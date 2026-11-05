"""Tests del cruce de tres fuentes que protege la importación desde Colab."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from tfm_edge.common.config import load_class_names
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import is_absolute_path_text  # noqa: F401
from tfm_edge.common.registry import (
    EXPERIMENTS_COLUMNS,
    TABLE_PRIMARY_KEYS,
    ExperimentRegistry,
    build_quality_columns,
)
from tfm_edge.compression.quantization.finetune import QatFinetuneReport
from tfm_edge.models.complexity import ModelComplexity


def _cargar_script() -> ModuleType:
    """Carga ``scripts/14_import_colab_registry.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "14_import_colab_registry.py"
    especificacion = importlib.util.spec_from_file_location("import_colab_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    # El módulo tiene que estar en sys.modules antes de ejecutarlo, porque ``@dataclass`` lo busca
    # ahí.
    sys.modules["import_colab_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """El script cargado una sola vez."""
    return _cargar_script()


class TestQueElVolcadoDeclaraSusParametros:
    """De dónde sale la primera de las tres fuentes, por técnica."""

    def test_un_volcado_de_poda(self, script: ModuleType) -> None:
        """La forma que ya funcionaba, para que siga funcionando."""
        volcado = {"pruning": {"parameters_before": 11_142_179, "parameters_after": 3_620_000}}
        parametros, procedencia = script.parametros_del_volcado(volcado)
        assert parametros == 3_620_000
        assert "pruning.parameters_after" in procedencia

    def test_un_volcado_de_qat(self, script: ModuleType) -> None:
        """La que fallaba en silencio."""
        informe = QatFinetuneReport(parameters=11_142_179)
        volcado = {"qat": informe.to_dict()}
        parametros, procedencia = script.parametros_del_volcado(volcado)
        assert parametros == 11_142_179
        assert "qat.parameters" in procedencia

    def test_un_volcado_sin_parametros_no_devuelve_un_numero(self, script: ModuleType) -> None:
        """Y la procedencia dice dónde se ha mirado, que es lo que hay que arreglar."""
        parametros, procedencia = script.parametros_del_volcado({"distillation": {"alumno": "n"}})
        assert parametros is None
        assert "pruning.parameters_after" in procedencia
        assert "qat.parameters" in procedencia

    def test_ninguna_tecnica_conocida_se_ha_quedado_fuera_por_descuido(
        self, script: ModuleType
    ) -> None:
        assert set(script.PARAMS_EN_EL_VOLCADO) >= {"pruning", "qat"}


# --- La disposición 'baselines' -------------------------------------------------

#: Un baseline de mentira, con los parámetros de un ``nn.Linear(2, 3)``: 2x3 + 3.
PARAMS_DE_PRUEBA = 9
EID = "l_baseline_default_188b9407"
ENV_ID = "3f0d6e692b5a"


def _escribir_csv(ruta: Path, columnas: tuple[str, ...], filas: list[dict[str, str]]) -> None:
    """Escribe un CSV del registro en el directorio de origen de un test."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8", newline="") as manejador:
        escritor = csv.DictWriter(manejador, fieldnames=list(columnas))
        escritor.writeheader()
        for fila in filas:
            escritor.writerow({c: fila.get(c, "") for c in columnas})


def _fila_de_experimento(**cambios: str) -> dict[str, str]:
    """Fila de ``experiments`` de un baseline, tal y como la trae el origen."""
    fila = {columna: "" for columna in EXPERIMENTS_COLUMNS}
    fila.update(
        {
            "experiment_id": EID,
            "timestamp": "2026-08-19T05:05:05+00:00",
            "model_family": "yolov8",
            "model_size": "l",
            "technique": "baseline",
            "technique_variant": "default",
            "config_hash": "188b9407",
            "config_path": "/content/drive/MyDrive/tfm-edge/runs/l_baseline_default/args.yaml",
            "compression_ratio": "1.0",
            "effective_params": str(PARAMS_DE_PRUEBA),
            "effective_params_ratio": "1.0",
            "precision": "fp32",
            "training_epochs": "100",
            "training_time_s": "36769.4",
            "training_device": "Tesla T4",
            "weights_path": "/content/drive/MyDrive/tfm-edge/runs/l_baseline_default/weights/best.pt",
            "status": "ok",
        }
    )
    fila.update(cambios)
    return fila


@pytest.fixture
def origen_baselines(tmp_path: Path) -> Path:
    """Monta un origen y unos pesos locales completos para la disposición baselines."""
    clases = load_class_names(
        get_project_paths().root / "configs" / "dataset" / "sh17_classes.json"
    )

    origen = tmp_path / "origen"
    _escribir_csv(origen / "experiments.csv", EXPERIMENTS_COLUMNS, [_fila_de_experimento()])

    columnas_calidad = build_quality_columns(clases)
    fila_calidad = {columna: "" for columna in columnas_calidad}
    fila_calidad.update(
        {
            "experiment_id": EID,
            "runtime": "pytorch",
            "precision": "fp32",
            "device_id": "auto",
            "environment_id": ENV_ID,
            "split": "val",
            "input_resolution": "640",
            "map_50": "0.629685",
            "map_50_95": "0.423482",
            "status": "ok",
            "notes": "evaluator=ultralytics",
        }
    )
    _escribir_csv(origen / "quality_metrics.csv", columnas_calidad, [fila_calidad])

    from tfm_edge.common.environment import ENVIRONMENT_COLUMNS

    fila_entorno = {columna: "" for columna in ENVIRONMENT_COLUMNS}
    fila_entorno.update(
        {
            "environment_id": ENV_ID,
            "device_id": "auto",
            "os": "Linux",
            "architecture": "x86_64",
            "python_version": "3.12.13",
            "torch_version": "2.8.0",
            "execution_context": "google_colab+container",
            "timestamp": "2026-08-18T18:35:51+00:00",
        }
    )
    _escribir_csv(origen / "environments.csv", ENVIRONMENT_COLUMNS, [fila_entorno])

    # La evidencia y los pesos, donde los busca la disposición: sin el hash en el
    # nombre del directorio, que es de lo que va 'dirname_sin_hash'.
    directorio = tmp_path / "models" / "l_baseline_default"
    directorio.mkdir(parents=True)
    complejidad = ModelComplexity(
        params_total=PARAMS_DE_PRUEBA,
        params_trainable=PARAMS_DE_PRUEBA,
        gflops=1.0,
        gmacs=0.5,
        resolution=640,
    )
    (directorio / "complexity.json").write_text(json.dumps(complejidad.to_dict()), encoding="utf-8")
    (directorio / "schedule.json").write_text(json.dumps({"epochs": 100}), encoding="utf-8")
    (directorio / "training_cost.json").write_text(
        json.dumps({"wall_time_s": 36769.4}), encoding="utf-8"
    )
    torch.save({"model": torch.nn.Linear(2, 3)}, directorio / "best.pt")
    return tmp_path


def _argumentos(script: ModuleType, montaje: Path, registro_dir: Path) -> object:
    """Los argumentos de la CLI para el montaje, pasando por ``parse_args`` de verdad."""
    return script.parse_args(
        [
            "--layout",
            "baselines",
            "--source-dir",
            str(montaje / "origen"),
            "--weights-root",
            str(montaje / "models"),
            "--registry-dir",
            str(registro_dir),
        ]
    )


def _registro_con_el_baseline(registro_dir: Path, **cambios: str) -> ExperimentRegistry:
    """Registro local con el baseline ya inscrito, como lo deja el paso 06."""
    clases = load_class_names(
        get_project_paths().root / "configs" / "dataset" / "sh17_classes.json"
    )
    registro = ExperimentRegistry(registry_dir=registro_dir, class_names=clases)
    registro.append_experiment(
        _fila_de_experimento(
            timestamp="2026-08-19T11:02:25+00:00",
            config_path="models/baselines/l_baseline_default/args.yaml",
            weights_path="models/baselines/l_baseline_default/best.pt",
            **cambios,
        ),
        on_duplicate="fail",
    )
    return registro


class TestLaDisposicionBaselines:
    """Lo que cambia cuando la evidencia no es un volcado crudo."""

    def test_el_directorio_del_experimento_pierde_el_hash(self, script: ModuleType) -> None:
        """Los baselines viven en 'l_baseline_default', sin el hash."""
        raiz = Path("/tmp/modelos")
        assert script.experiment_dir(EID, script.LAYOUTS["baselines"], raiz).name == (
            "l_baseline_default"
        )
        assert (
            script.experiment_dir(
                "s_pruning_l1global_r50_9419b26f", script.LAYOUTS["pruning"], raiz
            ).name
            == "s_pruning_l1global_r50_9419b26f"
        )

    def test_los_parametros_salen_del_complexity_json_real(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """La primera de las tres fuentes cuando no hay volcado crudo."""
        informe = ModelComplexity(
            params_total=43_642_947,
            params_trainable=43_642_931,
            gflops=164.9,
            gmacs=82.4,
            resolution=640,
        )
        (tmp_path / "complexity.json").write_text(json.dumps(informe.to_dict()), encoding="utf-8")
        parametros, procedencia = script.parametros_de_la_evidencia(
            tmp_path, script.LAYOUTS["baselines"]
        )
        assert parametros == 43_642_947
        assert "complexity.json:params_total" in procedencia

    def test_sin_evidencia_no_hay_numero_y_la_procedencia_lo_dice(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        parametros, procedencia = script.parametros_de_la_evidencia(
            tmp_path, script.LAYOUTS["baselines"]
        )
        assert parametros is None
        assert "complexity.json" in procedencia

    def test_la_disposicion_de_baselines_no_declara_volcado_crudo(self, script: ModuleType) -> None:
        assert script.LAYOUTS["baselines"].raw_subdir is None
        assert script.LAYOUTS["baselines"].evidence_files
        assert script.LAYOUTS["pruning"].raw_subdir == "pruning"

    def test_raw_dir_con_baselines_es_un_error_de_uso(self, script: ModuleType) -> None:
        """Y no una opción que se ignore en silencio, que es como se pierden cosas."""
        with pytest.raises(SystemExit):
            script.parse_args(
                ["--layout", "baselines", "--source-dir", "/tmp/x", "--raw-dir", "/tmp/y"]
            )


class TestQueUnaFilaInscritaNoSeReescribe:
    """La regla que hace segura la importación de baselines."""

    def test_la_fila_que_ya_esta_se_confirma_y_lo_que_falta_se_planifica(
        self, script: ModuleType, origen_baselines: Path, tmp_path: Path
    ) -> None:
        registro_dir = tmp_path / "registro"
        registro = _registro_con_el_baseline(registro_dir)
        args = _argumentos(script, origen_baselines, registro_dir)

        plan = script.build_plan(args, registro, get_project_paths())

        assert plan.experiment_rows == []
        assert [eid for eid, _ in plan.experiment_confirmations] == [EID]
        previstas = dict(plan.experiment_confirmations)[EID]
        assert any(d.startswith("config_path:") for d in previstas)
        assert any(d.startswith("weights_path:") for d in previstas)
        assert any(d.startswith("timestamp:") for d in previstas)
        assert len(plan.quality_rows) == 1
        assert plan.quality_rows[0]["map_50_95"] == "0.423482"
        assert [f["environment_id"] for f in plan.environment_rows] == [ENV_ID]
        assert plan.raw_copies == []

    def test_una_discrepancia_de_verdad_no_importa_nada(
        self, script: ModuleType, origen_baselines: Path, tmp_path: Path
    ) -> None:
        registro_dir = tmp_path / "registro"
        registro = _registro_con_el_baseline(registro_dir, training_time_s="12345.0")
        args = _argumentos(script, origen_baselines, registro_dir)

        with pytest.raises(TFMEdgeError) as excepcion:
            script.build_plan(args, registro, get_project_paths())

        mensaje = str(excepcion.value)
        assert "training_time_s" in mensaje
        assert "36769.4" in mensaje and "12345.0" in mensaje
        # Y el registro sigue como estaba: build_plan no escribe.
        assert registro.read_rows("quality_metrics") == []

    def test_unos_pesos_que_no_son_los_de_la_metrica_siguen_sin_pasar(
        self, script: ModuleType, origen_baselines: Path, tmp_path: Path
    ) -> None:
        """El cruce de tres fuentes sigue vivo con la evidencia nueva."""
        complejidad = origen_baselines / "models" / "l_baseline_default" / "complexity.json"
        datos = json.loads(complejidad.read_text(encoding="utf-8"))
        datos["params_total"] = PARAMS_DE_PRUEBA + 1
        complejidad.write_text(json.dumps(datos), encoding="utf-8")

        registro_dir = tmp_path / "registro"
        registro = _registro_con_el_baseline(registro_dir)
        args = _argumentos(script, origen_baselines, registro_dir)

        with pytest.raises(TFMEdgeError, match="no coincide entre las tres fuentes"):
            script.build_plan(args, registro, get_project_paths())

    def test_sin_la_evidencia_al_lado_del_checkpoint_no_se_importa(
        self, script: ModuleType, origen_baselines: Path, tmp_path: Path
    ) -> None:
        (origen_baselines / "models" / "l_baseline_default" / "training_cost.json").unlink()

        registro_dir = tmp_path / "registro"
        registro = _registro_con_el_baseline(registro_dir)
        args = _argumentos(script, origen_baselines, registro_dir)

        with pytest.raises(TFMEdgeError, match="training_cost.json"):
            script.build_plan(args, registro, get_project_paths())


class TestComoSeComparanLasFilas:
    """Qué diferencia bloquea y cuál solo se anota."""

    def test_las_rutas_el_timestamp_y_las_notas_no_bloquean(self, script: ModuleType) -> None:
        """Son las tres en las que el desacuerdo está previsto y explicado."""
        local = {"experiment_id": "x", "weights_path": "models/a.pt", "timestamp": "A", "notes": ""}
        origen = {
            "experiment_id": "x",
            "weights_path": "/content/a.pt",
            "timestamp": "B",
            "notes": "algo",
        }
        bloqueantes, previstas = script.comparar_filas("experiments", local, origen)
        assert bloqueantes == []
        assert len(previstas) == 3

    def test_cualquier_otra_columna_bloquea(self, script: ModuleType) -> None:
        bloqueantes, _ = script.comparar_filas(
            "experiments", {"effective_params": "9"}, {"effective_params": "10"}
        )
        assert len(bloqueantes) == 1 and "effective_params" in bloqueantes[0]

        bloqueantes, _ = script.comparar_filas("experiments", {}, {"columna_nueva": "v"})
        assert len(bloqueantes) == 1 and "columna_nueva" in bloqueantes[0]

    def test_dos_filas_iguales_no_dan_ninguna_diferencia(self, script: ModuleType) -> None:
        """Control, para que los dos tests de arriba no pasen por comparar de más."""
        fila = {"experiment_id": "x", "effective_params": "9"}
        assert script.comparar_filas("experiments", fila, dict(fila)) == ([], [])

    def test_la_clave_de_calidad_sale_del_registro_y_no_de_una_copia(
        self, script: ModuleType
    ) -> None:
        """Estaba escrita a mano en el script; dos definiciones de la misma clave divergen."""
        fila = {c: f"v{i}" for i, c in enumerate(TABLE_PRIMARY_KEYS["quality_metrics"])}
        assert script.clave_de_calidad(fila) == tuple(
            fila[c] for c in TABLE_PRIMARY_KEYS["quality_metrics"]
        )


# --- La disposición 'distillation-50ep' ------------------------------------------

EID_KD = "n_distillation_l2n_control_alpha0_f5c4a70e"
OTRO_KD = "n_distillation_l2n_combined_t2_ac9ff009"


def _checkpoint_de_destilacion(ruta: Path, proyecto: str, epocas: int | None = 50) -> Path:
    """Checkpoint con ``train_args`` como los que deja Ultralytics."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    argumentos: dict[str, object] = {"project": proyecto, "name": "distill"}
    if epocas is not None:
        argumentos["epochs"] = epocas
    torch.save({"model": torch.nn.Linear(2, 3), "train_args": argumentos}, ruta)
    return ruta


def _volcado_kd(eid: str = EID_KD, epocas: int = 50) -> dict[str, object]:
    """Lo mínimo de un volcado de 20_distill.py que lee la importación."""
    return {
        "experiment_id": eid,
        "effective_config": {"epochs": epocas},
        "preflight": {"student_params": PARAMS_DE_PRUEBA},
    }


def _fila_kd(eid: str = EID_KD, **cambios: str) -> dict[str, str]:
    """Fila de ``experiments`` de destilación como la escribe la sesión de Colab."""
    fila = {
        "experiment_id": eid,
        "config_path": f"results/raw/distillation/epochs50/{eid}.json",
        "training_epochs": "50",
    }
    fila.update(cambios)
    return fila


class TestLaDisposicionDeDestilacion:
    """Lo que añade la disposición de la campaña de 50 épocas."""

    DRIVE = "/content/drive/MyDrive/tfm-edge/distillation"

    def _layout(self, script: ModuleType) -> object:
        return script.LAYOUTS["distillation-50ep"]

    def test_la_forma_es_la_que_deja_el_cuaderno(self, script: ModuleType) -> None:
        """Las rutas de la sección 11 del cuaderno 03."""
        layout = self._layout(script)
        assert layout.csv_subdir == "registry"
        assert layout.raw_source_subdir == "raw_epochs50"
        assert layout.raw_subdir == "distillation/epochs50"
        assert layout.weights_relpath == Path("distill") / "weights" / "best.pt"
        assert layout.epochs_esperadas == 50
        assert layout.identidad_estricta
        assert "prefijo exacto" in layout.aviso

    def test_poda_y_baselines_no_cambian(self, script: ModuleType) -> None:
        for nombre in ("pruning", "baselines"):
            layout = script.LAYOUTS[nombre]
            assert (layout.csv_subdir, layout.raw_source_subdir) == ("", "")
            assert layout.epochs_esperadas is None
            assert not layout.identidad_estricta

    def test_los_parametros_salen_de_la_comprobacion_previa(self, script: ModuleType) -> None:
        """Con el ``to_dict()`` real del informe, para que un renombrado se vea."""
        from tfm_edge.compression.distillation.preflight import PreflightResult

        volcado = {"preflight": PreflightResult(student_params=3_014_163).to_dict()}
        parametros, procedencia = script.parametros_del_volcado(volcado)
        assert parametros == 3_014_163
        assert "preflight.student_params" in procedencia

    def test_sin_comprobacion_previa_no_hay_parametros(self, script: ModuleType) -> None:
        ruta = (
            get_project_paths().root
            / "results/raw/distillation/per_branch_v1"
            / "n_distillation_l2n_control_alpha0_pilot10_8826269c.json"
        )
        volcado = json.loads(ruta.read_text(encoding="utf-8"))
        assert volcado["preflight"] is None
        parametros, procedencia = script.parametros_del_volcado(volcado)
        assert parametros is None
        assert "preflight.student_params" in procedencia

    def test_todo_del_mismo_experimento_pasa(self, script: ModuleType, tmp_path: Path) -> None:
        pesos = _checkpoint_de_destilacion(tmp_path / "best.pt", f"{self.DRIVE}/{EID_KD}")
        assert (
            script.comprobar_identidad(
                EID_KD, self._layout(script), _volcado_kd(), _fila_kd(), pesos
            )
            == []
        )

    def test_la_fila_puede_traer_las_epocas_vacias(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """Vacío es «no medido» en el registro; no contradice a las otras dos."""
        pesos = _checkpoint_de_destilacion(tmp_path / "best.pt", f"{self.DRIVE}/{EID_KD}")
        fila = _fila_kd(training_epochs="")
        assert (
            script.comprobar_identidad(EID_KD, self._layout(script), _volcado_kd(), fila, pesos)
            == []
        )

    def test_los_pesos_de_otra_ejecucion_no_pasan(self, script: ModuleType, tmp_path: Path) -> None:
        pesos = _checkpoint_de_destilacion(tmp_path / "best.pt", f"{self.DRIVE}/{OTRO_KD}")
        problemas = script.comprobar_identidad(
            EID_KD, self._layout(script), _volcado_kd(), _fila_kd(), pesos
        )
        assert len(problemas) == 1 and OTRO_KD in problemas[0]

    def test_un_checkpoint_sin_train_args_no_pasa(self, script: ModuleType, tmp_path: Path) -> None:
        """No saber de qué entrenamiento es no se da por bueno."""
        ruta = tmp_path / "best.pt"
        torch.save({"model": torch.nn.Linear(2, 3)}, ruta)
        problemas = script.comprobar_identidad(
            EID_KD, self._layout(script), _volcado_kd(), _fila_kd(), ruta
        )
        assert any("train_args" in p for p in problemas)

    @pytest.mark.parametrize("fuente", ["volcado", "checkpoint", "fila"])
    def test_diez_epocas_en_cualquier_fuente_no_pasan(
        self, script: ModuleType, tmp_path: Path, fuente: str
    ) -> None:
        """Un volcado, una fila o unos pesos del piloto de 10 épocas no entran aquí."""
        pesos = _checkpoint_de_destilacion(
            tmp_path / "best.pt",
            f"{self.DRIVE}/{EID_KD}",
            epocas=10 if fuente == "checkpoint" else 50,
        )
        volcado = _volcado_kd(epocas=10 if fuente == "volcado" else 50)
        fila = _fila_kd(training_epochs="10" if fuente == "fila" else "50")
        problemas = script.comprobar_identidad(EID_KD, self._layout(script), volcado, fila, pesos)
        assert len(problemas) == 1 and "50 épocas" in problemas[0]

    def test_un_volcado_de_otro_experimento_no_pasa(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """El nombre del fichero y lo que declara dentro tienen que coincidir."""
        pesos = _checkpoint_de_destilacion(tmp_path / "best.pt", f"{self.DRIVE}/{EID_KD}")
        problemas = script.comprobar_identidad(
            EID_KD, self._layout(script), _volcado_kd(eid=OTRO_KD), _fila_kd(), pesos
        )
        assert len(problemas) == 1 and OTRO_KD in problemas[0]

    def test_una_fila_que_apunta_a_la_raiz_no_pasa(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """Un volcado escrito en la raíz de distillation/ no es de esta campaña."""
        pesos = _checkpoint_de_destilacion(tmp_path / "best.pt", f"{self.DRIVE}/{EID_KD}")
        fila = _fila_kd(config_path=f"results/raw/distillation/{EID_KD}.json")
        problemas = script.comprobar_identidad(
            EID_KD, self._layout(script), _volcado_kd(), fila, pesos
        )
        assert len(problemas) == 1 and "subdirectorio" in problemas[0]

    def test_una_ruta_absoluta_de_colab_tambien_vale(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """La fila puede llegar con la ruta sin normalizar; lo que cuenta es la cola."""
        pesos = _checkpoint_de_destilacion(tmp_path / "best.pt", f"{self.DRIVE}/{EID_KD}/")
        fila = _fila_kd(
            config_path=(
                "/content/tfm-edge-object-detection/results/raw/distillation/epochs50/"
                f"{EID_KD}.json"
            )
        )
        assert (
            script.comprobar_identidad(EID_KD, self._layout(script), _volcado_kd(), fila, pesos)
            == []
        )

    def test_las_disposiciones_sin_identidad_estricta_no_miran_nada_de_esto(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """En poda, ``config_path`` es el YAML y el checkpoint no se abre aquí."""
        problemas = script.comprobar_identidad(
            "s_pruning_l1global_r50_9419b26f",
            script.LAYOUTS["pruning"],
            {"experiment_id": "otro"},
            {"config_path": "configs/compression/pruning/l1global.yaml"},
            tmp_path / "no_existe.pt",
        )
        assert problemas == []

    def test_los_checkpoints_reales_del_piloto_traen_lo_que_se_lee(
        self, script: ModuleType
    ) -> None:
        raiz = get_project_paths().root / "models" / "distillation"
        pesos = sorted(raiz.glob("n_distillation_*/weights/best.pt"))
        if not pesos:
            pytest.skip(f"no hay checkpoints de destilación en '{raiz}' (models/ no se versiona)")
        for ruta in pesos:
            eid = ruta.parent.parent.name
            argumentos = script.leer_train_args(ruta)
            assert argumentos is not None
            assert str(argumentos["project"]).rstrip("/").rsplit("/", 1)[-1] == eid
            assert argumentos["epochs"] == 10
            assert script.count_checkpoint_parameters(ruta) == 3_014_163


class TestQueLaProcedenciaNoLlevaLaRutaDeEstaMaquina:
    """``notes`` no debe llevar rutas absolutas de esta máquina."""

    def test_la_nota_de_una_fila_nueva_no_lleva_ninguna_ruta_absoluta(
        self, script: ModuleType, origen_baselines: Path, tmp_path: Path
    ) -> None:
        """Se planifica contra un registro vacío, que es cuando la fila se escribe."""
        registro_dir = tmp_path / "registro"
        clases = load_class_names(
            get_project_paths().root / "configs" / "dataset" / "sh17_classes.json"
        )
        registro = ExperimentRegistry(registry_dir=registro_dir, class_names=clases)
        args = _argumentos(script, origen_baselines, registro_dir)

        plan = script.build_plan(args, registro, get_project_paths())

        assert len(plan.experiment_rows) == 1, "el plan no trae la fila nueva"
        notas = plan.experiment_rows[0]["notes"]
        absolutas = [
            trozo for trozo in notas.replace("'", " ").split() if is_absolute_path_text(trozo)
        ]
        assert not absolutas, f"la procedencia vuelve a llevar rutas de esta máquina: {absolutas}"
        assert "origen" in notas, "la procedencia ha dejado de decir de dónde viene"

    def test_las_doce_ya_inscritas_siguen_siendo_doce(self) -> None:
        """Las viejas no se tocan, pero tampoco pueden crecer."""
        ruta = get_project_paths().root / "results" / "registry" / "experiments.csv"
        if not ruta.is_file():
            pytest.skip("No está el registro en este árbol.")

        with ruta.open(encoding="utf-8") as manejador:
            con_ruta = [
                fila["experiment_id"]
                for fila in csv.DictReader(manejador)
                if any(
                    is_absolute_path_text(trozo)
                    for trozo in (fila.get("notes") or "").replace("'", " ").split()
                )
            ]

        assert len(con_ruta) == 12, (
            f"{len(con_ruta)} filas con una ruta absoluta dentro de 'notes'; deberían ser 12: "
            f"{sorted(set(con_ruta) )}"
        )
