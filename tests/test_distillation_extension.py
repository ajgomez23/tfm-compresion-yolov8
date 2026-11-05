"""Tests de la extensión de la destilación a 50 épocas."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import io
import json
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from tfm_edge.common.config import DistillationConfig, compute_config_hash, load_config
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.paths import build_experiment_id, get_project_paths
from tfm_edge.compression.distillation.reading import (
    EVALUATOR_READABILITY_FLOOR,
    PILOT_CONTROL_ID,
    STARTING_BASELINE_ID,
    read_extension,
)
from tfm_edge.compression.distillation.seed_variance import load_distillation_runs
from tfm_edge.evaluation.coco_eval import CRITERIA, CocoEvaluationResult, CriterionMetrics
from tfm_edge.training.trainer import mosaic_regime

RAIZ = get_project_paths().root
CONFIGS = RAIZ / "configs" / "compression" / "distillation"
CUADERNO = RAIZ / "notebooks" / "03_colab_distill.ipynb"

#: Las seis configuraciones de la campaña, en el orden en que se lanzan.
CAMPANA_50 = (
    "kd_control_50ep.yaml",
    "kd_combined.yaml",
    "kd_logit_only.yaml",
    "kd_feature_only.yaml",
    "kd_combined_t1.yaml",
    "kd_combined_t3.yaml",
)

#: Las dos que llegaron a ejecutarse; las cuatro ablaciones no se lanzaron. Sus filas están en el
#: registro del repositorio.
EJECUTADAS_50EP = ("kd_control_50ep.yaml", "kd_combined.yaml")

#: Lo que pasa el cuaderno 03 a cada ejecución, además de la configuración.
BANDERAS_DEL_CUADERNO = ("--seed", "42", "--batch-size", "16", "--workers", "2")

#: Identificadores con los que se inscribió la campaña, calculados con la fórmula anterior (sin los
#: puntos de partida), así que no se pueden recalcular.
INSCRITOS_50EP = {
    "kd_control_50ep.yaml": "n_distillation_l2n_control_alpha0_f5c4a70e",
    "kd_combined.yaml": "n_distillation_l2n_combined_t2_ac9ff009",
}

#: Contenido de los checkpoints falsos. Alumno y maestro son distintos porque sus huellas entran en
#: el experiment_id.
BYTES_ALUMNO = b"no es un modelo: el alumno de estos tests"
BYTES_MAESTRO = b"no es un modelo: el maestro de estos tests"


def _cargar_script() -> ModuleType:
    """Carga ``scripts/20_distill.py`` como módulo nuevo, sin ejecutar su ``main``."""
    ruta = RAIZ / "scripts" / "20_distill.py"
    especificacion = importlib.util.spec_from_file_location("distill_script_extension", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


class _PreviaFalsa:
    """Una comprobación previa que ha cabido, sin haber corrido nada."""

    completed = True
    failure_kind = None
    memory_measured = True
    steps_timed = False
    conditions_match_campaign = False
    peak_reserved_mb = 6496.0
    gpu_total_mb = 15095.0

    def summary(self) -> str:
        return "(comprobación previa sustituida por el test)"

    def to_dict(self) -> dict[str, Any]:
        return {"sustituida_por_el_test": True}


class _DetectorFalso:
    """Detector que no carga nada y cuenta los parámetros del `n` real."""

    @staticmethod
    def load(ruta: Path, input_resolution: int = 640) -> "_DetectorFalso":
        return _DetectorFalso()

    def count_parameters(self) -> int:
        return 3_014_163


def _informe(map_50_95: float) -> dict[str, Any]:
    """Informe de evaluación con las mismas claves que ``evaluate_checkpoint``."""
    resultado = CocoEvaluationResult(
        split="val",
        input_resolution=640,
        num_images=1213,
        num_annotations=8000,
        num_detections=1000,
        by_criterion={
            criterio: CriterionMetrics(criterion=criterio, map_50=0.5, map_50_95=map_50_95)
            for criterio in CRITERIA
        },
        per_class={indice: {"ap_50_95": 0.3} for indice in range(17)},
    )
    return {
        "result": resultado,
        "registry_columns": {
            **resultado.to_registry_columns(),
            "input_resolution": 640,
            "conf_threshold": 0.001,
            "iou_threshold": 0.7,
            "split": "val",
            "eval_time_s": 1.0,
        },
        "num_matched_detections": 1000,
    }


def _ejecutar_main(
    monkeypatch: pytest.MonkeyPatch,
    directorio: Path,
    config: str,
    *extra: str,
    dry_run: bool = True,
    map_50_95: float | None = None,
    registro: Path | None = None,
    previa: Any = None,
    crudo: Path | None = None,
    bytes_alumno: bytes = BYTES_ALUMNO,
    bytes_maestro: bytes = BYTES_MAESTRO,
    subdirectorio: str = "",
) -> tuple[int, dict[str, Any]]:
    """Ejecuta el ``main`` real con los pasos caros sustituidos."""
    script = _cargar_script()
    pesos = directorio / subdirectorio if subdirectorio else directorio
    alumno = pesos / "alumno.pt"
    maestro = pesos / "maestro.pt"
    for ruta, contenido in ((alumno, bytes_alumno), (maestro, bytes_maestro)):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(contenido)

    anotado: dict[str, Any] = {}

    def _destilador(**kwargs: Any) -> Any:
        anotado.update(kwargs)
        mejor = Path(kwargs["output_dir"]) / "distill" / "weights" / "best.pt"

        def _entrenar(run_name: str = "distill") -> Any:
            if dry_run:
                raise TFMEdgeError("parada del test: el destilador está anotado")
            return SimpleNamespace(
                steps=1,
                adapter_params=0,
                adapter_initial_norms={},
                adapter_final_norms={},
                adapters_learned=True,
                adapter_grad_was_always_zero=kwargs["config"].alpha == 0.0,
                epoch_components=[],
                # Un coste vacío: el registro lo inscribe con las columnas en
                # blanco, que es lo que significa «no medido».
                training=SimpleNamespace(
                    cost=SimpleNamespace(to_dict=lambda: {}), best_checkpoint=mejor
                ),
                components_csv=None,
            )

        return SimpleNamespace(train=_entrenar)

    # setup_logging pone propagate=False en el logger del proyecto y deja un
    # fichero en results/logs: sin sustituirlo, los tests que corren después ya no
    # capturan sus avisos con caplog. Es lo que hace test_qat_identity.py.
    monkeypatch.setattr(script, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        script, "ejecutar_comprobacion_previa", lambda *a, **k: previa or _PreviaFalsa()
    )
    monkeypatch.setattr(script, "Detector", _DetectorFalso)
    monkeypatch.setattr(script, "DistillationTrainer", _destilador)
    if not dry_run:
        monkeypatch.setattr(script.runner, "assert_evaluation_inputs_available", lambda *a, **k: {})
        monkeypatch.setattr(
            script.runner, "evaluate_checkpoint", lambda *a, **k: _informe(map_50_95)
        )

    argv = [
        "20_distill.py",
        "--config",
        str(CONFIGS / config),
        "--student-checkpoint",
        str(alumno),
        "--teacher-checkpoint",
        str(maestro),
        "--output-dir",
        str(directorio / "salida"),
        "--raw-dir",
        str(crudo or directorio / "crudo"),
        "--registry-dir",
        str(registro or directorio / "registro"),
        "--allow-cpu-train",
        *extra,
    ]
    if dry_run:
        argv += ["--dry-run", "--no-evaluate"]
    monkeypatch.setattr(sys, "argv", argv)
    return script.main(), anotado


def _identificador(anotado: dict[str, Any]) -> str:
    """Saca el ``experiment_id`` del directorio de salida que recibió el destilador."""
    return Path(anotado["output_dir"]).name


def _huellas_falsas() -> tuple[str, str]:
    """Huellas de los checkpoints de prueba, las que ``main`` calcula al correr aquí."""
    return (
        hashlib.sha256(BYTES_ALUMNO).hexdigest(),
        hashlib.sha256(BYTES_MAESTRO).hexdigest(),
    )


def _constante_del_cuaderno(nombre: str) -> Any:
    """Extrae una constante literal declarada en una celda de código del cuaderno."""
    cuaderno = json.loads(CUADERNO.read_text(encoding="utf-8"))
    literal = None
    for celda in cuaderno["cells"]:
        if celda["cell_type"] != "code":
            continue
        for nodo in ast.walk(ast.parse("".join(celda["source"]))):
            if isinstance(nodo, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == nombre for t in nodo.targets
            ):
                literal = ast.literal_eval(nodo.value)
    assert literal is not None, f"el cuaderno no declara {nombre}"
    return literal


def _identidad_de(nombre_config: str) -> str:
    """Calcula el ``experiment_id`` de una configuración con el código real."""
    esperado = _constante_del_cuaderno("ESPERADO")
    script = _cargar_script()
    config = load_config(CONFIGS / nombre_config).model_copy(update={"seed": 42})
    return str(
        script.construir_identidad(
            config,
            esperado["n"]["sha256"],
            esperado["l"]["sha256"],
            esperado["n"]["experiment_id"],
        )[2]
    )


def _registro_antes_de_la_campana(destino: Path) -> Path:
    """Copia el registro del repositorio sin las filas de la campaña de 50 épocas."""
    campana = {_identidad_de(nombre) for nombre in CAMPANA_50} | set(INSCRITOS_50EP.values())
    quitados: set[str] = set()
    destino.mkdir(parents=True, exist_ok=True)
    for origen in sorted((RAIZ / "results" / "registry").glob("*.csv")):
        with origen.open(newline="", encoding="utf-8") as entrada:
            lector = csv.DictReader(entrada)
            campos = list(lector.fieldnames or [])
            filas = []
            for fila in lector:
                if fila.get("experiment_id") in campana:
                    quitados.add(str(fila["experiment_id"]))
                else:
                    filas.append(fila)
        with (destino / origen.name).open("w", newline="", encoding="utf-8") as salida:
            escritor = csv.DictWriter(salida, fieldnames=campos)
            escritor.writeheader()
            escritor.writerows(filas)
    faltan = set(INSCRITOS_50EP.values()) - quitados
    assert not faltan, f"la campaña no está inscrita con estos identificadores: {faltan}"
    return destino


class TestLaRecetaLlegaAlEntrenamiento:
    """Lo que declara el YAML es lo que recibe el destilador."""

    @pytest.mark.parametrize("config", CAMPANA_50)
    def test_la_campana_corre_sin_mosaico_sin_paciencia_y_con_su_lote(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: str
    ) -> None:
        codigo, anotado = _ejecutar_main(
            monkeypatch, tmp_path, config, *BANDERAS_DEL_CUADERNO, "--allow-mosaic-free"
        )

        assert codigo == 2  # el destilador falso para en train(), con --dry-run
        assert anotado["config"].epochs == 50
        assert anotado["close_mosaic_epochs"] == 50
        assert anotado["allow_mosaic_free"] is True
        assert anotado["patience"] == 0
        assert anotado["batch_size"] == 16
        assert anotado["workers"] == 2
        assert anotado["scheduler"] == "linear"

    def test_el_piloto_sigue_corriendo_su_receta(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codigo, anotado = _ejecutar_main(
            monkeypatch,
            tmp_path,
            "kd_pilot_10ep.yaml",
            *BANDERAS_DEL_CUADERNO,
            "--allow-mosaic-free",
        )

        assert codigo == 2
        assert anotado["config"].epochs == 10
        assert anotado["close_mosaic_epochs"] == 10
        assert anotado["patience"] == 30

    def test_sin_la_bandera_la_guarda_para_antes_de_construir_nada(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codigo, anotado = _ejecutar_main(
            monkeypatch, tmp_path, "kd_combined.yaml", *BANDERAS_DEL_CUADERNO
        )

        assert codigo == 1
        assert anotado == {}

    @pytest.mark.parametrize("config", CAMPANA_50)
    def test_el_regimen_resuelto_es_sin_mosaico(self, config: str) -> None:
        script = _cargar_script()
        resuelta = load_config(CONFIGS / config)

        regimen, _ = mosaic_regime(
            resuelta.distillation.epochs, script.cierre_del_mosaico(resuelta)
        )

        assert regimen == "sin_mosaico"


class TestLaIdentidad:
    """Lo que decide la receta entra en el ``experiment_id``, y lo ya medido no se mueve."""

    def test_construir_identidad_da_lo_mismo_que_main(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Si ``main`` volviera a construir el payload por su cuenta, esto fallaría."""
        _, anotado = _ejecutar_main(
            monkeypatch,
            tmp_path,
            "kd_control_50ep.yaml",
            *BANDERAS_DEL_CUADERNO,
            "--allow-mosaic-free",
        )
        script = _cargar_script()
        config = load_config(CONFIGS / "kd_control_50ep.yaml").model_copy(update={"seed": 42})
        huella_alumno, huella_maestro = _huellas_falsas()

        _, config_hash, experiment_id = script.construir_identidad(
            config, huella_alumno, huella_maestro, STARTING_BASELINE_ID
        )

        assert _identificador(anotado) == experiment_id
        assert experiment_id.endswith(config_hash[:8])

    def test_otro_lote_es_otro_experimento(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, con_16 = _ejecutar_main(
            monkeypatch, tmp_path / "a", "kd_combined.yaml", "--allow-mosaic-free"
        )
        _, explicito = _ejecutar_main(
            monkeypatch,
            tmp_path / "b",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            "--batch-size",
            "16",
        )
        _, con_8 = _ejecutar_main(
            monkeypatch,
            tmp_path / "c",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            "--batch-size",
            "8",
        )

        assert _identificador(con_16) == _identificador(explicito)
        assert _identificador(con_8) != _identificador(con_16)
        assert con_8["batch_size"] == 8

    def test_otra_paciencia_es_otro_experimento(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Antes, --patience 5 llegaba al entrenamiento sin tocar el identificador."""
        _, sin = _ejecutar_main(
            monkeypatch, tmp_path / "a", "kd_combined.yaml", "--allow-mosaic-free"
        )
        _, con_5 = _ejecutar_main(
            monkeypatch,
            tmp_path / "b",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            "--patience",
            "5",
        )

        assert _identificador(con_5) != _identificador(sin)
        assert con_5["patience"] == 5

    def test_con_la_formula_retirada_salen_los_diez_identificadores_medidos(self) -> None:
        script = _cargar_script()

        def retirada(config: Any) -> str:
            previa = config.model_copy(
                update={"evaluation": config.evaluation.model_copy(update={"split": "test"})}
            )
            payload = {
                **previa.model_dump(mode="json"),
                "seeding_scheme": script.SEEDING_SCHEME,
                "warmup_epochs": script.WARMUP_EPOCHS_DISTILL,
            }
            return build_experiment_id(
                model_size=previa.model.size,
                technique="distillation",
                variant=previa.technique_variant,
                config_hash=compute_config_hash(payload),
            )

        en_disco = {
            p.stem
            for p in (RAIZ / "results" / "raw" / "distillation").glob("n_distillation_*.json")
        }
        pilotos = set()
        for nombre in ("kd_pilot_10ep.yaml", "kd_pilot_10ep_control.yaml"):
            # scheduler y workers, con los valores usados.
            base = load_config(
                CONFIGS / nombre,
                overrides=["training.scheduler=cosine", "training.workers=8"],
            )
            for semilla in (42, 43, 44, 45):
                pilotos.add(retirada(base.model_copy(update={"seed": semilla})))

        assert len(pilotos) == 8
        assert pilotos <= en_disco

        for nombre, inscrito in INSCRITOS_50EP.items():
            config = load_config(CONFIGS / nombre).model_copy(update={"seed": 42})
            assert retirada(config) == inscrito

    def test_ninguno_de_los_diez_medidos_se_reproduce_con_el_codigo_de_hoy(self) -> None:
        medidos = {
            p.stem
            for p in (RAIZ / "results" / "raw" / "distillation").glob("n_distillation_*.json")
        } | set(INSCRITOS_50EP.values())
        script = _cargar_script()
        huella_alumno, huella_maestro = _huellas_falsas()

        calculados = {_identidad_de(nombre) for nombre in CAMPANA_50}
        for nombre in ("kd_pilot_10ep.yaml", "kd_pilot_10ep_control.yaml"):
            base = load_config(
                CONFIGS / nombre,
                overrides=["training.scheduler=cosine", "training.workers=8"],
            )
            for semilla in (42, 43, 44, 45):
                calculados.add(
                    script.construir_identidad(
                        base.model_copy(update={"seed": semilla}),
                        huella_alumno,
                        huella_maestro,
                        STARTING_BASELINE_ID,
                    )[2]
                )

        assert not (calculados & medidos)

    def test_el_cierre_sin_declarar_no_aparece_al_serializar(self) -> None:
        """Si apareciera como null, movería el hash de poda, QAT y destilación."""
        sin = DistillationConfig(kd_type="logit")
        con = DistillationConfig(kd_type="logit", epochs=50, close_mosaic_epochs=50)

        assert "close_mosaic_epochs" not in sin.model_dump(mode="json")
        assert con.model_dump(mode="json")["close_mosaic_epochs"] == 50

    def test_un_cierre_que_no_puede_ocurrir_se_rechaza(self) -> None:
        """Con más cierre que épocas, todas llevarían mosaico y el YAML diría otra cosa."""
        with pytest.raises(ValueError, match="close_mosaic_epochs=51"):
            DistillationConfig(kd_type="logit", epochs=50, close_mosaic_epochs=51)

    def test_los_seis_identificadores_son_nuevos_y_distintos(self) -> None:
        """Ninguno de los seis choca con nada, y la evidencia de las dos que corrieron sigue ahí."""
        esperados = {nombre: _identidad_de(nombre) for nombre in CAMPANA_50}
        registro = "\n".join(
            p.read_text(encoding="utf-8") for p in (RAIZ / "results" / "registry").glob("*.csv")
        )
        volcados = {p.stem for p in (RAIZ / "results" / "raw" / "distillation").rglob("*.json")}

        assert len(set(esperados.values())) == 6

        for nombre, experiment_id in esperados.items():
            assert experiment_id not in registro, f"{nombre}: el identificador de hoy ya tiene fila"
            assert not any(experiment_id in v for v in volcados), f"{nombre} tiene volcado"

        filas = list(csv.DictReader((RAIZ / "results" / "registry" / "experiments.csv").open()))
        for nombre in EJECUTADAS_50EP:
            experiment_id = INSCRITOS_50EP[nombre]
            cuantas = sum(1 for f in filas if f["experiment_id"] == experiment_id)
            assert cuantas == 1, f"{nombre}: {cuantas} filas de experimento, se esperaba 1"
            assert experiment_id in volcados, f"{nombre}: inscrito sin volcado crudo"

    def test_el_cuaderno_espera_los_identificadores_que_calcula_el_script(self) -> None:
        """Un identificador esperado escrito a mano envejece sin avisar."""
        literal = _constante_del_cuaderno("IDENTIFICADORES_ESPERADOS")

        calculados = {
            f"configs/compression/distillation/{nombre}": _identidad_de(nombre)
            for nombre in CAMPANA_50
        }
        assert literal == calculados

    def test_el_cuaderno_declara_la_huella_de_los_dos_checkpoints(self) -> None:
        """Sin las huellas, los identificadores esperados del cuaderno no se pueden calcular."""
        esperado = _constante_del_cuaderno("ESPERADO")

        assert set(esperado) == {"n", "l"}
        for talla, datos in esperado.items():
            assert re.fullmatch(
                r"[0-9a-f]{64}", datos["sha256"]
            ), f"{talla}: sha256 no es un sha256"


class TestLosDosPuntosDePartidaEntranEnLaIdentidad:
    """Los checkpoints del alumno y del maestro entran en la identidad."""

    def test_otro_maestro_es_otro_experimento(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, con_uno = _ejecutar_main(
            monkeypatch, tmp_path / "a", "kd_combined.yaml", "--allow-mosaic-free"
        )
        _, con_otro = _ejecutar_main(
            monkeypatch,
            tmp_path / "b",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            bytes_maestro=b"otro maestro, mismos hiperparametros",
        )

        assert _identificador(con_uno) != _identificador(con_otro)

    def test_otro_alumno_es_otro_experimento(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Con ``student_from_pretrained``, el alumno es el punto de partida."""
        _, con_uno = _ejecutar_main(
            monkeypatch, tmp_path / "a", "kd_combined.yaml", "--allow-mosaic-free"
        )
        _, con_otro = _ejecutar_main(
            monkeypatch,
            tmp_path / "b",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            bytes_alumno=b"otro alumno, mismos hiperparametros",
        )

        assert _identificador(con_uno) != _identificador(con_otro)

    def test_el_mismo_contenido_en_otra_ruta_da_el_mismo_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _, aqui = _ejecutar_main(
            monkeypatch, tmp_path / "a", "kd_combined.yaml", "--allow-mosaic-free"
        )
        _, en_colab = _ejecutar_main(
            monkeypatch,
            tmp_path / "b",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            subdirectorio="content/drive/MyDrive/tfm-edge",
        )

        assert _identificador(aqui) == _identificador(en_colab)

    def test_el_padre_declarado_entra_en_la_identidad(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """El linaje con el que se lee la referencia del registro no es decorativo."""
        _, por_defecto = _ejecutar_main(
            monkeypatch, tmp_path / "a", "kd_combined.yaml", "--allow-mosaic-free"
        )
        _, otro_padre = _ejecutar_main(
            monkeypatch,
            tmp_path / "b",
            "kd_combined.yaml",
            "--allow-mosaic-free",
            "--parent-experiment-id",
            "n_pruning_l1global_r50_00000000",
        )

        assert _identificador(por_defecto) != _identificador(otro_padre)


class TestLaLecturaDeLaExtension:
    """Regla de lectura de la extensión."""

    PARTIDA = 0.325636
    CONTROL_10 = 0.311413

    def test_destilar_ayuda_y_se_sigue(self) -> None:
        lectura = read_extension(0.3300, 0.3200, self.PARTIDA, self.CONTROL_10, epochs=50)

        assert lectura.verdict == "destilar ayuda"
        assert lectura.readable
        assert lectura.continue_campaign
        assert lectura.control_state == "mejora con las épocas extra"

    def test_si_el_control_empeora_se_para_aunque_destilar_ayude(self) -> None:
        """El control de 50 por debajo del de 10 dice que la receta nueva degrada."""
        lectura = read_extension(0.3100, 0.3000, self.PARTIDA, self.CONTROL_10, epochs=50)

        assert lectura.verdict == "destilar ayuda"
        assert lectura.control_state == "EMPEORA con las épocas extra"
        assert not lectura.continue_campaign

    def test_destilar_perjudica_para(self) -> None:
        lectura = read_extension(0.3100, 0.3200, self.PARTIDA, self.CONTROL_10, epochs=50)

        assert lectura.verdict == "destilar perjudica a esta configuración"
        assert not lectura.continue_campaign

    def test_las_epocas_extra_ayudaron_para(self) -> None:
        lectura = read_extension(0.3350, 0.3330, self.PARTIDA, self.CONTROL_10, epochs=50)

        assert lectura.verdict == "las épocas extra ayudaron, destilar no se ve"
        assert not lectura.continue_campaign

    def test_dentro_del_suelo_no_bastan(self) -> None:
        lectura = read_extension(0.3260, 0.3258, self.PARTIDA, self.CONTROL_10, epochs=50)

        assert lectura.verdict == "50 épocas no bastan para verlo"
        assert "caen dentro del suelo" in lectura.statement
        assert not lectura.continue_campaign

    def test_por_debajo_de_la_partida_no_dice_que_caiga_dentro_del_suelo(self) -> None:
        lectura = read_extension(0.3120, 0.3114, self.PARTIDA, self.CONTROL_10, epochs=50)

        assert lectura.verdict == "50 épocas no bastan para verlo"
        assert "caen dentro del suelo" not in lectura.statement
        assert "por debajo del punto de partida" in lectura.statement

    def test_el_umbral_es_el_de_d110_y_tiene_borde(self) -> None:
        casi = EVALUATOR_READABILITY_FLOOR - 1e-9
        justo = EVALUATOR_READABILITY_FLOOR + 1e-9
        base = 0.3200

        assert (
            read_extension(base + casi, base, self.PARTIDA, self.CONTROL_10, 50).verdict
            != "destilar ayuda"
        )
        assert (
            read_extension(base + justo, base, self.PARTIDA, self.CONTROL_10, 50).verdict
            == "destilar ayuda"
        )

    def test_salvedades_y_confusor_salen_siempre(self) -> None:
        for cifras in (
            (0.33, 0.32, self.PARTIDA, self.CONTROL_10),
            (0.31, 0.32, self.PARTIDA, self.CONTROL_10),
            (0.326, 0.3258, self.PARTIDA, self.CONTROL_10),
        ):
            lectura = read_extension(*cifras, epochs=50)
            texto = lectura.summary()
            assert "una ejecución por brazo" in lectura.caveat
            assert "alpha y la tasa" in lectura.confound
            assert "PUNTO DE DECISIÓN" in texto
            assert "EL CONTROL" in texto
            # La selección de best.pt se indica con el veredicto.
            assert "misma partición" in lectura.selection.lower()
            assert "SELECCIÓN DE best.pt" in texto
            assert lectura.to_dict()["selection"] == lectura.selection

    def test_no_se_aplica_al_piloto_ni_sin_umbral(self) -> None:
        with pytest.raises(ValueError, match="read_pilot"):
            read_extension(0.33, 0.32, self.PARTIDA, self.CONTROL_10, epochs=10)
        with pytest.raises(ValueError, match="positivo"):
            read_extension(0.33, 0.32, self.PARTIDA, self.CONTROL_10, epochs=50, floor=0.0)


class TestLaCampanaDeExtremoAExtremo:
    """Control en una ejecución, variante en otra, y la regla aplicada al final."""

    def test_la_variante_se_lee_contra_el_control_de_otra_ejecucion(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Con el registro real copiado, como lo tiene un clon en Colab."""
        registro = _registro_antes_de_la_campana(tmp_path / "registro")

        codigo, anotado = _ejecutar_main(
            monkeypatch,
            tmp_path / "sesion1",
            "kd_control_50ep.yaml",
            *BANDERAS_DEL_CUADERNO,
            "--allow-mosaic-free",
            dry_run=False,
            map_50_95=0.320000,
            registro=registro,
        )
        assert codigo == 0
        control_id = _identificador(anotado)
        salida_control = capsys.readouterr().out
        assert f"--control-experiment-id {control_id}" in salida_control

        codigo, _ = _ejecutar_main(
            monkeypatch,
            tmp_path / "sesion2",
            "kd_combined.yaml",
            *BANDERAS_DEL_CUADERNO,
            "--allow-mosaic-free",
            "--control-experiment-id",
            control_id,
            dry_run=False,
            map_50_95=0.330000,
            registro=registro,
        )
        salida = capsys.readouterr()

        assert codigo == 0
        assert "LECTURA DE LA EXTENSIÓN A 50 ÉPOCAS" in salida.out
        assert "destilar ayuda" in salida.out
        assert "AVISO" not in salida.err

        volcado = json.loads(
            next(
                (tmp_path / "sesion2" / "crudo").glob("n_distillation_l2n_combined_t2_*.json")
            ).read_text(encoding="utf-8")
        )
        assert volcado["training_recipe"]["close_mosaic_epochs"] == 50
        assert volcado["training_recipe"]["mosaic_regime"] == "sin_mosaico"
        assert volcado["training_recipe"]["patience"] == 0

    def test_los_ids_de_la_lectura_estan_en_el_registro_real(self) -> None:
        """Las dos cifras fijas de la regla existen, medidas con pycocotools."""
        texto = (RAIZ / "results" / "registry" / "quality_metrics.csv").read_text(encoding="utf-8")
        assert f"{STARTING_BASELINE_ID},pytorch" in texto
        assert f"{PILOT_CONTROL_ID},pytorch" in texto


class _PreviaConParametros(_PreviaFalsa):
    """La comprobación previa sustituida, pero con el volcado del informe real."""

    def to_dict(self) -> dict[str, Any]:
        from tfm_edge.compression.distillation.preflight import PreflightResult

        return PreflightResult(
            completed=True, memory_measured=True, student_params=3_014_163
        ).to_dict()


class _AlumnoN(torch.nn.Module):
    """Un módulo con los 3.014.163 parámetros de un ``yolov8n``, sin ser uno."""

    def __init__(self) -> None:
        super().__init__()
        self.pesos = torch.nn.Parameter(torch.zeros(3_014_163))


def _cargar_importador() -> ModuleType:
    """Carga ``scripts/14_import_colab_registry.py`` como módulo nuevo."""
    ruta = RAIZ / "scripts" / "14_import_colab_registry.py"
    especificacion = importlib.util.spec_from_file_location("import_colab_extension", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    # @dataclass busca el módulo de la clase en sys.modules al decorarla.
    sys.modules["import_colab_extension"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


class TestLaImportacionDeLaCampana:
    """Lo que escribe una sesión de 20_distill.py, importado con 14 tal cual."""

    def test_la_sesion_del_control_se_importa_y_solo_ella(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # --- La sesión de Colab, con su clon en otra raíz -----------------------
        clon = tmp_path / "content" / "tfm-edge-object-detection"
        registro_colab = _registro_antes_de_la_campana(clon / "results" / "registry")
        crudo_colab = clon / "results" / "raw" / "distillation" / "epochs50"
        codigo, anotado = _ejecutar_main(
            monkeypatch,
            tmp_path / "sesion1",
            "kd_control_50ep.yaml",
            *BANDERAS_DEL_CUADERNO,
            "--allow-mosaic-free",
            dry_run=False,
            map_50_95=0.320000,
            registro=registro_colab,
            previa=_PreviaConParametros(),
            crudo=crudo_colab,
        )
        assert codigo == 0
        eid = _identificador(anotado)
        # La identidad incluye los checkpoints, que aquí son falsos, así que solo se comprueba la
        # variante.
        assert eid.startswith("n_distillation_l2n_control_alpha0_")
        assert eid != INSCRITOS_50EP["kd_control_50ep.yaml"]

        # --- salvar_registro() del cuaderno, sección 11 -------------------------
        drive = tmp_path / "drive" / "resultados_distillation_50ep"
        shutil.copytree(registro_colab, drive / "registry")
        shutil.copytree(crudo_colab, drive / "raw_epochs50")

        # --- La estación de trabajo ---------------------------------------------
        local = _registro_antes_de_la_campana(tmp_path / "local")
        crudo_local = tmp_path / "raw_local"
        modelos = tmp_path / "models"
        pesos = modelos / eid / "distill" / "weights" / "best.pt"
        pesos.parent.mkdir(parents=True)

        def _pesos(proyecto: str) -> None:
            torch.save(
                {"model": _AlumnoN(), "train_args": {"project": proyecto, "epochs": 50}}, pesos
            )

        importador = _cargar_importador()
        monkeypatch.setattr(importador, "setup_logging", lambda *a, **k: None)
        argumentos = [
            "--layout",
            "distillation-50ep",
            "--source-dir",
            str(drive),
            "--weights-root",
            str(modelos),
            "--raw-dir",
            str(crudo_local),
            "--registry-dir",
            str(local),
            "--source-root",
            str(clon),
        ]
        intacto = {p.name: p.read_bytes() for p in local.iterdir()}

        def _sin_tocar() -> bool:
            return {p.name: p.read_bytes() for p in local.iterdir()} == intacto and not (
                crudo_local.exists()
            )

        # Los pesos de otra ejecución: mismos parámetros, otro experimento.
        _pesos(
            "/content/drive/MyDrive/tfm-edge/distillation/n_distillation_l2n_combined_t2_ac9ff009"
        )
        capsys.readouterr()
        assert importador.main(argumentos + ["--dry-run"]) == 2
        assert _sin_tocar()

        # Los buenos, primero en seco.
        _pesos(f"/content/drive/MyDrive/tfm-edge/distillation/{eid}")
        assert importador.main(argumentos + ["--dry-run"]) == 0
        salida = capsys.readouterr().out
        assert f"+ experimento {eid}  params=3014163" in salida
        assert "AVISO (distillation-50ep)" in salida
        assert _sin_tocar()

        # Sin --apply no se escribe nada; con --apply, sí.
        assert importador.main(argumentos) == 0
        assert _sin_tocar(), "sin --apply no se escribe nada"

        assert importador.main(argumentos + ["--apply"]) == 0
        for tabla in ("experiments", "quality_metrics"):
            antes = list(csv.DictReader(io.StringIO(intacto[f"{tabla}.csv"].decode("utf-8"))))
            despues = list(csv.DictReader((local / f"{tabla}.csv").open(encoding="utf-8")))
            assert despues[: len(antes)] == antes, f"{tabla}: se ha reescrito una fila previa"
            nuevas = despues[len(antes) :]
            assert [f["experiment_id"] for f in nuevas] == [eid], tabla
        fila = list(csv.DictReader((local / "experiments.csv").open(encoding="utf-8")))[-1]
        assert fila["weights_path"] == str(pesos)
        assert fila["config_path"].endswith(f"results/raw/distillation/epochs50/{eid}.json")
        calidad = list(csv.DictReader((local / "quality_metrics.csv").open(encoding="utf-8")))[-1]
        assert float(calidad["map_50_95"]) == pytest.approx(0.32)
        assert (crudo_local / f"{eid}.json").read_bytes() == (
            crudo_colab / f"{eid}.json"
        ).read_bytes()
        # Los volcados de la comprobación previa no son de ningún experimento inscrito.
        assert sorted(p.name for p in crudo_local.iterdir()) == [f"{eid}.json"]

        # Una segunda importación no encuentra nada que traer.
        assert importador.main(argumentos + ["--apply"]) == 2


class TestLaLecturaDeSemillasNoVeLaExtension:
    """Un volcado de 50 épocas en su subdirectorio no entra en la campaña de semillas."""

    def test_el_subdirectorio_de_la_campana_no_se_mezcla(self, tmp_path: Path) -> None:
        crudo = tmp_path / "distillation"
        shutil.copytree(RAIZ / "results" / "raw" / "distillation", crudo)
        antes = {e.experiment_id for e in load_distillation_runs(crudo)}

        extension = crudo / "epochs50"
        extension.mkdir(exist_ok=True)
        (extension / "n_distillation_l2n_control_alpha0_00000000.json").write_text(
            json.dumps(
                {
                    "experiment_id": "n_distillation_l2n_control_alpha0_00000000",
                    "seed_base": 42,
                    "is_control": True,
                    "metrics": {"map_50": 0.5, "map_50_95": 0.32},
                }
            ),
            encoding="utf-8",
        )

        assert {e.experiment_id for e in load_distillation_runs(crudo)} == antes
        assert len(antes) == 8


class TestElRegistroSeEncadenaEntreSesiones:
    """``adopt_campaign_registry`` solo adopta un registro que continúa el actual."""

    @staticmethod
    def _csv(ruta: Path, filas: list[list[str]]) -> None:
        import csv

        ruta.parent.mkdir(parents=True, exist_ok=True)
        with ruta.open("w", encoding="utf-8", newline="") as manejador:
            csv.writer(manejador).writerows(filas)

    def test_primera_sesion_no_hay_nada_que_adoptar(self, tmp_path: Path) -> None:
        from tfm_edge.common.registry_chain import adopt_campaign_registry

        self._csv(tmp_path / "local" / "experiments.csv", [["id"], ["a"]])
        lineas = adopt_campaign_registry(tmp_path / "local", tmp_path / "no_existe")

        assert "primera sesión" in lineas[0]

    def test_una_copia_que_solo_anade_se_adopta(self, tmp_path: Path) -> None:
        from tfm_edge.common.registry_chain import adopt_campaign_registry

        self._csv(tmp_path / "local" / "experiments.csv", [["id"], ["a"]])
        self._csv(tmp_path / "campana" / "experiments.csv", [["id"], ["a"], ["control"]])

        adopt_campaign_registry(tmp_path / "local", tmp_path / "campana")

        assert "control" in (tmp_path / "local" / "experiments.csv").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "campana",
        [
            [["id"], ["a (tocada)"], ["control"]],  # reescribe una fila
            [["otra"], ["a"], ["control"]],  # otra cabecera
            [["id"]],  # tiene menos filas: el repositorio avanzó
        ],
        ids=["reescribe", "cabecera", "menos_filas"],
    )
    def test_lo_que_no_continua_no_se_adopta_y_no_se_copia_nada(
        self, tmp_path: Path, campana: list[list[str]]
    ) -> None:
        from tfm_edge.common.exceptions import RegistryChainError
        from tfm_edge.common.registry_chain import adopt_campaign_registry

        self._csv(tmp_path / "local" / "experiments.csv", [["id"], ["a"]])
        self._csv(tmp_path / "local" / "quality_metrics.csv", [["id"], ["q"]])
        self._csv(tmp_path / "campana" / "experiments.csv", campana)
        # Esta sí continuaría: comprueba que no se adopta media campaña.
        self._csv(tmp_path / "campana" / "quality_metrics.csv", [["id"], ["q"], ["nueva"]])
        antes = (tmp_path / "local" / "quality_metrics.csv").read_bytes()

        with pytest.raises(RegistryChainError):
            adopt_campaign_registry(tmp_path / "local", tmp_path / "campana")

        assert (tmp_path / "local" / "quality_metrics.csv").read_bytes() == antes
