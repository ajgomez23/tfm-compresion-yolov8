"""Tests del paso 17, que inscribe la fila de calidad de un checkpoint ya hecho."""

from __future__ import annotations

import csv
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn

# El informe falso se construye con el ``to_registry_columns`` real, reutilizando el de otro fichero
# de tests.
from test_script_registry_calls import CLASES, _informe_eval  # noqa: E402
from tfm_edge.common.paths import ProjectPaths, get_project_paths
from tfm_edge.common.registry import EXPERIMENTS_COLUMNS, ExperimentRegistry
from tfm_edge.compression.quantization import qat
from tfm_edge.evaluation.quality_notes import (
    EVALUATOR_PREFIX,
    QAT_FP32_NOTE,
    TECHNIQUES_REQUIRING_EXPLICIT_NOTE,
    build_quality_note,
)
from tfm_edge.models.detector import Detector

EXPERIMENT_ID = "s_qat_int8-noexcl_d4d403fa"


def _cargar_script() -> ModuleType:
    """Carga ``scripts/17_register_quality.py`` como módulo, sin ejecutar nada."""
    ruta = get_project_paths().root / "scripts" / "17_register_quality.py"
    especificacion = importlib.util.spec_from_file_location("register_quality_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """El script cargado una sola vez para todo el módulo."""
    return _cargar_script()


class _DetectorDeJuguete:
    """Doble con lo que mira :func:`comprobar_el_checkpoint`: el modelo y sus parámetros."""

    def __init__(self, modelo: nn.Module, size: str = "s") -> None:
        self.model = modelo
        # En el detector real el tamaño sale de la arquitectura; aquí se inyecta porque el modelo no
        # es un YOLOv8.
        self.size = size

    def count_parameters(self, only_trainable: bool = False) -> int:
        """Cuenta los parámetros, igual que el detector real."""
        return sum(
            p.numel() for p in self.model.parameters() if not only_trainable or p.requires_grad
        )


def _modelo_minimo() -> nn.Module:
    """Un módulo con parámetros y sin nada más."""
    return nn.Sequential(nn.Conv2d(3, 4, 3), nn.SiLU())


def _fila_de_experimento(**campos: Any) -> dict[str, Any]:
    """Construye una fila de ``experiments.csv`` completa."""
    fila: dict[str, Any] = {columna: "" for columna in EXPERIMENTS_COLUMNS}
    fila.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "model_family": "yolov8",
            "model_size": "s",
            "technique": "qat",
            "technique_variant": "int8-noexcl",
            "config_hash": "d4d403fa",
            "precision": "fp32",
            "status": "ok",
        }
    )
    fila.update(campos)
    return fila


def _piezas_de_evaluacion(tmp_path: Path) -> Path:
    """Crea las tres piezas que el paso 17 exige antes de tocar el checkpoint."""
    coco = tmp_path / "data" / "processed" / "sh17_coco.json"
    coco.parent.mkdir(parents=True, exist_ok=True)
    coco.write_text("{}", encoding="utf-8")

    lista = tmp_path / "data" / "splits" / "val.txt"
    lista.parent.mkdir(parents=True, exist_ok=True)
    lista.write_text("", encoding="utf-8")

    data_yaml = tmp_path / "data" / "resized_640" / "sh17_ultralytics.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: .\n", encoding="utf-8")
    return data_yaml


class TestQueLaNotaDiceQueMideLaFila:
    """La nota es lo único que impide leer una cifra de QAT como si fuera INT8."""

    def test_la_nota_de_qat_empieza_por_el_evaluador(self) -> None:
        nota = build_quality_note("qat")
        assert nota.startswith(EVALUATOR_PREFIX)
        assert "NO lo que rinde el INT8" in nota

    def test_la_nota_no_afirma_que_los_nodos_se_hayan_quitado(self) -> None:
        """Decía "ya sin nodos de simulación", y eso no es lo que ocurre."""
        assert "desactivada" in QAT_FP32_NOTE
        assert "sin nodos de simulación" not in QAT_FP32_NOTE

    def test_la_procedencia_y_el_texto_del_usuario_se_conservan(self) -> None:
        """Una fila reinscrita tiene que decir que lo es."""
        nota = build_quality_note("qat", provenance="Reevaluado.", extra="Nota del autor.")
        assert "Reevaluado." in nota
        assert "Nota del autor." in nota
        assert QAT_FP32_NOTE in nota

    def test_la_poda_no_tiene_nota_automatica(self) -> None:
        """No se puede deducir de los pesos si un modelo podado está reajustado."""
        assert "pruning" in TECHNIQUES_REQUIRING_EXPLICIT_NOTE
        assert build_quality_note("pruning") == EVALUATOR_PREFIX

    def test_el_paso_16_usa_esta_misma_nota(self) -> None:
        """Los dos scripts que escriben la fila de QAT dicen lo mismo."""
        ruta = get_project_paths().root / "scripts" / "16_qat.py"
        codigo = ruta.read_text(encoding="utf-8")
        assert "build_quality_note(" in codigo
        assert "PESOS AFINADOS CON QAT" not in codigo


class TestQueLaFilaDeCalidadSeEscribeYSeLee:
    """El tramo que murió en Colab, recorrido entero por el script nuevo."""

    @staticmethod
    def _fila(script: ModuleType, tmp_path: Path) -> dict[str, Any]:
        """Construye la fila con el código real del script."""
        return script.construir_fila_calidad(
            experiment_id=EXPERIMENT_ID,
            informe_eval=_informe_eval(),
            clases=list(CLASES),
            device_id="workstation_cpu",
            environment_id="0123456789abcdef",
            ruta_detalle=tmp_path / "detalle.json",
            nota=build_quality_note("qat", provenance="Reevaluado por el paso 17."),
        )

    def test_la_fila_se_inscribe_y_se_relee_del_csv(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        """Se escribe con un registro de verdad y se lee de vuelta del fichero."""
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=list(CLASES))
        assert registro.append_quality_metrics(self._fila(script, tmp_path), on_duplicate="fail")

        with (tmp_path / "quality_metrics.csv").open(encoding="utf-8", newline="") as manejador:
            filas = list(csv.DictReader(manejador))

        assert len(filas) == 1
        fila = filas[0]
        assert fila["experiment_id"] == EXPERIMENT_ID
        assert fila["runtime"] == "pytorch"
        assert fila["precision"] == "fp32"
        assert float(fila["map_50_95"]) == pytest.approx(0.371)
        assert fila["status"] == "ok"
        assert "Reevaluado por el paso 17." in fila["notes"]

    def test_el_ap_por_clase_llega_a_sus_columnas(self, script: ModuleType, tmp_path: Path) -> None:
        fila = self._fila(script, tmp_path)
        for indice, nombre in enumerate(CLASES):
            columna = f"ap_class_{indice:02d}_{nombre}"
            assert fila[columna] == pytest.approx(0.3 + indice / 100)

    def test_la_fila_dice_donde_esta_su_evidencia(self, script: ModuleType, tmp_path: Path) -> None:
        """``predictions_path`` apunta al volcado; sin él la fila no es auditable."""
        fila = self._fila(script, tmp_path)
        assert fila["predictions_path"] == str(tmp_path / "detalle.json")


class TestQueElCheckpointTieneQueSerElDelExperimento:
    """La guarda que impide escribir la medida de un modelo en la fila de otro."""

    def test_un_checkpoint_que_cuadra_no_da_motivos(self, script: ModuleType) -> None:
        detector = _DetectorDeJuguete(_modelo_minimo())
        fila = _fila_de_experimento(effective_params=str(detector.count_parameters()))
        assert script.comprobar_el_checkpoint(detector, fila, EXPERIMENT_ID) == []

    def test_un_numero_de_parametros_distinto_se_niega(self, script: ModuleType) -> None:
        """El síntoma de un ``--experiment-id`` equivocado si el tamaño coincide."""
        detector = _DetectorDeJuguete(_modelo_minimo())
        motivos = script.comprobar_el_checkpoint(
            detector, _fila_de_experimento(effective_params="11142179"), EXPERIMENT_ID
        )
        assert len(motivos) == 1
        assert "11.142.179" in motivos[0] or "11,142,179" in motivos[0]

    def test_sin_parametros_declarados_no_se_niega_pero_se_avisa(
        self, script: ModuleType, caplog: pytest.LogCaptureFixture
    ) -> None:
        detector = _DetectorDeJuguete(_modelo_minimo())
        with caplog.at_level("WARNING"):
            motivos = script.comprobar_el_checkpoint(
                detector, _fila_de_experimento(), EXPERIMENT_ID
            )
        assert motivos == []
        assert "effective_params" in caplog.text

    def test_unos_pesos_en_fp16_se_niegan(self, script: ModuleType) -> None:
        """La fila diría ``precision=fp32``, y eso es una afirmación."""
        detector = _DetectorDeJuguete(_modelo_minimo().half())
        motivos = script.comprobar_el_checkpoint(
            detector,
            _fila_de_experimento(effective_params=str(detector.count_parameters())),
            EXPERIMENT_ID,
        )
        assert any("FP32" in motivo for motivo in motivos)

    def test_la_simulacion_activa_se_niega_y_la_desactivada_no(self, script: ModuleType) -> None:
        """El par que de verdad discrimina."""
        import torch.ao.quantization as qao

        modelo = nn.Sequential(_modelo_minimo(), qao.FakeQuantize())
        detector = _DetectorDeJuguete(modelo)
        fila = _fila_de_experimento(effective_params=str(detector.count_parameters()))

        motivos_activo = script.comprobar_el_checkpoint(detector, fila, EXPERIMENT_ID)
        assert any("simulada siguen" in motivo for motivo in motivos_activo)

        modelo.apply(qao.disable_fake_quant)
        assert script.comprobar_el_checkpoint(detector, fila, EXPERIMENT_ID) == []

    def test_lo_que_deja_strip_fake_quantization_pasa_la_guarda(self, script: ModuleType) -> None:
        """La guarda tiene que aceptar justo lo que produce el paso anterior."""
        modelo = nn.Sequential(_modelo_minimo(), torch.ao.quantization.FakeQuantize())
        limpio = qat.strip_fake_quantization(modelo)
        detector = _DetectorDeJuguete(limpio)
        fila = _fila_de_experimento(effective_params=str(detector.count_parameters()))
        assert script.comprobar_el_checkpoint(detector, fila, EXPERIMENT_ID) == []


class TestQueElTamanoLoDiceLaArquitectura:
    """El tamaño del ``experiment_id`` y de ``model_size`` se contrasta con la arquitectura."""

    def test_el_tamano_que_cuadra_no_da_motivos(self, script: ModuleType) -> None:
        detector = _DetectorDeJuguete(_modelo_minimo(), size="s")
        fila = _fila_de_experimento(effective_params=str(detector.count_parameters()))
        assert script.comprobar_el_tamano(detector, fila, EXPERIMENT_ID) == []

    def test_un_experiment_id_de_otro_tamano_se_niega(self, script: ModuleType) -> None:
        detector = _DetectorDeJuguete(_modelo_minimo(), size="l")
        motivos = script.comprobar_el_tamano(detector, _fila_de_experimento(), EXPERIMENT_ID)

        assert any("'l'" in motivo and "experiment_id" in motivo for motivo in motivos)

    def test_una_columna_de_otro_tamano_se_niega(self, script: ModuleType) -> None:
        detector = _DetectorDeJuguete(_modelo_minimo(), size="s")
        motivos = script.comprobar_el_tamano(
            detector, _fila_de_experimento(model_size="m"), EXPERIMENT_ID
        )

        assert any("'m'" in motivo and "model_size" in motivo for motivo in motivos)

    def test_una_columna_vacia_se_niega_y_no_se_rellena(self, script: ModuleType) -> None:
        detector = _DetectorDeJuguete(_modelo_minimo(), size="s")
        motivos = script.comprobar_el_tamano(
            detector, _fila_de_experimento(model_size=""), EXPERIMENT_ID
        )

        assert any("no declara 'model_size'" in motivo for motivo in motivos)

    def test_la_guarda_entra_en_la_comprobacion_general(self, script: ModuleType) -> None:
        """Una guarda que existe y no la llama nadie no protege de nada."""
        detector = _DetectorDeJuguete(_modelo_minimo(), size="l")
        fila = _fila_de_experimento(effective_params=str(detector.count_parameters()))

        motivos = script.comprobar_el_checkpoint(detector, fila, EXPERIMENT_ID)

        assert any("experiment_id" in motivo for motivo in motivos)


class TestQueElDobleNoSeSeparaDelDetectorReal:
    """Un doble solo vale mientras se parezca a lo que sustituye."""

    def test_el_detector_real_expone_lo_que_mira_la_guarda(self) -> None:
        """Si ``Detector`` renombrara esto, la guarda moriría y el doble no."""
        assert hasattr(Detector, "model"), "Detector no expone 'model'."
        assert callable(getattr(Detector, "count_parameters", None))
        assert "self.size" in inspect.getsource(Detector.__init__), (
            "Detector no asigna 'size': la guarda de tamaño leería un atributo que no existe y "
            "el doble seguiría pasando."
        )


class TestQueSeNiegaAntesDeGastarLaEvaluacion:
    """Todo lo que puede fallar barato falla antes de evaluar."""

    @pytest.fixture(autouse=True)
    def _arbol_en_temporal(
        self, script: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Da al script un árbol de datos, modelos y resultados en el temporal."""
        reales = get_project_paths()
        monkeypatch.setattr(
            script,
            "get_project_paths",
            lambda: ProjectPaths(
                root=reales.root,
                configs=reales.configs,
                data=tmp_path / "data",
                models=tmp_path / "models",
                results=tmp_path / "results",
            ),
        )

    @staticmethod
    def _ejecutar(
        script: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        experiment_id: str,
        *extra: str,
    ) -> int:
        """Ejecuta ``main()`` con un checkpoint de mentira que sí existe."""
        checkpoint = tmp_path / "pesos.pt"
        checkpoint.write_bytes(b"no se llega a abrir")
        argv = [
            "17_register_quality.py",
            "--checkpoint",
            str(checkpoint),
            "--experiment-id",
            experiment_id,
            "--registry-dir",
            str(tmp_path / "registro"),
            *extra,
        ]
        monkeypatch.setattr("sys.argv", argv)
        return script.main()

    def test_la_poda_sin_notes_se_niega(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        """Su cifra no se puede interpretar sin saber si está reajustada."""
        codigo = self._ejecutar(script, monkeypatch, tmp_path, "s_pruning_l1global_r50_8db60bc4")
        assert codigo == 1
        assert "--notes" in capsys.readouterr().err

    def test_un_experimento_que_no_existe_se_niega(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        codigo = self._ejecutar(script, monkeypatch, tmp_path, EXPERIMENT_ID)
        assert codigo == 1
        assert "experiments.csv" in capsys.readouterr().err

    def test_un_experimento_fallido_se_niega(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        """Una fila de calidad contradiría a su propia fila de experimento."""
        registro = ExperimentRegistry(
            registry_dir=tmp_path / "registro", class_names=_clases_del_proyecto()
        )
        registro.append_experiment(
            _fila_de_experimento(status="fallido", notes="murió por AMP"), on_duplicate="fail"
        )
        codigo = self._ejecutar(script, monkeypatch, tmp_path, EXPERIMENT_ID)
        assert codigo == 1
        assert "fallido" in capsys.readouterr().err

    def test_un_checkpoint_ilegible_da_un_mensaje_y_no_una_traza(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        """Un ``.pt`` truncado sale de torch como ``UnpicklingError``."""
        registro = ExperimentRegistry(
            registry_dir=tmp_path / "registro", class_names=_clases_del_proyecto()
        )
        registro.append_experiment(_fila_de_experimento(), on_duplicate="fail")
        # Sin las tres piezas de evaluación, el script se para antes de abrir el
        # checkpoint y este test aprobaría el mensaje de otra guarda. Aquí hay
        # que ponerlas para llegar a la que se quiere probar.
        data_yaml = _piezas_de_evaluacion(tmp_path)

        codigo = self._ejecutar(
            script, monkeypatch, tmp_path, EXPERIMENT_ID, "--data", str(data_yaml)
        )
        assert codigo == 1
        assert "no se puede cargar el checkpoint" in capsys.readouterr().err

    def test_sin_las_piezas_de_evaluacion_se_para_antes_y_lo_dice(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        """La otra guarda, la que el test de arriba estuvo aprobando por error."""
        registro = ExperimentRegistry(
            registry_dir=tmp_path / "registro", class_names=_clases_del_proyecto()
        )
        registro.append_experiment(_fila_de_experimento(), on_duplicate="fail")

        codigo = self._ejecutar(
            script, monkeypatch, tmp_path, EXPERIMENT_ID, "--data", str(tmp_path / "no_existe.yaml")
        )

        assert codigo == 1
        error = capsys.readouterr().err
        assert "para poder evaluar" in error
        assert "no se puede cargar el checkpoint" not in error

    def test_no_pisa_una_fila_de_calidad_que_ya_trae_metricas(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
    ) -> None:
        clases = _clases_del_proyecto()
        registro = ExperimentRegistry(registry_dir=tmp_path / "registro", class_names=clases)
        registro.append_experiment(_fila_de_experimento(), on_duplicate="fail")

        fila = {columna: "" for columna in registro.columns_for("quality_metrics")}
        fila.update(
            {
                "experiment_id": EXPERIMENT_ID,
                "runtime": "pytorch",
                "precision": "fp32",
                "device_id": "workstation_cpu",
                "evaluator": "pycocotools",
                "split": "val",
                "map_50_95": "0.3710",
                "status": "ok",
            }
        )
        registro.append_quality_metrics(fila, on_duplicate="fail")

        codigo = self._ejecutar(
            script, monkeypatch, tmp_path, EXPERIMENT_ID, "--on-duplicate", "overwrite"
        )
        assert codigo == 1
        salida = capsys.readouterr().err
        assert "0.3710" in salida, "El mensaje tiene que decir qué se perdería."


class TestQueTestNoPisaAVal:
    """La partición y el dispositivo van en el directorio y en el volcado."""

    def test_el_directorio_del_validador_lleva_la_particion(self, script: ModuleType) -> None:
        """Sin ella, la evaluación sobre test escribía en el directorio de val."""
        val = script.validator_dir_for(Path("q"), EXPERIMENT_ID, "val")
        test = script.validator_dir_for(Path("q"), EXPERIMENT_ID, "test")
        assert val != test
        # El directorio antiguo no lleva partición y no se renombra.
        assert Path("q") / f"_val_{EXPERIMENT_ID}" not in (val, test)

    def test_el_volcado_lleva_particion_y_dispositivo(self, script: ModuleType) -> None:
        ruta = script.dump_path_for(Path("q"), EXPERIMENT_ID, "test", 640, "workstation_cpu")
        assert ruta.name == f"{EXPERIMENT_ID}_test_640_workstation_cpu.json"
        assert ruta != script.dump_path_for(Path("q"), EXPERIMENT_ID, "val", 640, "workstation_cpu")
        assert ruta != script.dump_path_for(Path("q"), EXPERIMENT_ID, "test", 640, "colab_t4_gpu")
        # El nombre antiguo no puede volver a escribirse.
        assert ruta.name != f"{EXPERIMENT_ID}_val_640.json"

    def test_un_dispositivo_con_barra_falla(self, script: ModuleType) -> None:
        """Un device_id con una barra no puede escribir en otro directorio."""
        from tfm_edge.common.exceptions import InvalidPathComponentError

        with pytest.raises(InvalidPathComponentError):
            script.dump_path_for(Path("q"), EXPERIMENT_ID, "test", 640, "a/b")


def _clases_del_proyecto() -> list[str]:
    """Devuelve los nombres de clase canónicos del proyecto."""
    from tfm_edge.common.config import load_class_names

    return load_class_names(get_project_paths().root / "configs" / "dataset" / "sh17_classes.json")
