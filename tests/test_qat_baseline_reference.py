"""Tests de la referencia FP32 que lee ``16_qat.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.config import load_class_names
from tfm_edge.common.exceptions import RegistryAmbiguousRowError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import ExperimentRegistry, build_quality_columns
from tfm_edge.common.registry_lookup import read_reference_quality
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR, ULTRALYTICS_EVALUATOR

RAIZ = get_project_paths().root
CONFIG_QAT = RAIZ / "configs" / "compression" / "quantization" / "qat_default.yaml"
BASELINE_S = "s_baseline_default_745e3d08"


@pytest.fixture(scope="module")
def clases() -> list[str]:
    """Nombres de clase canónicos del proyecto."""
    return load_class_names(RAIZ / "configs" / "dataset" / "sh17_classes.json")


def _registro(directorio: Path, clases: list[str]) -> ExperimentRegistry:
    """Crea un registro vacío en un directorio desechable."""
    return ExperimentRegistry(registry_dir=directorio, class_names=clases)


def _inscribir(
    registro: ExperimentRegistry,
    clases: list[str],
    *,
    experiment_id: str = BASELINE_S,
    evaluator: str = CANONICAL_EVALUATOR,
    device_id: str = "workstation_cpu",
    map_50: str = "0.586152",
    map_50_95: str = "0.376214",
    split: str = "val",
    runtime: str = "pytorch",
) -> None:
    """Inscribe una fila de calidad mínima pero completa."""
    fila = {columna: "" for columna in build_quality_columns(clases)}
    fila.update(
        {
            "experiment_id": experiment_id,
            "runtime": runtime,
            "precision": "fp32",
            "device_id": device_id,
            "evaluator": evaluator,
            "split": split,
            "map_50": map_50,
            "map_50_95": map_50_95,
            "status": "ok",
            "notes": f"evaluator={evaluator}. fila de test",
        }
    )
    registro.append_quality_metrics(fila, on_duplicate="fail")


def _el_caso_real_al_reves(registro: ExperimentRegistry, clases: list[str]) -> None:
    """Las dos filas reales del baseline ``s``, con la de Ultralytics primera."""
    _inscribir(
        registro,
        clases,
        evaluator=ULTRALYTICS_EVALUATOR,
        device_id="auto",
        map_50="0.593859",
        map_50_95="0.383774",
    )
    _inscribir(registro, clases)


class TestLaConsulta:
    """``read_reference_quality``, sin el script."""

    def test_devuelve_la_del_evaluador_pedido_aunque_vaya_la_ultima(
        self, clases: list[str], tmp_path: Path
    ) -> None:
        """El caso real con las filas al revés."""
        registro = _registro(tmp_path, clases)
        _el_caso_real_al_reves(registro, clases)

        referencia = read_reference_quality(
            registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR
        )

        assert referencia == {
            "map_50_95": 0.376214,
            "map_50": 0.586152,
            "device_id": "workstation_cpu",
            "evaluator": CANONICAL_EVALUATOR,
        }

    def test_sin_fila_del_evaluador_no_hay_referencia(
        self, clases: list[str], tmp_path: Path
    ) -> None:
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases, evaluator=ULTRALYTICS_EVALUATOR, device_id="auto")

        assert (
            read_reference_quality(registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR) == {}
        )

    def test_dos_dispositivos_distintos_sin_desempate_es_un_error(
        self, clases: list[str], tmp_path: Path
    ) -> None:
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50_95="0.376300")

        with pytest.raises(RegistryAmbiguousRowError) as excepcion:
            read_reference_quality(registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR)

        assert "0.376214" in str(excepcion.value)
        assert "0.376300" in str(excepcion.value)

    def test_el_dispositivo_desempata(self, clases: list[str], tmp_path: Path) -> None:
        """Con el dispositivo pedido presente, sale el suyo, vaya donde vaya."""
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50_95="0.376300")

        referencia = read_reference_quality(
            registro,
            BASELINE_S,
            "val",
            evaluator=CANONICAL_EVALUATOR,
            preferred_device_id="colab_t4_gpu",
        )

        assert referencia["map_50_95"] == 0.376300
        assert referencia["device_id"] == "colab_t4_gpu"

    def test_un_dispositivo_ausente_no_desempata(self, clases: list[str], tmp_path: Path) -> None:
        """El desempate no es un filtro, pero tampoco un permiso para elegir."""
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50_95="0.376300")

        with pytest.raises(RegistryAmbiguousRowError):
            read_reference_quality(
                registro,
                BASELINE_S,
                "val",
                evaluator=CANONICAL_EVALUATOR,
                preferred_device_id="jetson_orin",
            )

    def test_un_dispositivo_ausente_con_una_sola_fila_la_devuelve_con_su_dispositivo(
        self, clases: list[str], tmp_path: Path
    ) -> None:
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)

        referencia = read_reference_quality(
            registro,
            BASELINE_S,
            "val",
            evaluator=CANONICAL_EVALUATOR,
            preferred_device_id="colab_t4_gpu",
        )

        assert referencia["map_50_95"] == 0.376214
        assert referencia["device_id"] == "workstation_cpu"

    def test_dos_filas_que_dicen_lo_mismo_no_son_un_empate(
        self, clases: list[str], tmp_path: Path
    ) -> None:
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)
        _inscribir(registro, clases, device_id="colab_t4_gpu")

        referencia = read_reference_quality(
            registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR
        )

        assert referencia["map_50_95"] == 0.376214

    def test_coincidir_en_la_principal_no_basta(self, clases: list[str], tmp_path: Path) -> None:
        """Si difieren en mAP@50, devolver una sigue siendo elegir por el orden."""
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50="0.586999")

        with pytest.raises(RegistryAmbiguousRowError):
            read_reference_quality(registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR)

    @pytest.mark.parametrize(
        "cambio",
        [
            {"split": "test"},
            {"runtime": "onnxruntime"},
            {"experiment_id": "s_baseline_otro_00000000"},
            {"map_50_95": ""},
        ],
        ids=["otra_particion", "otro_motor", "otro_experimento", "sin_medir"],
    )
    def test_las_filas_que_no_encajan_no_son_candidatas(
        self, clases: list[str], tmp_path: Path, cambio: dict[str, str]
    ) -> None:
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases)
        _inscribir(registro, clases, **{"device_id": "colab_t4_gpu", "map_50_95": "0.1", **cambio})

        referencia = read_reference_quality(
            registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR
        )

        assert referencia["map_50_95"] == 0.376214

    def test_un_map_50_sin_medir_se_omite_no_se_rellena(
        self, clases: list[str], tmp_path: Path
    ) -> None:
        registro = _registro(tmp_path, clases)
        _inscribir(registro, clases, map_50="")

        referencia = read_reference_quality(
            registro, BASELINE_S, "val", evaluator=CANONICAL_EVALUATOR
        )

        assert "map_50" not in referencia
        assert referencia["map_50_95"] == 0.376214


def _cargar_16_qat() -> ModuleType:
    """Carga ``scripts/16_qat.py`` como módulo nuevo, sin ejecutar su ``main``."""
    ruta = RAIZ / "scripts" / "16_qat.py"
    especificacion = importlib.util.spec_from_file_location("qat_script_referencia", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


class _Parada(Exception):
    """Detiene ``main`` justo después del punto que se comprueba."""


class _DetectorFalso:
    """Sustituye a ``Detector`` y anota si se carga."""

    cargados: list[Path] = []

    @classmethod
    def load(cls, checkpoint: Path, **kwargs: Any) -> "_DetectorFalso":
        """Anota la carga y devuelve un detector que solo cuenta parámetros."""
        cls.cargados.append(checkpoint)
        return cls()

    def count_parameters(self) -> int:
        """Devuelve un recuento cualquiera."""
        return 1


def _ejecutar_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registro_dir: Path,
    padre: str,
    capturados: dict[str, Any],
) -> int:
    """Ejecuta el ``main`` real de ``16_qat.py`` hasta ``finetune_qat``, sin llamarlo."""
    script = _cargar_16_qat()
    _DetectorFalso.cargados = []

    def _finetune_que_captura(detector: object, config: object, **kwargs: Any) -> None:
        capturados.update(kwargs)
        raise _Parada

    monkeypatch.setattr(script, "setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(script, "Detector", _DetectorFalso)
    monkeypatch.setattr(script, "finetune_qat", _finetune_que_captura)

    checkpoint = tmp_path / "pesos" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pesos cualesquiera")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "16_qat.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(CONFIG_QAT),
            "--output-dir",
            str(tmp_path / "salida"),
            "--raw-dir",
            str(tmp_path / "crudo"),
            "--registry-dir",
            str(registro_dir),
            "--no-evaluate",
            "--allow-cpu-finetune",
            "--allow-mosaic-free",
            "--parent-experiment-id",
            padre,
        ],
    )
    return script.main()


class TestElScriptUsaLaConsulta:
    """El ``main`` real de ``16_qat.py``, parado antes de entrenar."""

    def test_con_el_caso_real_al_reves_pasa_la_de_pycocotools(
        self,
        clases: list[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        registro_dir = tmp_path / "registro"
        _el_caso_real_al_reves(_registro(registro_dir, clases), clases)
        capturados: dict[str, Any] = {}

        with pytest.raises(_Parada):
            _ejecutar_main(monkeypatch, tmp_path, registro_dir, BASELINE_S, capturados)

        assert capturados["baseline_metrics"] == {"map_50": 0.586152, "map_50_95": 0.376214}
        salida = capsys.readouterr().out
        assert "mAP@50-95 = 0.376214 [pycocotools, workstation_cpu]" in salida

    def test_ante_el_empate_falla_antes_de_cargar_el_modelo(
        self,
        clases: list[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        registro_dir = tmp_path / "registro"
        registro = _registro(registro_dir, clases)
        _inscribir(registro, clases, device_id="workstation_cpu")
        _inscribir(registro, clases, device_id="auto", map_50_95="0.376300")
        capturados: dict[str, Any] = {}

        codigo = _ejecutar_main(monkeypatch, tmp_path, registro_dir, BASELINE_S, capturados)

        assert codigo == 1
        assert _DetectorFalso.cargados == []
        assert capturados == {}
        assert "no se puede fijar la referencia FP32" in capsys.readouterr().err

    def test_sin_fila_la_referencia_queda_ausente(
        self, clases: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sin fila del evaluador canónico se sigue, con la referencia vacía."""
        registro_dir = tmp_path / "registro"
        _inscribir(
            _registro(registro_dir, clases),
            clases,
            evaluator=ULTRALYTICS_EVALUATOR,
            device_id="auto",
        )
        capturados: dict[str, Any] = {}

        with pytest.raises(_Parada):
            _ejecutar_main(monkeypatch, tmp_path, registro_dir, BASELINE_S, capturados)

        assert capturados["baseline_metrics"] == {}


class TestElRegistroReal:
    """Lectura de ``results/registry``: puntos de partida de la rama de poda y QAT."""

    PUNTOS_DE_PARTIDA = {
        BASELINE_S: 0.376214,
        "s_pruning_bnscale_r50_f51fef4c": 0.286544,
        "s_pruning_l1global_r25_4e91ddbc": 0.357927,
        "s_pruning_l1global_r25_randomctl_a658e5d4": 0.347943,
        "s_pruning_l1global_r50_9419b26f": 0.316322,
        "s_pruning_l1global_r50_iter4_d8fe066f": 0.331078,
        "s_pruning_l1global_r50_randomctl_d581bd22": 0.300312,
    }

    @pytest.mark.parametrize("experiment_id", sorted(PUNTOS_DE_PARTIDA))
    def test_cada_punto_de_partida_tiene_una_referencia_sin_ambiguedad(
        self, clases: list[str], experiment_id: str
    ) -> None:
        """Si esto falla con un empate, hay que decidir la fila antes de lanzar el afinado."""
        registro = ExperimentRegistry(class_names=clases)
        script = _cargar_16_qat()

        referencia = script.leer_baseline_del_registro(
            registro, experiment_id, "val", "colab_t4_gpu"
        )

        assert referencia["evaluator"] == CANONICAL_EVALUATOR
        assert referencia["map_50_95"] == self.PUNTOS_DE_PARTIDA[experiment_id]
