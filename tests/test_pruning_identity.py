"""Tests de la identidad y de la referencia de ``13_prune.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.config import load_class_names, load_config
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import ExperimentRegistry, build_quality_columns
from tfm_edge.data.download import compute_sha256
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR, ULTRALYTICS_EVALUATOR

RAIZ = get_project_paths().root
CONFIG_PODA = RAIZ / "configs" / "compression" / "pruning" / "l1_global_r25.yaml"
BASELINE_S = "s_baseline_default_745e3d08"


class _Centinela(BaseException):
    """Detiene ``main`` justo antes de podar, sin pasar por sus ``except``."""


def _cargar_13_prune() -> ModuleType:
    """Carga ``scripts/13_prune.py`` como módulo nuevo, sin ejecutar su ``main``."""
    ruta = RAIZ / "scripts" / "13_prune.py"
    especificacion = importlib.util.spec_from_file_location("prune_script_identidad", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _checkpoint(directorio: Path, contenido: bytes) -> Path:
    """Escribe un fichero que hace de checkpoint de partida."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / "best.pt"
    ruta.write_bytes(contenido)
    return ruta


def _ejecutar_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    *extra: str,
    padre: str = "",
    registro: Path | None = None,
) -> tuple[int | None, dict[str, Any]]:
    """Ejecuta el ``main`` real de ``13_prune.py`` hasta el umbral de la poda."""
    script = _cargar_13_prune()
    anotado: dict[str, Any] = {}

    def _campana_que_anota(**kwargs: Any) -> dict[str, Any]:
        anotado.update(kwargs)
        raise _Centinela()

    monkeypatch.setattr(script, "ejecutar_campana", _campana_que_anota)
    # setup_logging reconfigura el logging raíz y deja un fichero en results/logs.
    monkeypatch.setattr(script, "setup_logging", lambda *args, **kwargs: None)

    argumentos = [
        "13_prune.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(CONFIG_PODA),
        "--output-dir",
        str(tmp_path / "salida"),
        "--raw-dir",
        str(tmp_path / "crudo"),
        "--registry-dir",
        str(registro or (tmp_path / "registro")),
        "--no-evaluate",
        "--dry-run",
        *extra,
    ]
    if padre:
        argumentos += ["--parent-experiment-id", padre]
    monkeypatch.setattr(sys, "argv", argumentos)

    try:
        codigo: int | None = script.main()
    except _Centinela:
        codigo = None
    return codigo, anotado


def _identificador(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint: Path, *extra: str, padre: str = ""
) -> str:
    """Devuelve el ``experiment_id`` con el que ``main`` llega a podar."""
    codigo, anotado = _ejecutar_main(monkeypatch, tmp_path, checkpoint, *extra, padre=padre)
    assert codigo is None, "main debía llegar a la poda"
    return str(anotado["experiment_id"])


class TestElPuntoDePartidaEntraEnLaIdentidad:
    """Dos podas de pesos distintos no pueden compartir identificador."""

    def test_dos_checkpoints_distintos_no_comparten_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        baseline = _checkpoint(tmp_path / "baseline", b"pesos del baseline")
        otro = _checkpoint(tmp_path / "otro", b"otros pesos")

        de_baseline = _identificador(monkeypatch, tmp_path / "a", baseline, padre="p")
        de_otro = _identificador(monkeypatch, tmp_path / "b", otro, padre="p")

        assert de_baseline != de_otro

    def test_el_padre_declarado_entra_en_la_identidad(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pesos = _checkpoint(tmp_path / "pesos", b"los mismos pesos")

        con_a = _identificador(monkeypatch, tmp_path / "a", pesos, padre="padre_a")
        con_b = _identificador(monkeypatch, tmp_path / "b", pesos, padre="padre_b")

        assert con_a != con_b

    def test_el_mismo_contenido_en_otra_ruta_da_el_mismo_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        en_colab = _checkpoint(tmp_path / "content" / "tfm-edge", b"mismo fichero")
        en_local = _checkpoint(tmp_path / "models" / "baselines", b"mismo fichero")

        desde_colab = _identificador(monkeypatch, tmp_path / "a", en_colab, padre="p")
        desde_local = _identificador(monkeypatch, tmp_path / "b", en_local, padre="p")

        assert desde_colab == desde_local

    def test_construir_identidad_da_lo_mismo_que_main(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Si ``main`` volviera a construir el payload por su cuenta, esto fallaría."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos cualesquiera")
        de_main = _identificador(monkeypatch, tmp_path / "a", pesos, padre="padre_x")

        script = _cargar_13_prune()
        config = load_config(CONFIG_PODA)
        payload, config_hash, experiment_id = script.construir_identidad(
            config,
            config.pruning.criterion,
            config.technique_variant,
            False,
            compute_sha256(pesos),
            "padre_x",
        )

        assert experiment_id == de_main
        assert experiment_id.endswith(config_hash[:8])
        assert payload["starting_checkpoint_sha256"] == compute_sha256(pesos)
        assert payload["seeding_scheme"] == script.SEEDING_SCHEME


class TestElLoteEntraEnLaIdentidad:
    """Otro lote es otra receta, y el identificador tiene que decirlo."""

    def test_otro_lote_es_otro_experimento(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Antes, --batch-size 8 daba el mismo identificador con otra receta."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")

        por_defecto = _identificador(monkeypatch, tmp_path / "a", pesos)
        con_8 = _identificador(monkeypatch, tmp_path / "b", pesos, "--batch-size", "8")

        assert con_8 != por_defecto

    def test_el_lote_del_yaml_explicito_no_mueve_el_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Pasar el mismo lote que declara el YAML es la misma receta."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")
        del_yaml = load_config(CONFIG_PODA).training.batch_size

        por_defecto = _identificador(monkeypatch, tmp_path / "a", pesos)
        explicito = _identificador(
            monkeypatch, tmp_path / "b", pesos, "--batch-size", str(del_yaml)
        )

        assert explicito == por_defecto

    def test_el_lote_que_entra_en_el_hash_es_el_que_llega_al_reajuste(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Hash y entrenador leen el lote del mismo sitio."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")

        codigo, anotado = _ejecutar_main(monkeypatch, tmp_path, pesos, "--batch-size", "8")

        assert codigo is None
        assert anotado["config"].training.batch_size == 8
        assert anotado["trainer_kwargs"]["batch_size"] == 8


class TestElCalentamientoLlegaAlReajuste:
    """``warmup_epochs`` llega al reajuste."""

    def test_el_calentamiento_del_yaml_llega_al_entrenador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Se ejecuta el ``main`` real y se lee lo que recibiría el reajuste."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")
        del_yaml = load_config(CONFIG_PODA).training.warmup_epochs

        codigo, anotado = _ejecutar_main(monkeypatch, tmp_path, pesos)

        assert codigo is None
        assert anotado["trainer_kwargs"]["warmup_epochs"] == del_yaml

    def test_editar_el_yaml_cambia_lo_que_recibe_el_reajuste(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """La comprobación que separa «llega» de «coincide por casualidad»."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")

        codigo, anotado = _ejecutar_main(
            monkeypatch, tmp_path, pesos, "--override", "training.warmup_epochs=1.5"
        )

        assert codigo is None
        assert anotado["config"].training.warmup_epochs == 1.5
        assert anotado["trainer_kwargs"]["warmup_epochs"] == 1.5


@pytest.fixture(scope="module")
def clases() -> list[str]:
    """Nombres de clase canónicos del proyecto."""
    return load_class_names(RAIZ / "configs" / "dataset" / "sh17_classes.json")


def _inscribir(
    directorio: Path,
    clases: list[str],
    filas: list[tuple[str, str, str, str]],
) -> Path:
    """Crea un registro desechable con filas de calidad del baseline ``s``."""
    registro = ExperimentRegistry(registry_dir=directorio, class_names=clases)
    for evaluador, dispositivo, map_50, map_50_95 in filas:
        fila = {columna: "" for columna in build_quality_columns(clases)}
        fila.update(
            {
                "experiment_id": BASELINE_S,
                "runtime": "pytorch",
                "precision": "fp32",
                "device_id": dispositivo,
                "evaluator": evaluador,
                "split": "val",
                "map_50": map_50,
                "map_50_95": map_50_95,
                "status": "ok",
                "notes": f"evaluator={evaluador}. fila de test",
            }
        )
        registro.append_quality_metrics(fila, on_duplicate="fail")
    return directorio


class TestLaReferenciaNoSeEligePorElOrdenDelFichero:
    """La referencia se lee con el evaluador con que se mide."""

    def test_con_la_fila_de_ultralytics_primera_se_lee_la_de_pycocotools(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clases: list[str]
    ) -> None:
        registro = _inscribir(
            tmp_path / "registro",
            clases,
            [
                (ULTRALYTICS_EVALUATOR, "auto", "0.59", "0.383774"),
                (CANONICAL_EVALUATOR, "workstation_cpu", "0.586152", "0.376214"),
            ],
        )
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")

        codigo, anotado = _ejecutar_main(
            monkeypatch, tmp_path, pesos, padre=BASELINE_S, registro=registro
        )

        assert codigo is None
        assert anotado["baseline_metrics"] == {"map_50": 0.586152, "map_50_95": 0.376214}

    def test_dos_filas_del_mismo_evaluador_que_se_contradicen_paran_antes_de_podar(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clases: list[str]
    ) -> None:
        """Si el dispositivo no desempata, se falla en vez de elegir."""
        registro = _inscribir(
            tmp_path / "registro",
            clases,
            [
                (CANONICAL_EVALUATOR, "maquina_a", "0.586152", "0.376214"),
                (CANONICAL_EVALUATOR, "maquina_b", "0.5", "0.3"),
            ],
        )
        pesos = _checkpoint(tmp_path / "pesos", b"pesos")

        codigo, anotado = _ejecutar_main(
            monkeypatch, tmp_path, pesos, padre=BASELINE_S, registro=registro
        )

        assert codigo == 1
        assert anotado == {}, "main no debía llegar a podar con la referencia ambigua"
