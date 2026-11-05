"""El coste inscrito de la poda iterativa es la suma de sus cuatro entrenamientos."""

from __future__ import annotations

import hashlib
import json
import shutil
from decimal import Decimal
from pathlib import Path

import pytest

from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import ExperimentRegistry

EXPERIMENT_ID = "s_pruning_l1global_r50_iter4_d8fe066f"
STAGES = ("finetune_step1", "finetune_step2", "finetune_step3", "finetune")
PARTIAL_EPOCHS = 5
FINAL_EPOCHS = 30
ORIGINAL_SECONDS = "2830.26"

ROOT = get_project_paths().root
EVIDENCE_DIR = ROOT / "docs" / "evidencia" / "d185_coste_iterativa"
REGISTRY_DIR = ROOT / "results" / "registry"
DUMP_PATH = ROOT / "results" / "raw" / "pruning" / f"{EXPERIMENT_ID}.json"


def check_iterative_cost(registry_dir: Path, evidence_dir: Path, dump_path: Path) -> None:
    """Comprueba que celda, evidencia y volcado cuentan lo mismo."""
    huellas = json.loads((evidence_dir / "sha256.json").read_text(encoding="utf-8"))
    for nombre, datos in huellas.items():
        real = hashlib.sha256((evidence_dir / nombre).read_bytes()).hexdigest()
        assert real == datos["sha256"], f"{nombre} no coincide con su huella en sha256.json"

    segundos = Decimal("0")
    epocas = 0
    for tramo in STAGES:
        coste = json.loads(
            (evidence_dir / f"{tramo}_training_cost.json").read_text(encoding="utf-8")
        )
        assert coste["gpu_name"] == "Tesla T4", f"{tramo}: {coste['gpu_name']}"
        assert coste["interrupted"] is False, f"{tramo}: interrumpido"
        esperadas = FINAL_EPOCHS if tramo == "finetune" else PARTIAL_EPOCHS
        assert coste["epochs_completed"] == esperadas, f"{tramo}: {coste['epochs_completed']}"
        segundos += Decimal(str(coste["wall_time_s"]))
        epocas += coste["epochs_completed"]

    filas = [
        f
        for f in ExperimentRegistry(registry_dir).read_rows("experiments")
        if f["experiment_id"] == EXPERIMENT_ID
    ]
    assert len(filas) == 1, f"{len(filas)} filas de {EXPERIMENT_ID}"
    fila = filas[0]
    assert (
        Decimal(fila["training_time_s"]) == segundos
    ), f"training_time_s={fila['training_time_s']} y la evidencia suma {segundos}"
    assert (
        int(fila["training_epochs"]) == epocas
    ), f"training_epochs={fila['training_epochs']} y la evidencia suma {epocas}"
    # La constancia del valor original: la corrección no puede borrarla.
    assert "D-185" in fila["notes"] and ORIGINAL_SECONDS in fila["notes"]

    volcado = json.loads(dump_path.read_text(encoding="utf-8"))
    final = json.loads((evidence_dir / "finetune_training_cost.json").read_text(encoding="utf-8"))
    assert Decimal(str(volcado["finetune"]["cost"]["wall_time_s"])) == Decimal(
        str(final["wall_time_s"])
    ), "el sumando final no es el reajuste final que conserva el volcado"
    assert str(volcado["finetune"]["cost"]["wall_time_s"]) == ORIGINAL_SECONDS
    assert volcado["finetune"]["cost"]["epochs_completed"] == FINAL_EPOCHS


def test_celda_evidencia_y_volcado_cuadran() -> None:
    """La fila inscrita es la suma de la evidencia, y el sumando final, el del volcado."""
    check_iterative_cost(REGISTRY_DIR, EVIDENCE_DIR, DUMP_PATH)


def _copias(tmp_path: Path) -> tuple[Path, Path]:
    """Copia registro y evidencia a un directorio desechable."""
    registro = tmp_path / "registry"
    evidencia = tmp_path / "evidencia"
    shutil.copytree(REGISTRY_DIR, registro)
    shutil.copytree(EVIDENCE_DIR, evidencia)
    return registro, evidencia


def test_control_negativo_celda_tocada_sin_evidencia(tmp_path: Path) -> None:
    """Volver a la cifra corta en la celda, sin tocar la evidencia, tiene que fallar."""
    registro, evidencia = _copias(tmp_path)
    ExperimentRegistry(registro).update_cells(
        "experiments",
        [{"experiment_id": EXPERIMENT_ID, "training_time_s": ORIGINAL_SECONDS}],
        on_conflict="overwrite",
    )
    with pytest.raises(AssertionError, match="training_time_s"):
        check_iterative_cost(registro, evidencia, DUMP_PATH)


def test_control_negativo_evidencia_tocada(tmp_path: Path) -> None:
    """Cambiar un sumando de la evidencia sin actualizar su huella tiene que fallar."""
    registro, evidencia = _copias(tmp_path)
    fichero = evidencia / "finetune_step2_training_cost.json"
    datos = json.loads(fichero.read_text(encoding="utf-8"))
    datos["wall_time_s"] = 0.0
    fichero.write_text(json.dumps(datos), encoding="utf-8")
    with pytest.raises(AssertionError, match="sha256"):
        check_iterative_cost(registro, evidencia, DUMP_PATH)
