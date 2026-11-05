"""Lectura del coste de entrenamiento que el Trainer vuelca junto a cada checkpoint.

Las claves se declaran una sola vez y tests/test_training_cost.py las contrasta con TrainingCost,
porque una clave mal escrita deja la celda vacía sin dar error. En training_device se guarda el
nombre de la GPU (por ejemplo Tesla T4) y no ``cuda:0``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

TRAINING_COST_FILENAME = "training_cost.json"

#: Claves de training_cost.json que pasan al registro. Deben existir en TrainingCost.
COST_FIELD_EPOCHS = "epochs_completed"
COST_FIELD_SECONDS = "wall_time_s"
COST_FIELD_DEVICE = "device"
COST_FIELD_GPU_NAME = "gpu_name"

REGISTRY_COLUMNS = ("training_epochs", "training_time_s", "training_device")


def locate_run_file(checkpoint: Path | str, filename: str) -> Path | None:
    """Busca un fichero de la ejecución junto a un checkpoint.

    Contempla ``<run>/weights/best.pt`` (como lo deja Ultralytics) y ``<dir>/best.pt`` (pesos ya
    recogidos). Devuelve None si no está.
    """
    ruta = Path(checkpoint)
    for candidato in (ruta.parent / filename, ruta.parent.parent / filename):
        if candidato.is_file():
            return candidato
    return None


def locate_training_cost(checkpoint: Path | str) -> Path | None:
    """Localiza el ``training_cost.json`` que acompaña a un checkpoint."""
    return locate_run_file(checkpoint, TRAINING_COST_FILENAME)


def read_training_cost(checkpoint: Path | str) -> dict[str, Any]:
    """Lee el coste de entrenamiento de un checkpoint con las columnas del registro.

    Lo que no se pueda leer queda como cadena vacía, nunca como 0.
    """
    vacio: dict[str, Any] = {columna: "" for columna in REGISTRY_COLUMNS}

    ruta = locate_training_cost(checkpoint)
    if ruta is None:
        logger.debug("Sin '%s' junto a '%s'.", TRAINING_COST_FILENAME, checkpoint)
        return vacio

    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("No se ha podido leer '%s': %s", ruta, error)
        return vacio

    if not isinstance(datos, dict):
        logger.warning("'%s' no contiene un objeto JSON; se ignora.", ruta)
        return vacio

    return cost_dict_to_registry(datos, origen=str(ruta))


def cost_dict_to_registry(cost: dict[str, Any], origen: str = "") -> dict[str, Any]:
    """Traduce un coste ya cargado a las columnas del registro."""
    fila: dict[str, Any] = {columna: "" for columna in REGISTRY_COLUMNS}

    epocas = cost.get(COST_FIELD_EPOCHS, "")
    segundos = cost.get(COST_FIELD_SECONDS, "")

    # Cero épocas o cero segundos indica un entrenamiento que no llegó a correr.
    if isinstance(epocas, int) and epocas > 0:
        fila["training_epochs"] = epocas
    if isinstance(segundos, (int, float)) and segundos > 0:
        fila["training_time_s"] = round(float(segundos), 2)

    # gpu_name identifica la máquina; 'device' solo da el índice.
    fila["training_device"] = cost.get(COST_FIELD_GPU_NAME) or cost.get(COST_FIELD_DEVICE, "") or ""

    if not fila["training_epochs"] or not fila["training_time_s"]:
        logger.warning(
            "El coste de entrenamiento%s está incompleto: épocas=%r, segundos=%r. "
            "Las columnas que falten quedan vacías.",
            f" de '{origen}'" if origen else "",
            epocas,
            segundos,
        )

    return fila


def seconds_per_epoch(cost: dict[str, Any]) -> float | None:
    """Segundos por época a partir de un coste ya leído, o None si falta algún dato."""
    epocas = cost.get("training_epochs", "")
    segundos = cost.get("training_time_s", "")
    if not isinstance(epocas, int) or epocas <= 0:
        return None
    if not isinstance(segundos, (int, float)) or segundos <= 0:
        return None
    return float(segundos) / epocas
