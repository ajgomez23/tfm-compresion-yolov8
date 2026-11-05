"""Identidad de una evaluación de baseline, compartida por los pasos 05 y 06.

Los dos evaluadores tienen que dar el mismo experiment_id. El identificador depende de la
configuración de la evaluación (umbrales, resolución, partición, semilla), no de la del
entrenamiento.
"""

from __future__ import annotations

from typing import Any

from tfm_edge.common.config import compute_config_hash
from tfm_edge.common.paths import build_experiment_id


def build_baseline_evaluation_identity(
    *,
    model_size: str,
    num_classes: int,
    input_resolution: int,
    conf_threshold: float,
    iou_threshold: float,
    split: str,
    data: str,
    seed: int,
) -> tuple[dict[str, Any], str, str]:
    """Construye el payload de una evaluación de baseline y devuelve ``(payload, config_hash,
    experiment_id)``.

    ``data`` es el nombre del YAML del dataset y no su ruta, para que el identificador no dependa de
    la máquina.
    """
    payload = {
        "model_family": "yolov8",
        "model_size": model_size,
        "technique": "baseline",
        "variant": "default",
        "num_classes": num_classes,
        "input_resolution": input_resolution,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "split": split,
        "data": data,
        "seed": seed,
    }
    config_hash = compute_config_hash(payload)
    experiment_id = build_experiment_id(
        model_size=model_size,
        technique="baseline",
        variant="default",
        config_hash=config_hash,
    )
    return payload, config_hash, experiment_id
