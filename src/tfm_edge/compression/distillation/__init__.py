"""Destilación de conocimiento de un maestro grande a un alumno pequeño.

YOLOv8 clasifica con un sigmoide por clase y casi todos los anclajes son fondo, por eso se usa T=2 y
el término de respuesta se enmascara con la confianza del maestro. Módulos: losses, adapters,
feature_hooks y trainer.
"""

from tfm_edge.compression.distillation.adapters import FeatureAdapters, measure_channels
from tfm_edge.compression.distillation.feature_hooks import FeatureCollector, capture_pairs
from tfm_edge.compression.distillation.losses import (
    DistillationComponents,
    DistillationLoss,
    HeadGeometry,
    class_kl_bernoulli,
    dfl_kl,
    feature_mse,
    ground_truth_box_mask,
    head_geometry,
    masked_mean,
    split_head_output,
    teacher_confidence_mask,
)
from tfm_edge.compression.distillation.preflight import (
    PreflightError,
    PreflightResult,
    run_preflight,
)
from tfm_edge.compression.distillation.reading import (
    EVALUATOR_READABILITY_FLOOR,
    ExtensionReading,
    PilotReading,
    read_extension,
    read_pilot,
)
from tfm_edge.compression.distillation.trainer import (
    DistillationError,
    DistillationResult,
    DistillationTrainer,
)

__all__ = [
    "EVALUATOR_READABILITY_FLOOR",
    "DistillationComponents",
    "DistillationError",
    "DistillationLoss",
    "DistillationResult",
    "DistillationTrainer",
    "ExtensionReading",
    "FeatureAdapters",
    "FeatureCollector",
    "PreflightError",
    "PreflightResult",
    "HeadGeometry",
    "PilotReading",
    "capture_pairs",
    "class_kl_bernoulli",
    "dfl_kl",
    "feature_mse",
    "ground_truth_box_mask",
    "head_geometry",
    "masked_mean",
    "measure_channels",
    "read_extension",
    "read_pilot",
    "run_preflight",
    "split_head_output",
    "teacher_confidence_mask",
]
