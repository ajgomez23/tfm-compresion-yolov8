"""Utilidades comunes: configuración, rutas, logging, semillas y registro.

No depende de PyTorch ni de Ultralytics, para poder escribir el registro en un equipo que solo tenga
un runtime de inferencia.
"""

from __future__ import annotations

from tfm_edge.common.exceptions import (
    ConfigError,
    DependencyNotAvailableError,
    HardwareNotAvailableError,
    RegistryError,
    TFMEdgeError,
)
from tfm_edge.common.logging_utils import get_logger, setup_logging
from tfm_edge.common.paths import ProjectPaths, build_experiment_id, get_project_paths
from tfm_edge.common.seed import set_global_seed

__all__ = [
    "ConfigError",
    "DependencyNotAvailableError",
    "HardwareNotAvailableError",
    "ProjectPaths",
    "RegistryError",
    "TFMEdgeError",
    "build_experiment_id",
    "get_logger",
    "get_project_paths",
    "set_global_seed",
    "setup_logging",
]
