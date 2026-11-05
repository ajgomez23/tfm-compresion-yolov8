"""Obtención, validación, preparación y análisis del dataset SH17.

Se usa en el orden de los scripts 01 a 03: download, prepare, splits, statistics, calibration_set y
eda. ``processed`` no copia las imágenes, así que todo lo que lo usa comprueba antes que ``raw``
sigue presente.
"""

from __future__ import annotations

__all__ = [
    "calibration_set",
    "download",
    "eda",
    "prepare",
    "splits",
    "statistics",
]
