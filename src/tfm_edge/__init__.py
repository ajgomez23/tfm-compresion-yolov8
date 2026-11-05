"""Optimización de detectores YOLOv8 para inferencia en el borde.

Compara cuantización, poda estructurada y destilación con un protocolo de medida común
(tfm_edge.benchmark.protocol) y un registro único de resultados (tfm_edge.common.registry).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
