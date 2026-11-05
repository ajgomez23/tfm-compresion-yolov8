"""Entrenamiento, reajuste y destilación.

El Trainer usa la maquinaria de Ultralytics pero no llama a ``YOLO.train()``, que reconstruiría el
modelo desde el YAML y perdería la poda o los nodos de cuantización simulada.
"""

from __future__ import annotations

__all__ = ["callbacks", "schedule", "trainer"]
