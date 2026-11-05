"""Captura de activaciones intermedias para la destilación.

FeatureCollector rechaza capturar dos veces la misma capa sin vaciarse, para no mezclar pases, y
falla si una capa no devuelve un tensor. Los hooks son métodos de clase y no funciones locales
porque el checkpoint serializa el modelo entero y una función local no se puede guardar.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import torch
import torch.nn as nn

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.models.detector import drop_forward_hooks

logger = logging.getLogger(__name__)


class FeatureCaptureError(TFMEdgeError):
    """La captura de activaciones ha recibido algo que no puede usar."""


class FeatureCollector:
    """Guarda las activaciones de unas capas durante un paso. No es un nn.Module para no aparecer en
    el state_dict.
    """

    def __init__(self, owner: str = "modelo", strict: bool = True) -> None:
        """Prepara el colector. Con ``strict`` falla si una capa se captura dos veces sin vaciar."""
        self.owner = owner
        self.strict = strict
        self.features: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

    def _hook(self, nombre: str):
        """Crea el hook de una capa como método, para que el modelo se pueda serializar."""

        def _capturar(_modulo: nn.Module, _entrada: Any, salida: Any) -> None:
            if not isinstance(salida, torch.Tensor):
                raise FeatureCaptureError(
                    f"La capa {nombre!r} de {self.owner} devuelve {type(salida).__name__}, no un "
                    "tensor. Usa capas del troncal o del cuello."
                )
            if self.strict and nombre in self.features:
                raise FeatureCaptureError(
                    f"La capa {nombre!r} de {self.owner} se ha capturado dos veces. Llama a "
                    "clear() entre pases."
                )
            self.features[nombre] = salida

        return _capturar

    def attach(self, detector: Any, layer_names: Sequence[str]) -> None:
        """Registra los hooks en las capas indicadas."""
        if self._handles:
            raise FeatureCaptureError(
                f"El colector de {self.owner} ya tiene {len(self._handles)} hooks "
                "registrados. Retíralos con detach() antes de volver a registrar."
            )
        for nombre in layer_names:
            modulo = detector.get_module(nombre)
            self._handles.append(modulo.register_forward_hook(self._hook(nombre)))
        logger.info(
            "Instrumentadas %d capas de %s: %s", len(self._handles), self.owner, list(layer_names)
        )

    def detach(self) -> int:
        """Retira los hooks, vacía lo capturado y devuelve cuántos hooks había."""
        retirados = len(self._handles)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()
        if retirados:
            logger.info("Retirados %d hooks de %s", retirados, self.owner)
        return retirados

    def clear(self) -> None:
        """Suelta las activaciones del paso anterior para liberar su grafo y la memoria de GPU."""
        self.features.clear()

    def require(self, layer_names: Sequence[str]) -> dict[str, torch.Tensor]:
        """Devuelve las activaciones pedidas y falla si falta alguna."""
        faltan = [n for n in layer_names if n not in self.features]
        if faltan:
            raise FeatureCaptureError(
                f"No se han capturado las activaciones de {faltan} en {self.owner}. Capturadas: "
                f"{sorted(self.features)}."
            )
        return {n: self.features[n] for n in layer_names}

    @property
    def attached(self) -> int:
        """Número de hooks vivos."""
        return len(self._handles)

    def __deepcopy__(self, memo: dict) -> "FeatureCollector":
        """Copia vacía y sin hooks. Ultralytics hace una copia profunda del modelo al rehacer la
        EMA.
        """
        vacio = FeatureCollector(owner=self.owner, strict=self.strict)
        memo[id(self)] = vacio
        return vacio

    def __getstate__(self) -> dict:
        """Nunca se serializan ni las activaciones ni los hooks."""
        return {"owner": self.owner, "strict": self.strict, "features": {}, "_handles": []}


@contextmanager
def capture_pairs(
    teacher: Any, student: Any, pairs: Sequence[tuple[str, str]]
) -> Iterator[tuple[FeatureCollector, FeatureCollector]]:
    """Instrumenta las capas emparejadas de maestro y alumno y retira los hooks al salir, también si
    hay excepción.
    """
    colector_maestro = FeatureCollector(owner="maestro")
    colector_alumno = FeatureCollector(owner="alumno")
    try:
        colector_maestro.attach(teacher, [m for m, _ in pairs])
        colector_alumno.attach(student, [a for _, a in pairs])
        yield colector_maestro, colector_alumno
    finally:
        colector_maestro.detach()
        colector_alumno.detach()


def deepcopy_without_hooks(model: nn.Module) -> nn.Module:
    """Copia profunda de un modelo con los hooks ya soltados."""
    copia = copy.deepcopy(model)
    drop_forward_hooks(copia)
    return copia
