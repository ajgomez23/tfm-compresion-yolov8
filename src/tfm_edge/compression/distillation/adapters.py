"""Adaptadores de canales entre alumno y maestro.

Una convolución 1x1 por pareja de capas lleva los canales del alumno a los del maestro. Se entrena
con el alumno y se descarta al final, así que no afecta a latencia ni tamaño. Los canales se miden
con un pase de prueba, y la proyección se pone aunque coincidan porque el orden de canales de dos
modelos distintos no tiene por qué corresponderse.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from tfm_edge.common.exceptions import TFMEdgeError

logger = logging.getLogger(__name__)


class AdapterError(TFMEdgeError):
    """No se puede construir un adaptador para el emparejamiento pedido."""


def _clave(nombre: str) -> str:
    """Sustituye los puntos del nombre de capa, que ``nn.ModuleDict`` no admite como clave."""
    return nombre.replace(".", "__")


def measure_channels(
    detector: Any, layer_names: Sequence[str], resolution: int = 640, batch_size: int = 1
) -> dict[str, int]:
    """Mide con un pase de prueba los canales de salida de las capas indicadas.

    Lanza AdapterError si una capa no devuelve un tensor 4D, como pasa con las de la cabeza.
    """
    entrada = torch.zeros(
        batch_size, 3, resolution, resolution, device=detector.device, dtype=torch.float32
    )
    medidos: dict[str, int] = {}
    with torch.no_grad(), detector.capture_features(list(layer_names)) as capturadas:
        detector.forward(entrada)
        for nombre in layer_names:
            salida = capturadas.get(nombre)
            if not isinstance(salida, torch.Tensor) or salida.ndim != 4:
                tipo = type(salida).__name__
                raise AdapterError(
                    f"La capa {nombre!r} devuelve {tipo}, no un mapa de características. Usa "
                    "capas del troncal o del cuello."
                )
            medidos[nombre] = int(salida.shape[1])
    logger.info("Canales medidos con un pase real: %s", medidos)
    return medidos


class FeatureAdapters(nn.Module):
    """Convoluciones 1x1 de las características del alumno a las del maestro.

    Se guardan fuera del modelo para que no entren en el checkpoint, en la EMA ni en el recuento de
    parámetros.
    """

    def __init__(self, channel_pairs: dict[str, tuple[int, int]], bias: bool = False) -> None:
        """Crea un adaptador por pareja ``{capa_alumno: (canales_alumno, canales_maestro)}``. Sin
        sesgo por defecto, porque la normalización por canal de la pérdida lo cancela.
        """
        super().__init__()
        if not channel_pairs:
            raise AdapterError("No hay emparejamientos de los que construir adaptadores.")
        self._nombres = dict(channel_pairs)
        self.proyecciones = nn.ModuleDict(
            {
                _clave(nombre): nn.Conv2d(dentro, fuera, kernel_size=1, bias=bias)
                for nombre, (dentro, fuera) in channel_pairs.items()
            }
        )
        logger.info(
            "Adaptadores construidos: %s (%d parámetros, se descartan al terminar)",
            {n: f"{d}->{f}" for n, (d, f) in channel_pairs.items()},
            self.parameter_count(),
        )

    @classmethod
    def from_detectors(
        cls,
        student: Any,
        teacher: Any,
        pairs: Sequence[tuple[str, str]],
        resolution: int = 640,
    ) -> "FeatureAdapters":
        """Construye los adaptadores midiendo los canales de alumno y maestro."""
        capas_maestro = [m for m, _ in pairs]
        capas_alumno = [a for _, a in pairs]
        canales_maestro = measure_channels(teacher, capas_maestro, resolution)
        canales_alumno = measure_channels(student, capas_alumno, resolution)
        emparejados = {
            alumno: (canales_alumno[alumno], canales_maestro[maestro]) for maestro, alumno in pairs
        }
        adaptadores = cls(emparejados)
        return adaptadores.to(student.device)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Proyecta las activaciones del alumno al espacio del maestro."""
        proyectadas = {}
        for nombre, mapa in features.items():
            clave = _clave(nombre)
            if clave not in self.proyecciones:
                raise AdapterError(
                    f"No hay adaptador para {nombre!r}. Hay para: {sorted(self._nombres)}."
                )
            proyectadas[nombre] = self.proyecciones[clave](mapa)
        return proyectadas

    def parameter_count(self) -> int:
        """Número de parámetros de todos los adaptadores."""
        return sum(p.numel() for p in self.parameters())

    def weight_signature(self) -> dict[str, float]:
        """Norma de cada proyección. Si no cambia entre pasos, los adaptadores no están en el
        optimizador.
        """
        return {
            nombre: float(self.proyecciones[_clave(nombre)].weight.detach().norm())
            for nombre in self._nombres
        }

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """Parámetros que hay que entregar al optimizador de los adaptadores."""
        return self.parameters()
