"""Criterios de importancia de canal.

Traduce los nombres de PruningConfig a los de torch-pruning. Añade un criterio aleatorio de control
con generador propio, para que la arquitectura dependa solo de la semilla y no del estado del RNG
global, y comprueba que hay gradientes antes de usar Taylor. La normalización por defecto es
``mean`` para que el reparto global compare capas de distinta magnitud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch
import torch.nn as nn

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.pruning.dependency import PruningError, _importar_torch_pruning

logger = get_logger(__name__)

#: Criterios admitidos, en el vocabulario de ``PruningConfig``.
CRITERIA = ("l1", "l2", "bn_gamma", "taylor", "random")

#: Criterios que necesitan gradientes acumulados antes de podar.
CRITERIA_REQUIRING_GRADIENTS = ("taylor",)

#: Sin normalizar, el reparto global no compara bien capas de magnitudes distintas.
DEFAULT_NORMALIZER = "mean"


class MissingGradientsError(PruningError):
    """Se ha pedido un criterio que necesita gradientes y no los hay."""


@dataclass(frozen=True)
class CriterionSpec:
    """Descripción de un criterio de importancia."""

    name: str
    description: str
    requires_gradients: bool = False
    is_control: bool = False


CRITERION_SPECS: dict[str, CriterionSpec] = {
    "l1": CriterionSpec("l1", "norma L1 de los pesos del filtro"),
    "l2": CriterionSpec("l2", "norma L2 de los pesos del filtro"),
    "bn_gamma": CriterionSpec("bn_gamma", "escalado gamma de la normalización por lotes asociada"),
    "taylor": CriterionSpec(
        "taylor",
        "aproximación de Taylor de primer orden: |peso · gradiente|",
        requires_gradients=True,
    ),
    "random": CriterionSpec(
        "random",
        "azar; línea base de control, no un criterio",
        is_control=True,
    ),
}


def build_seeded_random_importance(seed: int) -> Any:
    """Construye el criterio aleatorio con generador propio.

    La clase se define dentro de la función porque torch-pruning se importa de forma diferida.
    """
    tp = _importar_torch_pruning()

    class SeededRandomImportance(tp.importance.Importance):
        """Importancia aleatoria reproducible: igual que ``tp.importance.RandomImportance`` pero con
        un ``torch.Generator`` propio.
        """

        def __init__(self, seed: int) -> None:
            """Crea el generador del sorteo."""
            self.seed = seed
            self._generator = torch.Generator(device="cpu").manual_seed(seed)

        @torch.no_grad()
        def __call__(self, group: Any, **kwargs: Any) -> torch.Tensor:
            """Sortea una importancia por canal del grupo."""
            _, idxs = group[0]
            return torch.rand(len(idxs), generator=self._generator)

    return SeededRandomImportance(seed)


def build_importance(
    criterion: str,
    normalizer: str | None = DEFAULT_NORMALIZER,
    seed: int | None = None,
) -> Any:
    """Construye el objeto de importancia de torch-pruning para un criterio.

    ``seed`` es obligatoria con ``random`` y no tiene valor por defecto.
    """
    tp = _importar_torch_pruning()
    if criterion not in CRITERIA:
        raise PruningError(
            f"Criterio de importancia desconocido: '{criterion}'.",
            f"Los admitidos son: {', '.join(CRITERIA)}.",
        )
    if criterion == "random" and seed is None:
        raise PruningError(
            "El criterio de control 'random' exige una semilla explícita.",
            "Pasa seed=...; sin ella el sorteo usaría el RNG global y dependería de lo "
            "ejecutado antes.",
        )

    constructores: dict[str, Callable[[], Any]] = {
        "l1": lambda: tp.importance.GroupMagnitudeImportance(p=1, normalizer=normalizer),
        "l2": lambda: tp.importance.GroupMagnitudeImportance(p=2, normalizer=normalizer),
        "bn_gamma": lambda: tp.importance.BNScaleImportance(normalizer=normalizer),
        "taylor": lambda: tp.importance.GroupTaylorImportance(normalizer=normalizer),
        # Los valores ya son uniformes, no hay nada que normalizar.
        "random": lambda: build_seeded_random_importance(seed),  # type: ignore[arg-type]
    }
    especificacion = CRITERION_SPECS[criterion]
    if especificacion.is_control:
        logger.warning(
            "El criterio 'random' es el control, no un criterio para producir modelos del estudio."
        )
    logger.info("Criterio de importancia '%s': %s", criterion, especificacion.description)
    return constructores[criterion]()


def has_gradients(model: nn.Module) -> bool:
    """Indica si el modelo tiene gradientes acumulados en sus convoluciones."""
    return any(isinstance(m, nn.Conv2d) and m.weight.grad is not None for m in model.modules())


def assert_gradients_available(model: nn.Module, criterion: str) -> None:
    """Comprueba que hay gradientes si el criterio los necesita."""
    if criterion not in CRITERIA_REQUIRING_GRADIENTS:
        return
    if has_gradients(model):
        logger.debug("Gradientes disponibles para el criterio '%s'", criterion)
        return
    raise MissingGradientsError(
        f"El criterio '{criterion}' necesita gradientes y el modelo no tiene " "ninguno acumulado.",
        "Acumula gradientes con accumulate_gradients() sobre lotes reales de entrenamiento, o "
        "usa 'l1' o 'l2', que solo miran los pesos.",
    )


def accumulate_gradients(
    model: nn.Module,
    batches: Iterable[Any],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    max_batches: int | None = None,
) -> int:
    """Acumula gradientes sobre lotes reales para los criterios que los necesitan.

    La pérdida la pasa quien llama, porque necesita las etiquetas en formato de Ultralytics.
    """
    model.zero_grad(set_to_none=True)
    model.train()
    procesados = 0
    for lote in batches:
        if max_batches is not None and procesados >= max_batches:
            break
        salida = model(lote[0] if isinstance(lote, (tuple, list)) else lote)
        perdida = loss_fn(salida, lote)
        perdida.backward()
        procesados += 1

    if procesados == 0 or not has_gradients(model):
        raise MissingGradientsError(
            f"Tras recorrer {procesados} lotes no hay gradientes acumulados.",
            "Comprueba que los lotes no están vacíos y que la pérdida depende de los pesos del "
            "modelo.",
        )
    logger.info("Gradientes acumulados sobre %d lotes reales", procesados)
    return procesados


def describe_criteria() -> str:
    """Genera una descripción legible de los criterios disponibles."""
    lineas = [f"{'criterio':10s} {'gradientes':11s} descripción", "-" * 72]
    for especificacion in CRITERION_SPECS.values():
        marca = "sí" if especificacion.requires_gradients else "no"
        sufijo = "  [CONTROL]" if especificacion.is_control else ""
        lineas.append(f"{especificacion.name:10s} {marca:11s} {especificacion.description}{sufijo}")
    return "\n".join(lineas)
