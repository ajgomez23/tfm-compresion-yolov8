"""Poda en una etapa frente a poda iterativa.

En una etapa se poda hasta el ratio objetivo y se reajusta al final; en la iterativa se alternan
podas pequeñas y reajustes parciales. Las dos se comparan al mismo ratio final. El reajuste lo pasa
quien llama como función; si devuelve None, la métrica queda vacía en el registro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

from tfm_edge.common.config import PruningConfig
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.pruning.structured import PruningResult, PruningSession

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Reajuste parcial entre ciclos: recibe el detector y el número de ciclo (desde 1) y devuelve la
#: métrica o None.
FinetuneCallback = Callable[["Detector", int], float | None]


@dataclass
class ScheduleStep:
    """Resultado de un ciclo de poda."""

    step: int
    parameters_after: int
    parameter_reduction: float
    metric_after_finetune: float | str = ""
    finetuned: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el ciclo como diccionario serializable."""
        return {
            "step": self.step,
            "parameters_after": self.parameters_after,
            "parameter_reduction": self.parameter_reduction,
            "metric_after_finetune": self.metric_after_finetune,
            "finetuned": self.finetuned,
        }


@dataclass
class ScheduleResult:
    """Resultado de una campaña de poda completa."""

    mode: str = "one_shot"
    steps: list[ScheduleStep] = field(default_factory=list)
    final: PruningResult | None = None
    notes: str = ""

    @property
    def num_finetunes(self) -> int:
        """Número de reajustes parciales ejecutados."""
        return sum(1 for s in self.steps if s.finetuned)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        return {
            "mode": self.mode,
            "num_steps": len(self.steps),
            "num_finetunes": self.num_finetunes,
            "steps": [s.to_dict() for s in self.steps],
            "final": self.final.to_dict() if self.final else None,
            "notes": self.notes,
        }

    def summary(self) -> str:
        """Genera un resumen legible."""
        lineas = [
            f"Campaña de poda '{self.mode}': {len(self.steps)} ciclo(s), "
            f"{self.num_finetunes} reajuste(s)",
            "",
            f"{'ciclo':>6s} {'parámetros':>13s} {'reducción':>10s} {'métrica':>10s}",
            "-" * 44,
        ]
        for paso in self.steps:
            metrica = (
                f"{paso.metric_after_finetune:>10.4f}"
                if isinstance(paso.metric_after_finetune, float)
                else f"{'—':>10s}"
            )
            lineas.append(
                f"{paso.step:>6d} {paso.parameters_after:>13,d} "
                f"{paso.parameter_reduction:>9.1%} {metrica}"
            )
        return "\n".join(lineas)


def run_pruning_schedule(
    detector: "Detector",
    config: PruningConfig,
    finetune: FinetuneCallback | None = None,
    example_input: torch.Tensor | None = None,
    random_seed: int | None = None,
) -> tuple["Detector", ScheduleResult]:
    """Ejecuta la campaña de poda de la configuración y devuelve el detector podado y el registro.

    Con ``iterative_steps=1`` es poda en una etapa. ``random_seed`` es obligatoria con el criterio
    ``random`` y se ignora con los demás.
    """
    sesion = PruningSession(detector, config, example_input=example_input, random_seed=random_seed)
    modo = "iterative" if sesion.total_steps > 1 else "one_shot"
    resultado = ScheduleResult(
        mode=modo,
        notes=(
            "El reajuste entre ciclos lo aporta quien llama. Una métrica ausente "
            "se registra como celda vacía, nunca como cero."
        ),
    )
    if finetune is None:
        logger.warning(
            "Campaña de poda '%s' sin reajuste: sirve para medir estructura y latencia, no "
            "precisión.",
            modo,
        )

    ultimo: PruningResult | None = None
    while sesion.steps_applied < sesion.total_steps:
        ultimo = sesion.step()
        paso = ScheduleStep(
            step=sesion.steps_applied,
            parameters_after=ultimo.parameters_after,
            parameter_reduction=ultimo.parameter_reduction,
        )
        # El reajuste parcial solo va entre ciclos; el final lo hace finetune.py.
        es_intermedio = sesion.steps_applied < sesion.total_steps
        if finetune is not None and es_intermedio:
            logger.info(
                "Reajuste parcial tras el ciclo %d de %d",
                sesion.steps_applied,
                sesion.total_steps,
            )
            metrica = finetune(sesion.detector, sesion.steps_applied)
            paso.finetuned = True
            paso.metric_after_finetune = metrica if metrica is not None else ""
            if metrica is None:
                logger.warning(
                    "El reajuste del ciclo %d no ha devuelto métrica: se registra "
                    "como ausente, no como cero.",
                    sesion.steps_applied,
                )
        resultado.steps.append(paso)

    resultado.final = ultimo
    logger.info(
        "Campaña '%s' terminada: %d ciclo(s), reducción final %.1f %%",
        modo,
        len(resultado.steps),
        100 * (ultimo.parameter_reduction if ultimo else 0.0),
    )
    return sesion.detector, resultado


def compare_schedules(one_shot: ScheduleResult, iterative: ScheduleResult) -> dict[str, Any]:
    """Compara una campaña en una etapa con una iterativa. Solo es válida si el ratio final
    coincide.
    """
    reduccion_una = one_shot.final.parameter_reduction if one_shot.final else 0.0
    reduccion_iter = iterative.final.parameter_reduction if iterative.final else 0.0
    diferencia = abs(reduccion_una - reduccion_iter)
    # Medio punto porcentual: por debajo, la diferencia viene del redondeo de canales.
    comparable = diferencia <= 0.005

    informe = {
        "one_shot_reduction": reduccion_una,
        "iterative_reduction": reduccion_iter,
        "reduction_gap": diferencia,
        "comparable": comparable,
        "iterative_steps": len(iterative.steps),
        "iterative_finetunes": iterative.num_finetunes,
    }
    if not comparable:
        aviso = (
            f"Las dos campañas no han llegado al mismo tamaño: "
            f"{reduccion_una:.1%} frente a {reduccion_iter:.1%}. La comparación "
            "mide la diferencia de tamaño, no la de estrategia."
        )
        informe["warning"] = aviso
        logger.warning(aviso)
    return informe
