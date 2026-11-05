"""Planificación del entrenamiento por sesiones de Colab y calendario de la receta.

A partir de los minutos por época medidos calcula cuántas épocas caben en una sesión y cuántas
sesiones hacen falta; sin medida, falla. También define calentamiento, decaimiento y cierre del
mosaico para que todas las variantes usen la misma receta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Horas útiles que se dan por seguras en una sesión de Colab. Es una cota conservadora, no un dato
#: oficial.
SESSION_LIMIT_HOURS = 12.0

#: Margen para arranque, descompresión del dataset, validación final y guardado.
SESSION_SAFETY_FRACTION = 0.15

#: Épocas finales sin mosaico, como en la receta estándar de YOLOv8.
DEFAULT_CLOSE_MOSAIC_EPOCHS = 10

DEFAULT_WARMUP_EPOCHS = 3.0


class ScheduleError(TFMEdgeError):
    """Error al planificar un entrenamiento."""


@dataclass
class TrainingSchedule:
    """Calendario de una campaña de entrenamiento."""

    epochs: int = 100
    warmup_epochs: float = DEFAULT_WARMUP_EPOCHS
    close_mosaic_epochs: int = DEFAULT_CLOSE_MOSAIC_EPOCHS
    initial_lr: float = 0.01
    final_lr_fraction: float = 0.01
    patience: int = 30
    checkpoint_every_epochs: int = 1

    def __post_init__(self) -> None:
        """Lanza ScheduleError si el calendario es incoherente."""
        if self.epochs < 1:
            raise ScheduleError(
                f"epochs debe ser >= 1 y es {self.epochs}.", "Usa 100 para la campaña completa."
            )
        if self.close_mosaic_epochs >= self.epochs:
            raise ScheduleError(
                f"close_mosaic_epochs ({self.close_mosaic_epochs}) no puede ser "
                f">= epochs ({self.epochs}).",
                "Entrenarías sin mosaico desde el principio, que es otra receta "
                "distinta y no comparable con el resto de variantes.",
            )
        if self.warmup_epochs >= self.epochs:
            raise ScheduleError(
                f"warmup_epochs ({self.warmup_epochs}) no puede ser >= epochs " f"({self.epochs}).",
                "El calentamiento tiene que terminar dentro del entrenamiento.",
            )
        if self.checkpoint_every_epochs < 1:
            raise ScheduleError(
                f"checkpoint_every_epochs debe ser >= 1 y es " f"{self.checkpoint_every_epochs}.",
                "Con sesiones que se cortan, volcar cada época es lo prudente.",
            )

    def to_ultralytics_overrides(self) -> dict[str, Any]:
        """Traduce el calendario a argumentos de Ultralytics."""
        return {
            "epochs": self.epochs,
            "warmup_epochs": self.warmup_epochs,
            "close_mosaic": self.close_mosaic_epochs,
            "lr0": self.initial_lr,
            "lrf": self.final_lr_fraction,
            "patience": self.patience,
            "save_period": self.checkpoint_every_epochs,
        }

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el calendario como diccionario serializable."""
        return asdict(self)


@dataclass
class SessionPlan:
    """Reparto de una campaña entre sesiones, a partir de minutos por época medidos."""

    model_size: str
    epochs: int
    measured_epoch_minutes: float
    total_hours: float
    usable_hours_per_session: float
    epochs_per_session: int
    sessions_needed: float
    fits_in_one_session: bool
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el plan como diccionario serializable."""
        return asdict(self)

    def summary(self) -> str:
        """Genera un resumen legible."""
        veredicto = "cabe en una sesión" if self.fits_in_one_session else "NO cabe en una sesión"
        return (
            f"yolov8{self.model_size}: {self.epochs} épocas x "
            f"{self.measured_epoch_minutes} min = {self.total_hours} h "
            f"({self.sessions_needed} sesiones, {veredicto})"
        )


def plan_sessions(
    model_size: str,
    measured_epoch_minutes: float,
    epochs: int = 100,
    session_limit_hours: float = SESSION_LIMIT_HOURS,
    safety_fraction: float = SESSION_SAFETY_FRACTION,
    source: str = "",
) -> SessionPlan:
    """Calcula cuántas sesiones necesita una campaña a partir del tiempo por época medido.

    Los minutos por época salen de scripts/measure_training_throughput.py. Lanza ScheduleError si el
    valor no es positivo o si no cabe ni una época.
    """
    if not isinstance(measured_epoch_minutes, (int, float)) or measured_epoch_minutes <= 0:
        raise ScheduleError(
            f"measured_epoch_minutes debe ser un número positivo y es "
            f"{measured_epoch_minutes!r}.",
            "Mídelo con scripts/measure_training_throughput.py. No se planifica "
            "sobre una estimación.",
        )
    if not 0.0 <= safety_fraction < 1.0:
        raise ScheduleError(
            f"safety_fraction debe estar en [0, 1) y es {safety_fraction}.", "Usa 0,15."
        )

    utiles = session_limit_hours * (1.0 - safety_fraction)
    por_sesion = int(utiles * 60.0 / measured_epoch_minutes)
    if por_sesion < 1:
        raise ScheduleError(
            f"Ni una sola época cabe en una sesión: una época tarda "
            f"{measured_epoch_minutes} min y solo hay {utiles:.1f} h útiles.",
            "Baja la resolución, el tamaño del modelo o el tamaño del dataset. "
            "Con este ritmo la campaña no es viable.",
        )

    horas = measured_epoch_minutes * epochs / 60.0
    return SessionPlan(
        model_size=model_size,
        epochs=epochs,
        measured_epoch_minutes=round(float(measured_epoch_minutes), 3),
        total_hours=round(horas, 2),
        usable_hours_per_session=round(utiles, 2),
        epochs_per_session=por_sesion,
        sessions_needed=round(horas / utiles, 2),
        fits_in_one_session=horas <= utiles,
        source=source,
    )


def plan_campaign(
    epoch_minutes_by_size: dict[str, float],
    epochs: int = 100,
    session_limit_hours: float = SESSION_LIMIT_HOURS,
    source: str = "",
) -> dict[str, Any]:
    """Planifica varios tamaños de modelo y devuelve un plan por tamaño con los totales."""
    if not epoch_minutes_by_size:
        raise ScheduleError(
            "No hay ninguna medida de tiempo por época.",
            "Ejecuta scripts/measure_training_throughput.py y pásale sus " "resultados.",
        )

    planes = {
        size: plan_sessions(
            size, minutos, epochs=epochs, session_limit_hours=session_limit_hours, source=source
        )
        for size, minutos in sorted(epoch_minutes_by_size.items())
    }
    horas_totales = sum(p.total_hours for p in planes.values())
    no_caben = [s for s, p in planes.items() if not p.fits_in_one_session]

    if no_caben:
        logger.warning(
            "Estos modelos NO caben en una sesión y necesitan reanudación: %s",
            no_caben,
        )

    return {
        "plans": {s: p.to_dict() for s, p in planes.items()},
        "total_hours": round(horas_totales, 2),
        "total_sessions": round(
            horas_totales / (session_limit_hours * (1.0 - SESSION_SAFETY_FRACTION)), 2
        ),
        "sizes_needing_resume": no_caben,
        "epochs": epochs,
        "source": source,
    }


def summarize_campaign(plan: dict[str, Any]) -> str:
    """Genera un resumen legible de un plan de campaña."""
    lineas = [
        f"PLAN DE CAMPAÑA: {plan['epochs']} épocas por modelo",
        "",
        f"{'modelo':9s} {'min/época':>10s} {'horas':>8s} {'ép./sesión':>11s} "
        f"{'sesiones':>9s}  cabe",
        "-" * 60,
    ]
    for size, p in plan["plans"].items():
        lineas.append(
            f"yolov8{size:2s} {p['measured_epoch_minutes']:>10.2f} "
            f"{p['total_hours']:>8.2f} {p['epochs_per_session']:>11d} "
            f"{p['sessions_needed']:>9.2f}  "
            f"{'sí' if p['fits_in_one_session'] else 'NO'}"
        )
    lineas += [
        "",
        f"Total de la campaña: {plan['total_hours']} h " f"({plan['total_sessions']} sesiones)",
    ]
    if plan["sizes_needing_resume"]:
        lineas.append(f"Necesitan reanudación desde checkpoint: {plan['sizes_needing_resume']}")
    if plan.get("source"):
        lineas.append(f"Medidas tomadas de: {plan['source']}")
    return "\n".join(lineas)
