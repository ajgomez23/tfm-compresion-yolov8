"""Selección de configuraciones que cumplen unas restricciones de despliegue.

Dados límites de latencia, memoria, tamaño y mAP en un dispositivo, devuelve las configuraciones
viables ordenadas. Cada restricción da cumple, en el margen (dentro del suelo de medida) o no
cumple, y las del margen se devuelven marcadas. El rango agrupa las configuraciones indistinguibles
de la mejor del grupo, comparando siempre con esa mejor porque estar dentro del suelo no es
transitivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tfm_edge.analysis.floors import FloorSet
from tfm_edge.analysis.pareto import Point, same_session
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

VERDICT_MEETS = "cumple"
VERDICT_BORDERLINE = "en el margen"
VERDICT_FAILS = "no cumple"

#: ``max`` es un techo y ``min`` un suelo.
LIMIT_DIRECTIONS = ("max", "min")


class SelectionError(TFMEdgeError):
    """Error al seleccionar configuraciones."""


@dataclass(frozen=True)
class Constraint:
    """Límite que una configuración tiene que respetar."""

    name: str
    column: str
    limit: float
    direction: str

    def __post_init__(self) -> None:
        """Lanza SelectionError si la dirección no es ``max`` ni ``min``."""
        if self.direction not in LIMIT_DIRECTIONS:
            raise SelectionError(
                f"La restricción '{self.name}' declara dirección '{self.direction}', y "
                f"solo valen {list(LIMIT_DIRECTIONS)}."
            )


@dataclass(frozen=True)
class ConstraintCheck:
    """Resultado de aplicar una restricción a una configuración."""

    constraint: Constraint
    value: float | None
    verdict: str
    tolerance: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Serializa la comprobación."""
        return {
            "name": self.constraint.name,
            "column": self.constraint.column,
            "limit": self.constraint.limit,
            "direction": self.constraint.direction,
            "value": self.value,
            "verdict": self.verdict,
            "tolerance": self.tolerance,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class Candidate:
    """Configuración evaluada contra todas las restricciones. Las indistinguibles comparten
    rango."""

    point: Point
    checks: tuple[ConstraintCheck, ...]
    rank: int

    @property
    def viable(self) -> bool:
        """True si ninguna restricción la descarta."""
        return all(c.verdict != VERDICT_FAILS for c in self.checks)

    @property
    def borderline(self) -> bool:
        """True si alguna restricción la deja en el margen."""
        return any(c.verdict == VERDICT_BORDERLINE for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        """Serializa el candidato."""
        return {
            "experiment_id": self.point.experiment_id,
            "runtime": self.point.runtime,
            "precision": self.point.precision,
            "device_id": self.point.device_id,
            "label": self.point.label,
            "rank": self.rank,
            "viable": self.viable,
            "borderline": self.borderline,
            "values": {k: v for k, v in sorted(self.point.values.items())},
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass(frozen=True)
class SelectionResult:
    """Resultado de una consulta de selección."""

    device_id: str
    constraints: tuple[Constraint, ...]
    order_by: str
    order_direction: str
    candidates: tuple[Candidate, ...]
    excluded: tuple[Candidate, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resultado."""
        return {
            "device_id": self.device_id,
            "order_by": self.order_by,
            "order_direction": self.order_direction,
            "constraints": [
                {
                    "name": c.name,
                    "column": c.column,
                    "limit": c.limit,
                    "direction": c.direction,
                }
                for c in self.constraints
            ],
            "counts": {
                "evaluated": len(self.candidates) + len(self.excluded),
                "viable": len(self.candidates),
                "borderline": sum(1 for c in self.candidates if c.borderline),
                "excluded": len(self.excluded),
            },
            "candidates": [c.to_dict() for c in self.candidates],
            "excluded": [c.to_dict() for c in self.excluded],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Scenario:
    """Conjunto de restricciones con nombre para los ejemplos del apartado 6.7."""

    name: str
    description: str
    constraints: tuple[Constraint, ...]
    order_by: str
    order_direction: str = "min"


def example_scenarios() -> tuple[Scenario, ...]:
    """Escenarios de ejemplo de la selección.

    Los límites son una elección para ilustrar el módulo, no un resultado: 100 ms de p95 (10 fps a
    lote 1), 500 MiB de memoria y 0,30 de mAP@50-95.
    """
    return (
        Scenario(
            name="tiempo_real_10fps",
            description=(
                "Latencia p95 por debajo de 100 ms a lote 1, con la calidad mínima "
                "que el estudio considera utilizable"
            ),
            constraints=(
                Constraint("latencia p95 máxima", "latency_p95_ms", 100.0, "max"),
                Constraint("mAP@50-95 mínimo", "map_50_95", 0.30, "min"),
            ),
            order_by="latency_p95_ms",
        ),
        Scenario(
            name="memoria_contenida",
            description=("Memoria pico por debajo de 500 MiB, sin exigir nada a la latencia"),
            constraints=(
                Constraint("memoria pico máxima", "peak_host_memory_mb", 500.0, "max"),
                Constraint("mAP@50-95 mínimo", "map_50_95", 0.30, "min"),
            ),
            order_by="peak_host_memory_mb",
        ),
        Scenario(
            name="calidad_primero",
            description=(
                "La mejor calidad alcanzable dentro de un presupuesto de latencia "
                "holgado, para ver qué se pierde al apretar"
            ),
            constraints=(Constraint("latencia p95 máxima", "latency_p95_ms", 200.0, "max"),),
            order_by="map_50_95",
            order_direction="max",
        ),
    )


def _tolerance(column: str, value: float, limit: float, floors: FloorSet) -> tuple[float, str]:
    """Margen al comparar un valor con un límite. Devuelve el margen y su origen.

    Se usa el suelo entre sesiones porque el límite se refiere a una ejecución futura.
    """
    suelo = floors.get(column)
    if suelo is None or not suelo.is_measured():
        return 0.0, (
            f"El eje '{column}' no tiene suelo medido, así que el límite se aplica como "
            f"una línea exacta: una configuración que se quede a un pelo quedará fuera "
            f"sin que nada avise (D-127)."
        )
    aplicable = suelo.applicable(same_session=False)
    if aplicable is None:
        return 0.0, f"El eje '{column}' no tiene suelo entre sesiones medido."
    if suelo.kind == "relativo":
        margen = aplicable * min(abs(value), abs(limit))
        return margen, (
            f"Margen del {aplicable:.1%} sobre el menor de valor y límite, que es la "
            f"resolución medida del eje ({suelo.provenance})."
        )
    return aplicable, (
        f"Margen absoluto de {aplicable:g}, que es la resolución medida del eje "
        f"({suelo.provenance})."
    )


def check_constraint(point: Point, constraint: Constraint, floors: FloorSet) -> ConstraintCheck:
    """Evalúa una restricción sobre un punto teniendo en cuenta el suelo de medida."""
    valor = point.values.get(constraint.column)
    if valor is None:
        return ConstraintCheck(
            constraint=constraint,
            value=None,
            verdict=VERDICT_FAILS,
            tolerance=0.0,
            explanation=(
                f"No hay valor medido de '{constraint.column}' para esta configuración. "
                f"Sin la medida no se puede afirmar que cumpla, y suponerlo sería "
                f"colarla en una lista de configuraciones viables."
            ),
        )

    margen, origen = _tolerance(constraint.column, valor, constraint.limit, floors)
    exceso = valor - constraint.limit if constraint.direction == "max" else constraint.limit - valor

    if exceso <= -margen:
        veredicto = VERDICT_MEETS
        explicacion = f"Cumple con holgura por encima de la resolución de la medida. {origen}"
    elif exceso <= margen:
        veredicto = VERDICT_BORDERLINE
        explicacion = (
            f"Se queda a {abs(exceso):g} del límite y la resolución del eje es {margen:g}: "
            f"la medida no distingue esta configuración de una que lo cumpla justo. "
            f"Se devuelve marcada en lugar de descartarla, porque descartarla sería "
            f"afirmar una diferencia que el instrumento no ve. {origen}"
        )
    else:
        veredicto = VERDICT_FAILS
        explicacion = f"Supera el límite en {exceso:g}, por encima de la resolución. {origen}"

    return ConstraintCheck(
        constraint=constraint,
        value=valor,
        verdict=veredicto,
        tolerance=margen,
        explanation=explicacion,
    )


def _assign_ranks(
    candidates: Sequence[tuple[Point, tuple[ConstraintCheck, ...]]],
    order_by: str,
    order_direction: str,
    floors: FloorSet,
) -> list[Candidate]:
    """Ordena los candidatos y agrupa en un mismo rango los indistinguibles del mejor de cada grupo.

    El suelo de cada comparación es el de la misma sesión o el de entre sesiones según la huella de
    entorno de los dos puntos (pareto.same_session).
    """
    faltan = [p.label for p, _ in candidates if order_by not in p.values]
    if faltan:
        raise SelectionError(
            f"No se puede ordenar por '{order_by}': {len(faltan)} configuraciones no lo tienen "
            f"medido ({faltan[:3]})."
        )

    restantes = sorted(
        candidates,
        key=lambda par: (
            par[0].values[order_by] if order_direction == "min" else -par[0].values[order_by],
            par[0].label,
        ),
    )

    suelo = floors.get(order_by)
    resultado: list[Candidate] = []
    rango = 1
    while restantes:
        punto_mejor = restantes[0][0]
        mejor = punto_mejor.values[order_by]
        grupo: list[tuple[Point, tuple[ConstraintCheck, ...]]] = []
        resto: list[tuple[Point, tuple[ConstraintCheck, ...]]] = []
        for punto, comprobaciones in restantes:
            valor = punto.values[order_by]
            if suelo is not None and suelo.is_measured():
                aplicable = suelo.applicable(same_session(punto, punto_mejor)) or 0.0
                margen = (
                    aplicable * min(abs(mejor), abs(valor))
                    if suelo.kind == "relativo"
                    else aplicable
                )
            else:
                margen = 0.0
            if abs(valor - mejor) <= margen:
                grupo.append((punto, comprobaciones))
            else:
                resto.append((punto, comprobaciones))
        for punto, comprobaciones in grupo:
            resultado.append(Candidate(point=punto, checks=comprobaciones, rank=rango))
        restantes = resto
        rango += 1
    return resultado


def select(
    points: Sequence[Point],
    constraints: Sequence[Constraint],
    floors: FloorSet,
    device_id: str,
    order_by: str,
    order_direction: str = "min",
) -> SelectionResult:
    """Configuraciones viables de un dispositivo, ordenadas, con las descartadas aparte."""
    if order_direction not in LIMIT_DIRECTIONS:
        raise SelectionError(
            f"La dirección de ordenación '{order_direction}' no vale; solo "
            f"{list(LIMIT_DIRECTIONS)}."
        )

    del_dispositivo = [p for p in points if p.device_id == device_id]
    if not del_dispositivo:
        raise SelectionError(f"No hay ningún punto medido en '{device_id}'.")

    evaluados = [
        (punto, tuple(check_constraint(punto, r, floors) for r in constraints))
        for punto in del_dispositivo
    ]
    viables = [par for par in evaluados if all(c.verdict != VERDICT_FAILS for c in par[1])]
    descartados = [par for par in evaluados if any(c.verdict == VERDICT_FAILS for c in par[1])]

    ordenados = _assign_ranks(viables, order_by, order_direction, floors)
    fuera = [Candidate(point=p, checks=c, rank=0) for p, c in descartados]
    fuera.sort(key=lambda c: c.point.label)

    notas = [
        "Las configuraciones marcadas «en el margen» se quedan a menos de la resolución "
        "de la medida del límite: la medida no las distingue de una que lo cumpla justo. "
        "Se devuelven porque descartarlas afirmaría una diferencia que el instrumento no "
        "ve, no porque se sepa que valen.",
        "Las que comparten rango son indistinguibles entre sí en el eje de ordenación. "
        "Presentar una de ellas como la mejor sería afirmar una ventaja por debajo de la "
        "resolución (D-128).",
    ]
    sin_suelo = sorted(
        {
            r.column
            for r in constraints
            if floors.get(r.column) is None or not floors.get(r.column).is_measured()  # type: ignore[union-attr]
        }
    )
    if sin_suelo:
        notas.append(
            f"Sin suelo medido, y por tanto aplicados como línea exacta: "
            f"{', '.join(sin_suelo)}. En esos ejes una configuración puede quedar fuera "
            f"por una diferencia que este proyecto no ha certificado (D-127)."
        )

    logger.info(
        "Selección en %s: %d configuraciones evaluadas, %d viables (%d en el margen), "
        "%d descartadas.",
        device_id,
        len(evaluados),
        len(ordenados),
        sum(1 for c in ordenados if c.borderline),
        len(fuera),
    )
    return SelectionResult(
        device_id=device_id,
        constraints=tuple(constraints),
        order_by=order_by,
        order_direction=order_direction,
        candidates=tuple(ordenados),
        excluded=tuple(fuera),
        notes=tuple(notas),
    )
