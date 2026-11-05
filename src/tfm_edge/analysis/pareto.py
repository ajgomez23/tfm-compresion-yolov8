"""Frontera de Pareto que tiene en cuenta el suelo de medida de cada eje.

Calcula dos fronteras. La estricta es la clásica y compara números. En la robusta un punto domina a
otro solo si le gana por encima del suelo en al menos un eje, con el suelo medido, y no le pierde en
ninguno, por poco que sea. El suelo solo se aplica a la ventaja, así que la frontera robusta siempre
contiene a la estricta. En un eje sin suelo medido (hoy ``map_50``) no se certifica ninguna ventaja.
La dominación robusta no es transitiva, por eso se compara cada punto contra todos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.analysis.floors import FloorSet, same_environment
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Modos de dominación.
MODE_STRICT: str = "estricta"
MODE_ROBUST: str = "robusta"

#: Claves con las que una fila de calidad y una de despliegue son la misma configuración.
#: build_points comprueba en cada ejecución que el cruce es uno a uno.
JOIN_KEYS: tuple[str, ...] = ("experiment_id", "runtime", "precision", "device_id")


class ParetoError(TFMEdgeError):
    """El análisis no se puede hacer con lo que hay."""


@dataclass(frozen=True)
class Objective:
    """Eje de la frontera."""

    name: str
    column: str
    source: str
    direction: str

    def __post_init__(self) -> None:
        """Lanza ParetoError si la dirección no es ``"min"`` ni ``"max"``."""
        if self.direction not in ("min", "max"):
            raise ParetoError(
                f"El objetivo '{self.name}' declara dirección '{self.direction}', "
                f"y solo valen 'min' y 'max'."
            )


@dataclass(frozen=True)
class Point:
    """Configuración medida con sus valores en todos los ejes."""

    experiment_id: str
    runtime: str
    precision: str
    device_id: str
    environment_id: str
    timestamp: str
    values: Mapping[str, float]

    @property
    def key(self) -> tuple[str, str, str, str]:
        """Identidad de la configuración."""
        return (self.experiment_id, self.runtime, self.precision, self.device_id)

    @property
    def label(self) -> str:
        """Etiqueta única con experimento, motor y precisión."""
        return f"{self.experiment_id} [{self.runtime}/{self.precision}]"


@dataclass(frozen=True)
class Comparison:
    """Resultado de comparar dos puntos en un eje.

    ``advantage`` es positiva si ``a`` es mejor, se maximice o se minimice el eje. ``certain``
    indica si el veredicto se apoya en un suelo medido.
    """

    objective: str
    column: str
    value_a: float
    value_b: float
    advantage: float
    threshold: float | None
    verdict: str
    certain: bool

    def to_dict(self) -> dict[str, Any]:
        """Serializa la comparación."""
        return {
            "objective": self.objective,
            "column": self.column,
            "value_a": self.value_a,
            "value_b": self.value_b,
            "advantage": self.advantage,
            "threshold": self.threshold,
            "verdict": self.verdict,
            "certain": self.certain,
        }


@dataclass(frozen=True)
class Domination:
    """Dominación de un punto sobre otro, con el detalle eje por eje."""

    dominator: str
    dominated: str
    mode: str
    same_session: bool
    comparisons: tuple[Comparison, ...]

    @property
    def deciding_objectives(self) -> tuple[str, ...]:
        """Ejes en los que la ventaja está certificada."""
        return tuple(c.objective for c in self.comparisons if c.verdict == "mejor" and c.certain)

    def to_dict(self) -> dict[str, Any]:
        """Serializa la dominación."""
        return {
            "dominator": self.dominator,
            "dominated": self.dominated,
            "mode": self.mode,
            "same_session": self.same_session,
            "deciding_objectives": list(self.deciding_objectives),
            "comparisons": [c.to_dict() for c in self.comparisons],
        }


@dataclass(frozen=True)
class FrontierResult:
    """Resultado de calcular una frontera."""

    mode: str
    nondominated: tuple[Point, ...]
    dominations: tuple[Domination, ...]
    indistinguishable: tuple[tuple[str, str], ...]

    def dominators_of(self, label: str) -> tuple[Domination, ...]:
        """Dominaciones que tienen al punto indicado como dominado."""
        return tuple(d for d in self.dominations if d.dominated == label)


def named_views() -> dict[str, tuple[Objective, ...]]:
    """Ejes de cada vista del estudio.

    Se definen aquí porque los usan el análisis, las figuras y las tablas. La primera vista es mAP,
    p95 y memoria; las otras dos usan la latencia mediana.
    """
    return {
        "spec_3d_map5095_p95_memoria": default_objectives(),
        "plano_map50_p50": (
            Objective("mAP@50", "map_50", "quality", "max"),
            Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
        ),
        "plano_map5095_p50": (
            Objective("mAP@50-95", "map_50_95", "quality", "max"),
            Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
        ),
    }


def default_objectives() -> tuple[Objective, ...]:
    """Objetivos por defecto: maximizar mAP@50-95 y minimizar la latencia p95 y la memoria pico.

    Se usa p95 aunque su suelo medido (14,7 %) sea peor que el de p50 (9,3 %).
    """
    return (
        Objective("mAP@50-95", "map_50_95", "quality", "max"),
        Objective("latencia p95", "latency_p95_ms", "deployment", "min"),
        Objective("memoria pico", "peak_host_memory_mb", "deployment", "min"),
    )


def build_points(
    quality_rows: Sequence[Mapping[str, str]],
    deployment_rows: Sequence[Mapping[str, str]],
    objectives: Sequence[Objective],
) -> tuple[Point, ...]:
    """Cruza calidad y despliegue y devuelve los puntos completos, ordenados por etiqueta.

    Una fila de despliegue sin pareja de calidad se descarta. Si casa con más de una, se lanza
    ParetoError.
    """
    indice: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for fila in quality_rows:
        indice.setdefault(tuple(fila[c] for c in JOIN_KEYS), []).append(fila)

    puntos: list[Point] = []
    sin_calidad = 0
    sin_valor = 0
    for fila in deployment_rows:
        clave = tuple(fila[c] for c in JOIN_KEYS)
        candidatas = indice.get(clave, [])
        if not candidatas:
            sin_calidad += 1
            logger.info(
                "Fila de despliegue sin métricas de calidad, descartada del "
                "análisis: %s / %s / %s",
                fila["experiment_id"],
                fila["runtime"],
                fila["precision"],
            )
            continue
        if len(candidatas) > 1:
            raise ParetoError(f"La configuración {clave} tiene {len(candidatas)} filas de calidad.")
        calidad = candidatas[0]

        valores: dict[str, float] = {}
        completo = True
        for objetivo in objectives:
            origen = calidad if objetivo.source == "quality" else fila
            bruto = (origen.get(objetivo.column, "") or "").strip()
            if bruto == "":
                completo = False
                break
            valores[objetivo.column] = float(bruto)
        if not completo:
            sin_valor += 1
            logger.info(
                "Configuración sin valor en algún objetivo, descartada: %s / %s",
                fila["experiment_id"],
                fila["runtime"],
            )
            continue

        puntos.append(
            Point(
                experiment_id=fila["experiment_id"],
                runtime=fila["runtime"],
                precision=fila["precision"],
                device_id=fila["device_id"],
                environment_id=fila.get("environment_id", ""),
                timestamp=fila.get("timestamp", ""),
                values=valores,
            )
        )

    if sin_calidad or sin_valor:
        logger.info(
            "Puntos construidos: %d. Descartados %d sin calidad y %d sin algún " "objetivo medido.",
            len(puntos),
            sin_calidad,
            sin_valor,
        )
    return tuple(sorted(puntos, key=lambda p: p.label))


def same_session(a: Point, b: Point) -> bool:
    """True si dos puntos comparten environment_id, es decir, sesión de medida.

    La huella incluye el commit y git_dirty, no el contenido del árbol, así que dos medidas sobre el
    mismo commit con cambios sin confirmar comparten huella aunque las separen días. git_dirty
    ignora results/. El criterio está en floors.same_environment para que degradación y selección lo
    usen igual.
    """
    return same_environment(a.environment_id, b.environment_id)


def compare(
    a: Point, b: Point, objective: Objective, floors: FloorSet, misma_sesion: bool
) -> Comparison:
    """Compara dos puntos en un eje aplicando el suelo de ese eje."""
    if objective.column not in a.values or objective.column not in b.values:
        raise ParetoError(
            f"No se puede comparar en '{objective.name}': falta el valor en uno "
            f"de los dos puntos ({a.label}, {b.label})."
        )
    va = a.values[objective.column]
    vb = b.values[objective.column]
    ventaja = (vb - va) if objective.direction == "min" else (va - vb)

    suelo = floors.get(objective.column)
    valor_suelo = suelo.applicable(misma_sesion) if suelo is not None else None

    if valor_suelo is None:
        umbral = None
    elif suelo is not None and suelo.kind == "relativo":
        umbral = valor_suelo * min(abs(va), abs(vb))
    else:
        umbral = valor_suelo

    if ventaja == 0:
        # Un empate exacto es indistinguible con cualquier suelo, medido o no.
        return Comparison(
            objective=objective.name,
            column=objective.column,
            value_a=va,
            value_b=vb,
            advantage=0.0,
            threshold=umbral,
            verdict="indistinguible",
            certain=True,
        )

    if umbral is not None and abs(ventaja) <= umbral:
        veredicto, certeza = "indistinguible", True
    else:
        veredicto = "mejor" if ventaja > 0 else "peor"
        certeza = umbral is not None

    return Comparison(
        objective=objective.name,
        column=objective.column,
        value_a=va,
        value_b=vb,
        advantage=ventaja,
        threshold=umbral,
        verdict=veredicto,
        certain=certeza,
    )


def dominates(
    a: Point,
    b: Point,
    objectives: Sequence[Objective],
    floors: FloorSet,
    mode: str = MODE_ROBUST,
) -> Domination | None:
    """Indica si ``a`` domina a ``b``.

    En modo ``estricta`` basta con no ser peor en ningún eje y mejor en alguno. En ``robusta`` la
    ventaja tiene que superar un suelo medido, y cualquier desventaja bloquea.
    """
    if mode not in (MODE_STRICT, MODE_ROBUST):
        raise ParetoError(f"Modo de dominación desconocido: '{mode}'.")

    misma = same_session(a, b)
    comparaciones = tuple(compare(a, b, o, floors, misma) for o in objectives)

    # La desventaja se exige sin suelo en los dos modos; perdonarla haría la dominación más
    # permisiva.
    if any(c.advantage < 0 for c in comparaciones):
        return None

    if mode == MODE_STRICT:
        if not any(c.advantage > 0 for c in comparaciones):
            return None
    else:
        if not any(c.verdict == "mejor" and c.certain for c in comparaciones):
            return None

    return Domination(
        dominator=a.label,
        dominated=b.label,
        mode=mode,
        same_session=misma,
        comparisons=comparaciones,
    )


def frontier(
    points: Sequence[Point],
    objectives: Sequence[Objective],
    floors: FloorSet,
    mode: str = MODE_ROBUST,
) -> FrontierResult:
    """Calcula la frontera y las dominaciones comparando cada punto con todos los demás."""
    dominaciones: list[Domination] = []
    dominados: set[str] = set()
    indistinguibles: list[tuple[str, str]] = []

    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                continue
            dom = dominates(a, b, objectives, floors, mode)
            if dom is not None:
                dominaciones.append(dom)
                dominados.add(b.label)
            if i < j:
                misma = same_session(a, b)
                comps = [compare(a, b, o, floors, misma) for o in objectives]
                if all(c.verdict == "indistinguible" for c in comps):
                    indistinguibles.append((a.label, b.label))

    primera = objectives[0].column
    no_dominados = tuple(
        sorted(
            (p for p in points if p.label not in dominados),
            key=lambda p: (p.values[primera], p.label),
        )
    )
    return FrontierResult(
        mode=mode,
        nondominated=no_dominados,
        dominations=tuple(dominaciones),
        indistinguishable=tuple(indistinguibles),
    )


def _normalized(
    point: Point, objectives: Sequence[Objective], bounds: Mapping[str, tuple[float, float]]
) -> tuple[float, ...]:
    """Coordenadas de minimización normalizadas a [0, 1]: 0 es lo mejor observado y 1 lo peor."""
    coords: list[float] = []
    for objetivo in objectives:
        minimo, maximo = bounds[objetivo.column]
        rango = maximo - minimo
        valor = point.values[objetivo.column]
        if rango == 0:
            coords.append(0.0)
            continue
        fraccion = (valor - minimo) / rango
        coords.append(fraccion if objetivo.direction == "min" else 1.0 - fraccion)
    return tuple(coords)


def observed_bounds(
    points: Sequence[Point], objectives: Sequence[Objective]
) -> dict[str, tuple[float, float]]:
    """Mínimo y máximo por eje sobre todos los puntos, para que los hipervolúmenes de grupos
    distintos sean comparables.
    """
    if not points:
        raise ParetoError("No hay puntos con los que fijar los límites de los ejes.")
    limites: dict[str, tuple[float, float]] = {}
    for objetivo in objectives:
        valores = [p.values[objetivo.column] for p in points]
        limites[objetivo.column] = (min(valores), max(valores))
    return limites


def _hypervolume_recursive(puntos: Sequence[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    """Hipervolumen exacto por rebanado de objetivos (HSO)."""
    if not puntos:
        return 0.0
    if len(ref) == 1:
        return max(0.0, ref[0] - min(p[0] for p in puntos))

    orden = sorted(puntos, key=lambda p: p[-1])
    total = 0.0
    for i, p in enumerate(orden):
        superior = orden[i + 1][-1] if i + 1 < len(orden) else ref[-1]
        grosor = superior - p[-1]
        if grosor <= 0:
            continue
        proyeccion = [q[:-1] for q in orden[: i + 1]]
        total += grosor * _hypervolume_recursive(proyeccion, ref[:-1])
    return total


def hypervolume(
    points: Sequence[Point],
    objectives: Sequence[Objective],
    bounds: Mapping[str, tuple[float, float]],
) -> float:
    """Hipervolumen normalizado entre 0 y 1, con referencia en el peor rincón de la caja observada.

    Es descriptivo: no tiene suelo y no sirve por sí solo para afirmar que un grupo es mejor.
    """
    if not points:
        return 0.0
    coords = [_normalized(p, objectives, bounds) for p in points]
    referencia = tuple(1.0 for _ in objectives)
    return _hypervolume_recursive(coords, referencia)


def distance_to_ideal(
    point: Point, objectives: Sequence[Objective], bounds: Mapping[str, tuple[float, float]]
) -> float:
    """Distancia euclídea normalizada al punto ideal (el mejor valor observado en cada eje)."""
    coords = _normalized(point, objectives, bounds)
    return math.sqrt(sum(c * c for c in coords))


def dominations_lost_to_floor(
    strict: FrontierResult, robust: FrontierResult
) -> tuple[Domination, ...]:
    """Dominaciones de la frontera estricta que no se mantienen en la robusta."""
    supervivientes = {(d.dominator, d.dominated) for d in robust.dominations}
    return tuple(d for d in strict.dominations if (d.dominator, d.dominated) not in supervivientes)


def analyze(
    points: Sequence[Point],
    objectives: Sequence[Objective],
    floors: FloorSet,
    group_keys: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Calcula todo lo necesario para una vista en cada agrupación pedida.

    Se agrupa por dispositivo y motor, y también solo por dispositivo, porque con el hardware fijo
    elegir motor es una decisión de despliegue. Los límites de normalización son comunes a todos los
    grupos.
    """
    limites = observed_bounds(points, objectives)
    resultado: dict[str, Any] = {
        "objectives": [
            {
                "name": o.name,
                "column": o.column,
                "source": o.source,
                "direction": o.direction,
            }
            for o in objectives
        ],
        "floors": floors.to_dict(),
        "bounds": {col: list(par) for col, par in sorted(limites.items())},
        "num_points": len(points),
        "groupings": {},
    }

    for claves in group_keys:
        nombre = "+".join(claves)
        grupos = group_points(points, claves)
        detalle: dict[str, Any] = {}
        for clave, ps in grupos.items():
            estricta = frontier(ps, objectives, floors, MODE_STRICT)
            robusta = frontier(ps, objectives, floors, MODE_ROBUST)
            perdidas = dominations_lost_to_floor(estricta, robusta)
            detalle["/".join(clave)] = {
                "num_points": len(ps),
                "hypervolume_normalized": hypervolume(ps, objectives, limites),
                "frontier_strict": [p.label for p in estricta.nondominated],
                "frontier_robust": [p.label for p in robusta.nondominated],
                "num_dominations_strict": len(estricta.dominations),
                "num_dominations_robust": len(robusta.dominations),
                "dominations_robust": [d.to_dict() for d in robusta.dominations],
                "dominations_lost_to_floor": [d.to_dict() for d in perdidas],
                "indistinguishable_pairs": [list(par) for par in robusta.indistinguishable],
                "distance_to_ideal": {
                    p.label: distance_to_ideal(p, objectives, limites)
                    for p in sorted(ps, key=lambda q: q.label)
                },
            }
        resultado["groupings"][nombre] = detalle

    return resultado


def group_points(
    points: Iterable[Point], group_keys: Sequence[str]
) -> dict[tuple[str, ...], list[Point]]:
    """Agrupa los puntos por los campos indicados."""
    grupos: dict[tuple[str, ...], list[Point]] = {}
    for punto in points:
        try:
            clave = tuple(getattr(punto, k) for k in group_keys)
        except AttributeError as exc:
            raise ParetoError(f"No se puede agrupar por {list(group_keys)}: {exc}") from exc
        grupos.setdefault(clave, []).append(punto)
    return {k: grupos[k] for k in sorted(grupos)}
