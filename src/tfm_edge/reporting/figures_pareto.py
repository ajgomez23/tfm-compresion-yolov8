"""Figuras de la frontera de Pareto (apartado 6.4).

Dibujan mAP frente a latencia con la frontera resaltada, una figura por dispositivo, y otra versión
con la memoria codificada en el tamaño del marcador (más legible impresa que una vista 3D). Las
figuras incluyen una barra con el suelo de medida del eje, resaltan los puntos que solo entran en la
frontera robusta y marcan los medidos en otra sesión. Si un eje no tiene suelo medido, lo indican.
La envolvente en escalera solo se dibuja si la frontera se calculó sobre esos dos ejes. No se
calcula nada aquí: puntos, suelos y fronteras vienen de analysis.pareto, y no hay barras de error
por punto porque el suelo no es una dispersión.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.analysis.floors import AxisFloor, FloorSet
from tfm_edge.analysis.pareto import (
    FrontierResult,
    Objective,
    Point,
    compare,
    dominates,
    same_session,
)
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import parse_experiment_id
from tfm_edge.reporting import figures_results, style
from tfm_edge.reporting.style import comma_axis_formatter, format_number

logger = get_logger(__name__)

#: Capítulo y apartado de estas figuras.
CHAPTER = 6
SECTION = 4

#: Área del marcador para el mínimo y el máximo de memoria observados. El mapeo es lineal sobre el
#: rango observado: con memorias entre 430 y 550 MiB, un área proporcional no se distinguiría.
MARKER_AREA_RANGE = (26.0, 220.0)

FLAT_MARKER_AREA = 34.0

SIZE_LEGEND_STEPS = 3

ROBUST_HALO_COLOR = "#E69F00"

FRONTIER_LINE_COLOR = "#555555"

FLOOR_COLOR = "#7F0000"

#: Motores de las figuras por motor, en el orden de los paneles. PyTorch no entra porque solo tiene
#: baselines.
ENGINE_RUNTIMES: tuple[str, ...] = ("onnxruntime", "openvino")
ENGINE_TITLES: dict[str, str] = {"onnxruntime": "ONNX Runtime", "openvino": "OpenVINO"}

#: Ancho y cuerpo de texto de las figuras por motor, que se dibujan a su tamaño final.
ENGINE_FIGURE_WIDTH = 16.0 / 2.54
ENGINE_FONTSIZE = 6.5
ENGINE_TITLE_FONTSIZE = 7.5

ENGINE_MARKER_AREA = 20.0

#: Opacidad de los puntos dominados: visibles, pero sin competir con la frontera.
DOMINATED_ALPHA = 0.28

#: Separación entre punto y etiqueta y entre etiquetas, en puntos.
LABEL_GAP_PT = 3.0
LABEL_SPACING_PT = 0.8

#: Holgura alrededor de cada marcador que no deben invadir las etiquetas, en puntos.
LABEL_CLEARANCE_PT = 0.6

RING_AREA_FACTOR = 5.0

#: Separación entre etiquetas y holgura en la figura de nano frente a small, mayores que en el resto
#: para que las etiquetas de OpenVINO no se toquen.
NANO_LABEL_SPACING_PT = 2.5
NANO_LABEL_CLEARANCE_PT = 1.6

#: Margen horizontal de la figura de nano frente a small, en fracción del rango.
NANO_X_MARGIN = 0.10

#: Desplazamiento horizontal máximo de una etiqueta, en puntos, en las figuras que la llevan a la
#: zona libre más próxima.
LABEL_MAX_SLIDE_PT = 90.0
LABEL_SLIDE_STEP_PT = 6.0

#: Órdenes de colocación aleatorios que se prueban además de los cuatro fijos, y su semilla.
LABEL_LAYOUT_RANDOM_ORDERS = 48
LABEL_LAYOUT_SEED = 190

#: Fondo blanco semitransparente de las etiquetas, para que las líneas no crucen el texto.
LABEL_BACKGROUND: dict[str, Any] = {
    "boxstyle": "square,pad=0.12",
    "facecolor": "white",
    "alpha": 0.8,
    "edgecolor": "none",
}

#: Relleno de la región que domina cada referencia de nano.
NANO_REGION_COLORS: dict[str, str] = {"baseline": "#9E9E9E", "ptq": "#0072B2"}
NANO_REGION_ALPHA = 0.14

#: Orden de las categorías en las leyendas, con la referencia delante.
ENGINE_CATEGORY_ORDER: tuple[str, ...] = ("baseline", *figures_results.COMPRESSION_CATEGORIES)

#: Rótulo de cada eje con su unidad; un eje desconocido se rotula sin unidad.
_AXIS_LABELS: dict[str, str] = {
    "latency_p50_ms": "latencia p50 (ms)",
    "latency_p95_ms": "latencia p95 (ms)",
    "peak_host_memory_mb": "memoria pico (MiB)",
}

#: Unidad de los ejes cuyo suelo relativo se puede expresar también en valor absoluto. Los de
#: calidad no están.
_AXIS_UNITS: dict[str, str] = {
    "latency_p50_ms": "ms",
    "latency_p95_ms": "ms",
    "peak_host_memory_mb": "MiB",
}


def technique_of(point: Point) -> str:
    """Familia de técnica de un punto, para su color y marcador. Un experiment_id mal formado lanza
    error.
    """
    return parse_experiment_id(point.experiment_id)["technique"]


def reference_environment(points: Sequence[Point]) -> str:
    """Entorno de medida con más puntos, que se toma como sesión de referencia.

    Los empates se resuelven por orden alfabético y se avisan en el log.
    """
    if not points:
        raise ValueError("No hay puntos de los que deducir la sesión de referencia.")
    cuenta = Counter(p.environment_id for p in points)
    mayor = max(cuenta.values())
    candidatos = sorted(env for env, n in cuenta.items() if n == mayor)
    if len(candidatos) > 1:
        logger.warning(
            "Hay %d entornos empatados a %d puntos (%s). Se toma '%s' como referencia por orden "
            "alfabético; revisa la figura antes de publicarla.",
            len(candidatos),
            mayor,
            ", ".join(candidatos),
            candidatos[0],
        )
    return candidatos[0]


def cross_session_labels(points: Sequence[Point], reference: str) -> set[str]:
    """Etiquetas de los puntos medidos fuera de la sesión de referencia."""
    return {p.label for p in points if p.environment_id != reference}


def _axis_label(objective: Objective) -> str:
    """Rótulo del eje, con unidad cuando la hay."""
    if objective.column in _AXIS_LABELS:
        return _AXIS_LABELS[objective.column]
    return objective.name


def _floor_caption(
    floor: AxisFloor | None,
    objective: Objective,
    anchor: float,
    *,
    cross_session_applies: bool,
) -> str:
    """Texto que declara el suelo de un eje.

    ``anchor`` permite dar un suelo relativo también en la unidad del eje. Si todos los puntos son
    de la misma sesión (``cross_session_applies`` False), el suelo entre sesiones se menciona
    aparte, porque no se aplica a ninguna comparación de la figura.
    """
    if floor is None or (floor.same_session is None and floor.cross_session is None):
        return (
            f"{objective.name}: sin suelo medido. "
            f"Sobre este eje no se certifica ninguna ventaja."
        )
    if floor.kind == "absoluto":
        valor = floor.same_session if floor.same_session is not None else floor.cross_session
        if valor is None:
            return f"{objective.name}: sin suelo medido."
        return f"suelo de {objective.name}: {format_number(valor, 4)}"

    # Un suelo relativo solo se da en la unidad del eje si hay un punto de anclaje, indicando que es
    # en ese punto.
    unidad = _AXIS_UNITS.get(objective.column, "")

    def en_unidad(fraccion: float) -> str:
        if unidad and anchor > 0:
            return f" ({format_number(anchor * fraccion, 2)} {unidad} aquí)"
        return ""

    # Si los dos suelos coinciden, se dice una vez.
    if floor.same_session is not None and floor.same_session == floor.cross_session:
        pct = format_number(100 * floor.same_session, 1)
        return (
            f"suelo de {objective.name}: {pct} %{en_unidad(floor.same_session)}, "
            f"igual dentro y entre sesiones"
        )

    # Todos los puntos de la misma sesión: el suelo entre sesiones se declara aparte.
    if not cross_session_applies and floor.same_session is not None:
        pct = format_number(100 * floor.same_session, 1)
        texto = (
            f"suelo de {objective.name}: {pct} %{en_unidad(floor.same_session)}, "
            f"misma sesión — el que gobierna todas las comparaciones de esta figura"
        )
        if floor.cross_session is not None and floor.cross_session != floor.same_session:
            entre = format_number(100 * floor.cross_session, 1)
            texto += (
                f". Entre sesiones está medido en {entre} %{en_unidad(floor.cross_session)}, "
                f"pero no se dibuja: aquí no hay ningún punto de otra sesión al que aplicarlo"
            )
        return texto

    partes = []
    for etiqueta, fraccion in (
        ("misma sesión", floor.same_session),
        ("entre sesiones", floor.cross_session),
    ):
        if fraccion is None:
            continue
        partes.append(f"{etiqueta} {format_number(100 * fraccion, 1)} %{en_unidad(fraccion)}")
    return f"suelo de {objective.name}: " + " · ".join(partes)


def _draw_relative_floor_bars(
    axes: Any,
    floor: AxisFloor,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    *,
    cross_session_applies: bool,
) -> float:
    """Barra de resolución del eje horizontal, como la escala de un mapa.

    El suelo de latencia es relativo, así que se ancla en un punto y el rótulo da su valor allí. Con
    una sola sesión no se dibuja la barra entre sesiones, que dominaría la figura sin aplicarse a
    nada. Devuelve el valor del eje donde se ancló.
    """
    x0 = xlim[0] + 0.50 * (xlim[1] - xlim[0])
    alto = ylim[1] - ylim[0]
    niveles: list[tuple[float, str]] = []
    if floor.same_session is not None:
        niveles.append((floor.same_session, "-"))
    dibujar_entre_sesiones = cross_session_applies or floor.same_session is None
    if (
        dibujar_entre_sesiones
        and floor.cross_session is not None
        and floor.cross_session != floor.same_session
    ):
        niveles.append((floor.cross_session, "--"))

    for indice, (fraccion, trazo) in enumerate(niveles):
        y = ylim[0] + (0.05 + 0.05 * indice) * alto
        axes.plot(
            [x0, x0 * (1.0 + fraccion)],
            [y, y],
            color=FLOOR_COLOR,
            linestyle=trazo,
            linewidth=1.2,
            marker="|",
            markersize=5,
            solid_capstyle="butt",
            zorder=1.5,
        )
    return x0


def _draw_absolute_floor_bar(
    axes: Any, floor: AxisFloor, xlim: tuple[float, float], ylim: tuple[float, float]
) -> None:
    """Dibuja la barra de resolución de un eje con suelo absoluto."""
    valor = floor.same_session if floor.same_session is not None else floor.cross_session
    if valor is None:
        return
    x = xlim[0] + 0.035 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    axes.plot(
        [x, x],
        [y0, y0 + valor],
        color=FLOOR_COLOR,
        linestyle="-",
        linewidth=1.2,
        marker="_",
        markersize=5,
        solid_capstyle="butt",
        zorder=1.5,
    )


def _scatter_points(
    axes: Any,
    points: Sequence[Point],
    objectives: Sequence[Objective],
    cross_session: set[str],
    sizes: Mapping[str, float] | None = None,
) -> set[str]:
    """Dibuja los puntos con el estilo de su familia y marca con un anillo los de otra sesión."""
    eje_x, eje_y = objectives[1].column, objectives[0].column
    familias: set[str] = set()
    for punto in points:
        familia = technique_of(punto)
        estilo = style.style_for(familia)
        area = FLAT_MARKER_AREA if sizes is None else sizes[punto.label]
        axes.scatter(
            punto.values[eje_x],
            punto.values[eje_y],
            s=area,
            marker=estilo.marker,
            facecolor=estilo.color,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        familias.add(familia)
        if punto.label in cross_session:
            axes.scatter(
                punto.values[eje_x],
                punto.values[eje_y],
                s=area * 3.2,
                marker="o",
                facecolor="none",
                edgecolor="black",
                linewidth=0.8,
                linestyle=(0, (2, 1)),
                zorder=2.5,
            )
    return familias


def _highlight_robust_only(
    axes: Any,
    objectives: Sequence[Objective],
    strict: FrontierResult,
    robust: FrontierResult,
    sizes: Mapping[str, float] | None = None,
) -> list[str]:
    """Resalta los puntos que solo están en la frontera robusta."""
    en_estricta = {p.label for p in strict.nondominated}
    solo_robusta = [p for p in robust.nondominated if p.label not in en_estricta]
    eje_x, eje_y = objectives[1].column, objectives[0].column
    for punto in solo_robusta:
        area = FLAT_MARKER_AREA if sizes is None else sizes[punto.label]
        axes.scatter(
            punto.values[eje_x],
            punto.values[eje_y],
            s=area * 6.0,
            marker="o",
            facecolor="none",
            edgecolor=ROBUST_HALO_COLOR,
            linewidth=1.5,
            zorder=2,
        )
    return [p.label for p in solo_robusta]


def _mark_dominated(
    axes: Any,
    points: Sequence[Point],
    objectives: Sequence[Objective],
    robust: FrontierResult,
) -> list[str]:
    """Tacha los puntos dominados en las vistas sin escalera; son minoría."""
    no_dominados = {p.label for p in robust.nondominated}
    eje_x, eje_y = objectives[1].column, objectives[0].column
    marcados: list[str] = []
    for punto in points:
        if punto.label in no_dominados:
            continue
        axes.scatter(
            punto.values[eje_x],
            punto.values[eje_y],
            s=FLAT_MARKER_AREA * 1.6,
            marker="x",
            color="black",
            linewidth=1.1,
            zorder=4,
        )
        marcados.append(punto.label)
    return marcados


def _draw_strict_envelope(
    axes: Any, strict: FrontierResult, objectives: Sequence[Objective]
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Dibuja la escalera de la frontera estricta.

    En escalera porque entre dos puntos solo se sabe que el mejor mAP alcanzable es el del punto más
    rápido. Solo la estricta, y solo si se calculó sobre estos dos ejes: la robusta no es monótona y
    una frontera de tres objetivos proyectada no es una envolvente. Devuelve los tramos para que las
    etiquetas los eviten.
    """
    eje_x, eje_y = objectives[1].column, objectives[0].column
    ordenados = sorted(strict.nondominated, key=lambda p: p.values[eje_x])
    if len(ordenados) < 2:
        return []
    tramos: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for a, b in zip(ordenados, ordenados[1:]):
        # «steps-post»: primero horizontal a la altura del punto, después vertical.
        esquina = (b.values[eje_x], a.values[eje_y])
        tramos.append(((a.values[eje_x], a.values[eje_y]), esquina))
        tramos.append((esquina, (b.values[eje_x], b.values[eje_y])))
    axes.plot(
        [p.values[eje_x] for p in ordenados],
        [p.values[eje_y] for p in ordenados],
        drawstyle="steps-post",
        color=FRONTIER_LINE_COLOR,
        linewidth=1.1,
        alpha=0.85,
        zorder=1.8,
    )
    return tramos


def _draw_indistinguishable(
    axes: Any,
    points: Sequence[Point],
    objectives: Sequence[Objective],
    robust: FrontierResult,
    sizes: Mapping[str, float] | None = None,
) -> int:
    """Marca con un rombo hueco los puntos que el suelo declara indistinguibles.

    Los dos miembros de un par caen casi uno encima del otro, así que el conector entre ellos no se
    ve; el rombo sí. El número de pares por panel va al pie.
    """
    por_etiqueta = {p.label: p for p in points}
    eje_x, eje_y = objectives[1].column, objectives[0].column
    presentes = [
        (por_etiqueta[a], por_etiqueta[b])
        for a, b in robust.indistinguishable
        if a in por_etiqueta and b in por_etiqueta
    ]
    for pa, pb in presentes:
        axes.plot(
            [pa.values[eje_x], pb.values[eje_x]],
            [pa.values[eje_y], pb.values[eje_y]],
            color=FLOOR_COLOR,
            linewidth=0.9,
            linestyle=":",
            zorder=1.6,
        )
    # Cada miembro se marca una sola vez aunque esté en dos pares.
    miembros = {p.label: p for par in presentes for p in par}
    for etiqueta, punto in miembros.items():
        area = FLAT_MARKER_AREA if sizes is None else sizes[etiqueta]
        axes.scatter(
            punto.values[eje_x],
            punto.values[eje_y],
            s=area * 4.5,
            marker="D",
            facecolor="none",
            edgecolor=FLOOR_COLOR,
            linewidth=1.0,
            linestyle=(0, (1, 1)),
            zorder=2.4,
        )
    return len(presentes)


def _memory_sizes(
    points: Sequence[Point], column: str
) -> tuple[dict[str, float], tuple[float, float]]:
    """Área del marcador según la memoria, con mapeo lineal sobre el rango observado. Devuelve
    también el rango.
    """
    valores = [p.values[column] for p in points]
    minimo, maximo = min(valores), max(valores)
    a0, a1 = MARKER_AREA_RANGE
    areas: dict[str, float] = {}
    for punto in points:
        if maximo == minimo:
            areas[punto.label] = (a0 + a1) / 2.0
        else:
            fraccion = (punto.values[column] - minimo) / (maximo - minimo)
            areas[punto.label] = a0 + fraccion * (a1 - a0)
    return areas, (minimo, maximo)


def declare_floors(
    axes: Any,
    floors: FloorSet,
    objectives: Sequence[Objective],
    *,
    cross_session_applies: bool,
) -> float:
    """Dibuja las barras del suelo en un panel ya poblado.

    Se hace en una segunda pasada porque la barra fija los límites del eje, y en un eje compartido
    eso congelaría los límites antes de dibujar los demás paneles. ``cross_session_applies`` se
    decide una vez para toda la figura.
    """
    xlim, ylim = axes.get_xlim(), axes.get_ylim()
    suelo_x = floors.get(objectives[1].column)
    suelo_y = floors.get(objectives[0].column)
    ancla = xlim[0] + 0.50 * (xlim[1] - xlim[0])
    if suelo_x is not None and suelo_x.kind == "relativo":
        ancla = _draw_relative_floor_bars(
            axes, suelo_x, xlim, ylim, cross_session_applies=cross_session_applies
        )
    if suelo_y is not None and suelo_y.kind == "absoluto":
        _draw_absolute_floor_bar(axes, suelo_y, xlim, ylim)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    return ancla


def plot_frontier_panel(
    axes: Any,
    points: Sequence[Point],
    objectives: Sequence[Objective],
    floors: FloorSet,
    strict: FrontierResult,
    robust: FrontierResult,
    *,
    title: str,
    reference: str | None = None,
    sizes: Mapping[str, float] | None = None,
    draw_envelope: bool = True,
    declare_floors_now: bool = True,
) -> dict[str, Any]:
    """Dibuja un panel de frontera sobre unos ejes ya creados, sin leyendas ni pie.

    ``reference`` se decide una vez por figura para que todos los paneles marquen igual los puntos
    de otra sesión. ``draw_envelope`` debe ser False si la frontera se calculó sobre más de dos
    ejes, y ``declare_floors_now`` False si el panel comparte ejes con otros aún sin dibujar.
    """
    if not points:
        raise ValueError(f"El panel '{title}' no tiene ningún punto que dibujar.")

    referencia = reference if reference is not None else reference_environment(points)
    otra_sesion = cross_session_labels(points, referencia)

    familias = _scatter_points(axes, points, objectives, otra_sesion, sizes)
    dominados: list[str] = []
    if draw_envelope:
        _draw_strict_envelope(axes, strict, objectives)
    else:
        dominados = _mark_dominated(axes, points, objectives, robust)
    resaltados = _highlight_robust_only(axes, objectives, strict, robust, sizes)
    indistinguibles = _draw_indistinguishable(axes, points, objectives, robust, sizes)

    axes.set_xlabel(_axis_label(objectives[1]))
    axes.set_ylabel(_axis_label(objectives[0]))
    axes.set_title(title)
    axes.xaxis.set_major_formatter(comma_axis_formatter())
    axes.yaxis.set_major_formatter(comma_axis_formatter())

    # Con declare_floors_now el panel es la figura entera, así que decidir aquí es correcto.
    ancla = (
        declare_floors(axes, floors, objectives, cross_session_applies=bool(otra_sesion))
        if declare_floors_now
        else 0.0
    )

    return {
        "reference_environment": referencia,
        "cross_session": sorted(otra_sesion),
        "robust_only": resaltados,
        "dominated_marked": sorted(dominados),
        "indistinguishable_drawn": indistinguibles,
        "families": sorted(familias),
        "floor_anchor": ancla,
        # Queda en el manifiesto si se dibujó la barra entre sesiones; None hasta declarar los
        # suelos.
        "cross_session_floor_drawn": bool(otra_sesion) if declare_floors_now else None,
        "num_points": len(points),
    }


def _family_handles(families: Iterable[str]) -> tuple[list[Any], list[str]]:
    """Elementos de leyenda por familia, con el marcador de la familia y no el de un punto concreto.
    """
    from matplotlib.lines import Line2D

    presentes = set(families)
    apoderados: list[Any] = []
    rotulos: list[str] = []
    for familia, estilo in style.TECHNIQUE_STYLES.items():
        if familia not in presentes:
            continue
        apoderados.append(
            Line2D(
                [],
                [],
                marker=estilo.marker,
                markerfacecolor=estilo.color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
                markersize=6,
            )
        )
        rotulos.append(estilo.label)
    return apoderados, rotulos


def _annotation_handles(
    *,
    cross_session: bool,
    halo: bool,
    envelope: bool,
    dominated: bool,
    indistinguishable: bool,
) -> tuple[list[Any], list[str]]:
    """Elementos de leyenda de las marcas propias de la figura."""
    from matplotlib.lines import Line2D

    apoderados: list[Any] = []
    rotulos: list[str] = []

    def anadir(handle: Any, rotulo: str) -> None:
        apoderados.append(handle)
        rotulos.append(rotulo)

    if cross_session:
        anadir(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="none",
                markersize=8,
            ),
            "medido en otra sesión",
        )
    if halo:
        anadir(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="none",
                markeredgecolor=ROBUST_HALO_COLOR,
                markeredgewidth=1.5,
                linestyle="none",
                markersize=9,
            ),
            "en la frontera solo al exigir el suelo",
        )
    if envelope:
        anadir(
            Line2D([], [], color=FRONTIER_LINE_COLOR, linewidth=1.1),
            "frontera estricta: mejor mAP por presupuesto",
        )
    if dominated:
        anadir(
            Line2D([], [], marker="x", color="black", linestyle="none", markersize=6),
            "dominado",
        )
    if indistinguishable:
        anadir(
            Line2D(
                [],
                [],
                color=FLOOR_COLOR,
                linewidth=0.9,
                linestyle=":",
                marker="D",
                markerfacecolor="none",
                markeredgecolor=FLOOR_COLOR,
                markeredgewidth=1.0,
                markersize=7,
            ),
            "par indistinguible bajo el suelo",
        )
    anadir(
        Line2D([], [], color=FLOOR_COLOR, linewidth=1.2, marker="|", markersize=5),
        "suelo de la medida",
    )
    return apoderados, rotulos


def _size_handles(rango: tuple[float, float]) -> tuple[list[Any], list[str]]:
    """Leyenda de tamaños de marcador con los valores reales de memoria."""
    from matplotlib.lines import Line2D

    minimo, maximo = rango
    a0, a1 = MARKER_AREA_RANGE
    apoderados: list[Any] = []
    rotulos: list[str] = []
    for indice in range(SIZE_LEGEND_STEPS):
        fraccion = indice / (SIZE_LEGEND_STEPS - 1)
        valor = minimo + fraccion * (maximo - minimo)
        area = a0 + fraccion * (a1 - a0)
        apoderados.append(
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="#999999",
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
                markersize=area**0.5,
            )
        )
        rotulos.append(f"{format_number(valor, 1)} MiB")
    return apoderados, rotulos


def _coverage_note(summary: Mapping[str, Any], runtimes: Sequence[str], total: int) -> str | None:
    """Nota sobre los motores que no tienen todas las técnicas medidas (por ejemplo, PyTorch solo
    tiene baselines). None si la cobertura es homogénea.
    """
    todas = {familia for datos in summary.values() for familia in datos["families"]}
    parciales = []
    for motor in runtimes:
        datos = summary[motor]
        if not todas - set(datos["families"]):
            continue
        parciales.append(
            f"{motor}, solo {', '.join(datos['families'])} "
            f"({datos['num_points']} de {total} puntos)"
        )
    if not parciales:
        return None
    return (
        "cobertura desigual entre motores: "
        + "; ".join(parciales)
        + ". Ese panel no está incompleto por error, es lo que llegó a medirse en ese motor"
    )


def _indistinguishable_note(counts: Sequence[tuple[str, int]]) -> str | None:
    """Nota al pie con los pares indistinguibles de cada panel, o None."""
    presentes = [(nombre, cuantos) for nombre, cuantos in counts if cuantos]
    if not presentes:
        return None
    if len(presentes) == 1 and not presentes[0][0]:
        detalle = str(presentes[0][1])
    else:
        detalle = ", ".join(f"{cuantos} en {nombre}" for nombre, cuantos in presentes)
    return (
        "pares que el suelo no llega a separar, con sus dos miembros marcados con "
        f"rombo y unidos por trazo punteado: {detalle}. Los dos puntos de un par caen "
        "casi en el mismo sitio, así que lo que se ve es el rombo, no el trazo"
    )


def _floor_footer(
    floors: FloorSet,
    objectives: Sequence[Objective],
    anchor: float,
    *,
    cross_session_applies: bool,
) -> str:
    """Pie que declara los suelos de los dos ejes. ``cross_session_applies`` debe coincidir con el
    usado al dibujar.
    """
    return "\n".join(
        [
            _floor_caption(
                floors.get(objectives[1].column),
                objectives[1],
                anchor,
                cross_session_applies=cross_session_applies,
            ),
            _floor_caption(
                floors.get(objectives[0].column),
                objectives[0],
                0.0,
                cross_session_applies=cross_session_applies,
            ),
        ]
    )


def _fill_legend_band(
    axes: Any,
    familias: Iterable[str],
    anotaciones: tuple[list[Any], list[str]],
    pie: str,
    tamanos: tuple[list[Any], list[str]] | None = None,
    notas: Sequence[str] = (),
) -> None:
    """Rellena la franja inferior con las leyendas y el pie, fuera de los paneles."""
    apoderados_familia, rotulos_familia = _family_handles(familias)
    leyenda_familia = axes.legend(
        apoderados_familia,
        rotulos_familia,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=7,
        title="Familia de técnica",
        title_fontsize=7,
        frameon=False,
        handletextpad=0.4,
    )
    axes.add_artist(leyenda_familia)

    leyenda_marcas = axes.legend(
        anotaciones[0],
        anotaciones[1],
        loc="upper left",
        bbox_to_anchor=(0.27, 1.0),
        fontsize=7,
        title="Marcas de esta figura",
        title_fontsize=7,
        frameon=False,
        handletextpad=0.6,
    )
    axes.add_artist(leyenda_marcas)

    if tamanos is not None:
        leyenda_tamanos = axes.legend(
            tamanos[0],
            tamanos[1],
            loc="upper left",
            bbox_to_anchor=(0.74, 1.0),
            fontsize=7,
            # La advertencia sobre el área va en el título de la leyenda de tamaños.
            title="Memoria pico\n(área lineal en el rango observado,\nno proporcional al valor)",
            title_fontsize=6.5,
            frameon=False,
            labelspacing=1.2,
            handletextpad=0.9,
        )
        axes.add_artist(leyenda_tamanos)

    # Suelos y notas en un solo bloque de texto gris; la entrada «suelo de la medida» de la leyenda
    # lleva el color de las barras.
    axes.text(
        0.0,
        -0.10,
        "\n".join([pie, *notas]),
        transform=axes.transAxes,
        fontsize=7,
        color="#333333",
        ha="left",
        va="top",
    )


def figure_frontier_by_device(
    points_by_runtime: Mapping[str, Sequence[Point]],
    frontiers_by_runtime: Mapping[str, tuple[FrontierResult, FrontierResult]],
    device_points: Sequence[Point],
    device_frontiers: tuple[FrontierResult, FrontierResult],
    objectives: Sequence[Objective],
    floors: FloorSet,
    device_id: str,
) -> tuple[Any, dict[str, Any]]:
    """mAP frente a latencia: un panel por motor y uno con el dispositivo completo.

    Cada panel de motor tiene su frontera, y el común sirve para comparar motores. Requiere
    exactamente dos objetivos.
    """
    import matplotlib.pyplot as plt

    if len(objectives) != 2:
        raise ValueError(
            "Esta figura necesita fronteras calculadas sobre los dos ejes que dibuja; se han "
            f"pasado {len(objectives)} objetivos."
        )
    motores = sorted(points_by_runtime)
    if not motores:
        raise ValueError(f"El dispositivo '{device_id}' no tiene ningún motor con puntos.")

    columnas = len(motores) + 1
    figura = plt.figure(figsize=(3.5 * columnas, 5.6))
    # El panel común es más ancho porque lleva todos los puntos.
    rejilla = figura.add_gridspec(
        2,
        columnas,
        height_ratios=[3.0, 1.05],
        width_ratios=[*([1.0] * len(motores)), 1.4],
        hspace=0.42,
        wspace=0.28,
    )
    paneles = [figura.add_subplot(rejilla[0, i]) for i in range(columnas)]
    for panel in paneles[1:]:
        panel.sharex(paneles[0])
        panel.sharey(paneles[0])
    franja = figura.add_subplot(rejilla[1, :])
    franja.axis("off")

    # La sesión de referencia se calcula con todos los puntos del dispositivo y se aplica a todos
    # los paneles.
    referencia = reference_environment(device_points)

    resumen: dict[str, Any] = {}
    for indice, motor in enumerate(motores):
        estricta, robusta = frontiers_by_runtime[motor]
        resumen[motor] = plot_frontier_panel(
            paneles[indice],
            points_by_runtime[motor],
            objectives,
            floors,
            estricta,
            robusta,
            title=motor,
            reference=referencia,
            declare_floors_now=False,
        )
    resumen["__dispositivo__"] = plot_frontier_panel(
        paneles[-1],
        device_points,
        objectives,
        floors,
        device_frontiers[0],
        device_frontiers[1],
        title=f"{device_id}: los {len(motores)} motores juntos",
        reference=referencia,
        declare_floors_now=False,
    )

    # Segunda pasada: con todos los paneles dibujados, el eje compartido ya abarca toda la nube.
    hay_otra_sesion = any(datos["cross_session"] for datos in resumen.values())
    for panel, clave in zip(paneles, [*motores, "__dispositivo__"]):
        resumen[clave]["floor_anchor"] = declare_floors(
            panel, floors, objectives, cross_session_applies=hay_otra_sesion
        )
        resumen[clave]["cross_session_floor_drawn"] = hay_otra_sesion

    familias = sorted({f for datos in resumen.values() for f in datos["families"]})
    _fill_legend_band(
        franja,
        familias,
        _annotation_handles(
            cross_session=hay_otra_sesion,
            halo=any(datos["robust_only"] for datos in resumen.values()),
            envelope=True,
            dominated=False,
            indistinguishable=any(datos["indistinguishable_drawn"] for datos in resumen.values()),
        ),
        _floor_footer(
            floors,
            objectives,
            resumen["__dispositivo__"]["floor_anchor"],
            cross_session_applies=hay_otra_sesion,
        ),
        notas=[
            nota
            for nota in (
                _coverage_note(resumen, motores, len(device_points)),
                _indistinguishable_note(
                    [
                        *((motor, resumen[motor]["indistinguishable_drawn"]) for motor in motores),
                        (
                            "el panel conjunto",
                            resumen["__dispositivo__"]["indistinguishable_drawn"],
                        ),
                    ]
                ),
            )
            if nota is not None
        ],
    )

    figura.suptitle(
        f"Frontera de Pareto en {device_id}: {objectives[0].name} frente a "
        f"{objectives[1].name}",
        fontsize=10,
    )
    return figura, resumen


def figure_frontier_with_memory(
    points: Sequence[Point],
    strict: FrontierResult,
    robust: FrontierResult,
    objectives: Sequence[Objective],
    floors: FloorSet,
    device_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Los tres objetivos, con la memoria en el tamaño del marcador.

    No lleva escalera, porque la frontera es de tres objetivos; se tachan los puntos dominados.
    """
    import matplotlib.pyplot as plt

    if len(objectives) != 3:
        raise ValueError(
            f"Esta figura codifica el tercer objetivo en el tamaño del marcador y "
            f"necesita exactamente tres; se han pasado {len(objectives)}."
        )

    areas, rango = _memory_sizes(points, objectives[2].column)
    # Nueve pulgadas para que quepa la leyenda de tamaños.
    figura = plt.figure(figsize=(9.0, 6.6))
    rejilla = figura.add_gridspec(2, 1, height_ratios=[3.0, 1.2], hspace=0.34)
    panel = figura.add_subplot(rejilla[0, 0])
    franja = figura.add_subplot(rejilla[1, 0])
    franja.axis("off")

    resumen = plot_frontier_panel(
        panel,
        points,
        objectives[:2],
        floors,
        strict,
        robust,
        title=(
            f"Frontera de Pareto en {device_id}: {objectives[0].name}, "
            f"{objectives[1].name} y {objectives[2].name}"
        ),
        sizes=areas,
        draw_envelope=False,
    )

    _fill_legend_band(
        franja,
        resumen["families"],
        _annotation_handles(
            cross_session=bool(resumen["cross_session"]),
            halo=bool(resumen["robust_only"]),
            envelope=False,
            dominated=bool(resumen["dominated_marked"]),
            indistinguishable=resumen["indistinguishable_drawn"] > 0,
        ),
        _floor_footer(
            floors,
            objectives[:2],
            resumen["floor_anchor"],
            cross_session_applies=bool(resumen["cross_session"]),
        ),
        tamanos=_size_handles(rango),
        notas=[
            nota
            for nota in (_indistinguishable_note([("", resumen["indistinguishable_drawn"])]),)
            if nota is not None
        ],
    )

    resumen["memory_range"] = list(rango)
    return figura, {"__dispositivo__": resumen}


#: Título corto de cada vista.
VIEW_TITLES: dict[str, str] = {
    "plano_map5095_p50": "mAP@50-95 / p50",
    "plano_map50_p50": "mAP@50 / p50",
    "spec_3d_map5095_p95_memoria": "mAP@50-95 / p95 / memoria",
}


#: La cadena de ascendencia está en figures_results y se reexporta aquí.
experiment_chain = figures_results.experiment_chain


def point_identities(
    points: Iterable[Point], experiments: Mapping[str, Mapping[str, str]]
) -> dict[str, dict[str, Any]]:
    """Etiqueta, categoría, si es control, tamaño y experiment_id de cada punto.

    Falla si dos puntos del mismo motor quedan con la misma etiqueta.
    """
    identidades: dict[str, dict[str, Any]] = {}
    por_motor: dict[str, list[str]] = {}
    for punto in points:
        etiqueta, categoria, control = figures_results.configuration_label(
            punto.experiment_id, experiment_chain(punto.experiment_id, experiments), experiments
        )
        identidades[punto.label] = {
            "label": etiqueta,
            "category": categoria,
            "control": control,
            "model_size": (experiments[punto.experiment_id].get("model_size", "") or "").strip(),
            "experiment_id": punto.experiment_id,
        }
        por_motor.setdefault(punto.runtime, []).append(etiqueta)
    for motor, etiquetas in sorted(por_motor.items()):
        repetidas = sorted({e for e in etiquetas if etiquetas.count(e) > 1})
        if repetidas:
            raise ValueError(
                f"En {motor} hay configuraciones distintas con la misma etiqueta ({repetidas}): "
                f"distínguelas antes de dibujarlas."
            )
    return identidades


def category_style(category: str) -> Any:
    """Estilo de una categoría de las figuras por motor, el mismo que en la figura del 6.2."""
    if category == "baseline":
        return style.TECHNIQUE_STYLES["baseline"]
    return figures_results.COMPRESSION_CATEGORIES[category][0]


def category_label(category: str) -> str:
    """Rótulo de una categoría en la leyenda de las figuras por motor."""
    if category == "baseline":
        return "FP32 (referencia)"
    return figures_results.COMPRESSION_CATEGORIES[category][1]


def _single_session(points: Sequence[Point]) -> bool:
    """True si todos los puntos comparten sesión; las figuras por motor solo admiten una."""
    return all(same_session(points[0], p) for p in points[1:])


def _engine_runtimes(points_by_runtime: Mapping[str, Sequence[Point]]) -> tuple[str, ...]:
    """Comprueba que están los dos motores y que todos los puntos son de una sesión."""
    faltan = [m for m in ENGINE_RUNTIMES if not points_by_runtime.get(m)]
    if faltan:
        raise ValueError(f"Las figuras por motor necesitan puntos de {faltan} y no los hay.")
    todos = [p for m in ENGINE_RUNTIMES for p in points_by_runtime[m]]
    if not _single_session(todos):
        raise ValueError(
            "Los puntos de la figura son de más de una sesión de medida: el suelo que "
            "declara no gobernaría todas sus comparaciones."
        )
    return ENGINE_RUNTIMES


def _draw_identity(
    axes: Any, x: float, y: float, identity: Mapping[str, Any], *, attenuated: bool
) -> None:
    """Dibuja un punto con el estilo de su categoría; los controles aleatorios van huecos."""
    estilo = category_style(identity["category"])
    hueco = bool(identity["control"])
    axes.scatter(
        [x],
        [y],
        s=ENGINE_MARKER_AREA,
        marker=estilo.marker,
        facecolor="white" if hueco else estilo.color,
        edgecolor=estilo.color if hueco else "black",
        linewidth=0.9 if hueco else 0.4,
        alpha=DOMINATED_ALPHA if attenuated else 1.0,
        zorder=2.2 if attenuated else 3,
    )


def _ring(axes: Any, x: float, y: float, color: str, *, dotted: bool = False) -> None:
    """Anillo alrededor de un punto: «solo en la robusta» o «bloqueado bajo el suelo»."""
    axes.scatter(
        [x],
        [y],
        s=ENGINE_MARKER_AREA * RING_AREA_FACTOR,
        marker="o",
        facecolor="none",
        edgecolor=color,
        linewidth=1.0,
        linestyle=(0, (1, 1)) if dotted else "-",
        zorder=2.6,
    )


def _place_labels(
    axes: Any,
    entries: Sequence[tuple[float, float, str]],
    *,
    obstacles: Sequence[tuple[float, float]] = (),
    avoid: Sequence[Any] = (),
    prefer: str = "left",
    fontsize: float = ENGINE_FONTSIZE,
    rings: Sequence[tuple[float, float]] = (),
    spacing_pt: float = LABEL_SPACING_PT,
    clearance_pt: float = LABEL_CLEARANCE_PT,
    edges: Sequence[tuple[tuple[float, float], tuple[float, float]]] = (),
    avoid_leaders: bool = False,
    regions: Sequence[tuple[float, float, float, float]] = (),
    max_slide_pt: float = 0.0,
    background: bool = False,
) -> list[dict[str, Any]]:
    """Coloca las etiquetas junto a sus puntos sin pisar otras etiquetas, marcadores ni anillos.

    Cada etiqueta prueba los dos lados de su punto y el menor desplazamiento vertical libre; si se
    aparta, se une al punto con un trazo. ``edges`` y ``regions`` son segmentos y rectángulos que no
    puede cruzar. Con ``max_slide_pt`` > 0 también se desplaza en horizontal y gana la posición más
    cercana al punto. Hay que llamarla con la figura a su tamaño final, porque trabaja en
    coordenadas de pantalla.
    """
    import math

    if not entries:
        return []
    figura = axes.figure
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    ppp = figura.dpi
    caja_ejes = axes.get_window_extent(dibujo)
    separacion = spacing_pt * ppp / 72.0
    ventaja = 2.0 * ppp / 72.0
    # Tolerancia para que el redondeo no deje una etiqueta atascada.
    tolerancia = 1e-6
    radio = (math.sqrt(ENGINE_MARKER_AREA) / 2.0 + clearance_pt) * ppp / 72.0
    # Radio exterior del anillo con medio trazo, en puntos.
    anillo_pt = math.sqrt(ENGINE_MARKER_AREA * RING_AREA_FACTOR) / 2.0 + 0.5
    radio_anillo = (anillo_pt + clearance_pt) * ppp / 72.0
    anillados = {(float(x), float(y)) for x, y in rings}

    ocupadas: list[tuple[float, float, float, float]] = []
    for x, y in obstacles:
        px, py = axes.transData.transform((x, y))
        ocupadas.append((px - radio, px + radio, py - radio, py + radio))
    for x, y in anillados:
        px, py = axes.transData.transform((x, y))
        ocupadas.append((px - radio_anillo, px + radio_anillo, py - radio_anillo, py + radio_anillo))
    # Un segmento ocupa su caja envolvente, ensanchada medio punto.
    grueso = 0.5 * ppp / 72.0
    for inicio, fin in edges:
        (ax0, ay0), (ax1, ay1) = axes.transData.transform([inicio, fin])
        ocupadas.append(
            (min(ax0, ax1) - grueso, max(ax0, ax1) + grueso, min(ay0, ay1) - grueso, max(ay0, ay1) + grueso)
        )
    # Las regiones se recortan a los ejes antes de pasarlas a pantalla; los bordes infinitos no se
    # transforman en un eje logarítmico.
    (lim_x0, lim_x1), (lim_y0, lim_y1) = axes.get_xlim(), axes.get_ylim()
    for rx0, rx1, ry0, ry1 in regions:
        esquinas = axes.transData.transform(
            [
                (min(max(rx0, lim_x0), lim_x1), min(max(ry0, lim_y0), lim_y1)),
                (min(max(rx1, lim_x0), lim_x1), min(max(ry1, lim_y0), lim_y1)),
            ]
        )
        (qx0, qy0), (qx1, qy1) = esquinas
        ocupadas.append(
            (min(qx0, qx1) - grueso, max(qx0, qx1) + grueso, min(qy0, qy1) - grueso, max(qy0, qy1) + grueso)
        )
    for artista in avoid:
        caja = artista.get_window_extent(dibujo)
        ocupadas.append((caja.x0, caja.x1, caja.y0, caja.y1))

    def empujar(a0: float, a1: float, yc: float, alto: float, sentido: int) -> float:
        # Cada paso sale de una caja ocupada y no vuelve a entrar, así que hay como mucho tantos
        # pasos como cajas.
        for _ in range(len(ocupadas) + 2):
            movida = False
            for b0, b1, c0, c1 in ocupadas:
                if (
                    a0 < b1 - tolerancia
                    and a1 > b0 + tolerancia
                    and yc - alto / 2 < c1 + separacion - tolerancia
                    and yc + alto / 2 > c0 - separacion + tolerancia
                ):
                    yc = c1 + separacion + alto / 2 if sentido > 0 else c0 - separacion - alto / 2
                    movida = True
            if not movida:
                break
        return yc

    lados = ("left", "right") if prefer == "left" else ("right", "left")
    medidas = []
    for x, y, texto in entries:
        provisional = axes.text(0.0, 0.0, texto, fontsize=fontsize)
        caja = provisional.get_window_extent(dibujo)
        provisional.remove()
        px, py = axes.transData.transform((x, y))
        medidas.append((x, y, texto, px, py, caja.width, caja.height))

    def corta(
        segmento: tuple[float, float, float, float], caja: tuple[float, float, float, float]
    ) -> bool:
        # Recorte de Liang-Barsky.
        x0, y0, x1, y1 = segmento
        b0, b1, c0, c1 = caja
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - b0), (dx, b1 - x0), (-dy, y0 - c0), (dy, c1 - y0)):
            if p == 0:
                if q < 0:
                    return False
                continue
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
        return True

    umbral_trazo = 1.5 * ppp / 72.0
    cajas_etiquetas: list[tuple[float, float, float, float]] = []
    trazos: list[tuple[float, float, float, float]] = []

    colocadas: list[dict[str, Any]] = [{} for _ in medidas]
    flecha_base = {
        "arrowstyle": "-",
        "linewidth": 0.4,
        "color": "#666666",
        "shrinkA": 0.0,
        "shrinkB": 2.5,
    }

    if max_slide_pt > 0:
        disposicion = _free_label_layout(
            medidas,
            ocupadas,
            anillados,
            caja_ejes=(caja_ejes.x0, caja_ejes.x1, caja_ejes.y0, caja_ejes.y1),
            ppp=ppp,
            anillo_pt=anillo_pt,
            separacion=separacion,
            max_slide_pt=max_slide_pt,
            prefer=prefer,
            avoid_leaders=avoid_leaders,
        )
        for i, (yc, a0, a1, ox, oy, con_trazo) in disposicion.items():
            x, y, texto, px, py, ancho, alto = medidas[i]
            flecha = None
            if con_trazo:
                flecha = dict(flecha_base)
                flecha["relpos"] = ((ox - a0) / ancho, (oy - (yc - alto / 2)) / alto)
            axes.annotate(
                texto,
                xy=(x, y),
                xytext=((a0 - px) * 72.0 / ppp, (yc - py) * 72.0 / ppp),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=fontsize,
                arrowprops=flecha,
                bbox=dict(LABEL_BACKGROUND) if background else None,
                zorder=4,
            )
            colocadas[i] = {
                "text": texto,
                "side": "left" if (a0 + a1) / 2.0 < px else "right",
                "offset_pt": round((yc - py) * 72.0 / ppp, 2),
                "dx_pt": round((a0 - px) * 72.0 / ppp, 2),
            }
        return colocadas

    for i in sorted(range(len(medidas)), key=lambda j: medidas[j][4]):
        x, y, texto, px, py, ancho, alto = medidas[i]
        # Con anillo, la etiqueta se aparta hasta fuera del anillo.
        hueco_pt = LABEL_GAP_PT + (anillo_pt if (float(x), float(y)) in anillados else 0.0)
        hueco = hueco_pt * ppp / 72.0
        mejor: tuple[float, str, float, float, float] | None = None
        mejor_cruzando: tuple[float, str, float, float, float] | None = None
        for orden, lado in enumerate(lados):
            a0 = px - hueco - ancho if lado == "left" else px + hueco
            a1 = a0 + ancho
            if a0 < caja_ejes.x0 + 1 or a1 > caja_ejes.x1 - 1:
                continue
            for sentido in (+1, -1):
                yc = empujar(a0, a1, py, alto, sentido)
                if yc - alto / 2 < caja_ejes.y0 or yc + alto / 2 > caja_ejes.y1:
                    continue
                coste = abs(yc - py) + (ventaja if orden else 0.0)
                candidata = (coste, lado, yc, a0, a1)
                caja = (a0, a1, yc - alto / 2, yc + alto / 2)
                # El trazo de annotate va del centro del texto al punto.
                trazo = ((a0 + a1) / 2.0, yc, px, py)
                cruza = avoid_leaders and (
                    (abs(yc - py) > umbral_trazo and any(corta(trazo, c) for c in cajas_etiquetas))
                    or any(corta(t, caja) for t in trazos)
                )
                if cruza:
                    if mejor_cruzando is None or coste < mejor_cruzando[0]:
                        mejor_cruzando = candidata
                    continue
                if mejor is None or coste < mejor[0]:
                    mejor = candidata
        if mejor is None and mejor_cruzando is not None:
            logger.warning("El trazo de la etiqueta '%s' cruza otra etiqueta.", texto.split("\n")[0])
            mejor = mejor_cruzando
        if mejor is None:
            # Si no cabe, se queda pegada al punto; la revisión visual decide si se lee.
            logger.warning("La etiqueta '%s' no cabe sin pisar algo.", texto.split("\n")[0])
            a0 = px - hueco - ancho if lados[0] == "left" else px + hueco
            mejor = (0.0, lados[0], py, a0, a0 + ancho)
        _, lado, yc, a0, a1 = mejor
        ocupadas.append((a0, a1, yc - alto / 2, yc + alto / 2))
        cajas_etiquetas.append((a0, a1, yc - alto / 2, yc + alto / 2))
        if abs(yc - py) > umbral_trazo:
            trazos.append(((a0 + a1) / 2.0, yc, px, py))
        desplazamiento = (yc - py) * 72.0 / ppp
        flecha = dict(flecha_base) if abs(desplazamiento) > 1.5 else None
        axes.annotate(
            texto,
            xy=(x, y),
            xytext=(-hueco_pt if lado == "left" else hueco_pt, desplazamiento),
            textcoords="offset points",
            ha="right" if lado == "left" else "left",
            va="center",
            fontsize=fontsize,
            arrowprops=flecha,
            bbox=dict(LABEL_BACKGROUND) if background else None,
            zorder=4,
        )
        colocadas[i] = {"text": texto, "side": lado, "offset_pt": round(desplazamiento, 2)}
    return colocadas


def _segment_hits_box(
    segment: tuple[float, float, float, float], box: tuple[float, float, float, float]
) -> bool:
    """True si el segmento atraviesa la caja (recorte de Liang-Barsky)."""
    x0, y0, x1, y1 = segment
    # Rozar el borde no cuenta: la caja se reduce un píxel por lado.
    b0, b1, c0, c1 = box[0] + 1.0, box[1] - 1.0, box[2] + 1.0, box[3] - 1.0
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - b0), (dx, b1 - x0), (-dy, y0 - c0), (dy, c1 - y0)):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def _segments_cross(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    """True si dos segmentos se cortan en un punto interior de ambos."""

    def lado(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> float:
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    d1 = lado(b[0], b[1], b[2], b[3], a[0], a[1])
    d2 = lado(b[0], b[1], b[2], b[3], a[2], a[3])
    d3 = lado(a[0], a[1], a[2], a[3], b[0], b[1])
    d4 = lado(a[0], a[1], a[2], a[3], b[2], b[3])
    return d1 * d2 < 0 and d3 * d4 < 0


def _free_label_layout(
    measures: Sequence[tuple[float, float, str, float, float, float, float]],
    occupied: Sequence[tuple[float, float, float, float]],
    ringed: set[tuple[float, float]],
    *,
    caja_ejes: tuple[float, float, float, float],
    ppp: float,
    anillo_pt: float,
    separacion: float,
    max_slide_pt: float,
    prefer: str,
    avoid_leaders: bool,
) -> dict[int, tuple[float, float, float, float, float, bool]]:
    """Coloca las etiquetas en la zona libre más cercana a su punto.

    Prueba posiciones a izquierda, derecha y encima con el menor desplazamiento vertical libre, y se
    queda la más cercana. Como el resultado depende del orden, se prueban varios órdenes y se elige
    el de menos cruces y menor coste.
    """
    tolerancia = 1e-6
    ventaja = 2.0 * ppp / 72.0
    umbral_trazo = 1.5 * ppp / 72.0
    paso = LABEL_SLIDE_STEP_PT * ppp / 72.0
    deslizamiento = max_slide_pt * ppp / 72.0

    def empujar(
        ocupadas: list[tuple[float, float, float, float]],
        a0: float,
        a1: float,
        yc: float,
        alto: float,
        sentido: int,
    ) -> float:
        # La separación entre etiquetas se exige también en horizontal.
        for _ in range(len(ocupadas) + 2):
            movida = False
            for b0, b1, c0, c1 in ocupadas:
                if (
                    a0 < b1 + separacion - tolerancia
                    and a1 > b0 - separacion + tolerancia
                    and yc - alto / 2 < c1 + separacion - tolerancia
                    and yc + alto / 2 > c0 - separacion + tolerancia
                ):
                    yc = c1 + separacion + alto / 2 if sentido > 0 else c0 - separacion - alto / 2
                    movida = True
            if not movida:
                break
        return yc

    def colocar(orden: Sequence[int]) -> tuple[int, float, dict[int, Any]]:
        ocupadas = list(occupied)
        cajas: list[tuple[float, float, float, float]] = []
        trazos: list[tuple[float, float, float, float]] = []
        resultado: dict[int, Any] = {}
        cruces, total = 0, 0.0
        for i in orden:
            x, y, _, px, py, ancho, alto = measures[i]
            hueco = (LABEL_GAP_PT + (anillo_pt if (float(x), float(y)) in ringed else 0.0)) * ppp / 72.0
            inicio = px - hueco - ancho - deslizamiento
            pasos = int((2 * deslizamiento + 2 * hueco + ancho) // paso) + 1
            posiciones = [px - hueco - ancho, px + hueco, px - ancho / 2.0]
            posiciones += [inicio + k * paso for k in range(pasos)]
            elegida = None
            elegida_cruzando = None
            for a0 in posiciones:
                a1 = a0 + ancho
                if a0 < caja_ejes[0] + 1 or a1 > caja_ejes[1] - 1:
                    continue
                preferido = ((a0 + a1) / 2.0 < px) == (prefer == "left")
                for sentido in (+1, -1):
                    yc = empujar(ocupadas, a0, a1, py, alto, sentido)
                    y0, y1 = yc - alto / 2, yc + alto / 2
                    if y0 < caja_ejes[2] or y1 > caja_ejes[3]:
                        continue
                    ox, oy = min(max(px, a0), a1), min(max(py, y0), y1)
                    distancia = math.hypot(ox - px, oy - py)
                    coste = distancia + (0.0 if preferido else ventaja)
                    # Trazo siempre que el texto no esté junto a su punto, para no confundirlo con
                    # un vecino.
                    con_trazo = distancia > hueco + umbral_trazo
                    caja = (a0, a1, y0, y1)
                    trazo = (ox, oy, px, py)
                    cruza = avoid_leaders and (
                        (
                            con_trazo
                            and (
                                any(_segment_hits_box(trazo, c) for c in cajas)
                                or any(_segments_cross(trazo, t) for t in trazos)
                            )
                        )
                        or any(_segment_hits_box(t, caja) for t in trazos)
                    )
                    candidata = (coste, yc, a0, a1, ox, oy, con_trazo)
                    if cruza:
                        if elegida_cruzando is None or coste < elegida_cruzando[0]:
                            elegida_cruzando = candidata
                        continue
                    if elegida is None or coste < elegida[0]:
                        elegida = candidata
            if elegida is None and elegida_cruzando is not None:
                cruces += 1
                elegida = elegida_cruzando
            if elegida is None:
                cruces += 1
                a0 = px - hueco - ancho if prefer == "left" else px + hueco
                elegida = (0.0, py, a0, a0 + ancho, a0 + ancho if prefer == "left" else a0, py, False)
            coste, yc, a0, a1, ox, oy, con_trazo = elegida
            total += coste
            ocupadas.append((a0, a1, yc - alto / 2, yc + alto / 2))
            cajas.append((a0, a1, yc - alto / 2, yc + alto / 2))
            if con_trazo:
                trazos.append((ox, oy, px, py))
            resultado[i] = (yc, a0, a1, ox, oy, con_trazo)
        return cruces, total, resultado

    import random

    indices = list(range(len(measures)))
    ordenes = [
        sorted(indices, key=lambda j: measures[j][4]),
        sorted(indices, key=lambda j: -measures[j][4]),
        sorted(indices, key=lambda j: measures[j][3]),
        sorted(indices, key=lambda j: -measures[j][3]),
    ]
    # Órdenes aleatorios con semilla fija para que la figura sea reproducible.
    azar = random.Random(LABEL_LAYOUT_SEED)
    for _ in range(LABEL_LAYOUT_RANDOM_ORDERS):
        orden = list(indices)
        azar.shuffle(orden)
        ordenes.append(orden)
    mejor = min((colocar(orden) for orden in ordenes), key=lambda r: (r[0], r[1]))
    if mejor[0]:
        logger.warning(
            "%d etiquetas no tienen sitio sin que su trazo cruce algo o sin pisar algo.", mejor[0]
        )
    return mejor[2]


def _scale_bar(
    axes: Any,
    floor: AxisFloor | None,
    objective: Objective,
    *,
    vertical: bool,
    single_session: bool,
    position: tuple[float, float],
) -> dict[str, Any]:
    """Dibuja el suelo de un eje como segmento de escala con su valor al lado.

    En eje logarítmico un suelo relativo mide lo mismo en cualquier punto.
    """
    texto_x, texto_y = position
    if floor is None or not floor.is_measured():
        axes.text(
            texto_x,
            texto_y,
            f"{objective.name}: sin suelo medido",
            transform=axes.transAxes,
            ha="right",
            va="center",
            fontsize=ENGINE_FONTSIZE,
            color=FLOOR_COLOR,
        )
        return {"objective": objective.name, "floor": None}
    valor = floor.applicable(single_session)
    if valor is None:
        valor = floor.cross_session
    a_datos = axes.transData.inverted()
    fin = a_datos.transform(axes.transAxes.transform(position))
    if vertical:
        inicio = (fin[0], fin[1] - valor if floor.kind == "absoluto" else fin[1] / (1 + valor))
    else:
        inicio = (fin[0] - valor if floor.kind == "absoluto" else fin[0] / (1 + valor), fin[1])
    axes.plot(
        [inicio[0], fin[0]],
        [inicio[1], fin[1]],
        color=FLOOR_COLOR,
        linewidth=1.3,
        solid_capstyle="butt",
        marker="_" if vertical else "|",
        markersize=4,
        markeredgewidth=0.8,
        zorder=3.5,
    )
    rotulo = (
        f"suelo de {objective.name}: {format_number(valor, 4)}"
        if floor.kind == "absoluto"
        else f"suelo de {objective.name}: {format_number(100 * valor, 1)} %"
    )
    inicio_pantalla = axes.transData.transform(inicio)
    fin_pantalla = axes.transData.transform(fin)
    izquierda = min(inicio_pantalla[0], fin_pantalla[0])
    medio = (inicio_pantalla[1] + fin_pantalla[1]) / 2.0
    ancla = axes.transAxes.inverted().transform((izquierda, medio))
    axes.annotate(
        rotulo,
        xy=tuple(ancla),
        xycoords="axes fraction",
        xytext=(-3.0, 0.0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=ENGINE_FONTSIZE,
        color=FLOOR_COLOR,
    )
    return {"objective": objective.name, "floor": valor, "kind": floor.kind}


def _format_engine_axes(
    axes: Any,
    *,
    log_x: bool,
    show_x: bool,
    xlabel: str,
    ylabel: str | None,
    fontsize: float = ENGINE_FONTSIZE,
) -> None:
    """Marcas, rótulos y cuerpos de letra de un panel de las figuras por motor."""
    from matplotlib.ticker import FixedLocator, NullLocator

    if log_x:
        izquierda, derecha = axes.get_xlim()
        marcas = [
            v
            for v in (10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500)
            if izquierda <= v <= derecha
        ]
        axes.xaxis.set_major_locator(FixedLocator(marcas))
        axes.xaxis.set_minor_locator(NullLocator())
    axes.xaxis.set_major_formatter(comma_axis_formatter())
    axes.yaxis.set_major_formatter(comma_axis_formatter())
    axes.tick_params(labelsize=fontsize, length=2.5, pad=2, labelbottom=show_x)
    if show_x:
        axes.set_xlabel(xlabel, fontsize=fontsize, labelpad=2)
    if ylabel is not None:
        axes.set_ylabel(ylabel, fontsize=fontsize, labelpad=2)


def _engine_legend(
    figure: Any,
    categories: Iterable[str],
    *,
    controls: bool,
    dominated: bool,
    envelope: bool,
    robust_only: bool,
    extra: Sequence[tuple[Any, str]] = (),
    ncol: int = 5,
    floor: bool = True,
    fontsize: float = ENGINE_FONTSIZE,
) -> Any:
    """Leyenda común de las figuras por motor, arriba y a todo el ancho."""
    from matplotlib.lines import Line2D

    presentes = set(categories)
    apoderados: list[Any] = []
    rotulos: list[str] = []
    for categoria in ENGINE_CATEGORY_ORDER:
        if categoria not in presentes:
            continue
        estilo = category_style(categoria)
        apoderados.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker=estilo.marker,
                markerfacecolor=estilo.color,
                markeredgecolor="black",
                markeredgewidth=0.4,
                markersize=4.5,
            )
        )
        rotulos.append(category_label(categoria))
    if controls:
        apoderados.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#333333",
                markeredgewidth=0.9,
                markersize=4.5,
            )
        )
        rotulos.append("control aleatorio (hueco)")
    if dominated:
        apoderados.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="#000000",
                markeredgecolor="#000000",
                alpha=DOMINATED_ALPHA,
                markersize=4.5,
            )
        )
        rotulos.append("dominado (atenuado)")
    if envelope:
        apoderados.append(Line2D([], [], color=FRONTIER_LINE_COLOR, linewidth=1.1))
        rotulos.append("frontera estricta")
    if robust_only:
        apoderados.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="none",
                markeredgecolor=ROBUST_HALO_COLOR,
                markeredgewidth=1.0,
                markersize=8,
            )
        )
        rotulos.append("solo en la frontera robusta")
    for apoderado, rotulo in extra:
        apoderados.append(apoderado)
        rotulos.append(rotulo)
    if floor:
        apoderados.append(
            Line2D([], [], color=FLOOR_COLOR, linewidth=1.3, marker="|", markersize=4)
        )
        rotulos.append("suelo, a escala")
    return figure.legend(
        apoderados,
        rotulos,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        borderaxespad=0.1,
    )


def _limits(values: Sequence[float], *, log: bool, margin: float) -> tuple[float, float]:
    """Límites de un eje con margen (multiplicativo si es logarítmico)."""
    minimo, maximo = min(values), max(values)
    if log:
        return minimo / (1.0 + margin), maximo * (1.0 + margin)
    rango = maximo - minimo or abs(maximo) or 1.0
    return minimo - margin * rango, maximo + margin * rango


def _panel_summary(
    points: Sequence[Point],
    strict: FrontierResult,
    robust: FrontierResult,
    identities: Mapping[str, Mapping[str, Any]],
    objectives: Sequence[Objective],
) -> dict[str, Any]:
    """Resumen de un panel por motor para el manifiesto."""
    estricta = {p.label for p in strict.nondominated}
    return {
        "num_points": len(points),
        "members": [
            {
                "label": identities[p.label]["label"],
                "point": p.label,
                **{o.column: p.values[o.column] for o in objectives},
            }
            for p in robust.nondominated
        ],
        "dominated": sorted(
            identities[p.label]["label"]
            for p in points
            if p.label not in {q.label for q in robust.nondominated}
        ),
        "robust_only": sorted(
            identities[p.label]["label"] for p in robust.nondominated if p.label not in estricta
        ),
    }


def figure_engine_frontier(
    points_by_runtime: Mapping[str, Sequence[Point]],
    frontiers_by_runtime: Mapping[str, tuple[FrontierResult, FrontierResult]],
    objectives: Sequence[Objective],
    floors: FloorSet,
    identities: Mapping[str, Mapping[str, Any]],
    device_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Frontera de dos objetivos por motor, con los miembros rotulados.

    Un panel por motor, uno sobre otro, con latencia compartida en escala logarítmica (las
    compresiones van de 25 a 150 ms). Los dominados se dibujan atenuados y sin nombre.
    """
    import matplotlib.pyplot as plt

    if len(objectives) != 2:
        raise ValueError(f"Esta figura dibuja dos objetivos; se han pasado {len(objectives)}.")
    motores = _engine_runtimes(points_by_runtime)
    calidad, latencia = objectives
    todos = [p for m in motores for p in points_by_runtime[m]]
    xlim = _limits([p.values[latencia.column] for p in todos], log=True, margin=0.3)
    ylim = _limits([p.values[calidad.column] for p in todos], log=False, margin=0.06)

    ancho = ENGINE_FIGURE_WIDTH
    izquierda, derecha, abajo, arriba = 0.42, 0.06, 0.33, 0.62
    alto_panel, hueco = 2.45, 0.24
    alto = abajo + len(motores) * alto_panel + (len(motores) - 1) * hueco + arriba
    figura = plt.figure(figsize=(ancho, alto))

    resumen: dict[str, Any] = {"device_id": device_id, "runtimes": {}}
    categorias: set[str] = set()
    controles = solo_robusta = False
    for indice, motor in enumerate(motores):
        y0 = abajo + (len(motores) - 1 - indice) * (alto_panel + hueco)
        eje = figura.add_axes(
            (izquierda / ancho, y0 / alto, (ancho - izquierda - derecha) / ancho, alto_panel / alto)
        )
        estricta, robusta = frontiers_by_runtime[motor]
        miembros = {p.label for p in robusta.nondominated}
        en_estricta = {p.label for p in estricta.nondominated}
        eje.set_xscale("log")
        for punto in points_by_runtime[motor]:
            identidad = identities[punto.label]
            categorias.add(identidad["category"])
            controles = controles or bool(identidad["control"])
            _draw_identity(
                eje,
                punto.values[latencia.column],
                punto.values[calidad.column],
                identidad,
                attenuated=punto.label not in miembros,
            )
        escalera = _draw_strict_envelope(eje, estricta, objectives)
        anillados: list[tuple[float, float]] = []
        for punto in robusta.nondominated:
            if punto.label not in en_estricta:
                solo_robusta = True
                _ring(
                    eje, punto.values[latencia.column], punto.values[calidad.column], ROBUST_HALO_COLOR
                )
                anillados.append((punto.values[latencia.column], punto.values[calidad.column]))
        eje.set_xlim(xlim)
        eje.set_ylim(ylim)
        eje.set_title(ENGINE_TITLES[motor], fontsize=ENGINE_TITLE_FONTSIZE, pad=3, loc="left")
        _format_engine_axes(
            eje,
            log_x=True,
            show_x=indice == len(motores) - 1,
            xlabel=f"{_axis_label(latencia)}, escala logarítmica",
            ylabel=calidad.name,
        )
        un_solo = _single_session(todos)
        suelos = [
            _scale_bar(
                eje,
                floors.get(latencia.column),
                latencia,
                vertical=False,
                single_session=un_solo,
                position=(0.97, 0.05),
            ),
            _scale_bar(
                eje,
                floors.get(calidad.column),
                calidad,
                vertical=True,
                single_session=un_solo,
                position=(0.97, 0.20),
            ),
        ]
        rotulos = _place_labels(
            eje,
            [
                (p.values[latencia.column], p.values[calidad.column], identities[p.label]["label"])
                for p in robusta.nondominated
            ],
            obstacles=[
                (p.values[latencia.column], p.values[calidad.column])
                for p in points_by_runtime[motor]
            ],
            # Mismas reglas que la figura de nano: escalera y anillos son obstáculos.
            rings=anillados,
            edges=escalera,
            avoid_leaders=True,
            spacing_pt=NANO_LABEL_SPACING_PT,
            clearance_pt=NANO_LABEL_CLEARANCE_PT,
            max_slide_pt=LABEL_MAX_SLIDE_PT,
            background=True,
        )
        resumen["runtimes"][motor] = {
            **_panel_summary(points_by_runtime[motor], estricta, robusta, identities, objectives),
            "floors": suelos,
            "labels": rotulos,
        }

    _engine_legend(
        figura,
        categorias,
        controls=controles,
        dominated=True,
        envelope=True,
        robust_only=solo_robusta,
    )
    return figura, resumen


def figure_engine_three_objectives(
    points_by_runtime: Mapping[str, Sequence[Point]],
    frontiers_by_runtime: Mapping[str, tuple[FrontierResult, FrontierResult]],
    objectives: Sequence[Objective],
    floors: FloorSet,
    identities: Mapping[str, Mapping[str, Any]],
    device_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Frontera de tres objetivos por motor, con la memoria en su propio eje.

    Cada motor tiene dos paneles con la misma ordenada: calidad frente a p95 y frente a memoria. Los
    miembros llevan un número, igual en los dos paneles, con la clave debajo. No lleva escalera.
    """
    import math

    import matplotlib.pyplot as plt

    if len(objectives) != 3:
        raise ValueError(f"Esta figura dibuja tres objetivos; se han pasado {len(objectives)}.")
    motores = _engine_runtimes(points_by_runtime)
    calidad, latencia, memoria = objectives
    todos = [p for m in motores for p in points_by_runtime[m]]
    un_solo = _single_session(todos)
    limites_x = {
        latencia.column: _limits([p.values[latencia.column] for p in todos], log=True, margin=0.3),
        memoria.column: _limits([p.values[memoria.column] for p in todos], log=False, margin=0.05),
    }
    ylim = _limits([p.values[calidad.column] for p in todos], log=False, margin=0.06)

    columnas_clave = 4
    alto_linea = 0.118
    filas_clave = {
        m: math.ceil(len(frontiers_by_runtime[m][1].nondominated) / columnas_clave) for m in motores
    }
    ancho = ENGINE_FIGURE_WIDTH
    izquierda, derecha, hueco_columnas = 0.42, 0.06, 0.16
    ancho_panel = (ancho - izquierda - derecha - hueco_columnas) / 2.0
    alto_panel, titulo, rotulo_x, arriba, abajo = 1.95, 0.2, 0.34, 0.5, 0.04
    bloques = {m: titulo + alto_panel + rotulo_x + filas_clave[m] * alto_linea + 0.08 for m in motores}
    alto = arriba + sum(bloques.values()) + abajo
    figura = plt.figure(figsize=(ancho, alto))

    resumen: dict[str, Any] = {"device_id": device_id, "runtimes": {}}
    categorias: set[str] = set()
    controles = solo_robusta = False
    techo = alto - arriba
    for motor in motores:
        estricta, robusta = frontiers_by_runtime[motor]
        miembros = sorted(robusta.nondominated, key=lambda p: (p.values[latencia.column], p.label))
        numero = {p.label: i + 1 for i, p in enumerate(miembros)}
        en_estricta = {p.label for p in estricta.nondominated}
        y_panel = techo - titulo - alto_panel
        paneles = {}
        for columna, objetivo in enumerate((latencia, memoria)):
            eje = figura.add_axes(
                (
                    (izquierda + columna * (ancho_panel + hueco_columnas)) / ancho,
                    y_panel / alto,
                    ancho_panel / ancho,
                    alto_panel / alto,
                )
            )
            paneles[objetivo.column] = eje
            logaritmico = objetivo is latencia
            if logaritmico:
                eje.set_xscale("log")
            for punto in points_by_runtime[motor]:
                identidad = identities[punto.label]
                categorias.add(identidad["category"])
                controles = controles or bool(identidad["control"])
                _draw_identity(
                    eje,
                    punto.values[objetivo.column],
                    punto.values[calidad.column],
                    identidad,
                    attenuated=punto.label not in numero,
                )
            for punto in miembros:
                if punto.label not in en_estricta:
                    solo_robusta = True
                    _ring(
                        eje,
                        punto.values[objetivo.column],
                        punto.values[calidad.column],
                        ROBUST_HALO_COLOR,
                    )
            eje.set_xlim(limites_x[objetivo.column])
            eje.set_ylim(ylim)
            _format_engine_axes(
                eje,
                log_x=logaritmico,
                show_x=True,
                xlabel=(
                    f"{_axis_label(objetivo)}, escala logarítmica"
                    if logaritmico
                    else _axis_label(objetivo)
                ),
                ylabel=calidad.name if columna == 0 else None,
            )
            if columna == 1:
                eje.tick_params(labelleft=False)
        paneles[latencia.column].set_title(
            ENGINE_TITLES[motor], fontsize=ENGINE_TITLE_FONTSIZE, pad=3, loc="left"
        )
        suelos = [
            _scale_bar(
                paneles[latencia.column],
                floors.get(latencia.column),
                latencia,
                vertical=False,
                single_session=un_solo,
                position=(0.97, 0.05),
            ),
            _scale_bar(
                paneles[latencia.column],
                floors.get(calidad.column),
                calidad,
                vertical=True,
                single_session=un_solo,
                position=(0.97, 0.2),
            ),
            _scale_bar(
                paneles[memoria.column],
                floors.get(memoria.column),
                memoria,
                vertical=False,
                single_session=un_solo,
                position=(0.97, 0.05),
            ),
        ]
        for objetivo in (latencia, memoria):
            _place_labels(
                paneles[objetivo.column],
                [
                    (p.values[objetivo.column], p.values[calidad.column], str(numero[p.label]))
                    for p in miembros
                ],
                obstacles=[
                    (p.values[objetivo.column], p.values[calidad.column])
                    for p in points_by_runtime[motor]
                ],
                prefer="right",
            )

        # Clave en columnas bajo los dos paneles de la fila.
        clave = figura.add_axes(
            (
                izquierda / ancho,
                (y_panel - rotulo_x - filas_clave[motor] * alto_linea - 0.04) / alto,
                (ancho - izquierda - derecha) / ancho,
                filas_clave[motor] * alto_linea / alto,
            )
        )
        clave.axis("off")
        for i, punto in enumerate(miembros):
            fila, columna = divmod(i, columnas_clave)
            clave.text(
                columna / columnas_clave,
                1.0 - (fila + 0.5) / filas_clave[motor],
                f"{numero[punto.label]}  {identities[punto.label]['label']}",
                transform=clave.transAxes,
                ha="left",
                va="center",
                fontsize=ENGINE_FONTSIZE,
            )
        techo -= bloques[motor]
        resumen["runtimes"][motor] = {
            **_panel_summary(points_by_runtime[motor], estricta, robusta, identities, objectives),
            "numbering": {identities[p.label]["label"]: numero[p.label] for p in miembros},
            "floors": suelos,
        }

    _engine_legend(
        figura,
        categorias,
        controls=controles,
        dominated=True,
        envelope=False,
        robust_only=solo_robusta,
    )
    return figura, resumen


def blocked_below_floor(
    a: Point, b: Point, objectives: Sequence[Objective], floors: FloorSet
) -> list[Any] | None:
    """Ejes en los que ``a`` no domina a ``b`` solo por un déficit menor que el suelo.

    Devuelve las comparaciones con ese déficit, o None si ``a`` no gana en ningún eje por encima del
    suelo o si el bloqueo tiene otra causa.
    """
    if dominates(a, b, objectives, floors) is not None:
        return None
    misma = same_session(a, b)
    comparaciones = [compare(a, b, o, floors, misma) for o in objectives]
    deficits = [c for c in comparaciones if c.advantage < 0]
    if not deficits or any(c.verdict != "indistinguible" for c in deficits):
        return None
    if not any(c.verdict == "mejor" and c.certain for c in comparaciones):
        return None
    return deficits


def dominated_region(
    reference: Point, objectives: Sequence[Objective], floors: FloorSet
) -> list[tuple[float, float]]:
    """Polígono de la región que un punto domina en el plano calidad-latencia, según
    pareto.dominates en modo robusto.

    Es el cuadrante de peor calidad y más latencia menos la esquina donde ambas ventajas quedan bajo
    el suelo.
    """
    calidad, latencia = objectives
    if calidad.direction != "max" or latencia.direction != "min":
        raise ValueError("La región supone calidad a maximizar y latencia a minimizar.")
    suelo_calidad = floors.get(calidad.column)
    suelo_latencia = floors.get(latencia.column)
    if suelo_calidad is None or suelo_latencia is None:
        raise ValueError("La región necesita los suelos de los dos ejes.")
    q = suelo_calidad.applicable(True)
    f = suelo_latencia.applicable(True)
    if q is None or f is None:
        raise ValueError("La región necesita los suelos de la misma sesión de los dos ejes.")
    x, y = reference.values[latencia.column], reference.values[calidad.column]
    umbral_x = x * (1 + f) if suelo_latencia.kind == "relativo" else x + f
    umbral_y = y - q if suelo_calidad.kind == "absoluto" else y / (1 + q)
    infinito = 1e9
    return [
        (umbral_x, y),
        (infinito, y),
        (infinito, -infinito),
        (x, -infinito),
        (x, umbral_y),
        (umbral_x, umbral_y),
    ]


def region_rectangles(
    reference: Point, objectives: Sequence[Objective], floors: FloorSet
) -> list[tuple[float, float, float, float]]:
    """La región de dominated_region como unión de dos rectángulos ``(x0, x1, y0, y1)``."""
    (umbral_x, y), (infinito, _), (_, bajo), (x, _), (_, umbral_y), _ = dominated_region(
        reference, objectives, floors
    )
    return [(umbral_x, infinito, bajo, y), (x, infinito, bajo, umbral_y)]


def figure_nano_against_small(
    points_by_runtime: Mapping[str, Sequence[Point]],
    frontiers_by_runtime: Mapping[str, tuple[FrontierResult, FrontierResult]],
    objectives: Sequence[Objective],
    floors: FloorSet,
    identities: Mapping[str, Mapping[str, Any]],
    device_id: str,
) -> tuple[Any, dict[str, Any]]:
    """n FP32 y n PTQ frente a las compresiones de small, por motor.

    Las regiones sombreadas son lo que cada referencia de nano domina por encima del suelo. Los
    puntos que no domina solo por un déficit de calidad menor que el suelo llevan un anillo
    punteado; el detalle va en el resumen (``blocked_below_floor``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon

    if len(objectives) != 2:
        raise ValueError(f"Esta figura dibuja dos objetivos; se han pasado {len(objectives)}.")
    motores = _engine_runtimes(points_by_runtime)
    calidad, latencia = objectives

    def referencias(motor: str) -> list[Point]:
        nano = [
            p
            for p in points_by_runtime[motor]
            if identities[p.label]["model_size"] == "n"
            and identities[p.label]["category"] in NANO_REGION_COLORS
        ]
        if sorted(identities[p.label]["category"] for p in nano) != sorted(NANO_REGION_COLORS):
            raise ValueError(
                f"En {motor} no hay exactamente un FP32 y una PTQ de nano: "
                f"{[identities[p.label]['label'] for p in nano]}."
            )
        return sorted(nano, key=lambda p: list(NANO_REGION_COLORS).index(
            identities[p.label]["category"]))

    seleccion = {
        m: [
            p
            for p in points_by_runtime[m]
            if identities[p.label]["model_size"] == "s" or p in referencias(m)
        ]
        for m in motores
    }
    todos = [p for m in motores for p in seleccion[m]]
    xlim = _limits(
        [p.values[latencia.column] for p in todos], log=False, margin=NANO_X_MARGIN
    )
    ylim = _limits([p.values[calidad.column] for p in todos], log=False, margin=0.07)

    ancho = ENGINE_FIGURE_WIDTH
    izquierda, derecha, abajo, arriba = 0.5, 0.06, 0.33, 0.8
    alto_panel, hueco = 2.55, 0.24
    alto = abajo + len(motores) * alto_panel + (len(motores) - 1) * hueco + arriba
    figura = plt.figure(figsize=(ancho, alto))

    resumen: dict[str, Any] = {"device_id": device_id, "runtimes": {}}
    categorias: set[str] = set()
    controles = bloqueos = False
    for indice, motor in enumerate(motores):
        y0 = abajo + (len(motores) - 1 - indice) * (alto_panel + hueco)
        eje = figura.add_axes(
            (izquierda / ancho, y0 / alto, (ancho - izquierda - derecha) / ancho, alto_panel / alto)
        )
        estricta, robusta = frontiers_by_runtime[motor]
        miembros = {p.label for p in robusta.nondominated}
        nano = referencias(motor)
        for referencia in nano:
            categoria = identities[referencia.label]["category"]
            eje.add_patch(
                Polygon(
                    dominated_region(referencia, objectives, floors),
                    closed=True,
                    facecolor=NANO_REGION_COLORS[categoria],
                    alpha=NANO_REGION_ALPHA,
                    edgecolor="none",
                    zorder=0.5,
                )
            )
            eje.add_patch(
                Polygon(
                    dominated_region(referencia, objectives, floors),
                    closed=True,
                    facecolor="none",
                    edgecolor=NANO_REGION_COLORS[categoria],
                    linewidth=0.6,
                    zorder=0.6,
                )
            )

        entradas = []
        regiones: dict[str, list[str]] = {identities[r.label]["label"]: [] for r in nano}
        bloqueados = []
        anillados: list[tuple[float, float]] = []
        for punto in seleccion[motor]:
            identidad = identities[punto.label]
            categorias.add(identidad["category"])
            controles = controles or bool(identidad["control"])
            x, y = punto.values[latencia.column], punto.values[calidad.column]
            _draw_identity(eje, x, y, identidad, attenuated=punto.label not in miembros)
            if punto in nano:
                entradas.append((x, y, identidad["label"]))
                continue
            for referencia in nano:
                if dominates(referencia, punto, objectives, floors) is not None:
                    regiones[identities[referencia.label]["label"]].append(identidad["label"])
            if punto.label not in miembros:
                continue
            for referencia in nano:
                deficits = blocked_below_floor(referencia, punto, objectives, floors)
                if not deficits:
                    continue
                bloqueos = True
                _ring(eje, x, y, FLOOR_COLOR, dotted=True)
                anillados.append((x, y))
                for c in deficits:
                    bloqueados.append(
                        {
                            "point": identidad["label"],
                            "reference": identities[referencia.label]["label"],
                            "objective": c.objective,
                            "column": c.column,
                            "deficit": -c.advantage,
                            "threshold": c.threshold,
                        }
                    )
            entradas.append((x, y, identidad["label"]))

        eje.set_xlim(xlim)
        eje.set_ylim(ylim)
        eje.set_title(ENGINE_TITLES[motor], fontsize=ENGINE_TITLE_FONTSIZE, pad=3, loc="left")
        _format_engine_axes(
            eje,
            log_x=False,
            show_x=indice == len(motores) - 1,
            xlabel=_axis_label(latencia),
            ylabel=calidad.name,
        )
        un_solo = _single_session(todos)
        suelos = [
            _scale_bar(
                eje,
                floors.get(latencia.column),
                latencia,
                vertical=False,
                single_session=un_solo,
                position=(0.97, 0.05),
            ),
            _scale_bar(
                eje,
                floors.get(calidad.column),
                calidad,
                vertical=True,
                single_session=un_solo,
                position=(0.97, 0.2),
            ),
        ]
        # Los anillos se explican en la leyenda; el detalle está en el manifiesto.
        _place_labels(
            eje,
            entradas,
            obstacles=[
                (p.values[latencia.column], p.values[calidad.column]) for p in seleccion[motor]
            ],
            rings=anillados,
            avoid_leaders=True,
            spacing_pt=NANO_LABEL_SPACING_PT,
            clearance_pt=NANO_LABEL_CLEARANCE_PT,
            # Las regiones completas son obstáculos para las etiquetas, que pueden desplazarse en
            # horizontal.
            regions=[
                rectangulo
                for referencia in nano
                for rectangulo in region_rectangles(referencia, objectives, floors)
            ],
            max_slide_pt=LABEL_MAX_SLIDE_PT,
            background=True,
        )
        resumen["runtimes"][motor] = {
            "references": [identities[r.label]["label"] for r in nano],
            "small_members": sorted(
                identities[p.label]["label"]
                for p in seleccion[motor]
                if p.label in miembros and p not in nano and identities[p.label]["model_size"] == "s"
            ),
            "dominated_by": {k: sorted(v) for k, v in regiones.items()},
            "blocked_below_floor": bloqueados,
            "floors": suelos,
        }

    extra: list[tuple[Any, str]] = []
    for categoria, color in NANO_REGION_COLORS.items():
        nombre = "n FP32" if categoria == "baseline" else "n PTQ"
        extra.append(
            (
                Patch(facecolor=color, alpha=NANO_REGION_ALPHA + 0.1, edgecolor=color, linewidth=0.6),
                f"dominado por {nombre}",
            )
        )
    if bloqueos:
        extra.append(
            (
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor=FLOOR_COLOR,
                    markeredgewidth=1.0,
                    markersize=8,
                ),
                "no dominado por un déficit bajo el suelo",
            )
        )
    _engine_legend(
        figura,
        categorias,
        controls=controles,
        dominated=True,
        envelope=False,
        robust_only=False,
        extra=extra,
        ncol=3,
    )
    return figura, resumen


#: Descripción y ubicación de la figura del resultado principal (capítulo 1). Se dibuja aquí porque
#: sale de las mismas fronteras que la del 6.4.
HEADLINE_DESCRIPTION = "resultado_frontera_por_motor"
HEADLINE_PLACEMENT = (1, 2)

#: Miembros que rotula la figura del resultado principal en los dos motores. Si alguno deja de estar
#: en la frontera, la figura falla.
HEADLINE_LABELS: tuple[str, ...] = ("n PTQ", "n FP32", "L1 r25", "s FP32", "m FP32", "l FP32")

#: Zona intermedia: miembros con poda r50 y cuantización, por categoría y ratio.
HEADLINE_ZONE_CATEGORIES: tuple[str, ...] = ("combined", "pruning_qat")
HEADLINE_ZONE_RATIO = "r50"
HEADLINE_ZONE_LABEL = "poda r50 con cuantización"
HEADLINE_ZONE_COLOR = "#6A3D9A"

#: Cuerpo de texto, el mínimo del capítulo 1.
HEADLINE_FONTSIZE = 7.0


def headline_zone(
    members: Sequence[Point], identities: Mapping[str, Mapping[str, Any]]
) -> list[Point]:
    """Miembros de la frontera con poda r50 y cuantización."""
    return [
        p
        for p in members
        if identities[p.label]["category"] in HEADLINE_ZONE_CATEGORIES
        and HEADLINE_ZONE_RATIO in identities[p.label]["label"].split()
    ]


def figure_headline_frontier(
    points_by_runtime: Mapping[str, Sequence[Point]],
    frontiers_by_runtime: Mapping[str, tuple[FrontierResult, FrontierResult]],
    objectives: Sequence[Objective],
    floors: FloorSet,
    identities: Mapping[str, Mapping[str, Any]],
    device_id: str,
) -> tuple[Any, dict[str, Any]]:
    """Resultado principal: la frontera por motor simplificada.

    Es la frontera de mAP@50-95 frente a p50 del apartado 6.4, con nombre solo en HEADLINE_LABELS y
    la zona de poda r50 con cuantización agrupada. No dibuja suelos.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    del floors  # La figura no dibuja suelos.
    if len(objectives) != 2:
        raise ValueError(f"Esta figura dibuja dos objetivos; se han pasado {len(objectives)}.")
    motores = _engine_runtimes(points_by_runtime)
    calidad, latencia = objectives
    todos = [p for m in motores for p in points_by_runtime[m]]
    xlim = _limits([p.values[latencia.column] for p in todos], log=True, margin=0.3)
    ylim = _limits([p.values[calidad.column] for p in todos], log=False, margin=0.06)

    ancho = ENGINE_FIGURE_WIDTH
    izquierda, derecha, abajo, arriba = 0.46, 0.06, 0.34, 0.5
    alto_panel, hueco = 2.05, 0.26
    alto = abajo + len(motores) * alto_panel + (len(motores) - 1) * hueco + arriba
    figura = plt.figure(figsize=(ancho, alto))

    resumen: dict[str, Any] = {"device_id": device_id, "runtimes": {}}
    categorias: set[str] = set()
    controles = False
    for indice, motor in enumerate(motores):
        y0 = abajo + (len(motores) - 1 - indice) * (alto_panel + hueco)
        eje = figura.add_axes(
            (izquierda / ancho, y0 / alto, (ancho - izquierda - derecha) / ancho, alto_panel / alto)
        )
        estricta, robusta = frontiers_by_runtime[motor]
        miembros = list(robusta.nondominated)
        por_etiqueta = {identities[p.label]["label"]: p for p in miembros}
        faltan = [e for e in HEADLINE_LABELS if e not in por_etiqueta]
        if faltan:
            raise ValueError(
                f"En {motor}, {faltan} no son miembros de la frontera robusta: la figura del "
                f"resultado principal los rotularía como si lo fueran."
            )
        zona = headline_zone(miembros, identities)
        if not zona:
            raise ValueError(
                f"En {motor} no hay ningún miembro de poda {HEADLINE_ZONE_RATIO} con "
                f"cuantización: la zona intermedia quedaría vacía."
            )

        eje.set_xscale("log")
        etiquetas_miembros = {p.label for p in miembros}
        for punto in points_by_runtime[motor]:
            identidad = identities[punto.label]
            categorias.add(identidad["category"])
            controles = controles or bool(identidad["control"])
            _draw_identity(
                eje,
                punto.values[latencia.column],
                punto.values[calidad.column],
                identidad,
                attenuated=punto.label not in etiquetas_miembros,
            )
        escalera = _draw_strict_envelope(eje, estricta, objectives)
        eje.set_xlim(xlim)
        eje.set_ylim(ylim)

        # La zona es un rectángulo con margen fijo en pantalla alrededor de sus puntos.
        figura.canvas.draw()
        pantalla = [
            eje.transData.transform((p.values[latencia.column], p.values[calidad.column]))
            for p in zona
        ]
        margen = 7.0 * figura.dpi / 72.0
        (zx0, zy0) = eje.transData.inverted().transform(
            (min(x for x, _ in pantalla) - margen, min(y for _, y in pantalla) - margen)
        )
        (zx1, zy1) = eje.transData.inverted().transform(
            (max(x for x, _ in pantalla) + margen, max(y for _, y in pantalla) + margen)
        )
        eje.add_patch(
            Rectangle(
                (zx0, zy0), zx1 - zx0, zy1 - zy0, facecolor=HEADLINE_ZONE_COLOR, alpha=0.08,
                edgecolor="none", zorder=1.2,
            )
        )
        eje.add_patch(
            Rectangle(
                (zx0, zy0), zx1 - zx0, zy1 - zy0, facecolor="none",
                edgecolor=HEADLINE_ZONE_COLOR, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1.3,
            )
        )
        bordes_zona = [
            ((zx0, zy0), (zx1, zy0)),
            ((zx1, zy0), (zx1, zy1)),
            ((zx1, zy1), (zx0, zy1)),
            ((zx0, zy1), (zx0, zy0)),
        ]
        eje.set_title(ENGINE_TITLES[motor], fontsize=7.5, pad=3, loc="left")
        _format_engine_axes(
            eje,
            log_x=True,
            show_x=indice == len(motores) - 1,
            xlabel=f"{_axis_label(latencia)}, escala logarítmica",
            ylabel=calidad.name,
            fontsize=HEADLINE_FONTSIZE,
        )
        centro_zona = (
            (zx0 * zx1) ** 0.5,
            (zy0 + zy1) / 2.0,
        )
        entradas = [
            (
                por_etiqueta[e].values[latencia.column],
                por_etiqueta[e].values[calidad.column],
                e,
            )
            for e in HEADLINE_LABELS
        ]
        entradas.append((*centro_zona, HEADLINE_ZONE_LABEL))
        rotulos = _place_labels(
            eje,
            entradas,
            obstacles=[
                (p.values[latencia.column], p.values[calidad.column])
                for p in points_by_runtime[motor]
            ],
            fontsize=HEADLINE_FONTSIZE,
            edges=[*escalera, *bordes_zona],
            regions=[(zx0, zx1, zy0, zy1)],
            avoid_leaders=True,
            spacing_pt=NANO_LABEL_SPACING_PT,
            clearance_pt=NANO_LABEL_CLEARANCE_PT,
            max_slide_pt=LABEL_MAX_SLIDE_PT,
            background=True,
        )
        resumen["runtimes"][motor] = {
            "labelled": list(HEADLINE_LABELS),
            "zone": sorted(identities[p.label]["label"] for p in zona),
            "members": sorted(identities[p.label]["label"] for p in miembros),
            "labels": rotulos,
        }

    _engine_legend(
        figura,
        categorias,
        controls=controles,
        dominated=True,
        envelope=True,
        robust_only=False,
        ncol=5,
        floor=False,
        fontsize=HEADLINE_FONTSIZE,
    )
    return figura, resumen


def figure_lost_dominations(
    counts: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Dominaciones estrictas frente a robustas por vista y agrupación.

    Un panel por agrupación con su propia escala. Las perdidas deben ser la diferencia entre
    estrictas y robustas, porque toda dominación robusta es estricta.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not counts:
        raise ValueError("No hay recuentos de dominaciones que dibujar.")
    for entrada in counts:
        if entrada["strict"] - entrada["robust"] != entrada["lost"]:
            raise ValueError(
                f"En {entrada['view']} / {entrada['group']} las perdidas ({entrada['lost']}) no "
                f"son la diferencia entre estrictas ({entrada['strict']}) y robustas "
                f"({entrada['robust']})."
            )
    grupos = [g for g in ("dispositivo", *ENGINE_RUNTIMES) if any(e["group"] == g for e in counts)]
    vistas = [v for v in VIEW_TITLES if any(e["view"] == v for e in counts)]
    indice = {(e["view"], e["group"]): e for e in counts}

    ancho = ENGINE_FIGURE_WIDTH
    izquierda, derecha, hueco, abajo, arriba = 1.42, 0.1, 0.26, 0.3, 0.22
    alto_fila = 0.36
    alto_panel = alto_fila * len(vistas)
    alto = abajo + alto_panel + 0.2 + arriba
    ancho_panel = (ancho - izquierda - derecha - hueco * (len(grupos) - 1)) / len(grupos)
    figura = plt.figure(figsize=(ancho, alto))
    barra = 0.34
    for columna, grupo in enumerate(grupos):
        eje = figura.add_axes(
            (
                (izquierda + columna * (ancho_panel + hueco)) / ancho,
                abajo / alto,
                ancho_panel / ancho,
                alto_panel / alto,
            )
        )
        maximo = max(indice[(v, grupo)]["strict"] for v in vistas if (v, grupo) in indice)
        for fila, vista in enumerate(vistas):
            if (vista, grupo) not in indice:
                continue
            e = indice[(vista, grupo)]
            eje.barh(fila - barra / 2, e["strict"], height=barra, color="#9E9E9E", edgecolor="black",
                     linewidth=0.4)
            eje.barh(fila + barra / 2, e["robust"], height=barra, color="white", edgecolor="black",
                     linewidth=0.4, hatch="////")
            eje.text(
                e["strict"] + 0.02 * maximo,
                fila - barra / 2,
                f"{e['strict']}",
                va="center",
                ha="left",
                fontsize=ENGINE_FONTSIZE,
            )
            eje.text(
                e["robust"] + 0.02 * maximo,
                fila + barra / 2,
                f"{e['robust']}  (−{e['lost']})" if e["lost"] else f"{e['robust']}  (0)",
                va="center",
                ha="left",
                fontsize=ENGINE_FONTSIZE,
                fontweight="bold" if e["lost"] else "normal",
            )
        eje.set_xlim(0, maximo * 1.45)
        eje.set_ylim(len(vistas) - 0.45, -0.55)
        eje.set_yticks(range(len(vistas)))
        eje.set_yticklabels(
            [VIEW_TITLES[v] for v in vistas] if columna == 0 else [""] * len(vistas),
            fontsize=ENGINE_FONTSIZE,
        )
        eje.tick_params(labelsize=ENGINE_FONTSIZE, length=2.5, pad=2)
        eje.grid(axis="y", visible=False)
        eje.set_xlabel("dominaciones", fontsize=ENGINE_FONTSIZE, labelpad=2)
        titulo = "Todos los motores" if grupo == "dispositivo" else ENGINE_TITLES[grupo]
        eje.set_title(titulo, fontsize=ENGINE_TITLE_FONTSIZE, pad=3, loc="left")
    figura.legend(
        [
            Patch(facecolor="#9E9E9E", edgecolor="black", linewidth=0.4),
            Patch(facecolor="white", edgecolor="black", linewidth=0.4, hatch="////"),
        ],
        [
            "estrictas: las que afirman los números",
            "robustas: las que sobreviven al suelo (perdidas entre paréntesis)",
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=ENGINE_FONTSIZE,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
        borderaxespad=0.1,
    )
    return figura, {
        "groups": grupos,
        "views": vistas,
        "counts": [dict(indice[(v, g)]) for g in grupos for v in vistas if (v, g) in indice],
    }


def figure_membership(
    frontiers: Mapping[str, Mapping[str, tuple[Sequence[Point], FrontierResult, FrontierResult]]],
    identities: Mapping[str, Mapping[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    """Matriz de pertenencia a la frontera por configuración, vista y motor."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columnas = [
        (vista, motor)
        for vista in VIEW_TITLES
        if vista in frontiers
        for motor in ENGINE_RUNTIMES
        if motor in frontiers[vista]
    ]
    if not columnas:
        raise ValueError("No hay fronteras por motor que resumir.")
    estado: dict[str, dict[str, str]] = {}
    conjuntos = []
    muestra: dict[str, Mapping[str, Any]] = {}
    for vista, motor in columnas:
        puntos, estricta, robusta = frontiers[vista][motor]
        en_estricta = {p.label for p in estricta.nondominated}
        en_robusta = {p.label for p in robusta.nondominated}
        etiquetas = set()
        for punto in puntos:
            identidad = identities[punto.label]
            etiquetas.add(identidad["label"])
            muestra.setdefault(identidad["label"], identidad)
            estado.setdefault(identidad["label"], {})[f"{vista}/{motor}"] = (
                "estricta"
                if punto.label in en_estricta
                else "solo robusta" if punto.label in en_robusta else "dominado"
            )
        conjuntos.append(etiquetas)
    if any(c != conjuntos[0] for c in conjuntos):
        raise ValueError(
            "Las columnas de la figura no tienen las mismas configuraciones: una fila "
            "vacía en una columna se leería como «dominado»."
        )
    tamanos = ("n", "s", "m", "l")
    filas = sorted(
        conjuntos[0],
        key=lambda e: (
            ENGINE_CATEGORY_ORDER.index(muestra[e]["category"]),
            tamanos.index(muestra[e]["model_size"]) if muestra[e]["model_size"] in tamanos else 9,
            e,
        ),
    )

    ancho = ENGINE_FIGURE_WIDTH
    alto_fila = 0.145
    izquierda, derecha, abajo, arriba, cabecera = 1.45, 0.25, 0.05, 0.22, 0.36
    alto_panel = alto_fila * len(filas)
    alto = abajo + alto_panel + cabecera + arriba
    figura = plt.figure(figsize=(ancho, alto))
    eje = figura.add_axes(
        (izquierda / ancho, abajo / alto, (ancho - izquierda - derecha) / ancho, alto_panel / alto)
    )
    for fila, etiqueta in enumerate(filas):
        identidad = muestra[etiqueta]
        estilo = category_style(identidad["category"])
        hueco = bool(identidad["control"])
        eje.scatter(
            [-0.85],
            [fila],
            s=16,
            marker=estilo.marker,
            facecolor="white" if hueco else estilo.color,
            edgecolor=estilo.color if hueco else "black",
            linewidth=0.9 if hueco else 0.4,
            clip_on=False,
        )
        for columna, (vista, motor) in enumerate(columnas):
            valor = estado[etiqueta][f"{vista}/{motor}"]
            if valor == "estricta":
                eje.scatter([columna], [fila], s=22, marker="s", color="black")
            elif valor == "solo robusta":
                eje.scatter(
                    [columna], [fila], s=22, marker="s", facecolor="white",
                    edgecolor=ROBUST_HALO_COLOR, linewidth=1.2,
                )
            else:
                eje.scatter([columna], [fila], s=3, marker="o", color="#BBBBBB")
    eje.set_xlim(-1.3, len(columnas) - 0.5)
    eje.set_ylim(len(filas) - 0.5, -0.5)
    eje.set_yticks(range(len(filas)))
    eje.set_yticklabels(filas, fontsize=ENGINE_FONTSIZE)
    eje.set_xticks(range(len(columnas)))
    eje.set_xticklabels(
        [ENGINE_TITLES[m] for _, m in columnas], fontsize=ENGINE_FONTSIZE, rotation=0
    )
    eje.xaxis.tick_top()
    eje.tick_params(length=0, pad=2, labelsize=ENGINE_FONTSIZE)
    eje.grid(False)
    for lado in ("left", "bottom", "top", "right"):
        eje.spines[lado].set_visible(False)
    for i, etiqueta in enumerate(filas[1:], start=1):
        if muestra[etiqueta]["category"] != muestra[filas[i - 1]]["category"]:
            eje.axhline(i - 0.5, color="#DDDDDD", linewidth=0.5, zorder=0)
    vistas = list(dict.fromkeys(v for v, _ in columnas))
    for indice_vista, vista in enumerate(vistas):
        posiciones = [i for i, (v, _) in enumerate(columnas) if v == vista]
        if indice_vista:
            eje.axvline(posiciones[0] - 0.5, color="#DDDDDD", linewidth=0.5, zorder=0)
        eje.annotate(
            VIEW_TITLES[vista],
            xy=((posiciones[0] + posiciones[-1]) / 2, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=ENGINE_FONTSIZE,
            fontweight="bold",
        )
    figura.legend(
        [
            Line2D([], [], linestyle="none", marker="s", color="black", markersize=4.5),
            Line2D(
                [], [], linestyle="none", marker="s", markerfacecolor="white",
                markeredgecolor=ROBUST_HALO_COLOR, markeredgewidth=1.2, markersize=4.5,
            ),
            Line2D([], [], linestyle="none", marker="o", color="#BBBBBB", markersize=2),
        ],
        ["en la frontera estricta", "solo en la robusta (al exigir el suelo)", "dominado"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=ENGINE_FONTSIZE,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
        borderaxespad=0.1,
    )
    nunca = sorted(e for e in filas if all(v == "dominado" for v in estado[e].values()))
    return figura, {
        "columns": [f"{v}/{m}" for v, m in columnas],
        "rows": [
            {
                "label": e,
                "category": muestra[e]["category"],
                "control": muestra[e]["control"],
                "states": dict(sorted(estado[e].items())),
            }
            for e in filas
        ],
        "never_member": nunca,
    }


def _slug(text: str) -> str:
    """Fragmento normalizado para el nombre de fichero."""
    from tfm_edge.common.paths import slugify

    return slugify(text)


def figure_description(view: str, device_id: str, kind: str) -> str:
    """Descripción para el nombre de fichero (``plano`` o ``memoria``)."""
    return _slug(f"frontera_{kind}_{view}_{device_id}")


def save_pareto_figures(
    figures: Iterable[tuple[str, Any]], output_dir: Path | str
) -> dict[str, dict[str, Path]]:
    """Guarda las figuras con el estilo y los nombres del proyecto."""
    escritas: dict[str, dict[str, Path]] = {}
    for descripcion, figura in figures:
        escritas[descripcion] = style.save_figure(
            figura, output_dir, *figure_placement(descripcion), descripcion
        )
    return escritas


def figure_placement(description: str) -> tuple[int, int]:
    """Capítulo y apartado de una figura: todas van al 6.4 salvo el resultado principal, que va al
    capítulo 1.
    """
    if description.startswith(HEADLINE_DESCRIPTION):
        return HEADLINE_PLACEMENT
    return CHAPTER, SECTION
