"""Figuras del capítulo 1 (introducción).

Tres son diagramas conceptuales (problema, técnicas y flujo del estudio); verify_study_scope
comprueba contra el registro los tamaños y técnicas que muestran. El espacio de decisión es
esquemático, con puntos sintéticos y ejes sin marcas. La muestra de SH17 monta cuatro imágenes con
sus anotaciones y toma la atribución de docs/SH17_ATTRIBUTIONS.md. La figura del resultado principal
la genera el paso 27. Todas se dibujan a 16 cm de ancho con texto de 7 pt o más.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.reporting import style

logger = get_logger(__name__)

#: Capítulo y apartado de cada figura (1.1 motivación, 1.2 objetivo, 1.3 metodología).
INTRO_CHAPTER = 1
PLACEMENT: dict[str, int] = {
    "problema_epi_en_el_borde": 1,
    "tecnicas_de_compresion": 1,
    "muestra_sh17": 1,
    "espacio_de_decision": 2,
    "flujo_del_estudio": 3,
}

#: Ancho de las figuras del capítulo, el de la caja de texto.
INTRO_FIGURE_WIDTH = 16.0 / 2.54

#: Margen lateral de los diagramas y ancho útil, en pulgadas. Al guardar se añaden 0,05 pulgadas por
#: lado.
DIAGRAM_MARGIN = 0.07
DIAGRAM_WIDTH = INTRO_FIGURE_WIDTH - 2 * DIAGRAM_MARGIN

#: Cuerpos de letra en puntos.
INTRO_FONTSIZE = 7.0
INTRO_TITLE_FONTSIZE = 8.0

#: Tipografía para los símbolos (✕); viene con Matplotlib.
SYMBOL_FONT = "DejaVu Sans"

#: Colores de zona del flujo del capítulo 4, para que los diagramas sean coherentes.
BOX_COLORS: dict[str, str] = {
    "captura": "#9DC3E6",
    "borde": "#9FD5B8",
    "salida": "#D4D4D4",
    "nube": "#F4B183",
}
EDGE_REGION_COLOR = "#DDF1E7"
BLOCKED_COLOR = "#B2182B"

#: Tamaños y técnicas que muestran los diagramas.
STUDY_MODEL_SIZES: tuple[str, ...] = ("n", "s", "m", "l")
STUDY_TECHNIQUES: tuple[str, ...] = ("ptq", "qat", "pruning", "distillation", "combined")

#: Imágenes de la muestra: escenas distintas, una persona, pocas cajas y equipos variados.
SAMPLE_IMAGES: tuple[str, ...] = (
    "pexels-photo-13930482.jpeg",
    "pexels-photo-2760241.jpeg",
    "pexels-photo-209230.jpeg",
    "pexels-photo-14793021.jpeg",
)

#: La muestra no difumina las cajas; se puede activar con ``blur=True`` en plot_sh17_sample.
SAMPLE_BLUR = False

#: Clases que se difuminan cuando se pide: cabeza y cara.
BLURRED_CLASSES: tuple[str, ...] = ("head", "face")

#: Radio del difuminado como fracción del lado mayor de la caja.
BLUR_FRACTION = 0.25

#: Las imágenes son de Pexels y SH17 las distribuye con CC BY-NC-SA 4.0; la figura es una obra
#: derivada con la misma licencia.
SAMPLE_LICENSES: dict[str, str] = {
    "origen": "Licencia de Pexels",
    "dataset": "CC BY-NC-SA 4.0 (SH17)",
    "figura": "CC BY-NC-SA 4.0, obra derivada",
}

#: Color de las cajas, el mismo que en los ejemplos del capítulo 3.
SAMPLE_BOX_COLOR = "#D55E00"


class IntroFigureError(TFMEdgeError):
    """Error al construir una figura del capítulo 1."""


# Piezas comunes de los diagramas


def _canvas(height: float) -> tuple[Any, Any]:
    """Figura de 16 cm de ancho con ejes en pulgadas, sin marcos ni marcas."""
    import matplotlib.pyplot as plt

    style.apply_style()
    figura = plt.figure(figsize=(INTRO_FIGURE_WIDTH, height))
    ejes = figura.add_axes(
        (DIAGRAM_MARGIN / INTRO_FIGURE_WIDTH, 0.0, DIAGRAM_WIDTH / INTRO_FIGURE_WIDTH, 1.0)
    )
    ejes.set_xlim(0.0, DIAGRAM_WIDTH)
    ejes.set_ylim(0.0, height)
    ejes.axis("off")
    return figura, ejes


def _box(
    axes: Any,
    cx: float,
    cy: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
    *,
    edgecolor: str = "black",
    linestyle: Any = "-",
) -> None:
    """Caja de diagrama con título en negrita y detalle debajo."""
    from matplotlib.patches import FancyBboxPatch

    axes.add_patch(
        FancyBboxPatch(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.06",
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=0.7,
            linestyle=linestyle,
            zorder=2,
        )
    )
    if not detail:
        axes.text(
            cx, cy, title, ha="center", va="center", fontsize=INTRO_TITLE_FONTSIZE,
            fontweight="bold", zorder=3, linespacing=1.1,
        )
        return
    lineas_titulo = title.count("\n") + 1
    lineas_detalle = detail.count("\n") + 1
    # Alto de línea en pulgadas.
    alto_titulo = lineas_titulo * INTRO_TITLE_FONTSIZE * 1.15 / 72.0
    alto_detalle = lineas_detalle * INTRO_FONTSIZE * 1.2 / 72.0
    hueco = 0.05
    total = alto_titulo + hueco + alto_detalle
    arriba = cy + total / 2
    axes.text(
        cx, arriba - alto_titulo / 2, title, ha="center", va="center",
        fontsize=INTRO_TITLE_FONTSIZE, fontweight="bold", zorder=3, linespacing=1.1,
    )
    axes.text(
        cx, arriba - alto_titulo - hueco - alto_detalle / 2, detail, ha="center",
        va="center", fontsize=INTRO_FONTSIZE, zorder=3, linespacing=1.2,
    )


def _arrow(
    axes: Any, start: tuple[float, float], end: tuple[float, float], **kwargs: Any
) -> None:
    """Flecha del diagrama; ``kwargs`` se pasa a FancyArrowPatch."""
    from matplotlib.patches import FancyArrowPatch

    opciones: dict[str, Any] = {
        "arrowstyle": "-|>",
        "mutation_scale": 9,
        "linewidth": 1.0,
        "color": "black",
        "shrinkA": 0,
        "shrinkB": 0,
        "zorder": 4,
    }
    opciones.update(kwargs)
    axes.add_patch(FancyArrowPatch(start, end, **opciones))


# Comprobación de los diagramas contra el registro


def verify_study_scope(experiments: Sequence[Mapping[str, str]]) -> None:
    """Comprueba que el registro tiene los tamaños y técnicas que se dibujan."""
    tamanos = {
        (f.get("model_size", "") or "").strip()
        for f in experiments
        if (f.get("technique", "") or "").strip() == "baseline"
    }
    tecnicas = {(f.get("technique", "") or "").strip() for f in experiments}
    faltan_tamanos = sorted(set(STUDY_MODEL_SIZES) - tamanos)
    faltan_tecnicas = sorted(set(STUDY_TECHNIQUES) - tecnicas)
    if faltan_tamanos or faltan_tecnicas:
        raise IntroFigureError(
            f"El registro no tiene lo que dibujan los diagramas del capítulo 1: "
            f"baselines {faltan_tamanos}, técnicas {faltan_tecnicas}.",
            "Corrige STUDY_MODEL_SIZES / STUDY_TECHNIQUES o el dibujo.",
        )


# Figura del problema


def plot_problem(output_dir: Path | str) -> dict[str, Path]:
    """Esquema del problema: de la cámara al aviso sin salir del dispositivo.

    Abajo, el procesamiento en el borde; arriba, la nube descartada por conectividad y privacidad.
    """
    from matplotlib.patches import FancyBboxPatch

    alto = 3.15
    figura, ejes = _canvas(alto)
    fila = 1.05

    _box(
        ejes, 0.78, fila, 1.36, 0.95, "Cámara\nindustrial", "capta a los\ntrabajadores",
        BOX_COLORS["captura"],
    )

    # El dispositivo en el borde, con su restricción.
    izq, der, abajo, arriba = 1.82, 4.62, 0.12, 2.02
    ejes.add_patch(
        FancyBboxPatch(
            (izq, abajo), der - izq, arriba - abajo,
            boxstyle="round,pad=0.0,rounding_size=0.08", facecolor=EDGE_REGION_COLOR,
            edgecolor="#2E7D4F", linewidth=1.0, linestyle=(0, (4, 2)), zorder=1,
        )
    )
    centro = (izq + der) / 2
    ejes.text(
        centro, arriba - 0.17, "Dispositivo en el borde", ha="center", va="center",
        fontsize=INTRO_TITLE_FONTSIZE, fontweight="bold", zorder=3,
    )
    _box(
        ejes, centro, fila + 0.08, 2.2, 0.82, "Detector de objetos",
        "¿lleva cada trabajador\nsu equipo de protección?", BOX_COLORS["borde"],
    )
    ejes.text(
        centro, abajo + 0.2, "memoria, cómputo y energía limitados", ha="center",
        va="center", fontsize=INTRO_FONTSIZE, style="italic", color="#1B5E20", zorder=3,
    )

    _box(
        ejes, 5.46, fila, 1.36, 0.95, "Aviso local", "la imagen no sale\nde la planta",
        BOX_COLORS["salida"],
    )
    _arrow(ejes, (1.46, fila), (2.0, fila + 0.08))
    ejes.text(1.73, fila + 0.2, "imágenes", ha="center", va="bottom", fontsize=INTRO_FONTSIZE)
    _arrow(ejes, (centro + 1.1, fila + 0.08), (4.78, fila))

    # La nube descartada: flecha discontinua cortada y sus dos motivos.
    nube_y = 2.72
    _box(
        ejes, centro, nube_y, 2.2, 0.52, "Procesamiento en la nube", "",
        BOX_COLORS["nube"], linestyle=(0, (3, 2)),
    )
    _arrow(
        ejes, (centro, arriba), (centro, nube_y - 0.26), color="#555555",
        linestyle=(0, (3, 2)), linewidth=0.9,
    )
    ejes.text(
        centro, (arriba + nube_y - 0.26) / 2, "✕", ha="center", va="center", fontsize=12,
        color=BLOCKED_COLOR, fontfamily=SYMBOL_FONT, fontweight="bold", zorder=5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )
    ejes.text(
        centro - 0.25, (arriba + nube_y - 0.26) / 2, "conectividad\nno garantizada",
        ha="right", va="center", fontsize=INTRO_FONTSIZE, color=BLOCKED_COLOR,
        linespacing=1.15,
    )
    ejes.text(
        centro + 0.25, (arriba + nube_y - 0.26) / 2, "imágenes de personas:\ndatos personales",
        ha="left", va="center", fontsize=INTRO_FONTSIZE, color=BLOCKED_COLOR, linespacing=1.15,
    )
    return style.save_figure(
        figura, output_dir, INTRO_CHAPTER, PLACEMENT["problema_epi_en_el_borde"],
        "problema_epi_en_el_borde",
    )


# Figura de las técnicas


def plot_techniques(output_dir: Path | str) -> dict[str, Path]:
    """Las tres técnicas de compresión, una viñeta por técnica."""
    from matplotlib.patches import Rectangle

    alto = 2.35
    figura, ejes = _canvas(alto)
    anchura = DIAGRAM_WIDTH / 3
    centros = [anchura * (k + 0.5) for k in range(3)]
    titulos = ("Cuantización", "Poda estructurada", "Destilación")
    lemas = (
        "cada peso pasa de 32 a 8 bits",
        "se eliminan canales enteros",
        "el alumno aprende del maestro",
    )
    for cx, titulo, lema in zip(centros, titulos, lemas):
        ejes.text(
            cx, alto - 0.18, titulo, ha="center", va="center",
            fontsize=INTRO_TITLE_FONTSIZE, fontweight="bold",
        )
        ejes.text(cx, 0.16, lema, ha="center", va="center", fontsize=INTRO_FONTSIZE)
    for x in (anchura, 2 * anchura):
        ejes.plot([x, x], [0.08, alto - 0.08], color="#BBBBBB", linewidth=0.6)

    # Cuantización: un registro de 32 celdas y otro de 8.
    cx = centros[0]
    celda = 0.05
    for fila_y, bits, nombre, color in ((1.55, 32, "FP32", "#9DC3E6"), (0.75, 8, "INT8", "#2E75B6")):
        x0 = cx - 32 * celda / 2 + 0.17
        for k in range(bits):
            ejes.add_patch(
                Rectangle(
                    (x0 + k * celda, fila_y - 0.11), celda, 0.22, facecolor=color,
                    edgecolor="white", linewidth=0.4,
                )
            )
        ejes.text(x0 - 0.06, fila_y, nombre, ha="right", va="center", fontsize=INTRO_FONTSIZE)
    _arrow(ejes, (cx + 0.17 - 32 * celda / 2 + 4 * celda, 1.4), (cx + 0.17 - 32 * celda / 2 + 4 * celda, 0.9))

    # Poda: ocho canales, se eliminan los tres menos importantes.
    cx = centros[1]
    alturas = (0.78, 0.30, 0.66, 0.85, 0.22, 0.72, 0.27, 0.6)
    quitados = {1, 4, 6}
    # Con este paso todo cabe en el tercio: 8 × 0,12 + 0,32 + 5 × 0,12 = 1,88 pulgadas de 2,10.
    ancho_canal, paso = 0.09, 0.12
    x0 = cx - 0.94
    base = 0.55
    for k, h in enumerate(alturas):
        quitado = k in quitados
        ejes.add_patch(
            Rectangle(
                (x0 + k * paso, base), ancho_canal, h,
                facecolor="white" if quitado else "#F4B183",
                edgecolor="#999999" if quitado else "black", linewidth=0.6,
                linestyle=(0, (2, 1.5)) if quitado else "-",
            )
        )
        if quitado:
            ejes.text(
                x0 + k * paso + ancho_canal / 2, base + h + 0.1, "✕", ha="center", va="center",
                fontsize=8, color=BLOCKED_COLOR, fontfamily=SYMBOL_FONT,
            )
    _arrow(ejes, (x0 + 8 * paso + 0.02, base + 0.4), (x0 + 8 * paso + 0.26, base + 0.4))
    x1 = x0 + 8 * paso + 0.32
    for j, h in enumerate(h for k, h in enumerate(alturas) if k not in quitados):
        ejes.add_patch(
            Rectangle(
                (x1 + j * paso, base), ancho_canal, h, facecolor="#F4B183",
                edgecolor="black", linewidth=0.6,
            )
        )
    ejes.text(x0 + 4 * paso, base - 0.13, "capa original", ha="center", va="center", fontsize=INTRO_FONTSIZE)
    ejes.text(x1 + 2.5 * paso, base - 0.13, "capa podada", ha="center", va="center", fontsize=INTRO_FONTSIZE)

    # Destilación: maestro grande y alumno pequeño.
    cx = centros[2]
    _box(ejes, cx - 0.6, 1.12, 0.78, 1.0, "Maestro", "YOLOv8l", "#C5E0B4")
    _box(ejes, cx + 0.64, 1.12, 0.66, 0.62, "Alumno", "YOLOv8n", "#9FD5B8")
    _arrow(ejes, (cx - 0.19, 1.12), (cx + 0.29, 1.12))
    ejes.text(cx + 0.05, 1.18, "salidas", ha="center", va="bottom", fontsize=INTRO_FONTSIZE)
    return style.save_figure(
        figura, output_dir, INTRO_CHAPTER, PLACEMENT["tecnicas_de_compresion"],
        "tecnicas_de_compresion",
    )


# Figura del flujo del estudio

#: Etapas del flujo de izquierda a derecha: título y detalle.
STUDY_FLOW: tuple[tuple[str, str], ...] = (
    ("SH17", "imágenes de EPI\nen la industria"),
    ("Entrenamiento", "YOLOv8\nn, s, m y l"),
    ("Compresión", "cuantización, poda,\ndestilación y\ncombinaciones"),
    ("Medida\nhomogénea", "calidad y\ndespliegue"),
    ("Frontera\nde Pareto", "precisión, latencia\ny memoria"),
)


def plot_study_flow(
    output_dir: Path | str, experiments: Sequence[Mapping[str, str]]
) -> dict[str, Path]:
    """Flujo del estudio de los datos a la frontera, contrastado con el registro."""
    verify_study_scope(experiments)
    alto = 1.45
    figura, ejes = _canvas(alto)
    colores = ("#D4D4D4", "#9DC3E6", "#F4B183", "#9FD5B8", "#C9B8E0")
    n = len(STUDY_FLOW)
    hueco = 0.27
    ancho = (DIAGRAM_WIDTH - 0.08 - (n - 1) * hueco) / n
    for k, ((titulo, detalle), color) in enumerate(zip(STUDY_FLOW, colores)):
        cx = 0.04 + ancho / 2 + k * (ancho + hueco)
        _box(ejes, cx, alto / 2, ancho, alto - 0.12, titulo, detalle, color)
        if k:
            _arrow(ejes, (cx - ancho / 2 - hueco + 0.03, alto / 2), (cx - ancho / 2 - 0.03, alto / 2))
    return style.save_figure(
        figura, output_dir, INTRO_CHAPTER, PLACEMENT["flujo_del_estudio"], "flujo_del_estudio"
    )


# Figura del espacio de decisión

DECISION_SPACE_SEED = 1
DECISION_SPACE_POINTS = 45

#: Restricción del dispositivo en unidades normalizadas: latencia y memoria máximas.
DECISION_SPACE_LIMITS = (0.55, 0.6)


def decision_space_ceiling(latency: Any, memory: Any) -> Any:
    """Frontera esquemática: mejor precisión alcanzable con una latencia y una memoria normalizadas.

    Crece con las dos y se satura, como la frontera medida.
    """
    import numpy as np

    return 1.0 - 0.8 * np.exp(-2.4 * (0.55 * latency + 0.45 * memory))


def decision_space_points() -> tuple[Any, tuple[float, float, float]]:
    """Configuraciones sintéticas por debajo de la frontera y la configuración elegida.

    La elegida es el punto de la frontera con más precisión dentro de la restricción.
    """
    import numpy as np

    azar = np.random.default_rng(DECISION_SPACE_SEED)
    latencia = azar.uniform(0.05, 1.0, DECISION_SPACE_POINTS)
    memoria = azar.uniform(0.05, 1.0, DECISION_SPACE_POINTS)
    precision = decision_space_ceiling(latencia, memoria) * azar.uniform(
        0.55, 0.95, DECISION_SPACE_POINTS
    )
    limite_l, limite_m = DECISION_SPACE_LIMITS
    elegida = (limite_l, limite_m, float(decision_space_ceiling(limite_l, limite_m)))
    return np.column_stack([latencia, memoria, precision]), elegida


def plot_decision_space(output_dir: Path | str) -> dict[str, Path]:
    """Esquema 3D del espacio de decisión: precisión, latencia y memoria.

    La superficie es decision_space_ceiling, los puntos grises son sintéticos, la caja discontinua
    es la región admisible y la estrella la configuración elegida.
    """
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    puntos, elegida = decision_space_points()
    limite_l, limite_m = DECISION_SPACE_LIMITS
    color_frontera = "#2E75B6"
    color_region = "#2E7D4F"

    style.apply_style()
    # Los ejes 3D se agrandan para que la caja llegue a la leyenda sin margen blanco.
    figura = plt.figure(figsize=(INTRO_FIGURE_WIDTH, 2.85))
    ejes = figura.add_axes((0.13, -0.17, 0.74, 1.22), projection="3d")
    malla_l, malla_m = np.meshgrid(np.linspace(0.05, 1.0, 30), np.linspace(0.05, 1.0, 30))
    ejes.plot_surface(
        malla_l, malla_m, decision_space_ceiling(malla_l, malla_m), color=color_frontera,
        alpha=0.28, linewidth=0, antialiased=True, shade=False,
    )
    ejes.plot_wireframe(
        malla_l, malla_m, decision_space_ceiling(malla_l, malla_m), rstride=5, cstride=5,
        color=color_frontera, linewidth=0.3, alpha=0.6,
    )
    ejes.scatter(
        puntos[:, 0], puntos[:, 1], puntos[:, 2], s=9, color="#8C8C8C", depthshade=False
    )
    # Región admisible: caja de aristas discontinuas.
    esquinas = list(itertools.product((0.0, limite_l), (0.0, limite_m), (0.0, 1.0)))
    for a, b in itertools.combinations(esquinas, 2):
        if sum(x != y for x, y in zip(a, b)) == 1:
            ejes.plot(*zip(a, b), color=color_region, linewidth=1.0, linestyle=(0, (3, 2)))
    ejes.scatter(
        [elegida[0]], [elegida[1]], [elegida[2]], s=130, marker="*", color="#D55E00",
        edgecolor="black", linewidth=0.5, depthshade=False, zorder=10,
    )
    ejes.set_xlim(0, 1)
    ejes.set_ylim(0, 1)
    ejes.set_zlim(0, 1)
    ejes.set_xticks([])
    ejes.set_yticks([])
    ejes.set_zticks([])
    ejes.set_xlabel("latencia", fontsize=INTRO_FONTSIZE, labelpad=-12)
    ejes.set_ylabel("memoria", fontsize=INTRO_FONTSIZE, labelpad=-12)
    ejes.set_zlabel("precisión", fontsize=INTRO_FONTSIZE, labelpad=-12)
    ejes.view_init(elev=20, azim=-62)
    ejes.grid(False)
    for eje in (ejes.xaxis, ejes.yaxis, ejes.zaxis):
        eje.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
        eje.pane.set_edgecolor("#BBBBBB")

    manejadores = [
        Line2D([], [], linestyle="none", marker="o", markersize=4, color="#8C8C8C"),
        Patch(facecolor=color_frontera, alpha=0.45, edgecolor=color_frontera, linewidth=0.5),
        Line2D([], [], color=color_region, linewidth=1.0, linestyle=(0, (3, 2))),
        Line2D(
            [], [], linestyle="none", marker="*", markersize=9, markerfacecolor="#D55E00",
            markeredgecolor="black", markeredgewidth=0.5,
        ),
    ]
    figura.legend(
        manejadores,
        [
            "configuración posible",
            "frontera de Pareto",
            "lo que admite el dispositivo",
            "configuración elegida",
        ],
        loc="upper center",
        ncol=4,
        fontsize=INTRO_FONTSIZE,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    return style.save_figure(
        figura, output_dir, INTRO_CHAPTER, PLACEMENT["espacio_de_decision"],
        "espacio_de_decision",
    )


# Muestra de SH17


@dataclass(frozen=True)
class Attribution:
    """Atribución de una imagen de SH17."""

    image: str
    photographer: str
    source: str


def load_attributions(path: Path | str) -> dict[str, Attribution]:
    """Lee la tabla de detalle por imagen de SH17_ATTRIBUTIONS.md."""
    patron = re.compile(r"^\| `([^`]+)` \| (.+?) \| (https?://\S+) \|$")
    resultado: dict[str, Attribution] = {}
    for linea in Path(path).read_text(encoding="utf-8").splitlines():
        coincidencia = patron.match(linea.strip())
        if coincidencia:
            nombre, fotografo, origen = coincidencia.groups()
            resultado[nombre] = Attribution(nombre, " ".join(fotografo.split()), origen)
    if not resultado:
        raise IntroFigureError(
            f"{path} no tiene la tabla de detalle por imagen.",
            "Regenera el fichero con tfm_edge.data.prepare.extract_attributions.",
        )
    return resultado


def sample_attributions(
    attributions: Mapping[str, Attribution], images: Sequence[str] = SAMPLE_IMAGES
) -> list[dict[str, str]]:
    """Atribución y licencia de cada imagen de la muestra. Sin atribución, la figura no se genera.
    """
    salida = []
    faltan = []
    for fichero in images:
        clave = Path(fichero).stem
        if clave not in attributions:
            faltan.append(fichero)
            continue
        entrada = attributions[clave]
        salida.append(
            {
                "image": fichero,
                "photographer": entrada.photographer,
                "source": entrada.source,
                **{f"license_{k}": v for k, v in SAMPLE_LICENSES.items()},
            }
        )
    if faltan:
        raise IntroFigureError(
            f"Imágenes de la muestra sin atribución en SH17_ATTRIBUTIONS.md: {faltan}.",
            "Elige otra imagen o regenera el fichero: la licencia exige la atribución.",
        )
    return salida


def blur_boxes(image: Any, boxes: Sequence[tuple[float, float, float, float]]) -> Any:
    """Difumina el contenido de unas cajas ``(x, y, ancho, alto)`` de una imagen."""
    from PIL import ImageFilter

    salida = image.copy()
    for x, y, ancho, alto in boxes:
        caja = (
            max(0, int(x)),
            max(0, int(y)),
            min(image.width, int(round(x + ancho)) + 1),
            min(image.height, int(round(y + alto)) + 1),
        )
        if caja[2] <= caja[0] or caja[3] <= caja[1]:
            continue
        radio = max(3.0, BLUR_FRACTION * max(ancho, alto))
        salida.paste(image.crop(caja).filter(ImageFilter.GaussianBlur(radio)), caja[:2])
    return salida


def _place_box_labels(axes: Any, labels: Sequence[tuple[float, float, float, float, str]]) -> None:
    """Coloca los rótulos de las cajas sin que se pisen.

    Prueba cuatro posiciones (encima o debajo, por fuera o por dentro) y usa la primera libre.
    """
    figura = axes.figure
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    marco = axes.get_window_extent(dibujo)
    ocupadas: list[Any] = []
    for x, y, ancho, alto, texto in labels:
        colocado = None
        # En la imagen la y crece hacia abajo.
        for ty, va in ((y, "bottom"), (y, "top"), (y + alto, "top"), (y + alto, "bottom")):
            candidato = axes.text(
                x, ty, texto, fontsize=INTRO_FONTSIZE, color=SAMPLE_BOX_COLOR, va=va,
                ha="left", bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 0.6},
            )
            caja = candidato.get_window_extent(dibujo)
            dentro = (
                caja.x0 >= marco.x0 and caja.x1 <= marco.x1
                and caja.y0 >= marco.y0 and caja.y1 <= marco.y1
            )
            if dentro and not any(caja.overlaps(o) for o in ocupadas):
                colocado = candidato
                ocupadas.append(caja)
                break
            candidato.remove()
        if colocado is None:
            logger.warning("El rótulo '%s' no cabe sin pisar otro: se omite.", texto)


def plot_sh17_sample(
    coco: Mapping[str, Any],
    images_dir: Path | str,
    attributions: Mapping[str, Attribution],
    output_dir: Path | str,
    images: Sequence[str] = SAMPLE_IMAGES,
    blur: bool = SAMPLE_BLUR,
) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """Muestra de SH17 con sus anotaciones, tomadas del COCO.

    Con ``blur=True`` difumina las cajas de cabeza y cara sobre la imagen ya reducida.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    atribuciones = sample_attributions(attributions, images)
    nombres = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    por_nombre = {img["file_name"]: img for img in coco["images"]}
    por_imagen: dict[int, list[Mapping[str, Any]]] = {}
    for anotacion in coco["annotations"]:
        por_imagen.setdefault(anotacion["image_id"], []).append(anotacion)

    style.apply_style()
    columnas = 2
    filas = (len(images) + columnas - 1) // columnas
    figura, ejes = plt.subplots(filas, columnas, figsize=(INTRO_FIGURE_WIDTH, 2.25 * filas))
    paneles = list(ejes.flat)
    resumen = []
    for panel, fichero, atribucion in zip(paneles, images, atribuciones):
        info = por_nombre.get(fichero)
        if info is None or not por_imagen.get(info["id"]):
            raise IntroFigureError(
                f"La imagen '{fichero}' de la muestra no está en el COCO o no tiene anotaciones.",
                "No se sustituye por otra en silencio: elige otra en SAMPLE_IMAGES.",
            )
        anotaciones = por_imagen[info["id"]]
        with Image.open(Path(images_dir) / fichero) as original:
            original.draft("RGB", (1400, 1400))
            imagen = original.convert("RGB")
        escala = imagen.width / info["width"]
        borrosas = (
            [
                tuple(v * escala for v in a["bbox"])
                for a in anotaciones
                if nombres[int(a["category_id"])] in BLURRED_CLASSES
            ]
            if blur
            else []
        )
        if blur and not borrosas:
            raise IntroFigureError(
                f"La imagen '{fichero}' no tiene ninguna caja de {BLURRED_CLASSES}.",
                "Una persona sin cabeza anotada no se puede difuminar con la anotación: "
                "elige otra imagen.",
            )
        panel.imshow(blur_boxes(imagen, borrosas) if blur else imagen)
        rotulos = []
        for anotacion in sorted(anotaciones, key=lambda a: -a["bbox"][2] * a["bbox"][3]):
            x, y, ancho, alto = (v * escala for v in anotacion["bbox"])
            panel.add_patch(
                Rectangle((x, y), ancho, alto, fill=False, edgecolor=SAMPLE_BOX_COLOR, linewidth=0.9)
            )
            rotulos.append((x, y, ancho, alto, nombres[int(anotacion["category_id"])]))
        panel.set_axis_off()
        resumen.append(
            {
                **atribucion,
                "blurred": blur,
                "blurred_boxes": len(borrosas),
                "classes": sorted({r[4] for r in rotulos}),
                "labels": rotulos,
            }
        )
    for panel in paneles[len(images):]:
        panel.axis("off")
    # Margen lateral del 1,2 % para que el PNG no supere los 16 cm.
    figura.subplots_adjust(left=0.012, right=0.988, top=0.995, bottom=0.005, wspace=0.02, hspace=0.03)
    for panel, entrada in zip(paneles, resumen):
        _place_box_labels(panel, entrada.pop("labels"))
    rutas = style.save_figure(
        figura, output_dir, INTRO_CHAPTER, PLACEMENT["muestra_sh17"], "muestra_sh17"
    )
    return rutas, resumen


# Generación conjunta


def generate_intro_figures(
    output_dir: Path | str,
    experiments: Sequence[Mapping[str, str]],
    attributions_path: Path | str,
    coco_path: Path | str | None,
    images_dir: Path | str | None,
) -> dict[str, Any]:
    """Genera las cinco figuras del capítulo 1 que no salen del paso 27.

    Sin data/ se omite la muestra de SH17 y el manifiesto lo indica.
    """
    verify_study_scope(experiments)
    atribuciones = load_attributions(attributions_path)
    sample_attributions(atribuciones)
    manifiesto: dict[str, Any] = {"figures": {}, "skipped": {}}
    for clave, construir in (
        ("problema_epi_en_el_borde", lambda: plot_problem(output_dir)),
        ("tecnicas_de_compresion", lambda: plot_techniques(output_dir)),
        ("espacio_de_decision", lambda: plot_decision_space(output_dir)),
        ("flujo_del_estudio", lambda: plot_study_flow(output_dir, experiments)),
    ):
        manifiesto["figures"][clave] = {"files": style.manifest_files(construir())}
    if coco_path is None or images_dir is None or not Path(coco_path).is_file():
        manifiesto["skipped"]["muestra_sh17"] = (
            "Sin data/processed/sh17_coco.json o sin data/raw/images: data/ no se versiona."
        )
        return manifiesto
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    rutas, resumen = plot_sh17_sample(coco, images_dir, atribuciones, output_dir)
    manifiesto["figures"]["muestra_sh17"] = {
        "files": style.manifest_files(rutas),
        "images": resumen,
    }
    return manifiesto
