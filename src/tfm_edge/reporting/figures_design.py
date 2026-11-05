"""Figuras del capítulo 4 (diseño del estudio experimental).

El sesgo de selección de best.pt dibuja las curvas por época del validador de Ultralytics y las
cuatro evaluaciones con pycocotools, contrastadas antes de dibujar. El flujo experimental es un
diagrama cuyas escrituras en el registro se comprueban contra scripts/. La cadena de causas de QAT
es un diagrama de la secuencia de fallos agrupada por etapas. El suelo de error de latencia dibuja
los vectores completos de las dos tomas. La matriz de cobertura clasifica cada perfil y motor a
partir del registro y los perfiles. Van a 16 cm, con texto de 7 pt o más, colores planos y sin
título ni notas dentro de la imagen; la matriz añade un símbolo por celda para la escala de grises.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.reporting.style import (
    TECHNIQUE_STYLES,
    apply_style,
    comma_axis_formatter,
    format_number,
    save_figure,
)

logger = get_logger(__name__)

DESIGN_CHAPTER = 4
SECTION_SELECTION = 1
SECTION_QAT_CHAIN = 2
SECTION_MEASUREMENT = 3
SECTION_COVERAGE = 5

#: Tipografía de los símbolos ✓ y ✕, que la serifa del proyecto no tiene; viene con Matplotlib.
SYMBOL_FONT = "DejaVu Sans"

#: Ancho de las figuras y cuerpo mínimo de letra.
DESIGN_FIGURE_WIDTH = 16.0 / 2.54
DESIGN_MIN_FONTSIZE = 7.0

#: Márgenes de los diagramas en fracción de la figura, para que el PNG no supere 16 cm.
DIAGRAM_MARGINS: dict[str, float] = {"left": 0.012, "right": 0.988, "bottom": 0.01, "top": 0.99}


class DesignFigureError(TFMEdgeError):
    """Error al construir una figura del capítulo 4."""


# Cadena de causas de la campaña QAT (apartado 4.2)


@dataclass(frozen=True)
class QatStage:
    """Etapa de la búsqueda de fallos de QAT: una campaña de GPU o la reproducción en local.

    ``campaign`` es None en la reproducción en local.
    """

    key: str
    label: str
    campaign: int | None


#: Etapas en orden; la reproducción en local va entre la primera y la segunda campaña.
QAT_STAGES: tuple[QatStage, ...] = (
    QatStage("campana_1", "Campaña 1", 1),
    QatStage("local", "Reproducción en local", None),
    QatStage("campana_2", "Campaña 2", 2),
    QatStage("campana_3", "Campaña 3", 3),
    QatStage("campana_4", "Campaña 4", 4),
)


@dataclass(frozen=True)
class QatCause:
    """Eslabón de la cadena de fallos de QAT.

    ``decision`` es la procedencia del dato y no se dibuja.
    """

    number: int
    label: str
    moment: str
    decision: str
    stage: str


#: Los seis eslabones en orden. La causa 1 apareció en la primera campaña, la 2 y la 3 al
#: reproducirla en local, y la 4, la 5 y la 6 en las campañas 2, 3 y 4.
QAT_CAUSE_CHAIN: tuple[QatCause, ...] = (
    QatCause(1, "Copia EMA\nprematura", "primer paso del\noptimizador", "D-086", "campana_1"),
    QatCause(2, "qconfig no\nserializable", "al guardar la\nprimera época", "D-086", "local"),
    QatCause(3, "Guardado desde\nmemoria", "guardado final\n(fallo latente)", "D-086", "local"),
    QatCause(4, "AMP\nincompatible", "primer paso en\nGPU con AMP", "D-087", "campana_2"),
    QatCause(5, "Observador\ndel DFL", "al arrancar la\nsegunda época", "D-088", "campana_3"),
    QatCause(6, "Errata de\nregistro", "tras 10 épocas,\nal inscribir", "D-089", "campana_4"),
)


def verify_cause_chain(chain: Sequence[QatCause], stages: Sequence[QatStage] = QAT_STAGES) -> None:
    """Comprueba que la cadena es coherente antes de dibujarla.

    Eslabones numerados desde 1, etapas que empiezan por una campaña, todo eslabón en una etapa,
    etapas que avanzan de una en una sin quedar vacías y etiquetas de tres palabras como máximo.
    """
    if [c.number for c in chain] != list(range(1, len(chain) + 1)):
        raise DesignFigureError(
            "Los eslabones de la cadena QAT no están numerados de forma consecutiva.",
            "Revisa QAT_CAUSE_CHAIN: el orden de la tupla es el orden del dibujo.",
        )
    claves = [e.key for e in stages]
    campanas = [e.campaign for e in stages if e.campaign is not None]
    if (
        not stages
        or stages[0].campaign is None
        or len(set(claves)) != len(claves)
        or campanas != list(range(1, len(campanas) + 1))
    ):
        raise DesignFigureError(
            f"Las etapas de la cadena QAT no están bien formadas: {claves}.",
            "Revisa QAT_STAGES: claves únicas, campañas numeradas 1, 2, 3... y la "
            "primera etapa una campaña; la reproducción en local sigue a una.",
        )
    sin_etapa = [c.number for c in chain if c.stage not in claves]
    if sin_etapa:
        raise DesignFigureError(
            f"Eslabones de la cadena QAT sin etapa declarada: {sin_etapa}.",
            "Cada eslabón tiene que caer en una campaña o en la reproducción en "
            "local de QAT_STAGES.",
        )
    posiciones = [claves.index(c.stage) for c in chain]
    if posiciones[0] != 0 or any(b - a not in (0, 1) for a, b in zip(posiciones, posiciones[1:])):
        raise DesignFigureError(
            f"Las etapas de la cadena QAT no avanzan de una en una: {[c.stage for c in chain]}.",
            "Cada eslabón tiene la etapa del anterior o la siguiente, empezando por la primera.",
        )
    vacias = [clave for clave in claves if clave not in {c.stage for c in chain}]
    if vacias:
        raise DesignFigureError(
            f"Etapas de la cadena QAT sin causas: {vacias}.",
            "Una etapa sin eslabones dibujaría una banda vacía; quítala de QAT_STAGES.",
        )
    largas = [c.label for c in chain if len(c.label.split()) > 3]
    if largas:
        raise DesignFigureError(
            f"Etiquetas de más de tres palabras: {largas}.",
            "El diagrama pide etiquetas breves; el detalle va en el texto del apartado.",
        )


def plot_qat_cause_chain(
    output_dir: Path | str,
    chain: Sequence[QatCause] = QAT_CAUSE_CHAIN,
    stages: Sequence[QatStage] = QAT_STAGES,
) -> dict[str, Path]:
    """Cadena de causas de QAT como línea horizontal, con bandas por etapa.

    Las campañas de GPU van en gris y la reproducción en local en blanco con borde discontinuo. No
    se dibuja la vuelta de la causa 5 a la 1, porque el texto presenta la cadena como lineal.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    verify_cause_chain(chain, stages)
    color_nodo = TECHNIQUE_STYLES["qat"].color
    n = len(chain)

    figura, ejes = plt.subplots(figsize=(DESIGN_FIGURE_WIDTH, 1.9))
    ejes.set_xlim(0, n)
    # Sin identificadores ni notas, el dibujo ocupa solo la banda superior.
    ejes.set_ylim(0.26, 1.0)
    ejes.axis("off")

    # 0,88 para que «Guardado desde» no toque los bordes del nodo.
    ancho_nodo, alto_nodo, y_nodo = 0.88, 0.26, 0.66
    # Bandas por etapa detrás de los nodos; la reproducción en local se distingue también sin color.
    for etapa in stages:
        indices = [i for i, c in enumerate(chain) if c.stage == etapa.key]
        izquierda, derecha = min(indices) + 0.05, max(indices) + 0.95
        es_local = etapa.campaign is None
        ejes.add_patch(
            Rectangle(
                (izquierda, 0.30),
                derecha - izquierda,
                0.58,
                facecolor="white" if es_local else "#EDEDED",
                edgecolor="#888888" if es_local else "none",
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                zorder=0,
            )
        )
        ejes.text(
            (izquierda + derecha) / 2,
            0.90,
            etapa.label,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    for i, causa in enumerate(chain):
        centro = i + 0.5
        ejes.add_patch(
            FancyBboxPatch(
                (centro - ancho_nodo / 2, y_nodo - alto_nodo / 2),
                ancho_nodo,
                alto_nodo,
                boxstyle="round,pad=0.0,rounding_size=0.04",
                facecolor=color_nodo,
                edgecolor="black",
                linewidth=0.6,
                zorder=2,
            )
        )
        ejes.text(
            centro,
            y_nodo,
            causa.label,
            ha="center",
            va="center",
            fontsize=DESIGN_MIN_FONTSIZE,
            zorder=3,
        )
        ejes.text(
            centro - ancho_nodo / 2 + 0.03,
            y_nodo + alto_nodo / 2 - 0.02,
            str(causa.number),
            ha="left",
            va="top",
            fontsize=DESIGN_MIN_FONTSIZE,
            fontweight="bold",
            zorder=3,
        )
        ejes.text(centro, 0.40, causa.moment, ha="center", va="center", fontsize=7, linespacing=1.2)
        if i + 1 < n:
            ejes.add_patch(
                FancyArrowPatch(
                    (centro + ancho_nodo / 2, y_nodo),
                    (centro + 1 - ancho_nodo / 2, y_nodo),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=0.9,
                    color="black",
                    zorder=4,
                )
            )

    figura.subplots_adjust(**DIAGRAM_MARGINS)
    return save_figure(figura, output_dir, DESIGN_CHAPTER, SECTION_QAT_CHAIN, "cadena_causas_qat")


# Las dos tomas de latencia (apartado 4.3.2)

#: Separación máxima entre las dos tomas, que deben ser dos procesos seguidos.
MAX_TAKE_GAP_S = 3600.0


@dataclass(frozen=True)
class LatencyTake:
    """Toma de latencia con su vector completo y lo que publica el volcado."""

    label: str
    source: Path
    started_at: datetime
    values_ms: tuple[float, ...]
    repetition_medians_ms: tuple[float, ...]
    mean_ms: float
    between_run_std_ms: float
    artifact_path: str
    backend: str
    precision: str
    device_id: str
    idle: bool


def load_latency_take(path: Path | str, label: str) -> LatencyTake:
    """Lee un volcado de latencia y comprueba que la media y la dispersión publicadas salen del
    vector.
    """
    ruta = Path(path)
    if not ruta.is_file():
        raise DesignFigureError(
            f"No existe el volcado de latencia '{ruta}'.",
            "Las dos tomas están en docs/evidencia/.",
        )
    datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
    repeticiones = datos.get("latency") or []
    vectores = [[t * 1000.0 for t in r.get("timings_s", [])] for r in repeticiones]
    if not vectores or any(not v for v in vectores):
        raise DesignFigureError(
            f"'{ruta.name}' no trae los vectores de latencia de cada repetición.",
            "Esta figura dibuja los valores, no el resumen: hace falta el volcado completo.",
        )
    valores = tuple(v for vector in vectores for v in vector)
    medianas = tuple(statistics.median(v) for v in vectores)

    media = float(datos["aggregated"]["latency_mean_ms"])
    if not math.isclose(media, statistics.fmean(valores), rel_tol=1e-9):
        raise DesignFigureError(
            f"'{ruta.name}': la media publicada ({media}) no es la de sus "
            f"{len(valores)} valores ({statistics.fmean(valores)}).",
            "Revisa si el protocolo descartó valores al agregar; la figura no "
            "puede marcar una media que no sale del vector que dibuja.",
        )
    dispersion = float(datos["between_run_std_ms"])
    recalculada = statistics.stdev(medianas) if len(medianas) > 1 else float("nan")
    if not math.isclose(dispersion, recalculada, abs_tol=1e-3):
        raise DesignFigureError(
            f"'{ruta.name}': la dispersión publicada ({dispersion}) no es la desviación típica "
            f"de las medianas por repetición ({recalculada:.4f}).",
            "La figura la presenta con esa definición; si el protocolo la cambió, "
            "hay que cambiar también el rótulo.",
        )

    carga = datos.get("system_load") or {}
    return LatencyTake(
        label=label,
        source=ruta,
        started_at=datetime.fromisoformat(datos["started_at"]),
        values_ms=valores,
        repetition_medians_ms=medianas,
        mean_ms=media,
        between_run_std_ms=dispersion,
        artifact_path=str((datos.get("artifact") or {}).get("path", "")),
        backend=str(datos.get("backend", "")),
        precision=str(datos.get("precision", "")),
        device_id=str(datos.get("device_id", "")),
        idle=bool(carga.get("acceptable")) and not bool(carga.get("forced")),
    )


def verify_comparable_takes(
    first: LatencyTake, second: LatencyTake, max_gap_s: float = MAX_TAKE_GAP_S
) -> None:
    """Comprueba que las dos tomas son del mismo artefacto, motor, precisión y perfil, en reposo y
    separadas por minutos.
    """
    distintos = [
        campo
        for campo in ("artifact_path", "backend", "precision", "device_id")
        if getattr(first, campo) != getattr(second, campo)
    ]
    if distintos:
        raise DesignFigureError(
            f"Las dos tomas no miden lo mismo: difieren en {', '.join(distintos)}.",
            "Las dos tomas tienen que ser del mismo artefacto y protocolo.",
        )
    separacion = (second.started_at - first.started_at).total_seconds()
    if not 0 < separacion <= max_gap_s:
        raise DesignFigureError(
            f"Las tomas están separadas por {separacion:.0f} s; se esperaban minutos.",
            "La segunda toma está en docs/evidencia/d095_toma_registrada_*.json.",
        )
    if not (first.idle and second.idle):
        raise DesignFigureError(
            "Alguna de las dos tomas no se hizo con la máquina en reposo.",
            "El pie de la figura lo afirma; sin reposo no puede afirmarlo.",
        )


def plot_latency_error_floor(
    first: LatencyTake, second: LatencyTake, output_dir: Path | str
) -> dict[str, Path]:
    """Dibuja las dos tomas de latencia: el vector completo y lo que publica el protocolo.

    A la izquierda, los 900 valores de cada toma y sus medias; a la derecha, la mediana de cada
    repetición y la dispersión publicada sobre la media. Hacen falta los dos paneles: la dispersión
    individual (9 a 12 ms) es mayor que la diferencia entre tomas.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    verify_comparable_takes(first, second)
    tomas = ((first, "#0072B2", "o"), (second, "#D55E00", "s"))
    delta = second.mean_ms - first.mean_ms
    relativo = 100.0 * delta / first.mean_ms

    figura, (izq, der) = plt.subplots(
        1, 2, figsize=(DESIGN_FIGURE_WIDTH, 3.0), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # Panel izquierdo: vectores completos
    todos = np.array(first.values_ms + second.values_ms)
    xmin = math.floor(float(todos.min()))
    xmax = math.ceil(float(np.percentile(todos, 99.5)))
    fuera = int((todos > xmax).sum())
    logger.info("%d de %d valores por encima de %d ms, fuera del eje", fuera, len(todos), xmax)
    bins = np.arange(xmin, xmax + 1.0, 1.0)
    for toma, color, _ in tomas:
        etiqueta = f"{toma.label} · media {format_number(toma.mean_ms, 2)} ms"
        izq.hist(
            toma.values_ms,
            bins=bins,
            histtype="stepfilled",
            alpha=0.45,
            color=color,
            label=etiqueta,
        )
        izq.hist(toma.values_ms, bins=bins, histtype="step", color=color, linewidth=0.8)
    tope = izq.get_ylim()[1]
    izq.set_ylim(0, tope * 1.25)
    # Las líneas de las medias terminan en la flecha del Δ para no cruzar el rótulo.
    for toma, color, _ in tomas:
        izq.vlines(toma.mean_ms, 0, tope * 1.06, color=color, linewidth=1.4)
    izq.annotate(
        "",
        xy=(second.mean_ms, tope * 1.06),
        xytext=(first.mean_ms, tope * 1.06),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
    )
    izq.text(
        (first.mean_ms + second.mean_ms) / 2,
        tope * 1.10,
        f"Δ {format_number(delta, 2)} ms ({format_number(relativo, 1)} %)",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    izq.set_xlim(xmin, xmax)
    izq.set_xlabel("Latencia (ms)")
    izq.set_ylabel("Frecuencia")
    izq.xaxis.set_major_formatter(comma_axis_formatter())
    izq.legend(loc="upper right", fontsize=DESIGN_MIN_FONTSIZE)
    izq.set_title(
        f"Las {len(first.values_ms)} latencias de cada toma", fontweight="bold", fontsize=9
    )

    # Panel derecho: lo que publica el protocolo
    for i, (toma, color, marcador) in enumerate(tomas):
        der.axhline(toma.mean_ms, color=color, linewidth=0.6, alpha=0.6, zorder=1)
        der.scatter(
            [i - 0.08, i, i + 0.08],
            toma.repetition_medians_ms,
            color=color,
            marker=marcador,
            s=22,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        der.errorbar(
            i + 0.24,
            toma.mean_ms,
            yerr=toma.between_run_std_ms,
            fmt="_",
            color=color,
            markersize=12,
            capsize=4,
            elinewidth=1.6,
            markeredgewidth=1.6,
            zorder=3,
        )
        # Rótulo encima de la barra para no caer sobre la línea de la media.
        der.text(
            i + 0.24,
            toma.mean_ms + toma.between_run_std_ms + 0.5,
            f"±{format_number(toma.between_run_std_ms, 2)} ms",
            ha="center",
            va="bottom",
            fontsize=DESIGN_MIN_FONTSIZE,
            zorder=4,
        )
    # Rótulo del Δ a la derecha, para no chocar con el «±» de la primera toma.
    der.annotate(
        "",
        xy=(0.62, second.mean_ms),
        xytext=(0.62, first.mean_ms),
        arrowprops={"arrowstyle": "<->", "linewidth": 0.9, "color": "black"},
    )
    der.text(
        0.66,
        (first.mean_ms + second.mean_ms) / 2,
        f"Δ {format_number(delta, 2)} ms",
        ha="left",
        va="center",
        fontsize=DESIGN_MIN_FONTSIZE,
    )
    der.set_xlim(-0.35, 1.55)
    # Margen sobre la barra más alta para su «±».
    techo = max(t.mean_ms + t.between_run_std_ms for t, _, _ in tomas)
    der.set_ylim(top=techo + 2.5)
    der.set_xticks([0, 1])
    der.set_xticklabels(
        [f"{t.label}\n{t.started_at:%H:%M:%S} UTC" for t, _, _ in tomas], fontsize=7.5
    )
    der.set_ylabel("Latencia (ms)")
    der.yaxis.set_major_formatter(comma_axis_formatter())
    der.grid(axis="x", visible=False)
    der.set_title("Lo que publica el protocolo", fontweight="bold", fontsize=9)

    figura.tight_layout()
    return save_figure(
        figura, output_dir, DESIGN_CHAPTER, SECTION_MEASUREMENT, "suelo_error_latencia"
    )


# Matriz de cobertura de perfiles y motores (apartado 4.5)

#: Filas y columnas de la matriz en orden de dibujo.
COVERAGE_PROFILES: tuple[str, ...] = (
    "workstation_cpu",
    "constrained_cpu",
    "workstation_gpu",
    "edge_cpu_arm",
    "edge_gpu_jetson",
    "cloud_gpu",
)
COVERAGE_ENGINES: tuple[str, ...] = ("pytorch", "onnxruntime", "openvino", "tensorrt", "tflite")

_SOLO_GPU_NVIDIA = "TensorRT solo corre sobre GPU NVIDIA"
_OPENVINO_INTEL = "OpenVINO es de Intel: no ejecuta sobre GPU NVIDIA"

#: Combinaciones imposibles por hardware; es lo único que se declara a mano. TFLite en
#: workstation_gpu correría en CPU. OpenVINO sí funciona en CPU ARM.
NOT_APPLICABLE: dict[tuple[str, str], str] = {
    ("workstation_cpu", "tensorrt"): _SOLO_GPU_NVIDIA,
    ("constrained_cpu", "tensorrt"): _SOLO_GPU_NVIDIA,
    ("edge_cpu_arm", "tensorrt"): _SOLO_GPU_NVIDIA,
    ("workstation_gpu", "openvino"): _OPENVINO_INTEL,
    ("edge_gpu_jetson", "openvino"): _OPENVINO_INTEL,
    ("workstation_gpu", "tflite"): "Sin camino CUDA en escritorio",
}

#: Celda enmarcada: donde se habría contrastado H1.
H1_CELL: tuple[str, str] = ("workstation_gpu", "tensorrt")


@dataclass(frozen=True)
class CoverageCategory:
    """Categoría de la matriz de cobertura, con color y símbolo."""

    key: str
    label: str
    color: str
    symbol: str
    symbol_color: str


#: Las cinco categorías en el orden de la leyenda.
COVERAGE_CATEGORIES: tuple[CoverageCategory, ...] = (
    CoverageCategory("medido", "Medido", "#009E73", "✓", "white"),
    CoverageCategory(
        "no_disponible", "No disponible, con motivo declarado", "#D55E00", "✕", "white"
    ),
    CoverageCategory("excluido", "Excluido por decisión metodológica", "#56B4E9", "—", "black"),
    CoverageCategory(
        "no_ejecutado", "No ejecutado, sin impedimento conocido", "#F0E442", "○", "black"
    ),
    CoverageCategory(
        "no_aplicable", "No aplicable: motor y hardware incompatibles", "#EEEEEE", "", "black"
    ),
)


@dataclass(frozen=True)
class CoverageCell:
    """Estado de una celda de la matriz."""

    category: str
    measured_rows: int = 0
    reason: str = ""


def coverage_matrix(
    deployment_rows: Sequence[Mapping[str, str]],
    profiles: Mapping[str, Any],
    grid_profiles: Sequence[str] = COVERAGE_PROFILES,
    grid_engines: Sequence[str] = COVERAGE_ENGINES,
    not_applicable: Mapping[tuple[str, str], str] = NOT_APPLICABLE,
) -> dict[tuple[str, str], CoverageCell]:
    """Clasifica cada celda perfil y motor a partir del registro y los perfiles.

    Por orden: medido (alguna fila ok), excluido (perfil no válido para medir), no aplicable
    (``not_applicable``), no disponible (fila no_disponible del perfil, del motor o del par) y no
    ejecutado. Falla si el registro tiene filas fuera de la rejilla.
    """
    faltan = [p for p in grid_profiles if p not in profiles]
    if faltan:
        raise DesignFigureError(
            f"Perfiles de la rejilla sin YAML en configs/benchmark: {', '.join(faltan)}.",
            "La categoría «excluido» se lee del perfil; sin él no se puede clasificar.",
        )

    medidas: dict[tuple[str, str], int] = {}
    ausencias: list[tuple[str, str, str]] = []
    fuera: list[str] = []
    for fila in deployment_rows:
        dispositivo, motor = fila.get("device_id", ""), fila.get("runtime", "")
        if fila.get("status") == "ok":
            if dispositivo not in grid_profiles or motor not in grid_engines:
                fuera.append(f"medida {dispositivo}/{motor}")
            medidas[(dispositivo, motor)] = medidas.get((dispositivo, motor), 0) + 1
        elif fila.get("status") == "no_disponible":
            if (dispositivo and dispositivo not in grid_profiles) or (
                motor and motor not in grid_engines
            ):
                fuera.append(f"no_disponible {dispositivo or '*'}/{motor or '*'}")
            ausencias.append((dispositivo, motor, fila.get("notes", "")))
    if fuera:
        raise DesignFigureError(
            "El registro tiene filas fuera de la rejilla de la figura: "
            f"{', '.join(sorted(set(fuera)))}.",
            "Amplía COVERAGE_PROFILES o COVERAGE_ENGINES: una figura de cobertura "
            "no puede dejar fuera algo que sí se midió o se declaró.",
        )

    celdas: dict[tuple[str, str], CoverageCell] = {}
    for perfil in grid_profiles:
        for motor in grid_engines:
            par = (perfil, motor)
            if medidas.get(par):
                celdas[par] = CoverageCell("medido", measured_rows=medidas[par])
            elif not getattr(profiles[perfil], "valid_for_measurement", True):
                celdas[par] = CoverageCell(
                    "excluido", reason="perfil no válido para medir latencia"
                )
            elif par in not_applicable:
                celdas[par] = CoverageCell("no_aplicable", reason=not_applicable[par])
            else:
                motivo = next(
                    (
                        nota
                        for disp, mot, nota in ausencias
                        if (disp == perfil and mot in ("", motor)) or (disp == "" and mot == motor)
                    ),
                    None,
                )
                celdas[par] = (
                    CoverageCell("no_disponible", reason=motivo)
                    if motivo is not None
                    else CoverageCell("no_ejecutado")
                )
    return celdas


def plot_coverage_matrix(
    cells: Mapping[tuple[str, str], CoverageCell],
    output_dir: Path | str,
    grid_profiles: Sequence[str] = COVERAGE_PROFILES,
    grid_engines: Sequence[str] = COVERAGE_ENGINES,
) -> dict[str, Path]:
    """Dibuja la matriz de cobertura con color, símbolo y número de filas medidas."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    por_clave = {c.key: c for c in COVERAGE_CATEGORIES}
    figura = plt.figure(figsize=(DESIGN_FIGURE_WIDTH, 3.2))
    rejilla = figura.add_gridspec(
        1, 2, width_ratios=[1.3, 1.0], wspace=0.05, left=0.17, right=0.99, bottom=0.17, top=0.99
    )
    ejes = figura.add_subplot(rejilla[0, 0])
    leyenda = figura.add_subplot(rejilla[0, 1])

    for fila, perfil in enumerate(grid_profiles):
        for columna, motor in enumerate(grid_engines):
            celda = cells[(perfil, motor)]
            categoria = por_clave[celda.category]
            ejes.add_patch(
                Rectangle(
                    (columna, fila),
                    1,
                    1,
                    facecolor=categoria.color,
                    edgecolor="white",
                    linewidth=2.0,
                )
            )
            if categoria.symbol:
                # En las medidas el símbolo sube para dejar sitio al recuento; en la de H1 baja.
                desplazamiento = (
                    -0.12 if celda.measured_rows else 0.1 if (perfil, motor) == H1_CELL else 0.0
                )
                ejes.text(
                    columna + 0.5,
                    fila + 0.5 + desplazamiento,
                    categoria.symbol,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontfamily=SYMBOL_FONT,
                    color=categoria.symbol_color,
                    fontweight="bold",
                )
            if celda.measured_rows:
                ejes.text(
                    columna + 0.5,
                    fila + 0.75,
                    f"{celda.measured_rows} filas",
                    ha="center",
                    va="center",
                    fontsize=DESIGN_MIN_FONTSIZE,
                    color=categoria.symbol_color,
                )

    fila_h1 = list(grid_profiles).index(H1_CELL[0])
    columna_h1 = list(grid_engines).index(H1_CELL[1])
    ejes.add_patch(
        Rectangle(
            (columna_h1 + 0.04, fila_h1 + 0.04),
            0.92,
            0.92,
            fill=False,
            edgecolor="black",
            linewidth=2.2,
            zorder=5,
        )
    )
    # Dentro del marco; el ✕ de esta celda baja para no tocarlo.
    ejes.text(
        columna_h1 + 0.12,
        fila_h1 + 0.12,
        "H1",
        ha="left",
        va="top",
        fontsize=DESIGN_MIN_FONTSIZE,
        fontweight="bold",
        color="white",
        zorder=6,
    )

    ejes.set_xlim(0, len(grid_engines))
    ejes.set_ylim(len(grid_profiles), 0)
    ejes.set_xticks([i + 0.5 for i in range(len(grid_engines))])
    # rotation_mode="anchor" gira el texto sobre su anclaje y lo deja bajo su columna.
    ejes.set_xticklabels(grid_engines, rotation=30, ha="right", rotation_mode="anchor")
    ejes.set_yticks([i + 0.5 for i in range(len(grid_profiles))])
    ejes.set_yticklabels(grid_profiles)
    ejes.tick_params(length=0)
    ejes.grid(False)
    for lado in ejes.spines.values():
        lado.set_visible(False)
    ejes.set_aspect("equal")

    # Leyenda dibujada a mano para que los símbolos usen la tipografía que los tiene.
    leyenda.axis("off")
    leyenda.set_xlim(0, 1)
    leyenda.set_ylim(0, 1)
    paso = 0.13
    for i, categoria in enumerate(COVERAGE_CATEGORIES):
        y = 0.86 - i * paso
        leyenda.add_patch(
            Rectangle(
                (0.02, y - 0.045),
                0.09,
                0.09,
                facecolor=categoria.color,
                edgecolor="#777777",
                linewidth=0.5,
            )
        )
        if categoria.symbol:
            leyenda.text(
                0.065,
                y,
                categoria.symbol,
                ha="center",
                va="center",
                fontsize=9,
                fontfamily=SYMBOL_FONT,
                color=categoria.symbol_color,
                fontweight="bold",
            )
        # En dos líneas para que el rótulo más largo quepa.
        leyenda.text(
            0.15,
            y,
            textwrap.fill(categoria.label, 24),
            ha="left",
            va="center",
            fontsize=7,
            linespacing=1.2,
        )
    y = 0.86 - len(COVERAGE_CATEGORIES) * paso
    leyenda.add_patch(
        Rectangle((0.02, y - 0.045), 0.09, 0.09, fill=False, edgecolor="black", linewidth=2.0)
    )
    leyenda.text(
        0.15,
        y,
        "Celda donde se habría\ncontrastado H1",
        ha="left",
        va="center",
        fontsize=7,
        linespacing=1.2,
    )
    return save_figure(
        figura, output_dir, DESIGN_CHAPTER, SECTION_COVERAGE, "cobertura_perfiles_motores"
    )


# Sesgo de selección de best.pt (apartado 4.1)

#: Columna de results.csv que Ultralytics maximiza al guardar best.pt; en 8.3.253 es el mAP@50-95 de
#: su validador.
ULTRALYTICS_MAP_COLUMN = "metrics/mAP50-95(B)"

#: Primera época del panel ampliado, ya en la meseta.
SELECTION_ZOOM_FROM = 36


@dataclass(frozen=True)
class SelectionArm:
    """Brazo de la extensión a 50 épocas: control o variante."""

    key: str
    label: str
    color: str
    marker: str


#: El control en negro y la variante con el color de la destilación.
SELECTION_ARMS: tuple[SelectionArm, ...] = (
    SelectionArm("control", "Control (α = 0)", TECHNIQUE_STYLES["baseline"].color, "o"),
    SelectionArm("variante", "kd_combined", TECHNIQUE_STYLES["distillation"].color, "v"),
)


@dataclass(frozen=True)
class SelectionArmData:
    """Curva de un brazo y sus evaluaciones con pycocotools."""

    arm: SelectionArm
    experiment_id: str
    epochs: tuple[int, ...]
    ultralytics: tuple[float, ...]
    best_epoch: int
    last_epoch: int
    best_pycocotools: float
    last_pycocotools: float

    def ultralytics_at(self, epoch: int) -> float:
        """mAP@50-95 de Ultralytics en una época."""
        return self.ultralytics[self.epochs.index(epoch)]

    @property
    def selection_premium(self) -> float:
        """Prima de selección: best.pt menos la última época, con pycocotools."""
        return self.best_pycocotools - self.last_pycocotools


def selection_curve_path(evidence_dir: Path | str, experiment_id: str) -> Path:
    """Ruta del results.csv extraído de un experimento; el nombre lleva el experiment_id."""
    return Path(evidence_dir) / f"d171_results_{experiment_id}.csv"


def load_training_curve(path: Path | str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Lee la curva de mAP@50-95 por época de un results.csv de Ultralytics.

    Recorta los espacios de las cabeceras y exige todas las épocas consecutivas.
    """
    ruta = Path(path)
    if not ruta.is_file():
        raise DesignFigureError(
            f"No existe la curva de entrenamiento '{ruta}'.",
            "Las curvas están en docs/evidencia/d171_results_*.csv, extraídas de "
            "models/distillation/.",
        )
    with ruta.open(encoding="utf-8", newline="") as fichero:
        filas = [{k.strip(): v.strip() for k, v in f.items()} for f in csv.DictReader(fichero)]
    if not filas or ULTRALYTICS_MAP_COLUMN not in filas[0] or "epoch" not in filas[0]:
        raise DesignFigureError(
            f"'{ruta.name}' no trae las columnas 'epoch' y '{ULTRALYTICS_MAP_COLUMN}'.",
            "¿Es el results.csv de Ultralytics, o ha cambiado su formato?",
        )
    epocas = tuple(int(float(f["epoch"])) for f in filas)
    if epocas != tuple(range(1, len(epocas) + 1)):
        raise DesignFigureError(
            f"'{ruta.name}': las épocas no son consecutivas desde 1.",
            "La figura necesita la curva completa; una reanudación puede dejar "
            "épocas repetidas o ausentes.",
        )
    return epocas, tuple(float(f[ULTRALYTICS_MAP_COLUMN]) for f in filas)


def load_selection_bias(
    evidence_json: Path | str, evidence_dir: Path | str
) -> tuple[SelectionArmData, ...]:
    """Lee las medidas y las curvas de los dos brazos y comprueba que corresponden al mismo
    entrenamiento.

    Comprueba que el evaluador es pycocotools, que best.pt es el máximo único de la curva en la
    época indicada, que la última medida es la última época y que el mAP de Ultralytics anotado
    coincide con la curva.
    """
    ruta = Path(evidence_json)
    if not ruta.is_file():
        raise DesignFigureError(
            f"No existe el fichero de evidencia '{ruta}'.",
            "Está en docs/evidencia/d171_sesgo_de_seleccion.json.",
        )
    datos: dict[str, Any] = json.loads(ruta.read_text(encoding="utf-8"))
    evaluador = (datos.get("protocolo") or {}).get("evaluador")
    if datos.get("decision") != "D-171" or evaluador != "pycocotools":
        raise DesignFigureError(
            f"'{ruta.name}' no es la evaluación con pycocotools esperada "
            f"(decisión {datos.get('decision')!r}, evaluador {evaluador!r}).",
            "La figura rotula las cifras como pycocotools; tienen que serlo.",
        )
    medidas: dict[str, Any] = datos.get("medidas") or {}

    brazos: list[SelectionArmData] = []
    for brazo in SELECTION_ARMS:
        mejor, ultima = medidas.get(f"{brazo.key}_best"), medidas.get(f"{brazo.key}_last")
        if not mejor or not ultima or mejor["experiment_id"] != ultima["experiment_id"]:
            raise DesignFigureError(
                f"'{ruta.name}' no trae las dos medidas del brazo '{brazo.key}' "
                "sobre un mismo experimento.",
                "Cada brazo necesita su best.pt y su última época.",
            )
        experimento = mejor["experiment_id"]
        epocas, valores = load_training_curve(selection_curve_path(evidence_dir, experimento))

        maximo = max(valores)
        elegidas = [e for e, v in zip(epocas, valores) if v == maximo]
        if elegidas != [int(mejor["epoca"])]:
            raise DesignFigureError(
                f"{experimento}: la curva tiene su máximo en la época {elegidas} y el fichero "
                f"sitúa best.pt en la {mejor['epoca']}.",
                "O la curva no es la de ese entrenamiento, o hay un empate que "
                "Ultralytics resolvió de un modo que la figura no puede reproducir.",
            )
        if int(ultima["epoca"]) != epocas[-1]:
            raise DesignFigureError(
                f"{experimento}: la última medida es de la época {ultima['epoca']} y "
                f"la curva acaba en la {epocas[-1]}.",
                "La comparación es contra la última época del entrenamiento.",
            )
        for medida in (mejor, ultima):
            en_curva = valores[epocas.index(int(medida["epoca"]))]
            if en_curva != float(medida["map_50_95_ultralytics"]):
                raise DesignFigureError(
                    f"{experimento}, época {medida['epoca']}: el fichero anota "
                    f"{medida['map_50_95_ultralytics']} de Ultralytics y la curva dice "
                    f"{en_curva}.",
                    "Las dos cifras salen del mismo validador; si no coinciden, "
                    "la curva no es la del entrenamiento medido.",
                )
        brazos.append(
            SelectionArmData(
                arm=brazo,
                experiment_id=experimento,
                epochs=epocas,
                ultralytics=valores,
                best_epoch=int(mejor["epoca"]),
                last_epoch=int(ultima["epoca"]),
                best_pycocotools=float(mejor["map_50_95_pycocotools"]),
                last_pycocotools=float(ultima["map_50_95_pycocotools"]),
            )
        )
    return tuple(brazos)


def _signed(value: float, decimals: int) -> str:
    """Número con signo explícito, coma decimal y signo menos tipográfico."""
    texto = format_number(abs(value), decimals)
    return f"+{texto}" if value >= 0 else f"−{texto}"


def plot_selection_bias(
    arms: Sequence[SelectionArmData],
    output_dir: Path | str,
    zoom_from: int = SELECTION_ZOOM_FROM,
) -> dict[str, Path]:
    """Sesgo de selección de best.pt en la extensión a 50 épocas.

    A la izquierda, las curvas del validador de Ultralytics; a la derecha, la meseta ampliada con
    best.pt (estrella) y lo que da pycocotools en ese checkpoint y en la última época. No hay curva
    de pycocotools por época.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figura, (completo, meseta) = plt.subplots(
        1, 2, figsize=(DESIGN_FIGURE_WIDTH, 3.6), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    for brazo in arms:
        estilo = brazo.arm
        for ejes, tam in ((completo, 2.2), (meseta, 3.5)):
            ejes.plot(
                brazo.epochs,
                brazo.ultralytics,
                color=estilo.color,
                marker=estilo.marker,
                markersize=tam,
                linewidth=1.0,
                label=estilo.label,
                zorder=2,
            )
            ejes.scatter(
                [brazo.best_epoch],
                [brazo.ultralytics_at(brazo.best_epoch)],
                marker="*",
                s=110 if ejes is meseta else 80,
                color=estilo.color,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )

        # pycocotools en best.pt (círculo) y en la última época (cuadrado), unidos a la curva.
        for epoca, canonico, marcador in (
            (brazo.best_epoch, brazo.best_pycocotools, "o"),
            (brazo.last_epoch, brazo.last_pycocotools, "s"),
        ):
            meseta.vlines(
                epoca,
                canonico,
                brazo.ultralytics_at(epoca),
                color=estilo.color,
                linewidth=0.9,
                linestyle=(0, (2, 1.5)),
                zorder=1,
            )
            meseta.scatter(
                [epoca],
                [canonico],
                marker=marcador,
                s=26,
                facecolor="white",
                edgecolor=estilo.color,
                linewidth=1.2,
                zorder=3,
            )

    # Posición de los rótulos del desfase, elegida para no pisar las curvas.
    for brazo in arms:
        desfase = brazo.ultralytics_at(brazo.best_epoch) - brazo.best_pycocotools
        medio = (brazo.ultralytics_at(brazo.best_epoch) + brazo.best_pycocotools) / 2
        if brazo.arm.key == "variante":
            x, y, ha = brazo.best_epoch + 0.35, medio, "left"
        else:
            x, y, ha = brazo.best_epoch - 0.35, brazo.best_pycocotools, "right"
        meseta.text(
            x,
            y,
            f"desfase en best.pt\n{_signed(desfase, 4)}",
            ha=ha,
            va="center",
            fontsize=DESIGN_MIN_FONTSIZE,
            color=brazo.arm.color,
            linespacing=1.15,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
            zorder=5,
        )

    todos = [v for b in arms for v in b.ultralytics]
    completo.set_ylim(math.floor(min(todos) * 100) / 100, math.ceil(max(todos) * 100) / 100 + 0.005)
    completo.axvspan(zoom_from - 0.5, max(arms[0].epochs) + 0.5, color="#EDEDED", zorder=0)
    completo.set_xlim(0, max(arms[0].epochs) + 1)
    completo.set_xlabel("Época")
    completo.set_ylabel("mAP@50-95")
    completo.yaxis.set_major_formatter(comma_axis_formatter())
    completo.set_title("Validador de Ultralytics, por época", fontweight="bold", fontsize=9)
    marcas = [
        Line2D([], [], color=b.arm.color, marker=b.arm.marker, markersize=3.5, label=b.arm.label)
        for b in arms
    ] + [
        Line2D(
            [],
            [],
            color="none",
            marker="*",
            markersize=9,
            markerfacecolor="#999999",
            markeredgecolor="black",
            label="best.pt (máximo de la curva)",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="pycocotools en best.pt",
        ),
        Line2D(
            [],
            [],
            color="none",
            marker="s",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor="#555555",
            label="pycocotools en la época 50",
        ),
    ]
    completo.set_xticks(range(0, max(arms[0].epochs) + 1, 10))
    # Leyenda debajo de los dos paneles, en dos filas.
    figura.legend(
        handles=marcas,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        fontsize=DESIGN_MIN_FONTSIZE,
        frameon=False,
        columnspacing=1.5,
    )

    ultima = max(arms[0].epochs)
    en_meseta = (
        [v for b in arms for e, v in zip(b.epochs, b.ultralytics) if e >= zoom_from]
        + [b.best_pycocotools for b in arms]
        + [b.last_pycocotools for b in arms]
    )
    meseta.set_xlim(zoom_from - 0.5, ultima + 0.9)
    # Marcas en épocas enteras.
    meseta.set_xticks(range(zoom_from, ultima + 1, 2))
    meseta.set_ylim(min(en_meseta) - 0.0008, max(en_meseta) + 0.0012)
    meseta.set_xlabel("Época")
    meseta.yaxis.set_major_formatter(comma_axis_formatter())
    meseta.set_title(
        f"Épocas {zoom_from} a {ultima}: dos evaluadores", fontweight="bold", fontsize=9
    )

    figura.tight_layout(rect=(0, 0.13, 1, 1))
    return save_figure(
        figura, output_dir, DESIGN_CHAPTER, SECTION_SELECTION, "sesgo_seleccion_best_pt"
    )


# Flujo experimental (apartado 4.3)

#: Pasos de scripts/ que aparecen en el diagrama y lo que hace cada uno.
FLOW_STEPS: dict[str, str] = {
    "04": "entrenamiento",
    "05": "evaluación del baseline e inscripción",
    "06": "evaluación COCO del checkpoint",
    "07": "medida de despliegue",
    "08": "exportación FP32",
    "10": "PTQ",
    "11": "evaluación de calidad del artefacto",
    "13": "poda",
    "16": "QAT",
    "18": "INT8 de la rama QAT",
    "20": "destilación",
    "22": "combinaciones",
}

#: Qué paso escribe en qué tabla del registro; se contrasta con el código.
FLOW_REGISTRY_WRITES: dict[str, tuple[str, ...]] = {
    "experiments": ("05", "06", "10", "13", "16", "20", "22"),
    "quality_metrics": ("11",),
    "deployment_metrics": ("07",),
}

#: Método del registro que escribe cada tabla.
_APPEND_METHOD = {
    "experiments": "append_experiment(",
    "quality_metrics": "append_quality_metrics(",
    "deployment_metrics": "append_deployment_metrics(",
}

#: Colores del diagrama: un tono claro por zona y uno medio para sus cajas.
FLOW_ZONES: dict[str, tuple[str, str]] = {
    "entrenamiento": ("#DCEAF6", "#9DC3E6"),
    "exportacion": ("#FBE3D0", "#F4B183"),
    "medida": ("#DDF1E7", "#9FD5B8"),
    "registro": ("#EDEDED", "#D4D4D4"),
}


def verify_flow_against_scripts(scripts_dir: Path | str) -> dict[str, Path]:
    """Comprueba que cada paso citado es un único fichero NN_*.py y que escribe en la tabla que
    indica el diagrama.
    """
    from tfm_edge.common.registry import TABLE_FILENAMES

    directorio = Path(scripts_dir)
    ficheros: dict[str, Path] = {}
    for paso in FLOW_STEPS:
        candidatos = sorted(directorio.glob(f"{paso}_*.py"))
        if len(candidatos) != 1:
            raise DesignFigureError(
                f"El paso {paso} del diagrama corresponde a {len(candidatos)} ficheros "
                f"en {directorio}.",
                "Revisa FLOW_STEPS: el diagrama cita los pasos por su número.",
            )
        ficheros[paso] = candidatos[0]

    fallos = []
    for tabla, pasos in FLOW_REGISTRY_WRITES.items():
        if tabla not in TABLE_FILENAMES:
            fallos.append(f"la tabla '{tabla}' no existe en el registro")
            continue
        for paso in pasos:
            if paso not in ficheros:
                fallos.append(f"el paso {paso} no está en FLOW_STEPS")
            elif _APPEND_METHOD[tabla] not in ficheros[paso].read_text(encoding="utf-8"):
                fallos.append(f"{ficheros[paso].name} no escribe en {tabla}")
    if fallos:
        raise DesignFigureError(
            f"El diagrama de flujo no cuadra con el código: {'; '.join(fallos)}.",
            "Corrige FLOW_REGISTRY_WRITES o el dibujo: el diagrama no puede atribuir "
            "una escritura a un paso que no la hace.",
        )
    return ficheros


def _flow_box(
    axes: Any,
    cx: float,
    cy: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
    fontsize: float = 7.0,
) -> None:
    """Caja del diagrama: título en negrita y detalle debajo."""
    from matplotlib.patches import FancyBboxPatch

    axes.add_patch(
        FancyBboxPatch(
            (cx - width / 2, cy - height / 2),
            width,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=2,
        )
    )
    if detail:
        lineas = detail.count("\n") + 1
        axes.text(
            cx,
            cy + 0.12 * lineas,
            title,
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            zorder=3,
        )
        axes.text(
            cx,
            cy - 0.12,
            detail,
            ha="center",
            va="center",
            fontsize=DESIGN_MIN_FONTSIZE,
            linespacing=1.2,
            zorder=3,
        )
    else:
        axes.text(cx, cy, title, ha="center", va="center", fontsize=fontsize, zorder=3)


def _flow_arrow(
    axes: Any,
    start: tuple[float, float],
    end: tuple[float, float],
    head: bool = True,
    **kwargs: Any,
) -> None:
    """Flecha del diagrama, o segmento sin punta."""
    from matplotlib.patches import FancyArrowPatch

    opciones = {"linewidth": 1.0, "color": "black"}
    opciones.update(kwargs)
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>" if head else "-",
            mutation_scale=8,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
            **opciones,
        )
    )


def plot_experimental_flow(output_dir: Path | str, scripts_dir: Path | str) -> dict[str, Path]:
    """Flujo experimental del entrenamiento al registro, en cuatro zonas.

    Entrenamiento, exportación, medida y registro. Calidad y despliegue se miden sobre el mismo
    artefacto exportado. El texto no incluye números de paso ni nombres de tabla; la correspondencia
    con el código se comprueba con FLOW_STEPS y FLOW_REGISTRY_WRITES.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    verify_flow_against_scripts(scripts_dir)
    figura, ejes = plt.subplots(figsize=(DESIGN_FIGURE_WIDTH, 3.6))
    ejes.set_xlim(0, 12)
    ejes.set_ylim(0, 5.75)
    ejes.axis("off")

    # Zonas. Entrenamiento y exportación necesitan más ancho que el registro.
    zonas = (
        ("entrenamiento", 0.05, 3.8, "ENTRENAMIENTO"),
        ("exportacion", 3.9, 6.75, "EXPORTACIÓN"),
        ("medida", 6.85, 10.0, "MEDIDA"),
        ("registro", 10.1, 11.95, "REGISTRO"),
    )
    y0, y1 = 0.85, 5.7
    for clave, izq, der, nombre in zonas:
        ejes.add_patch(
            Rectangle(
                (izq, y0),
                der - izq,
                y1 - y0,
                facecolor=FLOW_ZONES[clave][0],
                edgecolor="none",
                zorder=0,
            )
        )
        centro = (izq + der) / 2
        ejes.text(centro, 5.4, nombre, ha="center", va="center", fontsize=8, fontweight="bold")
    # La frontera se marca también con dos trazos gruesos.
    frontera_izq, frontera_der = zonas[1][1], zonas[1][2]
    for x in (frontera_izq, frontera_der):
        ejes.plot([x, x], [y0, y1], color="#C55A11", linewidth=1.8, zorder=1)

    caja_a, caja_b, caja_c, caja_d = (
        FLOW_ZONES[k][1] for k in ("entrenamiento", "exportacion", "medida", "registro")
    )

    # Entrenamiento
    a_cx, a_ancho = 1.925, 3.45
    _flow_box(ejes, a_cx, 4.3, a_ancho, 0.95, "Entrenamiento", "modelo entrenado en FP32", caja_a)
    _flow_box(
        ejes,
        a_cx,
        2.45,
        a_ancho,
        1.2,
        "Compresión, según la rama",
        "poda - cuantización durante\nel entrenamiento - destilación",
        caja_a,
    )
    _flow_arrow(ejes, (a_cx, 3.825), (a_cx, 3.05))

    # Exportación
    b_cx, b_ancho = 5.325, 2.77
    _flow_box(
        ejes,
        b_cx,
        4.3,
        b_ancho,
        1.05,
        "Exportación",
        "conversión al formato\nde despliegue, en FP32",
        caja_b,
    )
    _flow_box(
        ejes,
        b_cx,
        2.45,
        b_ancho,
        1.2,
        "Cuantización a INT8",
        "las dos ramas recorren\nel mismo camino",
        caja_b,
    )
    a_der, b_izq, b_der = a_cx + a_ancho / 2, b_cx - b_ancho / 2, b_cx + b_ancho / 2
    _flow_arrow(ejes, (a_der, 4.3), (b_izq, 4.3))
    _flow_arrow(ejes, (a_der, 2.75), (b_izq, 3.95))
    _flow_arrow(ejes, (b_cx, 3.775), (b_cx, 3.05))
    ejes.text(
        b_cx + 0.13,
        3.41,
        "donde\naplique",
        ha="left",
        va="center",
        fontsize=DESIGN_MIN_FONTSIZE,
        style="italic",
        color="#444444",
        linespacing=1.1,
    )

    # El mismo artefacto llega a las dos medidas. FP32 e INT8 van dentro de la zona de medida.
    medida_izq = zonas[2][1]
    bus_x = medida_izq + 0.65
    c_izq = bus_x + 0.38
    ejes.plot(
        [bus_x, bus_x], [2.2, 4.55], color="#333333", linewidth=2.2, zorder=3, solid_capstyle="butt"
    )
    _flow_arrow(ejes, (b_der, 4.3), (bus_x, 4.3), head=False)
    _flow_arrow(ejes, (b_der, 2.45), (bus_x, 2.45), head=False)
    x_rotulo = (medida_izq + bus_x) / 2
    ejes.text(x_rotulo, 4.4, "FP32", ha="center", va="bottom", fontsize=DESIGN_MIN_FONTSIZE)
    ejes.text(x_rotulo, 2.55, "INT8", ha="center", va="bottom", fontsize=DESIGN_MIN_FONTSIZE)
    ejes.text(
        bus_x + 0.19,
        3.375,
        "el mismo modelo exportado",
        rotation=90,
        ha="center",
        va="center",
        fontsize=DESIGN_MIN_FONTSIZE,
        color="#333333",
    )
    _flow_arrow(ejes, (bus_x, 4.55), (c_izq, 4.55))
    _flow_arrow(ejes, (bus_x, 2.2), (c_izq, 2.2))

    # Medida
    c_der = 9.92
    ancho_c = c_der - c_izq
    _flow_box(
        ejes,
        c_izq + ancho_c / 2,
        4.55,
        ancho_c,
        1.1,
        "Calidad",
        "evaluación de la\nprecisión media\n(mAP)",
        caja_c,
    )
    _flow_box(
        ejes,
        c_izq + ancho_c / 2,
        2.2,
        ancho_c,
        1.1,
        "Despliegue",
        "latencia, memoria\ny tamaño, en el\nequipo de medida",
        caja_c,
    )

    # Registro. Nombres en dos líneas para que quepan a 7 pt.
    d_cx, d_ancho, d_alto = 11.025, 1.7, 0.62
    for y, nombre in (
        (4.55, "registro de\ncalidad"),
        (2.2, "registro de\ndespliegue"),
        (1.3, "registro de\nexperimentos"),
    ):
        _flow_box(ejes, d_cx, y, d_ancho, d_alto, nombre, "", caja_d, fontsize=DESIGN_MIN_FONTSIZE)
    _flow_arrow(ejes, (c_der, 4.55), (d_cx - d_ancho / 2, 4.55))
    _flow_arrow(ejes, (c_der, 2.2), (d_cx - d_ancho / 2, 2.2))

    # Procedencia: la inscriben los pasos que producen pesos.
    gris = "#555555"
    trazo = (0, (3, 2))
    # Sale de las dos cajas: el baseline también se inscribe (05, 06).
    a_izq, x_procedencia = a_cx - a_ancho / 2, 0.11
    for y in (4.3, 2.45):
        ejes.plot(
            [a_izq, x_procedencia], [y, y], color=gris, linewidth=0.9, linestyle=trazo, zorder=1
        )
    ejes.plot(
        [x_procedencia, x_procedencia, d_cx],
        [4.3, 0.5, 0.5],
        color=gris,
        linewidth=0.9,
        linestyle=trazo,
        zorder=1,
    )
    _flow_arrow(
        ejes, (d_cx, 0.5), (d_cx, 1.3 - d_alto / 2), color=gris, linewidth=0.9, linestyle=trazo
    )
    ejes.text(
        5.6,
        0.34,
        "procedencia: qué configuración, de qué modelo parte y con qué pesos",
        ha="center",
        va="top",
        fontsize=DESIGN_MIN_FONTSIZE,
        color=gris,
    )

    figura.subplots_adjust(**DIAGRAM_MARGINS)
    return save_figure(
        figura, output_dir, DESIGN_CHAPTER, SECTION_MEASUREMENT, "flujo_experimental"
    )


def generate_design_figures(
    output_dir: Path | str,
    deployment_rows: Sequence[Mapping[str, str]],
    profiles: Mapping[str, Any],
    first_take: Path | str,
    second_take: Path | str,
    selection_evidence: Path | str,
    scripts_dir: Path | str,
) -> dict[str, dict[str, Path]]:
    """Genera las cinco figuras del capítulo 4.

    Todas las comprobaciones se hacen antes de dibujar la primera, para no dejar un lote a medias.
    """
    apply_style()
    primera = load_latency_take(first_take, "1.ª toma")
    segunda = load_latency_take(second_take, "2.ª toma")
    verify_comparable_takes(primera, segunda)
    celdas = coverage_matrix(deployment_rows, profiles)
    brazos = load_selection_bias(selection_evidence, Path(selection_evidence).parent)
    verify_flow_against_scripts(scripts_dir)
    verify_cause_chain(QAT_CAUSE_CHAIN)
    return {
        "sesgo_seleccion_best_pt": plot_selection_bias(brazos, output_dir),
        "cadena_causas_qat": plot_qat_cause_chain(output_dir),
        "flujo_experimental": plot_experimental_flow(output_dir, scripts_dir),
        "suelo_error_latencia": plot_latency_error_floor(primera, segunda, output_dir),
        "cobertura_perfiles_motores": plot_coverage_matrix(celdas, output_dir),
    }
