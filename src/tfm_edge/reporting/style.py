"""Estilo común de las figuras.

Cada familia de técnica tiene color, marcador y estilo de línea fijos en todas las figuras, para que
se lean también en escala de grises. Se exporta a PDF vectorial y PNG a 300 ppp. Los nombres siguen
``fig_{capitulo}_{apartado}_{descripcion}``: el prefijo indica capítulo y apartado, no número de
figura.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir, relative_to_root

logger = get_logger(__name__)

FIGURE_DPI = 300

FIGURE_FORMATS = ("pdf", "png")

#: Fecha fija (2026-01-01T00:00:00Z) para que PDF y XLSX salgan idénticos en dos ejecuciones
#: iguales.
FIXED_TIMESTAMP_EPOCH = 1767225600


def freeze_creation_date() -> None:
    """Fija la fecha de creación de los PDF de Matplotlib mediante SOURCE_DATE_EPOCH.

    Hay que llamarla antes de importar Matplotlib. No sobrescribe un valor que ya traiga el entorno.
    """
    import os

    os.environ.setdefault("SOURCE_DATE_EPOCH", str(FIXED_TIMESTAMP_EPOCH))


@dataclass(frozen=True)
class FamilyStyle:
    """Estilo visual de una familia de técnica."""

    color: str
    marker: str
    linestyle: str
    label: str


#: Paleta apta para daltonismo (basada en Okabe-Ito), con marcador y estilo de línea por familia.
TECHNIQUE_STYLES: dict[str, FamilyStyle] = {
    "baseline": FamilyStyle("#000000", "o", "-", "Referencia"),
    "ptq": FamilyStyle("#0072B2", "s", "--", "Cuantización PTQ"),
    "qat": FamilyStyle("#56B4E9", "D", "-.", "Cuantización QAT"),
    "pruning": FamilyStyle("#D55E00", "^", ":", "Poda estructurada"),
    "distillation": FamilyStyle("#009E73", "v", (0, (3, 1, 1, 1)), "Destilación"),
    "combined": FamilyStyle("#CC79A7", "P", (0, (5, 1)), "Combinaciones"),
}

#: QAT sobre un modelo podado. Se distingue del QAT sobre baseline por el marcador; no está en
#: TECHNIQUE_STYLES porque no es una técnica del registro.
PRUNING_QAT_STYLE = FamilyStyle(
    TECHNIQUE_STYLES["qat"].color, "X", TECHNIQUE_STYLES["qat"].linestyle, "Poda + QAT"
)

#: Estilo por rango de tamaño de objeto.
SIZE_STYLES: dict[str, FamilyStyle] = {
    "small": FamilyStyle("#D55E00", "^", ":", "Pequeño"),
    "medium": FamilyStyle("#0072B2", "s", "--", "Mediano"),
    "large": FamilyStyle("#009E73", "o", "-", "Grande"),
}

#: Estilo por partición. Colores de la misma paleta, distintos de los de SIZE_STYLES. Estas barras
#: se dibujan sin trama porque son tres series rotuladas con leyenda.
SPLIT_STYLES: dict[str, FamilyStyle] = {
    "train": FamilyStyle("#0072B2", "o", "-", "Entrenamiento"),
    "val": FamilyStyle("#F0E442", "s", "--", "Validación"),
    "test": FamilyStyle("#009E73", "^", ":", "Prueba"),
}

#: Estilo por clase de error (las claves de error_decomposition.ERROR_KINDS). missed se dibuja
#: aparte con su marcador porque no es un falso positivo.
ERROR_KIND_STYLES: dict[str, FamilyStyle] = {
    "classification": FamilyStyle("#E69F00", "o", "-", "Clase equivocada"),
    "localization": FamilyStyle("#56B4E9", "s", "--", "Mal localizada"),
    "both": FamilyStyle("#009E73", "^", ":", "Clase y localización"),
    "duplicate": FamilyStyle("#CC79A7", "D", "-.", "Duplicado"),
    "background": FamilyStyle("#0072B2", "v", (0, (3, 1, 1, 1)), "Fondo"),
    "missed": FamilyStyle("#D55E00", "o", "-", "No detectada"),
}

#: 'viridis' es perceptualmente uniforme y se ve bien en escala de grises.
SEQUENTIAL_COLORMAP = "viridis"

DIVERGING_COLORMAP = "RdBu_r"


def apply_style() -> None:
    """Aplica el estilo común a Matplotlib. Es idempotente."""
    try:
        import matplotlib
    except ImportError as error:
        from tfm_edge.common.exceptions import DependencyNotAvailableError

        raise DependencyNotAvailableError("matplotlib", "generar figuras") from error

    # 'Agg' no necesita entorno gráfico.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Tipografía con serifa, como el cuerpo del documento.
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            # Rejilla suave y sin adornos.
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            "errorbar.capsize": 3,
        }
    )


def style_for(technique: str) -> FamilyStyle:
    """Estilo de una familia de técnica. Si no está registrada usa el de baseline y lo avisa en el
    log.
    """
    if technique not in TECHNIQUE_STYLES:
        logger.warning(
            "Familia de técnica sin estilo: '%s'. Se usa el de referencia; añádela a "
            "TECHNIQUE_STYLES.",
            technique,
        )
        return TECHNIQUE_STYLES["baseline"]
    return TECHNIQUE_STYLES[technique]


def split_style_for(split: str) -> FamilyStyle:
    """Estilo de una partición (train, val o test). Un nombre desconocido lanza KeyError."""
    if split not in SPLIT_STYLES:
        raise KeyError(
            f"Partición '{split}' sin estilo asignado. Las registradas son: "
            f"{', '.join(SPLIT_STYLES)}. Añádela a SPLIT_STYLES si es nueva."
        )
    return SPLIT_STYLES[split]


def format_number(value: float, decimals: int = 2) -> str:
    """Formatea un número con coma decimal. No usa el locale del sistema para no depender del
    entorno.
    """
    return f"{value:.{decimals}f}".replace(".", ",")


def format_grouped(value: float, decimals: int = 0) -> str:
    """Formatea con punto de millares y coma decimal, por ejemplo ``"12.345"`` o ``"1.234,50"``.

    ``f"{valor:,}"`` usaría la coma inglesa, que en el documento es el separador decimal.
    """
    texto = f"{abs(value):,.{decimals}f}"
    entero, _, decimal = texto.partition(".")
    resultado = entero.replace(",", ".")
    if decimal:
        resultado = f"{resultado},{decimal}"
    # El signo solo se pone si el valor redondeado no es cero, para evitar «-0,00».
    if value < 0 and float(texto.replace(",", "")) != 0.0:
        resultado = f"-{resultado}"
    return resultado


def format_integer(value: int | float) -> str:
    """Formatea un entero con punto de millares, por ejemplo ``"12.345"``."""
    return format_grouped(value, 0)


#: Máximo de decimales en un eje; si hacen falta más conviene cambiar de unidad.
MAX_AXIS_DECIMALS = 3


def axis_decimals(ticks: Iterable[float], max_decimals: int = MAX_AXIS_DECIMALS) -> int:
    """Número de decimales de un eje: el mínimo con el que redondear no cambia ninguna marca.

    Si ninguno hasta ``max_decimals`` basta, devuelve el máximo y lo avisa en el log.
    """
    finitas = [float(t) for t in ticks if math.isfinite(float(t))]
    if not finitas:
        return 0
    tolerancia = 1e-9 * max(1.0, max(abs(t) for t in finitas))
    for decimales in range(max_decimals + 1):
        if all(abs(t - round(t, decimales)) <= tolerancia for t in finitas):
            return decimales
    logger.warning(
        "Con hasta %d decimales no se representan exactamente las marcas del eje (%s...). Se "
        "usan %d y las etiquetas quedan redondeadas.",
        max_decimals,
        ", ".join(repr(t) for t in finitas[:4]),
        max_decimals,
    )
    return max_decimals


def comma_axis_formatter(max_decimals: int = MAX_AXIS_DECIMALS) -> Any:
    """Formateador de eje con coma decimal y el mismo número de decimales en todas las marcas.

    Los decimales se deciden con las marcas visibles (set_locs) y los millares se agrupan con
    format_grouped. Cada eje necesita su propio formateador.
    """
    from matplotlib.ticker import Formatter

    class _CommaAxisFormatter(Formatter):
        """Rotula con coma decimal y las mismas cifras en todo el eje."""

        def __init__(self) -> None:
            # None mientras Matplotlib no haya pasado las marcas.
            self._decimales: int | None = None

        def set_locs(self, locs: Any) -> None:
            """Recibe las marcas del eje y fija los decimales."""
            visibles = [float(t) for t in locs]
            eje = getattr(self, "axis", None)
            if eje is not None:
                inferior, superior = sorted(eje.get_view_interval())
                margen = abs(superior - inferior) * 1e-9
                dentro = [t for t in visibles if inferior - margen <= t <= superior + margen]
                if dentro:
                    visibles = dentro
            self._decimales = axis_decimals(visibles, max_decimals)

        def __call__(self, x: float, pos: Any = None) -> str:
            """Etiqueta de una marca con coma decimal."""
            if self._decimales is None:
                # Sin marcas todavía, se usa el número de cifras que necesite el valor.
                return format_grouped(x, axis_decimals([x], max_decimals))
            return format_grouped(x, self._decimales)

    return _CommaAxisFormatter()


def figure_filename(chapter: int, section: int, description: str) -> str:
    """Nombre de fichero de una figura sin extensión, por ejemplo ``fig_5_3_map_vs_ratio_poda``."""
    from tfm_edge.common.paths import slugify

    return f"fig_{chapter}_{section}_{slugify(description)}"


def table_filename(chapter: int, section: int, description: str) -> str:
    """Nombre de fichero de una tabla sin extensión, por ejemplo
    ``tab_6_4_frontera_workstation_cpu``.
    """
    from tfm_edge.common.paths import slugify

    return f"tab_{chapter}_{section}_{slugify(description)}"


def save_figure(
    figure: Any,
    output_dir: Path | str,
    chapter: int,
    section: int,
    description: str,
    close: bool = True,
) -> dict[str, Path]:
    """Guarda una figura en PDF vectorial y PNG a 300 ppp y devuelve las rutas."""
    import matplotlib.pyplot as plt

    directorio = ensure_dir(output_dir)
    base = figure_filename(chapter, section, description)
    rutas: dict[str, Path] = {}
    for formato in FIGURE_FORMATS:
        destino = ensure_parent_dir(directorio / f"{base}.{formato}")
        figure.savefig(destino, format=formato)
        rutas[formato] = destino
    logger.info("Figura guardada: %s.{%s}", base, ",".join(FIGURE_FORMATS))
    if close:
        plt.close(figure)
    return rutas


def manifest_files(paths: dict[str, Path]) -> dict[str, str]:
    """Convierte las rutas escritas por un paso en rutas relativas a la raíz para el manifiesto."""
    return {formato: relative_to_root(ruta) for formato, ruta in paths.items()}


def grayscale_safe_bars(axes: Any, containers: Iterable[Any], per_bar: bool = False) -> None:
    """Añade tramas a las barras para distinguirlas en escala de grises.

    Por defecto la trama cambia por contenedor (barras apiladas); con ``per_bar`` cambia de barra en
    barra dentro del mismo contenedor.
    """
    tramas = ("", "///", "...", "xxx", "\\\\\\", "+++")
    for indice, contenedor in enumerate(containers):
        for posicion, barra in enumerate(contenedor):
            trama = tramas[(posicion if per_bar else indice) % len(tramas)]
            barra.set_hatch(trama)
            barra.set_edgecolor("black")
            barra.set_linewidth(0.4)
    _ = axes
