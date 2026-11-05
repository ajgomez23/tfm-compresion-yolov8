"""Exportación de tablas a XLSX, LaTeX y CSV.

Quien llama decide filas y columnas; aquí se resuelven el redondeo, las unidades y el marcado del
mejor valor. LaTeX lleva coma decimal y negritas; XLSX guarda los números completos con el redondeo
como formato de celda; CSV guarda los valores sin redondear. Si una columna declara su suelo de
medida, se marcan en negrita todos los valores indistinguibles del mejor, y si son toda la columna
no se marca ninguno y se indica al pie.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.analysis.floors import same_environment
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir
from tfm_edge.reporting import style

logger = get_logger(__name__)

TABLE_FORMATS = ("tex", "xlsx", "csv")

#: Redondeo: cuatro decimales en calidad y dos en latencias en ms; el resto según la magnitud.
QUALITY_DECIMALS = 4
LATENCY_DECIMALS = 2
MEMORY_DECIMALS = 1
SIZE_DECIMALS = 2
PERCENT_DECIMALS = 1
RATIO_DECIMALS = 3

#: Ancho máximo de columna en la hoja, en caracteres.
MAX_XLSX_COLUMN_WIDTH = 46

#: Tamaños de letra de LaTeX admitidos. Una orden desconocida no falla al generar el .tex sino al
#: compilar.
LATEX_FONT_SIZES = (
    "normalsize",
    "small",
    "footnotesize",
    "scriptsize",
    "tiny",
)

#: Orientaciones en LaTeX. ``landscape`` pone la tabla apaisada y la reduce hasta que quepa en una
#: página, sin ampliarla nunca; la reducción la calcula LaTeX al componer.
LATEX_LAYOUTS = ("portrait", "landscape")

#: Caracteres que pueden llegar a un .tex: ASCII imprimible y los no ASCII que pdflatex compone con
#: fontenc T1 (comprobado con pdfTeX 1.40.28, TeX Live 2025). Quedan fuera, entre otros, Δ ′ ″ − ≈ ≠
#: ≤ ≥ ○ ✓ ✕.
LATEX_SAFE_CHARACTERS = frozenset(chr(codigo) for codigo in range(0x20, 0x7F)) | frozenset(
    "¡¢£¤¥¦§¨©ª«¬®¯°±²³´µ¶·¸¹º»¼½¾¿"
    "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞß"
    "àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
    "–—‘’‚“”„†‡•…‰‹›€←↑→↓"
)

#: Caracteres que necesitan fontenc T1: los no ASCII no compilan en OT1 y seis ASCII compilan pero
#: se imprimen como otro carácter. Si una tabla lleva alguno, su cabecera pide fontenc. Un test lee
#: el PDF para comprobar que se imprimen bien.
T1_REQUIRED_CHARACTERS = frozenset('«»ÐÞðþ‚„‹›<>|"^~')

#: Caracteres que TeX une en ligadura con el siguiente igual (``--``, ``<<``, ``,,``); se separan
#: con un grupo vacío.
_LATEX_LIGATURE_CHARACTERS = frozenset("-<>,")

#: Fecha fija en los metadatos del XLSX, la misma de SOURCE_DATE_EPOCH, para que dos ejecuciones
#: iguales den el mismo fichero.
FIXED_TIMESTAMP_EPOCH = style.FIXED_TIMESTAMP_EPOCH


@dataclass(frozen=True)
class Column:
    """Columna de una tabla.

    ``floor`` o ``floor_relative`` es la resolución de la medida (absoluta o relativa al menor de
    los dos valores, como en pareto.compare) y requiere ``floor_note`` con su procedencia.
    ``floor_same_session`` es el suelo para dos filas de la misma sesión.
    """

    key: str
    header: str
    unit: str | None = None
    decimals: int | None = None
    better: str | None = None
    floor: float | None = None
    floor_relative: float | None = None
    floor_note: str | None = None
    floor_same_session: float | None = None

    def __post_init__(self) -> None:
        """Comprueba que la columna es coherente y lanza ValueError si no."""
        if self.better not in (None, "max", "min"):
            raise ValueError(
                f"La columna '{self.key}' declara better='{self.better}'; "
                f"solo se admiten 'max', 'min' o None."
            )
        if self.floor is not None and self.floor_relative is not None:
            raise ValueError(
                f"La columna '{self.key}' declara suelo absoluto y relativo a la vez; "
                f"solo uno de los dos."
            )
        if (self.floor is not None or self.floor_relative is not None) and not self.floor_note:
            raise ValueError(
                f"La columna '{self.key}' declara un suelo "
                f"({self.floor if self.floor is not None else self.floor_relative}) sin floor_note."
            )
        if self.floor_same_session is not None and not self.has_floor:
            raise ValueError(
                f"La columna '{self.key}' declara suelo de la misma sesión "
                f"({self.floor_same_session}) sin el de sesiones distintas."
            )
        if self.decimals is None and self.better is not None:
            raise ValueError(
                f"La columna de texto '{self.key}' no puede tener un mejor valor: "
                f"para comparar hace falta que sea numérica."
            )

    @property
    def has_floor(self) -> bool:
        """True si la columna declara suelo, absoluto o relativo."""
        return self.floor is not None or self.floor_relative is not None

    def tolerance(self, best: float, value: float, same_session: bool = False) -> float:
        """Umbral por debajo del cual ``value`` no se distingue de ``best``.

        Hace el mismo cálculo que pareto.compare. ``same_session`` es False por defecto, que es lo
        conservador.
        """
        if same_session and self.floor_same_session is not None:
            suelo = self.floor_same_session
        elif self.floor_relative is not None:
            suelo = self.floor_relative
        else:
            suelo = self.floor or 0.0
        if self.floor_relative is not None:
            return suelo * min(abs(best), abs(value))
        return suelo

    def _floor_value_text(self, value: float) -> str:
        """Valor de suelo como se cita al pie, con su unidad o en porcentaje."""
        if self.floor_relative is not None:
            return f"{style.format_number(value * 100.0, 1)} %"
        texto = style.format_number(value, self.decimals or 2)
        return f"{texto} {self.unit}" if self.unit else texto

    @property
    def floor_text(self) -> str:
        """Texto del suelo para el pie; si hay suelo de la misma sesión, los dos."""
        base = self.floor_relative if self.floor_relative is not None else self.floor
        if base is None:
            return ""
        if self.floor_same_session is None or self.floor_same_session == base:
            return self._floor_value_text(base)
        return (
            f"{self._floor_value_text(self.floor_same_session)} en la misma sesión, "
            f"{self._floor_value_text(base)} entre sesiones"
        )

    @property
    def is_numeric(self) -> bool:
        """True si la columna tiene redondeo declarado."""
        return self.decimals is not None

    @property
    def header_text(self) -> str:
        """Cabecera con la unidad, por ejemplo ``latencia p50 (ms)``."""
        return f"{self.header} ({self.unit})" if self.unit else self.header

    @property
    def number_format(self) -> str:
        """Formato de celda de Excel, por ejemplo ``0.0000``."""
        if self.decimals is None:
            raise ValueError(f"La columna de texto '{self.key}' no tiene formato numérico.")
        return "0" if self.decimals == 0 else "0." + "0" * self.decimals


def text_column(key: str, header: str) -> Column:
    """Columna de texto."""
    return Column(key=key, header=header)


def quality_column(
    key: str,
    header: str,
    *,
    better: str = "max",
    floor: float | None = None,
    floor_relative: float | None = None,
    floor_note: str | None = None,
    floor_same_session: float | None = None,
) -> Column:
    """Columna de una métrica de calidad, con cuatro decimales."""
    return Column(
        key=key,
        header=header,
        decimals=QUALITY_DECIMALS,
        better=better,
        floor=floor,
        floor_relative=floor_relative,
        floor_note=floor_note,
        floor_same_session=floor_same_session,
    )


def latency_column(
    key: str,
    header: str,
    *,
    floor: float | None = None,
    floor_relative: float | None = None,
    floor_note: str | None = None,
    floor_same_session: float | None = None,
) -> Column:
    """Columna de latencia en milisegundos, con dos decimales."""
    return Column(
        key=key,
        header=header,
        unit="ms",
        decimals=LATENCY_DECIMALS,
        better="min",
        floor=floor,
        floor_relative=floor_relative,
        floor_note=floor_note,
        floor_same_session=floor_same_session,
    )


def memory_column(
    key: str,
    header: str,
    *,
    unit: str = "MiB",
    floor: float | None = None,
    floor_relative: float | None = None,
    floor_note: str | None = None,
    floor_same_session: float | None = None,
) -> Column:
    """Columna de memoria. ``unit`` permite MB en lugar de MiB."""
    return Column(
        key=key,
        header=header,
        unit=unit,
        decimals=MEMORY_DECIMALS,
        better="min",
        floor=floor,
        floor_relative=floor_relative,
        floor_note=floor_note,
        floor_same_session=floor_same_session,
    )


def size_column(key: str, header: str, *, unit: str = "MB") -> Column:
    """Columna de tamaño de artefacto."""
    return Column(key=key, header=header, unit=unit, decimals=SIZE_DECIMALS, better="min")


def percent_column(key: str, header: str, *, better: str | None = None) -> Column:
    """Columna en tanto por ciento."""
    return Column(key=key, header=header, unit="%", decimals=PERCENT_DECIMALS, better=better)


def count_column(key: str, header: str, *, better: str | None = None) -> Column:
    """Columna de recuento, sin decimales."""
    return Column(key=key, header=header, decimals=0, better=better)


@dataclass(frozen=True)
class Table:
    """Tabla lista para escribir.

    ``latex_font`` es footnotesize por defecto porque los experiment_id completos no caben a tamaño
    normal. ``environment_key`` es la clave con la huella de entorno de cada fila; con ella
    best_marks usa el suelo de la misma sesión cuando corresponde.
    """

    name: str
    caption: str
    chapter: int
    section: int
    columns: tuple[Column, ...]
    rows: tuple[Mapping[str, Any], ...]
    notes: tuple[str, ...] = field(default_factory=tuple)
    latex_font: str = "footnotesize"
    latex_layout: str = "portrait"
    environment_key: str | None = None

    def __post_init__(self) -> None:
        """Comprueba que la tabla se puede escribir y lanza ValueError si no."""
        if not self.columns:
            raise ValueError(f"La tabla '{self.name}' no tiene ninguna columna.")
        if not self.rows:
            raise ValueError(
                f"La tabla '{self.name}' no tiene filas; no se escribe una tabla vacía."
            )
        if self.latex_font not in LATEX_FONT_SIZES:
            raise ValueError(
                f"La tabla '{self.name}' pide el tamaño de letra '{self.latex_font}', que no es "
                f"válido. Admitidos: {', '.join(LATEX_FONT_SIZES)}."
            )
        if self.latex_layout not in LATEX_LAYOUTS:
            raise ValueError(
                f"La tabla '{self.name}' pide la orientación '{self.latex_layout}'. "
                f"Admitidas: {', '.join(LATEX_LAYOUTS)}."
            )
        claves = [columna.key for columna in self.columns]
        repetidas = sorted({clave for clave in claves if claves.count(clave) > 1})
        if repetidas:
            raise ValueError(f"La tabla '{self.name}' repite las claves de columna {repetidas}.")

    @property
    def basename(self) -> str:
        """Nombre de fichero sin extensión, con el prefijo de capítulo y apartado."""
        return style.table_filename(self.chapter, self.section, self.name)

    @property
    def label(self) -> str:
        """Etiqueta de LaTeX, por ejemplo ``tab:6_4_frontera``."""
        return f"tab:{self.basename.removeprefix('tab_')}"


def cell_value(row: Mapping[str, Any], column: Column) -> float | str | None:
    """Valor de una celda: número, texto o None si está vacía.

    Una columna numérica que recibe algo que no es un número lanza ValueError.
    """
    bruto = row.get(column.key)
    if bruto is None or (isinstance(bruto, str) and not bruto.strip()):
        return None
    if not column.is_numeric:
        return str(bruto)
    if isinstance(bruto, bool):
        raise ValueError(f"La columna numérica '{column.key}' ha recibido un booleano ({bruto!r}).")
    try:
        valor = float(bruto)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"La columna numérica '{column.key}' ha recibido {bruto!r}, que no es un número. Si "
            "no se midió, la celda debe venir vacía."
        ) from error
    if valor != valor:
        raise ValueError(
            f"La columna numérica '{column.key}' ha recibido NaN. Un valor que no se "
            f"midió va como celda vacía, no como NaN."
        )
    return valor


def format_value(value: float | str | None, column: Column) -> str:
    """Texto de una celda; vacío si el valor falta."""
    if value is None:
        return ""
    if not column.is_numeric:
        return str(value)
    assert column.decimals is not None
    return style.format_number(float(value), column.decimals)


def best_marks(table: Table) -> tuple[dict[str, frozenset[int]], tuple[str, ...]]:
    """Decide qué celdas van en negrita y qué notas añadir al pie.

    Con suelo se marca el conjunto indistinguible del mejor; si es toda la columna no se marca nada.
    El suelo se decide pareja a pareja según si las dos filas comparten sesión, como en
    pareto.compare.
    """

    def huella(indice: int) -> str:
        if table.environment_key is None:
            return ""
        return str(table.rows[indice].get(table.environment_key) or "")

    marcas: dict[str, frozenset[int]] = {}
    notas: list[str] = []
    for columna in table.columns:
        if columna.better is None:
            continue
        valores = [(indice, cell_value(fila, columna)) for indice, fila in enumerate(table.rows)]
        numericos = [(indice, float(valor)) for indice, valor in valores if valor is not None]
        if len(numericos) < 2:
            # Con un solo valor no hay nada que comparar y no se marca.
            continue
        mejor = (
            max(valor for _, valor in numericos)
            if columna.better == "max"
            else min(valor for _, valor in numericos)
        )
        filas_mejores = [indice for indice, valor in numericos if valor == mejor]
        ganadores = frozenset(
            indice
            for indice, valor in numericos
            if any(
                (mejor - valor if columna.better == "max" else valor - mejor)
                <= columna.tolerance(
                    mejor, valor, same_session=same_environment(huella(indice), huella(otra))
                )
                for otra in filas_mejores
            )
        )
        if len(ganadores) == len(numericos):
            notas.append(
                f"En «{columna.header_text}» ninguna diferencia supera la resolución de "
                f"la medida ({columna.floor_text}, {columna.floor_note}), así que no se "
                f"marca ningún mejor valor."
                if columna.has_floor
                else f"En «{columna.header_text}» todos los valores empatan."
            )
            continue
        marcas[columna.key] = ganadores
    return marcas, tuple(notas)


def _escape_latex(text: str) -> str:
    """Escapa los caracteres especiales de LaTeX y los que se imprimirían cambiados (por ejemplo,
    los guiones bajos de los experiment_id).
    """
    reemplazos = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "'": r"\textquotesingle{}",
        "`": r"\textasciigrave{}",
    }
    trozos: list[str] = []
    for posicion, caracter in enumerate(text):
        trozos.append(reemplazos.get(caracter, caracter))
        if caracter in _LATEX_LIGATURE_CHARACTERS and text[posicion + 1 : posicion + 2] == caracter:
            trozos.append("{}")
    return "".join(trozos)


def _atomic_write_text(path: Path, content: str) -> Path:
    """Escribe un fichero de texto de forma atómica."""
    destino = ensure_parent_dir(path)
    temporal = destino.with_name(destino.name + ".tmp")
    temporal.write_text(content, encoding="utf-8", newline="\n")
    os.replace(temporal, destino)
    return destino


def render_latex(table: Table) -> str:
    r"""Contenido del fichero .tex, listo para ``\input``."""
    marcas, automaticas = best_marks(table)
    notas = [*table.notes, *automaticas]
    alineacion = "".join("r" if columna.is_numeric else "l" for columna in table.columns)

    # Todo el texto de la tabla antes de escapar: se valida y decide si hace falta T1.
    crudos: list[str] = [table.caption, *notas]
    crudos.extend(columna.header_text for columna in table.columns)
    celdas_por_fila: list[list[str]] = []
    for indice, fila in enumerate(table.rows):
        celdas: list[str] = []
        for columna in table.columns:
            crudo = format_value(cell_value(fila, columna), columna)
            crudos.append(crudo)
            texto = _escape_latex(crudo)
            if texto and indice in marcas.get(columna.key, frozenset()):
                texto = f"\\textbf{{{texto}}}"
            celdas.append(texto)
        celdas_por_fila.append(celdas)
    _check_latex_characters(table, crudos)

    cabecera = " & ".join(_escape_latex(columna.header_text) for columna in table.columns)
    tabular = [
        f"  \\begin{{tabular}}{{{alineacion}}}",
        r"    \toprule",
        "    " + cabecera + r" \\",
        r"    \midrule",
        *["    " + " & ".join(celdas) + r" \\" for celdas in celdas_por_fila],
        r"    \bottomrule",
        r"  \end{tabular}",
    ]
    titulo = [
        f"  \\caption{{{_escape_latex(table.caption)}}}",
        f"  \\label{{{table.label}}}",
    ]
    pie = (
        [
            r"  \begin{minipage}{\linewidth}\footnotesize",
            *[f"  {_escape_latex(nota)}" for nota in notas],
            r"  \end{minipage}",
        ]
        if notas
        else []
    )

    apaisada = table.latex_layout == "landscape"
    if apaisada:
        lineas = _landscape_body(table, titulo, tabular, pie)
    else:
        lineas = [
            r"\begin{table}[htbp]",
            r"  \centering",
            *titulo,
            f"  \\{table.latex_font}",
            *tabular,
            *([r"  \par\vspace{2pt}", *pie] if pie else []),
            r"\end{table}",
        ]

    paquetes = [r"\usepackage{booktabs}"]
    if any(caracter in T1_REQUIRED_CHARACTERS for texto in crudos for caracter in texto):
        paquetes.append(r"\usepackage[T1]{fontenc}")
    if apaisada:
        paquetes.extend([r"\usepackage{pdflscape}", r"\usepackage{adjustbox}"])
    cabecera_fichero = [
        "% Generado por tfm_edge.reporting.tables. No editar a mano: se regenera.",
        f"% Requiere {', '.join(paquetes)}.",
    ]
    return "\n".join([*cabecera_fichero, *lineas]) + "\n"


def _check_latex_characters(table: Table, texts: Sequence[str]) -> None:
    """Lanza ValueError si hay caracteres fuera de LATEX_SAFE_CHARACTERS."""
    fuera = sorted({c for texto in texts for c in texto if c not in LATEX_SAFE_CHARACTERS})
    if fuera:
        detalle = ", ".join(f"'{c}' (U+{ord(c):04X})" for c in fuera)
        raise ValueError(
            f"La tabla '{table.name}' lleva caracteres que pdflatex no compone con fontenc T1: "
            f"{detalle}. Sustitúyelos o, si compilan, añádelos a LATEX_SAFE_CHARACTERS."
        )


def _landscape_body(
    table: Table, title: list[str], tabular: list[str], notes: list[str]
) -> list[str]:
    r"""Cuerpo de una tabla apaisada que se reduce hasta caber en una página.

    Título y pie se miden en cajas y el resto de ``\textheight`` es el alto máximo del tabular;
    ``adjustbox`` con ``max`` nunca amplía. Se restan 6 pt más, medidos de forma empírica.
    """
    return [
        r"\ifdefined\tfmedgetitle\else\newsavebox{\tfmedgetitle}\fi",
        r"\ifdefined\tfmedgenotes\else\newsavebox{\tfmedgenotes}\fi",
        r"\begin{landscape}",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \sbox{\tfmedgetitle}{\parbox{\linewidth}{%",
        *title,
        r"  }}%",
        r"  \sbox{\tfmedgenotes}{%",
        # El «%» evita el espacio que añadiría el salto de línea tras \end{minipage}.
        *(notes[:-1] + [notes[-1] + "%"] if notes else [r"  \relax"]),
        r"  }%",
        r"  \usebox{\tfmedgetitle}\par",
        f"  \\{table.latex_font}",
        r"  \begin{adjustbox}{max width=\linewidth, max totalheight=\dimexpr\textheight"
        r"-\ht\tfmedgetitle-\dp\tfmedgetitle-\ht\tfmedgenotes-\dp\tfmedgenotes"
        r"-6pt\relax}",
        *tabular,
        r"  \end{adjustbox}",
        *([r"  \par\vspace{2pt}", r"  \usebox{\tfmedgenotes}"] if notes else []),
        r"\end{table}",
        r"\end{landscape}",
    ]


def write_latex(table: Table, path: Path | str) -> Path:
    """Escribe la tabla en LaTeX."""
    return _atomic_write_text(Path(path), render_latex(table))


def write_csv(table: Table, path: Path | str) -> Path:
    """Escribe la tabla en CSV sin redondear, con la clave de la columna como cabecera."""
    import io

    buffer = io.StringIO(newline="")
    escritor = csv.DictWriter(
        buffer,
        fieldnames=[columna.key for columna in table.columns],
        lineterminator="\n",
        extrasaction="ignore",
    )
    escritor.writeheader()
    for fila in table.rows:
        escritor.writerow(
            {
                columna.key: ("" if cell_value(fila, columna) is None else fila.get(columna.key))
                for columna in table.columns
            }
        )
    return _atomic_write_text(Path(path), buffer.getvalue())


#: Fecha fija en el formato de docProps/core.xml.
FIXED_TIMESTAMP_TEXT = "2026-01-01T00:00:00Z"

CORE_PROPERTIES_ENTRY = "docProps/core.xml"


def _freeze_modified_date(content: bytes) -> bytes:
    """Fija la fecha de modificación en docProps/core.xml.

    openpyxl la reescribe al guardar, así que se sustituye en el XML ya escrito. Si no se encuentra
    el elemento, falla.
    """
    import re

    patron = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")
    nuevo, sustituciones = patron.subn(
        rb"\g<1>" + FIXED_TIMESTAMP_TEXT.encode("ascii") + rb"\g<2>", content
    )
    if sustituciones != 1:
        raise ValueError(
            f"No se ha encontrado la fecha de modificación en {CORE_PROPERTIES_ENTRY} "
            f"({sustituciones} coincidencias). Puede que openpyxl haya cambiado el formato."
        )
    return nuevo


def _rewrite_zip_deterministic(path: Path) -> None:
    """Reescribe un XLSX con fechas fijas en las entradas del zip y en docProps/core.xml."""
    import zipfile

    fijo = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(path, "r") as origen:
        entradas = [(info, origen.read(info.filename)) for info in origen.infolist()]

    temporal = path.with_name(path.name + ".zip.tmp")
    with zipfile.ZipFile(temporal, "w", zipfile.ZIP_DEFLATED) as destino:
        for info, contenido in entradas:
            if info.filename == CORE_PROPERTIES_ENTRY:
                contenido = _freeze_modified_date(contenido)
            nueva = zipfile.ZipInfo(filename=info.filename, date_time=fijo)
            nueva.compress_type = info.compress_type
            nueva.external_attr = 0o600 << 16
            nueva.create_system = 3
            destino.writestr(nueva, contenido)
    os.replace(temporal, path)


def _sheet_name(name: str) -> str:
    """Nombre de hoja válido para Excel: 31 caracteres como máximo y sin los prohibidos."""
    limpio = "".join("_" if caracter in "[]:*?/\\" else caracter for caracter in name)
    return limpio[:31] or "tabla"


def write_xlsx(table: Table, path: Path | str) -> Path:
    """Escribe la tabla en XLSX con los números completos y el redondeo como formato de celda."""
    from datetime import datetime, timezone

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.utils import get_column_letter

    marcas, automaticas = best_marks(table)
    libro = Workbook()
    hoja = libro.active
    hoja.title = _sheet_name(table.name)

    negrita = Font(bold=True)
    for indice, columna in enumerate(table.columns, start=1):
        celda = hoja.cell(row=1, column=indice, value=columna.header_text)
        celda.font = negrita
        celda.alignment = Alignment(horizontal="center", vertical="bottom", wrap_text=True)

    for numero_fila, fila in enumerate(table.rows, start=2):
        for indice, columna in enumerate(table.columns, start=1):
            valor = cell_value(fila, columna)
            if valor is None:
                continue
            celda = hoja.cell(row=numero_fila, column=indice, value=valor)
            if columna.is_numeric:
                celda.number_format = columna.number_format
            if numero_fila - 2 in marcas.get(columna.key, frozenset()):
                celda.font = negrita

    fila_notas = len(table.rows) + 3
    for desplazamiento, nota in enumerate([*table.notes, *automaticas]):
        hoja.cell(row=fila_notas + desplazamiento, column=1, value=nota)

    for indice, columna in enumerate(table.columns, start=1):
        anchos = [max(len(parte) for parte in columna.header_text.split(" "))]
        anchos.extend(len(format_value(cell_value(fila, columna), columna)) for fila in table.rows)
        hoja.column_dimensions[get_column_letter(indice)].width = min(
            MAX_XLSX_COLUMN_WIDTH, max(anchos) + 2
        )

    hoja.freeze_panes = "A2"
    hoja.auto_filter.ref = f"A1:{get_column_letter(len(table.columns))}{len(table.rows) + 1}"

    # Metadatos con fecha fija para que el fichero sea reproducible.
    fijo = datetime.fromtimestamp(FIXED_TIMESTAMP_EPOCH, tz=timezone.utc).replace(tzinfo=None)
    libro.properties.creator = "tfm-edge-object-detection"
    libro.properties.lastModifiedBy = "tfm-edge-object-detection"
    libro.properties.created = fijo
    libro.properties.modified = fijo

    destino = ensure_parent_dir(Path(path))
    temporal = destino.with_name(destino.name + ".tmp")
    libro.save(temporal)
    _rewrite_zip_deterministic(temporal)
    os.replace(temporal, destino)
    return destino


def save_table(table: Table, output_dir: Path | str) -> dict[str, Path]:
    """Escribe la tabla en los tres formatos."""
    directorio = ensure_dir(output_dir)
    escritores = {"tex": write_latex, "xlsx": write_xlsx, "csv": write_csv}
    rutas = {
        formato: escritores[formato](table, directorio / f"{table.basename}.{formato}")
        for formato in TABLE_FORMATS
    }
    logger.info(
        "Tabla guardada: %s.{%s} (%d filas, %d columnas)",
        table.basename,
        ",".join(TABLE_FORMATS),
        len(table.rows),
        len(table.columns),
    )
    return rutas


def save_tables(tables: Iterable[Table], output_dir: Path | str) -> dict[str, dict[str, Path]]:
    """Escribe varias tablas."""
    return {tabla.basename: save_table(tabla, output_dir) for tabla in tables}


def table_summary(table: Table) -> dict[str, Any]:
    """Resumen de lo que afirma la tabla (negritas y notas), para el manifiesto."""
    marcas, automaticas = best_marks(table)
    return {
        "basename": table.basename,
        "caption": table.caption,
        "num_rows": len(table.rows),
        "columns": [columna.key for columna in table.columns],
        "best_marked": {clave: sorted(indices) for clave, indices in sorted(marcas.items())},
        "columns_without_distinction": [
            columna.key
            for columna in table.columns
            if columna.better is not None and columna.key not in marcas
        ],
        "automatic_notes": list(automaticas),
    }


def rows_from_sequence(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[Column]
) -> tuple[Mapping[str, Any], ...]:
    """Deja en cada fila solo las claves de las columnas."""
    claves = [columna.key for columna in columns]
    return tuple({clave: fila.get(clave, "") for clave in claves} for fila in rows)
