#!/usr/bin/env python
"""Convierte en relativas las rutas absolutas que ya estaban en el registro.

append ya normaliza las rutas al escribir; este script corrige las celdas escritas antes. Solo
modifica columnas de TABLE_PATH_COLUMNS y solo si la ruta nueva apunta al mismo fichero
(os.path.samefile). No convierte rutas de fuera del proyecto (las de Drive); eso lo hace
14_import_colab_registry.py. Por defecto solo muestra el plan (``--apply`` para escribir), exige el
registro limpio en git (salvo ``--allow-dirty``) y una cabecera igual al esquema, y después de
escribir relee y compara celda a celda, restaurando si algo no cuadra.

Códigos de salida: 0 si no hay nada que hacer, se muestra el plan o se aplica y verifica; 1 si hay
un error de uso o de entorno; 2 si hay incidencias o falla la verificación.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ProjectPaths, get_project_paths, relative_to_root  # noqa: E402
from tfm_edge.common.registry import (  # noqa: E402
    TABLE_PATH_COLUMNS,
    ExperimentRegistry,
    is_absolute_path_text,
)
from tfm_edge.common.registry_guards import check_git_clean, read_header  # noqa: E402

logger = get_logger(__name__)

TABLES: tuple[str, ...] = ("experiments", "quality_metrics", "deployment_metrics", "environments")


@dataclass(frozen=True)
class PathChange:
    """Celda de ruta que se va a reescribir."""

    table: str
    line: int
    row_index: int
    column: str
    current: str
    proposed: str


def plan_table(
    registry: ExperimentRegistry,
    table: str,
    paths: ProjectPaths,
) -> tuple[list[dict[str, str]], list[PathChange], list[str]]:
    """Decide qué celdas de una tabla se pueden reescribir y devuelve ``(filas, cambios,
    incidencias)``.
    """
    incidencias: list[str] = []
    columnas_ruta = TABLE_PATH_COLUMNS.get(table, ())
    filas = registry.read_rows(table)
    if not filas or not columnas_ruta:
        return filas, [], incidencias

    cabecera = read_header(registry.path_for(table))
    esquema = registry.columns_for(table)
    if cabecera != esquema:
        incidencias.append(
            f"[{table}] la cabecera en disco no coincide con el esquema "
            f"({len(cabecera)} columnas frente a {len(esquema)}); no se reescribe, "
            "porque el reescritor emite las columnas del esquema y se perdería la "
            "diferencia"
        )
        return filas, [], incidencias

    cambios: list[PathChange] = []
    for indice, fila in enumerate(filas):
        linea = indice + 2  # +1 por la cabecera y +1 porque las líneas empiezan en 1
        for columna in columnas_ruta:
            valor = (fila.get(columna) or "").strip()
            if not valor or not is_absolute_path_text(valor):
                continue
            origen = Path(valor).expanduser()
            if not origen.exists():
                incidencias.append(
                    f"[{table}] L{linea}/{columna}: la ruta absoluta apunta a un fichero "
                    f"que no existe en esta máquina; se deja como está ({valor})"
                )
                continue
            propuesta = relative_to_root(origen, paths)
            if is_absolute_path_text(propuesta):
                incidencias.append(
                    f"[{table}] L{linea}/{columna}: cae fuera del árbol del proyecto; "
                    f"se deja absoluta, reapuntarla es trabajo del importador ({valor})"
                )
                continue
            destino = paths.root / propuesta
            if not destino.exists() or not os.path.samefile(destino, origen):
                incidencias.append(
                    f"[{table}] L{linea}/{columna}: la ruta relativa propuesta no resuelve "
                    f"al mismo fichero; se deja como está ({valor} -> {propuesta})"
                )
                continue
            cambios.append(
                PathChange(
                    table=table,
                    line=linea,
                    row_index=indice,
                    column=columna,
                    current=valor,
                    proposed=propuesta,
                )
            )
    return filas, cambios, incidencias


def build_rows(filas: list[dict[str, str]], cambios: list[PathChange]) -> list[dict[str, str]]:
    """Filas nuevas con los cambios aplicados sobre una copia."""
    nuevas = [dict(fila) for fila in filas]
    for cambio in cambios:
        nuevas[cambio.row_index][cambio.column] = cambio.proposed
    return nuevas


def diff_cells(antes: list[dict[str, str]], despues: list[dict[str, str]]) -> list[str]:
    """Celdas distintas entre dos listas de filas, como ``"L{linea}/{columna}"``."""
    if len(antes) != len(despues):
        return [f"el número de filas ha cambiado: {len(antes)} -> {len(despues)}"]
    diferencias: list[str] = []
    for indice, (fila_a, fila_b) in enumerate(zip(antes, despues)):
        if fila_a.keys() != fila_b.keys():
            diferencias.append(f"L{indice + 2}: el conjunto de columnas ha cambiado")
            continue
        for columna in fila_a:
            if fila_a[columna] != fila_b[columna]:
                diferencias.append(f"L{indice + 2}/{columna}")
    return diferencias


def apply_table(
    registry: ExperimentRegistry,
    table: str,
    previas: list[dict[str, str]],
    cambios: list[PathChange],
) -> list[str]:
    """Escribe una tabla con _write_rows_atomic y verifica que solo ha cambiado lo planificado.

    Si algo no cuadra, restaura la versión anterior.
    """
    esperadas = build_rows(previas, cambios)

    previstas = {f"L{c.line}/{c.column}" for c in cambios}
    inesperadas = set(diff_cells(previas, esperadas)) - previstas
    if inesperadas:
        return [
            f"[{table}] las filas propuestas difieren en celdas no planificadas: "
            f"{', '.join(sorted(inesperadas))}; no se escribe nada"
        ]

    registry._write_rows_atomic(table, esperadas)  # noqa: SLF001 (ver docstring)

    releidas = registry.read_rows(table)
    discrepancias = diff_cells(esperadas, releidas)
    if discrepancias:
        registry._write_rows_atomic(table, previas)  # noqa: SLF001
        return [
            f"[{table}] lo escrito no coincide con lo planificado ("
            f"{', '.join(discrepancias[:10])}); se ha restaurado la imagen previa"
        ]
    logger.info("[%s] reescritas %d celdas de ruta.", table, len(cambios))
    return []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y analiza los argumentos de la CLI."""
    rutas = get_project_paths()
    analizador = argparse.ArgumentParser(
        description=(
            "Reescribe como relativas a la raíz del proyecto las rutas absolutas que "
            "ya están registradas. Por defecto solo enseña el plan."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué celdas cambiarían (no escribe nada)\n"
            "  python scripts/19_normalize_registry_paths.py\n\n"
            "  # Aplicarlo\n"
            "  python scripts/19_normalize_registry_paths.py --apply\n\n"
            "  # Sobre una copia desechable del registro\n"
            "  python scripts/19_normalize_registry_paths.py --registry-dir /tmp/reg --apply\n"
        ),
    )
    analizador.add_argument(
        "--registry-dir",
        type=Path,
        default=rutas.results_registry,
        help="Directorio de los CSV del registro (por defecto: %(default)s).",
    )
    analizador.add_argument(
        "--table",
        nargs="+",
        choices=TABLES,
        default=None,
        help="Limitar a estas tablas. Por defecto, las cuatro.",
    )
    analizador.add_argument(
        "--apply",
        action="store_true",
        help="Escribe los cambios. Sin esta bandera no se toca ningún fichero.",
    )
    analizador.add_argument(
        "--allow-dirty",
        action="store_true",
        help=(
            "Permite escribir aunque el registro tenga cambios sin confirmar en git. "
            "Sin git limpio no hay forma de deshacer la reescritura."
        ),
    )
    analizador.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return analizador.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada."""
    args = parse_args(argv)
    setup_logging("19_normalize_registry_paths", level=args.log_level)
    rutas = get_project_paths()

    try:
        clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
        registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    tablas = tuple(args.table) if args.table else TABLES
    plan: dict[str, tuple[list[dict[str, str]], list[PathChange]]] = {}
    incidencias: list[str] = []
    for tabla in tablas:
        filas, cambios, avisos = plan_table(registro, tabla, rutas)
        incidencias.extend(avisos)
        if cambios:
            plan[tabla] = (filas, cambios)

    total = sum(len(c) for _, c in plan.values())
    print("\n=== Plan de normalización de rutas ===")
    print(f"  registro : {args.registry_dir}")
    print(f"  raíz     : {rutas.root}")
    if not total:
        print("\n  Nada que reescribir: no queda ninguna ruta absoluta convertible.")
    for tabla, (_, cambios) in plan.items():
        print(f"\n  [{tabla}] {len(cambios)} celda(s)")
        for cambio in cambios:
            print(f"      L{cambio.line:>3} {cambio.column}")
            print(f"          de : {cambio.current}")
            print(f"          a  : {cambio.proposed}")

    if incidencias:
        print(f"\n=== Incidencias ({len(incidencias)}) ===")
        for texto in incidencias:
            print(f"  - {texto}")

    if not args.apply:
        if total:
            print(
                f"\nNo se ha escrito nada. Son {total} celda(s) en {len(plan)} tabla(s); "
                "vuelve a lanzarlo con --apply para aplicarlo."
            )
        return 2 if incidencias else 0

    if total:
        motivo = check_git_clean(args.registry_dir)
        if motivo is not None and not args.allow_dirty:
            print(
                f"\nERROR: el registro tiene cambios sin confirmar en git: {motivo}\n"
                "Confirma o descarta esos cambios, o pasa --allow-dirty.",
                file=sys.stderr,
            )
            return 1
        if motivo is not None:
            logger.warning("Se escribe con el registro sucio en git: %s", motivo)

    fallos: list[str] = []
    for tabla, (previas, cambios) in plan.items():
        fallos.extend(apply_table(registro, tabla, previas, cambios))

    if fallos:
        print(f"\n=== Fallos ({len(fallos)}) ===", file=sys.stderr)
        for texto in fallos:
            print(f"  - {texto}", file=sys.stderr)
        return 2

    if total:
        print(
            f"\nReescritas {total} celda(s) en {len(plan)} tabla(s), verificadas contra el "
            "fichero releído."
        )
        logger.info("Reescritas %d celdas de ruta en %d tablas.", total, len(plan))
    else:
        print("\nNo ha hecho falta reescribir nada.")
    return 2 if incidencias else 0


if __name__ == "__main__":
    sys.exit(main())
