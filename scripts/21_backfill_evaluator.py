#!/usr/bin/env python
"""Rellena la columna ``evaluator`` de quality_metrics en las filas que ya estaban inscritas.

Las filas antiguas solo llevan el evaluador en el prefijo ``evaluator=`` de notes. El script solo
escribe ``evaluator`` cuando está vacía y no toca nada más. Si una fila no tiene prefijo, si la
columna discrepa de notes o si la cabecera no es la esperada, aborta. Por defecto solo muestra el
plan (``--apply`` para escribir), exige el registro limpio en git (salvo ``--allow-dirty``) y
después de escribir relee el fichero y lo compara celda a celda, restaurándolo si hay diferencias.

Códigos de salida: 0 si no hay nada que hacer, se muestra el plan o se aplica y verifica; 1 si hay
un error de uso o de entorno; 2 si hay incidencias o falla la verificación.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import VALID_EVALUATORS, ExperimentRegistry  # noqa: E402
from tfm_edge.common.registry_guards import check_git_clean, read_header  # noqa: E402
from tfm_edge.evaluation.quality_notes import parse_evaluator_from_notes  # noqa: E402

logger = get_logger(__name__)

TABLE = "quality_metrics"


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos."""
    parser = argparse.ArgumentParser(
        description=(
            "Rellena la columna 'evaluator' de quality_metrics.csv a partir del prefijo "
            "'evaluator=' de 'notes'."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio del registro. Por defecto, results/registry.",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        default=None,
        metavar="RUTA",
        help="JSON de clases. Por defecto, configs/dataset/sh17_classes.json.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Escribe los cambios. Sin esta bandera solo se muestra el plan.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permite aplicar aunque el registro tenga cambios sin confirmar en git.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    return parser


def plan_backfill(
    rows: list[dict[str, str]],
) -> tuple[dict[int, str], list[str]]:
    """Decide el evaluador de cada fila y devuelve ``(cambios, incidencias)``."""
    cambios: dict[int, str] = {}
    incidencias: list[str] = []

    for indice, fila in enumerate(rows):
        linea = indice + 2  # +1 por la cabecera y +1 porque las líneas empiezan en 1
        etiqueta = f"L{linea} ({fila.get('experiment_id', '?')})"
        actual = (fila.get("evaluator") or "").strip()
        desde_notas = parse_evaluator_from_notes(fila.get("notes", ""))

        if not desde_notas:
            incidencias.append(f"{etiqueta}: 'notes' no declara evaluador, así que no se rellena.")
            continue
        if desde_notas not in VALID_EVALUATORS:
            incidencias.append(
                f"{etiqueta}: 'notes' declara evaluator='{desde_notas}', que no "
                f"está en VALID_EVALUATORS ({', '.join(VALID_EVALUATORS)})."
            )
            continue
        if actual and actual != desde_notas:
            incidencias.append(
                f"{etiqueta}: la columna dice '{actual}' y 'notes' dice "
                f"'{desde_notas}'. Dos fuentes que discrepan sobre el mismo "
                "número: se para, no se elige una."
            )
            continue
        if actual == desde_notas:
            continue  # ya migrada
        cambios[indice] = desde_notas

    return cambios, incidencias


def main() -> int:
    """Punto de entrada del script."""
    args = build_parser().parse_args()
    setup_logging("21_backfill_evaluator", level=args.log_level)

    try:
        rutas = get_project_paths()
        clases = load_class_names(
            args.classes or rutas.root / "configs" / "dataset" / "sh17_classes.json"
        )
        registro = ExperimentRegistry(args.registry_dir or rutas.results_registry, clases)
    except TFMEdgeError as error:
        logger.error("No se ha podido preparar el registro: %s", error)
        return 1

    destino = registro.path_for(TABLE)
    if not destino.is_file():
        logger.error("No existe '%s'.", destino)
        return 1

    esquema = registro.columns_for(TABLE)
    if "evaluator" not in esquema:
        logger.error(
            "El esquema de %s no tiene columna 'evaluator'.",
            TABLE,
        )
        return 1
    esquema_previo = tuple(c for c in esquema if c != "evaluator")

    cabecera = read_header(destino)
    if cabecera == esquema:
        logger.info("La cabecera ya incluye 'evaluator'.")
    elif cabecera == esquema_previo:
        logger.info("La cabecera es la anterior a 'evaluator'; se añadirá la columna.")
    else:
        logger.error(
            "La cabecera en disco no es el esquema (%d columnas) ni el esquema sin 'evaluator' "
            "(%d): tiene %d.",
            len(esquema),
            len(esquema_previo),
            len(cabecera),
        )
        return 2

    filas = registro.read_rows(TABLE)
    previas = [dict(fila) for fila in filas]
    cambios, incidencias = plan_backfill(filas)

    for incidencia in incidencias:
        logger.error("%s", incidencia)
    if incidencias:
        print(
            f"\nABORTADO: {len(incidencias)} fila(s) sin evaluador recuperable. "
            "No se ha escrito nada.",
            file=sys.stderr,
        )
        return 2

    reparto: dict[str, int] = {}
    for evaluador in cambios.values():
        reparto[evaluador] = reparto.get(evaluador, 0) + 1

    print(f"Tabla: {destino}")
    print(f"Filas: {len(filas)}")
    print(f"A rellenar: {len(cambios)}  ({reparto or 'ninguna'})")
    print(f"Ya migradas: {len(filas) - len(cambios)}")

    if not cambios and cabecera == esquema:
        print("\nNada que hacer.")
        return 0

    if not args.apply:
        for indice in sorted(cambios):
            linea = indice + 2
            print(f"  L{linea}: {filas[indice]['experiment_id']}: {cambios[indice]}")
        print("\nPlan mostrado. Nada escrito. Pasa --apply para aplicarlo.")
        return 0

    motivo = None if args.allow_dirty else check_git_clean(registro.registry_dir)
    if motivo:
        logger.error(
            "El registro no está limpio en git, así que la reescritura no sería " "reversible: %s",
            motivo,
        )
        print(
            "\nABORTADO. Confirma o descarta esos cambios, o pasa --allow-dirty " "a sabiendas.",
            file=sys.stderr,
        )
        return 2

    for indice, evaluador in cambios.items():
        filas[indice]["evaluator"] = evaluador

    # Escritura atómica del registro: temporal y os.replace.
    registro._write_rows_atomic(TABLE, filas)  # noqa: SLF001
    logger.info("Reescrito '%s' con %d celdas rellenadas.", destino, len(cambios))

    # Se relee y se compara celda a celda con lo planificado.
    releidas = registro.read_rows(TABLE)
    fallos: list[str] = []
    if len(releidas) != len(previas):
        fallos.append(f"el fichero tiene {len(releidas)} filas y tenía {len(previas)}")
    else:
        for indice, (antes, ahora) in enumerate(zip(previas, releidas)):
            linea = indice + 2
            esperado = cambios.get(indice, (antes.get("evaluator") or "").strip())
            if (ahora.get("evaluator") or "").strip() != esperado:
                fallos.append(
                    f"L{linea}: evaluator='{ahora.get('evaluator')}', esperado '{esperado}'"
                )
            for columna in esquema_previo:
                if (antes.get(columna) or "") != (ahora.get(columna) or ""):
                    fallos.append(
                        f"L{linea}/{columna}: '{antes.get(columna)}' -> "
                        f"'{ahora.get(columna)}' (no estaba previsto)"
                    )

    if fallos:
        logger.error("La verificación posterior ha fallado; se restaura la imagen previa.")
        for fallo in fallos[:20]:
            logger.error("  %s", fallo)
        registro._write_rows_atomic(TABLE, previas)  # noqa: SLF001
        print("\nRESTAURADO el estado previo. Revisa el log.", file=sys.stderr)
        return 2

    print(f"\nAplicado y verificado: {len(cambios)} celdas rellenadas en {destino}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
