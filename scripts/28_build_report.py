#!/usr/bin/env python
"""Genera de una vez las tablas y figuras de la memoria.

Escribe las tablas del capítulo 6 (medidas de la campaña y posición respecto a la frontera) y vuelve
a lanzar los pasos de figuras. Al final enumera lo que la memoria necesita y todavía no se puede
generar. Los números salen del registro con los mismos filtros que el análisis de Pareto, y los
suelos del volcado del paso 26; si registro y volcado no cuentan las mismas filas, se aborta.

Códigos de salida: 0 si se genera lo que se puede generar, 1 si hay un error de configuración o las
tablas ya existen sin indicar qué hacer, 2 si los datos no permiten construir las tablas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.reporting import build_report  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="28_build_report.py",
        description=(
            "Genera las tablas de la memoria y relanza el paso de figuras, y "
            "enumera qué piezas del informe siguen sin poder generarse."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Todo lo que se puede generar\n"
            "  python scripts/28_build_report.py\n\n"
            "  # Solo las tablas, sin tocar las figuras\n"
            "  python scripts/28_build_report.py --skip-figures\n\n"
            "  # Ver qué saldría, sin escribir nada\n"
            "  python scripts/28_build_report.py --dry-run\n\n"
            "  # Todo a directorios temporales, sin tocar results/. Los dos volcados\n"
            "  # se copian porque cada paso deja su manifiesto junto al volcado que lee.\n"
            "  mkdir -p /tmp/p /tmp/a\n"
            "  cp results/raw/pareto/*.json /tmp/p/\n"
            "  cp results/raw/analysis/results_analysis.json /tmp/a/\n"
            "  python scripts/28_build_report.py --tables-dir /tmp/t \\\n"
            "      --figures-dir /tmp/f --analysis-dir /tmp/p \\\n"
            "      --results-analysis-dir /tmp/a --on-existing overwrite\n"
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro. Por defecto, results/registry.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directorio del volcado del paso 26, del que salen suelos y fronteras (por defecto: "
            "results/raw/pareto)."
        ),
    )
    parser.add_argument(
        "--results-analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directorio del volcado del paso 29, que lee el paso 30 (por defecto: "
            "results/raw/analysis)."
        ),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="Destino de las tablas. Por defecto, results/tables.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directorio de figuras. Por defecto, results/figures.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Dispositivo del que se hacen las tablas. Por defecto, el único que "
            "haya en el registro; si hay más de uno, hay que elegir."
        ),
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="No relanzar el paso 27. Útil cuando solo han cambiado las tablas.",
    )
    parser.add_argument(
        "--on-existing",
        choices=("fail", "skip", "overwrite"),
        default="fail",
        help=("Qué hacer si las tablas o figuras ya existen. Por defecto 'fail'."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construirlo todo sin escribir nada, para ver qué saldría.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log.",
    )
    return parser


def elegir_dispositivo(registry_dir: Path, pedido: str | None) -> str:
    """Dispositivo del que se hacen las tablas. Si hay varios, hay que elegirlo con --device."""
    registro = ExperimentRegistry(registry_dir)
    filas, _ = registro.latency_comparable_rows()
    disponibles = sorted({fila["device_id"] for fila in filas})
    if not disponibles:
        raise ValueError(
            f"No hay ninguna fila de despliegue comparable en {registry_dir}: no hay "
            f"nada que tabular."
        )
    if pedido is not None:
        if pedido not in disponibles:
            raise ValueError(
                f"El dispositivo '{pedido}' no está en el registro. Hay: "
                f"{', '.join(disponibles)}."
            )
        return pedido
    if len(disponibles) > 1:
        raise ValueError(
            f"El registro tiene {len(disponibles)} dispositivos ({', '.join(disponibles)}); "
            "elige uno con --device."
        )
    return disponibles[0]


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("28_build_report", level=args.log_level)

    try:
        rutas = get_project_paths()
        registry_dir = args.registry_dir or rutas.results_registry
        analysis_dir = args.analysis_dir or (rutas.results_raw / "pareto")
        results_analysis_dir = args.results_analysis_dir or (rutas.results_raw / "analysis")
        tables_dir = args.tables_dir or rutas.results_tables
        figures_dir = args.figures_dir or rutas.results_figures
        dispositivo = elegir_dispositivo(registry_dir, args.device)

        construidas = build_report.build_tables(registry_dir, analysis_dir, dispositivo)
        existentes = [
            tabla.basename
            for tabla in construidas
            if (tables_dir / f"{tabla.basename}.tex").exists()
        ]
        if existentes and not args.dry_run:
            if args.on_existing == "skip":
                logger.info("Ya existen %d tabla(s) y --on-existing=skip.", len(existentes))
                print(f"Sin cambios: ya existen {len(existentes)} tabla(s).")
                return 0
            if args.on_existing == "fail":
                logger.error(
                    "Ya existen estas tablas: %s. Usa --on-existing overwrite y después 'git "
                    "status' para ver si han cambiado.",
                    ", ".join(existentes),
                )
                return 1

        manifiesto = build_report.build(
            rutas.root,
            registry_dir,
            analysis_dir,
            tables_dir,
            figures_dir,
            dispositivo,
            results_analysis_dir=results_analysis_dir,
            skip_figures=args.skip_figures,
            on_existing="overwrite" if args.on_existing == "overwrite" else "skip",
            dry_run=args.dry_run,
        )
    except TFMEdgeError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logger.error("Error de configuración o de entrada/salida: %s", exc)
        return 1

    for linea in build_report.summarize(manifiesto):
        print(linea)

    if args.dry_run:
        logger.info("--dry-run: no se ha escrito ninguna tabla ni el manifiesto.")
        print(
            "\n--dry-run: no se ha escrito ninguna tabla, ninguna figura ni el "
            "manifiesto. El log sí está en results/logs/."
        )
        return 0

    destino = build_report.write_manifest(manifiesto, analysis_dir)
    print(f"\n{len(manifiesto['tables'])} tabla(s) en {tables_dir} (LaTeX, XLSX y CSV).")
    print(f"Manifiesto: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
