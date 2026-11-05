#!/usr/bin/env python
"""Calcula la frontera de Pareto del registro teniendo en cuenta el suelo de medida.

Para cada vista y agrupación calcula la frontera estricta y la robusta, que solo cuenta ventajas por
encima del suelo; ``dominations_lost_to_floor`` lista las dominaciones que el suelo no sostiene. El
suelo se mide en cada ejecución con los pares de control (analysis.floors); map_50 no tiene suelo.
No mide, no escribe en el registro ni dibuja. El volcado no lleva fecha, así que con el mismo
registro sale idéntico y ``--on-existing overwrite`` seguido de ``git status`` indica si algo ha
cambiado.

Códigos de salida: 0 si la frontera se calcula y se escribe, 1 si hay un error de configuración o el
volcado ya existe sin indicar qué hacer, 2 si los datos no permiten calcularla.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.analysis import floors as floors_mod  # noqa: E402
from tfm_edge.analysis import pareto  # noqa: E402
from tfm_edge.analysis.splits import analysis_quality_rows  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402

logger = get_logger(__name__)

#: Un solo volcado con todas las vistas, que comparten suelos.
DUMP_NAME = "pareto_analysis.json"

#: Vistas que se calculan: mAP, p95 y memoria, y dos sobre la latencia mediana.
VIEWS: dict[str, tuple[pareto.Objective, ...]] = pareto.named_views()

#: Agrupaciones: por dispositivo y motor, y solo por dispositivo para comparar motores.
GROUPINGS: tuple[tuple[str, ...], ...] = (
    ("device_id", "runtime"),
    ("device_id",),
)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="26_analyze_pareto.py",
        description=(
            "Calcula la frontera de Pareto del registro en dos modos, la clásica y otra que "
            "solo admite ventajas por encima del suelo de medida, y escribe ambas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # El análisis completo\n"
            "  python scripts/26_analyze_pareto.py\n\n"
            "  # Comprobar que nada ha cambiado tras añadir medidas\n"
            "  python scripts/26_analyze_pareto.py --on-existing overwrite\n"
            "  git status --short results/raw/pareto/\n\n"
            "  # Solo la vista de la especificación, sin escribir nada\n"
            "  python scripts/26_analyze_pareto.py --view spec_3d_map5095_p95_memoria "
            "--dry-run\n"
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dónde volcar el análisis (por defecto: results/raw/pareto).",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=sorted(VIEWS),
        default=None,
        help=(
            "Vista a calcular; se puede repetir. Por defecto se calculan todas. Con menos "
            "vistas el volcado tendrá menos información."
        ),
    )
    parser.add_argument(
        "--on-existing",
        choices=("fail", "skip", "overwrite"),
        default="fail",
        help=(
            "Qué hacer si el volcado ya existe (por defecto: fail). Con 'overwrite' y 'git "
            "status' se ve si ha cambiado algo."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Calcula e informa por consola sin escribir el volcado. El log sí se escribe en "
            "results/logs/."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: INFO).",
    )
    return parser


def calcular(registry_dir: Path, project_root: Path, vistas: list[str]) -> dict[str, Any]:
    """Calcula el análisis completo a partir del registro."""
    registro = ExperimentRegistry(registry_dir)
    calidad, calidad_descartadas = analysis_quality_rows(registro)
    despliegue, descartes = registro.latency_comparable_rows()

    if not despliegue:
        raise pareto.ParetoError(
            "El registro de despliegue no tiene ninguna fila utilizable. Sin "
            "latencias no hay frontera que calcular."
        )

    analisis: dict[str, Any] = {
        "inputs": {
            "quality_rows_used": len(calidad),
            "quality_rows_discarded_status": calidad_descartadas,
            "deployment_rows_used": len(despliegue),
            "deployment_rows_discarded": dict(sorted(descartes.items())),
        },
        "control_pairs": [
            {
                "name": par.name,
                "kind": par.kind,
                "rationale": par.rationale,
                "evidence": list(par.evidence),
                "contaminated": par.contaminated,
                "contamination_reason": par.contamination_reason,
            }
            for par in floors_mod.default_control_pairs()
        ],
        "views": {},
    }

    for nombre in vistas:
        objetivos = VIEWS[nombre]
        columnas = [o.column for o in objetivos]
        suelos = floors_mod.measure_floors(despliegue, columnas, project_root)
        puntos = pareto.build_points(calidad, despliegue, objetivos)
        if not puntos:
            raise pareto.ParetoError(
                f"La vista '{nombre}' se queda sin puntos: ninguna configuración "
                f"tiene medidos a la vez {columnas}."
            )
        logger.info("Vista '%s': %d puntos completos.", nombre, len(puntos))
        analisis["views"][nombre] = pareto.analyze(puntos, objetivos, suelos, GROUPINGS)

    return analisis


def resumir(analisis: dict[str, Any]) -> list[str]:
    """Líneas del resumen para el usuario."""
    lineas: list[str] = []
    for nombre, vista in analisis["views"].items():
        ejes = ", ".join(o["name"] for o in vista["objectives"])
        lineas.append(f"Vista '{nombre}' ({ejes}), {vista['num_points']} puntos:")
        sin_suelo = [
            eje
            for eje, s in vista["floors"]["floors"].items()
            if s["same_session"] is None and s["cross_session"] is None
        ]
        if sin_suelo:
            lineas.append(
                f"    ejes SIN suelo medido, en los que no se certifica ninguna "
                f"ventaja: {', '.join(sin_suelo)}"
            )
        for agrupacion, grupos in vista["groupings"].items():
            for clave, datos in grupos.items():
                perdidas = len(datos["dominations_lost_to_floor"])
                lineas.append(
                    "    [%s] %s: %d puntos, frontera %d (estricta) / %d (robusta), "
                    "%d dominaciones no sobreviven al suelo, HV=%.4f"
                    % (
                        agrupacion,
                        clave,
                        datos["num_points"],
                        len(datos["frontier_strict"]),
                        len(datos["frontier_robust"]),
                        perdidas,
                        datos["hypervolume_normalized"],
                    )
                )
    return lineas


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("26_analyze_pareto", level=args.log_level)

    try:
        rutas = get_project_paths()
        registry_dir = args.registry_dir or rutas.results_registry
        output_dir = args.output_dir or (rutas.results_raw / "pareto")
        destino = output_dir / DUMP_NAME
        vistas = sorted(set(args.view)) if args.view else sorted(VIEWS)

        if destino.exists() and not args.dry_run:
            if args.on_existing == "skip":
                logger.info("El volcado ya existe y --on-existing=skip: no se toca.")
                print(f"Sin cambios: {destino} ya existe.")
                return 0
            if args.on_existing == "fail":
                logger.error(
                    "El volcado %s ya existe. Usa --on-existing overwrite y después 'git "
                    "status' para ver si ha cambiado algo.",
                    destino,
                )
                return 1

        analisis = calcular(registry_dir, rutas.root, vistas)

    except TFMEdgeError as exc:
        logger.error("%s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logger.error("Error de configuración o de entrada/salida: %s", exc)
        return 1

    resumen = resumir(analisis)
    for linea in resumen:
        print(linea)

    if args.dry_run:
        logger.info("--dry-run: no se ha tocado results/raw/.")
        print(f"\n--dry-run: no se ha escrito {destino}. El log sí está en results/logs/.")
        return 0

    ensure_parent_dir(destino)
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    temporal.write_text(
        json.dumps(analisis, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporal, destino)
    logger.info("Análisis volcado en %s", destino)
    print(f"\nVolcado: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
