#!/usr/bin/env python
"""Escribe el análisis del capítulo 5: cobertura, degradación, compromisos y selección.

Lee results/registry/ y no mide nada. No afirma diferencias por debajo del suelo de medida: con los
tres niveles de compresión de la campaña el punto de ruptura no es determinable, algunas
configuraciones quedan en el margen de las restricciones y cada curva indica si es un punto, un
segmento o una curva. Los escenarios de selección (100 ms, 500 MiB, 0,30 de mAP) son ilustrativos y
están en selection.example_scenarios(). El volcado no lleva fecha, así que ``--on-existing
overwrite`` seguido de ``git status`` indica si ha cambiado algo.

Códigos de salida: 0 si se calcula y escribe el análisis, 1 si hay un error de configuración o el
volcado ya existe sin indicar qué hacer, 2 si los datos no permiten calcularlo.
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

from tfm_edge.analysis import aggregate  # noqa: E402
from tfm_edge.analysis import degradation  # noqa: E402
from tfm_edge.analysis import pareto  # noqa: E402
from tfm_edge.analysis import selection  # noqa: E402
from tfm_edge.analysis import tradeoffs  # noqa: E402
from tfm_edge.analysis import floors as floors_mod  # noqa: E402
from tfm_edge.analysis.splits import analysis_quality_rows  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402

logger = get_logger(__name__)

#: Un solo volcado con las cuatro secciones, que comparten suelos.
DUMP_NAME = "results_analysis.json"

#: Ejes cuyo suelo se mide: los de las restricciones, las eficiencias y la ordenación, todos a la
#: vez.
FLOOR_AXES: tuple[str, ...] = (
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_host_memory_mb",
    "artifact_size_bytes",
    "map_50",
    "map_50_95",
)

#: Ejes que se llevan a los puntos de selección.
SELECTION_OBJECTIVES: tuple[pareto.Objective, ...] = (
    pareto.Objective("mAP@50", "map_50", "quality", "max"),
    pareto.Objective("mAP@50-95", "map_50_95", "quality", "max"),
    pareto.Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
    pareto.Objective("latencia p95", "latency_p95_ms", "deployment", "min"),
    pareto.Objective("memoria pico", "peak_host_memory_mb", "deployment", "min"),
    pareto.Objective("tamaño del artefacto", "artifact_size_bytes", "deployment", "min"),
)

#: Columnas que se llevan a los puntos de las curvas de compromiso.
CURVE_COLUMNS: tuple[str, ...] = (
    "map_50",
    "map_50_95",
    "latency_p50_ms",
    "latency_p95_ms",
    "peak_host_memory_mb",
    "artifact_size_bytes",
)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="29_analyze_results.py",
        description=(
            "Deriva del registro la cobertura de la campaña, la degradación por "
            "clase y por tamaño, las curvas de compromiso por método y la "
            "selección por restricciones, y lo vuelca todo junto."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # El análisis completo\n"
            "  python scripts/29_analyze_results.py\n\n"
            "  # Comprobar que nada ha cambiado tras añadir medidas\n"
            "  python scripts/29_analyze_results.py --on-existing overwrite\n"
            "  git status --short results/raw/analysis/\n\n"
            "  # Ver qué saldría, sin escribir nada\n"
            "  python scripts/29_analyze_results.py --dry-run\n"
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
        help="Dónde volcar el análisis (por defecto: results/raw/analysis).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Dispositivo de la selección. Por defecto, el único medido; si hay varios, hay que "
            "elegir."
        ),
    )
    parser.add_argument(
        "--size-criterion",
        default=degradation.DEFAULT_SIZE_CRITERION,
        choices=("inference_res", "original_res"),
        help=(
            "Criterio de tamaño de objeto para el desglose de la degradación "
            f"(por defecto: {degradation.DEFAULT_SIZE_CRITERION})."
        ),
    )
    parser.add_argument(
        "--on-existing",
        default="fail",
        choices=("fail", "skip", "overwrite"),
        help=("Qué hacer si el volcado ya existe (por defecto: fail)."),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Calcula y resume por consola sin escribir el volcado. El log sí se escribe."),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: INFO).",
    )
    return parser


def elegir_dispositivo(deployment_rows: list[dict[str, str]], pedido: str | None) -> str:
    """Dispositivo de la selección. Falla si el pedido no está medido o si hay varios y no se
    elige."""
    presentes = sorted({fila["device_id"] for fila in deployment_rows})
    if pedido is not None:
        if pedido not in presentes:
            raise ValueError(
                f"No hay medidas de despliegue en '{pedido}'. Los dispositivos con "
                f"medidas son: {presentes}."
            )
        return pedido
    if not presentes:
        raise ValueError("No hay ninguna fila de despliegue utilizable.")
    if len(presentes) > 1:
        raise ValueError(f"Hay medidas en {presentes}; elige el dispositivo con --device.")
    return presentes[0]


def calcular(
    registry_dir: Path, project_root: Path, dispositivo: str | None, size_criterion: str
) -> dict[str, Any]:
    """Calcula las cuatro secciones del análisis."""
    registro = ExperimentRegistry(registry_dir)
    experimentos, experimentos_descartados = registro.valid_rows("experiments")
    calidad, calidad_descartada = analysis_quality_rows(registro)
    despliegue, descartes = registro.latency_comparable_rows()

    if not calidad:
        raise ValueError(
            "El registro de calidad no tiene ninguna fila utilizable. Sin métricas "
            "de calidad no hay degradación que calcular."
        )

    descartadas = {
        "experiment_rows_discarded_status": experimentos_descartados,
        "quality_rows_discarded_status": calidad_descartada,
        "deployment_rows_discarded": dict(sorted(descartes.items())),
    }
    suelos = floors_mod.measure_floors(despliegue, FLOOR_AXES, project_root)

    cobertura = aggregate.coverage_report(experimentos, calidad, despliegue, descartadas)
    filas_degradacion = degradation.degradation_rows(
        experimentos, calidad, despliegue, suelos, size_criterion=size_criterion
    )
    curvas = tradeoffs.family_curves(experimentos, calidad, despliegue, CURVE_COLUMNS)

    analisis: dict[str, Any] = {
        "inputs": {
            "experiment_rows_used": len(experimentos),
            "quality_rows_used": len(calidad),
            "deployment_rows_used": len(despliegue),
            "size_criterion": size_criterion,
            **descartadas,
        },
        "floors": suelos.to_dict(),
        "coverage": cobertura.to_dict(),
        "degradation": {
            "rows": [fila.to_dict() for fila in filas_degradacion],
            "breakpoints": _breakpoints(curvas),
        },
        "tradeoffs": {"curves": [curva.to_dict() for curva in curvas]},
        "selection": {},
    }

    if despliegue:
        elegido = elegir_dispositivo(despliegue, dispositivo)
        puntos = pareto.build_points(calidad, despliegue, SELECTION_OBJECTIVES)
        escenarios = {}
        for escenario in selection.example_scenarios():
            resultado = selection.select(
                puntos,
                escenario.constraints,
                suelos,
                elegido,
                escenario.order_by,
                escenario.order_direction,
            )
            escenarios[escenario.name] = {
                "description": escenario.description,
                **resultado.to_dict(),
            }
        analisis["selection"] = {
            "device_id": elegido,
            "num_points": len([p for p in puntos if p.device_id == elegido]),
            "limits_are_not_a_result": (
                "Los límites de estos escenarios los elige el autor para ilustrar el "
                "apartado 6.7. No salen de ninguna medida ni de ningún requisito del "
                "estudio, y cambiarlos cambia la lista de configuraciones viables sin "
                "que nada esté mal."
            ),
            "scenarios": escenarios,
        }
    else:
        analisis["selection"] = {
            "unavailable": (
                "No hay ninguna fila de despliegue utilizable, así que no se puede "
                "responder a ninguna restricción de latencia o de memoria."
            )
        }

    return analisis


def _breakpoints(curvas: tuple[tradeoffs.FamilyCurve, ...]) -> dict[str, Any]:
    """Punto de ruptura de cada curva con niveles suficientes."""
    resultado: dict[str, Any] = {}
    for curva in curvas:
        pares = [
            (punto.level, punto.values["map_50_95"])
            for punto in curva.points
            if "map_50_95" in punto.values
        ]
        if not pares:
            continue
        resultado[curva.label] = degradation.find_breakpoint(pares).to_dict()
    return dict(sorted(resultado.items()))


def resumir(analisis: dict[str, Any]) -> list[str]:
    """Líneas del resumen para el usuario."""
    lineas: list[str] = []

    cobertura = analisis["coverage"]
    recuentos = cobertura["counts"]
    lineas.append(
        f"Cobertura: {recuentos['cells_present']} de {recuentos['cells_expected']} "
        f"celdas esperadas, {recuentos['cells_missing']} ausentes en "
        f"{recuentos['experiments_with_gaps']} experimentos."
    )
    por_experimento: dict[str, list[str]] = {}
    for hueco in cobertura["gaps"]:
        etiqueta = "/".join(hueco["context"].values())
        por_experimento.setdefault(hueco["experiment_id"], []).append(
            f"{hueco['table']}:{etiqueta}"
        )
    for experimento, faltan in sorted(por_experimento.items()):
        lineas.append(f"    {experimento}: faltan {len(faltan)}")
    independencia = cobertura["device_independence"]
    lineas.append(
        f"    mAP frente al dispositivo: {independencia['checked']} configuraciones evaluadas en "
        "más de una máquina, "
        f"{'SE DETECTA DEPENDENCIA' if independencia['dependence_detected'] else 'sin dependencia'}."
    )
    if cobertura["inconsistencies"]:
        lineas.append(f"    INCOHERENCIAS del registro: {len(cobertura['inconsistencies'])}")
        for incoherencia in cobertura["inconsistencies"]:
            lineas.append(f"        [{incoherencia['kind']}] {incoherencia['detail']}")

    filas = analisis["degradation"]["rows"]
    sin_certificar = sum(
        1
        for fila in filas
        for eficiencia in fila["efficiency"].values()
        if eficiencia["points_per_factor"] is None
    )
    total_eficiencias = sum(len(fila["efficiency"]) for fila in filas)
    lineas.append("")
    lineas.append(
        f"Degradación: {len(filas)} configuraciones comparadas con su referencia. "
        f"Eficiencias calculadas: {total_eficiencias - sin_certificar} de "
        f"{total_eficiencias} ({sin_certificar} sin certificar por el suelo)."
    )
    determinables = [
        etiqueta
        for etiqueta, ruptura in analisis["degradation"]["breakpoints"].items()
        if ruptura["determinable"]
    ]
    lineas.append(
        f"    Puntos de ruptura determinables: {len(determinables)} de "
        f"{len(analisis['degradation']['breakpoints'])}."
    )

    curvas = analisis["tradeoffs"]["curves"]
    formas: dict[str, int] = {}
    for curva in curvas:
        formas[curva["shape"]] = formas.get(curva["shape"], 0) + 1
    lineas.append("")
    lineas.append(
        f"Compromisos: {len(curvas)} curvas "
        f"({', '.join(f'{n} {f}' for f, n in sorted(formas.items()))})."
    )

    seleccion = analisis["selection"]
    lineas.append("")
    if "unavailable" in seleccion:
        lineas.append(f"Selección: no disponible. {seleccion['unavailable']}")
    else:
        lineas.append(f"Selección en {seleccion['device_id']} ({seleccion['num_points']} puntos):")
        for nombre, escenario in seleccion["scenarios"].items():
            recuento = escenario["counts"]
            lineas.append(
                f"    {nombre}: {recuento['viable']} viables "
                f"({recuento['borderline']} en el margen), {recuento['excluded']} fuera."
            )
    return lineas


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("29_analyze_results", level=args.log_level)

    try:
        rutas = get_project_paths()
        registry_dir = args.registry_dir or rutas.results_registry
        output_dir = args.output_dir or (rutas.results_raw / "analysis")
        destino = output_dir / DUMP_NAME

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

        analisis = calcular(registry_dir, rutas.root, args.device, args.size_criterion)

    except TFMEdgeError as exc:
        logger.error("%s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logger.error("Error de configuración o de entrada/salida: %s", exc)
        return 1

    for linea in resumir(analisis):
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
