"""Protección de los volcados de results/raw frente a sobrescrituras.

Si el destino existe, el script falla salvo que se pase --overwrite-dump. No se mira el contenido.
Los scripts de entrenamiento tienen además su propia comprobación para --no-evaluate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = [
    "BANDERA",
    "DUMP_GUARD_BY_SCRIPT",
    "OVERWRITE_DUMP_FLAG",
    "add_overwrite_dump_argument",
    "refuse_to_overwrite_dump",
]

#: Nombre de la bandera, compartido por todos los parsers.
OVERWRITE_DUMP_FLAG = "--overwrite-dump"

#: Cómo protege su escritura cada script que vuelca en results/raw. tests/test_evidence_guard.py lo
#: contrasta con los scripts.
BANDERA = "se niega con --overwrite-dump, antes de medir"

DUMP_GUARD_BY_SCRIPT: dict[str, str] = {
    # Los que miden y se niegan con la bandera común.
    "06_evaluate_coco.py": BANDERA,
    "07_benchmark.py": BANDERA,
    "09_sensitivity.py": BANDERA,
    "11_evaluate_artifact.py": BANDERA,
    "17_register_quality.py": BANDERA,
    # Reconstruye desde las detecciones por instancia, que no se versionan, así que su volcado no se
    # puede regenerar en un clon limpio.
    "34_register_error_decomposition.py": BANDERA,
    "measure_memory_envelope.py": BANDERA,
    "measure_training_throughput.py": BANDERA,
    # Miran el contenido porque con --no-evaluate escriben menos.
    "13_prune.py": "guarda propia: se niega si esta ejecución no midió y la anterior sí",
    "16_qat.py": "guarda propia: se niega si esta ejecución no midió y la anterior sí",
    "20_distill.py": "guarda propia: se niega si esta ejecución no midió y la anterior sí",
    # Derivados: se recalculan desde volcados protegidos.
    "12_score_resolution.py": "derivado de dos volcados de artifact_eval",
    "23_operating_curve.py": "derivado del registro y de los volcados de calidad",
    "24_operating_analysis.py": "derivado del volcado del paso 23",
    "25_matching_ranking.py": "derivado: empareja filas del registro, no mide nada",
    "26_analyze_pareto.py": "derivado: agrega filas del registro en el volcado del análisis",
    "27_build_pareto_figures.py": "derivado del volcado del paso 26",
    "29_analyze_results.py": "derivado: agrega filas del registro para las figuras del 30",
    "30_build_results_figures.py": "derivado del volcado del paso 29",
    "31_register_export_cost.py": "derivado de los volcados de exportación",
    # Sin data/ falta la muestra de SH17, que se recalcula.
    "35_build_intro_figures.py": "derivado del registro, de las anotaciones y de las atribuciones",
    # Los que leen results/raw y no escriben nada dentro.
    "14_import_colab_registry.py": "no escribe bajo results/raw: comprueba que el destino no "
    "existe",
    "15_normalize_pruned_layout.py": "no escribe bajo results/raw: mueve pesos, y solo con --apply",
    "28_build_report.py": "no escribe bajo results/raw: lee los volcados de análisis",
    "32_distill_seed_variance.py": "no escribe bajo results/raw: lee los volcados de destilación",
    "benchmark_grid.py": "no escribe bajo results/raw: invoca al paso 07, que sí se protege",
}


def add_overwrite_dump_argument(parser: argparse.ArgumentParser, what: str) -> None:
    """Añade al parser la bandera para sobrescribir un volcado. ``what`` describe el volcado en la
    ayuda.
    """
    parser.add_argument(
        OVERWRITE_DUMP_FLAG,
        action="store_true",
        help=(f"Permite sobrescribir {what} si ya existe."),
    )


def refuse_to_overwrite_dump(destination: Path, *, overwrite: bool, what: str) -> None:
    """Lanza FileExistsError si ``destination`` existe y no se ha pedido sobrescribirlo."""
    if not destination.exists() or overwrite:
        return
    raise FileExistsError(
        f"ya existe '{destination}' ({what}). Bórralo a mano o pasa {OVERWRITE_DUMP_FLAG}."
    )
