#!/usr/bin/env python
"""Lee la campaña de semillas del piloto de destilación y emite su veredicto.

Empareja cada piloto con el control de su semilla, calcula diferencias emparejadas, dispersión e
intervalo al 95 % y aplica la regla de seed_variance. Mide la sensibilidad a la inicialización de
los adaptadores con los datos fijos, no la variabilidad entre entrenamientos. Solo escribe el
informe en results/raw/analysis/ (nada con --dry-run).

Uso: python scripts/32_distill_seed_variance.py python scripts/32_distill_seed_variance.py --dry-run
python scripts/32_distill_seed_variance.py --raw-dir /ruta/traida/de/drive
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tfm_edge.analysis.splits import select_analysis_split  # noqa: E402
from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.compression.distillation.seed_variance import (  # noqa: E402
    SeedCampaignError,
    attach_environment_ids,
    load_distillation_runs,
    pair_by_seed,
    read_seed_campaign,
)

logger = get_logger(__name__)

#: Nombre fijo del informe, para que regenerarlo sustituya al anterior.
NOMBRE_INFORME = "distillation_seed_variance.json"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de la línea de órdenes."""
    parser = argparse.ArgumentParser(
        description=(
            "Lee la campaña de semillas del piloto de destilación: diferencias emparejadas, "
            "dispersión entre semillas y veredicto."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "El criterio de lectura está fijado en "
            "src/tfm_edge/compression/distillation/seed_variance.py, escrito antes "
            "de ver los resultados. Este script no decide nada: lo aplica."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directorio de volcados de destilación. Por defecto, "
            "results/raw/distillation del repositorio."
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directorio del registro. Por defecto, results/registry.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Dónde dejar el informe. Por defecto, results/raw/analysis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calcula e imprime, pero NO escribe el informe en disco.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    return parser


def volcar_atomico(destino: Path, contenido: dict) -> None:
    """Escribe el informe JSON con un fichero temporal y os.replace."""
    ensure_dir(destino.parent)
    descriptor, temporal = tempfile.mkstemp(dir=str(destino.parent), suffix=".tmp")
    with os.fdopen(descriptor, "w", encoding="utf-8") as fichero:
        json.dump(contenido, fichero, indent=2, ensure_ascii=False)
    os.replace(temporal, destino)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args(argv)
    setup_logging("32_distill_seed_variance", level=args.log_level)
    rutas = get_project_paths()

    raw_dir = args.raw_dir or rutas.results / "raw" / "distillation"
    registry_dir = args.registry_dir or rutas.results / "registry"
    output_dir = args.output_dir or rutas.results / "raw" / "analysis"

    logger.info("Volcados     : %s", raw_dir)
    logger.info("Registro     : %s", registry_dir)

    try:
        ejecuciones = load_distillation_runs(raw_dir)
        clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
        registro = ExperimentRegistry(registry_dir, clases)
        # Solo val: las filas de test de los mismos experimentos tienen otra cifra.
        calidad = select_analysis_split(registro.read_rows("quality_metrics"))
        ejecuciones = attach_environment_ids(ejecuciones, calidad)
        parejas = pair_by_seed(ejecuciones)
    except (SeedCampaignError, TFMEdgeError) as error:
        logger.error("%s", error)
        return 1

    semillas_a_medias = sorted({e.seed_base for e in ejecuciones} - {p.seed for p in parejas})
    if semillas_a_medias:
        logger.warning(
            "Semillas con piloto o control pero no los dos, omitidas: %s",
            semillas_a_medias,
        )

    print()
    print("=" * 78)
    print("CAMPAÑA DE SEMILLAS DEL PILOTO DE DESTILACIÓN")
    print("=" * 78)
    print()
    print(
        f"{'semilla':>8}  {'mAP@50-95 pil.':>15}  {'control':>10}  {'delta':>11}  "
        f"{'delta@50':>11}  {'entorno':>9}"
    )
    print("-" * 78)
    for pareja in parejas:
        entorno = pareja.environment_state
        print(
            f"{pareja.seed:>8}  {pareja.pilot.map_50_95:>15.6f}  "
            f"{pareja.control.map_50_95:>10.6f}  {pareja.delta_50_95:>+11.6f}  "
            f"{pareja.delta_50:>+11.6f}  {entorno:>9}"
        )
    print("-" * 78)

    try:
        lectura = read_seed_campaign(parejas)
    except SeedCampaignError as error:
        logger.error("%s", error)
        return 1

    print()
    print(f"Semillas leídas          : {lectura.n_seeds}  {list(lectura.seeds)}")
    print(f"Diferencia media         : {lectura.mean_delta:+.6f}")
    print(f"Dispersión entre semillas: {lectura.sd_delta:.6f}  (desv. típica muestral)")
    print(
        f"IC 95 % (t_{lectura.n_seeds - 1}={lectura.t_critical})    : "
        f"{lectura.ci_low:+.6f} a {lectura.ci_high:+.6f}"
        f"{'   CRUZA EL CERO' if lectura.ci_crosses_zero else ''}"
    )
    print()
    print("--- Los controles ---")
    print(f"mAP@50-95 medio de los controles     : {lectura.control_mean:.6f}")
    print(f"Desviación típica de los controles   : {lectura.control_sd:.6f}")
    if lectura.controls_identical:
        print(
            "    Los controles son la misma ejecución: con alpha=0 los adaptadores\n"
            "    son inertes y son lo único que la semilla cambia. Cero por\n"
            "    construcción, no por estabilidad de la receta."
        )
    print(f"Suelo entre evaluadores                : {lectura.process_floor:.6f}")
    print()
    print("--- Predicciones fechadas, escritas antes de medir ---")
    print(f"[1] {lectura.floor_prediction}")
    if lectura.floor_prediction_testable:  # pragma: no cover - no ocurre en esta receta
        print(f"    SE CUMPLE: {lectura.floor_is_lower_bound}")
    else:
        print("    NO CONTRASTABLE.")
        print(f"    {lectura.floor_prediction_status}")
    print(f"[2] {lectura.tail_prediction}")
    print(f"    SE CUMPLE: {lectura.tail_prediction_holds}")
    print()
    print("=" * 78)
    print(f"VEREDICTO: {lectura.verdict}")
    print("=" * 78)
    print(lectura.statement)
    print()
    for aviso in lectura.caveats:
        print(f"  - {aviso}")
    if not lectura.all_pairs_same_environment:
        distintas = [p.seed for p in parejas if p.environment_state == "DISTINTO"]
        sin_dato = [p.seed for p in parejas if p.environment_state == "sin dato"]
        if distintas:
            print(
                f"  - Las semillas {distintas} tienen piloto y control medidos en "
                f"entornos DISTINTOS:\n"
                f"    su diferencia mezcla el efecto con el cambio de entorno."
            )
        if sin_dato:
            print(
                f"  - De las semillas {sin_dato} no consta el environment_id de alguna de las dos\n"
                f"    filas; no se ha podido comprobar si coinciden."
            )
    print()

    informe = {
        "reading": lectura.to_dict(),
        "pairs": [
            {
                "seed": pareja.seed,
                "pilot": {
                    "experiment_id": pareja.pilot.experiment_id,
                    "map_50": pareja.pilot.map_50,
                    "map_50_95": pareja.pilot.map_50_95,
                    "alpha": pareja.pilot.alpha,
                    "environment_id": pareja.pilot.environment_id,
                    "training_source": pareja.pilot.training_source,
                    "preflight_recorded": pareja.pilot.preflight_recorded,
                    "seed_branch": pareja.pilot.seed_branch,
                    "adapter_initial_norms": dict(pareja.pilot.adapter_initial_norms),
                },
                "control": {
                    "experiment_id": pareja.control.experiment_id,
                    "map_50": pareja.control.map_50,
                    "map_50_95": pareja.control.map_50_95,
                    "alpha": pareja.control.alpha,
                    "environment_id": pareja.control.environment_id,
                    "training_source": pareja.control.training_source,
                    "preflight_recorded": pareja.control.preflight_recorded,
                    "seed_branch": pareja.control.seed_branch,
                    "adapter_initial_norms": dict(pareja.control.adapter_initial_norms),
                },
                "delta_map_50_95": pareja.delta_50_95,
                "delta_map_50": pareja.delta_50,
                "same_environment": pareja.same_environment,
                "environment_state": pareja.environment_state,
                "effect_is_in_the_tail": pareja.effect_is_in_the_tail,
            }
            for pareja in parejas
        ],
        "seeds_incomplete": semillas_a_medias,
        "source_dumps": sorted(str(e.dump_path.name) for e in ejecuciones),
    }

    if args.dry_run:
        logger.info("--dry-run: no se escribe el informe.")
        return 0

    destino = output_dir / NOMBRE_INFORME
    volcar_atomico(destino, informe)
    logger.info("Informe escrito en %s", destino)
    print(f"Informe: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
