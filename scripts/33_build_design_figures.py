#!/usr/bin/env python
"""Dibuja las cinco figuras del capítulo 4 (diseño del estudio experimental).

Sesgo de selección de best.pt y suelo de error de latencia, leídos de docs/evidencia/; cadena de
causas de QAT y flujo experimental (diagramas, el segundo contrastado con scripts/); y cobertura de
perfiles y motores, deducida de deployment_metrics.csv y configs/benchmark/. Solo escribe en el
directorio de figuras.

Códigos de salida: 0 si se generan las figuras, 1 si falta un dato o falla una comprobación.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Matplotlib lee SOURCE_DATE_EPOCH al importarse, así que se fija antes.
from tfm_edge.reporting import style as _style  # noqa: E402

_style.freeze_creation_date()

from tfm_edge.benchmark.profiles import load_all_profiles  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.reporting import figures_design  # noqa: E402

logger = get_logger(__name__)

#: Las dos tomas de latencia están fuera de results/: la segunda se extrajo de un commit anterior
#: porque la remedida la sobrescribió.
FIRST_TAKE_NAME = "d095_toma_descartada_qat_onnxruntime_int8.json"
SECOND_TAKE_NAME = "d095_toma_registrada_qat_onnxruntime_int8.json"

#: Fichero con las cuatro evaluaciones; las dos curvas están en el mismo directorio.
SELECTION_EVIDENCE_NAME = "d171_sesgo_de_seleccion.json"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="33_build_design_figures.py",
        description="Dibuja las cinco figuras del capítulo 4 de la memoria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Las cinco figuras en results/figures\n"
            "  python scripts/33_build_design_figures.py\n\n"
            "  # Revisarlas antes de tocar results/\n"
            "  python scripts/33_build_design_figures.py --output-dir /tmp/figuras\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dónde dejar las figuras (por defecto: results/figures).",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=None,
        help=(
            "Directorio con las dos tomas de latencia y las curvas del sesgo de selección (por "
            "defecto: docs/evidencia)."
        ),
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Perfiles de medida (por defecto: configs/benchmark).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("33_build_design_figures", level=args.log_level)

    rutas = get_project_paths()
    salida = ensure_dir(args.output_dir or rutas.results_figures)
    evidencia = Path(args.evidence_dir or (rutas.root / "docs" / "evidencia"))
    perfiles_dir = Path(args.profiles_dir or (rutas.root / "configs" / "benchmark"))

    print("=" * 78)
    print("PASO 33: FIGURAS DEL CAPÍTULO 4 (DISEÑO DEL ESTUDIO)")
    print("=" * 78)
    try:
        filas = ExperimentRegistry(args.registry_dir or (rutas.results / "registry")).read_rows(
            "deployment_metrics"
        )
        perfiles = load_all_profiles(perfiles_dir)
        generadas = figures_design.generate_design_figures(
            salida,
            filas,
            perfiles,
            evidencia / FIRST_TAKE_NAME,
            evidencia / SECOND_TAKE_NAME,
            evidencia / SELECTION_EVIDENCE_NAME,
            rutas.root / "scripts",
        )

        celdas = figures_design.coverage_matrix(filas, perfiles)
        recuento: dict[str, int] = {}
        for celda in celdas.values():
            recuento[celda.category] = recuento.get(celda.category, 0) + 1
        print("\nMatriz de cobertura, celdas por categoría:")
        for categoria in figures_design.COVERAGE_CATEGORIES:
            print(f"  {categoria.label:<48s} {recuento.get(categoria.key, 0):>3d}")
        print("\nFilas medidas por celda:")
        for (perfil, motor), celda in celdas.items():
            if celda.measured_rows:
                print(f"  {perfil}/{motor:<12s} {celda.measured_rows:>3d}")

        print()
        for nombre, formatos in generadas.items():
            print(f"  {nombre:32s}: {formatos['pdf'].name}, {formatos['png'].name}")
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print(f"{len(generadas)} figuras generadas en {salida}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
