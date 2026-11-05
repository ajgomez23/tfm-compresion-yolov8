#!/usr/bin/env python
"""Dibuja las figuras del capítulo 1 (introducción).

Genera el problema, las técnicas de compresión, la muestra anotada de SH17 (necesita data/; sin ella
se omite), el espacio de decisión y el flujo del estudio. La figura del resultado principal la
dibuja el paso 27. El manifiesto results/raw/intro/intro_figures_manifest.json guarda la atribución
y la licencia de cada imagen de la muestra.

Códigos de salida: 0 si se generan las figuras, 1 si falta un dato o falla una comprobación.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Matplotlib lee SOURCE_DATE_EPOCH al importarse, así que se fija antes.
from tfm_edge.reporting import style as _style  # noqa: E402

_style.freeze_creation_date()

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.reporting import figures_intro  # noqa: E402

logger = get_logger(__name__)

MANIFEST_NAME = "intro_figures_manifest.json"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="35_build_intro_figures.py",
        description="Dibuja las figuras del capítulo 1 de la memoria, salvo el resultado "
        "principal, que sale del paso 27.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Las figuras en results/figures y el manifiesto en results/raw/intro\n"
            "  python scripts/35_build_intro_figures.py\n\n"
            "  # Revisarlas antes de tocar results/\n"
            "  python scripts/35_build_intro_figures.py --output-dir /tmp/figuras "
            "--manifest-dir /tmp/figuras\n"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Dónde dejar las figuras (por defecto: results/figures).",
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=None,
        help="Dónde dejar el manifiesto (por defecto: results/raw/intro).",
    )
    parser.add_argument(
        "--registry-dir", type=Path, default=None,
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("35_build_intro_figures", level=args.log_level)

    rutas = get_project_paths()
    salida = ensure_dir(args.output_dir or rutas.results_figures)
    destino = ensure_parent_dir(
        Path(args.manifest_dir or (rutas.results_raw / "intro")) / MANIFEST_NAME
    )
    print("=" * 78)
    print("PASO 35: FIGURAS DEL CAPÍTULO 1 (INTRODUCCIÓN)")
    print("=" * 78)
    try:
        experimentos, _ = ExperimentRegistry(
            args.registry_dir or rutas.results_registry
        ).valid_rows("experiments")
        manifiesto = figures_intro.generate_intro_figures(
            salida,
            experimentos,
            rutas.root / "docs" / "SH17_ATTRIBUTIONS.md",
            rutas.root / "data" / "processed" / "sh17_coco.json",
            rutas.root / "data" / "raw" / "images",
        )
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    temporal = destino.with_suffix(destino.suffix + ".tmp")
    temporal.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporal, destino)

    for nombre, entrada in manifiesto["figures"].items():
        print(f"  {nombre:28s}: {Path(entrada['files']['pdf']).name}")
    for nombre, motivo in manifiesto["skipped"].items():
        print(f"  {nombre:28s} omitida: {motivo}")
    muestra = manifiesto["figures"].get("muestra_sh17")
    if muestra:
        print("\nAtribución de la muestra:")
        for imagen in muestra["images"]:
            print(f"  {imagen['image']}: {imagen['photographer']}, {imagen['source']}")
    print(f"\nManifiesto: {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
