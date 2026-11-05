#!/usr/bin/env python
"""Genera una copia del dataset con las imágenes redimensionadas (paso opcional entre el 02 y el
03).

En Colab el cuello de botella era la CPU, que decodifica y redimensiona en cada época JPEG de hasta
8192x5462. data/raw no se modifica; la copia va a data/resized_<N>/. El tamaño de los objetos no
cambia porque las áreas se calculan sobre las dimensiones originales. La resolución del estudio no
puede superar la de la copia.

Códigos de salida: 0 si la copia se genera y verifica, 1 si falta el dataset original o no se
conserva el recuento.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.data import prepare, resize  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="02b_resize_dataset.py",
        description=(
            "Genera una copia del dataset con las imágenes redimensionadas al lado "
            "largo indicado, conservando la relación de aspecto. No modifica "
            "data/raw. Las etiquetas YOLO se copian sin tocar, porque son "
            "normalizadas. Acelera el entrenamiento cuando el cuello es la CPU."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Copia a 640 px, que es la resolución del estudio\n"
            "  python scripts/02b_resize_dataset.py\n"
            "\n"
            "  # A 512 px y con más hilos\n"
            "  python scripts/02b_resize_dataset.py --target-size 512 --workers 8\n"
            "\n"
            "  # Regenerar una copia existente\n"
            "  python scripts/02b_resize_dataset.py --force\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dataset original (por defecto: data/raw). NO se modifica.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Destino de la copia (por defecto: data/resized_<N>).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=640,
        metavar="N",
        help=(
            "Lado largo objetivo, en píxeles (por defecto: %(default)s). La "
            "resolución del estudio no podrá superar este valor."
        ),
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=resize.DEFAULT_JPEG_QUALITY,
        metavar="N",
        help=(
            "Calidad JPEG de la copia (por defecto: %(default)s). Alta para no introducir "
            "artefactos de compresión."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Hilos de redimensionado (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenera las imágenes que ya existan en el destino.",
    )
    parser.add_argument(
        "--no-dataset-yaml",
        action="store_true",
        help="No genera el YAML de Ultralytics que apunta a la copia.",
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
    setup_logging("02b_resize_dataset", level=args.log_level)

    rutas = get_project_paths()
    crudo = args.raw_dir or rutas.data_raw
    destino = args.output_dir or (rutas.data / f"resized_{args.target_size}")

    print("=" * 78)
    print(f"PASO 02b: COPIA REDIMENSIONADA A {args.target_size} px")
    print("=" * 78)
    print(f"Origen  : {crudo}   (NO se modifica)")
    print(f"Destino : {destino}")
    print()
    print("Las areas de los objetos y las estadisticas del estudio NO cambian:")
    print("salen del COCO, calculado sobre las dimensiones ORIGINALES.")
    print(f"La resolucion del estudio no podra superar {args.target_size} px.")
    print()

    libre = shutil.disk_usage(rutas.data if rutas.data.exists() else rutas.root).free
    print(f"Espacio libre: {libre / 2**30:.1f} GiB\n")

    try:
        informe = resize.resize_dataset(
            raw_dir=crudo,
            output_dir=destino,
            target_size=args.target_size,
            quality=args.quality,
            workers=args.workers,
            force=args.force,
        )
        print()
        print(informe.summary())
        if informe.errors:
            print(f"\nERRORES ({len(informe.errors)}), primeros 5:")
            for error in informe.errors[:5]:
                print(f"  - {error}")

        if not args.no_dataset_yaml:
            clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
            ruta_yaml = prepare.write_ultralytics_dataset_yaml(
                dataset_dir=destino,
                splits_dir=rutas.data_splits,
                class_names=clases,
            )
            print(f"\nYAML para Ultralytics apuntando a la copia: {ruta_yaml}")
            # Se comprueba que la copia no lleva rutas de esta máquina antes de darla por buena.
            prepare.assert_dataset_portable(destino)
            print("Portabilidad verificada: ninguna ruta absoluta en el YAML ni en las listas")

        comprobacion = resize.verify_resized_dataset(destino, study_resolution=args.target_size)
        print(f"Copia verificada: {comprobacion['num_images']:,} imágenes")

    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print()
    print("=" * 78)
    print("Copia lista. Es portable: el YAML y las listas van dentro y no")
    print("contienen ninguna ruta absoluta, asi que funciona alla donde se")
    print("descomprima, sin regenerar nada.")
    print()
    print("Para medir el efecto en el rendimiento:")
    print("  python scripts/measure_training_throughput.py \\")
    print(f"      --data {destino / prepare.ULTRALYTICS_DATASET_FILENAME} \\")
    print("      --sizes s --sweep-workers 2")
    print()
    print("Para empaquetarla y subirla a Drive (es lo que espera el cuaderno de")
    print("Colab, que la busca exactamente por ese nombre):")
    print(f"  tar -C {destino.parent} -czf sh17_resized{args.target_size}.tar.gz {destino.name}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
