#!/usr/bin/env python
"""Genera las figuras del análisis exploratorio del dataset.

Parte del COCO preparado por 02_prepare_dataset.py y produce las figuras del capítulo 3 en PDF y PNG
a 300 ppp. La distribución de tamaños se dibuja con los dos criterios en paneles contiguos.

Códigos de salida: 0 si se generan las figuras, 1 si falta el dataset preparado o su raw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Matplotlib lee SOURCE_DATE_EPOCH al importarse; se fija antes para que dos ejecuciones iguales den
# PDF idénticos.
from tfm_edge.reporting import style as _style  # noqa: E402

_style.freeze_creation_date()

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.data import eda, prepare, splits, statistics  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="03_dataset_eda.py",
        description=(
            "Genera las figuras del análisis exploratorio del dataset SH17 a "
            "partir del COCO preparado. Exporta cada figura simultáneamente en "
            "PDF vectorial y PNG a 300 ppp, con nombres que llevan el prefijo de "
            "capítulo y apartado de la memoria."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Caso por defecto: todas las figuras\n"
            "  python scripts/03_dataset_eda.py\n"
            "\n"
            "  # Rápido: sin el mosaico de ejemplos, que abre imágenes grandes\n"
            "  python scripts/03_dataset_eda.py --no-examples\n"
            "\n"
            "  # Recalcular las estadísticas con otra resolución de inferencia\n"
            "  python scripts/03_dataset_eda.py --recompute --inference-resolution 416\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio con el COCO preparado (por defecto: data/processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de las figuras (por defecto: results/figures).",
    )
    parser.add_argument(
        "--inference-resolution",
        type=int,
        default=640,
        metavar="N",
        help=(
            "Resolución de entrada del modelo, que determina el criterio de tamaño primario "
            "(por defecto: %(default)s). Queda registrada en las estadísticas."
        ),
    )
    parser.add_argument(
        "--primary-criterion",
        type=str,
        default=statistics.DEFAULT_PRIMARY_CRITERION,
        choices=list(statistics.SIZE_CRITERIA),
        help=(
            "Cuál de los dos criterios de tamaño es el primario (por defecto: "
            "%(default)s). Ambos se calculan siempre."
        ),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Directorio con las listas de partición train/val/test.txt (por "
            "defecto: data/splits). De aquí sale la figura del apartado 3.3.2."
        ),
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Directorio con las etiquetas YOLO (por defecto: el 'labels' del dataset crudo). Se "
            "usa para contar instancias por clase y partición."
        ),
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recalcula las estadísticas desde el COCO en lugar de leer el JSON ya volcado.",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Omite el mosaico de ejemplos anotados, que es la figura más lenta.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Usa el COCO reducido del smoke test, si existe.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def volcar_conteos_particion(
    conteos: dict[str, dict[str, int]], clases: list[str], destino: Path
) -> Path:
    """Escribe en CSV las instancias por clase y partición que muestra la figura."""
    import pandas as pd

    filas = [
        {"class_name": nombre, "class_index": indice, **{p: conteos[p][nombre] for p in conteos}}
        for indice, nombre in enumerate(clases)
    ]
    marco = pd.DataFrame(filas)
    marco["total"] = marco[list(conteos)].sum(axis=1)
    ensure_parent_dir(destino)
    marco.to_csv(destino, index=False, na_rep="")
    return destino


def volcar_coocurrencia_condicional(coocurrencia: dict, destino: Path) -> Path:
    """Escribe en CSV la coocurrencia condicional dirigida, con su definición en una columna."""
    import pandas as pd

    nombres = coocurrencia["class_names"]
    marco = pd.DataFrame(
        [[coocurrencia["matrix"][fila][columna] for columna in nombres] for fila in nombres],
        index=nombres,
        columns=nombres,
    )
    marco.index.name = "row_class_A"
    ensure_parent_dir(destino)
    marco.to_csv(destino, na_rep="")
    destino.with_suffix(".txt").write_text(coocurrencia["definition"] + "\n", encoding="utf-8")
    return destino


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("03_dataset_eda", level=args.log_level)

    rutas = get_project_paths()
    procesado = args.processed_dir or rutas.data_processed
    salida = ensure_dir(args.output_dir or rutas.results_figures)

    print("=" * 78)
    print("PASO 03: ANÁLISIS EXPLORATORIO DEL DATASET")
    print("=" * 78)

    try:
        sufijo = "_smoke" if args.smoke_test else ""
        ruta_coco = Path(procesado) / f"sh17_coco{sufijo}.json"
        if not ruta_coco.is_file():
            raise TFMEdgeError(
                f"No existe '{ruta_coco}'.",
                "Prepara antes el dataset con scripts/02_prepare_dataset.py.",
            )

        # processed depende de raw; se comprueba antes de empezar.
        crudo = prepare.verify_raw_available(processed_dir=procesado)
        print(f"Dataset crudo verificado en: {crudo}")

        coco = json.loads(ruta_coco.read_text(encoding="utf-8"))
        clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")

        ruta_estadisticas = Path(procesado) / "dataset_statistics.json"
        if args.recompute or not ruta_estadisticas.is_file():
            print("Calculando estadísticas...")
            estadisticas = statistics.compute_statistics(
                coco,
                clases,
                inference_resolution=args.inference_resolution,
                primary_criterion=args.primary_criterion,
            )
            estadisticas.to_json(ruta_estadisticas)
            estadisticas.to_csv(procesado)
        else:
            estadisticas = statistics.DatasetStatistics(
                payload=json.loads(ruta_estadisticas.read_text(encoding="utf-8"))
            )
            print(f"Estadísticas leídas de {ruta_estadisticas}")

        criterios = estadisticas.payload["size_criteria"]
        print(f"\nCriterio de tamaño primario : {criterios['primary']}")
        print(f"Resolución de inferencia    : {criterios['inference_resolution']} px")
        print(f"Modo de redimensionado      : {criterios['resize_mode']}")
        print("\nReparto de tamaños de objeto:")
        for criterio in statistics.SIZE_CRITERIA:
            reparto = estadisticas.payload["size_distribution"][criterio]["fractions"]
            marca = "PRIMARIO  " if criterio == criterios["primary"] else "secundario"
            print(
                f"  [{marca}] {criterio:14s} "
                + "  ".join(f"{r}={reparto[r]:.1%}" for r in statistics.SIZE_RANGES)
            )

        # Recuento por partición (apartado 3.3.2)
        dir_splits = Path(args.splits_dir or rutas.data_splits)
        dir_etiquetas = Path(args.labels_dir or (crudo / "labels"))
        conteos = None
        if all((dir_splits / f"{n}.txt").is_file() for n in splits.SPLIT_NAMES):
            listas = {
                n: splits.load_split_file(dir_splits / f"{n}.txt") for n in splits.SPLIT_NAMES
            }
            conteos = splits.count_instances_per_class(listas, dir_etiquetas, clases)
            print("\nInstancias por clase y partición:")
            for nombre in splits.SPLIT_NAMES:
                total = sum(conteos[nombre].values())
                print(
                    f"  {nombre:5s} {_style.format_integer(total):>8s} instancias "
                    f"en {_style.format_integer(len(listas[nombre]))} imágenes"
                )
            # Si alguna clase falta en alguna partición, se aborta sin generar la figura.
            splits.verify_all_classes_present(conteos, clases)
            print("  Las 17 clases están en train, val y test.")
            ruta_conteos = volcar_conteos_particion(
                conteos, list(clases), Path(procesado) / "stats_split_class_instances.csv"
            )
            print(f"  Recuento volcado en {ruta_conteos}")
        else:
            print(
                f"\nAVISO: no hay listas de partición en {dir_splits}. Se omite la "
                "figura del apartado 3.3.2."
            )

        # Coocurrencia condicional (apartado 3.4)
        coocurrencia = statistics.compute_conditional_cooccurrence(coco, clases)
        ruta_coocurrencia = volcar_coocurrencia_condicional(
            coocurrencia, Path(procesado) / "stats_conditional_cooccurrence.csv"
        )
        print(f"Coocurrencia condicional volcada en {ruta_coocurrencia}")

        print("\nGenerando figuras...")
        generadas = eda.generate_all_figures(
            estadisticas,
            coco,
            clases,
            images_dir=crudo / "images",
            output_dir=salida,
            include_examples=not args.no_examples,
            split_counts=conteos,
            conditional_cooccurrence=coocurrencia,
        )
        for nombre, formatos in generadas.items():
            print(f"  {nombre:45s}: {formatos['pdf'].name}, {formatos['png'].name}")

    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interacción del usuario
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print(f"{len(generadas)} figuras generadas en {salida}")
    print("Cada una en PDF vectorial y PNG a 300 ppp.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
