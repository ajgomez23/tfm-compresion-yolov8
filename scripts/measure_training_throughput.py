#!/usr/bin/env python
"""Mide si el cuello de botella del entrenamiento es la CPU o la GPU.

Script auxiliar. La máquina de Colab tiene una GPU razonable y un solo núcleo físico, y el aumento
de datos de Ultralytics consume CPU. Se mide sobre el dataset real, comparando el tiempo de iterar
solo los datos con el del paso completo: si se parecen, la GPU está esperando a la CPU.

Códigos de salida: 0 si la medición termina, 1 si falta el dataset, PyTorch o la GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.environment import (  # noqa: E402
    capture_environment,
    detect_execution_context,
    summarize_environment,
)
from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.training import profiling  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="measure_training_throughput.py",
        description=(
            "Mide el tiempo por época sobre el dataset real y diagnostica si el "
            "cuello de botella es la CPU (carga de datos y aumento) o la GPU "
            "(cómputo). Extrapola las horas totales de entrenamiento."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Lo que hay que ejecutar en Colab: los tres modelos candidatos\n"
            "  python scripts/measure_training_throughput.py \\\n"
            "      --data configs/dataset/sh17_ultralytics.yaml \\\n"
            "      --sizes s m l --batch-size 16 --epochs 100\n"
            "\n"
            "  # Barrido de workers sobre un solo modelo, para hallar el óptimo\n"
            "  python scripts/measure_training_throughput.py \\\n"
            "      --data configs/dataset/sh17_ultralytics.yaml \\\n"
            "      --sizes s --sweep-workers 0 1 2 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        metavar="RUTA",
        help="YAML del dataset en formato Ultralytics. OBLIGATORIO: se mide sobre datos reales.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["s", "m", "l"],
        metavar="T",
        help="Tamaños de modelo a medir (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        metavar="N",
        help="Resolución de entrada (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Tamaño de lote (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--sweep-workers",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Valores de workers a probar. Si se omite, se prueba "
            f"{list(profiling.DEFAULT_WORKER_OPTIONS)} solo con el primer tamaño, "
            "y el resto se mide con el óptimo encontrado."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="Épocas previstas, para extrapolar el tiempo total (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=profiling.DEFAULT_MEASURE_BATCHES,
        metavar="N",
        help="Lotes a cronometrar por configuración (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="False",
        choices=["False", "disk", "ram"],
        help=(
            "Caché de Ultralytics (por defecto: %(default)s). 'disk' guarda las imágenes "
            "decodificadas en .npy junto a las originales; el script mide el espacio que ocupa."
        ),
    )
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        metavar="P",
        help=(
            "Probabilidad del aumento por mosaico (por defecto: %(default)s). "
            "Ponerlo a 0 desactiva el aumento mas caro en CPU."
        ),
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        metavar="N",
        help="Epocas finales sin mosaico (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        metavar="ETIQUETA",
        help="Etiqueta de la variante, para identificarla en la tabla comparativa.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Mide sin precisión mixta.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=17,
        metavar="N",
        help="Número de clases del detector (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--session-limit-hours",
        type=float,
        default=profiling.COLAB_SESSION_LIMIT_HOURS,
        metavar="H",
        help=(
            "Límite práctico de una sesión en horas (por defecto: %(default)s); es una cota "
            "conservadora."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Fichero JSON de salida (por defecto: results/raw/throughput_<device_id>.json).",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="",
        metavar="ID",
        help="Identificador del entorno donde se mide.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    add_overwrite_dump_argument(parser, "el volcado de rendimiento")
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("measure_training_throughput", level=args.log_level)
    set_global_seed(42, deterministic=False)

    print("=" * 78)
    print("RENDIMIENTO DEL ENTRENAMIENTO: ¿el cuello es la CPU o la GPU?")
    print("=" * 78)

    if not args.data.is_file():
        print(f"ERROR: no existe el YAML del dataset '{args.data}'.", file=sys.stderr)
        print(
            "       Se mide sobre datos REALES: sin dataset no hay nada que medir.", file=sys.stderr
        )
        return 1

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch no está instalado.", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print(
            "ERROR: no hay GPU. Este script mide el reparto CPU/GPU del entrenamiento.",
            file=sys.stderr,
        )
        return 1

    contexto = detect_execution_context()
    device_id = args.device_id or ("colab_gpu" if "google_colab" in contexto else "workstation_gpu")
    huella = capture_environment(device_id=device_id)

    # Antes de medir, porque cada configuración tarda.
    salida = args.output or (rutas.results_raw / f"throughput_{device_id}.json")
    try:
        refuse_to_overwrite_dump(
            salida, overwrite=args.overwrite_dump, what="el volcado de rendimiento"
        )
    except FileExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(summarize_environment(huella))
    print()
    print(
        f"Nucleos fisicos: {huella.get('cpu_cores_physical')}  "
        f"logicos: {huella.get('cpu_cores_logical')}"
    )
    print("El aumento de datos de Ultralytics corre en la CPU: con pocos nucleos,")
    print("la GPU puede pasarse el entrenamiento esperando datos.\n")

    rutas = get_project_paths()
    trabajo = ensure_dir(rutas.results_raw / "profiling")
    amp = not args.no_amp
    cache = False if args.cache == "False" else args.cache
    tipo_dataset = "resized" if "resized" in str(args.data) else "original"
    variante = args.variant or (
        f"{tipo_dataset}"
        + (f"+cache={args.cache}" if cache else "")
        + (f"+mosaic={args.mosaic}" if args.mosaic != 1.0 else "")
    )
    print(
        f"Variante: {variante}  (dataset {tipo_dataset}, cache {args.cache}, "
        f"mosaic {args.mosaic})\n"
    )
    medidas: list[profiling.ThroughputMeasurement] = []

    try:
        # Barrido de workers sobre el primer tamaño para encontrar el mejor valor.
        opciones = args.sweep_workers or list(profiling.DEFAULT_WORKER_OPTIONS)
        print(f"[1/2] Barrido de workers sobre yolov8{args.sizes[0]}: {opciones}")
        barrido = profiling.sweep_workers(
            size=args.sizes[0],
            resolution=args.resolution,
            batch_size=args.batch_size,
            data_config=args.data,
            output_dir=trabajo,
            worker_options=opciones,
            amp=amp,
            num_classes=args.num_classes,
            measure_batches=args.batches,
            cache=cache,
            mosaic=args.mosaic,
            variant=variante,
            dataset_kind=tipo_dataset,
        )
        medidas.extend(barrido)

        utiles = [m for m in barrido if isinstance(m.full_step_s_per_batch, (int, float))]
        if not utiles:
            print("\nERROR: ninguna configuración ha producido una medida válida.", file=sys.stderr)
            for m in barrido:
                if m.error:
                    print(f"  workers={m.workers}: {m.error}", file=sys.stderr)
            return 1
        mejor = min(utiles, key=lambda m: m.full_step_s_per_batch)
        print(f"\n  Óptimo: workers={mejor.workers} con {mejor.full_step_s_per_batch} s/lote\n")

        # El resto de tamaños con ese valor.
        restantes = args.sizes[1:]
        if restantes:
            print(f"[2/2] Midiendo {restantes} con workers={mejor.workers}")
            for size in restantes:
                medidas.append(
                    profiling.measure_throughput(
                        size=size,
                        resolution=args.resolution,
                        batch_size=args.batch_size,
                        workers=mejor.workers,
                        data_config=args.data,
                        output_dir=trabajo,
                        amp=amp,
                        num_classes=args.num_classes,
                        measure_batches=args.batches,
                        cache=cache,
                        mosaic=args.mosaic,
                        close_mosaic=args.close_mosaic,
                        variant=variante,
                        dataset_kind=tipo_dataset,
                    )
                )

    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    proyecciones = profiling.project_training_time(
        medidas, epochs=args.epochs, session_limit_hours=args.session_limit_hours
    )
    profiling.save_measurements(medidas, proyecciones, salida, environment=huella)

    print()
    print(profiling.summarize(medidas, proyecciones))
    print()
    print("=" * 78)
    print(f"Medidas completas en: {salida}")
    cuellos = {m.bottleneck for m in medidas if m.bottleneck}
    if "cpu" in cuellos:
        print()
        print("DIAGNOSTICO: al menos una configuracion esta limitada por la CPU.")
        print("  Mas VRAM o un modelo mas pequeno NO mejoraran el tiempo por epoca.")
        print("  Opciones: reducir el aumento de datos, bajar la resolucion, o")
        print("  entrenar en una maquina con mas nucleos.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
