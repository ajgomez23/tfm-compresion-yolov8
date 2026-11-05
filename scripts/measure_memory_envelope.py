#!/usr/bin/env python
"""Mide qué combinaciones de modelo, resolución y lote caben en la GPU.

Es un script auxiliar que se ejecuta antes de entrenar: su resultado decide el par maestro-alumno y
la resolución del estudio. Cada celda ejecuta un paso de entrenamiento real y lee
``torch.cuda.max_memory_reserved()``. Se ejecutó en Colab (16 GB) y en la estación local (4 GB).

Códigos de salida: 0 si la medición termina, 1 si no hay GPU o falta PyTorch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.environment import capture_environment, summarize_environment  # noqa: E402
from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.models import memory_envelope as env  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="measure_memory_envelope.py",
        description=(
            "Mide qué combinaciones de modelo, resolución y lote caben en la memoria de la GPU, "
            "con y sin precisión mixta. Cada celda se mide en un subproceso independiente."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Envolvente completo (lo que hay que ejecutar en Colab)\n"
            "  python scripts/measure_memory_envelope.py\n"
            "\n"
            "  # Solo el par maestro-alumno candidato, a la resolución del estudio\n"
            "  python scripts/measure_memory_envelope.py --sizes s l --resolutions 640\n"
            "\n"
            "  # Rápido, para una GPU pequeña\n"
            "  python scripts/measure_memory_envelope.py --sizes n s --batch-sizes 1 2 4 8\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=list(env.DEFAULT_SIZES),
        metavar="T",
        help="Tamaños de modelo a probar (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(env.DEFAULT_RESOLUTIONS),
        metavar="N",
        help="Resoluciones de entrada a probar (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(env.DEFAULT_BATCH_SIZES),
        metavar="N",
        help="Tamaños de lote a probar (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Prueba solo sin precisión mixta, en lugar de con y sin.",
    )
    parser.add_argument(
        "--amp-only",
        action="store_true",
        help="Prueba solo con precisión mixta.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=17,
        metavar="N",
        help="Número de clases del detector (por defecto: %(default)s, las de SH17).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Dispositivo sobre el que medir (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Fichero JSON de salida (por defecto: " "results/raw/memory_envelope_<device_id>.json)."
        ),
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Identificador del entorno donde se mide, para el nombre del fichero "
            "y la huella. Si se omite, se deduce del contexto de ejecución."
        ),
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help=(
            "No omite los lotes mayores tras el primer fallo de memoria. Más lento "
            "y sin información adicional: si no cabe con 8, no cabe con 16."
        ),
    )
    parser.add_argument(
        "--isolation",
        type=str,
        default="subprocess",
        choices=["subprocess", "inprocess"],
        help=(
            "Cómo se aísla cada medida (por defecto: %(default)s). 'subprocess' arranca un "
            "proceso por combinación; 'inprocess' es más rápido pero arrastra estado y solo "
            "sirve para depurar."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    add_overwrite_dump_argument(parser, "el volcado de la envolvente de memoria")
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("measure_memory_envelope", level=args.log_level)
    set_global_seed(42, deterministic=False)

    print("=" * 78)
    print("ENVOLVENTE DE MEMORIA: qué combinaciones caben en esta GPU")
    print("=" * 78)

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch no está instalado.", file=sys.stderr)
        return 1

    from tfm_edge.common.environment import detect_execution_context

    contexto = detect_execution_context()
    device_id = args.device_id or ("colab_gpu" if "google_colab" in contexto else "workstation_gpu")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("ERROR: se pidió CUDA y no hay ninguna GPU disponible.", file=sys.stderr)
        print(
            "       Este script no simula medidas: sin GPU no hay nada que medir.", file=sys.stderr
        )
        return 1

    huella = capture_environment(device_id=device_id)

    # La comprobación va antes de medir, porque la medida tarda minutos.
    salida = args.output or (get_project_paths().results_raw / f"memory_envelope_{device_id}.json")
    try:
        refuse_to_overwrite_dump(
            salida, overwrite=args.overwrite_dump, what="el volcado de la envolvente de memoria"
        )
    except FileExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(summarize_environment(huella))
    print()

    if args.amp_only:
        amp_options = (True,)
    elif args.no_amp:
        amp_options = (False,)
    else:
        amp_options = (True, False)

    total = len(args.sizes) * len(args.resolutions) * len(args.batch_sizes) * len(amp_options)
    print(f"Combinaciones a probar (cota superior): {total}")
    print("Cada una ejecuta un paso completo de entrenamiento. Puede tardar.\n")

    try:
        medidas = env.measure_envelope(
            sizes=args.sizes,
            resolutions=args.resolutions,
            batch_sizes=args.batch_sizes,
            amp_options=amp_options,
            num_classes=args.num_classes,
            device=args.device,
            stop_on_first_oom=not args.no_early_stop,
            isolation=args.isolation,
        )
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    # Si el pico no crece con el lote o la resolución, la medida está contaminada.
    coherencia = env.check_monotonicity(medidas)

    env.save_measurements(medidas, salida, environment=huella, monotonicity=coherencia)

    print()
    print(env.summarize(medidas))
    print()
    print(coherencia.summary())
    print()
    print("=" * 78)
    caben = sum(1 for m in medidas if m.fits)
    print(f"{caben} de {len(medidas)} combinaciones probadas caben en memoria.")
    print(f"Medidas completas en: {salida}")
    if not coherencia.is_coherent:
        print()
        print("AVISO: la medicion NO es coherente. Los valores no son fiables.")
        print("       Reejecuta con --isolation subprocess si has usado 'inprocess'.")
    print("=" * 78)
    return 0 if coherencia.is_coherent else 2


if __name__ == "__main__":
    sys.exit(main())
