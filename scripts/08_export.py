#!/usr/bin/env python
"""Exporta un modelo a los formatos de despliegue y verifica cada artefacto.

Cada artefacto se compara numéricamente con el modelo de PyTorch; si no pasa, se registra como
``fallido``. Un runtime ausente se registra como ``no_disponible`` con su motivo y la campaña sigue.

Códigos de salida: 0 si todos los formatos disponibles se exportan y verifican, 1 si hay un error de
configuración, 2 si algún formato falla la verificación.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.export import exporter  # noqa: E402
from tfm_edge.models.detector import DEFAULT_ONNX_OPSET, Detector  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="08_export.py",
        description=(
            "Exporta un checkpoint a ONNX, OpenVINO, TensorRT y TFLite, y "
            "verifica que cada artefacto es numéricamente equivalente al modelo "
            "de PyTorch. Los formatos no disponibles se registran con su motivo."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué formatos se pueden exportar en esta máquina\n"
            "  python scripts/08_export.py --list-formats\n"
            "\n"
            "  # Exportar el baseline principal\n"
            "  python scripts/08_export.py \\\n"
            "      --checkpoint results/runs/s_baseline_default/weights/best.pt\n"
            "\n"
            "  # Solo ONNX, con un opset concreto\n"
            "  python scripts/08_export.py --checkpoint ... --formats onnx --opset 17\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, metavar="RUTA", help="Pesos a exportar."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de los artefactos (por defecto: results/exports).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        metavar="NOMBRE",
        help="Nombre base de los artefactos (por defecto: el de la ejecución).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["onnx", "openvino", "tensorrt", "tflite"],
        choices=sorted(exporter.EXPORT_FORMATS),
        metavar="FMT",
        help="Formatos a intentar (por defecto: todos).",
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de entrada."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Conjunto de operadores ONNX (por defecto: {DEFAULT_ONNX_OPSET}, "
            "elegido por interoperabilidad y no por ser el más nuevo)."
        ),
    )
    parser.add_argument(
        "--check-samples",
        type=int,
        default=3,
        metavar="N",
        help=("Muestras para comprobar la equivalencia (por defecto: %(default)s)."),
    )
    parser.add_argument("--no-simplify", action="store_true", help="No simplifica el grafo ONNX.")
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="Muestra qué formatos se pueden exportar aquí, y termina.",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("08_export", level=args.log_level)
    set_global_seed(args.seed)

    if args.list_formats:
        print("FORMATOS DE EXPORTACIÓN")
        print()
        print(f"{'formato':12s} {'disponible':12s} {'verificado':12s}  motivo")
        print("-" * 78)
        for formato in sorted(exporter.EXPORT_FORMATS):
            hay, motivo = exporter.check_format_available(formato)
            verificado = "sí" if formato in exporter.VERIFIED_FORMATS else "no"
            print(f"{formato:12s} {'sí' if hay else 'NO':12s} {verificado:12s}  {motivo}")
        print()
        print("'verificado' = se ha ejecutado y comprobado en el entorno de")
        print("desarrollo. Los demás están implementados pero no se han podido")
        print("probar aquí; se registrarán como no_disponible con su motivo.")
        return 0

    if not args.checkpoint:
        print(
            "ERROR: hace falta --checkpoint. Usa --list-formats para ver los formatos.",
            file=sys.stderr,
        )
        return 1
    if not args.checkpoint.is_file():
        print(f"ERROR: no existe el checkpoint '{args.checkpoint}'.", file=sys.stderr)
        return 1

    rutas = get_project_paths()
    salida = ensure_dir(args.output_dir or (rutas.results / "exports"))
    nombre = args.name or args.checkpoint.parent.parent.name

    print("=" * 78)
    print("PASO 08: EXPORTACIÓN Y VERIFICACIÓN DE ARTEFACTOS")
    print("=" * 78)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Destino    : {salida}")
    print(f"Formatos   : {args.formats}   resolución: {args.resolution} px")
    print()

    try:
        detector = Detector.load(args.checkpoint, input_resolution=args.resolution)
        detector.assert_float32()
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    resultados = exporter.export_all(
        detector,
        salida,
        nombre,
        formats=args.formats,
        num_check_samples=args.check_samples,
    )

    print(exporter.summarize_exports(resultados))

    informe = salida / f"{nombre}_export_report.json"
    informe.write_text(
        json.dumps(
            {formato: r.to_dict() for formato, r in resultados.items()},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nInforme detallado en: {informe}")

    fallidos = [f for f, r in resultados.items() if r.status == "fallido"]
    exportados = [f for f, r in resultados.items() if r.ok]
    print()
    print("=" * 78)
    print(f"Exportados y verificados: {exportados}")
    if fallidos:
        print(f"FALLIDOS: {fallidos}")
        for formato in fallidos:
            print(f"  {formato}: {resultados[formato].notes}")
    no_disponibles = [f for f, r in resultados.items() if r.status == "no_disponible"]
    if no_disponibles:
        print(f"No disponibles en esta máquina: {no_disponibles}")
    print("=" * 78)
    return 2 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
