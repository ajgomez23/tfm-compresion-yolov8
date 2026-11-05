#!/usr/bin/env python
"""Mide la sensibilidad a la cuantización capa a capa.

Cuantiza una sola capa cada vez, deja el resto en coma flotante y mide cuánto cambia la salida. El
resultado es un orden de sensibilidad para decidir qué excluir; no captura interacciones entre
capas. Usa las mismas imágenes que la calibración de PTQ. No escribe en el registro; los resultados
van a results/raw/sensitivity/.

Códigos de salida: 0 si el análisis termina, 1 si hay un error de configuración, 2 si no se pudo
medir ninguna capa.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths, relative_to_root  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.compression.quantization.qat import QatConfig  # noqa: E402
from tfm_edge.compression.quantization.sensitivity import (  # noqa: E402
    SensitivityError,
    measure_layer_sensitivity,
    suggest_exclusions,
)
from tfm_edge.data.calibration_set import (  # noqa: E402
    load_calibration_set,
    resolve_calibration_images,
)
from tfm_edge.models.detector import Detector  # noqa: E402

logger = get_logger(__name__)


def preprocesar(rutas: list[Path], resolution: int) -> list:
    """Prepara las imágenes con el preprocesado de InferenceBackend, el mismo que en latencia y
    calidad.
    """
    import cv2

    from tfm_edge.benchmark.backends import PyTorchBackend

    preprocesador = PyTorchBackend(input_resolution=resolution, device="cpu")
    entradas = []
    for ruta in rutas:
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise TFMEdgeError(
                f"No se ha podido leer la imagen '{ruta}'.",
                "El fichero existe pero OpenCV no lo decodifica: puede estar "
                "truncado. Revisa la preparación del dataset.",
            )
        entradas.append(preprocesador.preprocess(imagen).numpy())
    return entradas


def volcar_csv(informe, destino: Path) -> None:
    """Escribe el orden de sensibilidad como CSV."""
    ensure_dir(destino.parent)
    columnas = [
        "rank",
        "layer",
        "output_mse",
        "output_max_abs",
        "box_max_abs",
        "score_max_abs",
        "num_parameters",
        "in_head",
        "error",
    ]
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    with temporal.open("w", encoding="utf-8", newline="") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=columnas)
        escritor.writeheader()
        posicion = {capa.layer: i for i, capa in enumerate(informe.ranked, start=1)}
        for capa in informe.layers:
            escritor.writerow(
                {
                    # Las capas que no se pudieron medir no tienen puesto: celda vacía.
                    "rank": posicion.get(capa.layer, ""),
                    "layer": capa.layer,
                    "output_mse": capa.output_mse,
                    "output_max_abs": capa.output_max_abs,
                    "box_max_abs": capa.box_max_abs,
                    "score_max_abs": capa.score_max_abs,
                    "num_parameters": capa.num_parameters,
                    "in_head": "sí" if capa.in_head else "no",
                    "error": capa.error,
                }
            )
    temporal.replace(destino)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="09_sensitivity.py",
        description=(
            "Mide capa a capa cuánto cambia la salida al cuantizar a INT8, sobre las imágenes "
            "de calibración. Da el orden de sensibilidad para decidir qué capas excluir en PTQ "
            "y QAT."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Análisis completo del baseline principal\n"
            "  python scripts/09_sensitivity.py \\\n"
            "      --checkpoint models/baselines/s_baseline_default/best.pt\n"
            "\n"
            "  # Comprobación rápida (pocas capas, pocas imágenes)\n"
            "  python scripts/09_sensitivity.py --checkpoint ... --smoke-test\n"
            "\n"
            "  # Solo unas capas concretas\n"
            "  python scripts/09_sensitivity.py --checkpoint ... \\\n"
            "      --layers model.0.conv model.22.cv2.0.0.conv\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Pesos a analizar. Tiene que ser un modelo ENTRENADO: la "
        "sensibilidad de unos pesos sin afinar no dice nada del desplegado.",
    )
    parser.add_argument(
        "--calibration-set",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Lista de calibración (por defecto: data/calibration/calibration_128.txt).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de imágenes (por defecto: data/resized_640/images).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        metavar="N",
        help=(
            "Imágenes con las que medir (por defecto: %(default)s). Más imágenes "
            "dan un orden más estable y cuestan tiempo lineal."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=None,
        metavar="RUTA",
        help="Capas a analizar (por defecto: todas las convoluciones).",
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de entrada."
    )
    parser.add_argument(
        "--qengine",
        type=str,
        default=QatConfig().qengine,
        metavar="MOTOR",
        help="Motor de cuantización (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--top", type=int, default=15, metavar="N", help="Capas a mostrar en el resumen."
    )
    parser.add_argument(
        "--suggest",
        type=int,
        default=5,
        metavar="N",
        help=(
            "Cuántas exclusiones proponer (por defecto: %(default)s). Es una "
            "SUGERENCIA: excluir mejora la precisión y empeora tamaño y latencia."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de salida (por defecto: results/raw/sensitivity).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Subconjunto reducido: 2 imágenes y las 6 primeras convoluciones.",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    add_overwrite_dump_argument(parser, "el volcado de sensibilidad")
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("09_sensitivity", level=args.log_level)
    set_global_seed(args.seed)

    if not args.checkpoint.is_file():
        print(f"ERROR: no existe el checkpoint '{args.checkpoint}'.", file=sys.stderr)
        return 1

    rutas = get_project_paths()
    lista = args.calibration_set or (rutas.data_calibration / "calibration_128.txt")
    directorio_imagenes = args.images_dir or (rutas.data / "resized_640" / "images")
    salida = ensure_dir(args.output_dir or (rutas.results / "raw" / "sensitivity"))
    num_imagenes = 2 if args.smoke_test else args.num_images

    # El destino se calcula antes de medir, que tarda minutos.
    nombre = args.checkpoint.parent.name + ("_smoke" if args.smoke_test else "")
    try:
        refuse_to_overwrite_dump(
            salida / f"{nombre}_sensitivity.json",
            overwrite=args.overwrite_dump,
            what="el volcado de sensibilidad",
        )
    except FileExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("PASO 09: SENSIBILIDAD A LA CUANTIZACIÓN, CAPA A CAPA")
    print("=" * 78)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Calibración : {lista}")
    print(f"Imágenes    : {num_imagenes} a {args.resolution} px   motor: {args.qengine}")
    print()

    try:
        stems = load_calibration_set(lista)
        imagenes = resolve_calibration_images(stems, directorio_imagenes, num_imagenes)
        entradas = preprocesar(imagenes, args.resolution)

        detector = Detector.load(args.checkpoint, input_resolution=args.resolution)
        detector.assert_float32()

        capas = args.layers
        if capas is None and args.smoke_test:
            capas = list(detector.get_named_conv_layers())[:6]
        if capas is None:
            capas = list(detector.get_named_conv_layers())
        print(f"Capas a medir: {len(capas)}")
        print()

        informe = measure_layer_sensitivity(
            detector, entradas, layers=capas, config=QatConfig(qengine=args.qengine)
        )
    except (TFMEdgeError, SensitivityError) as error:
        logger.error("%s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1 if not isinstance(error, SensitivityError) else 2

    print(informe.summary(top=args.top))

    propuestas = suggest_exclusions(informe, num_layers=args.suggest)
    print()
    print(f"Exclusiones propuestas ({len(propuestas)}), de más a menos sensible:")
    for capa in propuestas:
        print(f"  {capa}")
    print()
    print("Es una SUGERENCIA, no una decisión: excluir capas mejora la precisión")
    print("y empeora el tamaño y la latencia. Dónde cortar es del estudio, y la")
    print("decisión se toma en función del compromiso entre precisión y tamaño.")
    print()
    print("AVISO: esta medida cuantiza la convolución con la BatchNorm")
    print("todavía SEPARADA, y los runtimes de PTQ y QAT sí la fusionan. Medido,")
    print("el orden global aguanta (Spearman 0,93) pero el corte no: de las cinco")
    print("capas propuestas solo tres sobreviven a la fusión. Hasta que la fusión")
    print("esté en la ruta de cuantización, esta lista no fija exclusiones.")

    # Una prueba de humo mide pocas capas; se escribe con otro sufijo para no sustituir el resultado
    # completo.
    detalle = salida / f"{nombre}_sensitivity.json"
    datos = informe.to_dict()
    # Rutas relativas a la raíz, porque este volcado se versiona.
    datos["checkpoint"] = relative_to_root(args.checkpoint, rutas)
    datos["calibration_set"] = relative_to_root(lista, rutas)
    datos["calibration_images"] = [i.name for i in imagenes]
    datos["suggested_exclusions"] = list(propuestas)
    detalle.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")

    ranking = salida / f"{nombre}_sensitivity.csv"
    volcar_csv(informe, ranking)

    print()
    print(f"Detalle : {detalle}")
    print(f"Orden   : {ranking}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
