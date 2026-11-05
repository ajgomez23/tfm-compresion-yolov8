#!/usr/bin/env python
"""Coste en mAP de la resolución de puntuación de un artefacto INT8, aislado del resto.

El INT8 de ONNX Runtime pierde más mAP que el de OpenVINO y tiene muchos menos valores distintos de
puntuación, pero los dos cuantizadores difieren en más cosas. Aquí se toman las detecciones del FP32
y solo se redondean sus puntuaciones a la rejilla de niveles del INT8, sin tocar las cajas. Antes se
reevalúan las detecciones sin modificar y se comprueba que reproducen el mAP registrado, porque la
traducción de categorías se hace sin el class_map del validador.

Códigos de salida: 0 si la ablación termina, 1 si hay un error de configuración, 2 si la
reevaluación no reproduce el registro.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.evaluation import coco_eval, runner  # noqa: E402

logger = get_logger(__name__)

#: Tolerancia de la comprobación: un error en las categorías daría diferencias de décimas.
GUARD_TOLERANCE = 1e-5


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="12_score_resolution.py",
        description=(
            "Mide el coste en mAP de la resolución de puntuación de un artefacto INT8: redondea "
            "las puntuaciones del FP32 a la rejilla del INT8 sin tocar las cajas."
        ),
        epilog=(
            "Ejemplo:\n"
            "  python scripts/12_score_resolution.py \\\n"
            "      --reference results/raw/artifact_eval/"
            "validator_s_baseline_default_745e3d08_onnxruntime_fp32 \\\n"
            "      --quantized results/raw/artifact_eval/"
            "validator_s_ptq_int8-noexcl_5bcd71f0_onnxruntime_int8 \\\n"
            "      --expected-map-50 0.586152 --expected-map-50-95 0.376214\n"
            "\n"
            "Algunos directorios antiguos de 11_evaluate_artifact.py se llaman\n"
            "'validator_{runtime}_{precision}', sin el experimento.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Directorio del validador con las detecciones FP32 (predictions.json).",
    )
    parser.add_argument(
        "--quantized",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Directorio del validador con las detecciones INT8, del que sale la rejilla.",
    )
    parser.add_argument(
        "--expected-map-50",
        type=float,
        required=True,
        metavar="M",
        help="mAP@50 que la reevaluación del FP32 tiene que reproducir (la del registro).",
    )
    parser.add_argument(
        "--expected-map-50-95",
        type=float,
        required=True,
        metavar="M",
        help="mAP@50-95 que la reevaluación del FP32 tiene que reproducir.",
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=None,
        metavar="RUTA",
        help="COCO de referencia (por defecto: data/processed/sh17_coco.json).",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"], help="Partición evaluada."
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de inferencia."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="RUTA",
        help="JSON de salida (por defecto: results/raw/artifact_eval/score_resolution.json).",
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


def cargar_detecciones(directorio: Path) -> list[dict[str, Any]]:
    """Lee las detecciones de un validador y las pasa a la numeración de clases del proyecto (desde
    0).

    Ultralytics numera desde 1 en datasets que no son COCO; el desfase se aplica aquí y el script lo
    comprueba reproduciendo el mAP registrado.
    """
    fichero = Path(directorio) / "predictions.json"
    if not fichero.is_file():
        raise TFMEdgeError(
            f"No existe '{fichero}'.",
            "Lo escribe scripts/11_evaluate_artifact.py al evaluar el artefacto.",
        )
    detecciones = json.loads(fichero.read_text(encoding="utf-8"))
    return [{**d, "category_id": int(d["category_id"]) - 1} for d in detecciones]


def evaluar(
    detecciones: list[dict[str, Any]],
    referencia: dict[str, Any],
    images_dir: Path,
    split: str,
    input_resolution: int,
) -> Any:
    """Evalúa unas detecciones por el mismo camino que el runner."""
    emparejadas = coco_eval.map_detections_to_ground_truth(detecciones, referencia)
    reescaladas, _ = coco_eval.rescale_detections_to_original(emparejadas, referencia, images_dir)
    return coco_eval.evaluate_coco(
        referencia,
        reescaladas,
        input_resolution=input_resolution,
        split=split,
        conf_threshold=0.001,
        iou_threshold=0.7,
        with_per_class=False,
    )


def aplastar_puntuaciones(
    detecciones: list[dict[str, Any]], niveles: list[float]
) -> list[dict[str, Any]]:
    """Redondea cada puntuación al nivel más cercano de la rejilla, sin tocar las cajas."""
    import numpy as np

    rejilla = np.asarray(sorted(niveles), dtype=float)
    aplastadas: list[dict[str, Any]] = []
    for deteccion in detecciones:
        copia = dict(deteccion)
        copia["score"] = float(rejilla[int(np.abs(rejilla - deteccion["score"]).argmin())])
        aplastadas.append(copia)
    return aplastadas


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("12_score_resolution", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = args.output or (rutas.results_raw / "artifact_eval" / "score_resolution.json")

    print("=" * 78)
    print("PASO 12: COSTE AISLADO DE LA RESOLUCIÓN DE PUNTUACIÓN")
    print("=" * 78)

    try:
        completo = json.loads(coco.read_text(encoding="utf-8"))
        nombres = runner.load_split_file_names(rutas.data_splits, args.split, completo)
        referencia = coco_eval.subset_ground_truth(completo, nombres)
        fp32 = cargar_detecciones(args.reference)
        int8 = cargar_detecciones(args.quantized)
    except (TFMEdgeError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    imagenes = rutas.data / f"resized_{args.resolution}" / "images"
    niveles = sorted({float(d["score"]) for d in int8})
    print(f"\nRejilla del artefacto cuantizado: {len(niveles)} niveles distintos")
    print(f"  {[round(n, 5) for n in niveles[:20]]}{' ...' if len(niveles) > 20 else ''}")

    print("\nGUARDA: la reevaluación del FP32 tiene que reproducir el registro")
    try:
        base = evaluar(fp32, referencia, imagenes, args.split, args.resolution).primary
    except TFMEdgeError as error:
        # map_detections_to_ground_truth rechaza categorías inexistentes; la comparación con el
        # registro cubre el caso en que el desfase deja los índices en rango.
        print(
            f"\nERROR: la reevaluación del FP32 ha fallado: {error}\n"
            "Revisa la traducción de la numeración de categorías.",
            file=sys.stderr,
        )
        return 2
    print(f"  mAP@50    obtenido {base.map_50:.6f}   esperado {args.expected_map_50:.6f}")
    print(f"  mAP@50-95 obtenido {base.map_50_95:.6f}   esperado {args.expected_map_50_95:.6f}")
    if (
        abs(base.map_50 - args.expected_map_50) > GUARD_TOLERANCE
        or abs(base.map_50_95 - args.expected_map_50_95) > GUARD_TOLERANCE
    ):
        print(
            "\nERROR: la reevaluación no reproduce el registro. Se aborta sin escribir nada.",
            file=sys.stderr,
        )
        return 2
    print("  OK.")

    aplastadas = aplastar_puntuaciones(fp32, niveles)
    cuenta = Counter(d["score"] for d in aplastadas)
    empatadas = sum(n for n in cuenta.values() if n > 1)
    ablacion = evaluar(aplastadas, referencia, imagenes, args.split, args.resolution).primary

    print(f"\nFP32 con la puntuación aplastada a {len(cuenta)} niveles")
    print(f"  detecciones empatadas: {100 * empatadas / len(aplastadas):.1f}%")
    print(f"  mAP@50    {ablacion.map_50:.6f}   ({ablacion.map_50 - base.map_50:+.6f})")
    print(f"  mAP@50-95 {ablacion.map_50_95:.6f}   ({ablacion.map_50_95 - base.map_50_95:+.6f})")

    informe = {
        "reference": str(args.reference),
        "quantized": str(args.quantized),
        "split": args.split,
        "input_resolution": args.resolution,
        "num_detections": len(fp32),
        "num_levels": len(niveles),
        "levels": [round(n, 8) for n in niveles],
        "tied_fraction": round(empatadas / len(aplastadas), 6),
        "reference_map_50": round(float(base.map_50), 6),
        "reference_map_50_95": round(float(base.map_50_95), 6),
        "ablated_map_50": round(float(ablacion.map_50), 6),
        "ablated_map_50_95": round(float(ablacion.map_50_95), 6),
        "cost_map_50": round(float(base.map_50 - ablacion.map_50), 6),
        "cost_map_50_95": round(float(base.map_50_95 - ablacion.map_50_95), 6),
    }
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVolcado en {salida}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
