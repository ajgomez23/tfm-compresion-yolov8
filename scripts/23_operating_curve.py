#!/usr/bin/env python
"""Curva de precisión y recall a umbral de confianza fijo, sin volver a inferir.

Las columnas precision_at_best_f1 y recall_at_best_f1 se miden en el umbral óptimo de cada modelo,
que cambia entre FP32 e INT8; aquí se miden todos en la misma rejilla. La evaluación se reconstruye
desde las detecciones guardadas en validator_*/predictions.json, y antes de escribir se comprueba
que reproduce a seis cifras el mAP@50 y el punto de operación registrados. No escribe en el
registro; las curvas van a sus volcados.

Códigos de salida: 0 si se miden y escriben las curvas, 1 si hay un error de configuración, 2 si la
reconstrucción no reproduce lo registrado o la medida falla.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths, relativize_dump_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.evaluation import coco_eval, operating_curve, reconstruction  # noqa: E402
from tfm_edge.evaluation.runner import load_split_file_names  # noqa: E402

logger = get_logger(__name__)

#: Motores y precisiones para trocear el nombre de un directorio de validador.
KNOWN_RUNTIMES = reconstruction.KNOWN_RUNTIMES
KNOWN_PRECISIONS = reconstruction.KNOWN_PRECISIONS


class Objetivo:
    """Evaluación ya hecha de la que se quiere la curva."""

    def __init__(
        self, experiment_id: str, runtime: str, precision: str, validator_dir: Path, dump: Path
    ) -> None:
        self.experiment_id = experiment_id
        self.runtime = runtime
        self.precision = precision
        self.validator_dir = validator_dir
        self.dump = dump

    @property
    def etiqueta(self) -> str:
        """Nombre corto para logs y ficheros."""
        return f"{self.experiment_id}_{self.runtime}_{self.precision}"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="23_operating_curve.py",
        description=(
            "Mide precisión y recall en una rejilla fija de umbrales reconstruyendo la "
            "evaluación desde las detecciones guardadas. Comprueba que reproduce lo registrado "
            "antes de escribir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Toda la rejilla de combinaciones y sus referencias FP32\n"
            "  python scripts/23_operating_curve.py --all\n\n"
            "  # Una sola evaluación, con el directorio del validador a mano\n"
            "  python scripts/23_operating_curve.py \\\n"
            "      --experiment-id s_baseline_default_745e3d08 \\\n"
            "      --runtime onnxruntime --precision fp32 \\\n"
            "      --validator-dir results/raw/artifact_eval/validator_onnxruntime_fp32\n"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Mide todas las evaluaciones cuyo directorio de validador lleva el experiment_id y "
            "tienen volcado registrado."
        ),
    )
    parser.add_argument("--experiment-id", type=str, default=None, help="Experimento a medir.")
    parser.add_argument(
        "--runtime", type=str, default=None, choices=list(KNOWN_RUNTIMES), help="Motor."
    )
    parser.add_argument(
        "--precision", type=str, default=None, choices=list(KNOWN_PRECISIONS), help="Precisión."
    )
    parser.add_argument(
        "--validator-dir",
        type=Path,
        default=None,
        help=(
            "Directorio con predictions.json, si no se puede deducir del experiment_id (los "
            "volcados antiguos no lo llevan en el nombre)."
        ),
    )
    parser.add_argument(
        "--confidences",
        type=float,
        nargs="+",
        default=list(operating_curve.DEFAULT_CONFIDENCES),
        metavar="C",
        help="Rejilla de umbrales (por defecto: 0,05 a 0,90 de 0,05 en 0,05).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directorio de las evaluaciones (por defecto: results/raw/artifact_eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dónde volcar las curvas (por defecto: results/raw/operating_curve).",
    )
    parser.add_argument(
        "--on-existing",
        type=str,
        default="fail",
        choices=["fail", "skip", "overwrite"],
        help=("Qué hacer si la curva ya existe (por defecto: %(default)s)."),
    )
    parser.add_argument("--coco", type=Path, default=None, help="COCO de referencia.")
    parser.add_argument("--data", type=Path, default=None, help="YAML de dataset.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--resolution", type=int, default=640, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def trocear_directorio(nombre: str) -> tuple[str, str, str] | None:
    """Trocea ``validator_{experiment_id}_{runtime}_{precision}`` desde la derecha, porque el
    experiment_id puede llevar guiones bajos.
    """
    if not nombre.startswith("validator_"):
        return None
    partes = nombre[len("validator_") :].split("_")
    if len(partes) < 3:
        return None
    if partes[-1] not in KNOWN_PRECISIONS or partes[-2] not in KNOWN_RUNTIMES:
        return None
    return "_".join(partes[:-2]), partes[-2], partes[-1]


def descubrir_objetivos(eval_dir: Path, split: str, resolucion: int) -> list[Objetivo]:
    """Evaluaciones con detecciones y volcado registrado, ordenadas por etiqueta.

    Sin volcado no hay con qué comprobar la reconstrucción.
    """
    objetivos: list[Objetivo] = []
    for directorio in sorted(eval_dir.glob("validator_*")):
        if not directorio.is_dir():
            continue
        troceado = trocear_directorio(directorio.name)
        if troceado is None:
            logger.debug("Se ignora '%s': el nombre no lleva experiment_id", directorio.name)
            continue
        experimento, motor, precision = troceado
        if not (directorio / "predictions.json").is_file():
            logger.warning("'%s' no tiene predictions.json; se ignora", directorio.name)
            continue
        volcado = eval_dir / f"{experimento}_{motor}_{precision}_{split}_{resolucion}.json"
        if not volcado.is_file():
            logger.warning(
                "'%s' no tiene volcado registrado (%s); sin ancla no se mide",
                directorio.name,
                volcado.name,
            )
            continue
        objetivos.append(Objetivo(experimento, motor, precision, directorio, volcado))
    return objetivos


def medir_objetivo(
    objetivo: Objetivo,
    referencia_completa: dict[str, Any],
    splits_dir: Path,
    images_dir: Path,
    split: str,
    resolucion: int,
    confidences: list[float],
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Reconstruye una evaluación y mide su curva a umbral fijo."""
    inicio = time.perf_counter()
    volcado = json.loads(objetivo.dump.read_text(encoding="utf-8"))

    nombres = load_split_file_names(splits_dir, split, referencia_completa)
    referencia = coco_eval.subset_ground_truth(referencia_completa, nombres)

    crudas = json.loads((objetivo.validator_dir / "predictions.json").read_text(encoding="utf-8"))
    logger.info("%s: %s detecciones en disco", objetivo.etiqueta, f"{len(crudas):,}")

    reconstruida = reconstruction.rebuild_evaluation(
        crudas, volcado, referencia, images_dir, resolucion, len(class_names)
    )
    logger.info(
        "%s: anclada contra el volcado registrado (peor diferencia %.1e)",
        objetivo.etiqueta,
        reconstruida.anchor["max_difference"],
    )

    curva = operating_curve.precision_recall_at_confidences(
        reconstruida.evaluator, confidences=confidences, class_names=class_names
    )
    return {
        "experiment_id": objetivo.experiment_id,
        "runtime": objetivo.runtime,
        "precision": objetivo.precision,
        "split": split,
        "input_resolution": resolucion,
        "size_criterion": "inference_resolution",
        "num_detections": reconstruida.num_detections,
        "source": {
            "predictions": str(objetivo.validator_dir / "predictions.json"),
            "recorded_dump": str(objetivo.dump),
            "note": (
                "Las detecciones NO se han vuelto a generar: son las de la "
                "evaluación ya registrada, leídas otra vez en umbrales fijos."
            ),
        },
        "anchor": reconstruida.anchor,
        "curve": curva,
        "elapsed_s": round(time.perf_counter() - inicio, 2),
    }


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("23_operating_curve", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    eval_dir = args.eval_dir or (rutas.results_raw / "artifact_eval")
    salida = args.output_dir or (rutas.results_raw / "operating_curve")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")

    print("=" * 78)
    print("PASO 23: PRECISIÓN Y RECALL A UMBRAL FIJO")
    print("=" * 78)

    if not coco.is_file():
        print(f"ERROR: no existe el COCO de referencia '{coco}'.", file=sys.stderr)
        return 1
    if not eval_dir.is_dir():
        print(f"ERROR: no existe el directorio de evaluaciones '{eval_dir}'.", file=sys.stderr)
        return 1

    try:
        data_yaml = args.data or resize.find_dataset_for_resolution(rutas.data, args.resolution)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Qué se va a medir
    if args.all:
        objetivos = descubrir_objetivos(eval_dir, args.split, args.resolution)
        if not objetivos:
            print("ERROR: no se ha encontrado ninguna evaluación anclable.", file=sys.stderr)
            return 1
    else:
        if not (args.experiment_id and args.runtime and args.precision):
            print(
                "ERROR: sin --all hay que dar --experiment-id, --runtime y --precision.",
                file=sys.stderr,
            )
            return 1
        directorio = args.validator_dir or (
            eval_dir / f"validator_{args.experiment_id}_{args.runtime}_{args.precision}"
        )
        volcado = (
            eval_dir / f"{args.experiment_id}_{args.runtime}_{args.precision}_"
            f"{args.split}_{args.resolution}.json"
        )
        for ruta, que in ((directorio / "predictions.json", "detecciones"), (volcado, "volcado")):
            if not ruta.is_file():
                print(f"ERROR: no existe el {que} '{ruta}'.", file=sys.stderr)
                return 1
        objetivos = [
            Objetivo(args.experiment_id, args.runtime, args.precision, directorio, volcado)
        ]

    # load_class_names devuelve una lista en orden de índice.
    clases = dict(
        enumerate(load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json"))
    )
    print(f"\nEvaluaciones a medir: {len(objetivos)}")
    print(f"Umbrales: {', '.join(f'{c:.2f}' for c in sorted(args.confidences))}")
    print(f"Salida:   {salida}\n")

    referencia_completa = json.loads(coco.read_text(encoding="utf-8"))
    images_dir = Path(data_yaml).parent / "images"
    salida.mkdir(parents=True, exist_ok=True)

    medidas, saltadas, fallidas = 0, 0, 0
    for objetivo in objetivos:
        destino = salida / f"{objetivo.etiqueta}_{args.split}_{args.resolution}.json"
        if destino.exists():
            if args.on_existing == "skip":
                logger.info("%s: ya volcada, se salta", objetivo.etiqueta)
                saltadas += 1
                continue
            if args.on_existing == "fail":
                print(
                    f"ERROR: '{destino}' ya existe. Usa --on-existing skip para continuar u "
                    "overwrite para rehacerla.",
                    file=sys.stderr,
                )
                return 1

        try:
            resultado = medir_objetivo(
                objetivo,
                referencia_completa,
                rutas.data_splits,
                images_dir,
                args.split,
                args.resolution,
                list(args.confidences),
                clases,
            )
        except TFMEdgeError as error:
            print(f"FALLIDA {objetivo.etiqueta}: {error}", file=sys.stderr)
            fallidas += 1
            continue

        temporal = destino.with_suffix(".json.tmp")
        # La procedencia va en el volcado con rutas relativas, porque se versiona.
        temporal.write_text(
            json.dumps(relativize_dump_paths(resultado), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporal.replace(destino)
        medidas += 1

        referencia_punto = operating_curve.point_at(
            resultado["curve"], operating_curve.REFERENCE_CONFIDENCE
        )
        print(
            f"  {objetivo.etiqueta}: "
            f"P={referencia_punto['precision']:.4f} "
            f"R={referencia_punto['recall']:.4f} "
            f"en conf={operating_curve.REFERENCE_CONFIDENCE:.2f} "
            f"({resultado['elapsed_s']:.0f} s)"
        )

    print("\n" + "-" * 78)
    print(f"Medidas: {medidas}   Saltadas: {saltadas}   Fallidas: {fallidas}")
    print("-" * 78)
    if fallidas:
        print(
            "\nAlguna reconstrucción no reproduce lo registrado; sus curvas no se han escrito.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
