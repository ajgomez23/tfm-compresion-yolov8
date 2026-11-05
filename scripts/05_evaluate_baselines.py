#!/usr/bin/env python
"""Evalúa los modelos de referencia con el validador de Ultralytics y los inscribe en el registro.

Registra mAP@50, mAP@50-95, precisión, recall y AP por clase. Las columnas de desglose por tamaño
quedan vacías: las rellena 06_evaluate_coco.py con los dos criterios de tamaño.

Códigos de salida: 0 si se evalúan e inscriben todos los modelos, 1 si hay un error de configuración
o nada que evaluar, 2 si alguno falla (los demás quedan registrados).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.environment import capture_environment, persist_environment  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import (  # noqa: E402
    ExperimentRegistry,
    build_ap_class_columns,
    utc_timestamp,
)
from tfm_edge.common.registry_lookup import preserve_registered_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.evaluation.identity import build_baseline_evaluation_identity  # noqa: E402
from tfm_edge.evaluation.quality_notes import ULTRALYTICS_EVALUATOR  # noqa: E402
from tfm_edge.models.complexity import measure_complexity  # noqa: E402
from tfm_edge.models.detector import VALID_SIZES, Detector  # noqa: E402
from tfm_edge.training.cost import locate_run_file, read_training_cost  # noqa: E402

logger = get_logger(__name__)

#: Columnas por tamaño que este script deja vacías.
SIZE_BREAKDOWN_COLUMNS = (
    "ap_small_inference_res",
    "ap_medium_inference_res",
    "ap_large_inference_res",
    "ap_small_original_res",
    "ap_medium_original_res",
    "ap_large_original_res",
    "ar_small_inference_res",
    "ar_medium_inference_res",
    "ar_large_inference_res",
    "ar_small_original_res",
    "ar_medium_original_res",
    "ar_large_original_res",
)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="05_evaluate_baselines.py",
        description=(
            "Evalúa los modelos de referencia ya entrenados y escribe sus "
            "métricas en results/registry/. Es idempotente: volver a lanzarlo no "
            "duplica filas."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Evaluar todo lo que haya entrenado, sobre validación\n"
            "  python scripts/05_evaluate_baselines.py\n"
            "\n"
            "  # Solo el baseline principal, sobre la partición de test\n"
            "  python scripts/05_evaluate_baselines.py --sizes s --split test\n"
            "\n"
            "  # Recalcular sobrescribiendo las filas anteriores\n"
            "  python scripts/05_evaluate_baselines.py --on-duplicate overwrite\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        choices=list(VALID_SIZES),
        metavar="TAM",
        help="Tamaños a evaluar. Por defecto, todos los que se encuentren entrenados.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de ejecuciones (por defecto: results/runs).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="RUTA",
        help="YAML de dataset (por defecto: el de data/resized_640).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Partición a evaluar (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Tamaño de lote de la evaluación (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        metavar="N",
        help="Resolución de evaluación (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        metavar="C",
        help=(
            "Umbral de confianza (por defecto: %(default)s). Es el del protocolo "
            "COCO: subirlo infla la precisión a costa del recall."
        ),
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        metavar="I",
        help="Umbral de IoU de la supresión de no máximos (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        metavar="DEV",
        help="Dispositivo. Vacío deja que Ultralytics elija.",
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="skip",
        choices=["skip", "overwrite", "version", "fail"],
        help=(
            "Qué hacer si el experimento ya está en el registro (por defecto: "
            "%(default)s, que permite reanudar sin repetir)."
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Directorio del registro (por defecto: results/registry). Usa otro al hacer pruebas."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evalúa y muestra los resultados, pero no escribe en el registro.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def localizar_checkpoints(runs_dir: Path, sizes: list[str] | None) -> dict[str, Path]:
    """Busca el mejor checkpoint de cada tamaño, en ``<run>/weights/`` o directamente en
    ``<run>/``."""
    encontrados: dict[str, Path] = {}
    for size in sizes or list(VALID_SIZES):
        directorio = runs_dir / f"{size}_baseline_default"
        for candidato in (
            directorio / "weights" / "best.pt",
            directorio / "weights" / "last.pt",
            directorio / "best.pt",
            directorio / "last.pt",
        ):
            if candidato.is_file():
                encontrados[size] = candidato
                break
        else:
            logger.debug("Sin checkpoint para yolov8%s en '%s'", size, directorio)
    return encontrados


def construir_fila_experimento(
    size: str, checkpoint: Path, complejidad: dict, args: argparse.Namespace, experiment_id: str
) -> dict:
    """Construye la fila de experiments.csv."""
    # Las claves de training_cost.json están en training/cost.py.
    coste = read_training_cost(checkpoint)
    # args.yaml puede estar junto al checkpoint o un nivel por encima; si no aparece, se anota la
    # ubicación canónica.
    configuracion = locate_run_file(checkpoint, "args.yaml")

    return {
        "experiment_id": experiment_id,
        # Un baseline no tiene padre: celda vacía.
        "parent_experiment_id": "",
        "timestamp": utc_timestamp(),
        "model_family": "yolov8",
        "model_size": size,
        "technique": "baseline",
        "technique_variant": "default",
        "config_hash": experiment_id.rsplit("_", 1)[-1],
        "config_path": str(configuracion) if configuracion else "",
        # El ratio de compresión de un baseline es 1.
        "compression_ratio": 1.0,
        "effective_params": complejidad["params_total"],
        "effective_params_ratio": 1.0,
        "precision": "fp32",
        **coste,
        "weights_path": str(checkpoint),
        "status": "ok",
        "notes": "",
    }


def construir_fila_calidad(
    experiment_id: str,
    metricas: dict,
    complejidad: dict,
    args: argparse.Namespace,
    class_names: list[str],
    environment_id: str,
    eval_seconds: float,
) -> dict:
    """Construye la fila de quality_metrics.csv con las columnas por tamaño vacías."""
    fila: dict = {
        "experiment_id": experiment_id,
        "runtime": "pytorch",
        "precision": "fp32",
        "device_id": args.device or "auto",
        "environment_id": environment_id,
        # Este script usa el validador de Ultralytics, no pycocotools, y la columna lo indica.
        "evaluator": ULTRALYTICS_EVALUATOR,
        "split": metricas["split"],
        "input_resolution": metricas["input_resolution"],
        "conf_threshold": metricas["conf_threshold"],
        "iou_threshold": metricas["iou_threshold"],
        "map_50": metricas["map_50"],
        "map_50_95": metricas["map_50_95"],
        # DetMetrics no expone AP@75 por separado; se deja vacío.
        "ap_75": "",
        "size_criterion_primary": "inference_resolution",
        "size_criterion_inference_resolution": metricas["input_resolution"],
        "ar_1": "",
        "ar_10": "",
        "ar_100": "",
        "precision_at_best_f1": metricas["mean_precision"],
        "recall_at_best_f1": metricas["mean_recall"],
        "f1_best": "",
        "error_classification": "",
        "error_localization": "",
        "error_both": "",
        "error_duplicate": "",
        "error_background": "",
        "error_missed": "",
        "num_images": "",
        "num_detections": "",
        "eval_time_s": round(eval_seconds, 2),
        "predictions_path": "",
        "status": "ok",
        # El evaluador va también en notes; estas filas no se usan en el capítulo 5 sin reevaluarlas
        # con 06_evaluate_coco.py.
        "notes": (
            "evaluator=ultralytics; desglose por tamaño de objeto pendiente de la "
            "evaluación COCO con los dos criterios de D-038."
        ),
    }
    for columna in SIZE_BREAKDOWN_COLUMNS:
        fila[columna] = ""

    # AP por clase con el mapeo canónico de índice a nombre.
    columnas_ap = build_ap_class_columns(class_names)
    por_clase = metricas["per_class"]
    for indice, columna in enumerate(columnas_ap):
        datos = por_clase.get(indice)
        fila[columna] = datos["ap_50_95"] if datos else ""
    return fila


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("05_evaluate_baselines", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    runs = args.runs_dir or (rutas.results / "runs")

    try:
        data_yaml = (
            Path(args.data)
            if args.data
            else resize.find_dataset_for_resolution(rutas.data, args.resolution)
        )
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("PASO 05: EVALUACIÓN DE LOS MODELOS DE REFERENCIA")
    print("=" * 78)
    print(f"Ejecuciones: {runs}")
    print(f"Dataset    : {data_yaml}   partición: {args.split}")
    print(f"Resolución : {args.resolution} px   conf: {args.conf}   IoU: {args.iou}")
    print()

    if not data_yaml.is_file():
        print(f"ERROR: no existe el YAML de dataset '{data_yaml}'.", file=sys.stderr)
        return 1

    checkpoints = localizar_checkpoints(runs, args.sizes)
    if not checkpoints:
        print(
            f"ERROR: no hay ningún modelo entrenado en '{runs}'.\n"
            "Entrena primero con scripts/04_train_baselines.py.",
            file=sys.stderr,
        )
        return 1
    print(f"Modelos encontrados: {sorted(checkpoints)}\n")

    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)
    # El entorno se inscribe primero porque las filas de calidad lo referencian.
    entorno = capture_environment(device_id=args.device or "auto", repo_dir=rutas.root)
    environment_id = (
        entorno["environment_id"]
        if args.dry_run
        else persist_environment(entorno, registry_dir=args.registry_dir)
    )
    print(f"Entorno    : {environment_id}\n")

    evaluados: list[dict] = []
    fallidos: list[str] = []
    for size, checkpoint in sorted(checkpoints.items()):
        print("-" * 78)
        print(f"yolov8{size}: {checkpoint}")
        try:
            detector = Detector.load(checkpoint, input_resolution=args.resolution)
            complejidad = measure_complexity(
                detector, resolution=args.resolution, checkpoint_path=checkpoint
            )
            inicio = time.perf_counter()
            metricas = detector.validate(
                data_yaml,
                split=args.split,
                batch_size=args.batch_size,
                device=args.device,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                save_dir=checkpoint.parent.parent / f"eval_{args.split}",
            )
            transcurrido = time.perf_counter() - inicio

            # El hash sale de la configuración efectiva de la evaluación. La fórmula está en
            # evaluation/identity.py y la comparte el paso 06, que debe dar el mismo identificador.
            _, config_hash, experiment_id = build_baseline_evaluation_identity(
                model_size=size,
                num_classes=len(clases),
                input_resolution=args.resolution,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                split=args.split,
                data=data_yaml.name,
                seed=args.seed,
            )
            print(
                f"  mAP@50 = {metricas['map_50']:.4f}   "
                f"mAP@50-95 = {metricas['map_50_95']:.4f}   "
                f"({transcurrido:.1f} s)"
            )

            if args.dry_run:
                evaluados.append({"size": size, "metrics": metricas})
                continue

            registro.append_experiment(
                # Se reinscribe la fila sin cambiar sus rutas, igual que en el paso 06.
                preserve_registered_paths(
                    registro,
                    construir_fila_experimento(
                        size, checkpoint, complejidad.to_dict(), args, experiment_id
                    ),
                ),
                on_duplicate=args.on_duplicate,
            )
            registro.append_quality_metrics(
                construir_fila_calidad(
                    experiment_id,
                    metricas,
                    complejidad.to_dict(),
                    args,
                    clases,
                    environment_id,
                    transcurrido,
                ),
                on_duplicate=args.on_duplicate,
            )
            evaluados.append({"size": size, "experiment_id": experiment_id, "metrics": metricas})
        except TFMEdgeError as error:
            logger.error("yolov8%s ha fallado: %s", size, error)
            print(f"  ERROR: {error}", file=sys.stderr)
            fallidos.append(size)
        except Exception as error:  # pragma: no cover - depende del entorno
            logger.exception("yolov8%s ha fallado de forma inesperada", size)
            print(f"  ERROR inesperado: {error}", file=sys.stderr)
            fallidos.append(size)

    print()
    print("=" * 78)
    print(f"Evaluados: {len(evaluados)}   fallidos: {len(fallidos)}")
    if not args.dry_run and evaluados:
        print()
        print(registro.validate().summary())
    if args.dry_run:
        print("--dry-run: no se ha escrito en el registro.")
    print("=" * 78)
    return 2 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
