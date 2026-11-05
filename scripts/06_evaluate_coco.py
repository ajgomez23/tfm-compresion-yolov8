#!/usr/bin/env python
"""Evaluación COCO con los dos criterios de tamaño de objeto.

Es el único script que rellena las doce columnas de desglose por tamaño. Genera las detecciones una
vez en coordenadas originales y evalúa dos veces: tal cual (resolución original) y con referencia y
detecciones escaladas por el factor de letterbox (resolución de inferencia, el primario). El mAP
global tiene que coincidir en las dos evaluaciones; si no, falla.

Códigos de salida: 0 si se evalúan todos los modelos, 1 si hay un error de configuración, 2 si
alguno falla (los demás quedan registrados).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.environment import capture_environment, persist_environment  # noqa: E402
from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import (  # noqa: E402
    get_project_paths,
    relative_to_root,
    validate_path_component,
)
from tfm_edge.common.registry import (  # noqa: E402
    ExperimentRegistry,
    build_ap_class_columns,
    utc_timestamp,
)
from tfm_edge.common.registry_lookup import preserve_registered_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.evaluation import runner  # noqa: E402
from tfm_edge.evaluation.identity import build_baseline_evaluation_identity  # noqa: E402
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR  # noqa: E402
from tfm_edge.models.complexity import measure_complexity  # noqa: E402
from tfm_edge.models.detector import VALID_SIZES, Detector  # noqa: E402
from tfm_edge.training.cost import locate_run_file, read_training_cost  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="06_evaluate_coco.py",
        description=(
            "Evalúa con pycocotools con los dos criterios de tamaño de objeto y rellena las "
            "columnas de desglose por tamaño del registro."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # El baseline principal, sobre validación\n"
            "  python scripts/06_evaluate_coco.py --sizes s\n"
            "\n"
            "  # Todo lo entrenado, sobre la partición de test\n"
            "  python scripts/06_evaluate_coco.py --split test\n"
            "\n"
            "  # Ver el resultado sin tocar el registro\n"
            "  python scripts/06_evaluate_coco.py --sizes s --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=None,
        choices=list(VALID_SIZES),
        metavar="TAM",
        help="Tamaños a evaluar. Por defecto, todos los que se encuentren.",
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None, metavar="RUTA", help="Directorio de ejecuciones."
    )
    parser.add_argument("--data", type=Path, default=None, metavar="RUTA", help="YAML de dataset.")
    parser.add_argument(
        "--coco",
        type=Path,
        default=None,
        metavar="RUTA",
        help="COCO de referencia (por defecto: data/processed/sh17_coco.json).",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"], help="Partición a evaluar."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        metavar="N",
        help=(
            "Resolución de inferencia (por defecto: %(default)s). Determina el "
            "criterio de tamaño primario, así que va al registro."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16, metavar="N", help="Lote.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        metavar="C",
        help="Umbral de confianza (por defecto: %(default)s, el del protocolo COCO).",
    )
    parser.add_argument("--iou", type=float, default=0.7, metavar="I", help="Umbral de IoU.")
    parser.add_argument("--device", type=str, default="", metavar="DEV", help="Dispositivo.")
    # --device es el dispositivo de torch (vacío para que elija Ultralytics); --device-id es el
    # identificador del registro y debe coincidir con el de deployment_metrics.
    parser.add_argument(
        "--device-id",
        type=str,
        default="workstation_cpu",
        metavar="ID",
        help=(
            "Identificador de dispositivo del registro (por defecto: %(default)s). Debe "
            "coincidir con el de deployment_metrics."
        ),
    )
    parser.add_argument(
        "--no-per-class",
        action="store_true",
        help="No calcula el AP por clase. Ahorra una evaluación por clase.",
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="overwrite",
        choices=["skip", "overwrite", "version", "fail"],
        help=(
            "Qué hacer si ya hay fila (por defecto: %(default)s, porque este "
            "script COMPLETA las columnas que el 05 dejó vacías)."
        ),
    )
    parser.add_argument(
        "--registry-dir", type=Path, default=None, metavar="RUTA", help="Directorio del registro."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde volcar el JSON detallado (por defecto: results/raw/coco_eval).",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "No escribe en el registro, pero sí el volcado de detalle. Nunca sobrescribe un "
            "volcado existente."
        ),
    )
    add_overwrite_dump_argument(parser, "un volcado de detalle")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    return parser


def localizar_checkpoints(runs_dir: Path, sizes: list[str] | None) -> dict[str, Path]:
    """Busca el checkpoint de cada tamaño, en ``<run>/weights/`` o directamente en ``<run>/``."""
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
    return encontrados


def build_dump_name(size: str, split: str, resolution: int, device_id: str) -> str:
    """Nombre del volcado de detalle de una evaluación, con el device_id para que dos dispositivos
    no escriban el mismo fichero.

    Los volcados antiguos llevan el nombre sin device_id y no se renombran.
    """
    nombre = f"{size}_baseline_{split}_{resolution}_{device_id}.json"
    validate_path_component(nombre)
    return nombre


def volcar_detalle(
    result: Any,
    *,
    output_dir: Path,
    size: str,
    split: str,
    resolution: int,
    device_id: str,
    environment_id: str,
    checkpoint: Path | str,
    experiment_id: str,
    identity_payload: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Escribe el volcado de detalle de una evaluación sin sobrescribir ninguno."""
    destino = output_dir / build_dump_name(size, split, resolution, device_id)
    # La protección está en common/evidence.py.
    refuse_to_overwrite_dump(
        destino,
        overwrite=overwrite,
        what="el volcado de una medida anterior con este mismo dispositivo",
    )
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                **result.to_dict(),
                # Experimento al que pertenece la medida; el nombre del fichero no lo dice.
                "experiment_id": experiment_id,
                # El payload exacto que se hashea; un test comprueba que su hash da el sufijo del
                # experiment_id. Identifica la evaluación, no el entrenamiento.
                "experiment_identity_payload": dict(identity_payload),
                # La procedencia va dentro del volcado, no solo en el nombre.
                "provenance": {
                    "device_id": device_id,
                    "environment_id": environment_id,
                    "evaluator": CANONICAL_EVALUATOR,
                    "checkpoint": str(checkpoint),
                    "split": split,
                    "input_resolution": resolution,
                    "measured_at_utc": utc_timestamp(),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return destino


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("06_evaluate_coco", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    runs = args.runs_dir or (rutas.results / "runs")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = args.output_dir or (rutas.results_raw / "coco_eval")

    print("=" * 78)
    print("PASO 06: EVALUACIÓN COCO CON LOS DOS CRITERIOS DE TAMAÑO")
    print("=" * 78)

    try:
        data_yaml = (
            Path(args.data)
            if args.data
            else resize.find_dataset_for_resolution(rutas.data, args.resolution)
        )
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Dataset  : {data_yaml}")
    print(f"COCO     : {coco}")
    print(f"Partición: {args.split}   resolución: {args.resolution} px")
    print()

    if not coco.is_file():
        print(f"ERROR: no existe el COCO de referencia '{coco}'.", file=sys.stderr)
        return 1

    checkpoints = localizar_checkpoints(runs, args.sizes)
    if not checkpoints:
        print(
            f"ERROR: no hay ningún modelo entrenado en '{runs}'.\n"
            "Entrena con scripts/04_train_baselines.py.",
            file=sys.stderr,
        )
        return 1
    print(f"Modelos: {sorted(checkpoints)}\n")

    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)
    entorno = capture_environment(device_id=args.device_id, repo_dir=rutas.root)
    environment_id = (
        entorno["environment_id"]
        if args.dry_run
        else persist_environment(entorno, registry_dir=args.registry_dir)
    )

    fallidos: list[str] = []
    for size, checkpoint in sorted(checkpoints.items()):
        print("-" * 78)
        print(f"yolov8{size}: {checkpoint}")
        try:
            informe = runner.evaluate_checkpoint(
                checkpoint,
                data_config=data_yaml,
                coco_path=coco,
                splits_dir=rutas.data_splits,
                split=args.split,
                input_resolution=args.resolution,
                batch_size=args.batch_size,
                device=args.device,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                save_dir=checkpoint.parent.parent / f"coco_{args.split}",
                with_per_class=not args.no_per_class,
            )
            resultado = informe["result"]
            print()
            print(resultado.summary())

            # La identidad se calcula antes del volcado, que la incluye. La fórmula es la misma que
            # en el paso 05 (evaluation/identity.py).
            payload_identidad, config_hash, experiment_id = build_baseline_evaluation_identity(
                model_size=size,
                num_classes=len(clases),
                input_resolution=args.resolution,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                split=args.split,
                data=data_yaml.name,
                seed=args.seed,
            )

            try:
                destino = volcar_detalle(
                    resultado,
                    output_dir=salida,
                    size=size,
                    split=args.split,
                    resolution=args.resolution,
                    device_id=args.device_id,
                    environment_id=environment_id,
                    checkpoint=relative_to_root(checkpoint, rutas),
                    experiment_id=experiment_id,
                    identity_payload=payload_identidad,
                    overwrite=args.overwrite_dump,
                )
            except FileExistsError as error:
                logger.error("[%s] %s", size, error)
                print(f"ERROR: {error}", file=sys.stderr)
                fallidos.append(f"{size}: {error}")
                continue
            print(f"\nDetalle volcado en {destino}")

            if args.dry_run:
                continue

            fila = {
                "experiment_id": experiment_id,
                "runtime": "pytorch",
                "precision": "fp32",
                "device_id": args.device_id,
                "environment_id": environment_id,
                # Sale de la misma constante que el prefijo de notes.
                "evaluator": CANONICAL_EVALUATOR,
                **informe["registry_columns"],
                "num_detections": informe["num_matched_detections"],
                "error_classification": "",
                "error_localization": "",
                "error_both": "",
                "error_duplicate": "",
                "error_background": "",
                "error_missed": "",
                "predictions_path": str(destino),
                "status": "ok",
                # El evaluador va en notes para que cada fila diga de cuál viene.
                "notes": (
                    "evaluator=pycocotools; evaluación COCO con los dos criterios "
                    "de tamaño (D-038)."
                ),
            }
            for indice, columna in enumerate(build_ap_class_columns(clases)):
                datos = resultado.per_class.get(indice)
                fila[columna] = datos["ap_50_95"] if datos else ""

            # La fila de experimento tiene que existir antes que la de calidad. Se sobrescribe para
            # poder completarla (por ejemplo, con el coste de entrenamiento traído de Colab); trae
            # todo lo que traía la del paso 05.
            complejidad = measure_complexity(
                Detector.load(checkpoint, input_resolution=args.resolution),
                resolution=args.resolution,
                checkpoint_path=checkpoint,
            )
            fila_experimento = {
                "experiment_id": experiment_id,
                "parent_experiment_id": "",
                "timestamp": utc_timestamp(),
                "model_family": "yolov8",
                "model_size": size,
                "technique": "baseline",
                "technique_variant": "default",
                "config_hash": config_hash,
                "config_path": (
                    str(locate_run_file(checkpoint, "args.yaml"))
                    if locate_run_file(checkpoint, "args.yaml")
                    else ""
                ),
                "compression_ratio": 1.0,
                "effective_params": complejidad.params_total,
                "effective_params_ratio": 1.0,
                "precision": "fp32",
                # El coste sale de training_cost.json; si no está, las columnas quedan vacías.
                **read_training_cost(checkpoint),
                "weights_path": str(checkpoint),
                "status": "ok",
                "notes": "",
            }
            # Se completa la fila, pero sus rutas no se cambian por las de esta máquina.
            registro.append_experiment(
                preserve_registered_paths(registro, fila_experimento),
                on_duplicate="overwrite",
            )
            registro.append_quality_metrics(fila, on_duplicate=args.on_duplicate)
            print(f"Inscrito en el registro como {experiment_id}")

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
    print(f"Evaluados: {len(checkpoints) - len(fallidos)}   fallidos: {len(fallidos)}")
    if not args.dry_run and len(fallidos) < len(checkpoints):
        print()
        print(registro.validate().summary())
    print("=" * 78)
    return 2 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
