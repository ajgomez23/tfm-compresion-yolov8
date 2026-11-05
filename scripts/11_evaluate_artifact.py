#!/usr/bin/env python
"""Evalúa la calidad de un artefacto exportado (ONNX, IR de OpenVINO).

Usa el mismo validador de Ultralytics que con un checkpoint: mismo cargador, letterbox, NMS y
reescalado. Para ello se prepara una copia con los metadatos que Ultralytics espera
(``export/staging.py``), sin tocar el artefacto medido. Las métricas del validador se registran
junto a las de pycocotools; como salen de las mismas detecciones, la diferencia se debe solo al
evaluador.

Códigos de salida: 0 si el artefacto se evalúa y queda registrado, 1 si hay un error de
configuración, 2 si la evaluación falla.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    parse_experiment_id,
    relativize_dump_paths,
    validate_path_component,
)
from tfm_edge.common.registry import ExperimentRegistry, build_ap_class_columns  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.data.download import compute_sha256  # noqa: E402
from tfm_edge.data.smoke_subset import SMOKE_IMAGES, prepare_smoke_subset  # noqa: E402
from tfm_edge.evaluation import runner  # noqa: E402
from tfm_edge.evaluation.quality_notes import (  # noqa: E402
    CANONICAL_EVALUATOR,
    EVALUATOR_PREFIX,
    TECHNIQUES_REQUIRING_EXPLICIT_NOTE,
)

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="11_evaluate_artifact.py",
        description=(
            "Evalúa la calidad de un artefacto exportado con pycocotools y los dos criterios de "
            "tamaño, por el mismo camino que un checkpoint, y registra la diferencia frente al "
            "evaluador de Ultralytics."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # El INT8 de ONNX Runtime de la variante PTQ\n"
            "  python scripts/11_evaluate_artifact.py \\\n"
            "      --artifact results/exports/s_baseline_default_ptq/"
            "s_baseline_default_int8_onnxruntime.onnx \\\n"
            "      --runtime onnxruntime --precision int8 \\\n"
            "      --experiment-id s_ptq_int8-noexcl_5bcd71f0\n"
            "\n"
            "  # Comprobar el camino entero en segundos, sin tocar el registro\n"
            "  python scripts/11_evaluate_artifact.py --artifact ... "
            "--runtime openvino \\\n"
            "      --precision fp32 --experiment-id ... --smoke-test\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        metavar="RUTA",
        help=(
            "Artefacto a evaluar: el .onnx, o el .xml del IR de OpenVINO (o su "
            "directorio). No se modifica."
        ),
    )
    parser.add_argument(
        "--runtime",
        type=str,
        required=True,
        choices=["onnxruntime", "openvino"],
        help=(
            "Motor que lo ejecuta. TensorRT y TFLite no están disponibles en "
            "esta máquina y no tienen evaluación de calidad implementada."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["fp32", "fp16", "int8"],
        help="Precisión del artefacto. Va al registro tal cual.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        metavar="ID",
        help=(
            "Experimento al que pertenece el artefacto. Tiene que existir ya en "
            "experiments.csv: este script mide una variante, no la crea."
        ),
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
            "Resolución de inferencia (por defecto: %(default)s). Tiene que ser "
            "la de la exportación: el artefacto es de entrada fija."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Lote (por defecto: %(default)s). Las exportaciones del proyecto son "
            "de lote fijo 1; Ultralytics lo impone leyendo los metadatos."
        ),
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        metavar="C",
        help="Umbral de confianza (por defecto: %(default)s, el del protocolo COCO).",
    )
    parser.add_argument("--iou", type=float, default=0.7, metavar="I", help="Umbral de IoU.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        metavar="DEV",
        help="Dispositivo (por defecto: %(default)s).",
    )
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
        "--notes",
        type=str,
        default="",
        metavar="TEXTO",
        help=(
            "Texto que se añade a la nota de la fila. Obligatorio para: "
            f"{', '.join(sorted(TECHNIQUES_REQUIRING_EXPLICIT_NOTE))}."
        ),
    )
    parser.add_argument(
        "--no-per-class",
        action="store_true",
        help="No calcula el AP por clase.",
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="overwrite",
        choices=["skip", "overwrite", "version", "fail"],
        help="Qué hacer si ya hay fila (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--registry-dir", type=Path, default=None, metavar="RUTA", help="Directorio del registro."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde volcar el JSON detallado (por defecto: results/raw/artifact_eval).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Dónde dejar la copia preparada del artefacto (por defecto: "
            "results/exports/_staged). El original no se toca."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            f"Evalúa solo {SMOKE_IMAGES} imágenes para comprobar el camino. "
            "Implica --dry-run: las cifras que salen de aquí no son resultados."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Evalúa pero no escribe en el registro."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    add_overwrite_dump_argument(parser, "el volcado de la evaluacion")
    return parser


def validator_dir_for(
    output_dir: Path,
    experiment_id: str,
    runtime: str,
    precision: str,
    smoke_test: bool,
    split: str,
) -> Path:
    """Directorio donde el validador deja las detecciones crudas.

    Incluye experimento, partición y marca de prueba de humo para que dos evaluaciones no escriban
    el mismo ``predictions.json``, que es la entrada de ``12_score_resolution.py``. Los directorios
    antiguos sin partición son de val y no se renombran.
    """
    sufijo = "_smoke" if smoke_test else ""
    return output_dir / f"validator_{experiment_id}_{runtime}_{precision}_{split}{sufijo}"


def dump_path_for(
    output_dir: Path,
    experiment_id: str,
    runtime: str,
    precision: str,
    split: str,
    resolution: int,
    device_id: str,
    smoke_test: bool,
) -> Path:
    """Ruta del volcado de detalle, con partición y dispositivo en el nombre.

    Los volcados antiguos sin dispositivo no se renombran. Lanza InvalidPathComponentError si el
    nombre no es válido como fichero en Linux y Windows.
    """
    sufijo = "_smoke" if smoke_test else ""
    nombre = f"{experiment_id}_{runtime}_{precision}_{split}_{resolution}_{device_id}{sufijo}.json"
    validate_path_component(nombre)
    return output_dir / nombre


def artifact_files_fingerprint(artifact: Path) -> list[dict[str, Any]]:
    """Calcula el sha256 y el tamaño de cada fichero del artefacto original (no de la copia
    preparada).

    Un IR de OpenVINO son dos ficheros y se devuelven los dos.
    """
    if artifact.is_dir():
        ficheros = sorted(f for f in artifact.rglob("*") if f.is_file())
    elif artifact.suffix == ".xml":
        ficheros = [artifact, artifact.with_suffix(".bin")]
    else:
        ficheros = [artifact]
    return [
        {"file": f.name, "bytes": f.stat().st_size, "sha256": compute_sha256(f)} for f in ficheros
    ]


def imprimir_brecha(brecha: dict[str, Any]) -> None:
    """Imprime la comparación entre evaluadores."""
    print()
    print("Diferencia entre evaluadores, sobre las mismas detecciones:")
    print(f"  {'métrica':<12} {'pycocotools':>12} {'ultralytics':>12} {'brecha':>12}")

    def _fmt(valor: Any) -> str:
        return f"{valor:.4f}" if isinstance(valor, (int, float)) else str(valor or "—")

    for metrica in ("map_50", "map_50_95"):
        print(
            f"  {metrica:<12} {_fmt(brecha[f'pycocotools_{metrica}']):>12} "
            f"{_fmt(brecha[f'ultralytics_{metrica}']):>12} "
            f"{_fmt(brecha[f'gap_{metrica}']):>12}"
        )


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("11_evaluate_artifact", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = args.output_dir or (rutas.results_raw / "artifact_eval")
    preparacion = args.staging_dir or (rutas.results / "exports" / "_staged")
    # La prueba de humo no escribe en el registro.
    dry_run = args.dry_run or args.smoke_test

    # Se comprueba antes de evaluar para no perder la evaluación entera.
    try:
        destino = dump_path_for(
            salida,
            args.experiment_id,
            args.runtime,
            args.precision,
            args.split,
            args.resolution,
            args.device_id,
            args.smoke_test,
        )
        refuse_to_overwrite_dump(
            destino, overwrite=args.overwrite_dump, what="el volcado de la evaluación"
        )
    except (FileExistsError, TFMEdgeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("PASO 11: CALIDAD DE UN ARTEFACTO EXPORTADO")
    print("=" * 78)

    if not args.artifact.exists():
        print(f"ERROR: no existe el artefacto '{args.artifact}'.", file=sys.stderr)
        return 1
    if not coco.is_file():
        print(f"ERROR: no existe el COCO de referencia '{coco}'.", file=sys.stderr)
        return 1

    try:
        data_yaml = (
            Path(args.data)
            if args.data
            else resize.find_dataset_for_resolution(rutas.data, args.resolution)
        )
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Misma exigencia que en 17_register_quality.py, porque este paso también inscribe filas de esas
    # técnicas. Se comprueba también con --dry-run.
    try:
        tecnica = parse_experiment_id(args.experiment_id)["technique"]
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if tecnica in TECHNIQUES_REQUIRING_EXPLICIT_NOTE and not args.notes.strip():
        print(
            f"ERROR: la técnica '{tecnica}' exige --notes.\n"
            "  Del artefacto no se deduce, por ejemplo, si el podado se reajustó.\n"
            "  Ver docs/ESQUEMA_RESULTADOS.md, apartado 4.",
            file=sys.stderr,
        )
        return 1

    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)

    # El experimento tiene que existir para no dejar una fila de calidad huérfana.
    if not dry_run:
        existentes = {f["experiment_id"] for f in registro.read_rows("experiments")}
        if args.experiment_id not in existentes:
            print(
                f"ERROR: el experimento '{args.experiment_id}' no está en el "
                f"registro. Los que hay: {sorted(existentes)}",
                file=sys.stderr,
            )
            return 1

    splits_dir = rutas.data_splits
    if args.smoke_test:
        try:
            data_yaml, splits_dir = prepare_smoke_subset(
                data_yaml, splits_dir, args.split, SMOKE_IMAGES, salida / "_smoke_subset"
            )
        except TFMEdgeError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1

    directorio_validador = validator_dir_for(
        salida, args.experiment_id, args.runtime, args.precision, args.smoke_test, args.split
    )
    # Huella de lo que se mide, tomada antes de evaluar.
    try:
        huella = artifact_files_fingerprint(args.artifact)
    except FileNotFoundError as error:
        print(f"ERROR: el artefacto está incompleto: {error}", file=sys.stderr)
        return 1

    print(f"Artefacto : {args.artifact}")
    print(f"Motor     : {args.runtime}   precisión: {args.precision}")
    print(f"Experimento: {args.experiment_id}")
    print(f"Dataset   : {data_yaml}")
    print(f"Partición : {args.split}   resolución: {args.resolution} px")
    if args.smoke_test:
        print(f"PRUEBA DE HUMO: {SMOKE_IMAGES} imágenes. Las cifras NO son resultados.")
    print()

    try:
        informe = runner.evaluate_artifact(
            args.artifact,
            runtime=args.runtime,
            class_names=clases,
            data_config=data_yaml,
            coco_path=coco,
            splits_dir=splits_dir,
            staging_dir=preparacion,
            split=args.split,
            input_resolution=args.resolution,
            batch_size=args.batch_size,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_dir=directorio_validador,
            with_per_class=not args.no_per_class,
        )
    except TFMEdgeError as error:
        logger.error("La evaluación del artefacto ha fallado: %s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - depende del entorno
        logger.exception("La evaluación del artefacto ha fallado de forma inesperada")
        print(f"ERROR inesperado: {error}", file=sys.stderr)
        return 2

    resultado = informe["result"]
    print()
    print(resultado.summary())
    imprimir_brecha(informe["evaluator_gap"])

    destino.parent.mkdir(parents=True, exist_ok=True)
    # Las rutas se relativizan al serializar porque el volcado se versiona.
    destino.write_text(
        json.dumps(
            relativize_dump_paths(
                {
                    "artifact": informe["artifact"],
                    "artifact_files": huella,
                    "staged_artifact": informe["staged_artifact"],
                    # Así la relación entre este volcado y sus detecciones crudas queda en el propio
                    # fichero.
                    "validator_dir": str(directorio_validador),
                    "runtime": informe["runtime"],
                    "precision": args.precision,
                    "experiment_id": args.experiment_id,
                    "split": args.split,
                    "device_id": args.device_id,
                    "smoke_test": args.smoke_test,
                    "coco": resultado.to_dict(),
                    "ultralytics_metrics": informe["ultralytics_metrics"],
                    "evaluator_gap": informe["evaluator_gap"],
                    "num_raw_detections": informe["num_raw_detections"],
                    "num_matched_detections": informe["num_matched_detections"],
                    "rescaling": informe["rescaling"],
                    "eval_time_s": informe["eval_time_s"],
                }
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDetalle volcado en {destino}")

    if dry_run:
        print("\n(--dry-run: no se escribe en el registro)")
        return 0

    entorno = capture_environment(device_id=args.device_id, repo_dir=rutas.root)
    environment_id = persist_environment(entorno, registry_dir=args.registry_dir)

    fila: dict[str, Any] = {
        "experiment_id": args.experiment_id,
        "runtime": args.runtime,
        "precision": args.precision,
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
        # El evaluador va en notes para que cada fila diga de cuál viene, y la diferencia también
        # porque el esquema no tiene columnas para ella. No se usa build_quality_note porque la nota
        # de QAT dice que se mide FP32, y aquí se evalúa el INT8.
        "notes": (
            f"{EVALUATOR_PREFIX} Artefacto exportado evaluado por el mismo "
            f"validador que el checkpoint; brecha frente a ultralytics: "
            f"map_50={informe['evaluator_gap']['gap_map_50']}, "
            f"map_50_95={informe['evaluator_gap']['gap_map_50_95']}."
            + (f" {args.notes.strip()}" if args.notes.strip() else "")
        ),
    }
    for indice, columna in enumerate(build_ap_class_columns(clases)):
        datos = resultado.per_class.get(indice)
        fila[columna] = datos["ap_50_95"] if datos else ""

    registro.append_quality_metrics(fila, on_duplicate=args.on_duplicate)
    print(f"\nInscrito en el registro: {args.experiment_id} / {args.runtime} / {args.precision}")
    print()
    print(registro.validate().summary())
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
