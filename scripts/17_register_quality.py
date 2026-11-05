#!/usr/bin/env python
"""Evalúa un checkpoint existente e inscribe su fila de calidad.

Separa medir la calidad de producir los pesos: si la evaluación falla al final de un entrenamiento,
basta con volver a evaluar el checkpoint, que es determinista. No escribe en experiments.csv, así
que el experimento tiene que estar inscrito. Para la poda exige ``--notes``, porque los pesos no
dicen si el modelo se reajustó.

Antes de evaluar comprueba que el experimento existe con status ok, que el checkpoint tiene los
parámetros que declara la fila, que todos están en FP32, que la cuantización simulada está
desactivada y que no hay ya una fila con métricas para esa clave.

Con ``--dry-run`` no se escriben la fila ni el volcado de detalle, pero sí el log, el directorio de
trabajo del validador (``results/raw/quality/_val_{experiment_id}_{split}/``) y, con
``--smoke-test``, el subconjunto en ``results/raw/quality/_smoke_subset/``.

Códigos de salida: 0 si la fila queda inscrita (o se simula con --dry-run), 1 si hay un error de
uso, de configuración o de comprobación previa, 2 si la evaluación falla.
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

import torch  # noqa: E402

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.environment import capture_environment, persist_environment  # noqa: E402
from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import (  # noqa: E402
    ensure_dir,
    get_project_paths,
    parse_experiment_id,
    validate_path_component,
)
from tfm_edge.common.registry import ExperimentRegistry, build_ap_class_columns  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.compression.quantization.finetune import (  # noqa: E402
    count_active_fake_quant_modules,
    count_fake_quant_modules,
)
from tfm_edge.data.download import compute_sha256  # noqa: E402
from tfm_edge.data.smoke_subset import SMOKE_IMAGES, prepare_smoke_subset  # noqa: E402
from tfm_edge.evaluation import runner  # noqa: E402
from tfm_edge.evaluation.quality_notes import (  # noqa: E402
    CANONICAL_EVALUATOR,
    TECHNIQUES_REQUIRING_EXPLICIT_NOTE,
    build_quality_note,
)
from tfm_edge.models.detector import Detector  # noqa: E402

logger = get_logger(__name__)

#: La fila describe siempre un checkpoint de PyTorch; las de artefactos exportados las escribe
#: 11_evaluate_artifact.py (ver docs/ESQUEMA_RESULTADOS.md).
RUNTIME = "pytorch"

#: Precisión de la fila, comprobada contra los pesos antes de escribirla.
PRECISION = "fp32"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos."""
    parser = argparse.ArgumentParser(
        prog="17_register_quality.py",
        description=(
            "Evalúa un checkpoint con pycocotools y los dos criterios de tamaño e inscribe su "
            "fila de calidad en un experimento que ya existe. No crea el experimento, no "
            "entrena y no exporta."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Inscribir la fila de calidad de un checkpoint de QAT\n"
            "  python scripts/17_register_quality.py \\\n"
            "      --checkpoint models/qat/s_qat_int8-noexcl_d4d403fa/"
            "qat_stripped_fp32.pt \\\n"
            "      --experiment-id s_qat_int8-noexcl_d4d403fa \\\n"
            "      --device-id workstation_cpu\n"
            "\n"
            "  # Ver qué saldría, sin tocar el registro\n"
            "  python scripts/17_register_quality.py ... --dry-run\n"
            "\n"
            "  # Prueba con unas pocas imágenes\n"
            "  python scripts/17_register_quality.py ... --smoke-test\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Pesos a evaluar. Tienen que ser los del experimento que se indica.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        metavar="ID",
        help=(
            "Experimento al que pertenece la fila. Tiene que estar ya inscrito en "
            "experiments.csv: este script no lo crea."
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
        metavar="PX",
        help="Resolución de inferencia (por defecto: %(default)s).",
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
    parser.add_argument(
        "--device-id",
        type=str,
        default="workstation_cpu",
        metavar="ID",
        help=(
            "Identificador de dispositivo del registro (por defecto: %(default)s). Forma parte "
            "de la clave de la fila."
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
    parser.add_argument("--no-per-class", action="store_true", help="No calcula el AP por clase.")
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="fail",
        choices=["fail", "skip", "overwrite"],
        help=(
            "Qué hacer si ya hay fila con esa clave (por defecto: %(default)s). Una fila con "
            "métricas no se sobrescribe ni con 'overwrite'."
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
        help="Dónde volcar el detalle (por defecto: results/raw/quality).",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Evalúa y muestra la fila sin escribirla ni escribir el volcado de detalle. El log "
            "y el directorio del validador sí se escriben."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Subconjunto de {SMOKE_IMAGES} imágenes. Implica --dry-run.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    add_overwrite_dump_argument(parser, "el volcado de calidad")
    return parser


def validator_dir_for(output_dir: Path, experiment_id: str, split: str) -> Path:
    """Directorio de trabajo del validador de Ultralytics, con la partición en el nombre."""
    return output_dir / f"_val_{experiment_id}_{split}"


def dump_path_for(
    output_dir: Path, experiment_id: str, split: str, resolution: int, device_id: str
) -> Path:
    """Ruta del volcado de detalle, con partición y dispositivo en el nombre.

    Los volcados antiguos sin dispositivo no se renombran. Lanza InvalidPathComponentError si el
    nombre no es válido como fichero en Linux y Windows.
    """
    nombre = f"{experiment_id}_{split}_{resolution}_{device_id}.json"
    validate_path_component(nombre)
    return output_dir / nombre


def buscar_fila_de_experimento(
    registro: ExperimentRegistry, experiment_id: str
) -> dict[str, str] | None:
    """Devuelve la fila de experiments.csv de ese experimento, o ``None`` si no está inscrito."""
    for fila in registro.read_rows("experiments"):
        if fila.get("experiment_id") == experiment_id:
            return fila
    return None


def buscar_fila_de_calidad(
    registro: ExperimentRegistry, experiment_id: str, device_id: str, split: str
) -> dict[str, str] | None:
    """Devuelve la fila de calidad con la clave ``(experiment_id, runtime, precision, device_id,
    split)``, o ``None``.
    """
    for fila in registro.read_rows("quality_metrics"):
        if (
            fila.get("experiment_id") == experiment_id
            and fila.get("runtime") == RUNTIME
            and fila.get("precision") == PRECISION
            and fila.get("device_id") == device_id
            and fila.get("split") == split
        ):
            return fila
    return None


def comprobar_el_tamano(
    detector: Detector, fila_experimento: dict[str, str], experiment_id: str
) -> list[str]:
    """Comprueba el tamaño deducido de la arquitectura contra el prefijo del ``experiment_id`` y
    contra la columna ``model_size`` de la fila, por separado. Devuelve los motivos de rechazo.
    """
    motivos: list[str] = []
    deducido = detector.size

    del_id = parse_experiment_id(experiment_id)["model_size"]
    if del_id != deducido:
        motivos.append(
            f"la arquitectura del checkpoint es de tamaño '{deducido}' y el "
            f"experiment_id dice '{del_id}'. O el checkpoint no es el de ese "
            f"experimento, o el identificador se construyó con un tamaño incorrecto."
        )

    de_la_fila = (fila_experimento.get("model_size") or "").strip()
    if not de_la_fila:
        motivos.append(
            "la fila del experimento no declara 'model_size'. Aquí no es una "
            "medida ausente sino parte de la identidad, y sin ella la fila de "
            "calidad no se puede atribuir a un tamaño."
        )
    elif de_la_fila != deducido:
        motivos.append(
            f"la arquitectura del checkpoint es de tamaño '{deducido}' y la "
            f"columna 'model_size' de su fila dice '{de_la_fila}'."
        )

    return motivos


def comprobar_el_checkpoint(
    detector: Detector, fila_experimento: dict[str, str], experiment_id: str
) -> list[str]:
    """Comprueba que el checkpoint corresponde a la fila del experimento y devuelve los motivos de
    rechazo, si los hay.
    """
    motivos: list[str] = []
    modelo = detector.model

    motivos.extend(comprobar_el_tamano(detector, fila_experimento, experiment_id))

    parametros = detector.count_parameters()
    declarados = (fila_experimento.get("effective_params") or "").strip()
    if not declarados:
        # Vacío significa no medido, y entonces no hay con qué contrastar.
        logger.warning(
            "La fila del experimento no declara 'effective_params': no se puede "
            "contrastar que estos pesos sean los suyos."
        )
    elif int(float(declarados)) != parametros:
        motivos.append(
            f"el checkpoint tiene {parametros:,} parámetros y la fila del "
            f"experimento declara {int(float(declarados)):,}. O no son los pesos "
            f"de ese experimento, o la fila está mal."
        )

    no_fp32 = {
        nombre: str(parametro.dtype)
        for nombre, parametro in modelo.named_parameters()
        if parametro.dtype != torch.float32
    }
    if no_fp32:
        primero = next(iter(no_fp32.items()))
        motivos.append(
            f"hay {len(no_fp32)} parámetros que no están en FP32 (por ejemplo "
            f"'{primero[0]}' en {primero[1]}), y la fila diría 'precision=fp32'. "
            f"Convertirlos aquí no recuperaría la precisión perdida."
        )

    nodos = count_fake_quant_modules(modelo)
    activos = count_active_fake_quant_modules(modelo)
    if nodos:
        logger.info(
            "El modelo trae %d nodos de cuantización simulada; activos: %d.", nodos, activos
        )
    if activos:
        motivos.append(
            f"{activos} de los {nodos} nodos de cuantización simulada siguen "
            f"activos: esta medida no sería la de los pesos FP32. Evalúa el "
            f"checkpoint que deja strip_fake_quantization()."
        )

    return motivos


def construir_fila_calidad(
    experiment_id: str,
    informe_eval: dict[str, Any],
    clases: list[str],
    device_id: str,
    environment_id: str,
    ruta_detalle: Path,
    nota: str,
) -> dict[str, Any]:
    """Compone la fila de quality_metrics."""
    resultado = informe_eval["result"]
    fila: dict[str, Any] = {
        "experiment_id": experiment_id,
        "runtime": RUNTIME,
        "precision": PRECISION,
        "device_id": device_id,
        "environment_id": environment_id,
        # Sale de la misma constante que el prefijo de notes.
        "evaluator": CANONICAL_EVALUATOR,
        **informe_eval["registry_columns"],
        "num_detections": informe_eval["num_matched_detections"],
        "error_classification": "",
        "error_localization": "",
        "error_both": "",
        "error_duplicate": "",
        "error_background": "",
        "error_missed": "",
        "predictions_path": str(ruta_detalle),
        "status": "ok",
        "notes": nota,
    }
    for indice, columna in enumerate(build_ap_class_columns(clases)):
        datos = resultado.per_class.get(indice)
        fila[columna] = datos["ap_50_95"] if datos else ""
    return fila


def _abortar(mensaje: str) -> int:
    """Imprime un error de uso y devuelve 1."""
    print(f"ERROR: {mensaje}", file=sys.stderr)
    return 1


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("17_register_quality", level=args.log_level)
    set_global_seed(args.seed)
    rutas = get_project_paths()

    dry_run = args.dry_run or args.smoke_test
    if args.smoke_test:
        logger.warning("--smoke-test implica --dry-run: no se escribe en el registro.")

    print("=" * 78)
    print("PASO 17: INSCRIPCIÓN DE LA FILA DE CALIDAD DE UN CHECKPOINT")
    print("=" * 78)

    if not args.checkpoint.is_file():
        return _abortar(f"no existe el checkpoint '{args.checkpoint}'.")

    try:
        componentes = parse_experiment_id(args.experiment_id)
    except TFMEdgeError as error:
        return _abortar(str(error))
    tecnica = componentes["technique"]

    if tecnica in TECHNIQUES_REQUIRING_EXPLICIT_NOTE and not args.notes.strip():
        return _abortar(
            f"la técnica '{tecnica}' exige --notes.\n"
            f"  Su cifra no se puede interpretar sin saber algo que NO está en los "
            f"pesos: un modelo podado sin reajustar y otro reajustado dan ficheros "
            f"indistinguibles y significan cosas opuestas.\n"
            f"  Ver docs/ESQUEMA_RESULTADOS.md, apartado 4."
        )

    data_yaml = args.data or (rutas.root / "data" / "resized_640" / "sh17_ultralytics.yaml")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = args.output_dir or (rutas.results_raw / "quality")
    # Se comprueba el destino antes de evaluar, con la misma guarda que los demás productores.
    try:
        detalle = dump_path_for(
            salida, args.experiment_id, args.split, args.resolution, args.device_id
        )
    except TFMEdgeError as error:
        return _abortar(str(error))
    if not dry_run:
        try:
            refuse_to_overwrite_dump(
                detalle, overwrite=args.overwrite_dump, what="el volcado de calidad"
            )
        except FileExistsError as error:
            return _abortar(str(error))
    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
    splits_dir = rutas.data_splits

    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)

    # Comprobaciones previas a la medida.
    fila_experimento = buscar_fila_de_experimento(registro, args.experiment_id)
    if fila_experimento is None:
        return _abortar(
            f"'{args.experiment_id}' no está en experiments.csv, y este script no "
            f"lo crea.\n"
            f"  La configuración, el entrenamiento y el experimento padre no se "
            f"pueden deducir del .pt.\n"
            f"  Si la campaña corrió en Colab, trae su fila con "
            f"scripts/14_import_colab_registry.py."
        )
    estado = fila_experimento.get("status", "")
    if estado != "ok":
        return _abortar(
            f"'{args.experiment_id}' está inscrito con status='{estado}'. Una fila "
            f"de calidad de un experimento que no terminó bien afirmaría lo "
            f"contrario que su propia fila de experimento."
        )

    previa = buscar_fila_de_calidad(registro, args.experiment_id, args.device_id, args.split)
    if previa is not None and not dry_run:
        if (previa.get("map_50_95") or "").strip():
            return _abortar(
                f"ya hay una fila de calidad medida para esa clave "
                f"(mAP@50-95 = {previa['map_50_95']}).\n"
                f"  Para rehacerla, bórrala del registro a mano o usa otro "
                f"--registry-dir."
            )
        if args.on_duplicate == "fail":
            return _abortar(
                "ya hay una fila con esa clave (sin métricas) y la política es "
                "'fail'. Con --on-duplicate overwrite se sustituye."
            )

    if args.smoke_test:
        data_yaml, splits_dir = prepare_smoke_subset(
            data_yaml, splits_dir, args.split, SMOKE_IMAGES, salida / "_smoke_subset"
        )
        print(f"PRUEBA DE HUMO: {SMOKE_IMAGES} imágenes. Las cifras NO son resultados.")

    try:
        runner.assert_evaluation_inputs_available(coco, splits_dir, args.split, data_yaml)
    except TFMEdgeError as error:
        return _abortar(str(error))

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Experimento: {args.experiment_id}  (técnica '{tecnica}', status '{estado}')")
    print(f"Dataset    : {data_yaml}")
    print(f"Partición  : {args.split}   resolución: {args.resolution} px")
    print()

    try:
        detector = Detector.load(args.checkpoint, input_resolution=args.resolution)
    except TFMEdgeError as error:
        return _abortar(f"no se puede cargar el checkpoint: {error}")
    except Exception as error:  # un .pt ilegible sale de torch, no de aquí
        # Se captura cualquier excepción porque aquí llegan errores de torch y pickle.
        return _abortar(
            f"no se puede cargar el checkpoint '{args.checkpoint}' "
            f"({type(error).__name__}: {error}).\n"
            f"  Comprueba que es un .pt de PyTorch completo: un fichero truncado "
            f"por una descarga a medias da exactamente este error."
        )

    motivos = comprobar_el_checkpoint(detector, fila_experimento, args.experiment_id)
    if motivos:
        return _abortar(
            "el checkpoint no cuadra con el experimento indicado:\n  - " + "\n  - ".join(motivos)
        )
    print(f"Parámetros : {detector.count_parameters():,} (coinciden con la fila del experimento)")
    # La huella de los pesos medidos va al volcado.
    huella = compute_sha256(args.checkpoint)
    print(f"Huella     : sha256 {huella}")
    print()

    # La medida.
    try:
        informe = runner.evaluate_checkpoint(
            args.checkpoint,
            data_config=data_yaml,
            coco_path=coco,
            splits_dir=splits_dir,
            split=args.split,
            input_resolution=args.resolution,
            batch_size=args.batch_size,
            device=args.device,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save_dir=validator_dir_for(salida, args.experiment_id, args.split),
            with_per_class=not args.no_per_class,
        )
    except TFMEdgeError as error:
        logger.error("La evaluación ha fallado: %s", error)
        print(f"\nLA EVALUACIÓN HA FALLADO: {error}", file=sys.stderr)
        return 2

    resultado = informe["result"]
    print()
    print(resultado.summary())

    # El destino ya se comprobó antes de evaluar.
    if dry_run:
        print(f"\n(--dry-run: no se vuelca el detalle en {detalle})")
    else:
        ensure_dir(salida)
        detalle.write_text(
            json.dumps(
                {
                    "experiment_id": args.experiment_id,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": huella,
                    "checkpoint_bytes": args.checkpoint.stat().st_size,
                    "parameters": detector.count_parameters(),
                    "split": args.split,
                    "device_id": args.device_id,
                    "input_resolution": args.resolution,
                    "conf_threshold": args.conf,
                    "iou_threshold": args.iou,
                    "seed": args.seed,
                    "registered_by": "17_register_quality.py",
                    "result": resultado.to_dict(),
                    "rescaling": informe["rescaling"],
                    "num_raw_detections": informe["num_raw_detections"],
                    "num_matched_detections": informe["num_matched_detections"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nDetalle volcado en {detalle}")

    nota = build_quality_note(
        tecnica,
        provenance=(
            f"Fila inscrita por 17_register_quality.py reevaluando "
            f"'{args.checkpoint.name}': la evaluación es determinista dados los "
            f"pesos, la partición, la resolución y los umbrales, así que es una "
            f"REMEDICIÓN del mismo artefacto (D-090)."
        ),
        extra=args.notes.strip(),
    )

    if dry_run:
        print("\n(--dry-run: no se escribe en el registro)")
        print(f"La nota habría sido:\n  {nota}")
        return 0

    entorno = capture_environment(device_id=args.device_id, repo_dir=rutas.root)
    environment_id = persist_environment(entorno, registry_dir=args.registry_dir)

    registro.append_quality_metrics(
        construir_fila_calidad(
            experiment_id=args.experiment_id,
            informe_eval=informe,
            clases=clases,
            device_id=args.device_id,
            environment_id=environment_id,
            ruta_detalle=detalle,
            nota=nota,
        ),
        on_duplicate=args.on_duplicate,
    )
    print(f"\nInscrita la fila de calidad de {args.experiment_id}.")
    print()
    print(registro.validate().summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
