#!/usr/bin/env python
"""Afinado consciente de la cuantización (QAT) sobre unos pesos ya entrenados.

Produce pesos FP32 afinados con cuantización simulada, no un modelo INT8. El INT8 lo materializa
después cada runtime por el mismo camino y con el mismo conjunto de calibración que la rama PTQ; la
conversión eager de PyTorch no es viable en YOLOv8 porque no hay kernel cuantizado para SiLU. Aquí
se mide el coste en FP32 de entrenar con el ruido de cuantización; la comparación entre ramas en
INT8 la hace el paso 11.

No afina en CPU salvo que se pida, aborta si los observadores no han visto datos y no sobrescribe un
volcado con métricas por otro sin ellas. El ``experiment_id`` incluye la huella SHA-256 del
checkpoint de partida y el ``--parent-experiment-id``.

Códigos de salida: 0 si el afinado termina y queda inscrito, 1 si hay un error de configuración o de
uso, 2 si el afinado falla (queda registrado con su motivo).
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

import torch  # noqa: E402

from tfm_edge.common.config import (  # noqa: E402
    ProjectConfig,
    compute_config_hash,
    experiment_identity_payload,
    load_class_names,
    load_config,
)
from tfm_edge.common.environment import capture_environment, persist_environment  # noqa: E402
from tfm_edge.common.exceptions import RegistryAmbiguousRowError, TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import build_experiment_id, ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import (  # noqa: E402
    ExperimentRegistry,
    build_ap_class_columns,
    utc_timestamp,
)
from tfm_edge.common.registry_lookup import read_reference_quality  # noqa: E402
from tfm_edge.common.seed import derive_seed, set_global_seed  # noqa: E402
from tfm_edge.compression.quantization.finetune import finetune_qat  # noqa: E402
from tfm_edge.data.download import compute_sha256  # noqa: E402
from tfm_edge.data.smoke_subset import SMOKE_IMAGES, prepare_smoke_subset  # noqa: E402
from tfm_edge.evaluation import runner  # noqa: E402
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR, build_quality_note  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402
from tfm_edge.training.cost import REGISTRY_COLUMNS, cost_dict_to_registry  # noqa: E402
from tfm_edge.training.trainer import CLOSE_MOSAIC_POR_DEFECTO, mosaic_regime  # noqa: E402

logger = get_logger(__name__)

#: Esquema de siembra por rama, igual que en poda. Entra en el hash para que campañas con distinto
#: esquema no compartan experiment_id; v2 pasa también la semilla de rama a Ultralytics.
SEEDING_SCHEME = "per_branch_v2"

#: El calentamiento de Ultralytics tarda 3 épocas en llegar a la tasa configurada, el 30 % de un
#: afinado de 10 sobre pesos ya convergidos, así que se desactiva. Ultralytics lo admite: 'nw = ...
#: if warmup_epochs > 0 else -1'.
WARMUP_EPOCHS_QAT = 0.0


def construir_identidad(
    config: ProjectConfig, huella_checkpoint: str, parent_experiment_id: str
) -> tuple[dict[str, Any], str, str]:
    """Construye el payload que identifica un afinado QAT y devuelve ``(payload, config_hash,
    experiment_id)``.

    Incluye la huella del contenido del checkpoint (no la ruta, para que sea portable), el
    ``parent_experiment_id`` declarado y las secciones de ``EXPERIMENT_IDENTITY_SECTIONS`` de la
    configuración.
    """
    payload = {
        **experiment_identity_payload(config),
        "seeding_scheme": SEEDING_SCHEME,
        # El calentamiento desactivado no está en el esquema, así que se añade al hash aquí.
        "warmup_epochs": WARMUP_EPOCHS_QAT,
        # Punto de partida.
        "starting_checkpoint_sha256": huella_checkpoint,
        "parent_experiment_id": parent_experiment_id,
    }
    config_hash = compute_config_hash(payload)
    experiment_id = build_experiment_id(
        model_size=config.model.size,
        technique="qat",
        variant=config.technique_variant,
        config_hash=config_hash,
    )
    return payload, config_hash, experiment_id


class CheckpointEvaluator:
    """Evalúa checkpoints con pycocotools y guarda el informe completo en ``reports``.

    Comprueba sus entradas al construirse para no descubrir que falta algo después del afinado.
    """

    def __init__(
        self,
        data_config: Path,
        coco_path: Path,
        splits_dir: Path,
        split: str,
        input_resolution: int,
        batch_size: int,
        device: str,
        with_per_class: bool,
    ) -> None:
        """Configura el evaluador. Lanza EvaluationRunnerError si falta el COCO de referencia, la
        lista de la partición o el YAML del dataset.
        """
        runner.assert_evaluation_inputs_available(coco_path, splits_dir, split, data_config)
        self.data_config = data_config
        self.coco_path = coco_path
        self.splits_dir = splits_dir
        self.split = split
        self.input_resolution = input_resolution
        self.batch_size = batch_size
        self.device = device
        self.with_per_class = with_per_class
        self.reports: dict[str, dict[str, Any]] = {}

    def __call__(self, checkpoint: Path) -> dict[str, float]:
        """Evalúa un checkpoint y devuelve las dos métricas principales."""
        logger.info("Evaluando '%s' por el camino canónico (pycocotools)", checkpoint.name)
        informe = runner.evaluate_checkpoint(
            checkpoint,
            data_config=self.data_config,
            coco_path=self.coco_path,
            splits_dir=self.splits_dir,
            split=self.split,
            input_resolution=self.input_resolution,
            batch_size=self.batch_size,
            device=self.device,
            with_per_class=self.with_per_class,
        )
        self.reports[str(checkpoint)] = informe
        primario = informe["result"].primary
        return {"map_50": primario.map_50, "map_50_95": primario.map_50_95}


def construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos."""
    parser = argparse.ArgumentParser(
        prog="16_qat.py",
        description=(
            "Afina un checkpoint con cuantización simulada (QAT) y lo inscribe en el registro. "
            "Produce pesos FP32 que siguen el mismo camino de materialización INT8 que los del "
            "baseline."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Campaña completa en GPU\n"
            "  python scripts/16_qat.py \\\n"
            "      --checkpoint models/baselines/s_baseline_default/best.pt \\\n"
            "      --parent-experiment-id s_baseline_default_745e3d08\n"
            "\n"
            "  # Ensayo del camino, sin escribir en el registro\n"
            "  python scripts/16_qat.py --checkpoint ... --smoke-test\n"
            "\n"
            "  # Solo preparar y comprobar, sin gastar GPU\n"
            "  python scripts/16_qat.py --checkpoint ... --override "
            "quantization.qat_epochs=0 --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="RUTA",
        help=(
            "Pesos de partida: el baseline FP32 (el mismo que usa PTQ) o el modelo del que se "
            "parta. Su SHA-256 entra en el experiment_id."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/compression/quantization/qat_default.yaml"),
        metavar="RUTA",
        help="Configuración de QAT (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="CLAVE=VALOR",
        help="Sobrescribe claves de la configuración con notación de punto.",
    )
    parser.add_argument(
        "--parent-experiment-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Experimento del que se toma el baseline de referencia. Su métrica se lee del "
            "registro, no se reevalúa."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde dejar los checkpoints (por defecto: models/qat).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde volcar el detalle (por defecto: results/raw/qat).",
    )
    parser.add_argument(
        "--registry-dir", type=Path, default=None, metavar="RUTA", help="Registro de resultados."
    )
    parser.add_argument("--data", type=Path, default=None, metavar="RUTA", help="YAML de dataset.")
    parser.add_argument(
        "--coco", type=Path, default=None, metavar="RUTA", help="COCO de referencia."
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="PX", help="Resolución de inferencia."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Lote del afinado y de la evaluación. Por defecto, training.batch_size de la "
            "configuración. Equivale a --override training.batch_size=N y entra en el "
            "experiment_id."
        ),
    )
    parser.add_argument("--device", type=str, default="", metavar="DEV", help="Dispositivo.")
    parser.add_argument(
        "--device-id",
        type=str,
        default="colab_t4_gpu",
        metavar="ID",
        help="Identificador de dispositivo para el registro (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cargadores de datos. Por defecto, training.workers de la configuración. Equivale a "
            "--override training.workers=N y entra en el experiment_id."
        ),
    )
    parser.add_argument(
        "--allow-mosaic-free",
        action="store_true",
        help=(
            "Permite el régimen en el que ningún lote lleva mosaico (épocas == close_mosaic), "
            "que es el del afinado QAT de 10 épocas. Sin esta opción el Trainer lo rechaza."
        ),
    )
    parser.add_argument("--no-evaluate", action="store_true", help="No mide nada.")
    parser.add_argument("--no-per-class", action="store_true", help="No calcula el AP por clase.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="No comprueba si hay una ejecución anterior a medias.",
    )
    parser.add_argument(
        "--allow-cpu-finetune",
        action="store_true",
        help=(
            "Permite afinar sin GPU. Solo para probar el camino: un QAT en CPU no "
            "es lento, es inviable."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help="Semilla efectiva del experimento. Sobrescribe la de la configuración.",
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="fail",
        choices=["fail", "skip", "overwrite"],
        help="Qué hacer si el experimento ya está inscrito (por defecto: %(default)s).",
    )
    parser.add_argument("--dry-run", action="store_true", help="No escribe en el registro.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=f"Subconjunto de {SMOKE_IMAGES} imágenes. Implica --dry-run.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def _degradaria_la_evidencia(detalle: Path, informe: Any) -> bool:
    """Indica si sobrescribir ese volcado cambiaría métricas medidas por valores vacíos."""
    if not detalle.is_file():
        return False
    if informe.metrics_baseline or informe.metrics_after:
        return False  # esta ejecución ha medido y puede sustituirlo
    try:
        anterior = json.loads(detalle.read_text(encoding="utf-8")).get("qat", {})
    except (OSError, json.JSONDecodeError):
        # Un volcado ilegible no es evidencia que proteger.
        return False
    return bool(anterior.get("metrics_baseline") or anterior.get("metrics_after"))


def volcar_detalle(
    output_dir: Path,
    *,
    experiment_id: str,
    checkpoint: Path | str,
    config_path: Path | str,
    smoke_test: bool,
    config: ProjectConfig,
    seed_branch: int,
    identity_payload: Mapping[str, Any],
    informe: Any,
    baseline_reference: Any,
    dry_run: bool,
) -> Path | None:
    """Escribe el volcado crudo del afinado, salvo con ``dry_run`` o si sustituiría métricas por
    vacío.
    """
    destino = output_dir / f"{experiment_id}.json"
    if dry_run:
        print(f"\n(--dry-run: no se vuelca el detalle en {destino})")
        return None
    if _degradaria_la_evidencia(destino, informe):
        print(
            "\n"
            f"No se sobrescribe '{destino}': la campaña anterior tenía métricas medidas y esta "
            "no.\n"
            "  Para descartar aquella medida, borra el fichero a mano.",
            file=sys.stderr,
        )
        return None

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "checkpoint": str(checkpoint),
                "config_path": str(config_path),
                "smoke_test": smoke_test,
                "seed_base": config.seed,
                "seed_branch": seed_branch,
                "seeding_scheme": SEEDING_SCHEME,
                "warmup_epochs": WARMUP_EPOCHS_QAT,
                "effective_config": config.quantization.model_dump(mode="json"),
                # El payload exacto que se hashea, con la huella del checkpoint de partida; un test
                # comprueba que su hash da el sufijo del experiment_id.
                "experiment_identity_payload": dict(identity_payload),
                "baseline_reference": baseline_reference,
                "qat": informe.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDetalle volcado en {destino}")
    return destino


def _fila_previa_del_experimento(
    registro: ExperimentRegistry, experiment_id: str
) -> dict[str, str] | None:
    """Devuelve la fila de experimentos con ese identificador, o ``None``."""
    for fila in registro.read_rows("experiments"):
        if fila.get("experiment_id") == experiment_id:
            return fila
    return None


def _inscribir_el_fallo(
    registro: ExperimentRegistry,
    fila: dict[str, Any],
    politica: str,
    previa: dict[str, str] | None,
) -> None:
    """Inscribe un afinado fallido sin lanzar excepciones ni sustituir una fila con resultado.

    Si la fila anterior también era un fallo, se sustituye.
    """
    if previa is not None and previa.get("status") != "fallido":
        print(
            "\n"
            f"El fallo no se ha inscrito: ya hay una fila '{previa.get('status')}' para este "
            "experimento.\n"
            "  Para descartarla, bórrala del registro a mano.",
            file=sys.stderr,
        )
        return

    if previa is not None and politica == "fail":
        # La anterior también era un fallo, así que se sustituye.
        politica = "overwrite"

    try:
        registro.append_experiment(fila, on_duplicate=politica)
    except TFMEdgeError as error:
        print(
            f"\nEl fallo NO se ha podido inscribir ({error}). El motivo real del "
            f"fallo es el de arriba, no este.",
            file=sys.stderr,
        )


def leer_baseline_del_registro(
    registro: ExperimentRegistry, parent_id: str, split: str, device_id: str
) -> dict[str, Any]:
    """Lee del registro la métrica FP32 del experimento de referencia.

    Filtra por el evaluador con el que mide este script y usa el dispositivo solo para desempatar.
    Devuelve ``map_50``, ``map_50_95``, ``device_id`` y ``evaluator``, o un diccionario vacío si no
    hay fila. Lanza RegistryAmbiguousRowError si quedan filas con métricas distintas.
    """
    return read_reference_quality(
        registro,
        parent_id,
        split,
        evaluator=CANONICAL_EVALUATOR,
        runtime="pytorch",
        preferred_device_id=device_id,
    )


def construir_fila_calidad(
    experiment_id: str,
    informe_eval: dict[str, Any],
    clases: list[str],
    device_id: str,
    environment_id: str,
    ruta_detalle: Path,
) -> dict[str, Any]:
    """Compone la fila de quality_metrics del modelo afinado con QAT."""
    resultado = informe_eval["result"]
    fila: dict[str, Any] = {
        "experiment_id": experiment_id,
        "runtime": "pytorch",
        # Se mide en FP32 sin los nodos de simulación; el INT8 lo produce después cada runtime.
        "precision": "fp32",
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
        # La nota está en evaluation/quality_notes.py porque también la usa el paso 17.
        "notes": build_quality_note("qat"),
    }
    for indice, columna in enumerate(build_ap_class_columns(clases)):
        datos = resultado.per_class.get(indice)
        fila[columna] = datos["ap_50_95"] if datos else ""
    return fila


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("16_qat", level=args.log_level)
    rutas = get_project_paths()

    dry_run = args.dry_run or args.smoke_test
    if args.smoke_test:
        logger.warning("--smoke-test implica --dry-run: no se escribe en el registro.")

    if not args.checkpoint.is_file():
        print(f"ERROR: no existe el checkpoint '{args.checkpoint}'.", file=sys.stderr)
        return 1
    if not args.config.is_file():
        print(f"ERROR: no existe la configuración '{args.config}'.", file=sys.stderr)
        return 1

    try:
        # --workers y --batch-size entran por la configuración para formar parte del config_hash.
        overrides = list(args.override)
        if args.workers is not None:
            overrides.append(f"training.workers={args.workers}")
        if args.batch_size is not None:
            overrides.append(f"training.batch_size={args.batch_size}")
        config: ProjectConfig = load_config(args.config, overrides=overrides)
    except TFMEdgeError as error:
        print(f"ERROR de configuración: {error}", file=sys.stderr)
        return 1

    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    # La semilla de rama se deriva de la base y de la técnica, igual que en poda, porque el reajuste
    # baraja y aumenta con el RNG global.
    semilla_rama = derive_seed(config.seed, "qat", config.quantization.mode)
    set_global_seed(semilla_rama)
    logger.info(
        "Semilla base %d, semilla de la rama %d (esquema '%s').",
        config.seed,
        semilla_rama,
        SEEDING_SCHEME,
    )

    afina = config.quantization.qat_epochs > 0
    if afina and not torch.cuda.is_available() and not args.allow_cpu_finetune:
        # No se pasa a CPU sin pedirlo.
        print(
            "ERROR: el afinado QAT necesita GPU y no hay ninguna disponible.\n"
            "  En CPU es inviable: el modelo lleva 64 observadores de peso y 121 de activación.\n"
            "  Ejecútalo en Colab (ver cloud/README.md). Para probar el camino aquí usa "
            "--allow-cpu-finetune, y para preparar sin entrenar, "
            "--override quantization.qat_epochs=0.",
            file=sys.stderr,
        )
        return 1

    regimen, descripcion_regimen = mosaic_regime(
        config.quantization.qat_epochs, CLOSE_MOSAIC_POR_DEFECTO
    )
    if afina and regimen == "sin_mosaico" and not args.allow_mosaic_free:
        # Misma comprobación que el Trainer, adelantada para no preparar el modelo en vano.
        print(
            f"ERROR: {descripcion_regimen}.\n"
            "  Es la receta del afinado QAT, no la de los baselines. Para usarla, pasa\n"
            "  --allow-mosaic-free.",
            file=sys.stderr,
        )
        return 1

    data_yaml = args.data or (rutas.root / "data" / "resized_640" / "sh17_ultralytics.yaml")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = ensure_dir(args.output_dir or (rutas.root / "models" / "qat"))
    crudo = ensure_dir(args.raw_dir or (rutas.results / "raw" / "qat"))
    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")

    splits_dir = rutas.data_splits
    if args.smoke_test:
        data_yaml, splits_dir = prepare_smoke_subset(
            data_yaml, splits_dir, args.split, SMOKE_IMAGES, crudo / "_smoke_subset"
        )
        print(f"PRUEBA DE HUMO: {SMOKE_IMAGES} imágenes. Las cifras NO son resultados.")

    huella_checkpoint = compute_sha256(args.checkpoint)
    payload_identidad, config_hash, experiment_id = construir_identidad(
        config, huella_checkpoint, args.parent_experiment_id
    )

    print("=" * 78)
    print("PASO 16: AFINADO CONSCIENTE DE LA CUANTIZACIÓN (QAT)")
    print("=" * 78)
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Huella       : sha256 {huella_checkpoint}")
    print(f"Padre        : {args.parent_experiment_id or '(no declarado)'}")
    print(f"Configuración: {args.config}")
    print(f"Experimento  : {experiment_id}")
    print(
        f"Afinado      : {config.quantization.qat_epochs} épocas, "
        f"tasa {config.quantization.qat_learning_rate:g}, SGD, sin calentamiento"
    )
    print(
        "Congelación  : observadores en la época " f"{config.quantization.freeze_observers_epoch}"
        if config.quantization.freeze_observers_epoch is not None
        else "Congelación  : ninguna"
    )
    print()
    print(
        "Este paso produce pesos FP32, no un modelo INT8. El INT8 lo\n"
        "materializa cada runtime por el mismo camino que PTQ."
    )
    print()

    evaluador = None
    if not args.no_evaluate:
        try:
            evaluador = CheckpointEvaluator(
                data_config=data_yaml,
                coco_path=coco,
                splits_dir=splits_dir,
                split=args.split,
                input_resolution=args.resolution,
                batch_size=config.training.batch_size,
                device=args.device,
                with_per_class=not args.no_per_class,
            )
        except TFMEdgeError as error:
            print(f"ERROR: no se puede evaluar: {error}", file=sys.stderr)
            return 1

    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)

    # Se comprueba aquí si el experimento ya está inscrito para no fallar después del afinado.
    fila_previa = _fila_previa_del_experimento(registro, experiment_id)
    if fila_previa is not None and not dry_run:
        estado_previo = fila_previa.get("status", "?")
        if args.on_duplicate == "fail":
            print(
                f"ERROR: '{experiment_id}' ya está inscrito con status='{estado_previo}' y la "
                "política de duplicados es 'fail'.\n"
                "  Si viene de un intento fallido, usa --on-duplicate overwrite.\n"
                "  Si la fila tiene una medición válida, usa otro --registry-dir.",
                file=sys.stderr,
            )
            return 1
        logger.warning(
            "'%s' ya está inscrito con status='%s'; al terminar se aplicará la " "política '%s'.",
            experiment_id,
            estado_previo,
            args.on_duplicate,
        )

    # El baseline se lee del registro. En modo humo no se usa, porque su cifra es de la partición
    # entera.
    baseline_metrics: dict[str, float] = {}
    # Fila de la que sale la referencia, para el volcado.
    referencia_baseline: dict[str, str] = {}
    if args.smoke_test and args.parent_experiment_id:
        logger.warning(
            "--smoke-test: NO se usa el baseline del registro. Su cifra es de la "
            "partición completa y aquí se miden %d imágenes.",
            SMOKE_IMAGES,
        )
    elif args.parent_experiment_id:
        try:
            fila_baseline = leer_baseline_del_registro(
                registro, args.parent_experiment_id, args.split, args.device_id
            )
        except RegistryAmbiguousRowError as error:
            # Se comprueba antes de cargar el modelo.
            print(f"ERROR: no se puede fijar la referencia FP32: {error}", file=sys.stderr)
            return 1
        if fila_baseline:
            baseline_metrics = {
                clave: fila_baseline[clave]
                for clave in ("map_50", "map_50_95")
                if clave in fila_baseline
            }
            referencia_baseline = {
                "experiment_id": args.parent_experiment_id,
                "evaluator": fila_baseline["evaluator"],
                "device_id": fila_baseline["device_id"],
            }
            print(
                f"Baseline de referencia ({args.parent_experiment_id}): "
                f"mAP@50-95 = {baseline_metrics['map_50_95']:.6f} "
                f"[{fila_baseline['evaluator']}, {fila_baseline['device_id'] or 'sin dispositivo'}]"
            )
            if fila_baseline["device_id"] != args.device_id:
                logger.warning(
                    "La referencia de '%s' se midió en '%s' y el afinado se inscribirá "
                    "como '%s': la diferencia frente al FP32 mezcla dos dispositivos.",
                    args.parent_experiment_id,
                    fila_baseline["device_id"] or "(sin declarar)",
                    args.device_id,
                )
        else:
            logger.warning(
                "No hay fila de calidad de '%s' con %s en el registro: la "
                "diferencia frente al FP32 quedará ausente, no a cero.",
                args.parent_experiment_id,
                CANONICAL_EVALUATOR,
            )

    detector = Detector.load(args.checkpoint, input_resolution=config.model.input_resolution)
    logger.info("Punto de partida: %s parámetros", f"{detector.count_parameters():,}")

    trainer_kwargs: dict[str, Any] = {
        # Sin esta clave Ultralytics usa su semilla 0 por defecto y el resultado no dependería de
        # --seed.
        "seed": semilla_rama,
        "batch_size": config.training.batch_size,
        "device": args.device,
        # Planificador, lrf, momentum, weight_decay y workers salen de la configuración que entra en
        # el hash.
        **config.training.trainer_recipe_kwargs(),
        "allow_mosaic_free": args.allow_mosaic_free,
        # Este entrenador no puede reanudar: con True, un afinado interrumpido falla al relanzarlo
        # en vez de empezar de cero.
        "resume_if_possible": not args.no_resume,
        # Se pasa como parámetro porque el Trainer ya envía warmup_epochs y lo rechazaría en
        # extra_args.
        "warmup_epochs": WARMUP_EPOCHS_QAT,
        # Modo determinista declarado en la configuración.
        "deterministic": config.deterministic,
    }

    try:
        _, informe = finetune_qat(
            detector,
            config.quantization,
            data_config=data_yaml,
            output_dir=salida / experiment_id,
            evaluator=evaluador,
            baseline_metrics=baseline_metrics,
            trainer_kwargs=trainer_kwargs,
        )
    except TFMEdgeError as error:
        # Si los observadores no vieron datos, lo entrenado no era QAT; se registra con su motivo.
        print(f"\nEL AFINADO QAT HA FALLADO: {error}", file=sys.stderr)
        if not dry_run:
            _inscribir_el_fallo(
                registro,
                {
                    "experiment_id": experiment_id,
                    "parent_experiment_id": args.parent_experiment_id,
                    "timestamp": utc_timestamp(),
                    "model_family": config.model.family,
                    "model_size": config.model.size,
                    "technique": "qat",
                    "technique_variant": config.technique_variant,
                    "config_hash": config_hash,
                    "config_path": str(args.config),
                    "compression_ratio": "",
                    "effective_params": "",
                    "effective_params_ratio": "",
                    "precision": "fp32",
                    "training_epochs": "",
                    "training_time_s": "",
                    "training_device": "",
                    "weights_path": "",
                    "status": "fallido",
                    "notes": str(error).replace("\n", " ")[:500],
                },
                politica=args.on_duplicate,
                previa=fila_previa,
            )
        return 2

    detalle = crudo / f"{experiment_id}.json"
    volcar_detalle(
        crudo,
        experiment_id=experiment_id,
        checkpoint=args.checkpoint,
        config_path=args.config,
        smoke_test=args.smoke_test,
        config=config,
        seed_branch=semilla_rama,
        identity_payload=payload_identidad,
        informe=informe,
        baseline_reference=referencia_baseline,
        dry_run=dry_run,
    )

    print()
    print(informe.summary())

    if dry_run:
        print("\n(--dry-run: no se escribe en el registro)")
        return 0

    entorno = capture_environment(device_id=args.device_id, repo_dir=rutas.root)
    environment_id = persist_environment(entorno, registry_dir=args.registry_dir)

    registro.append_experiment(
        {
            "experiment_id": experiment_id,
            "parent_experiment_id": args.parent_experiment_id,
            "timestamp": utc_timestamp(),
            "model_family": config.model.family,
            "model_size": config.model.size,
            "technique": "qat",
            "technique_variant": config.technique_variant,
            "config_hash": config_hash,
            "config_path": str(detalle),
            # QAT no quita parámetros; la compresión viene del INT8 posterior, así que el ratio se
            # deja vacío.
            "compression_ratio": "",
            "effective_params": informe.parameters,
            "effective_params_ratio": "",
            "precision": "fp32",
            **(
                cost_dict_to_registry(informe.cost, origen="afinado QAT")
                if informe.cost
                else {columna: "" for columna in REGISTRY_COLUMNS}
            ),
            "weights_path": informe.checkpoint_after,
            "status": "ok",
            "notes": (
                f"QAT REAL: {informe.preparation.get('num_weight_observers', '?')} "
                f"observadores de peso y "
                f"{informe.preparation.get('num_activation_observers', '?')} de "
                f"activación, ejercitados durante el afinado (comprobado, no "
                f"supuesto). {config.quantization.qat_epochs} épocas, SGD a "
                f"{config.quantization.qat_learning_rate:g} sin calentamiento. "
                f"Pesos FP32 sin simulación, listos para el mismo camino de "
                f"materialización INT8 que el baseline (D-018)."
            ),
        },
        on_duplicate=args.on_duplicate,
    )

    if evaluador is not None and informe.checkpoint_after:
        informe_eval = evaluador.reports.get(informe.checkpoint_after)
        if informe_eval is not None:
            registro.append_quality_metrics(
                construir_fila_calidad(
                    experiment_id=experiment_id,
                    informe_eval=informe_eval,
                    clases=clases,
                    device_id=args.device_id,
                    environment_id=environment_id,
                    ruta_detalle=detalle,
                ),
                on_duplicate=args.on_duplicate,
            )

    print()
    print(f"Inscrito como {experiment_id}.")
    print(
        "SIGUIENTE PASO: materializar el INT8 de esta rama con el mismo conjunto de calibración\n"
        "que la rama PTQ y contrastarlo con compare_with_qat()."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
