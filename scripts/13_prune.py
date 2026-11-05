#!/usr/bin/env python
"""Poda estructurada de unos pesos ya entrenados.

La importancia de cada canal depende de los pesos, así que se poda el modelo afinado en SH17. El
``ratio`` es el que se pide y ``effective_params_ratio`` el que se obtiene, más alto porque quitar
un canal de salida arrastra las entradas que dependen de él; el apartado 5.3 usa el medido.

También se mide el modelo recién podado, sin reajustar, y la fila lo indica en ``notes``. Con
``--random-control`` se repite la campaña con criterio aleatorio para distinguir el efecto del
criterio del de la propia poda. Cada rama deriva su semilla de la base y del criterio, y el esquema
de siembra entra en el hash.

Códigos de salida: 0 si la campaña termina y queda inscrita, 1 si hay un error de configuración o de
uso, 2 si la poda falla de forma controlada (validación de formas, mínimo de canales); queda
registrada con su motivo.
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
from tfm_edge.compression.pruning import finetune as finetune_module  # noqa: E402
from tfm_edge.compression.pruning.schedule import run_pruning_schedule  # noqa: E402
from tfm_edge.data.download import compute_sha256  # noqa: E402
from tfm_edge.data.smoke_subset import SMOKE_IMAGES, prepare_smoke_subset  # noqa: E402
from tfm_edge.evaluation import runner  # noqa: E402
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402
from tfm_edge.training.cost import REGISTRY_COLUMNS, cost_dict_to_registry  # noqa: E402

logger = get_logger(__name__)

#: Esquema de siembra por rama; entra en el hash para que campañas con distinto esquema no compartan
#: experiment_id. v2 pasa además la semilla de rama al reajuste. Si el esquema cambia, cambia esta
#: cadena.
SEEDING_SCHEME = "per_branch_v2"


def construir_identidad(
    config: ProjectConfig,
    criterio: str,
    variante: str,
    con_reajuste: bool,
    huella_checkpoint: str,
    parent_experiment_id: str,
) -> tuple[dict[str, Any], str, str]:
    """Construye el payload que identifica una rama de poda y devuelve ``(payload, config_hash,
    experiment_id)``.

    Incluye la huella del contenido del checkpoint, el ``parent_experiment_id`` declarado, el
    criterio y las secciones de ``EXPERIMENT_IDENTITY_SECTIONS`` de la configuración.
    """
    payload = {
        **experiment_identity_payload(config),
        "pruning": {**config.pruning.model_dump(mode="json"), "criterion": criterio},
        "finetuned": con_reajuste,
        # El esquema de siembra entra en el hash.
        "seeding_scheme": SEEDING_SCHEME,
        # Punto de partida.
        "starting_checkpoint_sha256": huella_checkpoint,
        "parent_experiment_id": parent_experiment_id,
    }
    config_hash = compute_config_hash(payload)
    experiment_id = build_experiment_id(
        model_size=config.model.size,
        technique="pruning",
        variant=variante,
        config_hash=config_hash,
    )
    return payload, config_hash, experiment_id


class CheckpointEvaluator:
    """Evalúa checkpoints con pycocotools y guarda el informe completo en ``reports``.

    Comprueba sus entradas al construirse, porque se usa por primera vez después de podar (ver
    :func:`runner.assert_evaluation_inputs_available`).
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
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="13_prune.py",
        description=(
            "Poda un checkpoint entrenado según una configuración de "
            "configs/compression/pruning/, valida el resultado, mide la reducción real de "
            "parámetros y lo inscribe en el registro."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Poda al 50 % del baseline principal, sin reajuste\n"
            "  python scripts/13_prune.py \\\n"
            "      --checkpoint models/baselines/s_baseline_default/best.pt \\\n"
            "      --config configs/compression/pruning/l1_global_r50.yaml\n"
            "\n"
            "  # La misma, más el control aleatorio al mismo ratio\n"
            "  python scripts/13_prune.py --checkpoint ... --config ... \\\n"
            "      --random-control\n"
            "\n"
            "  # Solo la estructura, sin evaluar (rápido)\n"
            "  python scripts/13_prune.py --checkpoint ... --config ... \\\n"
            "      --no-evaluate\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Pesos ENTRENADOS de partida. La importancia se calcula sobre ellos.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="RUTA",
        help="YAML de poda de configs/compression/pruning/.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="CLAVE=VALOR",
        help="Overrides con notación de punto, p. ej. pruning.ratio=0.6.",
    )
    parser.add_argument(
        "--parent-experiment-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Experimento del que desciende esta variante. Sin él, la fila queda "
            "sin enlazar con el baseline del que salió."
        ),
    )
    parser.add_argument(
        "--random-control",
        action="store_true",
        help=(
            "Repite la campaña con criterio aleatorio, al mismo ratio y con las mismas "
            "protecciones."
        ),
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help=(
            "Reajusta tras podar, con las épocas y la tasa de aprendizaje de la "
            "configuración. Exige GPU salvo --allow-cpu-finetune."
        ),
    )
    parser.add_argument(
        "--allow-cpu-finetune",
        action="store_true",
        help="Permite reajustar en CPU. Es inviable en la práctica; existe para pruebas.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Procesos de carga de datos del reajuste. Por defecto, training.workers de la "
            "configuración. Equivale a --override training.workers=N y entra en el "
            "experiment_id."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Rehace el reajuste aunque haya una ejecución anterior. Sin esta opción, relanzar "
            "un reajuste interrumpido falla, porque el entrenador no puede reanudar."
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
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de inferencia."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Lote del reajuste y de la evaluación. Por defecto, training.batch_size de la "
            "configuración. Equivale a --override training.batch_size=N y entra en el "
            "experiment_id."
        ),
    )
    parser.add_argument("--device", type=str, default="", metavar="DEV", help="Dispositivo.")
    parser.add_argument(
        "--device-id",
        type=str,
        default="workstation_cpu",
        metavar="ID",
        help=(
            "Identificador de dispositivo del registro (por defecto: "
            "%(default)s). Tiene que coincidir con el de deployment_metrics."
        ),
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="No mide mAP. Solo estructura: parámetros, canales y validación de formas.",
    )
    parser.add_argument("--no-per-class", action="store_true", help="No calcula el AP por clase.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde dejar los checkpoints podados (por defecto: models/pruned).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde volcar el JSON detallado (por defecto: results/raw/pruning).",
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
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Semilla del experimento. Si se omite, se usa la de la configuración; si se da, la "
            "sustituye y entra en el hash."
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            f"Campaña reducida: evalúa {SMOKE_IMAGES} imágenes en lugar de la "
            "partición entera. Implica --dry-run."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Poda y mide pero no escribe en el registro."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    return parser


def _degradaria_la_evidencia(detalle: Path, informe: Any) -> bool:
    """Indica si sobrescribir ese volcado cambiaría métricas medidas por valores vacíos."""
    if not detalle.is_file():
        return False
    if informe.metrics_before or informe.metrics_after:
        return False  # esta ejecución ha medido y puede sustituirlo
    try:
        anterior = json.loads(detalle.read_text(encoding="utf-8")).get("finetune", {})
    except (OSError, json.JSONDecodeError):
        # Un volcado ilegible no es evidencia que proteger.
        return False
    return bool(anterior.get("metrics_before") or anterior.get("metrics_after"))


def volcar_detalle(
    output_dir: Path,
    *,
    experiment_id: str,
    checkpoint: Path | str,
    config_path: Path | str,
    criterio: str,
    es_control: bool,
    smoke_test: bool,
    config: ProjectConfig,
    seed_base: int,
    seed_branch: int,
    identity_payload: Mapping[str, Any],
    schedule: Any,
    resultado_poda: Any,
    informe_reajuste: Any,
    dry_run: bool,
) -> Path | None:
    """Escribe el volcado crudo de una rama de poda, salvo con ``dry_run`` o si sustituiría métricas
    por vacío.
    """
    destino = output_dir / f"{experiment_id}.json"
    if dry_run:
        print(f"\n(--dry-run: no se vuelca el detalle en {destino})")
        return None
    if _degradaria_la_evidencia(destino, informe_reajuste):
        print(
            "\n"
            f"No se sobrescribe '{destino}': la campaña anterior tenía métricas medidas y esta "
            "no (--no-evaluate).\n"
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
                "criterion": criterio,
                "is_random_control": es_control,
                "smoke_test": smoke_test,
                # Semilla efectiva de la rama y esquema con el que se derivó.
                "seed_base": seed_base,
                "seed_branch": seed_branch,
                "seeding_scheme": SEEDING_SCHEME,
                "effective_config": config.model_dump(mode="json"),
                # El payload exacto que se hashea, con la huella del checkpoint de partida; un test
                # comprueba que su hash da el sufijo del experiment_id.
                "experiment_identity_payload": dict(identity_payload),
                "schedule": schedule.to_dict(),
                "pruning": resultado_poda.to_dict(),
                "finetune": informe_reajuste.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDetalle volcado en {destino}")
    return destino


def ejecutar_campana(
    checkpoint: Path,
    config: ProjectConfig,
    criterio: str,
    experiment_id: str,
    output_dir: Path,
    data_config: Path,
    evaluador: CheckpointEvaluator | None,
    baseline_metrics: dict[str, float],
    con_reajuste: bool,
    trainer_kwargs: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    """Poda un checkpoint y, si procede, lo reajusta.

    Devuelve el resultado de la poda, el informe de reajuste y el detector resultante. Lanza
    TFMEdgeError si la poda falla de forma controlada.
    """
    poda = config.pruning.model_copy(update={"criterion": criterio})
    if not con_reajuste:
        # Con cero épocas la métrica posterior queda ausente, que no es lo mismo que un reajuste sin
        # efecto.
        poda = poda.model_copy(update={"finetune_epochs": 0, "finetune_epochs_per_step": 0})

    detector = Detector.load(checkpoint, input_resolution=config.model.input_resolution)
    logger.info(
        "Punto de partida: %s parámetros, %d convoluciones",
        f"{detector.count_parameters():,}",
        len(detector.get_named_conv_layers()),
    )

    generador = torch.Generator().manual_seed(config.seed)
    entrada = torch.randn(
        1, 3, config.model.input_resolution, config.model.input_resolution, generator=generador
    )

    # Ultralytics resiembra con su 'seed' al construir el trainer; sin esta clave usaría 0 y las dos
    # ramas reajustarían igual. Se pone aquí y no en el diccionario común de main porque cada rama
    # tiene su semilla.
    trainer_kwargs = {**trainer_kwargs, "seed": random_seed}

    reajuste_parcial = None
    if con_reajuste and poda.iterative_steps > 1:
        reajuste_parcial = finetune_module.make_partial_finetune_callback(
            poda,
            data_config=data_config,
            output_dir=output_dir / experiment_id,
            evaluator=evaluador,
            trainer_kwargs=trainer_kwargs,
        )

    podado, campana = run_pruning_schedule(
        detector,
        poda,
        finetune=reajuste_parcial,
        example_input=entrada,
        random_seed=random_seed,
    )
    print()
    print(campana.summary())
    if campana.final is not None:
        print()
        print(campana.final.summary())

    reajustado, informe = finetune_module.finetune_pruned(
        podado,
        poda,
        data_config=data_config,
        output_dir=output_dir / experiment_id,
        evaluator=evaluador,
        baseline_metrics=baseline_metrics,
        trainer_kwargs=trainer_kwargs,
    )
    return {
        "schedule": campana,
        "finetune": informe,
        "detector": reajustado,
        "config": poda,
        "random_seed": random_seed,
    }


def construir_fila_calidad(
    experiment_id: str,
    informe_eval: dict[str, Any],
    clases: list[str],
    device_id: str,
    environment_id: str,
    ruta_detalle: Path,
    reajustado: bool,
) -> dict[str, Any]:
    """Compone la fila de quality_metrics de un modelo podado."""
    resultado = informe_eval["result"]
    aviso = (
        ""
        if reajustado
        else (
            " MODELO PODADO SIN REAJUSTAR: esta cifra mide lo que degrada la poda "
            "por sí sola (apartado 5.3.2.2) y NO es comparable con el baseline "
            "como resultado de la técnica."
        )
    )
    fila: dict[str, Any] = {
        "experiment_id": experiment_id,
        "runtime": "pytorch",
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
        "notes": f"evaluator=pycocotools.{aviso}".strip(),
    }
    for indice, columna in enumerate(build_ap_class_columns(clases)):
        datos = resultado.per_class.get(indice)
        fila[columna] = datos["ap_50_95"] if datos else ""
    return fila


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("13_prune", level=args.log_level)
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
        config = load_config(args.config, overrides=overrides)
    except TFMEdgeError as error:
        print(f"ERROR de configuración: {error}", file=sys.stderr)
        return 1

    # Una sola semilla efectiva, la misma que entra en el hash.
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})
    set_global_seed(config.seed)
    logger.info("Semilla efectiva del experimento: %d", config.seed)

    if args.finetune and not torch.cuda.is_available() and not args.allow_cpu_finetune:
        # No se pasa a CPU sin pedirlo.
        print(
            "ERROR: se ha pedido --finetune y no hay GPU disponible.\n"
            "  El reajuste se ejecuta en Colab (ver cloud/README.md). Para probar el camino "
            "aquí, usa --allow-cpu-finetune.",
            file=sys.stderr,
        )
        return 1

    data_yaml = args.data or (rutas.root / "data" / "resized_640" / "sh17_ultralytics.yaml")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = ensure_dir(args.output_dir or (rutas.root / "models" / "pruned"))
    crudo = ensure_dir(args.raw_dir or (rutas.results / "raw" / "pruning"))
    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")

    splits_dir = rutas.data_splits
    if args.smoke_test:
        data_yaml, splits_dir = prepare_smoke_subset(
            data_yaml, splits_dir, args.split, SMOKE_IMAGES, crudo / "_smoke_subset"
        )
        print(f"PRUEBA DE HUMO: {SMOKE_IMAGES} imágenes. Las cifras NO son resultados.")

    print("=" * 78)
    print("PASO 13: PODA ESTRUCTURADA SOBRE LOS PESOS AFINADOS")
    print("=" * 78)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Configuración: {args.config}")
    print(
        f"Poda        : criterio '{config.pruning.criterion}', reparto "
        f"'{config.pruning.distribution}', ratio {config.pruning.ratio:.0%}, "
        f"{config.pruning.iterative_steps} ciclo(s)"
    )
    print(f"Reajuste    : {'sí' if args.finetune else 'NO (el modelo no queda recuperado)'}")
    print()

    evaluador = None
    if not args.no_evaluate:
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

    # El baseline se toma del registro en lugar de reevaluarlo. En modo humo no se usa, porque su
    # cifra es de la partición entera y aquí se miden cuarenta imágenes.
    baseline_metrics: dict[str, float] = {}
    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)
    if args.smoke_test and args.parent_experiment_id:
        logger.warning(
            "--smoke-test: no se usa el baseline del registro, porque es de la partición "
            "completa y aquí se miden %d imágenes.",
            SMOKE_IMAGES,
        )
    elif args.parent_experiment_id:
        # Se filtra por el evaluador con el que mide este script; si quedan filas que no coinciden,
        # falla.
        try:
            referencia = read_reference_quality(
                registro,
                args.parent_experiment_id,
                args.split,
                evaluator=CANONICAL_EVALUATOR,
                runtime="pytorch",
                preferred_device_id=args.device_id,
            )
        except RegistryAmbiguousRowError as error:
            # Se comprueba antes de podar.
            print(f"ERROR: no se puede fijar la referencia: {error}", file=sys.stderr)
            return 1
        baseline_metrics = {
            clave: referencia[clave] for clave in ("map_50", "map_50_95") if clave in referencia
        }
        if baseline_metrics:
            print(
                f"Baseline de referencia ({args.parent_experiment_id}): "
                f"mAP@50-95 = {baseline_metrics['map_50_95']:.4f} "
                f"({referencia['evaluator']}, {referencia['device_id']})"
            )
        if not baseline_metrics:
            logger.warning(
                "No hay fila de calidad de '%s' en el registro: la fracción "
                "recuperada quedará ausente, no a cero.",
                args.parent_experiment_id,
            )

    trainer_kwargs: dict[str, Any] = {
        "batch_size": config.training.batch_size,
        "device": args.device,
        # Planificador, lrf, momentum, weight_decay y workers salen de la configuración que entra en
        # el hash.
        **config.training.trainer_recipe_kwargs(),
        # El calentamiento también sale de la configuración.
        "warmup_epochs": config.training.warmup_epochs,
        # El modo determinista está en la raíz de la configuración y no sale de
        # trainer_recipe_kwargs().
        "deterministic": config.deterministic,
        # Este entrenador no puede reanudar: con True, un reajuste interrumpido falla al relanzarlo
        # en vez de empezar de cero.
        "resume_if_possible": not args.no_resume,
    }

    # El contenido de lo que se poda entra en la identidad de cada rama.
    huella_checkpoint = compute_sha256(args.checkpoint)
    print(f"Huella      : sha256 {huella_checkpoint}")

    criterios = [config.pruning.criterion]
    if args.random_control:
        criterios.append("random")

    salidas: list[int] = []
    for criterio in criterios:
        es_control = criterio != config.pruning.criterion
        variante = config.technique_variant + ("_randomctl" if es_control else "")

        # Cada rama parte de su propia semilla derivada, tanto para el sorteo del control como para
        # el reajuste.
        semilla_rama = derive_seed(config.seed, "pruning", criterio)
        set_global_seed(semilla_rama)
        logger.info(
            "Rama '%s': semilla derivada %d (base %d). La rama no depende de lo "
            "ejecutado antes en este proceso.",
            criterio,
            semilla_rama,
            config.seed,
        )

        payload_identidad, config_hash, experiment_id = construir_identidad(
            config, criterio, variante, args.finetune, huella_checkpoint, args.parent_experiment_id
        )
        print()
        print("-" * 78)
        etiqueta = "CONTROL ALEATORIO" if es_control else "CRITERIO CONFIGURADO"
        print(f"{etiqueta}: criterio '{criterio}'  =>  {experiment_id}")
        print("-" * 78)

        try:
            campana = ejecutar_campana(
                checkpoint=args.checkpoint,
                config=config,
                criterio=criterio,
                experiment_id=experiment_id,
                output_dir=salida,
                data_config=data_yaml,
                evaluador=evaluador,
                baseline_metrics=baseline_metrics,
                con_reajuste=args.finetune,
                trainer_kwargs=trainer_kwargs,
                random_seed=semilla_rama,
            )
        except TFMEdgeError as error:
            # Un fallo de validación de formas se registra con su motivo y la campaña sigue con el
            # siguiente criterio.
            print(f"\nLA PODA HA FALLADO: {error}", file=sys.stderr)
            if not dry_run:
                registro.append_experiment(
                    {
                        "experiment_id": experiment_id,
                        "parent_experiment_id": args.parent_experiment_id,
                        "timestamp": utc_timestamp(),
                        "model_family": config.model.family,
                        "model_size": config.model.size,
                        "technique": "pruning",
                        "technique_variant": variante,
                        "config_hash": config_hash,
                        "config_path": str(args.config),
                        "compression_ratio": config.pruning.ratio,
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
                    on_duplicate=args.on_duplicate,
                )
            salidas.append(2)
            continue

        resultado_poda = campana["schedule"].final
        informe_reajuste = campana["finetune"]
        assert resultado_poda is not None  # run_pruning_schedule siempre deja uno

        # El volcado no se escribe con --dry-run, y uno sin métricas no sustituye a otro que las
        # tiene. Una campaña repetida entera sí lo sustituye.
        detalle = crudo / f"{experiment_id}.json"
        volcar_detalle(
            crudo,
            experiment_id=experiment_id,
            checkpoint=args.checkpoint,
            config_path=args.config,
            criterio=criterio,
            es_control=es_control,
            smoke_test=args.smoke_test,
            config=campana["config"],
            seed_base=config.seed,
            seed_branch=campana["random_seed"],
            identity_payload=payload_identidad,
            schedule=campana["schedule"],
            resultado_poda=resultado_poda,
            informe_reajuste=informe_reajuste,
            dry_run=dry_run,
        )
        print()
        print(informe_reajuste.summary())

        if dry_run:
            print("\n(--dry-run: no se escribe en el registro)")
            salidas.append(0)
            continue

        entorno = capture_environment(device_id=args.device_id, repo_dir=rutas.root)
        environment_id = persist_environment(entorno, registry_dir=args.registry_dir)

        checkpoint_final = informe_reajuste.checkpoint_after or informe_reajuste.checkpoint_before
        registro.append_experiment(
            {
                "experiment_id": experiment_id,
                "parent_experiment_id": args.parent_experiment_id,
                "timestamp": utc_timestamp(),
                "model_family": config.model.family,
                "model_size": config.model.size,
                "technique": "pruning",
                "technique_variant": variante,
                "config_hash": config_hash,
                "config_path": str(detalle),
                # Ratio nominal pedido; el medido va en effective_params_ratio.
                "compression_ratio": config.pruning.ratio,
                "effective_params": resultado_poda.parameters_after,
                "effective_params_ratio": round(
                    resultado_poda.parameters_after / resultado_poda.parameters_before, 6
                ),
                "precision": "fp32",
                # La conversión de coste a columnas del registro está en un solo sitio.
                **(
                    cost_dict_to_registry(informe_reajuste.cost, origen="reajuste")
                    if args.finetune
                    else {columna: "" for columna in REGISTRY_COLUMNS}
                ),
                "weights_path": checkpoint_final,
                "status": "ok",
                "notes": (
                    f"Poda REAL, criterio '{criterio}', reparto "
                    f"'{resultado_poda.distribution}'. Ratio pedido "
                    f"{resultado_poda.ratio_requested:.0%}, reducción MEDIDA de "
                    f"parámetros {resultado_poda.parameter_reduction:.1%}. "
                    + (
                        "Reajustado."
                        if args.finetune
                        else "SIN REAJUSTAR: la métrica no es comparable con el baseline."
                    )
                ),
            },
            on_duplicate=args.on_duplicate,
        )

        if evaluador is not None:
            medido = informe_reajuste.checkpoint_after or informe_reajuste.checkpoint_before
            informe_eval = evaluador.reports.get(medido)
            if informe_eval is not None:
                registro.append_quality_metrics(
                    construir_fila_calidad(
                        experiment_id,
                        informe_eval,
                        clases,
                        args.device_id,
                        environment_id,
                        detalle,
                        reajustado=args.finetune,
                    ),
                    on_duplicate=args.on_duplicate,
                )
        print(f"\nInscrito en el registro: {experiment_id}")
        salidas.append(0)

    if not dry_run:
        print()
        print(registro.validate().summary())
    print("=" * 78)
    return max(salidas) if salidas else 1


if __name__ == "__main__":
    sys.exit(main())
