#!/usr/bin/env python
"""Destilación de conocimiento de un maestro grande a un alumno pequeño (9.4).

Primero mide si maestro y alumno caben juntos en memoria ejecutando el bucle sobre unas pocas
imágenes, al lote y la resolución de la campaña. Esta comprobación no se puede saltar
(``--preflight-only`` solo la ejecuta) y, si no cabe, el script falla sin bajar el lote. Después
destila con el maestro congelado, evalúa el checkpoint con pycocotools y lo inscribe en el registro.

El control es el mismo camino con ``alpha=0`` (``kd_pilot_10ep_control.yaml``, o
``kd_control_50ep.yaml`` en la extensión a 50 épocas), que equivale a un afinado normal. El baseline
``n`` no sirve de control porque es el punto de partida. Al terminar se aplica la regla de lectura
de :mod:`tfm_edge.compression.distillation.reading` contra el control inscrito (``read_extension``
con otro número de épocas).

``--reuse-run`` completa un entrenamiento ya hecho sin repetirlo: evalúa, vuelca e inscribe. Lo que
no está en disco va a ``null`` en el volcado (ver :mod:`tfm_edge.compression.distillation.reuse`) y
no se hace la comprobación de memoria (``"preflight": null``).

Códigos de salida: 0 si todo va bien o ``--preflight-only`` confirma la memoria, 1 si hay un error
de configuración, de entradas o de entorno, 2 si la destilación falla (se inscribe con
``status=fallido``), 3 si la comprobación previa indica que no cabe, 4 si la comprobación previa
falla por algo que no es memoria, 5 si ya hay un entrenamiento en el directorio de salida, 6 si se
pide ``--reuse-run`` y no hay entrenamiento que reutilizar.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import (  # noqa: E402
    ProjectConfig,
    compute_config_hash,
    experiment_identity_payload,
    load_class_names,
    load_config,
)
from tfm_edge.common.environment import capture_environment, persist_environment  # noqa: E402
from tfm_edge.common.exceptions import (  # noqa: E402
    AmbiguousDistillationControlError,
    RegistryAmbiguousRowError,
    TFMEdgeError,
)
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import build_experiment_id, ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import (  # noqa: E402
    ExperimentRegistry,
    build_ap_class_columns,
    utc_timestamp,
)
from tfm_edge.common.registry_lookup import read_reference_quality  # noqa: E402
from tfm_edge.common.seed import derive_seed, set_global_seed  # noqa: E402
from tfm_edge.compression.distillation.preflight import (  # noqa: E402
    FAILURE_MEMORY,
    PREFLIGHT_TRAIN_IMAGES,
    PreflightResult,
    run_preflight,
)
from tfm_edge.compression.distillation.reading import (  # noqa: E402
    PILOT_CONTROL_ID,
    PILOT_EPOCHS,
    STARTING_BASELINE_ID,
    read_extension,
    read_pilot,
)
from tfm_edge.compression.distillation.reuse import ReusedRun, load_finished_run  # noqa: E402
from tfm_edge.compression.distillation.trainer import (  # noqa: E402
    DistillationResult,
    DistillationTrainer,
)
from tfm_edge.data.download import compute_sha256  # noqa: E402
from tfm_edge.data.smoke_subset import SMOKE_IMAGES, prepare_smoke_subset  # noqa: E402
from tfm_edge.evaluation import runner  # noqa: E402
from tfm_edge.evaluation.quality_notes import (  # noqa: E402
    CANONICAL_EVALUATOR,
    DISTILLATION_CONTROL_MARKER,
    build_distillation_note,
    read_seed_from_note,
)
from tfm_edge.models.detector import Detector  # noqa: E402
from tfm_edge.training.cost import REGISTRY_COLUMNS, cost_dict_to_registry  # noqa: E402
from tfm_edge.training.trainer import CLOSE_MOSAIC_POR_DEFECTO, mosaic_regime  # noqa: E402

logger = get_logger(__name__)

#: Esquema de siembra por rama, igual que en poda y QAT. Entra en el hash; v2 pasa también la
#: semilla de rama a Ultralytics.
SEEDING_SCHEME = "per_branch_v2"

#: El alumno parte de pesos convergidos, así que el calentamiento se desactiva, como en QAT.
WARMUP_EPOCHS_DISTILL = 0.0


def construir_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos."""
    parser = argparse.ArgumentParser(
        prog="20_distill.py",
        description=(
            "Destila un maestro grande en un alumno pequeño y lo inscribe en el registro. Antes "
            "de entrenar mide si maestro y alumno caben juntos en la GPU al lote de la campaña."
        ),
        epilog=(
            "La campaña piloto son dos ejecuciones, en este orden:\n"
            "\n"
            "  # 1. Control sin destilar (alpha=0).\n"
            "  python scripts/20_distill.py \\\n"
            "      --config configs/compression/distillation/kd_pilot_10ep_control.yaml\n"
            "\n"
            "  # 2. Piloto destilado; al terminar se lee contra el control.\n"
            "  python scripts/20_distill.py \\\n"
            "      --config configs/compression/distillation/kd_pilot_10ep.yaml\n"
            "\n"
            "  # Solo la comprobación de memoria\n"
            "  python scripts/20_distill.py --config ... --preflight-only\n"
            "\n"
            "  # Prueba del camino completo, sin escribir en el registro\n"
            "  python scripts/20_distill.py --config ... --smoke-test\n"
            "\n"
            "  # Completar una campaña que falló después de entrenar\n"
            "  python scripts/20_distill.py --config ... --reuse-run\n"
            "\n"
            "La extensión a 50 épocas es otra campaña: primero el control\n"
            "kd_control_50ep.yaml y después cada variante con --control-experiment-id\n"
            "apuntando a él. Todas declaran el cierre del mosaico y la paciencia en su\n"
            "YAML, así que necesitan --allow-mosaic-free.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/compression/distillation/kd_pilot_10ep.yaml"),
        metavar="RUTA",
        help="Configuración de destilación (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="CLAVE=VALOR",
        help=(
            "Sobrescribe claves de la configuración con notación de punto. "
            "'distillation.epochs' se define en el fichero porque piloto y control tienen que "
            "usar las mismas."
        ),
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Pesos de partida del alumno. Por defecto, el baseline del tamaño que "
            "declara la configuración. Se ignora si "
            "'distillation.student_from_pretrained' es falso."
        ),
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Pesos del maestro. Por defecto, los de 'distillation.teacher_weights'.",
    )
    parser.add_argument(
        "--parent-experiment-id",
        type=str,
        default=STARTING_BASELINE_ID,
        metavar="ID",
        help=(
            "Experimento del que parte el alumno (por defecto: %(default)s). NO es "
            "el control: es el punto de partida de los dos."
        ),
    )
    parser.add_argument(
        "--control-experiment-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Control sin destilar con el que contrastar al terminar. Si se omite, "
            "se busca en el registro la fila de destilación con alpha=0 del mismo "
            "tamaño de alumno."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde dejar los checkpoints (por defecto: models/distillation).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde volcar el detalle (por defecto: results/raw/distillation).",
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
        "--resolution", type=int, default=640, metavar="PX", help="Resolución del estudio."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Lote. Por defecto, training.batch_size de la configuración. Equivale a --override "
            "training.batch_size=N y entra en el experiment_id."
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
            "que es el del piloto de 10 épocas y la campaña de semillas. Sin esta opción el "
            "Trainer lo rechaza."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Épocas sin mejora antes de parar; 0 la desactiva. Por defecto, training.patience "
            "de la configuración. Equivale a --override training.patience=N y entra en el "
            "experiment_id."
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Comprueba la memoria y para. No entrena ni escribe en el registro.",
    )
    parser.add_argument(
        "--preflight-images",
        type=int,
        default=PREFLIGHT_TRAIN_IMAGES,
        metavar="N",
        help=(
            "Imágenes de entrenamiento de la comprobación previa (por defecto: "
            "%(default)s). Se eligen las MÁS POBLADAS de la partición."
        ),
    )
    parser.add_argument("--no-evaluate", action="store_true", help="No mide nada.")
    parser.add_argument("--no-per-class", action="store_true", help="No calcula el AP por clase.")
    parser.add_argument(
        "--allow-cpu-train",
        action="store_true",
        help=(
            "Permite destilar sin GPU. Solo para probar el camino: con el maestro "
            "`l` en el bucle, una época en CPU no es lenta, es inviable."
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
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help=(
            "Entrena aunque el directorio de salida ya tenga un entrenamiento, sustituyéndolo. "
            "Sin esta opción el script se detiene, porque no reanuda. También se aplica a "
            "--smoke-test, que comparte experiment_id con la campaña."
        ),
    )
    parser.add_argument(
        "--reuse-run",
        action="store_true",
        help=(
            "No entrena: lee el entrenamiento del directorio de salida y completa la "
            "evaluación, el volcado y la inscripción. No hace la comprobación de memoria. Lo "
            "que no esté en disco va a 'null' en el volcado."
        ),
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


def volcar_detalle(
    output_dir: Path,
    *,
    experiment_id: str,
    es_control: bool,
    alumno_pt: Path | str,
    maestro_pt: Path | str,
    config_path: Path | str,
    smoke_test: bool,
    config: ProjectConfig,
    seed_branch: int,
    identity_payload: Mapping[str, Any],
    lote: int,
    cierre_mosaico: int,
    regimen: str,
    previa: Any,
    metricas: dict[str, float],
    resultado: Any,
    reutilizado: bool,
    dry_run: bool,
) -> Path | None:
    """Escribe el volcado crudo de una rama de destilación, salvo con ``dry_run`` o si sustituiría
    métricas por vacío.
    """
    destino = output_dir / f"{experiment_id}.json"
    if dry_run:
        print(f"\n(--dry-run: no se vuelca el detalle en {destino})")
        return None
    if _degradaria_la_evidencia(destino, metricas):
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
                "is_control": es_control,
                "student_checkpoint": str(alumno_pt),
                "teacher_checkpoint": str(maestro_pt),
                "config_path": str(config_path),
                "smoke_test": smoke_test,
                "seed_base": config.seed,
                "seed_branch": seed_branch,
                "seeding_scheme": SEEDING_SCHEME,
                "warmup_epochs": WARMUP_EPOCHS_DISTILL,
                "effective_config": config.distillation.model_dump(mode="json"),
                # El payload exacto que se hashea, con las huellas del alumno y del maestro; un test
                # comprueba que su hash da el sufijo del experiment_id.
                "experiment_identity_payload": dict(identity_payload),
                # Parámetros de la receta que no están en la sección de destilación, con el valor
                # usado.
                "training_recipe": {
                    "batch_size": lote,
                    "patience": config.training.patience,
                    "close_mosaic_epochs": cierre_mosaico,
                    "close_mosaic_declared": (config.distillation.close_mosaic_epochs is not None),
                    "mosaic_regime": regimen,
                    **config.training.trainer_recipe_kwargs(),
                },
                "preflight": previa.to_dict() if previa is not None else None,
                "metrics": metricas,
                **campos_del_entrenamiento(resultado, reutilizado),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nDetalle volcado en {destino}")
    return destino


def _degradaria_la_evidencia(detalle: Path, metricas: dict[str, float]) -> bool:
    """Indica si sobrescribir ese volcado cambiaría métricas medidas por valores vacíos."""
    if not detalle.is_file():
        return False
    if metricas:
        return False  # esta ejecución ha medido y puede sustituirlo
    try:
        anterior = json.loads(detalle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False  # un volcado ilegible no se protege
    return bool(anterior.get("metrics"))


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
    """Inscribe una destilación fallida sin lanzar excepciones ni sustituir una fila con
    resultado."""
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
        politica = "overwrite"
    try:
        registro.append_experiment(fila, on_duplicate=politica)
    except TFMEdgeError as error:
        print(
            f"\nEl fallo NO se ha podido inscribir ({error}). El motivo real del "
            f"fallo es el de arriba, no este.",
            file=sys.stderr,
        )


def leer_metrica_del_registro(
    registro: ExperimentRegistry,
    experiment_id: str,
    split: str,
    *,
    evaluator: str = CANONICAL_EVALUATOR,
    preferred_device_id: str | None = None,
) -> dict[str, Any]:
    """Lee del registro la métrica de calidad de un experimento con
    :func:`~tfm_edge.common.registry_lookup.read_reference_quality`.

    Filtra por evaluador y usa ``preferred_device_id`` solo para desempatar. Devuelve ``map_50_95``,
    ``device_id``, ``evaluator`` y ``map_50`` si está, o un diccionario vacío. Lanza
    RegistryAmbiguousRowError si quedan filas con métricas distintas.
    """
    return read_reference_quality(
        registro,
        experiment_id,
        split,
        evaluator=evaluator,
        runtime="pytorch",
        preferred_device_id=preferred_device_id,
    )


def buscar_control(registro: ExperimentRegistry, model_size: str) -> str:
    """Busca en el registro el control sin destilar del mismo tamaño de alumno.

    Se identifica por la marca de ``notes``, que deriva de la ``alpha`` ejecutada, no por el nombre
    de la variante. Si hay varios controles distintos (por ejemplo, uno por semilla) lanza
    AmbiguousDistillationControlError; varias filas del mismo ``experiment_id`` no cuentan como
    empate. Devuelve una cadena vacía si no hay control.
    """
    candidatas = [
        fila
        for fila in registro.read_rows("experiments")
        if fila.get("technique") == "distillation"
        and fila.get("model_size") == model_size
        and fila.get("status") == "ok"
        and DISTILLATION_CONTROL_MARKER in fila.get("notes", "")
    ]
    if not candidatas:
        return ""

    identificadores = {fila["experiment_id"] for fila in candidatas}
    if len(identificadores) == 1:
        return candidatas[0]["experiment_id"]

    def _describir(fila: dict[str, str]) -> str:
        semilla = read_seed_from_note(fila.get("notes", ""))
        etiqueta = f"semilla base {semilla}" if semilla is not None else "sin marca de semilla"
        return f"  - {fila['experiment_id']} ({etiqueta}, {fila.get('timestamp', 'sin fecha')})"

    detalle = "\n".join(sorted(_describir(fila) for fila in candidatas))
    raise AmbiguousDistillationControlError(
        f"technique=distillation, model_size={model_size!r}, status=ok",
        detalle,
    )


def memoria_para_la_nota(previa: PreflightResult) -> str:
    """Compone la parte de ``notes`` sobre el pico de memoria, indicando si no se ha medido."""
    if not previa.memory_measured:
        return "Pico de memoria de GPU NO medido: la comprobación previa corrió sin GPU."
    return (
        f"Pico de memoria medido antes de lanzar: {previa.peak_reserved_mb:.0f} MiB "
        f"reservados de {previa.gpu_total_mb:.0f} MiB."
    )


def construir_fila_calidad(
    experiment_id: str,
    informe_eval: dict[str, Any],
    clases: list[str],
    device_id: str,
    environment_id: str,
    ruta_detalle: Path,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    """Compone la fila de quality_metrics del alumno destilado.

    ``alpha`` decide si la nota marca la fila como control, y la semilla va a la nota porque el
    registro no tiene columna para ella.
    """
    resultado = informe_eval["result"]
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
        "notes": build_distillation_note(alpha, seed=seed),
    }
    for indice, columna in enumerate(build_ap_class_columns(clases)):
        datos = resultado.per_class.get(indice)
        fila[columna] = datos["ap_50_95"] if datos else ""
    return fila


def campos_del_entrenamiento(
    resultado: DistillationResult | None, reutilizado: ReusedRun | None
) -> dict[str, Any]:
    """Compone los campos del volcado que describen el entrenamiento.

    Al reutilizar un entrenamiento se usan otros nombres para lo que se mide de otra forma
    (``adapter_norms_first_epoch`` frente a ``adapter_initial_norms``) y se deja a ``None`` lo que
    no está en disco. Lanza ValueError si no se pasa exactamente uno de los dos argumentos.
    """
    if (resultado is None) == (reutilizado is None):
        raise ValueError("campos_del_entrenamiento espera exactamente uno de los dos.")

    if resultado is not None:
        return {
            "training_source": "entrenado en esta ejecución",
            "steps": resultado.steps,
            "adapter_params": resultado.adapter_params,
            "adapter_initial_norms": resultado.adapter_initial_norms,
            "adapter_final_norms": resultado.adapter_final_norms,
            "adapters_learned": resultado.adapters_learned,
            "adapter_grad_always_zero": resultado.adapter_grad_was_always_zero,
            "epoch_components": resultado.epoch_components,
            "cost": resultado.training.cost.to_dict(),
        }

    assert reutilizado is not None  # garantizado por la comprobación anterior
    return {
        "training_source": "reutilizado de disco (--reuse-run)",
        "reused_training": reutilizado.to_dict(),
        # Lo que no está en disco va a null, no a 0 ni a {}.
        "steps": None,
        "adapter_params": None,
        "adapter_initial_norms": None,
        "adapter_final_norms": None,
        "adapters_learned": None,
        # Lo que sí está, con el nombre de lo que es.
        "steps_in_closed_epochs": reutilizado.steps_in_closed_epochs,
        "adapter_norms_first_epoch": reutilizado.adapter_norms_first_epoch,
        "adapter_norms_last_epoch": reutilizado.adapter_norms_last_epoch,
        "adapter_norms_moved": reutilizado.adapter_norms_moved,
        "adapter_grad_always_zero": reutilizado.adapter_grad_always_zero,
        "epoch_components": reutilizado.epoch_components,
        "cost": reutilizado.cost,
    }


def entrenamiento_previo(directorio: Path) -> tuple[Path, int] | None:
    """Comprueba si el directorio de salida ya tiene un entrenamiento.

    ``DistillationTrainer`` usa ``resume_if_possible=False`` y ``exist_ok=True``, así que relanzar
    sobre un directorio con pesos entrenaría desde cero encima. Devuelve ``(checkpoint, épocas)`` o
    ``None``; las épocas son 0 si no hay ``results.csv``.
    """
    mejor = directorio / "weights" / "best.pt"
    if not mejor.is_file():
        return None
    resultados = directorio / "results.csv"
    epocas = 0
    if resultados.is_file():
        filas = [f for f in resultados.read_text(encoding="utf-8").splitlines() if f.strip()]
        epocas = max(len(filas) - 1, 0)
    return mejor, epocas


def resolver_checkpoints(
    args: argparse.Namespace, config: ProjectConfig, rutas: Any
) -> tuple[Path, Path]:
    """Decide de qué ficheros salen alumno y maestro. Lanza TFMEdgeError si falta alguno o si la
    configuración no declara los pesos del maestro.
    """
    destilacion = config.distillation
    if args.teacher_checkpoint is not None:
        maestro = args.teacher_checkpoint
    elif destilacion.teacher_weights:
        maestro = rutas.root / destilacion.teacher_weights
    else:
        raise TFMEdgeError(
            "La configuración no declara los pesos del maestro.",
            "Ponlos en 'distillation.teacher_weights' o pásalos con " "--teacher-checkpoint.",
        )

    if args.student_checkpoint is not None:
        alumno = args.student_checkpoint
    else:
        alumno = rutas.models_baselines / f"{config.model.size}_baseline_default" / "best.pt"

    for etiqueta, ruta in (("alumno", alumno), ("maestro", maestro)):
        if not Path(ruta).is_file():
            raise TFMEdgeError(
                f"No existe el checkpoint del {etiqueta}: '{ruta}'.",
                "Entrena los baselines con scripts/04_train_baselines.py, o pásalo "
                "con --student-checkpoint / --teacher-checkpoint.",
            )
    return Path(alumno), Path(maestro)


def ejecutar_comprobacion_previa(
    args: argparse.Namespace,
    config: ProjectConfig,
    alumno_pt: Path,
    maestro_pt: Path,
    data_yaml: Path,
    splits_dir: Path,
    trabajo: Path,
    semilla: int,
) -> PreflightResult:
    """Mide si maestro y alumno caben juntos antes de lanzar la campaña.

    Los detectores se descartan al terminar porque la época de prueba modifica el alumno. La semilla
    es necesaria porque esa época entrena.
    """
    # El directorio de la prueba no cuelga de results/ y se borra para que el pico medido sea de una
    # ejecución limpia.
    if (trabajo / "run").exists():
        logger.info("Borrando la comprobación previa anterior en '%s'.", trabajo / "run")
        shutil.rmtree(trabajo / "run")

    alumno = Detector.load(alumno_pt, input_resolution=args.resolution)
    maestro = Detector.load(maestro_pt, input_resolution=args.resolution)
    try:
        return run_preflight(
            student=alumno,
            teacher=maestro,
            config=config.distillation,
            data_config=data_yaml,
            splits_dir=splits_dir,
            work_dir=trabajo,
            batch_size=config.training.batch_size,
            resolution=args.resolution,
            workers=config.training.workers,
            device=args.device,
            seed=semilla,
            train_images=args.preflight_images,
            allow_cpu=args.allow_cpu_train,
        )
    finally:
        del alumno, maestro
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def construir_identidad(
    config: ProjectConfig,
    huella_alumno: str,
    huella_maestro: str,
    parent_experiment_id: str,
) -> tuple[dict[str, Any], str, str]:
    """Construye el payload que identifica una destilación y devuelve ``(payload, config_hash,
    experiment_id)``.

    Incluye las secciones de ``EXPERIMENT_IDENTITY_SECTIONS`` (con lote, paciencia y, si se declara,
    el cierre del mosaico) y las huellas del contenido del alumno y del maestro. El cuaderno y los
    tests usan esta misma función.
    """
    payload = {
        **experiment_identity_payload(config),
        "seeding_scheme": SEEDING_SCHEME,
        "warmup_epochs": WARMUP_EPOCHS_DISTILL,
        # Los dos puntos de partida.
        "student_checkpoint_sha256": huella_alumno,
        "teacher_checkpoint_sha256": huella_maestro,
        "parent_experiment_id": parent_experiment_id,
    }
    config_hash = compute_config_hash(payload)
    experiment_id = build_experiment_id(
        model_size=config.model.size,
        technique="distillation",
        variant=config.technique_variant,
        config_hash=config_hash,
    )
    return payload, config_hash, experiment_id


def cierre_del_mosaico(config: ProjectConfig) -> int:
    """Devuelve el ``close_mosaic`` del entrenamiento: el declarado o, si no, el de Ultralytics. Lo
    usan la guarda del régimen, el Trainer y el volcado.
    """
    declarado = config.distillation.close_mosaic_epochs
    return CLOSE_MOSAIC_POR_DEFECTO if declarado is None else declarado


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("20_distill", level=args.log_level)
    rutas = get_project_paths()

    dry_run = args.dry_run or args.smoke_test or args.preflight_only
    if args.smoke_test:
        logger.warning("--smoke-test implica --dry-run: no se escribe en el registro.")

    if args.reuse_run and args.overwrite_run:
        print(
            "ERROR: --reuse-run y --overwrite-run son incompatibles.",
            file=sys.stderr,
        )
        return 1

    if args.reuse_run and args.preflight_only:
        print(
            "ERROR: --reuse-run no hace la comprobación previa y --preflight-only solo hace eso.",
            file=sys.stderr,
        )
        return 1

    if not args.config.is_file():
        print(f"ERROR: no existe la configuración '{args.config}'.", file=sys.stderr)
        return 1

    try:
        # --workers, --batch-size y --patience entran por la configuración para formar parte del
        # config_hash.
        overrides = list(args.override)
        if args.workers is not None:
            overrides.append(f"training.workers={args.workers}")
        if args.batch_size is not None:
            overrides.append(f"training.batch_size={args.batch_size}")
        if args.patience is not None:
            overrides.append(f"training.patience={args.patience}")
        config: ProjectConfig = load_config(args.config, overrides=overrides)
    except TFMEdgeError as error:
        print(f"ERROR de configuración: {error}", file=sys.stderr)
        return 1

    if config.technique != "distillation":
        print(
            f"ERROR: '{args.config}' declara technique='{config.technique}'. Este "
            f"script solo destila.",
            file=sys.stderr,
        )
        return 1

    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    # La semilla de rama sale de la base, la técnica y el tipo de destilación, pero no de alpha,
    # para que piloto y control vean los datos en el mismo orden.
    semilla_rama = derive_seed(config.seed, "distillation", config.distillation.kd_type)
    set_global_seed(semilla_rama)
    logger.info(
        "Semilla base %d, semilla de la rama %d (esquema '%s').",
        config.seed,
        semilla_rama,
        SEEDING_SCHEME,
    )

    if not torch.cuda.is_available() and not args.allow_cpu_train:
        # No se pasa a CPU sin pedirlo.
        print(
            "ERROR: la destilación necesita GPU y no hay ninguna disponible.\n"
            "  Con el maestro `l` en cada paso, entrenar en CPU es inviable.\n"
            "  Ejecútala en Colab (ver cloud/README.md). Para probar el camino aquí, usa "
            "--allow-cpu-train.",
            file=sys.stderr,
        )
        return 1

    try:
        alumno_pt, maestro_pt = resolver_checkpoints(args, config, rutas)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    data_yaml = args.data or (rutas.root / "data" / "resized_640" / "sh17_ultralytics.yaml")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    salida = ensure_dir(args.output_dir or (rutas.root / "models" / "distillation"))
    crudo = ensure_dir(args.raw_dir or (rutas.results / "raw" / "distillation"))
    clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
    splits_dir = rutas.data_splits

    huella_alumno = compute_sha256(alumno_pt)
    huella_maestro = compute_sha256(maestro_pt)
    payload_identidad, config_hash, experiment_id = construir_identidad(
        config, huella_alumno, huella_maestro, args.parent_experiment_id
    )
    es_control = config.distillation.alpha == 0.0
    lote = config.training.batch_size
    cierre_mosaico = cierre_del_mosaico(config)

    print("=" * 78)
    print("PASO 20: DESTILACIÓN DE CONOCIMIENTO")
    print("=" * 78)
    print(f"Configuración: {args.config}")
    print(f"Experimento  : {experiment_id}")
    print(f"Alumno       : '{config.model.size}' desde {alumno_pt}")
    print(f"               sha256 {huella_alumno}")
    print(f"Maestro      : '{config.distillation.teacher_size}' desde {maestro_pt}")
    print(f"               sha256 {huella_maestro}")
    print(
        f"Destilación  : {config.distillation.kd_type}, T={config.distillation.temperature:g}, "
        f"alpha={config.distillation.alpha:g}"
    )
    print(
        f"Campaña      : {config.distillation.epochs} épocas, tasa "
        f"{config.distillation.learning_rate:g}, lote {lote}, "
        f"{args.resolution} px, FP32 sin AMP"
    )
    print(
        f"Receta      : close_mosaic={cierre_mosaico}"
        f"{'' if config.distillation.close_mosaic_epochs is not None else ' (no declarado: defecto)'}"
        f", paciencia {config.training.patience}"
        f"{' (desactivada)' if config.training.patience == 0 else ''}, workers "
        f"{config.training.workers}, planificador {config.training.scheduler}"
    )
    if es_control:
        print()
        print(
            "EJECUCIÓN DE CONTROL (alpha=0). El término de destilación se multiplica\n"
            "por cero, así que equivale a un afinado normal por el mismo camino. Su\n"
            "cifra no es un resultado de la técnica."
        )
    print()

    # 0. Comprobaciones previas al uso de GPU.
    regimen, descripcion_regimen = mosaic_regime(config.distillation.epochs, cierre_mosaico)
    print(f"Mosaico     : {descripcion_regimen}")
    print()
    entrena = not args.reuse_run and not args.preflight_only
    if regimen == "sin_mosaico" and not args.allow_mosaic_free and entrena:
        # Misma comprobación que el Trainer, adelantada para no gastar la comprobación previa.
        print(
            f"ERROR: {descripcion_regimen}.\n"
            "  Es la receta del piloto de 10 épocas y de la campaña de semillas, no la\n"
            "  de los baselines. Para usarla, pasa --allow-mosaic-free.",
            file=sys.stderr,
        )
        return 1

    if entrena and not args.overwrite_run:
        # Se comprueba antes de la comprobación previa de memoria.
        previo = entrenamiento_previo(salida / experiment_id / "distill")
        if previo is not None:
            mejor_previo, epocas_previas = previo
            cuantas = (
                f"{epocas_previas} épocas registradas en results.csv"
                if epocas_previas
                else "no hay results.csv, así que no se sabe cuántas épocas son"
            )
            print(
                f"\nNO SE ENTRENA: '{salida / experiment_id / 'distill'}' ya tiene un "
                f"entrenamiento\n"
                f"  ({mejor_previo.name}, {cuantas}).\n"
                "  Este script no reanuda; seguir entrenaría desde cero encima de esos pesos.\n"
                "  Opciones:\n"
                "    1. --reuse-run, si lo que falló fue posterior al entrenamiento. Evalúa,\n"
                "       vuelca e inscribe a partir de estos pesos sin reentrenar.\n"
                "    2. --output-dir a otro directorio.\n"
                "    3. --overwrite-run para descartar lo que hay (haz antes una copia).",
                file=sys.stderr,
            )
            return 5

    # 1. Comprobación previa de memoria.
    previa: PreflightResult | None = None
    volcado_previa: Path | None = None
    if args.reuse_run:
        # Con --reuse-run no se entrena, así que no hace falta medir la memoria.
        print(
            "No se hace la comprobación previa de memoria: con --reuse-run no se\n"
            "entrena.\n"
            "  El volcado tendrá 'preflight': null."
        )
        print()
    else:
        trabajo = salida / "_preflight" / experiment_id
        try:
            previa = ejecutar_comprobacion_previa(
                args,
                config,
                alumno_pt,
                maestro_pt,
                data_yaml,
                splits_dir,
                trabajo,
                semilla_rama,
            )
        except TFMEdgeError as error:
            print(f"ERROR en la comprobación previa: {error}", file=sys.stderr)
            return 1

        print(previa.summary(), flush=True)
        print()

        # El volcado lleva marca de tiempo para no sobrescribir una medida anterior.
        marca = utc_timestamp().replace(":", "").replace("+0000", "Z")
        volcado_previa = crudo / f"preflight_{experiment_id}_{marca}.json"
        volcado_previa.write_text(
            json.dumps(
                {"experiment_id": experiment_id, "preflight": previa.to_dict()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Comprobación previa volcada en {volcado_previa}")

        if not previa.completed and previa.failure_kind == FAILURE_MEMORY:
            print(
                "\nNO SE LANZA LA CAMPAÑA: maestro y alumno no caben juntos en este\n"
                f"dispositivo al lote {lote}.\n"
                f"  Motivo: {previa.reason}\n"
                "  El script no cambia el lote, porque cambia los pasos por época y la\n"
                "  cota de coste. Para probar otro, usa --batch-size; entra en el\n"
                "  experiment_id, así que será otro experimento.",
                file=sys.stderr,
            )
            return 3

        if not previa.completed:
            # Aquí no se sugiere bajar el lote, porque no hay ninguna medida que apunte a él.
            print(
                "\nNO SE LANZA LA CAMPAÑA: la comprobación previa ha fallado antes de\n"
                "medir la memoria.\n"
                f"  Motivo: {previa.reason}\n"
                f"  La traza completa está en el campo\n"
                f"  'failure_traceback' de {volcado_previa}; el motivo de arriba puede\n"
                "  venir envuelto por Ultralytics.\n"
                "  No cambies el lote, porque no hay ninguna medida que apunte a él.",
                file=sys.stderr,
            )
            return 4

        if not previa.memory_measured:
            # Sin GPU no se ha medido memoria, así que no se puede decir que cabe.
            print(
                "\nAVISO: esta comprobación no ha medido memoria de GPU. Ha recorrido el\n"
                "camino, pero no dice si maestro y alumno caben juntos en la T4.\n"
                "Vuelve a ejecutarla en Colab antes de lanzar.",
                file=sys.stderr,
            )

        if previa.steps_timed and previa.conditions_match_campaign:
            print(
                f"\nCota prevista: 20,45 min/época. Medido aquí: "
                f"{previa.measured_epoch_s / 60:.2f} min de cómputo por época "
                f"({previa.measured_epoch_s * config.distillation.epochs / 3600:.2f} h "
                f"la campaña, sin contar validación ni arranque)."
            )
        elif previa.steps_timed:
            # En otras condiciones (resolución, lote o CPU) no se compara con la cota.
            print(
                f"\nTiempo medido: {previa.measured_epoch_s / 60:.2f} min por época a "
                f"{previa.resolution} px, lote {previa.batch_size}, en "
                f"'{previa.gpu_name or previa.device}'.\n"
                f"  No se compara con la cota, que es para T4 a 640 px y lote 16."
            )

    if args.preflight_only:
        print("\n(--preflight-only: no se entrena ni se escribe en el registro)")
        return 0

    # 2. Preparación del registro y del evaluador.
    if args.smoke_test:
        # Las dos particiones se recortan desde el YAML original, no desde el ya reducido.
        original_yaml, original_splits = data_yaml, splits_dir
        prepare_smoke_subset(
            original_yaml, original_splits, "train", SMOKE_IMAGES, crudo / "_smoke_subset"
        )
        data_yaml, splits_dir = prepare_smoke_subset(
            original_yaml, original_splits, args.split, SMOKE_IMAGES, crudo / "_smoke_subset"
        )
        print(f"PRUEBA DE HUMO: {SMOKE_IMAGES} imágenes. Las cifras NO son resultados.")

    evaluador_listo = False
    if not args.no_evaluate:
        try:
            runner.assert_evaluation_inputs_available(coco, splits_dir, args.split, data_yaml)
            evaluador_listo = True
        except TFMEdgeError as error:
            print(f"ERROR: no se puede evaluar: {error}", file=sys.stderr)
            return 1

    registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)

    # Se comprueba aquí si el experimento ya está inscrito para no fallar al final de la campaña.
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
            "'%s' ya está inscrito con status='%s'; al terminar se aplicará la política '%s'.",
            experiment_id,
            estado_previo,
            args.on_duplicate,
        )

    # 3. Campaña. Los detectores se recargan porque los de la comprobación previa ya han entrenado
    # una época.
    directorio_run = salida / experiment_id / "distill"
    resultado: DistillationResult | None = None
    reutilizado: ReusedRun | None = None

    if args.reuse_run:
        try:
            reutilizado = load_finished_run(directorio_run)
        except TFMEdgeError as error:
            print(f"\nERROR: {error}", file=sys.stderr)
            return 6
        print()
        print(reutilizado.summary())
        checkpoint = reutilizado.best_checkpoint
        # El número de parámetros se lee del checkpoint reutilizado, que es el que se inscribe.
        alumno = Detector.load(checkpoint, input_resolution=args.resolution)
        print(f"\nCheckpoint del alumno destilado: {checkpoint}")
        if reutilizado.components_csv:
            print(f"Desglose de la pérdida por época: {reutilizado.components_csv}")
    else:
        # La comprobación de entrenamiento existente ya se hizo al principio.
        set_global_seed(semilla_rama)
        alumno = Detector.load(alumno_pt, input_resolution=args.resolution)
        maestro = Detector.load(maestro_pt, input_resolution=args.resolution)
        logger.info(
            "Punto de partida: alumno %s parámetros, maestro %s parámetros.",
            f"{alumno.count_parameters():,}",
            f"{maestro.count_parameters():,}",
        )

        destilador = DistillationTrainer(
            student=alumno,
            teacher=maestro,
            data_config=data_yaml,
            output_dir=salida / experiment_id,
            config=config.distillation,
            resolution=args.resolution,
            # Lote, paciencia y cierre del mosaico salen de la configuración que entra en el hash.
            batch_size=lote,
            device=args.device,
            patience=config.training.patience,
            seed=semilla_rama,
            # Se pasa como parámetro porque el Trainer ya envía warmup_epochs.
            warmup_epochs=WARMUP_EPOCHS_DISTILL,
            # Modo determinista declarado en la configuración.
            deterministic=config.deterministic,
            # Planificador, lrf, momentum, weight_decay y workers salen de la configuración que
            # entra en el hash.
            **config.training.trainer_recipe_kwargs(),
            close_mosaic_epochs=cierre_mosaico,
            allow_mosaic_free=args.allow_mosaic_free,
        )

        try:
            resultado = destilador.train(run_name="distill")
        except TFMEdgeError as error:
            print(f"\nLA DESTILACIÓN HA FALLADO: {error}", file=sys.stderr)
            if not dry_run:
                _inscribir_el_fallo(
                    registro,
                    {
                        "experiment_id": experiment_id,
                        "parent_experiment_id": args.parent_experiment_id,
                        "timestamp": utc_timestamp(),
                        "model_family": config.model.family,
                        "model_size": config.model.size,
                        "technique": "distillation",
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

        checkpoint = resultado.training.best_checkpoint
        print(f"\nCheckpoint del alumno destilado: {checkpoint}")
        if resultado.components_csv:
            print(f"Desglose de la pérdida por época: {resultado.components_csv}")

    # 4. Evaluación.
    informe_eval: dict[str, Any] | None = None
    metricas: dict[str, float] = {}
    if evaluador_listo:
        logger.info("Evaluando '%s' por el camino canónico (pycocotools)", checkpoint.name)
        informe_eval = runner.evaluate_checkpoint(
            checkpoint,
            data_config=data_yaml,
            coco_path=coco,
            splits_dir=splits_dir,
            split=args.split,
            input_resolution=args.resolution,
            batch_size=lote,
            device=args.device,
            with_per_class=not args.no_per_class,
        )
        primario = informe_eval["result"].primary
        metricas = {"map_50": primario.map_50, "map_50_95": primario.map_50_95}
        print(
            f"Evaluado: mAP@50 = {metricas['map_50']:.6f}, "
            f"mAP@50-95 = {metricas['map_50_95']:.6f}"
        )

    # 5. Volcado crudo.
    detalle = crudo / f"{experiment_id}.json"
    volcar_detalle(
        crudo,
        experiment_id=experiment_id,
        es_control=es_control,
        alumno_pt=alumno_pt,
        maestro_pt=maestro_pt,
        config_path=args.config,
        smoke_test=args.smoke_test,
        config=config,
        seed_branch=semilla_rama,
        identity_payload=payload_identidad,
        lote=lote,
        cierre_mosaico=cierre_mosaico,
        regimen=regimen,
        previa=previa,
        metricas=metricas,
        resultado=resultado,
        reutilizado=reutilizado,
        dry_run=dry_run,
    )

    if dry_run:
        print("\n(--dry-run: no se escribe en el registro)")
        return 0

    # 6. Inscripción.
    if reutilizado is not None:
        coste_registro = reutilizado.cost
        origen_del_coste = f"reutilizado de {reutilizado.run_dir}"
        # La fila indica que el entrenamiento no es de esta ejecución, aunque la evaluación sí lo
        # sea.
        nota_de_procedencia = (
            f"Entrenamiento REUTILIZADO de '{reutilizado.run_dir}', no repetido "
            f"({reutilizado.closed_epochs} épocas registradas). La EVALUACIÓN sí es "
            f"de esta ejecución. Sin comprobación previa de memoria: en esta "
            f"ejecución no se ha entrenado."
        )
    else:
        assert resultado is not None and previa is not None  # garantizado en la sección 3
        coste_registro = resultado.training.cost.to_dict() if resultado.training.cost else None
        origen_del_coste = "destilación"
        nota_de_procedencia = memoria_para_la_nota(previa)

    entorno = capture_environment(device_id=args.device_id, repo_dir=rutas.root)
    environment_id = persist_environment(entorno, registry_dir=args.registry_dir)

    registro.append_experiment(
        {
            "experiment_id": experiment_id,
            "parent_experiment_id": args.parent_experiment_id,
            "timestamp": utc_timestamp(),
            "model_family": config.model.family,
            "model_size": config.model.size,
            "technique": "distillation",
            "technique_variant": config.technique_variant,
            "config_hash": config_hash,
            "config_path": str(detalle),
            # La destilación no quita parámetros al alumno, así que el ratio se deja vacío.
            "compression_ratio": "",
            "effective_params": alumno.count_parameters(),
            "effective_params_ratio": "",
            "precision": "fp32",
            **(
                cost_dict_to_registry(coste_registro, origen=origen_del_coste)
                if coste_registro
                else {columna: "" for columna in REGISTRY_COLUMNS}
            ),
            "weights_path": str(checkpoint),
            "status": "ok",
            "notes": build_distillation_note(config.distillation.alpha, seed=config.seed)
            + (
                f" Maestro '{config.distillation.teacher_size}', T="
                f"{config.distillation.temperature:g}, {config.distillation.epochs} épocas, "
                f"lote {lote}, FP32 sin AMP. "
            )
            + nota_de_procedencia,
        },
        on_duplicate=args.on_duplicate,
    )

    if informe_eval is not None:
        registro.append_quality_metrics(
            construir_fila_calidad(
                experiment_id=experiment_id,
                informe_eval=informe_eval,
                clases=clases,
                device_id=args.device_id,
                environment_id=environment_id,
                ruta_detalle=detalle,
                alpha=config.distillation.alpha,
                seed=config.seed,
            ),
            on_duplicate=args.on_duplicate,
        )

    print()
    print(f"Inscrito como {experiment_id}.")

    # 7. Regla de lectura, si hay con qué comparar.
    if es_control:
        if config.distillation.epochs == PILOT_EPOCHS:
            print(
                "\nSIGUIENTE PASO: el piloto destilado, con la misma configuración y\n"
                "alpha=0,5:\n"
                "  python scripts/20_distill.py --config "
                "configs/compression/distillation/kd_pilot_10ep.yaml"
            )
        else:
            print(
                f"\nSIGUIENTE PASO: las variantes destiladas a "
                f"{config.distillation.epochs} épocas, cada una leída contra este "
                f"control:\n  --control-experiment-id {experiment_id}"
            )
        return 0

    if not metricas:
        return 0

    # Un fallo aquí se avisa sin propagarlo: el entrenamiento y las filas ya están; solo falta la
    # lectura, que se puede repetir con --control-experiment-id.
    try:
        control_id = args.control_experiment_id or buscar_control(registro, config.model.size)
    except AmbiguousDistillationControlError as error:
        print(f"\nAVISO: {error}", file=sys.stderr)
        return 0

    if not control_id:
        print(
            "\nAVISO: no hay control sin destilar inscrito, así que esta cifra no indica\n"
            "si destilar ayuda. Ejecuta antes el control:\n"
            "  python scripts/20_distill.py --config "
            "configs/compression/distillation/kd_pilot_10ep_control.yaml",
            file=sys.stderr,
        )
        return 0

    try:
        control = leer_metrica_del_registro(
            registro, control_id, args.split, preferred_device_id=args.device_id
        )
        partida = leer_metrica_del_registro(
            registro, args.parent_experiment_id, args.split, preferred_device_id=args.device_id
        )
    except RegistryAmbiguousRowError as error:
        print(f"\nAVISO: {error}", file=sys.stderr)
        print(
            "No se aplica la regla de lectura porque las filas no coinciden.",
            file=sys.stderr,
        )
        return 0

    if not control or not partida:
        print(
            f"\nAVISO: falta la fila de calidad del control ('{control_id}') o del punto\n"
            f"de partida ('{args.parent_experiment_id}') medida con "
            f"{CANONICAL_EVALUATOR}, así que no se aplica la regla de lectura.",
            file=sys.stderr,
        )
        return 0

    if config.distillation.epochs == PILOT_EPOCHS:
        lectura = read_pilot(
            distilled_map_50_95=metricas["map_50_95"],
            control_map_50_95=control["map_50_95"],
            starting_map_50_95=partida["map_50_95"],
        )
    else:
        # La regla de la extensión necesita además el control de 10 épocas; sin él no se aplica.
        try:
            control_piloto = leer_metrica_del_registro(
                registro, PILOT_CONTROL_ID, args.split, preferred_device_id=args.device_id
            )
        except RegistryAmbiguousRowError as error:
            print(f"\nAVISO: {error}", file=sys.stderr)
            return 0
        if not control_piloto:
            print(
                f"\nAVISO: falta la fila de calidad del control de 10 épocas "
                f"('{PILOT_CONTROL_ID}'), necesaria para la regla de la extensión. No se "
                f"aplica la regla.",
                file=sys.stderr,
            )
            return 0
        lectura = read_extension(
            distilled_map_50_95=metricas["map_50_95"],
            control_map_50_95=control["map_50_95"],
            starting_map_50_95=partida["map_50_95"],
            pilot_control_map_50_95=control_piloto["map_50_95"],
            epochs=config.distillation.epochs,
        )
    print()
    print(lectura.summary())
    print()

    # Las tres cifras deberían venir del mismo dispositivo; si no, se indica cuál es cada una y qué
    # parte del resultado se ve afectada.
    print(f"Dispositivo de cada cifra ({CANONICAL_EVALUATOR} en las tres):")
    print(f"  destilado : {args.device_id}")
    print(f"  control   : {control['device_id'] or '(sin declarar)'}")
    print(f"  partida   : {partida['device_id'] or '(sin declarar)'}")

    dispositivos = {args.device_id, control["device_id"], partida["device_id"]}
    if len(dispositivos) > 1:
        veredicto_tocado = control["device_id"] != args.device_id
        print()
        print(
            "AVISO: las tres cifras NO vienen del mismo dispositivo.\n"
            + (
                "  El VEREDICTO queda tocado: se apoya en destilado - control, y esas\n"
                "  dos se midieron en sitios distintos.\n"
                if veredicto_tocado
                else "  El veredicto se sostiene: destilado y control comparten dispositivo.\n"
            )
            + "  La línea 'control - partida' mezcla dispositivos y vale como contexto,\n"
            "  no como cifra citable. Para cerrarla, evalúa el punto de partida en\n"
            f"  '{args.device_id}' con scripts/06_evaluate_coco.py.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
