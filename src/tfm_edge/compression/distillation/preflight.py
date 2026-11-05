"""Comprobación previa de la destilación: si maestro y alumno caben a la vez en la GPU.

Ejecuta una época del DistillationTrainer real sobre un subconjunto de las imágenes con más
anotaciones, con el lote y la resolución de la campaña, y mide el pico de memoria y el tiempo por
paso. El asignador de Ultralytics rellena objetivos hasta el máximo de anotaciones de una imagen del
lote, por eso se eligen las más pobladas. No garantiza el máximo exacto (el mosaico puede juntar
más), no mide calidad y el tiempo de época que estima no incluye la validación.
"""

from __future__ import annotations

import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from tfm_edge.common.config import DistillationConfig
from tfm_edge.common.exceptions import HardwareNotAvailableError, TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.distillation.trainer import DistillationTrainer
from tfm_edge.data.smoke_subset import count_labels, prepare_smoke_subset
from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Imágenes de entrenamiento: con lote 16 son ocho lotes y dos pasos de optimizador (nbs=64). Hace
#: falta al menos uno para que existan los búferes de momento.
PREFLIGHT_TRAIN_IMAGES = 128

#: Imágenes de validación; con pocas, la validación no domina el tiempo.
PREFLIGHT_VAL_IMAGES = 16

#: Pasos iniciales que no se promedian: el primero crea los búferes del optimizador y hace la
#: primera reserva de CUDA.
WARMUP_STEPS = 2

#: Dos tipos de fallo distintos. "memoria": el par no cabe. "camino": el bucle se rompió por otra
#: causa (dependencias, entorno, dataset) y no se sabe si cabe.
FAILURE_MEMORY = "memoria"
FAILURE_PATH = "camino"


class PreflightError(TFMEdgeError):
    """La comprobación previa no ha llegado a medir nada."""


@dataclass
class PreflightResult:
    """Lo medido por la comprobación previa.

    ``memory_measured`` es False en CPU: en ese caso ``completed`` solo indica que el camino
    funciona. ``failure_kind`` distingue memoria de camino y ``failure_traceback`` guarda la traza
    completa. ``measured_epoch_s`` sale de multiplicar el tiempo por paso por los pasos reales y no
    incluye la validación.
    """

    completed: bool = False
    memory_measured: bool = False
    reason: str = ""
    failure_kind: str = ""
    failure_traceback: str = ""
    device: str = ""
    gpu_name: str = ""
    # Cadena vacía para no medido, como en el registro; 0.0 parecería una medida.
    gpu_total_mb: float | str = ""
    peak_allocated_mb: float | str = ""
    peak_reserved_mb: float | str = ""
    headroom_mb: float | str = ""
    batch_size: int = 0
    resolution: int = 0
    alpha: float = 0.0
    teacher_size: str = ""
    student_size: str = ""
    teacher_params: int = 0
    student_params: int = 0
    adapter_params: int = 0
    steps_timed: int = 0
    seconds_per_step: float = 0.0
    seconds_per_step_max: float = 0.0
    train_images_full_split: int = 0
    steps_per_epoch_full: int = 0
    measured_epoch_s: float = 0.0
    max_labels_subset: int = 0
    max_labels_split: int = 0
    run_dir: str = ""
    step_times_s: list[float] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        """True si el bucle terminó y además se midió la memoria."""
        return self.completed and self.memory_measured

    @property
    def conditions_match_campaign(self) -> bool:
        """True si se midió en las condiciones de la campaña: GPU, 640 px y lote 16."""
        return self.memory_measured and self.resolution == 640 and self.batch_size == 16

    def to_dict(self) -> dict[str, Any]:
        """Diccionario serializable, incluidos ``fits`` y ``conditions_match_campaign``, que
        ``asdict`` no recoge.
        """
        return {
            **asdict(self),
            "fits": self.fits,
            "conditions_match_campaign": self.conditions_match_campaign,
        }

    def summary(self) -> str:
        """Resumen legible de la comprobación."""
        dispositivo = self.gpu_name or self.device
        if self.memory_measured:
            dispositivo += f" ({self.gpu_total_mb:.0f} MiB)"
        lineas = [
            "COMPROBACIÓN PREVIA DE MEMORIA: maestro y alumno juntos",
            f"  Dispositivo      : {dispositivo}",
            f"  Par              : maestro '{self.teacher_size}' "
            f"({self.teacher_params:,} par.) + alumno '{self.student_size}' "
            f"({self.student_params:,} par.) + adaptadores "
            f"({self.adapter_params:,} par.)",
            f"  Condiciones      : lote {self.batch_size}, {self.resolution} px, "
            f"FP32 sin AMP, alpha={self.alpha:g}",
        ]
        if not self.completed:
            if self.failure_kind == FAILURE_MEMORY:
                lineas.append(f"  RESULTADO        : NO CABE. {self.reason}")
            else:
                lineas.append(
                    f"  RESULTADO        : SIN MEDIR. El bucle se ha roto antes de "
                    f"llegar a mirar la memoria, y por algo que no es memoria: "
                    f"{self.reason}"
                )
                lineas.append(
                    "                     Esto NO dice que el par no quepa. No dice "
                    "nada: la medida no llegó a tomarse."
                )
            return "\n".join(lineas)

        if self.memory_measured:
            lineas += [
                f"  Pico asignado    : {self.peak_allocated_mb:.0f} MiB",
                f"  Pico reservado   : {self.peak_reserved_mb:.0f} MiB",
                f"  Margen           : {self.headroom_mb:.0f} MiB "
                f"({100 * self.headroom_mb / self.gpu_total_mb:.1f} % del total)",
            ]
        else:
            lineas.append(
                "  Memoria          : NO MEDIDA. Sin GPU no hay memoria de GPU que "
                "mirar. Los campos de pico van vacíos, no a cero."
            )

        lineas.append(
            f"  Peor caso visto  : {self.max_labels_subset} anotaciones en una "
            f"imagen (máximo de la partición: {self.max_labels_split})"
        )
        if self.steps_timed:
            lineas.append(
                f"  Tiempo por paso  : {self.seconds_per_step:.3f} s de media, "
                f"{self.seconds_per_step_max:.3f} s el más lento "
                f"({self.steps_timed} pasos cronometrados)"
            )
            lineas.append(
                f"  Época real       : {self.steps_per_epoch_full} pasos "
                f"({self.train_images_full_split} imágenes): "
                f"{self.measured_epoch_s / 60:.2f} min de cómputo por época "
                f"EN ESTAS CONDICIONES"
            )
        if self.fits:
            lineas.append("  RESULTADO        : CABE")
        else:
            lineas.append(
                "  RESULTADO        : CAMINO RECORRIDO, MEMORIA SIN COMPROBAR. Esto "
                "NO autoriza a lanzar la campaña."
            )
        return "\n".join(lineas)


class _StepTimer:
    """Cronómetro de paso con sincronización de CUDA a ambos lados."""

    def __init__(self, en_cuda: bool) -> None:
        """Prepara el cronómetro."""
        self.en_cuda = en_cuda
        self.times: list[float] = []
        self._inicio = 0.0

    def _sincronizar(self) -> None:
        """Espera a que la GPU haya terminado lo encolado."""
        if self.en_cuda:
            torch.cuda.synchronize()

    def start(self, _trainer: Any) -> None:
        """Arranca el reloj al empezar un lote."""
        self._sincronizar()
        self._inicio = time.perf_counter()

    def stop(self, _trainer: Any) -> None:
        """Para el reloj al cerrar un lote."""
        self._sincronizar()
        self.times.append(time.perf_counter() - self._inicio)


def classify_failure(error: BaseException) -> tuple[str, str]:
    """Clasifica un fallo del bucle como de memoria o de camino. Devuelve ``(clase, motivo)``.

    Solo es de memoria si el propio error lo dice. Ultralytics envuelve en RuntimeError cualquier
    fallo al comprobar el dataset, así que un RuntimeError no basta. El motivo se recorta a 300
    caracteres.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return FAILURE_MEMORY, f"OutOfMemoryError: {str(error)[:300]}"
    if isinstance(error, RuntimeError) and "out of memory" in str(error).lower():
        return FAILURE_MEMORY, f"Sin memoria: {str(error)[:300]}"
    return FAILURE_PATH, f"{type(error).__name__}: {str(error)[:300]}"


def _max_labels(data_yaml: Path, lista: Path) -> int:
    """Anotaciones de la imagen más poblada de una lista de partición."""
    entradas = [linea for linea in lista.read_text(encoding="utf-8").splitlines() if linea.strip()]
    return max((count_labels(data_yaml, entrada) for entrada in entradas), default=0)


def _contar(lista: Path) -> int:
    """Número de entradas no vacías de una lista de partición."""
    return sum(1 for linea in lista.read_text(encoding="utf-8").splitlines() if linea.strip())


def run_preflight(
    student: Detector,
    teacher: Detector,
    config: DistillationConfig,
    data_config: Path | str,
    splits_dir: Path | str,
    work_dir: Path | str,
    batch_size: int = 16,
    resolution: int = 640,
    workers: int = 2,
    device: str = "",
    seed: int | None = None,
    train_images: int = PREFLIGHT_TRAIN_IMAGES,
    val_images: int = PREFLIGHT_VAL_IMAGES,
    allow_cpu: bool = False,
) -> PreflightResult:
    """Comprueba midiendo que la destilación cabe en la GPU.

    Los detectores se modifican durante la época de prueba, así que hay que volver a cargarlos para
    la campaña. ``batch_size`` y ``resolution`` deben ser los de la campaña. No lanza error si no
    cabe: devuelve ``fits=False`` con el motivo.
    """
    en_cuda = torch.cuda.is_available() and not device.startswith("cpu")
    if not en_cuda and not allow_cpu:
        raise HardwareNotAvailableError(
            "una GPU CUDA",
            "comprobar si maestro y alumno caben juntos en memoria",
        )

    data_yaml = Path(data_config)
    if not data_yaml.is_file():
        raise PreflightError(
            f"No existe el YAML de dataset '{data_yaml}'.",
            "Prepara el dataset con scripts/02_prepare_dataset.py.",
        )

    trabajo = Path(work_dir)
    trabajo.mkdir(parents=True, exist_ok=True)
    subconjunto = trabajo / "subset"

    # Entrenamiento: las imágenes más pobladas. Validación: las primeras.
    yaml_reducido, _ = prepare_smoke_subset(
        data_yaml, Path(splits_dir), "train", train_images, subconjunto, order="most_labels"
    )
    prepare_smoke_subset(data_yaml, Path(splits_dir), "val", val_images, subconjunto, order="first")

    lista_completa = data_yaml.parent / "ultralytics_train.txt"
    lista_reducida = subconjunto / "ultralytics_train.txt"
    if not lista_reducida.is_file() or _contar(lista_reducida) == 0:
        raise PreflightError(
            f"El subconjunto de la comprobación previa ha salido vacío ('{lista_reducida}').",
            "Comprueba que 'ultralytics_train.txt' tiene entradas.",
        )

    imagenes_completas = _contar(lista_completa) if lista_completa.is_file() else 0
    resultado = PreflightResult(
        device=device or ("cuda" if en_cuda else "cpu"),
        batch_size=batch_size,
        resolution=resolution,
        alpha=config.alpha,
        teacher_size=teacher.size,
        student_size=student.size,
        teacher_params=teacher.count_parameters(),
        student_params=student.count_parameters(),
        train_images_full_split=imagenes_completas,
        steps_per_epoch_full=(
            math.ceil(imagenes_completas / batch_size) if imagenes_completas else 0
        ),
        max_labels_subset=_max_labels(data_yaml, lista_reducida),
        max_labels_split=_max_labels(data_yaml, lista_completa) if lista_completa.is_file() else 0,
    )
    if en_cuda:
        propiedades = torch.cuda.get_device_properties(torch.cuda.current_device())
        resultado.gpu_name = propiedades.name
        resultado.gpu_total_mb = round(propiedades.total_memory / 2**20, 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cronometro = _StepTimer(en_cuda=en_cuda)
    destilador = DistillationTrainer(
        student=student,
        teacher=teacher,
        data_config=yaml_reducido,
        output_dir=trabajo / "run",
        config=config.model_copy(update={"epochs": 1}),
        resolution=resolution,
        batch_size=batch_size,
        workers=workers,
        device=device,
        patience=1,
        seed=seed,
        extra_args={"val": True, "plots": False, "save_period": -1},
        extra_callbacks=[
            ("on_train_batch_start", cronometro.start),
            ("on_train_batch_end", cronometro.stop),
        ],
    )

    logger.info(
        "Comprobación previa: una época sobre %d imágenes (las más pobladas, hasta "
        "%d anotaciones), lote %d, %d px, FP32.",
        _contar(lista_reducida),
        resultado.max_labels_subset,
        batch_size,
        resolution,
    )

    try:
        entrenamiento = destilador.train(run_name="preflight")
    except (RuntimeError, TFMEdgeError) as error:
        # RuntimeError incluye OutOfMemoryError, pero también los fallos que Ultralytics envuelve al
        # cargar el dataset y las guardas del propio bucle. classify_failure decide cuál es.
        resultado.failure_kind, resultado.reason = classify_failure(error)
        resultado.failure_traceback = traceback.format_exc()
    else:
        resultado.completed = True
        resultado.run_dir = str(entrenamiento.training.run_dir)
        resultado.adapter_params = entrenamiento.adapter_params

    if en_cuda:
        resultado.memory_measured = True
        resultado.peak_allocated_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        resultado.peak_reserved_mb = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        resultado.headroom_mb = round(resultado.gpu_total_mb - resultado.peak_reserved_mb, 1)

    tiempos = cronometro.times[WARMUP_STEPS:]
    resultado.step_times_s = [round(t, 4) for t in cronometro.times]
    if tiempos:
        resultado.steps_timed = len(tiempos)
        resultado.seconds_per_step = round(sum(tiempos) / len(tiempos), 4)
        resultado.seconds_per_step_max = round(max(tiempos), 4)
        resultado.measured_epoch_s = round(
            resultado.seconds_per_step * resultado.steps_per_epoch_full, 1
        )
    elif resultado.completed:
        # Si cabe pero no se cronometró ningún paso, el tiempo queda a cero.
        logger.warning(
            "No hay pasos cronometrados tras descartar los %d de calentamiento: "
            "el subconjunto es demasiado pequeño para medir tiempo.",
            WARMUP_STEPS,
        )
    return resultado
