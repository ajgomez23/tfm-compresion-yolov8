"""Rendimiento real del entrenamiento: si el cuello de botella es la GPU o la CPU.

La sesión de Colab tiene una T4 y una CPU de un solo núcleo físico, y el aumento de datos de
Ultralytics corre en CPU. Se cronometra el mismo dataloader real de dos formas: solo cargando datos
y con el paso completo. Si los tiempos se parecen, la GPU espera a la CPU. La utilización de la GPU
se mide aparte como confirmación.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tfm_edge.common.exceptions import DependencyNotAvailableError, TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir

logger = get_logger(__name__)

DEFAULT_MEASURE_BATCHES = 30

#: Lotes de calentamiento: el primero incluye la caché del dataset y la compilación de kernels.
DEFAULT_WARMUP_BATCHES = 5

#: Valores de workers por defecto. Con un solo núcleo, 0 (carga en el proceso principal) puede ser
#: más rápido.
DEFAULT_WORKER_OPTIONS = (0, 1, 2, 4)

#: Horas útiles por sesión de Colab; es una cota conservadora, no un dato oficial.
COLAB_SESSION_LIMIT_HOURS = 12.0


class ProfilingError(TFMEdgeError):
    """Error al medir el rendimiento del entrenamiento."""


# Muestreo de la utilización de la GPU


#: Muestras mínimas para fiarse de la lectura de utilización.
MIN_GPU_SAMPLES = 20

#: Fracción mínima de la ventana que debe cubrir el muestreo.
MIN_GPU_COVERAGE = 0.5


class GpuUtilizationSampler:
    """Muestrea la utilización de la GPU en un hilo secundario durante toda la ventana de medida.

    Usa torch.cuda.utilization() (NVML, en microsegundos) y nvidia-smi como reserva; con nvidia-smi
    salían muy pocas muestras y una lectura llegó a dar 0 % en un paso limitado por GPU. Se guardan
    percentiles, número de muestras y cobertura, y check_gpu_coherence marca las lecturas dudosas.
    Sin backend los estadísticos quedan vacíos.

    >>> with GpuUtilizationSampler() as muestreador:
    ...     entrenar_unos_lotes()
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        """Configura el muestreador."""
        self.interval_s = interval_s
        self.samples: list[float] = []
        self.memory_samples: list[float] = []
        self.backend: str = ""
        self.unavailable_reason: str = ""
        self.window_s: float = 0.0
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None
        self._inicio: float = 0.0

    def _probar_nvml(self) -> bool:
        """True si torch.cuda.utilization() responde."""
        try:
            import torch

            if not torch.cuda.is_available():
                self.unavailable_reason = "torch no ve ninguna GPU"
                return False
            torch.cuda.utilization()
            return True
        except Exception as error:
            # Normalmente falta pynvml, que torch usa por debajo.
            self.unavailable_reason = f"torch.cuda.utilization no disponible: {error}"
            return False

    def _leer_nvml(self) -> tuple[float, float] | None:
        """Utilización y memoria vía NVML, o None."""
        try:
            import torch

            return float(torch.cuda.utilization()), float(torch.cuda.memory_usage())
        except Exception:
            return None

    def _probar_smi(self) -> bool:
        """True si nvidia-smi devuelve una lectura válida."""
        if shutil.which("nvidia-smi") is None:
            self.unavailable_reason += "; nvidia-smi no está en el PATH"
            return False
        return self._leer_smi() is not None

    def _leer_smi(self) -> tuple[float, float] | None:
        """Utilización y memoria (MiB) vía nvidia-smi, o None."""
        try:
            salida = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if salida.returncode != 0 or not salida.stdout.strip():
                return None
            partes = salida.stdout.strip().splitlines()[0].split(",")
            return float(partes[0].strip()), float(partes[1].strip())
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return None

    def _muestrear(self) -> None:
        """Bucle del hilo secundario."""
        leer = self._leer_nvml if self.backend == "nvml" else self._leer_smi
        while not self._parar.is_set():
            lectura = leer()
            if lectura is not None:
                self.samples.append(lectura[0])
                self.memory_samples.append(lectura[1])
            self._parar.wait(self.interval_s)

    def __enter__(self) -> "GpuUtilizationSampler":
        """Elige backend y arranca el muestreo."""
        if self._probar_nvml():
            self.backend = "nvml"
        elif self._probar_smi():
            self.backend = "nvidia-smi"
        else:
            self.backend = "no_disponible"
            logger.warning(
                "No se puede medir la utilización de la GPU: %s. "
                "La medida quedará con utilización vacía, no con cero.",
                self.unavailable_reason,
            )
            return self

        self._parar.clear()
        self._inicio = time.perf_counter()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *_excepcion: object) -> None:
        """Detiene el muestreo, espera al hilo y cierra la ventana."""
        if self._hilo is None:
            return
        self._parar.set()
        self._hilo.join(timeout=10)
        self.window_s = time.perf_counter() - self._inicio

    def _percentil(self, fraccion: float) -> float | str:
        """Percentil de las muestras, o cadena vacía si no hay."""
        if not self.samples:
            return ""
        ordenadas = sorted(self.samples)
        indice = min(int(fraccion * len(ordenadas)), len(ordenadas) - 1)
        return round(ordenadas[indice], 1)

    @property
    def mean_utilization(self) -> float | str:
        """Utilización media de la GPU, en tanto por ciento."""
        return round(statistics.mean(self.samples), 1) if self.samples else ""

    @property
    def median_utilization(self) -> float | str:
        """Utilización mediana de la GPU, en tanto por ciento."""
        return round(statistics.median(self.samples), 1) if self.samples else ""

    @property
    def p10_utilization(self) -> float | str:
        """Percentil 10 de la utilización."""
        return self._percentil(0.10)

    @property
    def p90_utilization(self) -> float | str:
        """Percentil 90 de la utilización."""
        return self._percentil(0.90)

    @property
    def min_utilization(self) -> float | str:
        """Mínimo de la utilización."""
        return round(min(self.samples), 1) if self.samples else ""

    @property
    def max_utilization(self) -> float | str:
        """Máximo de la utilización."""
        return round(max(self.samples), 1) if self.samples else ""

    @property
    def num_samples(self) -> int:
        """Número de muestras recogidas."""
        return len(self.samples)

    @property
    def effective_interval_s(self) -> float | str:
        """Intervalo real entre muestras. Si es mucho mayor que ``interval_s``, el backend frena el
        muestreo.
        """
        if not self.samples or self.window_s <= 0:
            return ""
        return round(self.window_s / len(self.samples), 4)

    @property
    def coverage(self) -> float | str:
        """Fracción de la ventana efectivamente muestreada, en tanto por uno."""
        if not self.samples or self.window_s <= 0:
            return ""
        esperadas = self.window_s / self.interval_s
        return round(min(len(self.samples) / esperadas, 1.0), 3)


# Resultados


@dataclass
class ThroughputMeasurement:
    """Rendimiento medido para una configuración de entrenamiento.

    ``data_only_s_per_batch`` itera solo el dataloader y ``full_step_s_per_batch`` el paso completo.
    ``bottleneck`` es ``cpu``, ``gpu`` o ``indeterminado``.
    """

    size: str
    resolution: int
    batch_size: int
    workers: int
    amp: bool
    variant: str = ""
    cache: str = "False"
    mosaic: float = 1.0
    dataset_kind: str = "original"
    data_only_s_per_batch: float | str = ""
    full_step_s_per_batch: float | str = ""
    gpu_utilization_mean: float | str = ""
    gpu_utilization_median: float | str = ""
    gpu_utilization_p10: float | str = ""
    gpu_utilization_p90: float | str = ""
    gpu_utilization_min: float | str = ""
    gpu_utilization_max: float | str = ""
    gpu_samples: int = 0
    gpu_sampler_backend: str = ""
    gpu_sampling_window_s: float | str = ""
    gpu_effective_interval_s: float | str = ""
    gpu_coverage: float | str = ""
    gpu_utilization_status: str = ""
    warnings: list[str] = field(default_factory=list)
    batches_measured: int = 0
    dataset_batches_per_epoch: int | str = ""
    estimated_epoch_minutes: float | str = ""
    bottleneck: str = ""
    data_fraction: float | str = ""
    error: str = ""

    @property
    def gpu_utilization_is_trustworthy(self) -> bool:
        """True si la lectura de utilización es fiable."""
        return self.gpu_utilization_status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        return asdict(self)


@dataclass
class TrainingProjection:
    """Extrapolación del tiempo total de entrenamiento."""

    size: str
    epochs: int
    epoch_minutes: float
    total_hours: float
    sessions_needed: float
    fits_in_one_session: bool

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la proyección como diccionario serializable."""
        return asdict(self)


# Medida


def _construir_trainer(
    size: str,
    resolution: int,
    batch_size: int,
    workers: int,
    amp: bool,
    data_config: Path | str,
    num_classes: int,
    output_dir: Path | str,
    cache: str | bool = False,
    mosaic: float = 1.0,
    close_mosaic: int = 10,
) -> Any:
    """Construye un DetectionTrainer con sus dataloaders montados.

    ``cache="disk"`` guarda las imágenes decodificadas en .npy; ``mosaic=0`` desactiva el aumento
    más caro.
    """
    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
    except ImportError as error:
        raise DependencyNotAvailableError("ultralytics", "medir el rendimiento") from error

    from tfm_edge.models.detector import Detector

    overrides = {
        "model": f"yolov8{size}.yaml",
        "data": str(data_config),
        "epochs": 1,
        "batch": batch_size,
        "imgsz": resolution,
        "workers": workers,
        "amp": amp,
        "project": str(output_dir),
        "name": f"profile_{size}_{resolution}_b{batch_size}_w{workers}",
        "exist_ok": True,
        "verbose": False,
        "plots": False,
        "val": False,
        "cache": cache,
        "mosaic": mosaic,
        "close_mosaic": close_mosaic,
    }
    try:
        trainer = DetectionTrainer(overrides=overrides)
        detector = Detector.from_architecture(
            size=size, num_classes=num_classes, input_resolution=resolution, pretrained=False
        )
        # Igual que el Trainer del proyecto: se asigna el modelo para que setup_model() no lo
        # reconstruya.
        trainer.model = detector.model
        trainer._setup_train()
    except Exception as error:
        raise ProfilingError(
            f"No se ha podido montar el entrenamiento: {type(error).__name__}: {error}",
            "Comprueba que el YAML del dataset existe y que el dataset está preparado.",
        ) from error
    return trainer


def measure_throughput(
    size: str,
    resolution: int,
    batch_size: int,
    workers: int,
    data_config: Path | str,
    output_dir: Path | str,
    amp: bool = True,
    num_classes: int = 17,
    measure_batches: int = DEFAULT_MEASURE_BATCHES,
    warmup_batches: int = DEFAULT_WARMUP_BATCHES,
    cache: str | bool = False,
    mosaic: float = 1.0,
    close_mosaic: int = 10,
    variant: str = "",
    dataset_kind: str = "original",
) -> ThroughputMeasurement:
    """Mide el rendimiento sobre el dataset real, separando la carga de datos del paso completo."""
    import torch

    medida = ThroughputMeasurement(
        size=size,
        resolution=resolution,
        batch_size=batch_size,
        workers=workers,
        amp=amp,
        variant=variant or "base",
        cache=str(cache),
        mosaic=mosaic,
        dataset_kind=dataset_kind,
    )
    try:
        trainer = _construir_trainer(
            size,
            resolution,
            batch_size,
            workers,
            amp,
            data_config,
            num_classes,
            output_dir,
            cache=cache,
            mosaic=mosaic,
            close_mosaic=close_mosaic,
        )
    except TFMEdgeError as error:
        medida.error = str(error)[:300]
        return medida

    cargador = trainer.train_loader
    medida.dataset_batches_per_epoch = len(cargador)

    # Fase 1: solo datos
    tiempos_datos: list[float] = []
    iterador = iter(cargador)
    for indice in range(warmup_batches + measure_batches):
        inicio = time.perf_counter()
        try:
            next(iterador)
        except StopIteration:
            break
        transcurrido = time.perf_counter() - inicio
        if indice >= warmup_batches:
            tiempos_datos.append(transcurrido)
    medida.data_only_s_per_batch = (
        round(statistics.median(tiempos_datos), 4) if tiempos_datos else ""
    )

    # Fase 2: datos y cómputo, muestreando la GPU
    modelo = trainer.model
    modelo.train()
    optimizador = trainer.optimizer
    escalador = getattr(trainer, "scaler", None)
    dispositivo = trainer.device

    tiempos_completos: list[float] = []
    iterador = iter(cargador)
    with GpuUtilizationSampler() as muestreador:
        for indice in range(warmup_batches + measure_batches):
            inicio = time.perf_counter()
            try:
                lote = next(iterador)
            except StopIteration:
                break
            try:
                lote = trainer.preprocess_batch(lote)
                optimizador.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type="cuda", enabled=amp and str(dispositivo).startswith("cuda")
                ):
                    perdida, _ = modelo(lote)
                total = perdida.sum() if perdida.ndim > 0 else perdida
                if escalador is not None:
                    escalador.scale(total).backward()
                    escalador.step(optimizador)
                    escalador.update()
                else:
                    total.backward()
                    optimizador.step()
                if str(dispositivo).startswith("cuda"):
                    torch.cuda.synchronize()
            except Exception as error:  # pragma: no cover - depende del modelo
                medida.error = f"{type(error).__name__}: {str(error)[:200]}"
                break
            transcurrido = time.perf_counter() - inicio
            if indice >= warmup_batches:
                tiempos_completos.append(transcurrido)

    medida.batches_measured = len(tiempos_completos)
    medida.gpu_utilization_mean = muestreador.mean_utilization
    medida.gpu_utilization_median = muestreador.median_utilization
    medida.gpu_utilization_p10 = muestreador.p10_utilization
    medida.gpu_utilization_p90 = muestreador.p90_utilization
    medida.gpu_utilization_min = muestreador.min_utilization
    medida.gpu_utilization_max = muestreador.max_utilization
    medida.gpu_samples = muestreador.num_samples
    medida.gpu_sampler_backend = muestreador.backend
    medida.gpu_sampling_window_s = round(muestreador.window_s, 2) if muestreador.window_s else ""
    medida.gpu_effective_interval_s = muestreador.effective_interval_s
    medida.gpu_coverage = muestreador.coverage

    if tiempos_completos:
        por_lote = statistics.median(tiempos_completos)
        medida.full_step_s_per_batch = round(por_lote, 4)
        if isinstance(medida.dataset_batches_per_epoch, int):
            medida.estimated_epoch_minutes = round(
                por_lote * medida.dataset_batches_per_epoch / 60.0, 2
            )
        if tiempos_datos:
            fraccion = statistics.median(tiempos_datos) / por_lote
            medida.data_fraction = round(fraccion, 3)
            # El diagnóstico sale de comparar tiempos, así que una lectura de GPU dudosa no lo
            # invalida.
            medida.bottleneck = _diagnosticar(fraccion, muestreador.median_utilization)

    # La medida se valida antes de devolverla.
    check_gpu_coherence(medida, muestreador.unavailable_reason)
    for aviso in medida.warnings:
        logger.warning("[%s yolov8%s w=%d] %s", medida.variant, medida.size, medida.workers, aviso)

    return medida


def check_gpu_coherence(medida: ThroughputMeasurement, unavailable_reason: str = "") -> None:
    """Marca como sospechosa una lectura de utilización poco creíble.

    Motivos: 0 % con el paso limitado por GPU, menos de MIN_GPU_SAMPLES muestras o cobertura pobre
    de la ventana. Modifica la medida en el sitio.
    """
    avisos: list[str] = []

    if medida.gpu_sampler_backend == "no_disponible" or medida.gpu_samples == 0:
        medida.gpu_utilization_status = "no_disponible"
        motivo = unavailable_reason or "no se recogió ninguna muestra"
        medida.warnings.append(f"Utilización de GPU no medida: {motivo}")
        return

    if medida.bottleneck == "gpu" and medida.gpu_utilization_median == 0.0:
        avisos.append(
            "Utilización de GPU de 0,0 % con el cuello clasificado como GPU: "
            "contradictorio, la lectura es un artefacto de muestreo y NO debe "
            "usarse. El diagnóstico del cuello sigue siendo válido porque sale de "
            "comparar tiempos, no de esta lectura."
        )

    if medida.gpu_samples < MIN_GPU_SAMPLES:
        avisos.append(
            f"Solo {medida.gpu_samples} muestras de GPU (mínimo {MIN_GPU_SAMPLES}): "
            f"la mediana no es representativa. Backend "
            f"'{medida.gpu_sampler_backend}', intervalo efectivo "
            f"{medida.gpu_effective_interval_s} s."
        )

    if isinstance(medida.gpu_coverage, float) and medida.gpu_coverage < MIN_GPU_COVERAGE:
        avisos.append(
            f"El muestreo solo cubrió el {medida.gpu_coverage:.0%} de la ventana: "
            f"el backend '{medida.gpu_sampler_backend}' está frenando la lectura."
        )

    medida.warnings.extend(avisos)
    medida.gpu_utilization_status = "sospechosa" if avisos else "ok"


def _diagnosticar(data_fraction: float, gpu_utilization: float | str) -> str:
    """Diagnóstico del cuello de botella: ``"cpu"``, ``"gpu"`` o ``"indeterminado"``."""
    # Si la carga de datos ocupa la mayor parte del paso, la GPU espera. 0,6 es un umbral
    # conservador.
    if data_fraction >= 0.6:
        return "cpu"
    if data_fraction <= 0.3:
        return "gpu"
    # En la zona intermedia desempata la utilización de la GPU.
    if isinstance(gpu_utilization, (int, float)):
        if gpu_utilization < 50:
            return "cpu"
        if gpu_utilization > 80:
            return "gpu"
    return "indeterminado"


def sweep_workers(
    size: str,
    resolution: int,
    batch_size: int,
    data_config: Path | str,
    output_dir: Path | str,
    worker_options: Sequence[int] = DEFAULT_WORKER_OPTIONS,
    amp: bool = True,
    num_classes: int = 17,
    measure_batches: int = DEFAULT_MEASURE_BATCHES,
    cache: str | bool = False,
    mosaic: float = 1.0,
    variant: str = "",
    dataset_kind: str = "original",
) -> list[ThroughputMeasurement]:
    """Mide con varios valores de workers. Con un solo núcleo, más workers no siempre es mejor."""
    medidas: list[ThroughputMeasurement] = []
    for workers in worker_options:
        logger.info("Midiendo yolov8%s con workers=%d...", size, workers)
        medida = measure_throughput(
            size=size,
            resolution=resolution,
            batch_size=batch_size,
            workers=workers,
            data_config=data_config,
            output_dir=output_dir,
            amp=amp,
            num_classes=num_classes,
            measure_batches=measure_batches,
            cache=cache,
            mosaic=mosaic,
            variant=variant,
            dataset_kind=dataset_kind,
        )
        medidas.append(medida)
        logger.info(
            "  workers=%d: %s s/lote, GPU al %s %%, cuello: %s",
            workers,
            medida.full_step_s_per_batch,
            medida.gpu_utilization_median,
            medida.bottleneck or "?",
        )
    return medidas


def project_training_time(
    medidas: Sequence[ThroughputMeasurement],
    epochs: int,
    session_limit_hours: float = COLAB_SESSION_LIMIT_HOURS,
) -> list[TrainingProjection]:
    """Horas totales estimadas a partir del tiempo por época, con la mejor configuración de workers
    por tamaño.
    """
    mejores: dict[str, ThroughputMeasurement] = {}
    for m in medidas:
        if not isinstance(m.estimated_epoch_minutes, (int, float)):
            continue
        actual = mejores.get(m.size)
        if actual is None or m.estimated_epoch_minutes < actual.estimated_epoch_minutes:
            mejores[m.size] = m

    proyecciones: list[TrainingProjection] = []
    for size, m in sorted(mejores.items()):
        horas = float(m.estimated_epoch_minutes) * epochs / 60.0
        proyecciones.append(
            TrainingProjection(
                size=size,
                epochs=epochs,
                epoch_minutes=float(m.estimated_epoch_minutes),
                total_hours=round(horas, 2),
                sessions_needed=round(horas / session_limit_hours, 2),
                fits_in_one_session=horas <= session_limit_hours,
            )
        )
    return proyecciones


def summarize(
    medidas: Sequence[ThroughputMeasurement], proyecciones: Sequence[TrainingProjection]
) -> str:
    """Informe legible del rendimiento."""
    lineas = [
        "RENDIMIENTO DEL ENTRENAMIENTO: ¿el cuello es la CPU o la GPU?",
        "",
        f"{'variante':22s} {'modelo':8s} {'work':>5s} {'s/lote':>8s} "
        f"{'datos':>8s} {'%datos':>7s} {'GPU%':>7s} {'n':>5s} {'min/epoca':>10s}  cuello",
        "-" * 100,
    ]
    sospechosas = 0
    for m in medidas:
        etiqueta = (m.variant or "base")[:22]
        if m.error:
            lineas.append(
                f"{etiqueta:22s} yolov8{m.size:2s} {m.workers:>5d}   ERROR: {m.error[:40]}"
            )
            continue
        # Las lecturas que no pasan la comprobación se marcan en la tabla.
        if m.gpu_utilization_status == "ok":
            gpu = f"{m.gpu_utilization_median}"
        elif m.gpu_utilization_status == "no_disponible":
            gpu = "n/d"
        else:
            gpu = f"{m.gpu_utilization_median}!"
            sospechosas += 1
        lineas.append(
            f"{etiqueta:22s} yolov8{m.size:2s} {m.workers:>5d} "
            f"{m.full_step_s_per_batch:>8} {m.data_only_s_per_batch:>8} "
            f"{m.data_fraction:>7} {gpu:>7} {m.gpu_samples:>5d} "
            f"{m.estimated_epoch_minutes:>10}  {m.bottleneck}"
        )

    if sospechosas:
        lineas += [
            "",
            f"AVISO: {sospechosas} lectura(s) de GPU marcadas con '!' NO son fiables "
            "y no deben usarse en la memoria:",
        ]
        for m in medidas:
            for aviso in m.warnings:
                lineas.append(f"  - [yolov8{m.size} w={m.workers}] {aviso}")
        lineas.append("  El diagnostico del cuello NO depende de esta lectura: sale de comparar")
        lineas.append("  el tiempo de solo-datos con el del paso completo.")

    if proyecciones:
        lineas += [
            "",
            "PROYECCION DEL TIEMPO TOTAL",
            "",
            f"{'modelo':9s} {'epocas':>7s} {'min/epoca':>10s} {'horas':>8s} "
            f"{'sesiones':>9s}  cabe en una sesion",
            "-" * 66,
        ]
        for p in proyecciones:
            lineas.append(
                f"yolov8{p.size:2s} {p.epochs:>7d} {p.epoch_minutes:>10.2f} "
                f"{p.total_hours:>8.2f} {p.sessions_needed:>9.2f}  "
                f"{'si' if p.fits_in_one_session else 'NO'}"
            )
    lineas += [
        "",
        "%datos = fraccion del tiempo del paso que consume la carga de datos.",
        "  >= 0,60 -> el cuello es la CPU: la GPU espera. Mas VRAM no ayuda.",
        "  <= 0,30 -> el cuello es el computo de la GPU.",
        "n = muestras de utilizacion de GPU recogidas. '!' = lectura no fiable.",
    ]
    return "\n".join(lineas)


def save_measurements(
    medidas: Sequence[ThroughputMeasurement],
    proyecciones: Sequence[TrainingProjection],
    path: Path | str,
    environment: dict[str, Any] | None = None,
) -> Path:
    """Escribe las medidas y proyecciones en JSON."""
    destino = ensure_parent_dir(path)
    contenido = {
        "environment": environment or {},
        "note": (
            "Medido sobre el DATASET REAL, no sintetico: el aumento de datos es lo "
            "que consume CPU, y con ruido aleatorio no se ejecuta. Se cronometra "
            "por separado la iteracion del dataloader y el paso completo, para "
            "poder decidir si el cuello es la CPU o la GPU."
        ),
        "measurements": [m.to_dict() for m in medidas],
        "projections": [p.to_dict() for p in proyecciones],
    }
    destino.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Medidas de rendimiento escritas en %s", destino)
    return destino
