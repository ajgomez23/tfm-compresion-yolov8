"""Protocolo de medida común a todas las variantes (ver docs/PROTOCOLO_MEDIDA.md).

Un perfil no válido para medir se rechaza con error. Se guarda el vector completo de tiempos, se
repite el bloque R veces para separar la variabilidad entre ejecuciones de la de dentro, y se mide
sobre imágenes reales, porque el postprocesado depende del número de detecciones. En GPU se registra
la temperatura al inicio y al final para detectar estrangulamiento térmico.
"""

from __future__ import annotations

import gzip
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir
from tfm_edge.common.registry import utc_timestamp
from tfm_edge.common.timing import compute_timing_statistics

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.benchmark.backends import InferenceBackend

logger = get_logger(__name__)

#: Iteraciones de calentamiento que se descartan.
DEFAULT_WARMUP_ITERATIONS = 50

#: Iteraciones medidas por repetición.
DEFAULT_MEASURE_ITERATIONS = 300

#: Repeticiones del bloque completo.
DEFAULT_REPETITIONS = 3

#: Lotes del barrido de rendimiento.
DEFAULT_THROUGHPUT_BATCH_SIZES = (1, 2, 4, 8, 16)

#: Segundos de ventana por tamaño de lote.
DEFAULT_THROUGHPUT_WINDOW_S = 10.0

#: Carga media por encima de la cual no se mide sin ``--force``.
MAX_SYSTEM_LOAD_PER_CORE = 0.5

#: Intervalo de muestreo de la memoria residente, en segundos.
RSS_SAMPLING_INTERVAL_S = 0.05


class ProtocolError(TFMEdgeError):
    """Error del protocolo de medida."""


class InvalidProfileError(TFMEdgeError):
    """Se ha intentado medir con un perfil no válido; la campaña se detiene."""

    def __init__(self, device_id: str, motivo: str) -> None:
        super().__init__(
            f"El perfil '{device_id}' NO es válido para medir latencia: {motivo}",
            "Los percentiles de latencia sobre una máquina compartida no son "
            "comparables ni entre sí, y contaminarían la frontera de Pareto del "
            "apartado 6.4.\n"
            "     Colab y SageMaker sirven para entrenar; la latencia se mide en "
            "hardware propio.\n"
            "     Si de verdad quieres una medida no comparable, márcala "
            "explícitamente como tal en la configuración del perfil.",
        )


# Resultados


@dataclass
class LatencyMeasurement:
    """Latencia de una repetición con el vector completo de tiempos."""

    repetition: int
    timings_s: list[float] = field(default_factory=list)
    preprocess_s: list[float] = field(default_factory=list)
    postprocess_s: list[float] = field(default_factory=list)
    detections_per_image: list[int] = field(default_factory=list)
    statistics: dict[str, float | int] = field(default_factory=dict)
    temperature_start_c: float | str = ""
    temperature_end_c: float | str = ""
    started_at: str = ""
    ended_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        return asdict(self)


@dataclass
class MemoryMeasurement:
    """Memoria consumida, separando pesos de activaciones."""

    gpu_peak_allocated_bytes: int | str = ""
    gpu_peak_reserved_bytes: int | str = ""
    rss_idle_bytes: int | str = ""
    rss_peak_bytes: int | str = ""
    rss_samples: int = 0
    notes: str = ""

    @property
    def activations_bytes(self) -> int | str:
        """Memoria atribuible a las activaciones: pico menos reposo."""
        if isinstance(self.rss_peak_bytes, int) and isinstance(self.rss_idle_bytes, int):
            return max(0, self.rss_peak_bytes - self.rss_idle_bytes)
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        datos = asdict(self)
        datos["activations_bytes"] = self.activations_bytes
        return datos


@dataclass
class ThroughputMeasurement:
    """Rendimiento sostenido para un tamaño de lote."""

    batch_size: int
    images_per_second: float | str = ""
    window_s: float = 0.0
    batches_completed: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        return asdict(self)


@dataclass
class ArtifactSize:
    """Tamaño del artefacto en disco y comprimido con gzip."""

    path: str
    disk_bytes: int
    gzip_bytes: int

    @property
    def compression_ratio(self) -> float:
        """Factor de compresión de gzip."""
        return round(self.disk_bytes / self.gzip_bytes, 3) if self.gzip_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        datos = asdict(self)
        datos["compression_ratio"] = self.compression_ratio
        return datos


@dataclass
class BenchmarkResult:
    """Resultado completo de medir un artefacto.

    ``between_run_std_ms`` es la dispersión entre repeticiones de la latencia mediana.
    ``measurement_regime`` es la condición de medida declarada de la campaña.
    """

    device_id: str
    backend: str
    precision: str = "fp32"
    #: Proveedor de ejecución del motor; distingue GPU de CPU dentro del mismo motor.
    execution_provider: str = ""
    input_resolution: int = 640
    num_threads: int | str = ""
    cpu_affinity: str = ""
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS
    measure_iterations: int = DEFAULT_MEASURE_ITERATIONS
    repetitions: int = DEFAULT_REPETITIONS
    latency: list[LatencyMeasurement] = field(default_factory=list)
    aggregated: dict[str, float | int] = field(default_factory=dict)
    between_run_std_ms: float | str = ""
    memory: MemoryMeasurement = field(default_factory=MemoryMeasurement)
    throughput: list[ThroughputMeasurement] = field(default_factory=list)
    artifact: ArtifactSize | None = None
    system_load: dict[str, Any] = field(default_factory=dict)
    environment_id: str = ""
    status: str = "ok"
    measurement_regime: str = ""
    notes: str = ""
    started_at: str = ""
    ended_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        datos = asdict(self)
        datos["artifact"] = self.artifact.to_dict() if self.artifact else None
        datos["memory"] = self.memory.to_dict()
        return datos

    def summary(self) -> str:
        """Genera un resumen legible."""
        a = self.aggregated
        lineas = [
            f"MEDIDA: {self.device_id} / {self.backend} / {self.precision} "
            f"@ {self.input_resolution} px",
            "",
            f"  iteraciones : {self.measure_iterations} x {self.repetitions} "
            f"repeticiones ({self.warmup_iterations} de calentamiento descartadas)",
            f"  hilos       : {self.num_threads}   afinidad: {self.cpu_affinity or 'sin fijar'}",
            "",
        ]
        if a:
            lineas += [
                f"  latencia media : {a.get('latency_mean_ms', 0):.3f} ms  "
                f"(desv. {a.get('latency_std_ms', 0):.3f})",
                f"  p50 / p90      : {a.get('latency_p50_ms', 0):.3f} / "
                f"{a.get('latency_p90_ms', 0):.3f} ms",
                f"  p95 / p99      : {a.get('latency_p95_ms', 0):.3f} / "
                f"{a.get('latency_p99_ms', 0):.3f} ms",
                f"  min / max      : {a.get('latency_min_ms', 0):.3f} / "
                f"{a.get('latency_max_ms', 0):.3f} ms",
                f"  dispersión ENTRE repeticiones: "
                + (
                    f"{self.between_run_std_ms} ms"
                    if isinstance(self.between_run_std_ms, float)
                    else "n/d (hace falta más de una repetición)"
                ),
            ]
        temperaturas = [
            (m.temperature_start_c, m.temperature_end_c)
            for m in self.latency
            if isinstance(m.temperature_start_c, float)
        ]
        if temperaturas:
            lineas.append(
                f"  temperatura    : {temperaturas[0][0]:.0f} -> " f"{temperaturas[-1][1]:.0f} °C"
            )
        if self.throughput:
            lineas += ["", "  Rendimiento sostenido:"]
            for t in self.throughput:
                valor = f"{t.images_per_second} img/s" if not t.error else f"ERROR: {t.error[:40]}"
                lineas.append(f"    lote {t.batch_size:>3d}: {valor}")
        if self.artifact:
            lineas += [
                "",
                f"  artefacto      : {self.artifact.disk_bytes / 2**20:.2f} MiB "
                f"({self.artifact.gzip_bytes / 2**20:.2f} MiB comprimido, "
                f"{self.artifact.compression_ratio}x)",
            ]
        return "\n".join(lineas)


# Comprobaciones previas


def check_system_load(force: bool = False) -> dict[str, Any]:
    """Comprueba que la máquina está lo bastante libre para medir.

    Con carga alta los percentiles reflejan la contención. Lanza ProtocolError salvo que se pase
    ``force``.
    """
    nucleos = os.cpu_count() or 1
    try:
        carga_1, carga_5, carga_15 = os.getloadavg()
    except (OSError, AttributeError):  # pragma: no cover - Windows
        return {
            "available": False,
            "reason": "getloadavg no está disponible en esta plataforma",
            "forced": force,
        }

    por_nucleo = carga_1 / nucleos
    aceptable = por_nucleo <= MAX_SYSTEM_LOAD_PER_CORE
    informe = {
        "available": True,
        "load_1min": round(carga_1, 2),
        "load_5min": round(carga_5, 2),
        "load_15min": round(carga_15, 2),
        "cpu_count": nucleos,
        "load_per_core": round(por_nucleo, 3),
        "threshold": MAX_SYSTEM_LOAD_PER_CORE,
        "acceptable": aceptable,
        "forced": force,
    }

    if not aceptable:
        if not force:
            raise ProtocolError(
                f"La carga del sistema es {por_nucleo:.2f} por núcleo, por encima "
                f"del umbral {MAX_SYSTEM_LOAD_PER_CORE}.",
                "Cierra lo que esté corriendo y vuelve a intentarlo, o pasa --force (quedará "
                "anotado en 'notes').",
            )
        logger.warning(
            "Se mide con carga alta (%.2f por núcleo) porque se ha pasado --force. "
            "Los percentiles no serán comparables con los de otras medidas.",
            por_nucleo,
        )
    return informe


def read_gpu_temperature() -> float | str:
    """Temperatura de la GPU en grados Celsius, o cadena vacía si no se puede leer."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return ""
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if salida.returncode != 0 or not salida.stdout.strip():
            return ""
        return float(salida.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return ""


def configure_cpu(num_threads: int | None = None, affinity: Sequence[int] | None = None) -> dict:
    """Fija los hilos y la afinidad de núcleos del proceso.

    Es lo que distingue workstation_cpu de constrained_cpu, que usan la misma CPU.
    """
    import torch

    aplicado: dict[str, Any] = {"num_threads": "", "cpu_affinity": ""}

    if num_threads is not None:
        if num_threads < 1:
            raise ProtocolError(
                f"num_threads debe ser >= 1 y es {num_threads}.",
                "Usa 2 para el perfil restringido.",
            )
        torch.set_num_threads(num_threads)
        aplicado["num_threads"] = torch.get_num_threads()
    else:
        aplicado["num_threads"] = torch.get_num_threads()

    if affinity:
        nucleos = sorted(set(int(c) for c in affinity))
        if not hasattr(os, "sched_setaffinity"):  # pragma: no cover - no Linux
            raise ProtocolError(
                "Este sistema no permite fijar la afinidad de núcleos.",
                "El perfil 'constrained_cpu' necesita afinidad fijada para ser "
                "reproducible. Mide en Linux, o declara el perfil como no válido.",
            )
        disponibles = os.sched_getaffinity(0)
        if invalidos := [c for c in nucleos if c not in disponibles and c >= (os.cpu_count() or 0)]:
            raise ProtocolError(
                f"Los núcleos {invalidos} no existen en esta máquina "
                f"({os.cpu_count()} núcleos).",
                "Ajusta 'cpu_affinity' en el perfil.",
            )
        os.sched_setaffinity(0, nucleos)
        aplicado["cpu_affinity"] = ",".join(str(c) for c in sorted(os.sched_getaffinity(0)))

    logger.info(
        "CPU configurada: %s hilos, afinidad '%s'",
        aplicado["num_threads"],
        aplicado["cpu_affinity"] or "sin fijar",
    )
    return aplicado


# Medidas


class _RssSampler:
    """Muestrea la memoria residente del proceso en un hilo secundario."""

    def __init__(self, interval_s: float = RSS_SAMPLING_INTERVAL_S) -> None:
        """Configura el muestreador."""
        self.interval_s = interval_s
        self.samples: list[int] = []
        self._parar = threading.Event()
        self._hilo: threading.Thread | None = None

    def _muestrear(self) -> None:
        """Bucle del hilo."""
        import psutil

        proceso = psutil.Process()
        while not self._parar.is_set():
            try:
                self.samples.append(int(proceso.memory_info().rss))
            except Exception:  # pragma: no cover - proceso desapareciendo
                pass
            self._parar.wait(self.interval_s)

    def __enter__(self) -> "_RssSampler":
        """Arranca el muestreo."""
        self._parar.clear()
        self._hilo = threading.Thread(target=self._muestrear, daemon=True)
        self._hilo.start()
        return self

    def __exit__(self, *_excepcion: object) -> None:
        """Detiene el muestreo."""
        self._parar.set()
        if self._hilo is not None:
            self._hilo.join(timeout=5)


def measure_latency(
    backend: "InferenceBackend",
    images: Sequence[Any],
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    measure_iterations: int = DEFAULT_MEASURE_ITERATIONS,
    repetitions: int = DEFAULT_REPETITIONS,
    with_temperature: bool = False,
    conf: float = 0.25,
    iou: float = 0.7,
) -> list[LatencyMeasurement]:
    """Mide la latencia: calentamiento, iteraciones medidas y repeticiones.

    ``images`` deben ser varias imágenes reales. La temperatura es obligatoria en GPU.
    """
    if len(images) < 2:
        raise ProtocolError(
            f"Se han pasado {len(images)} imágenes y hacen falta al menos 2.",
            "Hacen falta varias imágenes: el coste del postprocesado depende del número de "
            "detecciones.",
        )
    if measure_iterations < 1 or repetitions < 1:
        raise ProtocolError(
            "measure_iterations y repetitions deben ser >= 1.",
            "Los valores del protocolo son 300 y 3.",
        )

    medidas: list[LatencyMeasurement] = []
    for repeticion in range(repetitions):
        logger.info("Repetición %d de %d...", repeticion + 1, repetitions)
        medida = LatencyMeasurement(
            repetition=repeticion,
            started_at=utc_timestamp(),
            temperature_start_c=read_gpu_temperature() if with_temperature else "",
        )

        for indice in range(warmup_iterations):
            backend.run(images[indice % len(images)], conf=conf, iou=iou)

        for indice in range(measure_iterations):
            # Se rota por las imágenes en orden fijo, igual en todas las repeticiones y variantes.
            salida = backend.run(images[indice % len(images)], conf=conf, iou=iou)
            medida.timings_s.append(salida.inference_s)
            medida.preprocess_s.append(salida.preprocess_s)
            medida.postprocess_s.append(salida.postprocess_s)
            medida.detections_per_image.append(salida.num_detections)

        medida.temperature_end_c = read_gpu_temperature() if with_temperature else ""
        medida.ended_at = utc_timestamp()
        medida.statistics = compute_timing_statistics(medida.timings_s, discarded=warmup_iterations)
        # Si no hubo ninguna detección, la NMS no se ejecutó y ese tiempo no representa nada.
        if medida.detections_per_image and not any(medida.detections_per_image):
            aviso = (
                "Ninguna de las iteraciones produjo detecciones: el tiempo de "
                "postprocesado NO es representativo, porque la supresión de no "
                "máximos no llegó a ejecutarse. La latencia de inferencia sí es "
                "válida. Suele indicar un modelo sin entrenar o un umbral de "
                "confianza demasiado alto."
            )
            logger.warning("%s", aviso)
            medida.statistics["postprocess_representative"] = 0
        else:
            medida.statistics["postprocess_representative"] = 1
        medidas.append(medida)

        if with_temperature and isinstance(medida.temperature_start_c, float):
            delta = float(medida.temperature_end_c) - medida.temperature_start_c
            logger.info(
                "  p50 %.3f ms, temperatura de %.0f a %.0f °C (%+.0f)",
                medida.statistics["latency_p50_ms"],
                medida.temperature_start_c,
                medida.temperature_end_c,
                delta,
            )
    return medidas


def aggregate_latency(medidas: Sequence[LatencyMeasurement]) -> tuple[dict, float | str]:
    """Agrega las repeticiones y devuelve ``(estadísticos, desviación entre repeticiones en
    ms)``."""
    if not medidas:
        raise ProtocolError(
            "No hay ninguna repetición que agregar.", "Ejecuta measure_latency antes."
        )
    todos: list[float] = []
    for medida in medidas:
        todos.extend(medida.timings_s)
    agregados = compute_timing_statistics(todos)

    medianas = [float(m.statistics["latency_p50_ms"]) for m in medidas if m.statistics]
    entre = round(statistics.stdev(medianas), 4) if len(medianas) > 1 else ""
    return agregados, entre


def measure_memory(
    backend: "InferenceBackend",
    images: Sequence[Any],
    iterations: int = 30,
    conf: float = 0.25,
    iou: float = 0.7,
) -> MemoryMeasurement:
    """Mide la memoria en reposo (modelo ya cargado) y durante la inferencia."""
    import torch

    medida = MemoryMeasurement()

    try:
        import psutil

        medida.rss_idle_bytes = int(psutil.Process().memory_info().rss)
    except ImportError:
        medida.notes = "psutil no disponible: no se ha medido la memoria residente"

    hay_gpu = torch.cuda.is_available() and str(backend.device).startswith("cuda")
    if hay_gpu:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    else:
        medida.notes = (
            medida.notes + "; " if medida.notes else ""
        ) + "sin GPU en este perfil: la memoria de GPU queda vacía, no a cero"

    with _RssSampler() as muestreador:
        for indice in range(iterations):
            backend.run(images[indice % len(images)], conf=conf, iou=iou)

    medida.rss_samples = len(muestreador.samples)
    if muestreador.samples:
        medida.rss_peak_bytes = max(muestreador.samples)

    if hay_gpu:
        torch.cuda.synchronize()
        medida.gpu_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
        medida.gpu_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    return medida


def measure_throughput(
    backend: "InferenceBackend",
    images: Sequence[Any],
    batch_sizes: Sequence[int] = DEFAULT_THROUGHPUT_BATCH_SIZES,
    window_s: float = DEFAULT_THROUGHPUT_WINDOW_S,
) -> list[ThroughputMeasurement]:
    """Mide el rendimiento sostenido para varios tamaños de lote.

    Es una medida aparte de la latencia, que se toma con lote 1. Un fallo de memoria en un lote
    grande se registra y el barrido sigue.
    """
    import torch

    resultados: list[ThroughputMeasurement] = []
    for lote in batch_sizes:
        resultado = ThroughputMeasurement(batch_size=lote)
        try:
            tensores = torch.cat(
                [backend.preprocess(images[i % len(images)]) for i in range(lote)], dim=0
            )
            # Calentamiento breve: el primer lote reserva memoria y elige kernels.
            for _ in range(3):
                backend.infer(tensores)
            backend.synchronize()

            inicio = time.perf_counter()
            completados = 0
            while time.perf_counter() - inicio < window_s:
                backend.infer(tensores)
                backend.synchronize()
                completados += 1
            transcurrido = time.perf_counter() - inicio

            resultado.window_s = round(transcurrido, 3)
            resultado.batches_completed = completados
            resultado.images_per_second = round(completados * lote / transcurrido, 2)
        except Exception as error:
            resultado.error = f"{type(error).__name__}: {str(error)[:150]}"
            logger.warning("Lote %d ha fallado: %s", lote, resultado.error)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        resultados.append(resultado)
    return resultados


def measure_artifact_size(path: Path | str) -> ArtifactSize:
    """Tamaño del artefacto en disco y con gzip; un fichero o un directorio."""
    ruta = Path(path)
    if not ruta.exists():
        raise ProtocolError(
            f"No existe el artefacto '{ruta}'.", "Comprueba la ruta del modelo exportado."
        )

    if ruta.is_dir():
        ficheros = [f for f in ruta.rglob("*") if f.is_file()]
        disco = sum(f.stat().st_size for f in ficheros)
        comprimido = sum(len(gzip.compress(f.read_bytes(), compresslevel=6)) for f in ficheros)
    else:
        disco = ruta.stat().st_size
        comprimido = len(gzip.compress(ruta.read_bytes(), compresslevel=6))

    return ArtifactSize(path=str(ruta), disk_bytes=disco, gzip_bytes=comprimido)


def to_deployment_row(
    result: BenchmarkResult,
    experiment_id: str,
    raw_timings_path: Path | str = "",
    artifact_path: Path | str = "",
    export_time_s: float | str = "",
    engine_build_time_s: float | str = "",
) -> dict[str, Any]:
    """Convierte un resultado en la fila de deployment_metrics.csv. Lo no medido queda vacío.

    El vector de tiempos no cabe en el CSV; la fila guarda la ruta del JSON en ``raw_timings_path``.
    """
    a = result.aggregated or {}
    memoria = result.memory

    def media(valores: Sequence[list[float]]) -> float | str:
        """Promedia en milisegundos una lista de listas de segundos."""
        planos = [x for lista in valores for x in lista]
        return round(1000.0 * sum(planos) / len(planos), 4) if planos else ""

    # El rendimiento del registro es el de lote 1; el resto está en el JSON.
    lote1 = next((t for t in result.throughput if t.batch_size == 1 and not t.error), None)

    def mib(valor: int | str) -> float | str:
        """Convierte bytes a MiB, dejando vacío lo no medido."""
        return round(valor / 2**20, 3) if isinstance(valor, int) else ""

    fila: dict[str, Any] = {
        "experiment_id": experiment_id,
        "runtime": result.backend,
        "precision": result.precision,
        "device_id": result.device_id,
        "environment_id": result.environment_id,
        "input_resolution": result.input_resolution,
        "batch_size": 1,
        "num_threads": result.num_threads,
        "cpu_affinity": result.cpu_affinity,
        "execution_provider": result.execution_provider,
        "warmup_iters": result.warmup_iterations,
        "measure_iters": result.measure_iterations,
        "num_repeats": result.repetitions,
        "preprocess_mean_ms": media([m.preprocess_s for m in result.latency]),
        "postprocess_mean_ms": media([m.postprocess_s for m in result.latency]),
        "inference_mean_ms": a.get("latency_mean_ms", ""),
        "throughput_img_s": lote1.images_per_second if lote1 else "",
        "peak_gpu_memory_mb": mib(memoria.gpu_peak_reserved_bytes),
        "peak_host_memory_mb": mib(memoria.rss_peak_bytes),
        "model_load_memory_mb": mib(memoria.rss_idle_bytes),
        "artifact_size_bytes": result.artifact.disk_bytes if result.artifact else "",
        "artifact_size_gzip_bytes": result.artifact.gzip_bytes if result.artifact else "",
        "artifact_path": str(artifact_path or (result.artifact.path if result.artifact else "")),
        "export_time_s": export_time_s,
        "engine_build_time_s": engine_build_time_s,
        "temperature_start_c": result.latency[0].temperature_start_c if result.latency else "",
        "temperature_end_c": result.latency[-1].temperature_end_c if result.latency else "",
        "raw_timings_path": str(raw_timings_path),
        "latency_representative": "",
        "timestamp": result.ended_at or utc_timestamp(),
        "status": result.status,
        "notes": result.notes,
    }
    # Estadísticos de latencia tal como salen del protocolo.
    for clave in (
        "latency_mean_ms",
        "latency_std_ms",
        "latency_min_ms",
        "latency_max_ms",
        "latency_p50_ms",
        "latency_p90_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ):
        fila[clave] = round(a[clave], 4) if clave in a else ""
    return fila


def save_benchmark_result(result: BenchmarkResult, path: Path | str) -> Path:
    """Escribe el resultado en JSON con el vector completo de tiempos."""
    destino = ensure_parent_dir(path)
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    temporal.write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporal.replace(destino)
    logger.info("Resultado de la medida volcado en %s", destino)
    return destino
