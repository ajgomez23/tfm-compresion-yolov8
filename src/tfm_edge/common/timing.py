"""Medida de tiempos y estadísticos de latencia.

Las llamadas a CUDA son asíncronas: sin sincronizar, perf_counter mide el encolado y no la
ejecución. Toda medida en GPU llama a synchronize_device antes de arrancar y antes de parar el
cronómetro; Timer lo hace solo si se le pasa el dispositivo.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Callable, Iterable, Sequence, TypeVar

from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

#: Percentiles que se reportan.
REQUIRED_PERCENTILES = (50, 90, 95, 99)


def synchronize_device(device: str | None = None) -> bool:
    """Sincroniza CUDA antes o después de medir.

    Devuelve True si ha sincronizado. En CPU no hace nada; si se pide CUDA y no está disponible lo
    deja en el log.
    """
    if device is None or not str(device).lower().startswith("cuda"):
        return False

    try:
        import torch
    except ImportError:
        logger.error(
            "Se pidió sincronizar el dispositivo '%s' pero PyTorch no está instalado. "
            "La medida de tiempo resultante NO es válida.",
            device,
        )
        return False

    if not torch.cuda.is_available():
        logger.error(
            "Se pidió sincronizar el dispositivo '%s' pero CUDA no está disponible. "
            "La medida de tiempo resultante NO es válida.",
            device,
        )
        return False

    torch.cuda.synchronize(device=device if ":" in str(device) else None)
    return True


@dataclass
class TimingResult:
    """Resultado de una medida de tiempo."""

    wall_time_s: float
    cpu_time_s: float
    label: str
    synchronized: bool

    @property
    def wall_time_ms(self) -> float:
        """Tiempo de pared en milisegundos."""
        return self.wall_time_s * 1000.0


class Timer:
    """Cronómetro de tiempo de pared y de CPU. Con un dispositivo CUDA sincroniza al entrar y al
    salir.

    >>> with Timer("inferencia", device="cuda") as t:
    ...     salida = modelo(entrada)
    """

    def __init__(self, label: str = "bloque", device: str | None = None, log: bool = True) -> None:
        """Inicializa el cronómetro."""
        self.label = label
        self.device = device
        self.log = log
        self.result: TimingResult | None = None
        self._wall_inicio: float = 0.0
        self._cpu_inicio: float = 0.0
        self._sincronizado: bool = False

    def __enter__(self) -> "Timer":
        """Arranca el cronómetro, sincronizando antes si procede."""
        self._sincronizado = synchronize_device(self.device)
        self._wall_inicio = time.perf_counter()
        self._cpu_inicio = time.process_time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Para el cronómetro, sincronizando antes si procede."""
        sincronizado_salida = synchronize_device(self.device)
        wall = time.perf_counter() - self._wall_inicio
        cpu = time.process_time() - self._cpu_inicio
        self.result = TimingResult(
            wall_time_s=wall,
            cpu_time_s=cpu,
            label=self.label,
            synchronized=self._sincronizado and sincronizado_salida,
        )
        if self.log:
            logger.debug(
                "%s: %.3f ms de pared, %.3f ms de CPU%s",
                self.label,
                wall * 1000.0,
                cpu * 1000.0,
                "" if self.result.synchronized or self.device is None else " (SIN sincronizar)",
            )


def timed(label: str | None = None, device: str | None = None) -> Callable[[F], F]:
    """Decorador que mide el tiempo de una función."""

    def decorador(func: F) -> F:
        @functools.wraps(func)
        def envoltorio(*args: Any, **kwargs: Any) -> Any:
            with Timer(label or func.__name__, device=device):
                return func(*args, **kwargs)

        return envoltorio  # type: ignore[return-value]

    return decorador


def compute_timing_statistics(
    timings_s: Sequence[float] | Iterable[float],
    discarded: int = 0,
) -> dict[str, float | int]:
    """Estadísticos de un vector de tiempos en segundos: media, desviación, mínimo, máximo y
    percentiles, en milisegundos.

    La desviación típica es muestral (ddof=1). Lanza ValueError si el vector está vacío o tiene
    negativos.
    """
    import numpy as np

    tiempos = np.asarray(list(timings_s), dtype=np.float64)
    if tiempos.size == 0:
        raise ValueError(
            "No se puede calcular estadística sobre un vector de tiempos vacío. "
            "Comprueba que el bucle de medida llegó a ejecutarse."
        )
    if np.any(tiempos < 0):
        raise ValueError("Hay tiempos negativos en el vector; revisa cómo se usa el cronómetro.")

    tiempos_ms = tiempos * 1000.0
    estadisticos: dict[str, float | int] = {
        "latency_mean_ms": float(np.mean(tiempos_ms)),
        # Con una sola muestra la desviación muestral no está definida y se deja en 0.0.
        "latency_std_ms": float(np.std(tiempos_ms, ddof=1)) if tiempos_ms.size > 1 else 0.0,
        "latency_min_ms": float(np.min(tiempos_ms)),
        "latency_max_ms": float(np.max(tiempos_ms)),
        "num_measurements": int(tiempos_ms.size),
        "num_discarded": int(discarded),
    }
    for percentil in REQUIRED_PERCENTILES:
        # Se fija 'linear' para que el percentil no dependa del valor por defecto de NumPy.
        estadisticos[f"latency_p{percentil}_ms"] = float(
            np.percentile(tiempos_ms, percentil, method="linear")
        )
    return estadisticos


def compute_robust_statistics(timings_s: Sequence[float]) -> dict[str, float]:
    """Mediana, desviación absoluta mediana y rango intercuartílico en milisegundos, menos sensibles
    a picos aislados.
    """
    import numpy as np

    tiempos = np.asarray(list(timings_s), dtype=np.float64)
    if tiempos.size == 0:
        raise ValueError("No se puede calcular estadística robusta sobre un vector vacío.")
    tiempos_ms = tiempos * 1000.0
    mediana = float(np.median(tiempos_ms))
    return {
        "median_ms": mediana,
        "mad_ms": float(np.median(np.abs(tiempos_ms - mediana))),
        "iqr_ms": float(np.percentile(tiempos_ms, 75) - np.percentile(tiempos_ms, 25)),
    }
