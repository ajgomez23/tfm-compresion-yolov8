"""Envolvente de memoria: qué combinaciones de tamaño, resolución y lote caben en la GPU.

Cada combinación se mide en un subproceso propio. Medirlas en bucle en el mismo proceso daba picos
no monótonos, porque quedaban referencias vivas (detector, optimizador, escalador) y empty_cache no
liberaba nada. Se registran max_memory_allocated (consumo real) y max_memory_reserved (lo que decide
si cabe), y check_monotonicity comprueba que el pico crece con el lote y la resolución.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from tfm_edge.common.exceptions import DependencyNotAvailableError, HardwareNotAvailableError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir

logger = get_logger(__name__)

DEFAULT_SIZES = ("n", "s", "m", "l")
DEFAULT_RESOLUTIONS = (416, 512, 640)

#: Lotes hasta 128 para encontrar el límite real y no el final del rango probado.
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)

SUBPROCESS_TIMEOUT = 600


@dataclass
class MemoryMeasurement:
    """Resultado de medir una combinación."""

    size: str
    resolution: int
    batch_size: int
    amp: bool
    fits: bool
    peak_allocated_mb: float | str = ""
    peak_reserved_mb: float | str = ""
    params: int | str = ""
    step_time_s: float | str = ""
    error: str = ""
    isolation: str = "subprocess"

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        return asdict(self)


# Medida en el proceso actual (la ejecuta el subproceso)


def measure_one_inprocess(
    size: str,
    resolution: int,
    batch_size: int,
    amp: bool,
    num_classes: int = 17,
    device: str = "cuda",
) -> MemoryMeasurement:
    """Mide una combinación en el proceso actual.

    Está pensada para ejecutarse en un subproceso nuevo; para medir desde código, usar measure_one.
    """
    try:
        import torch
    except ImportError as error:
        raise DependencyNotAvailableError("torch", "medir el consumo de memoria") from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise HardwareNotAvailableError("una GPU CUDA", "medir el envolvente de memoria")

    import time

    from tfm_edge.models.detector import Detector

    medida = MemoryMeasurement(
        size=size,
        resolution=resolution,
        batch_size=batch_size,
        amp=amp,
        fits=False,
        isolation="inprocess",
    )
    en_cuda = device.startswith("cuda")
    if en_cuda:
        torch.cuda.reset_peak_memory_stats()

    try:
        detector = Detector.from_architecture(
            size=size, num_classes=num_classes, input_resolution=resolution, pretrained=False
        )
        modelo = detector.model.to(device)
        modelo.train()
        medida.params = detector.count_parameters()

        optimizador = torch.optim.SGD(modelo.parameters(), lr=0.01, momentum=0.9)
        escalador = torch.amp.GradScaler(device="cuda", enabled=amp and en_cuda)
        entrada = torch.randn(batch_size, 3, resolution, resolution, device=device)

        inicio = time.perf_counter()
        # Dos pasos: los búferes del optimizador no existen hasta después del primero.
        for _ in range(2):
            optimizador.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp and en_cuda):
                salida = modelo(entrada)
                tensor = salida[0] if isinstance(salida, (list, tuple)) else salida
                perdida = tensor.float().pow(2).mean()
            escalador.scale(perdida).backward()
            escalador.step(optimizador)
            escalador.update()
        if en_cuda:
            torch.cuda.synchronize()
        medida.step_time_s = round((time.perf_counter() - inicio) / 2, 4)

        if en_cuda:
            medida.peak_allocated_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            medida.peak_reserved_mb = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        medida.fits = True

    except torch.cuda.OutOfMemoryError as error:
        medida.error = f"OutOfMemoryError: {str(error)[:200]}"
    except RuntimeError as error:
        texto = str(error).lower()
        medida.error = (
            f"OOM: {str(error)[:200]}"
            if "out of memory" in texto
            else f"{type(error).__name__}: {str(error)[:200]}"
        )
    return medida


# Medida aislada en subproceso


def measure_one(
    size: str,
    resolution: int,
    batch_size: int,
    amp: bool,
    num_classes: int = 17,
    device: str = "cuda",
    timeout: int = SUBPROCESS_TIMEOUT,
) -> MemoryMeasurement:
    """Mide una combinación en un subproceso nuevo.

    Si el subproceso muere sin salida (normalmente porque el sistema lo mata por memoria) se
    devuelve ``fits=False`` con el motivo.
    """
    comando = [
        sys.executable,
        "-m",
        "tfm_edge.models.memory_envelope",
        "--size",
        size,
        "--resolution",
        str(resolution),
        "--batch-size",
        str(batch_size),
        "--num-classes",
        str(num_classes),
        "--device",
        device,
    ]
    if amp:
        comando.append("--amp")

    try:
        resultado = subprocess.run(
            comando, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return MemoryMeasurement(
            size=size,
            resolution=resolution,
            batch_size=batch_size,
            amp=amp,
            fits=False,
            error=f"el subproceso ha superado los {timeout} s",
        )

    salida = resultado.stdout.strip().splitlines()
    for linea in reversed(salida):
        if linea.startswith("{"):
            try:
                return MemoryMeasurement(**json.loads(linea))
            except (json.JSONDecodeError, TypeError):
                continue

    # Sin JSON el proceso ha muerto; lo normal es que el sistema lo haya matado por memoria del
    # host.
    return MemoryMeasurement(
        size=size,
        resolution=resolution,
        batch_size=batch_size,
        amp=amp,
        fits=False,
        error=(
            f"el subproceso terminó con código {resultado.returncode} sin producir "
            f"medida. stderr: {resultado.stderr.strip()[:200]}"
        ),
    )


def measure_envelope(
    sizes: Sequence[str] = DEFAULT_SIZES,
    resolutions: Sequence[int] = DEFAULT_RESOLUTIONS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    amp_options: Sequence[bool] = (True, False),
    num_classes: int = 17,
    device: str = "cuda",
    stop_on_first_oom: bool = True,
    isolation: str = "subprocess",
) -> list[MemoryMeasurement]:
    """Mide todas las combinaciones pedidas.

    ``isolation="inprocess"`` es más rápido pero arrastra estado; solo sirve para depurar.
    """
    if isolation not in ("subprocess", "inprocess"):
        raise ValueError(f"Modo de aislamiento desconocido: '{isolation}'.")
    if isolation == "inprocess":
        logger.warning(
            "Aislamiento 'inprocess': las medidas pueden arrastrar estado entre "
            "combinaciones y salir NO monótonas. Solo para depurar."
        )

    medir = measure_one if isolation == "subprocess" else measure_one_inprocess
    medidas: list[MemoryMeasurement] = []
    for amp in amp_options:
        for size in sizes:
            for resolucion in resolutions:
                for lote in sorted(batch_sizes):
                    medida = medir(size, resolucion, lote, amp, num_classes, device)
                    medidas.append(medida)
                    estado = (
                        f"{medida.peak_reserved_mb} MiB reservados, "
                        f"{medida.peak_allocated_mb} asignados"
                        if medida.fits
                        else "NO CABE"
                    )
                    logger.info(
                        "yolov8%s %dpx lote %-3d amp=%-5s: %s",
                        size,
                        resolucion,
                        lote,
                        amp,
                        estado,
                    )
                    if not medida.fits and stop_on_first_oom:
                        break
    return medidas


# Validación de la medida


@dataclass
class MonotonicityReport:
    """Resultado de comprobar la coherencia de las medidas."""

    violations: list[str] = field(default_factory=list)
    checked: int = 0
    ceiling_not_reached: list[str] = field(default_factory=list)

    @property
    def is_coherent(self) -> bool:
        """True si no hay incoherencias."""
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return {
            "coherent": self.is_coherent,
            "comparisons": self.checked,
            "violations": self.violations,
            "ceiling_not_reached": self.ceiling_not_reached,
        }

    def summary(self) -> str:
        """Genera un resumen legible del informe."""
        lineas = [f"Coherencia de la medida: {self.checked} comparaciones"]
        if self.violations:
            lineas.append(f"  {len(self.violations)} INCOHERENCIAS (la medida es sospechosa):")
            lineas.extend(f"    - {v}" for v in self.violations[:15])
        else:
            lineas.append("  Sin incoherencias: el pico crece con el lote y con la resolución.")
        if self.ceiling_not_reached:
            lineas.append(
                f"  {len(self.ceiling_not_reached)} combinaciones NO alcanzaron su límite "
                "en el rango probado:"
            )
            lineas.extend(f"    - {c}" for c in self.ceiling_not_reached[:15])
        return "\n".join(lineas)


def check_monotonicity(medidas: Iterable[MemoryMeasurement]) -> MonotonicityReport:
    """Comprueba que el pico de memoria crece con el lote y con la resolución."""
    informe = MonotonicityReport()
    lista = [m for m in medidas if m.fits and isinstance(m.peak_reserved_mb, (int, float))]

    # Lote creciente, mismo modelo y resolución.
    por_serie: dict[tuple[str, int, bool], list[MemoryMeasurement]] = {}
    for m in lista:
        por_serie.setdefault((m.size, m.resolution, m.amp), []).append(m)
    for (size, resolucion, amp), serie in sorted(por_serie.items()):
        serie.sort(key=lambda m: m.batch_size)
        for anterior, siguiente in zip(serie, serie[1:]):
            informe.checked += 1
            if float(siguiente.peak_reserved_mb) < float(anterior.peak_reserved_mb):
                informe.violations.append(
                    f"yolov8{size} {resolucion}px amp={amp}: el lote {siguiente.batch_size} "
                    f"reserva {siguiente.peak_reserved_mb} MiB, MENOS que el lote "
                    f"{anterior.batch_size} con {anterior.peak_reserved_mb} MiB"
                )

    # Resolución creciente, mismo modelo y lote.
    por_resolucion: dict[tuple[str, int, bool], list[MemoryMeasurement]] = {}
    for m in lista:
        por_resolucion.setdefault((m.size, m.batch_size, m.amp), []).append(m)
    for (size, lote, amp), serie in sorted(por_resolucion.items()):
        serie.sort(key=lambda m: m.resolution)
        for anterior, siguiente in zip(serie, serie[1:]):
            informe.checked += 1
            if float(siguiente.peak_reserved_mb) < float(anterior.peak_reserved_mb):
                informe.violations.append(
                    f"yolov8{size} lote {lote} amp={amp}: {siguiente.resolution}px "
                    f"reserva {siguiente.peak_reserved_mb} MiB, MENOS que "
                    f"{anterior.resolution}px con {anterior.peak_reserved_mb} MiB"
                )

    # Combinaciones que no llegaron a su límite en el rango probado.
    todas: dict[tuple[str, int, bool], list[MemoryMeasurement]] = {}
    for m in medidas:
        todas.setdefault((m.size, m.resolution, m.amp), []).append(m)
    for (size, resolucion, amp), serie in sorted(todas.items()):
        serie.sort(key=lambda m: m.batch_size)
        if serie and serie[-1].fits:
            informe.ceiling_not_reached.append(
                f"yolov8{size} {resolucion}px amp={amp}: cabe con el lote mayor probado "
                f"({serie[-1].batch_size}); el límite real está por encima"
            )
    return informe


def summarize(medidas: Iterable[MemoryMeasurement]) -> str:
    """Tabla con el lote máximo que cabe en cada combinación."""
    lista = list(medidas)
    maximos: dict[tuple[str, int, bool], MemoryMeasurement] = {}
    techo_alcanzado: dict[tuple[str, int, bool], bool] = {}
    for m in lista:
        clave = (m.size, m.resolution, m.amp)
        if m.fits and (clave not in maximos or m.batch_size > maximos[clave].batch_size):
            maximos[clave] = m
        if not m.fits:
            techo_alcanzado[clave] = True

    lineas = [
        "LOTE MAXIMO QUE CABE, por modelo, resolucion y precision",
        "",
        f"{'modelo':9s} {'resol':>6s} {'AMP':>6s} {'lote max':>9s} "
        f"{'asignado':>10s} {'reservado':>10s}  limite",
        "-" * 72,
    ]
    for size in DEFAULT_SIZES:
        for resolucion in sorted({m.resolution for m in lista}):
            for amp in (True, False):
                clave = (size, resolucion, amp)
                if clave in maximos:
                    m = maximos[clave]
                    limite = "real" if techo_alcanzado.get(clave) else "NO ALCANZADO"
                    lineas.append(
                        f"yolov8{size:2s} {resolucion:>6d} {str(amp):>6s} {m.batch_size:>9d} "
                        f"{m.peak_allocated_mb:>10} {m.peak_reserved_mb:>10}  {limite}"
                    )
                elif any(
                    m.size == size and m.resolution == resolucion and m.amp == amp for m in lista
                ):
                    lineas.append(
                        f"yolov8{size:2s} {resolucion:>6d} {str(amp):>6s} {'NO CABE':>9s}"
                    )
    lineas.append("")
    lineas.append("asignado = max_memory_allocated (consumo real del modelo y activaciones)")
    lineas.append("reservado = max_memory_reserved (lo que PyTorch pide al driver; decide si cabe)")
    return "\n".join(lineas)


def save_measurements(
    medidas: Iterable[MemoryMeasurement],
    path: Path | str,
    environment: dict[str, Any] | None = None,
    monotonicity: MonotonicityReport | None = None,
) -> Path:
    """Escribe las medidas en JSON junto con la huella del entorno."""
    destino = ensure_parent_dir(path)
    lista = list(medidas)
    contenido = {
        "environment": environment or {},
        "isolation": "subprocess",
        "note": (
            "Cada combinacion se ha medido en un SUBPROCESO independiente que "
            "arranca, mide y muere: no puede arrastrar estado de la anterior. "
            "Se reportan las dos metricas: allocated (consumo real) y reserved "
            "(lo que decide si cabe). Los casos fits=false agotaron la memoria; "
            "no se extrapola ninguno."
        ),
        "monotonicity_check": monotonicity.to_dict() if monotonicity else {},
        "measurements": [m.to_dict() for m in lista],
    }
    destino.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Medidas de memoria escritas en %s", destino)
    return destino


# Punto de entrada del subproceso


def _worker_main() -> int:
    """Mide una combinación y escribe el resultado en JSON por stdout. Devuelve 0 también si no
    cabe.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Mide una combinacion de memoria. Uso interno del envolvente."
    )
    parser.add_argument("--size", required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-classes", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    medida = measure_one_inprocess(
        size=args.size,
        resolution=args.resolution,
        batch_size=args.batch_size,
        amp=args.amp,
        num_classes=args.num_classes,
        device=args.device,
    )
    medida.isolation = "subprocess"
    print(json.dumps(medida.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main())
