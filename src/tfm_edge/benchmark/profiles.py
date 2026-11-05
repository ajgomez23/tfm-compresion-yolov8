"""Perfiles de medida: qué máquina mide, cómo y cuál no puede medir.

El protocolo rechaza medir con un perfil marcado como no válido. Una instancia compartida (Colab,
SageMaker) sirve para entrenar y evaluar calidad, pero sus p95 y p99 dependen de la carga de otros
usuarios.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from tfm_edge.common.config import StrictModel
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)


class ProfileError(TFMEdgeError):
    """Error en la definición de un perfil de medida."""


class BenchmarkProfile(StrictModel):
    """Definición de un perfil de medida.

    Si el perfil no es válido o no está disponible, el motivo es obligatorio. Los perfiles con GPU
    NVIDIA registran temperatura.
    """

    device_id: str = Field(min_length=1)
    description: str = ""
    device_type: Literal["cpu", "cuda"] = "cpu"
    valid_for_measurement: bool = True
    invalid_reason: str = ""
    available: bool = True
    unavailable_reason: str = ""
    num_threads: int | None = Field(default=None, ge=1)
    cpu_affinity: list[int] = Field(default_factory=list)
    requires_temperature: bool = False
    backends: list[str] = Field(default_factory=lambda: ["pytorch"])
    warmup_iterations: int = Field(default=50, ge=0)
    measure_iterations: int = Field(default=300, ge=1)
    repetitions: int = Field(default=3, ge=1)
    throughput_batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    latency_representative: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def _comprobar_coherencia(self) -> "BenchmarkProfile":
        """Comprueba que el perfil es coherente y lanza ValueError si no lo es."""
        if not self.valid_for_measurement and not self.invalid_reason:
            raise ValueError(
                f"El perfil '{self.device_id}' no es válido para medir y no dice por qué (falta "
                "'invalid_reason')."
            )
        if not self.available and not self.unavailable_reason:
            raise ValueError(
                f"El perfil '{self.device_id}' se declara no disponible pero no "
                "dice por qué. 'unavailable_reason' es obligatorio."
            )
        if self.device_type == "cuda" and not self.requires_temperature:
            raise ValueError(
                f"El perfil '{self.device_id}' usa GPU y debe registrar temperatura para "
                "distinguir el estrangulamiento térmico."
            )
        if self.cpu_affinity and self.num_threads is None:
            raise ValueError(
                f"El perfil '{self.device_id}' fija afinidad de núcleos sin fijar el número de "
                "hilos."
            )
        if self.cpu_affinity and self.num_threads and self.num_threads > len(self.cpu_affinity):
            raise ValueError(
                f"El perfil '{self.device_id}' pide {self.num_threads} hilos sobre "
                f"{len(self.cpu_affinity)} núcleos."
            )
        return self

    def assert_can_measure(self) -> None:
        """Lanza un error si el perfil no es válido para medir latencia o su hardware no está
        disponible.
        """
        from tfm_edge.benchmark.protocol import InvalidProfileError

        if not self.valid_for_measurement:
            raise InvalidProfileError(self.device_id, self.invalid_reason)
        if not self.available:
            raise ProfileError(
                f"El perfil '{self.device_id}' no está disponible: " f"{self.unavailable_reason}",
                "Se registrará con status=no_disponible.",
            )

    def unavailable_coverage_row(self) -> dict[str, Any]:
        """Fila de cobertura de un perfil que no se puede medir.

        Lleva device_id y el motivo en notes; experiment_id, runtime, precision y batch_size quedan
        vacíos porque la ausencia es del perfil, no de un experimento.
        """
        from tfm_edge.common.registry import DEPLOYMENT_COLUMNS, utc_timestamp

        motivo = self.invalid_reason if not self.valid_for_measurement else self.unavailable_reason
        fila: dict[str, Any] = {columna: "" for columna in DEPLOYMENT_COLUMNS}
        fila.update(
            {
                "device_id": self.device_id,
                "timestamp": utc_timestamp(),
                "status": "no_disponible",
                "notes": f"COBERTURA, no medida. {motivo}",
            }
        )
        return fila

    def to_registry_fields(self) -> dict[str, Any]:
        """Devuelve los campos que acompañan a cada medida en el registro."""
        return {
            "device_id": self.device_id,
            "num_threads": self.num_threads if self.num_threads is not None else "",
            "cpu_affinity": ",".join(str(c) for c in self.cpu_affinity),
            "latency_representative": self.latency_representative,
        }


def load_profile(path: Path | str) -> BenchmarkProfile:
    """Carga un perfil desde su YAML."""
    import yaml
    from pydantic import ValidationError

    ruta = Path(path)
    if not ruta.is_file():
        raise ProfileError(
            f"No existe el perfil '{ruta}'.",
            "Los perfiles del proyecto están en configs/benchmark/.",
        )
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
    try:
        return BenchmarkProfile(**datos)
    except ValidationError as error:
        raise ProfileError(
            f"El perfil '{ruta.name}' no es válido:\n{error}",
            "Los YAML de configuración se validan con extra='forbid': una clave "
            "desconocida es un error, nunca se ignora en silencio.",
        ) from error


def load_all_profiles(directory: Path | str) -> dict[str, BenchmarkProfile]:
    """Carga los perfiles de un directorio como ``{device_id: perfil}``."""
    raiz = Path(directory)
    if not raiz.is_dir():
        raise ProfileError(
            f"No existe el directorio de perfiles '{raiz}'.",
            "Deberían estar en configs/benchmark/.",
        )
    perfiles: dict[str, BenchmarkProfile] = {}
    for ruta in sorted(raiz.glob("*.yaml")):
        perfil = load_profile(ruta)
        if perfil.device_id in perfiles:
            raise ProfileError(
                f"Hay dos perfiles con device_id '{perfil.device_id}'.",
                "El identificador va en cada fila del registro y tiene que ser "
                "único, o dos máquinas distintas se confundirían en el análisis.",
            )
        perfiles[perfil.device_id] = perfil
    return perfiles


def describe_profiles(perfiles: dict[str, BenchmarkProfile]) -> str:
    """Tabla de perfiles con su estado y motivo."""
    lineas = [
        f"{'perfil':20s} {'tipo':6s} {'mide':>5s} {'dispon.':>8s} {'repr.':>6s}  motivo",
        "-" * 100,
    ]
    for device_id, p in sorted(perfiles.items()):
        motivo = p.invalid_reason or p.unavailable_reason or p.notes or ""
        lineas.append(
            f"{device_id:20s} {p.device_type:6s} "
            f"{'sí' if p.valid_for_measurement else 'NO':>5s} "
            f"{'sí' if p.available else 'NO':>8s} "
            f"{'sí' if p.latency_representative else 'NO':>6s}  {motivo[:52]}"
        )
    return "\n".join(lineas)
