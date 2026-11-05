"""Complejidad estática de un modelo: parámetros, operaciones y tamaño en disco.

Las operaciones se cuentan con un paso hacia delante real bajo FlopCounterMode, así que reflejan los
canales que quedan tras una poda. Se dan ``gflops`` (operaciones de coma flotante, lo que cuenta
PyTorch) y ``gmacs`` (la mitad). El tamaño en disco se mide sobre el fichero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import torch
    from torch import nn

    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Resolución del paso de conteo; las operaciones dependen de ella.
DEFAULT_COMPLEXITY_RESOLUTION = 640


class ComplexityError(TFMEdgeError):
    """Error al medir la complejidad de un modelo."""


@dataclass
class ModelComplexity:
    """Complejidad estática medida de un modelo."""

    params_total: int
    params_trainable: int
    gflops: float
    gmacs: float
    resolution: int
    batch_size: int = 1
    checkpoint_bytes: int | str = ""
    state_dict_bytes: int = 0
    flops_by_module: dict[str, float] = field(default_factory=dict)
    params_by_module: dict[str, int] = field(default_factory=dict)
    counter_backend: str = "torch.utils.flop_counter"

    @property
    def params_millions(self) -> float:
        """Parámetros totales, en millones."""
        return round(self.params_total / 1e6, 3)

    @property
    def checkpoint_mib(self) -> float | str:
        """Tamaño del checkpoint, en MiB."""
        if not isinstance(self.checkpoint_bytes, int):
            return ""
        return round(self.checkpoint_bytes / 2**20, 2)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        datos = asdict(self)
        datos["params_millions"] = self.params_millions
        datos["checkpoint_mib"] = self.checkpoint_mib
        return datos

    def summary(self) -> str:
        """Genera un resumen legible."""
        lineas = [
            f"Parámetros      : {self.params_total:,} ({self.params_millions} M)",
            f"  entrenables   : {self.params_trainable:,}",
            f"GFLOPs @{self.resolution}px : {self.gflops}  ({self.gmacs} GMACs)",
            f"Pesos en memoria: {self.state_dict_bytes / 2**20:.2f} MiB",
        ]
        if isinstance(self.checkpoint_bytes, int):
            lineas.append(f"Checkpoint      : {self.checkpoint_mib} MiB")
        return "\n".join(lineas)


def count_parameters(module: "nn.Module") -> tuple[int, int]:
    """Parámetros ``(totales, entrenables)`` de un módulo."""
    total = sum(p.numel() for p in module.parameters())
    entrenables = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, entrenables


def state_dict_bytes(module: "nn.Module") -> int:
    """Bytes de parámetros y buffers, sin metadatos ni optimizador."""
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers())
    return total


def checkpoint_size_bytes(path: Path | str) -> int:
    """Devuelve el tamaño de un fichero de pesos."""
    ruta = Path(path)
    if not ruta.is_file():
        raise ComplexityError(
            f"No existe el checkpoint '{ruta}'.",
            "Comprueba la ruta, o entrena el modelo antes de medir su tamaño.",
        )
    return ruta.stat().st_size


def _desglosar_por_capa(conteos: dict[str, dict[Any, int]], total: int) -> dict[str, float]:
    """Reparte las operaciones entre capas sin contarlas dos veces.

    get_flop_counts devuelve un ámbito por nivel de la jerarquía y cada uno incluye a sus hijos. Se
    busca el nivel cuyos ámbitos suman el total (en YOLOv8, ``DetectionModel.model.<i>``).
    """
    utiles = {k: v for k, v in conteos.items() if k not in ("Global", "")}
    profundidades = sorted({len(k.split(".")) for k in utiles} - {1})

    for profundidad in profundidades:
        del_nivel = {k: v for k, v in utiles.items() if len(k.split(".")) == profundidad}
        suma = sum(sum(ops.values()) for ops in del_nivel.values())
        if suma == total:
            return {
                # Se quita el nombre de la clase raíz. Seis decimales para que la suma siga
                # cuadrando con el total.
                ".".join(nombre.split(".")[1:]): round(sum(ops.values()) / 1e9, 6)
                for nombre, ops in sorted(del_nivel.items())
            }

    logger.warning(
        "Ningún nivel de la jerarquía suma el total de operaciones: el desglose "
        "por capa se omite. El total sí es válido."
    )
    return {}


def _contar_operaciones(
    module: "nn.Module", entrada: "torch.Tensor"
) -> tuple[int, dict[str, float]]:
    """Cuenta operaciones con un paso hacia delante real y devuelve ``(flops_totales,
    GFLOPs_por_capa)``.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    contador = FlopCounterMode(display=False)
    try:
        with torch.no_grad(), contador:
            module(entrada)
    except Exception as error:
        raise ComplexityError(
            f"El paso hacia delante falló al contar operaciones: "
            f"{type(error).__name__}: {error}",
            "Comprueba que la resolución es múltiplo de 32 y que el modelo acepta "
            "la entrada. Si vienes de una poda, valida antes la estructura.",
        ) from error

    total = contador.get_total_flops()
    return total, _desglosar_por_capa(contador.get_flop_counts(), total)


def measure_complexity(
    detector: "Detector",
    resolution: int | None = None,
    batch_size: int = 1,
    checkpoint_path: Path | str | None = None,
) -> ModelComplexity:
    """Mide la complejidad estática de un detector. Las cifras se dan por imagen."""
    import torch

    resolucion = int(resolution or detector.input_resolution)
    if resolucion % 32 != 0:
        raise ComplexityError(
            f"La resolución {resolucion} no es múltiplo de 32.",
            "YOLOv8 submuestrea por 32; usa por ejemplo 416, 512, 640 u 800.",
        )
    if batch_size < 1:
        raise ComplexityError(f"batch_size debe ser >= 1 y es {batch_size}.", "Usa 1.")

    modelo = detector.model
    estaba_entrenando = modelo.training
    modelo.eval()
    try:
        entrada = torch.zeros(
            batch_size, 3, resolucion, resolucion, device=detector.device, dtype=detector.dtype
        )
        flops_totales, por_modulo = _contar_operaciones(modelo, entrada)
    finally:
        modelo.train(estaba_entrenando)

    total, entrenables = count_parameters(modelo)
    # Por imagen, para que el lote no afecte a la comparación.
    gflops = round(flops_totales / batch_size / 1e9, 4)

    complejidad = ModelComplexity(
        params_total=total,
        params_trainable=entrenables,
        gflops=gflops,
        gmacs=round(gflops / 2.0, 4),
        resolution=resolucion,
        batch_size=batch_size,
        state_dict_bytes=state_dict_bytes(modelo),
        flops_by_module={k: round(v / batch_size, 6) for k, v in por_modulo.items()},
        params_by_module=params_by_top_level_module(modelo),
    )
    if checkpoint_path is not None:
        complejidad.checkpoint_bytes = checkpoint_size_bytes(checkpoint_path)

    logger.info(
        "Complejidad medida: %s parámetros, %s GFLOPs a %d px",
        f"{total:,}",
        gflops,
        resolucion,
    )
    return complejidad


def params_by_top_level_module(module: "nn.Module") -> dict[str, int]:
    """Parámetros por módulo de primer nivel; en YOLOv8, las 23 capas."""
    desglose: dict[str, int] = {}
    for nombre, hijo in module.named_children():
        # En YOLOv8 el hijo directo es un nn.Sequential; se baja un nivel.
        nietos = list(hijo.named_children())
        if nietos:
            for sub_nombre, nieto in nietos:
                desglose[f"{nombre}.{sub_nombre}"] = sum(p.numel() for p in nieto.parameters())
        else:
            desglose[nombre] = sum(p.numel() for p in hijo.parameters())
    return desglose


def compare_complexity(referencia: ModelComplexity, comprimido: ModelComplexity) -> dict[str, Any]:
    """Compara la complejidad de un modelo comprimido con la de su referencia.

    Si parámetros y operaciones no bajan, la reducción no es estructural y se avisa. Lanza
    ComplexityError si las resoluciones no coinciden.
    """
    if referencia.resolution != comprimido.resolution:
        raise ComplexityError(
            f"Las medidas no son comparables: una a {referencia.resolution} px y "
            f"otra a {comprimido.resolution} px.",
            "Mide las dos a la misma resolución.",
        )

    def reduccion(antes: float, despues: float) -> float:
        """Calcula la reducción porcentual."""
        return round(100.0 * (antes - despues) / antes, 2) if antes else 0.0

    params = reduccion(referencia.params_total, comprimido.params_total)
    flops = reduccion(referencia.gflops, comprimido.gflops)
    resultado: dict[str, Any] = {
        "params_reduction_pct": params,
        "gflops_reduction_pct": flops,
        "params_before": referencia.params_total,
        "params_after": comprimido.params_total,
        "gflops_before": referencia.gflops,
        "gflops_after": comprimido.gflops,
        "structural": params > 0.5 and flops > 0.5,
        "notes": "",
    }
    if not resultado["structural"]:
        resultado["notes"] = (
            "Ni los parámetros ni las operaciones bajan de forma apreciable: la "
            "compresión NO es estructural. Si el fichero ocupa menos, será por la "
            "precisión de los pesos, no por haber eliminado cómputo, y la latencia "
            "no tiene por qué mejorar."
        )
        logger.warning("%s", resultado["notes"])
    return resultado
