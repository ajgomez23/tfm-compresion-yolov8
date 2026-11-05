"""Semillas y determinismo.

PyTorch se importa dentro de las funciones para que el módulo funcione sin él; lo que no se puede
sembrar queda anotado en el informe. Límites: PYTHONHASHSEED solo afecta a procesos hijo, algunos
kernels de CUDA no tienen variante determinista (por eso warn_only), el modo determinista baja el
rendimiento y no se usa al medir latencia, y el orden con varios workers no se fija del todo.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any

from tfm_edge.common.exceptions import DependencyNotAvailableError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Necesario para que cuBLAS sea determinista; sin él PyTorch falla en modo estricto.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass
class SeedReport:
    """Qué se ha sembrado y qué no. Se guarda junto a los resultados del experimento."""

    seed: int
    deterministic: bool
    seeded: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "seeded": list(self.seeded),
            "skipped": dict(self.skipped),
            "warnings": list(self.warnings),
        }


def set_global_seed(seed: int, deterministic: bool = True) -> SeedReport:
    """Siembra random, NumPy, PyTorch y CUDA y fija PYTHONHASHSEED.

    Con ``deterministic=True`` activa además los algoritmos deterministas; no debe usarse al medir
    latencia.
    """
    if seed < 0:
        raise ValueError(f"La semilla debe ser un entero no negativo, y se recibió {seed}.")

    informe = SeedReport(seed=seed, deterministic=deterministic)

    os.environ["PYTHONHASHSEED"] = str(seed)
    informe.seeded.append("PYTHONHASHSEED (solo afecta a procesos hijo)")
    informe.warnings.append(
        "PYTHONHASHSEED se ha fijado dentro del proceso, por lo que no altera la "
        "aleatorización de hash del proceso actual; solo la de los procesos hijo."
    )

    random.seed(seed)
    informe.seeded.append("random")

    try:
        import numpy as np

        np.random.seed(seed)
        informe.seeded.append("numpy.random")
    except ImportError:
        informe.skipped["numpy"] = "NumPy no está instalado"

    try:
        import torch
    except ImportError:
        informe.skipped["torch"] = "PyTorch no está instalado"
        informe.skipped["torch.cuda"] = "PyTorch no está instalado"
        logger.warning("PyTorch no está instalado: no se han sembrado torch ni CUDA.")
        _log_informe(informe)
        return informe

    torch.manual_seed(seed)
    informe.seeded.append("torch")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        informe.seeded.append("torch.cuda (todas las GPU)")
    else:
        informe.skipped["torch.cuda"] = "No hay ninguna GPU CUDA disponible"

    if deterministic:
        _activar_determinismo(torch, informe)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        informe.warnings.append(
            "Modo NO determinista: cudnn.benchmark está activo. Es lo correcto "
            "para medir latencia, y lo incorrecto para reproducir un entrenamiento."
        )

    _log_informe(informe)
    return informe


def _activar_determinismo(torch: Any, informe: SeedReport) -> None:
    """Activa las banderas de determinismo de PyTorch y cuDNN."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    informe.seeded.append("cudnn.deterministic=True, cudnn.benchmark=False")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)

    try:
        # warn_only=True: unas pocas operaciones no tienen versión determinista y abortarían el
        # entrenamiento.
        torch.use_deterministic_algorithms(True, warn_only=True)
        informe.seeded.append("torch.use_deterministic_algorithms(True, warn_only=True)")
        informe.warnings.append(
            "Se ha pedido determinismo con warn_only=True: las operaciones sin "
            "implementación determinista emitirán un aviso y seguirán siendo no "
            "deterministas. Revisa el log para saber cuáles han aparecido."
        )
    except Exception as error:  # pragma: no cover - depende de la versión de torch
        informe.skipped["use_deterministic_algorithms"] = str(error)
        logger.warning("No se ha podido activar el modo determinista de PyTorch: %s", error)

    informe.warnings.append(
        "El modo determinista desactiva la búsqueda de algoritmos de cuDNN y "
        "reduce el rendimiento. NO debe usarse en campañas de medida de latencia."
    )


def _log_informe(informe: SeedReport) -> None:
    """Vuelca el informe de siembra al log."""
    logger.info("Semilla global fijada a %d (determinista=%s)", informe.seed, informe.deterministic)
    for componente in informe.seeded:
        logger.debug("  sembrado: %s", componente)
    for componente, motivo in informe.skipped.items():
        logger.info("  NO sembrado: %s (%s)", componente, motivo)


def get_dataloader_generator(seed: int) -> Any:
    """Generador de PyTorch sembrado para el DataLoader."""
    try:
        import torch
    except ImportError as error:
        raise DependencyNotAvailableError(
            "torch",
            "construir un generador reproducible para el DataLoader",
        ) from error

    generador = torch.Generator()
    generador.manual_seed(seed)
    return generador


def derive_seed(base_seed: int, *labels: str) -> int:
    """Deriva una semilla estable a partir de una base y unas etiquetas.

    Cada rama de una campaña recibe su semilla sin depender del orden de ejecución. Se usa SHA-256
    porque ``hash`` cambia con PYTHONHASHSEED.
    """
    import hashlib

    material = "|".join([str(base_seed), *labels])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**31)


def seed_worker(worker_id: int) -> None:
    """Siembra un worker del DataLoader. Se pasa como ``worker_init_fn``."""
    import torch

    semilla_worker = torch.initial_seed() % 2**32
    random.seed(semilla_worker)
    try:
        import numpy as np

        np.random.seed(semilla_worker)
    except ImportError:  # pragma: no cover - NumPy es dependencia dura del entrenamiento
        logger.warning("NumPy no disponible en el worker %d: no se ha sembrado.", worker_id)
