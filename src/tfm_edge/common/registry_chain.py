"""Encadena el registro de una campaña de Colab entre sesiones.

Cada sesión adopta la copia de Drive solo si el registro local es un prefijo exacto de ella, tabla a
tabla. Se compara por filas, no por bytes, y si una tabla no cumple no se copia ninguna.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from tfm_edge.common.exceptions import RegistryChainError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

__all__ = ["adopt_campaign_registry"]


def _leer_filas(ruta: Path) -> list[list[str]]:
    """Lee un CSV como lista de filas, cabecera incluida."""
    with ruta.open(encoding="utf-8", newline="") as manejador:
        return list(csv.reader(manejador))


def adopt_campaign_registry(local_dir: Path, campaign_dir: Path) -> list[str]:
    """Adopta la copia de campaña si solo añade filas al registro local.

    Devuelve una línea por tabla adoptada y lanza RegistryChainError sin copiar nada si alguna no es
    prefijo.
    """
    copias = sorted(campaign_dir.glob("*.csv")) if campaign_dir.is_dir() else []
    if not copias:
        return [f"No hay copia de campaña en '{campaign_dir}': primera sesión, nada que adoptar."]

    plan: list[tuple[Path, Path, int, int]] = []
    for copia in copias:
        local = local_dir / copia.name
        de_campana = _leer_filas(copia)
        propias = _leer_filas(local) if local.is_file() else []
        if propias and not de_campana:
            raise RegistryChainError(
                f"La copia de campaña '{copia}' está vacía y la local '{local}' no.",
                "Una copia vacía no continúa a nada. Averigua qué la vació antes de "
                "lanzar: adoptarla borraría el registro de esta sesión.",
            )
        if propias and propias[0] != de_campana[0]:
            raise RegistryChainError(
                f"'{copia.name}': la cabecera de la copia de campaña no es la del "
                f"registro local.",
                "El esquema del registro ha cambiado entre sesiones. Importa la copia de "
                "campaña en la estación de trabajo y vuelve a clonar.",
            )
        comun = min(len(propias), len(de_campana))
        for indice in range(comun):
            if propias[indice] != de_campana[indice]:
                raise RegistryChainError(
                    f"'{copia.name}': la fila {indice} del registro local no coincide con la "
                    "copia de campaña.",
                    "No se adopta: copiarla encima perdería lo que la local dice y la "
                    "otra no. Compara las dos filas a mano antes de lanzar nada.",
                )
        if len(propias) > len(de_campana):
            raise RegistryChainError(
                f"'{copia.name}': el registro local tiene {len(propias) - 1} filas y la "
                f"copia de campaña {max(len(de_campana) - 1, 0)}.",
                "El clon trae filas que la copia de campaña no tiene. Importa antes la copia de "
                "campaña en la estación de trabajo.",
            )
        plan.append((copia, local, len(propias), len(de_campana)))

    local_dir.mkdir(parents=True, exist_ok=True)
    lineas = []
    for copia, local, antes, despues in plan:
        shutil.copy2(copia, local)
        nuevas = despues - antes if antes else max(despues - 1, 0)
        lineas.append(f"{copia.name}: adoptada ({nuevas} filas de sesiones anteriores).")
        logger.info("Registro de campaña adoptado: %s", lineas[-1])
    return lineas
