"""Consultas de lectura sobre el registro que fallan ante un empate en vez de elegir.

Un mismo experimento puede tener varias filas (dos evaluadores, dos dispositivos). El evaluador
filtra, el dispositivo desempata, las filas idénticas cuentan como una y si aun así quedan valores
distintos se lanza un error. Lo usan 13_prune.py, 16_qat.py y 20_distill.py.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from tfm_edge.common.exceptions import RegistryAmbiguousRowError
from tfm_edge.common.registry import (
    TABLE_PATH_COLUMNS,
    TABLE_PRIMARY_KEYS,
    ExperimentRegistry,
    normalize_value,
)

__all__ = ["preserve_registered_paths", "read_reference_quality"]

logger = logging.getLogger(__name__)


def preserve_registered_paths(
    registry: ExperimentRegistry, row: Mapping[str, Any], table: str = "experiments"
) -> dict[str, Any]:
    """Restaura en una fila candidata las rutas que ya estaban inscritas en el registro.

    Los pasos 05 y 06 reinscriben la fila del experimento para completarla, pero no deben cambiar
    config_path ni weights_path por la ruta local del checkpoint. Reapuntar rutas es trabajo de
    14_import_colab_registry.py.
    """
    columnas = TABLE_PATH_COLUMNS.get(table, ())
    if not columnas:
        return dict(row)

    clave = TABLE_PRIMARY_KEYS[table]
    buscada = tuple(normalize_value(row.get(c)) for c in clave)
    previa = next(
        (
            f
            for f in registry.read_rows(table)
            if tuple(normalize_value(f.get(c)) for c in clave) == buscada
        ),
        None,
    )
    if previa is None:
        return dict(row)

    resultado = dict(row)
    for columna in columnas:
        inscrita = (previa.get(columna) or "").strip()
        if not inscrita:
            # Una celda vacía sí se puede rellenar.
            continue
        candidata = normalize_value(resultado.get(columna))
        if candidata and candidata != inscrita:
            logger.warning(
                "Se conserva '%s' = %r de la fila inscrita de '%s' en lugar de %r.",
                columna,
                inscrita,
                buscada[0],
                candidata,
            )
        resultado[columna] = inscrita
    return resultado


def read_reference_quality(
    registry: ExperimentRegistry,
    experiment_id: str,
    split: str,
    *,
    evaluator: str,
    runtime: str = "pytorch",
    preferred_device_id: str | None = None,
) -> dict[str, Any]:
    """Lee la métrica de calidad de referencia de un experimento.

    Solo cuentan las filas con map_50_95 medido. El evaluador es obligatorio y el dispositivo solo
    desempata. Devuelve un diccionario vacío si no hay referencia y lanza RegistryAmbiguousRowError
    si quedan filas contradictorias.
    """
    candidatas = [
        fila
        for fila in registry.read_rows("quality_metrics")
        if fila.get("experiment_id") == experiment_id
        and fila.get("runtime") == runtime
        and fila.get("split") == split
        and fila.get("map_50_95")
        and fila.get("evaluator") == evaluator
    ]
    if not candidatas:
        return {}

    if len(candidatas) > 1 and preferred_device_id:
        del_dispositivo = [f for f in candidatas if f.get("device_id") == preferred_device_id]
        if del_dispositivo:
            candidatas = del_dispositivo

    # Se comparan las dos métricas: coincidir solo en mAP@50-95 no basta.
    distintas = {(fila.get("map_50", ""), fila["map_50_95"]) for fila in candidatas}
    if len(distintas) > 1:
        detalle = "\n".join(
            f"  - device_id={fila.get('device_id', '')!r}, "
            f"map_50={fila.get('map_50', '')}, map_50_95={fila['map_50_95']}, "
            f"notes={(fila.get('notes') or '')[:60]!r}"
            for fila in candidatas
        )
        raise RegistryAmbiguousRowError(
            f"experiment_id={experiment_id!r}, runtime={runtime!r}, split={split!r}, "
            f"evaluator={evaluator!r}",
            detalle,
        )

    elegida = candidatas[0]
    referencia: dict[str, Any] = {"map_50_95": float(elegida["map_50_95"])}
    # Un mAP@50 sin medir se omite, igual que la celda vacía del registro.
    if elegida.get("map_50"):
        referencia["map_50"] = float(elegida["map_50"])
    referencia["device_id"] = elegida.get("device_id", "")
    referencia["evaluator"] = elegida.get("evaluator", "")
    return referencia
