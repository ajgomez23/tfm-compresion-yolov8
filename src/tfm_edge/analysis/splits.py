"""Partición sobre la que se hace el análisis: solo val.

Test se evaluó para contrastar las conclusiones, no para decidir con él. Sus filas comparten
experimento, motor, precisión y dispositivo con las de val, así que se filtran aquí.
"""

from __future__ import annotations

from typing import Iterable, Mapping, TypeVar

from tfm_edge.common.registry import ExperimentRegistry

__all__ = ["ANALYSIS_SPLIT", "select_analysis_split", "analysis_quality_rows"]

ANALYSIS_SPLIT = "val"

_Fila = TypeVar("_Fila", bound=Mapping[str, str])


def select_analysis_split(rows: Iterable[_Fila]) -> list[_Fila]:
    """Se queda con las filas de calidad de la partición de análisis."""
    return [fila for fila in rows if fila.get("split", "") == ANALYSIS_SPLIT]


def analysis_quality_rows(registry: ExperimentRegistry) -> tuple[list[dict[str, str]], int]:
    """Como ``valid_rows("quality_metrics")`` pero solo con val. Devuelve las filas y cuántas se
    descartaron.
    """
    todas = select_analysis_split(registry.read_rows("quality_metrics"))
    validas = [fila for fila in todas if fila.get("status", "") == "ok"]
    return validas, len(todas) - len(validas)
