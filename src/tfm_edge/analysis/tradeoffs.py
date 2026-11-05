"""Curvas de compromiso por familia de técnica.

Cada curva indica su forma (punto, segmento o curva) según cuántos niveles la sostienen, porque una
recta entre dos puntos parece una tendencia. El nivel sale de compression_ratio y se contrasta con
el que lleva el nombre de la variante. Se agrupa por método, etapa (sin reajustar o reajustado) y
tamaño de modelo. El baseline se usa como nivel cero solo si tiene la misma precisión.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tfm_edge.analysis.degradation import (
    STAGE_FINETUNED,
    STAGE_RAW,
    resolve_reference,
    training_stage,
)
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import parse_experiment_id

logger = get_logger(__name__)

#: Nivel dentro del nombre de una variante: un token ``rNN``. ``l1global_r50_iter4`` da nivel 50 y
#: método ``l1global_iter4``.
LEVEL_TOKEN = re.compile(r"[-_]r(\d{1,3})(?=[-_]|$)")

#: Tolerancia de redondeo entre el nivel del nombre y compression_ratio.
LEVEL_TOLERANCE = 1e-6

SHAPE_BY_POINTS: dict[int, str] = {0: "vacia", 1: "punto", 2: "segmento"}

#: Las etapas y la función que las decide están en ``degradation``; se reexportan aquí.


class TradeoffError(TFMEdgeError):
    """Error al construir las curvas de compromiso."""


@dataclass(frozen=True)
class CurvePoint:
    """Punto de una curva de compromiso."""

    experiment_id: str
    level: float
    values: Mapping[str, float]
    is_anchor: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serializa el punto."""
        return {
            "experiment_id": self.experiment_id,
            "level": self.level,
            "values": {k: v for k, v in sorted(self.values.items())},
            "is_anchor": self.is_anchor,
        }


@dataclass(frozen=True)
class FamilyCurve:
    """Comportamiento de un método a lo largo de su rango de agresividad."""

    technique: str
    model_size: str
    method: str
    stage: str
    runtime: str
    precision: str
    device_id: str
    points: tuple[CurvePoint, ...]
    anchor_note: str = ""

    @property
    def levels(self) -> tuple[float, ...]:
        """Niveles distintos que sostienen la curva."""
        return tuple(sorted({p.level for p in self.points}))

    @property
    def shape(self) -> str:
        """Forma de la curva según sus niveles distintos: punto, segmento o curva."""
        return SHAPE_BY_POINTS.get(len(self.levels), "curva")

    @property
    def label(self) -> str:
        """Etiqueta única con método, etapa, tamaño, motor y precisión."""
        etapa = "" if self.stage == STAGE_FINETUNED else f" {self.stage}"
        return f"{self.method}{etapa} ({self.model_size}) [{self.runtime}/{self.precision}]"

    def to_dict(self) -> dict[str, Any]:
        """Serializa la curva."""
        return {
            "technique": self.technique,
            "model_size": self.model_size,
            "method": self.method,
            "stage": self.stage,
            "runtime": self.runtime,
            "precision": self.precision,
            "device_id": self.device_id,
            "shape": self.shape,
            "num_levels": len(self.levels),
            "levels": list(self.levels),
            "points": [p.to_dict() for p in self.points],
            "anchor_note": self.anchor_note,
        }


def split_variant(variant: str) -> tuple[str, float | None]:
    """Separa el método del nivel de agresividad de una variante. El nivel es None si el nombre no
    lo lleva.
    """
    encontrados = LEVEL_TOKEN.findall(variant)
    if len(encontrados) > 1:
        raise TradeoffError(
            f"La variante '{variant}' lleva {len(encontrados)} niveles en el nombre "
            f"({encontrados})."
        )
    if not encontrados:
        return variant, None
    metodo = LEVEL_TOKEN.sub("", variant, count=1)
    # El token se lleva su separador izquierdo; si iba en medio hay que recomponer el nombre.
    metodo = re.sub(r"[-_]{2,}", "_", metodo).strip("-_")
    return metodo, int(encontrados[0]) / 100.0


def _declared_level(row: Mapping[str, str]) -> float | None:
    """Nivel declarado en compression_ratio, o None si está vacío."""
    bruto = (row.get("compression_ratio", "") or "").strip()
    if not bruto:
        return None
    try:
        return float(bruto)
    except ValueError as error:
        raise TradeoffError(
            f"El experimento {row.get('experiment_id', '?')} declara "
            f"compression_ratio='{bruto}', que no es un número."
        ) from error


def level_of(row: Mapping[str, str]) -> tuple[str, float | None]:
    """Método y nivel de un experimento. Falla si el nombre y compression_ratio no coinciden."""
    variante = (row.get("technique_variant", "") or "").strip()
    metodo, del_nombre = split_variant(variante)
    declarado = _declared_level(row)

    if del_nombre is not None and declarado is not None:
        if abs(del_nombre - declarado) > LEVEL_TOLERANCE:
            raise TradeoffError(
                f"El experimento {row.get('experiment_id', '?')} se llama '{variante}' (nivel "
                f"{del_nombre}) pero declara compression_ratio={declarado}."
            )
    nivel = declarado if declarado is not None else del_nombre
    return metodo, nivel


def _values_of(
    quality_row: Mapping[str, str] | None,
    deployment_row: Mapping[str, str] | None,
    columns: Sequence[str],
) -> dict[str, float]:
    """Valores de las columnas pedidas presentes en las dos filas."""
    valores: dict[str, float] = {}
    for fila in (quality_row, deployment_row):
        if fila is None:
            continue
        for columna in columns:
            bruto = (fila.get(columna, "") or "").strip()
            if bruto and columna not in valores:
                try:
                    valores[columna] = float(bruto)
                except ValueError:
                    logger.warning(
                        "Valor no numérico en '%s' de %s: '%s'. No entra en la curva.",
                        columna,
                        fila.get("experiment_id", "?"),
                        bruto,
                    )
    return valores


def family_curves(
    experiment_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
    deployment_rows: Sequence[Mapping[str, str]],
    columns: Sequence[str] = ("map_50", "map_50_95", "latency_p95_ms", "peak_host_memory_mb"),
) -> tuple[FamilyCurve, ...]:
    """Construye una curva por método, etapa, tamaño, motor y precisión.

    Lanza TradeoffError si dos experimentos de la misma curva comparten nivel.
    """
    experimentos = {fila["experiment_id"]: fila for fila in experiment_rows}
    calidad = {(f["experiment_id"], f["runtime"]): f for f in quality_rows}
    despliegue = {(f["experiment_id"], f["runtime"], f["precision"]): f for f in deployment_rows}

    # Se agrupa por técnica, tamaño, método, etapa, motor, precisión y dispositivo.
    grupos: dict[tuple[str, str, str, str, str, str], list[CurvePoint]] = {}
    anclas: dict[tuple[str, str, str, str, str, str], str] = {}

    for fila_calidad in quality_rows:
        experiment_id = fila_calidad["experiment_id"]
        experimento = experimentos.get(experiment_id)
        if experimento is None:
            continue
        tecnica = (experimento.get("technique", "") or "").strip()
        if tecnica == "baseline":
            continue
        metodo, nivel = level_of(experimento)
        if nivel is None:
            logger.info(
                "%s no declara nivel de compresión: su familia no tiene rango de "
                "agresividad medido y no forma curva.",
                experiment_id,
            )
            continue

        etapa = training_stage(experiment_id, experimentos)
        motor = fila_calidad["runtime"]
        precision = fila_calidad["precision"]
        fila_despliegue = despliegue.get((experiment_id, motor, precision))
        dispositivo = fila_despliegue["device_id"] if fila_despliegue is not None else ""
        # El tamaño del modelo forma parte de la clave para no mezclar, por ejemplo, la PTQ de n y
        # la de s.
        tamano = parse_experiment_id(experiment_id)["model_size"]
        clave = (tecnica, tamano, metodo, etapa, motor, precision, dispositivo)

        punto = CurvePoint(
            experiment_id=experiment_id,
            level=nivel,
            values=_values_of(fila_calidad, fila_despliegue, columns),
        )
        grupos.setdefault(clave, []).append(punto)

        if clave not in anclas:
            raiz, _ = resolve_reference(experiment_id, experimentos)
            fila_ancla = calidad.get((raiz, motor))
            if fila_ancla is None:
                anclas[clave] = (
                    f"Sin ancla: la referencia {raiz} no está medida en {motor}, así que "
                    f"la curva no tiene nivel cero."
                )
            elif fila_ancla["precision"] != precision:
                anclas[clave] = (
                    f"Sin ancla: la referencia {raiz} está medida en {motor} a "
                    f"{fila_ancla['precision']} y esta curva es de {precision}. Un modelo "
                    f"{fila_ancla['precision']} no es el nivel cero de una curva "
                    f"{precision}: entre los dos hay otra técnica."
                )
            elif etapa == STAGE_RAW:
                anclas[clave] = (
                    f"Sin ancla: esta curva es de compresión sin reajustar y {raiz} sí "
                    f"está entrenado. El nivel cero de una curva sin reajuste sería un "
                    f"modelo sin entrenar, que no existe en la campaña."
                )
            else:
                ancla_despliegue = despliegue.get((raiz, motor, precision))
                grupos[clave].append(
                    CurvePoint(
                        experiment_id=raiz,
                        level=0.0,
                        values=_values_of(fila_ancla, ancla_despliegue, columns),
                        is_anchor=True,
                    )
                )
                anclas[clave] = f"Ancla en el nivel cero: {raiz} en {motor}/{precision}."

    curvas: list[FamilyCurve] = []
    for clave, puntos in sorted(grupos.items()):
        tecnica, tamano, metodo, etapa, motor, precision, dispositivo = clave
        por_nivel: dict[float, str] = {}
        for punto in puntos:
            if punto.level in por_nivel and por_nivel[punto.level] != punto.experiment_id:
                raise TradeoffError(
                    f"El método '{metodo}' ({etapa}) sobre '{tamano}' en {motor}/{precision} "
                    f"tiene dos experimentos en el nivel {punto.level}: {por_nivel[punto.level]} "
                    f"y {punto.experiment_id}."
                )
            por_nivel[punto.level] = punto.experiment_id

        curva = FamilyCurve(
            technique=tecnica,
            model_size=tamano,
            method=metodo,
            stage=etapa,
            runtime=motor,
            precision=precision,
            device_id=dispositivo,
            points=tuple(sorted(puntos, key=lambda p: (p.level, p.experiment_id))),
            anchor_note=anclas.get(clave, ""),
        )
        curvas.append(curva)

    formas: dict[str, int] = {}
    for curva in curvas:
        formas[curva.shape] = formas.get(curva.shape, 0) + 1
    logger.info(
        "Curvas de compromiso: %d en total (%s).",
        len(curvas),
        ", ".join(f"{n} de forma '{f}'" for f, n in sorted(formas.items())) or "ninguna",
    )
    return tuple(curvas)
