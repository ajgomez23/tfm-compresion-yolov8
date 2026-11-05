"""Consolidación del registro y cobertura de la campaña.

Lo esperado se deduce por familia de técnica: si un experimento de la familia se midió en un
entorno, se espera que todos los de la familia estén ahí. Un entorno donde no se midió nada no
genera huecos. La calidad se agrupa por (motor, precisión, evaluador) y el despliegue por (motor,
precisión, dispositivo), porque el mAP no depende de la máquina; device_independence lo comprueba en
cada ejecución. También se cuentan filas huérfanas, experimentos sin métricas y claves duplicadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.common.exceptions import InvalidExperimentIdError, TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import parse_experiment_id

logger = get_logger(__name__)

#: Ejes de un entorno de calidad, sin dispositivo.
QUALITY_CONTEXT_KEYS: tuple[str, ...] = ("runtime", "precision", "evaluator")

#: Ejes de un entorno de despliegue, con dispositivo.
DEPLOYMENT_CONTEXT_KEYS: tuple[str, ...] = ("runtime", "precision", "device_id")

CONTEXT_KEYS: dict[str, tuple[str, ...]] = {
    "quality": QUALITY_CONTEXT_KEYS,
    "deployment": DEPLOYMENT_CONTEXT_KEYS,
}

#: Claves para detectar duplicados. Incluyen el dispositivo también en calidad: evaluar en dos
#: máquinas no es un duplicado.
ROW_KEYS: dict[str, tuple[str, ...]] = {
    "quality": ("experiment_id", "runtime", "precision", "evaluator", "device_id"),
    "deployment": ("experiment_id", "runtime", "precision", "device_id"),
}

#: Columnas de calidad con las que se comprueba la independencia del dispositivo.
DEVICE_INDEPENDENCE_COLUMNS: tuple[str, ...] = ("map_50", "map_50_95")

METRIC_TABLES: tuple[str, ...] = ("quality", "deployment")

#: Familia de un experimento con identificador ilegible; no se deja vacía.
UNKNOWN_TECHNIQUE = "desconocida"


class AggregateError(TFMEdgeError):
    """Error al consolidar el registro."""


@dataclass(frozen=True)
class Context:
    """Ejes de un entorno de medida, sin el experimento."""

    table: str
    values: tuple[str, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """Ejes que definen este entorno."""
        return CONTEXT_KEYS[self.table]

    @property
    def label(self) -> str:
        """Etiqueta con los valores separados por barras."""
        return "/".join(self.values)

    def as_dict(self) -> dict[str, str]:
        """El entorno como diccionario de campo a valor."""
        return dict(zip(self.keys, self.values))


@dataclass(frozen=True)
class Coverage:
    """Indica si un experimento tiene métrica en un entorno esperado."""

    table: str
    experiment_id: str
    technique: str
    context: Context
    present: bool

    def to_dict(self) -> dict[str, Any]:
        """Serializa la cobertura."""
        return {
            "table": self.table,
            "experiment_id": self.experiment_id,
            "technique": self.technique,
            "context": self.context.as_dict(),
            "present": self.present,
        }


@dataclass(frozen=True)
class Inconsistency:
    """Incoherencia del registro, distinta de un hueco de la campaña."""

    kind: str
    detail: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        """Serializa la incoherencia."""
        return {"kind": self.kind, "detail": self.detail, "explanation": self.explanation}


@dataclass(frozen=True)
class DeviceSpread:
    """Diferencia de una métrica de calidad entre dispositivos."""

    experiment_id: str
    context: Context
    devices: tuple[str, ...]
    spread: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serializa la observación."""
        return {
            "experiment_id": self.experiment_id,
            "context": self.context.as_dict(),
            "devices": list(self.devices),
            "spread": {k: v for k, v in sorted(self.spread.items())},
        }


@dataclass(frozen=True)
class CoverageReport:
    """Informe de cobertura: celdas esperadas, huecos, incoherencias y dependencia del
    dispositivo."""

    rule: str
    environments: Mapping[str, Mapping[str, tuple[Context, ...]]]
    cells: tuple[Coverage, ...]
    inconsistencies: tuple[Inconsistency, ...]
    device_spreads: tuple[DeviceSpread, ...]
    discarded: Mapping[str, Any]

    @property
    def gaps(self) -> tuple[Coverage, ...]:
        """Celdas esperadas que no están."""
        return tuple(c for c in self.cells if not c.present)

    @property
    def device_dependence_detected(self) -> bool:
        """True si alguna métrica de calidad cambia con el dispositivo."""
        return any(valor != 0.0 for obs in self.device_spreads for valor in obs.spread.values())

    def gaps_by_experiment(self) -> dict[str, tuple[Coverage, ...]]:
        """Agrupa los huecos por experimento."""
        agrupado: dict[str, list[Coverage]] = {}
        for celda in self.gaps:
            agrupado.setdefault(celda.experiment_id, []).append(celda)
        return {k: tuple(agrupado[k]) for k in sorted(agrupado)}

    def to_dict(self) -> dict[str, Any]:
        """Serializa el informe entero."""
        return {
            "rule": self.rule,
            "environments": {
                tabla: {
                    familia: [contexto.label for contexto in contextos]
                    for familia, contextos in sorted(por_familia.items())
                }
                for tabla, por_familia in sorted(self.environments.items())
            },
            "counts": {
                "cells_expected": len(self.cells),
                "cells_present": len(self.cells) - len(self.gaps),
                "cells_missing": len(self.gaps),
                "experiments_with_gaps": len(self.gaps_by_experiment()),
                "inconsistencies": len(self.inconsistencies),
            },
            "gaps": [celda.to_dict() for celda in self.gaps],
            "inconsistencies": [i.to_dict() for i in self.inconsistencies],
            "device_independence": {
                "checked": len(self.device_spreads),
                "dependence_detected": self.device_dependence_detected,
                "observations": [o.to_dict() for o in self.device_spreads],
            },
            "discarded": dict(sorted(self.discarded.items())),
        }


def _technique_of(experiment_id: str, experiments: Mapping[str, Mapping[str, str]]) -> str:
    """Familia de técnica de un experimento, del registro o deducida del identificador si es
    huérfano.
    """
    fila = experiments.get(experiment_id)
    if fila is not None and (fila.get("technique", "") or "").strip():
        return fila["technique"].strip()
    try:
        return parse_experiment_id(experiment_id)["technique"]
    except InvalidExperimentIdError:
        # Un identificador ilegible se señala en el informe en lugar de lanzar la excepción.
        return UNKNOWN_TECHNIQUE


def context_of(row: Mapping[str, str], table: str) -> Context:
    """Entorno de una fila de métricas. Falla si le falta algún eje."""
    if table not in CONTEXT_KEYS:
        raise AggregateError(f"Tabla desconocida: '{table}'. Solo hay {METRIC_TABLES}.")
    claves = CONTEXT_KEYS[table]
    faltan = [k for k in claves if not (row.get(k, "") or "").strip()]
    if faltan:
        raise AggregateError(
            f"La fila de {row.get('experiment_id', '(sin experimento)')} en '{table}' no declara "
            f"{faltan}."
        )
    return Context(table=table, values=tuple(row[k].strip() for k in claves))


def observed_environments(
    rows: Iterable[Mapping[str, str]],
    table: str,
    experiments: Mapping[str, Mapping[str, str]],
) -> dict[str, tuple[Context, ...]]:
    """Entornos en los que se midió cada familia de técnica."""
    por_familia: dict[str, set[Context]] = {}
    for fila in rows:
        familia = _technique_of(fila["experiment_id"], experiments)
        por_familia.setdefault(familia, set()).add(context_of(fila, table))
    return {
        familia: tuple(sorted(contextos, key=lambda c: c.label))
        for familia, contextos in sorted(por_familia.items())
    }


def device_independence(
    quality_rows: Sequence[Mapping[str, str]],
    columns: Sequence[str] = DEVICE_INDEPENDENCE_COLUMNS,
) -> tuple[DeviceSpread, ...]:
    """Comprueba que el mAP no cambia entre dispositivos para filas con el mismo experimento, motor,
    precisión y evaluador.
    """
    agrupado: dict[tuple[str, tuple[str, ...]], list[Mapping[str, str]]] = {}
    for fila in quality_rows:
        contexto = context_of(fila, "quality")
        agrupado.setdefault((fila["experiment_id"], contexto.values), []).append(fila)

    observaciones: list[DeviceSpread] = []
    for (experiment_id, valores), filas in sorted(agrupado.items()):
        dispositivos = sorted({(f.get("device_id", "") or "").strip() for f in filas})
        if len(dispositivos) < 2:
            continue
        diferencias: dict[str, float] = {}
        for columna in columns:
            numeros = []
            for fila in filas:
                bruto = (fila.get(columna, "") or "").strip()
                if bruto:
                    numeros.append(float(bruto))
            if len(numeros) >= 2:
                diferencias[columna] = max(numeros) - min(numeros)
        observaciones.append(
            DeviceSpread(
                experiment_id=experiment_id,
                context=Context(table="quality", values=valores),
                devices=tuple(dispositivos),
                spread=diferencias,
            )
        )
    return tuple(observaciones)


def _index_rows(
    rows: Iterable[Mapping[str, str]], keys: Sequence[str]
) -> dict[tuple[str, ...], list[Mapping[str, str]]]:
    """Indexa filas por unas claves."""
    indice: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for fila in rows:
        clave = tuple((fila.get(k, "") or "").strip() for k in keys)
        indice.setdefault(clave, []).append(fila)
    return indice


def _find_inconsistencies(
    experiments: Mapping[str, Mapping[str, str]],
    rows_by_table: Mapping[str, Sequence[Mapping[str, str]]],
) -> tuple[Inconsistency, ...]:
    """Incoherencias del registro, ordenadas por tipo y detalle."""
    encontradas: list[Inconsistency] = []

    for tabla in METRIC_TABLES:
        filas = rows_by_table[tabla]
        huerfanos = sorted(
            {f["experiment_id"] for f in filas if f["experiment_id"] not in experiments}
        )
        for experimento in huerfanos:
            encontradas.append(
                Inconsistency(
                    kind="metrica_sin_experimento",
                    detail=f"{tabla}: {experimento}",
                    explanation=(
                        "Hay métricas de un experimento que no está en experiments.csv. "
                        "O falta la fila del experimento, o la métrica se registró con un "
                        "identificador equivocado; en los dos casos el registro no puede "
                        "decir de qué configuración son esos números."
                    ),
                )
            )

        for clave, repetidas in sorted(_index_rows(filas, ROW_KEYS[tabla]).items()):
            if len(repetidas) > 1:
                encontradas.append(
                    Inconsistency(
                        kind="fila_duplicada",
                        detail=f"{tabla}: {'/'.join(clave)} ({len(repetidas)} filas)",
                        explanation=(
                            "La misma fila aparece más de una vez. El análisis de Pareto "
                            "se niega a elegir una de ellas, así que esta configuración "
                            "no llega a ser un punto."
                        ),
                    )
                )

    ilegibles: set[str] = set()
    for tabla in METRIC_TABLES:
        for fila in rows_by_table[tabla]:
            experiment_id = fila["experiment_id"]
            if experiment_id in experiments or experiment_id in ilegibles:
                continue
            try:
                parse_experiment_id(experiment_id)
            except InvalidExperimentIdError as error:
                ilegibles.add(experiment_id)
                encontradas.append(
                    Inconsistency(
                        kind="identificador_ilegible",
                        detail=f"{tabla}: {experiment_id}",
                        explanation=(
                            f"El identificador no sigue la convención de la sección 16.2 "
                            f"({error}). Ni siquiera se puede deducir de qué técnica es "
                            f"esa fila."
                        ),
                    )
                )

    con_metricas = {f["experiment_id"] for tabla in METRIC_TABLES for f in rows_by_table[tabla]}
    for experimento in sorted(set(experiments) - con_metricas):
        estado = (experiments[experimento].get("status", "") or "").strip()
        encontradas.append(
            Inconsistency(
                kind="experimento_sin_metricas",
                detail=f"{experimento} (status={estado or 'sin declarar'})",
                explanation=(
                    "El experimento está registrado y no tiene ninguna métrica válida. "
                    "Si su status es 'fallido' es coherente; si es 'ok', se generó algo "
                    "que nunca se llegó a evaluar."
                ),
            )
        )

    return tuple(sorted(encontradas, key=lambda i: (i.kind, i.detail)))


def coverage_report(
    experiment_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
    deployment_rows: Sequence[Mapping[str, str]],
    discarded: Mapping[str, Any] | None = None,
) -> CoverageReport:
    """Consolida el registro y señala las combinaciones ausentes."""
    experimentos = {fila["experiment_id"]: fila for fila in experiment_rows}
    por_tabla: dict[str, Sequence[Mapping[str, str]]] = {
        "quality": quality_rows,
        "deployment": deployment_rows,
    }

    entornos = {
        tabla: observed_environments(por_tabla[tabla], tabla, experimentos)
        for tabla in METRIC_TABLES
    }
    presentes = {
        tabla: {(f["experiment_id"], context_of(f, tabla).values) for f in por_tabla[tabla]}
        for tabla in METRIC_TABLES
    }

    celdas: list[Coverage] = []
    for experiment_id in sorted(experimentos):
        familia = _technique_of(experiment_id, experimentos)
        for tabla in METRIC_TABLES:
            for contexto in entornos[tabla].get(familia, ()):
                celdas.append(
                    Coverage(
                        table=tabla,
                        experiment_id=experiment_id,
                        technique=familia,
                        context=contexto,
                        present=(experiment_id, contexto.values) in presentes[tabla],
                    )
                )

    informe = CoverageReport(
        rule=(
            "Se espera de cada experimento lo que alcanzó algún experimento de su misma "
            "familia de técnica: si alguna variante de poda se midió en onnxruntime/fp32, "
            "se esperan ahí todas las de poda. La regla se deduce de la campaña y no de "
            "una lista escrita a mano, así que no puede inventarse un entorno que nadie "
            "alcanzó. Los ejes del entorno no son los mismos en las dos tablas: la "
            "calidad se agrupa por motor, precisión y evaluador, y el despliegue por "
            "motor, precisión y dispositivo, porque la latencia es una propiedad de la "
            "máquina y el mAP no —lo que se comprueba en 'device_independence' en lugar "
            "de suponerlo—. El límite de la regla es el simétrico de su virtud: un "
            "entorno en el que no se midió nada no genera ningún hueco, de modo que este "
            "informe no detecta un dispositivo entero sin medir."
        ),
        environments=entornos,
        cells=tuple(celdas),
        inconsistencies=_find_inconsistencies(experimentos, por_tabla),
        device_spreads=device_independence(quality_rows),
        discarded=dict(discarded or {}),
    )

    if informe.device_dependence_detected:
        logger.warning(
            "El mAP cambia con el dispositivo en alguna configuración; revisa "
            "'device_independence' en el volcado antes de usar este informe."
        )
    logger.info(
        "Cobertura: %d celdas esperadas, %d ausentes en %d experimentos. Incoherencias: %d. "
        "Evaluadas en más de un dispositivo: %d.",
        len(informe.cells),
        len(informe.gaps),
        len(informe.gaps_by_experiment()),
        len(informe.inconsistencies),
        len(informe.device_spreads),
    )
    return informe
