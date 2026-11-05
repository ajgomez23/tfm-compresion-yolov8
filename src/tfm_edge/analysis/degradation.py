"""Degradación por compresión y su coste en eficiencia.

La referencia de una variante es la raíz de su cadena de parent_experiment_id. La comparación se
hace en el mismo motor y con el mismo evaluador; si quedan varias filas de referencia que solo
difieren en el dispositivo, deben dar el mismo valor o se aborta. La eficiencia (puntos de mAP por
factor de reducción) solo se calcula si el factor supera el suelo de medida, el de la misma sesión o
el de entre sesiones según las huellas de entorno. El punto de ruptura necesita al menos cuatro
niveles; con menos se devuelve "no determinable", y con cuatro el resultado es descriptivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tfm_edge.analysis.floors import FloorSet, same_environment
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Niveles mínimos para hablar de aceleración: con tres solo hay una diferencia segunda.
MIN_LEVELS_FOR_BREAKPOINT = 4

CLASS_COLUMN_PREFIX = "ap_class_"

SIZE_RANGES: tuple[str, ...] = ("small", "medium", "large")

#: Criterio de tamaño por defecto, el de resolución de inferencia.
DEFAULT_SIZE_CRITERION = "inference_res"

#: Etapas de una variante: una poda sin reajustar y la misma reajustada comparten nombre y ratio.
STAGE_FINETUNED = "reajustado"
STAGE_RAW = "sin_reajuste"


class DegradationError(TFMEdgeError):
    """Error al calcular la degradación."""


@dataclass(frozen=True)
class Reference:
    """Configuración contra la que se mide una variante."""

    experiment_id: str
    chain: tuple[str, ...]
    runtime: str
    precision: str

    def to_dict(self) -> dict[str, Any]:
        """Serializa la referencia."""
        return {
            "experiment_id": self.experiment_id,
            "chain": list(self.chain),
            "runtime": self.runtime,
            "precision": self.precision,
        }


@dataclass(frozen=True)
class Delta:
    """Diferencia de una variante en una columna.

    ``relative`` es None si la referencia vale cero.
    """

    column: str
    reference: float
    value: float
    absolute: float
    relative: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serializa la diferencia."""
        return {
            "column": self.column,
            "reference": self.reference,
            "value": self.value,
            "absolute": self.absolute,
            "relative": self.relative,
        }


@dataclass(frozen=True)
class Efficiency:
    """Calidad perdida por cada unidad de factor de reducción.

    ``points_per_factor`` es None si el factor no está certificado. ``same_session`` indica qué
    suelo se aplicó.
    """

    axis: str
    quality_column: str
    reference: float
    value: float
    factor: float
    quality_loss: float
    points_per_factor: float | None
    certified: bool
    same_session: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializa la eficiencia."""
        return {
            "axis": self.axis,
            "quality_column": self.quality_column,
            "reference": self.reference,
            "value": self.value,
            "factor": self.factor,
            "quality_loss": self.quality_loss,
            "points_per_factor": self.points_per_factor,
            "certified": self.certified,
            "same_session": self.same_session,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DegradationRow:
    """Degradación completa de una configuración medida.

    ``stage`` distingue una poda sin reajustar de la misma poda reajustada.
    """

    experiment_id: str
    technique: str
    stage: str
    runtime: str
    precision: str
    device_id: str
    reference: Reference
    overall: Mapping[str, Delta]
    per_class: Mapping[str, Delta]
    per_size: Mapping[str, Delta]
    efficiency: Mapping[str, Efficiency]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serializa la fila."""
        return {
            "experiment_id": self.experiment_id,
            "technique": self.technique,
            "stage": self.stage,
            "runtime": self.runtime,
            "precision": self.precision,
            "device_id": self.device_id,
            "reference": self.reference.to_dict(),
            "overall": {k: v.to_dict() for k, v in sorted(self.overall.items())},
            "per_class": {k: v.to_dict() for k, v in sorted(self.per_class.items())},
            "per_size": {k: v.to_dict() for k, v in sorted(self.per_size.items())},
            "efficiency": {k: v.to_dict() for k, v in sorted(self.efficiency.items())},
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Breakpoint:
    """Nivel donde se acelera la degradación, si se puede determinar."""

    determinable: bool
    level: float | None
    second_differences: tuple[tuple[float, float], ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serializa el resultado."""
        return {
            "determinable": self.determinable,
            "level": self.level,
            "second_differences": [list(par) for par in self.second_differences],
            "reason": self.reason,
        }


def class_columns(row: Mapping[str, str]) -> tuple[str, ...]:
    """Columnas de AP por clase presentes en una fila, en el orden del registro."""
    return tuple(c for c in row if c.startswith(CLASS_COLUMN_PREFIX))


def class_label(column: str) -> str:
    """Nombre de clase legible a partir de ``ap_class_NN_nombre``. Falla si la columna no tiene esa
    forma.
    """
    if not column.startswith(CLASS_COLUMN_PREFIX):
        raise DegradationError(
            f"'{column}' no es una columna de AP por clase: debe empezar por "
            f"'{CLASS_COLUMN_PREFIX}'."
        )
    resto = column[len(CLASS_COLUMN_PREFIX) :]
    indice, _, nombre = resto.partition("_")
    if not indice.isdigit() or not nombre:
        raise DegradationError(
            f"'{column}' no sigue el formato ap_class_NN_nombre, así que no se puede "
            f"deducir de qué clase es sin inventárselo."
        )
    return nombre.replace("_", " ")


def size_columns(criterion: str = DEFAULT_SIZE_CRITERION) -> dict[str, str]:
    """Columnas de AP por rango de tamaño para un criterio."""
    return {rango: f"ap_{rango}_{criterion}" for rango in SIZE_RANGES}


def resolve_reference(
    experiment_id: str, experiments: Mapping[str, Mapping[str, str]]
) -> tuple[str, tuple[str, ...]]:
    """Sube por la cadena de ascendencia hasta la raíz y devuelve la raíz y la cadena recorrida.

    Falla si un padre no está en el registro o si la cadena es cíclica.
    """
    cadena: list[str] = [experiment_id]
    vistos = {experiment_id}
    actual = experiment_id
    while True:
        fila = experiments.get(actual)
        if fila is None:
            raise DegradationError(
                f"La cadena de ascendencia de '{experiment_id}' pasa por '{actual}', que no está "
                "en experiments.csv."
            )
        padre = (fila.get("parent_experiment_id", "") or "").strip()
        if not padre:
            return actual, tuple(cadena)
        if padre in vistos:
            raise DegradationError(
                f"La cadena de ascendencia de '{experiment_id}' es cíclica: '{padre}' "
                f"ya aparecía en {cadena}."
            )
        vistos.add(padre)
        cadena.append(padre)
        actual = padre


def _epochs_of(row: Mapping[str, str]) -> int:
    """Épocas declaradas por un experimento, 0 si la celda está vacía."""
    bruto = (row.get("training_epochs", "") or "").strip()
    if not bruto:
        return 0
    try:
        return int(float(bruto))
    except ValueError as error:
        raise DegradationError(
            f"El experimento {row.get('experiment_id', '?')} declara "
            f"training_epochs='{bruto}', que no es un número de épocas."
        ) from error


def training_stage(experiment_id: str, experiments: Mapping[str, Mapping[str, str]]) -> str:
    """Indica si la compresión se reajustó con entrenamiento, mirando toda la cadena menos la raíz.

    Un modelo combinado no declara épocas, pero desciende de una poda reajustada. Las épocas de la
    raíz son las del baseline y no cuentan.
    """
    _, cadena = resolve_reference(experiment_id, experiments)
    for eslabon in cadena[:-1]:
        if _epochs_of(experiments[eslabon]) > 0:
            return STAGE_FINETUNED
    return STAGE_RAW


@dataclass(frozen=True)
class ChainCost:
    """Coste de producir una variante sumando su cadena."""

    experiment_id: str
    seconds: float | None
    contributors: tuple[tuple[str, float], ...]
    missing: tuple[str, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializa el coste."""
        return {
            "experiment_id": self.experiment_id,
            "seconds": self.seconds,
            "contributors": [list(par) for par in self.contributors],
            "missing": list(self.missing),
            "reason": self.reason,
        }


def chain_cost(
    experiment_id: str,
    experiments: Mapping[str, Mapping[str, str]],
    column: str = "training_time_s",
    extra_costs: Mapping[str, float] | None = None,
) -> ChainCost:
    """Suma el coste de los pasos que llevan del baseline a una variante, sin contar la raíz.

    Los eslabones de PTQ o combinación no entrenan; su coste de exportar y cuantizar, que depende
    del runtime, entra por ``extra_costs``. Un eslabón sin coste no se cuenta como cero: si falta
    alguno, no se devuelve total.
    """
    _, cadena = resolve_reference(experiment_id, experiments)
    eslabones = cadena[:-1]

    extra = dict(extra_costs or {})
    aportes: list[tuple[str, float]] = []
    faltan: list[str] = []
    for eslabon in eslabones:
        valor = _value(experiments[eslabon], column)
        adicional = extra.get(eslabon)
        if valor is None and adicional is None:
            faltan.append(eslabon)
            continue
        aportes.append((eslabon, (valor or 0.0) + (adicional or 0.0)))

    if not eslabones:
        return ChainCost(
            experiment_id=experiment_id,
            seconds=None,
            contributors=(),
            missing=(),
            reason=(
                "Es la raíz de su cadena: su coste es el del entrenamiento del "
                "baseline, no el de comprimir nada."
            ),
        )
    if faltan:
        return ChainCost(
            experiment_id=experiment_id,
            seconds=None,
            contributors=tuple(aportes),
            missing=tuple(faltan),
            reason=(
                f"No hay coste registrado de {list(faltan)}, ni de entrenamiento ni de "
                f"exportación. Sumar solo los eslabones que sí lo tienen daría una "
                f"cifra menor que la real con aspecto de medida."
            ),
        )
    return ChainCost(
        experiment_id=experiment_id,
        seconds=sum(valor for _, valor in aportes),
        contributors=tuple(aportes),
        missing=(),
    )


def _value(row: Mapping[str, str], column: str) -> float | None:
    """Número de una celda, o None si está vacía. Falla si no es un número."""
    bruto = (row.get(column, "") or "").strip()
    if not bruto:
        return None
    try:
        return float(bruto)
    except ValueError as error:
        raise DegradationError(
            f"La columna '{column}' de {row.get('experiment_id', '?')} contiene "
            f"'{bruto}', que no es un número."
        ) from error


def delta(reference_row: Mapping[str, str], row: Mapping[str, str], column: str) -> Delta | None:
    """Diferencia de una columna entre variante y referencia, o None si falta algún valor."""
    referencia = _value(reference_row, column)
    valor = _value(row, column)
    if referencia is None or valor is None:
        return None
    absoluta = valor - referencia
    relativa = absoluta / referencia if referencia != 0 else None
    return Delta(
        column=column,
        reference=referencia,
        value=valor,
        absolute=absoluta,
        relative=relativa,
    )


def compression_efficiency(
    quality_loss: float,
    quality_column: str,
    reference_value: float,
    value: float,
    axis: str,
    floors: FloorSet,
    same_session: bool,
) -> Efficiency:
    """Pérdida de calidad por unidad de factor de reducción, si el factor está certificado.

    ``same_session`` no tiene valor por defecto: decide qué suelo se aplica y lo determina quien
    llama con las dos filas.
    """
    if value == 0:
        raise DegradationError(
            f"La variante mide 0 en '{axis}', así que el factor de reducción sería "
            f"infinito. Un cero ahí es una medida rota, no una reducción."
        )
    factor = reference_value / value

    suelo = floors.get(axis)
    if suelo is None or not suelo.is_measured():
        return Efficiency(
            axis=axis,
            quality_column=quality_column,
            reference=reference_value,
            value=value,
            factor=factor,
            quality_loss=quality_loss,
            points_per_factor=None,
            certified=False,
            same_session=same_session,
            reason=(
                f"El eje '{axis}' no tiene suelo de medida medido, así que no se puede "
                f"certificar que la reducción sea real (D-127). Repartir la pérdida de "
                f"calidad entre un factor sin certificar daría una cifra que parece una "
                f"conclusión sin serlo."
            ),
        )

    # Los suelos de latencia y memoria son relativos al menor de los dos valores, como en
    # pareto.compare; el de mAP es absoluto.
    tolerancia = suelo.applicable(same_session)
    alcance = "de la misma sesión" if same_session else "entre sesiones"
    diferencia = abs(reference_value - value)
    if suelo.kind == "relativo":
        margen = (tolerancia or 0.0) * min(abs(reference_value), abs(value))
        margen_texto = f"{(tolerancia or 0.0):.1%} del menor de los dos valores (suelo {alcance})"
    else:
        margen = tolerancia or 0.0
        margen_texto = f"{margen:g} en las unidades del eje (suelo {alcance})"
    if tolerancia is None or diferencia <= margen:
        return Efficiency(
            axis=axis,
            quality_column=quality_column,
            reference=reference_value,
            value=value,
            factor=factor,
            quality_loss=quality_loss,
            points_per_factor=None,
            certified=False,
            same_session=same_session,
            reason=(
                f"La reducción en '{axis}' es de {diferencia:g} y la resolución de la "
                f"medida es de {margen_texto}: dividir por un factor que la medida no "
                f"distingue de 1 no da un número, da ruido amplificado."
            ),
        )

    if factor <= 1.0:
        return Efficiency(
            axis=axis,
            quality_column=quality_column,
            reference=reference_value,
            value=value,
            factor=factor,
            quality_loss=quality_loss,
            points_per_factor=None,
            certified=True,
            same_session=same_session,
            reason=(
                f"La variante no reduce '{axis}': el factor es {factor:.3f}. Repartir la "
                f"pérdida entre una reducción que no existe no significa nada."
            ),
        )

    return Efficiency(
        axis=axis,
        quality_column=quality_column,
        reference=reference_value,
        value=value,
        factor=factor,
        quality_loss=quality_loss,
        points_per_factor=quality_loss / (factor - 1.0),
        certified=True,
        same_session=same_session,
    )


def find_breakpoint(levels: Sequence[tuple[float, float]]) -> Breakpoint:
    """Busca el nivel en el que la degradación se acelera.

    Con menos de MIN_LEVELS_FOR_BREAKPOINT niveles distintos no es determinable. Falla si un nivel
    aparece con dos calidades distintas.
    """
    por_nivel: dict[float, float] = {}
    for nivel, calidad in levels:
        if nivel in por_nivel and por_nivel[nivel] != calidad:
            raise DegradationError(
                f"El nivel {nivel} aparece con dos calidades distintas ({por_nivel[nivel]} y "
                f"{calidad})."
            )
        por_nivel[nivel] = calidad

    ordenados = sorted(por_nivel.items())
    if len(ordenados) < MIN_LEVELS_FOR_BREAKPOINT:
        return Breakpoint(
            determinable=False,
            level=None,
            second_differences=(),
            reason=(
                f"Hay {len(ordenados)} niveles de compresión y hacen falta al menos "
                f"{MIN_LEVELS_FOR_BREAKPOINT}. Detectar que algo se acelera exige "
                f"comparar dos curvaturas, y con tres niveles solo hay una: cualquier "
                f"valor no nulo parecería una ruptura."
            ),
        )

    segundas: list[tuple[float, float]] = []
    for i in range(1, len(ordenados) - 1):
        (x0, y0), (x1, y1), (x2, y2) = ordenados[i - 1], ordenados[i], ordenados[i + 1]
        pendiente_izquierda = (y1 - y0) / (x1 - x0)
        pendiente_derecha = (y2 - y1) / (x2 - x1)
        segundas.append((x1, pendiente_derecha - pendiente_izquierda))

    # La calidad cae al comprimir: acelerarse es la diferencia segunda más negativa.
    nivel_ruptura = min(segundas, key=lambda par: (par[1], par[0]))[0]
    return Breakpoint(
        determinable=True,
        level=nivel_ruptura,
        second_differences=tuple(segundas),
        reason=(
            "Es el nivel donde la caída más se acelera. Es una descripción de los "
            "puntos medidos, no un contraste: este estudio no tiene dispersión medida "
            "con la que decidir si la aceleración supera el ruido, así que no "
            "debe citarse como significativa."
        ),
    )


def _pick_reference(
    candidates: Sequence[Mapping[str, str]],
    device_id: str,
    columns: Sequence[str],
) -> Mapping[str, str] | None:
    """Elige la fila de referencia entre las que comparten motor y evaluador.

    Prefiere el mismo dispositivo que la variante; si no lo hay, admite otra solo si todas las
    candidatas dan el mismo valor.
    """
    if not candidates:
        return None

    mismas = [c for c in candidates if (c.get("device_id", "") or "").strip() == device_id]
    utiles = mismas or list(candidates)
    if len(utiles) == 1:
        return utiles[0]

    ordenadas = sorted(utiles, key=lambda c: (c.get("device_id", ""), c.get("precision", "")))
    for columna in columns:
        valores = {(c.get(columna, "") or "").strip() for c in ordenadas}
        if len(valores) > 1:
            raise DegradationError(
                f"La referencia {ordenadas[0]['experiment_id']} en {ordenadas[0]['runtime']} tiene "
                f"{len(ordenadas)} filas que no coinciden en '{columna}' ({sorted(valores)})."
            )
    return ordenadas[0]


def degradation_rows(
    experiment_rows: Sequence[Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
    deployment_rows: Sequence[Mapping[str, str]],
    floors: FloorSet,
    quality_columns: Sequence[str] = ("map_50", "map_50_95"),
    efficiency_axes: Sequence[str] = ("latency_p95_ms", "artifact_size_bytes"),
    size_criterion: str = DEFAULT_SIZE_CRITERION,
) -> tuple[DegradationRow, ...]:
    """Degradación de todas las variantes con referencia medida en el mismo motor.

    Las que no tienen referencia comparable no aparecen y el motivo queda en el log.
    """
    experimentos = {fila["experiment_id"]: fila for fila in experiment_rows}
    candidatas_referencia: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for fila in quality_rows:
        clave = (
            fila["experiment_id"],
            fila["runtime"],
            (fila.get("evaluator", "") or "").strip(),
        )
        candidatas_referencia.setdefault(clave, []).append(fila)
    despliegue_por_clave: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for fila in deployment_rows:
        despliegue_por_clave[(fila["experiment_id"], fila["runtime"], fila["precision"])] = fila

    resultado: list[DegradationRow] = []
    sin_referencia = 0
    for fila in quality_rows:
        experiment_id = fila["experiment_id"]
        motor = fila["runtime"]
        if experiment_id not in experimentos:
            continue
        raiz, cadena = resolve_reference(experiment_id, experimentos)
        if raiz == experiment_id:
            continue

        evaluador = (fila.get("evaluator", "") or "").strip()
        fila_referencia = _pick_reference(
            candidatas_referencia.get((raiz, motor, evaluador), ()),
            fila.get("device_id", ""),
            quality_columns,
        )
        if fila_referencia is None:
            sin_referencia += 1
            logger.info(
                "Sin referencia de %s en %s con el evaluador '%s': no se compara.",
                experiment_id,
                motor,
                evaluador,
            )
            continue

        notas: list[str] = []
        globales = {}
        for columna in quality_columns:
            diferencia = delta(fila_referencia, fila, columna)
            if diferencia is None:
                notas.append(f"Sin valor de '{columna}' en la variante o en la referencia.")
            else:
                globales[columna] = diferencia

        por_clase = {}
        for columna in class_columns(fila):
            diferencia = delta(fila_referencia, fila, columna)
            if diferencia is not None:
                por_clase[class_label(columna)] = diferencia
        if not por_clase:
            notas.append("Sin AP por clase medido: el desglose por clase queda vacío.")

        por_tamano = {}
        for rango, columna in size_columns(size_criterion).items():
            diferencia = delta(fila_referencia, fila, columna)
            if diferencia is not None:
                por_tamano[rango] = diferencia
        if not por_tamano:
            notas.append(
                f"Sin AP por tamaño con el criterio '{size_criterion}': el desglose por "
                f"tamaño queda vacío."
            )

        eficiencias = {}
        precision = fila["precision"]
        despliegue_variante = despliegue_por_clave.get((experiment_id, motor, precision))
        dispositivo = ""
        if despliegue_variante is not None:
            dispositivo = despliegue_variante.get("device_id", "")
            precision_referencia = (fila_referencia.get("precision", "") or "").strip()
            despliegue_referencia = despliegue_por_clave.get((raiz, motor, precision_referencia))
            if despliegue_referencia is None:
                notas.append(
                    f"La referencia no tiene despliegue medido en {motor}/"
                    f"{precision_referencia}: no se calcula la eficiencia."
                )
            else:
                principal = quality_columns[-1]
                # La sesión se decide sobre las dos filas comparadas, como en el paso 26.
                misma_sesion = same_environment(
                    despliegue_variante.get("environment_id", ""),
                    despliegue_referencia.get("environment_id", ""),
                )
                if principal in globales:
                    perdida = -globales[principal].absolute
                    for eje in efficiency_axes:
                        valor_referencia = _value(despliegue_referencia, eje)
                        valor = _value(despliegue_variante, eje)
                        if valor_referencia is None or valor is None:
                            notas.append(f"Sin '{eje}' medido en los dos lados.")
                            continue
                        eficiencias[eje] = compression_efficiency(
                            quality_loss=perdida,
                            quality_column=principal,
                            reference_value=valor_referencia,
                            value=valor,
                            axis=eje,
                            floors=floors,
                            same_session=misma_sesion,
                        )

        resultado.append(
            DegradationRow(
                experiment_id=experiment_id,
                technique=experimentos[experiment_id].get("technique", ""),
                stage=training_stage(experiment_id, experimentos),
                runtime=motor,
                precision=precision,
                device_id=dispositivo,
                reference=Reference(
                    experiment_id=raiz,
                    chain=cadena,
                    runtime=motor,
                    precision=(fila_referencia.get("precision", "") or "").strip(),
                ),
                overall=globales,
                per_class=por_clase,
                per_size=por_tamano,
                efficiency=eficiencias,
                notes=tuple(notas),
            )
        )

    if sin_referencia:
        logger.info(
            "Configuraciones sin referencia en su mismo motor, fuera del análisis de "
            "degradación: %d.",
            sin_referencia,
        )
    return tuple(sorted(resultado, key=lambda r: (r.experiment_id, r.runtime, r.precision)))
