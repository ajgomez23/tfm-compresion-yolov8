"""Suelo de lectura de cada eje, medido a partir de pares de control.

Una diferencia menor que el suelo no se presenta como resultado. El suelo depende del eje: sobre el
mismo par, latency_p50_ms varía un 9,3 % y latency_p95_ms un 14,7 %. Un par de control son dos
medidas de coste equivalente, así que su diferencia es error de medida; cada par indica si es entre
procesos de la misma sesión o entre sesiones, y si está contaminado. Los pares se identifican por
volcados en disco y no por filas del registro, porque una fila puede reescribirse; _check_pair_kind
comprueba que las fechas de las tomas cuadran con el tipo declarado. Es una cota inferior del ruido,
no una estimación de la varianza.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

# El suelo de calidad se importa de reading para no duplicar la cifra.
from tfm_edge.compression.distillation.reading import EVALUATOR_READABILITY_FLOOR

logger = get_logger(__name__)

#: Bytes por MiB; peak_host_memory_mb está en MiB.
BYTES_PER_MIB: int = 1024 * 1024


class FloorMeasurementError(TFMEdgeError):
    """El suelo no se puede medir con lo que hay en disco. No se devuelve ningún valor por
    defecto."""


@dataclass(frozen=True)
class PairMember:
    """Una de las dos tomas de un par de control: una fila del registro o un volcado suelto."""

    label: str
    experiment_id: str | None = None
    runtime: str | None = None
    dump: str | None = None

    def __post_init__(self) -> None:
        """Comprueba que la toma se identifica de una sola forma."""
        del_registro = self.experiment_id is not None and self.runtime is not None
        if del_registro == (self.dump is not None):
            raise FloorMeasurementError(
                f"La toma '{self.label}' debe identificarse por (experiment_id, runtime) o por "
                "un volcado, y solo de una de las dos formas."
            )


@dataclass(frozen=True)
class ControlPair:
    """Dos medidas cuya diferencia es error de medida.

    ``kind`` es ``"entre_procesos"`` (mismo día y campaña) o ``"entre_sesiones"``. Se comprueba que
    los ficheros de ``evidence`` existen.
    """

    name: str
    kind: str
    rationale: str
    evidence: tuple[str, ...]
    members: tuple[PairMember, PairMember]
    contaminated: bool = False
    contamination_reason: str = ""


@dataclass(frozen=True)
class AxisFloor:
    """Suelo de un eje y su procedencia.

    ``same_session`` y ``cross_session`` son los suelos dentro de una sesión y entre sesiones; None
    si no están medidos.
    """

    axis: str
    kind: str
    same_session: float | None
    cross_session: float | None
    provenance: str
    caveat: str = ""

    def applicable(self, same_session: bool) -> float | None:
        """Suelo aplicable según si los dos puntos comparten sesión, o None."""
        return self.same_session if same_session else self.cross_session

    def is_measured(self) -> bool:
        """True si al menos uno de los dos suelos está medido."""
        return self.same_session is not None or self.cross_session is not None


def same_environment(environment_a: str, environment_b: str) -> bool:
    """True si las dos huellas de entorno existen y son iguales.

    Es el criterio común para elegir suelo en pareto, degradation, selection y tables. Una huella
    vacía cuenta como sesión distinta, que es lo conservador.
    """
    a = (environment_a or "").strip()
    b = (environment_b or "").strip()
    return bool(a) and a == b


@dataclass(frozen=True)
class FloorSet:
    """Suelos de todos los ejes con las observaciones de las que salen."""

    floors: dict[str, AxisFloor]
    pair_observations: dict[str, dict[str, float]] = field(default_factory=dict)
    skipped_pairs: dict[str, str] = field(default_factory=dict)

    def get(self, axis: str) -> AxisFloor | None:
        """Suelo de un eje, o None si no tiene."""
        return self.floors.get(axis)

    def to_dict(self) -> dict[str, Any]:
        """Serializa el conjunto para volcarlo junto al resultado."""
        return {
            "floors": {
                eje: {
                    "axis": suelo.axis,
                    "kind": suelo.kind,
                    "same_session": suelo.same_session,
                    "cross_session": suelo.cross_session,
                    "provenance": suelo.provenance,
                    "caveat": suelo.caveat,
                }
                for eje, suelo in sorted(self.floors.items())
            },
            "pair_observations": {
                par: {eje: valor for eje, valor in sorted(ejes.items())}
                for par, ejes in sorted(self.pair_observations.items())
            },
            "skipped_pairs": dict(sorted(self.skipped_pairs.items())),
        }


#: Rutas de los pares de control, relativas a la raíz.
_EVIDENCE_DIR = "docs/evidencia"
_ARCHIVE_DIR = "results/archive/2026-09-01_latencia_pre_regimen_declarado/raw_benchmark"
#: Volcados de la remedida archivada antes de la campaña única. Los pares entre sesiones leen de
#: aquí porque la campaña única sobrescribe los volcados de results/raw/benchmark.
_D136_ARCHIVE_DIR = (
    "results/archive/2026-09-17_latencia_regimen_d136_y_sesiones_declaradas/raw_benchmark"
)
_LIVE_DIR = "results/raw/benchmark"

#: Perfil y resolución de los nombres de volcado de la campaña de CPU.
_DUMP_PROFILE = "workstation_cpu"
_DUMP_IMGSZ = 640


def dump_filename(experiment_id: str, runtime: str, precision: str) -> str:
    """Nombre del volcado de benchmark de una combinación medida."""
    return f"{_DUMP_PROFILE}_{experiment_id}_{runtime}_{precision}_{_DUMP_IMGSZ}.json"


#: Tres controles de contorno: el mismo ONNX medido al principio, a la mitad y al final de una
#: sesión de tres horas. Se tomaron para detectar deriva y aquí se reutilizan como control entre
#: procesos.
_D136_CONTROLS: tuple[str, ...] = (
    "d136_control_contorno_1.json",
    "d136_control_contorno_2.json",
    "d136_control_contorno_3.json",
)

#: Resumen de la campaña, que confirma que los tres controles son de la misma sesión y su orden.
_D136_SUMMARY = "d136_resumen_campana.json"

#: Combinaciones medidas dos veces, en el archivo de agosto y en la remedida, como ``(experiment_id,
#: runtime, precision)``. La lista se declara a mano y verify_declared_pairs_match_disk la contrasta
#: con el disco.
_REMEASURED: tuple[tuple[str, str, str], ...] = (
    ("l_baseline_default_188b9407", "onnxruntime", "fp32"),
    ("l_baseline_default_188b9407", "openvino", "fp32"),
    ("l_baseline_default_188b9407", "pytorch", "fp32"),
    ("l_ptq_int8-noexcl_4a52a7e5", "onnxruntime", "int8"),
    ("l_ptq_int8-noexcl_4a52a7e5", "openvino", "int8"),
    ("m_baseline_default_dfe22158", "onnxruntime", "fp32"),
    ("m_baseline_default_dfe22158", "openvino", "fp32"),
    ("m_baseline_default_dfe22158", "pytorch", "fp32"),
    ("m_ptq_int8-noexcl_b1a6e9e7", "onnxruntime", "int8"),
    ("m_ptq_int8-noexcl_b1a6e9e7", "openvino", "int8"),
    ("n_baseline_default_fab9d45c", "onnxruntime", "fp32"),
    ("n_baseline_default_fab9d45c", "pytorch", "fp32"),
    ("n_ptq_int8-noexcl_a78f566a", "openvino", "int8"),
    ("s_baseline_default_745e3d08", "onnxruntime", "fp32"),
    ("s_baseline_default_745e3d08", "openvino", "fp32"),
    ("s_baseline_default_745e3d08", "pytorch", "fp32"),
    ("s_combined_prune-bnscale-r50-ptqint8_7712cb36", "onnxruntime", "int8"),
    ("s_combined_prune-bnscale-r50-ptqint8_7712cb36", "openvino", "int8"),
    ("s_combined_prune-l1global-r25-ptqint8_4a1d5adc", "onnxruntime", "int8"),
    ("s_combined_prune-l1global-r25-ptqint8_4a1d5adc", "openvino", "int8"),
    ("s_combined_prune-l1global-r25-randomctl-ptqint8_e70dc4d6", "onnxruntime", "int8"),
    ("s_combined_prune-l1global-r25-randomctl-ptqint8_e70dc4d6", "openvino", "int8"),
    ("s_combined_prune-l1global-r50-iter4-ptqint8_6f322ad5", "onnxruntime", "int8"),
    ("s_combined_prune-l1global-r50-iter4-ptqint8_6f322ad5", "openvino", "int8"),
    ("s_combined_prune-l1global-r50-ptqint8_9067dba2", "onnxruntime", "int8"),
    ("s_combined_prune-l1global-r50-ptqint8_9067dba2", "openvino", "int8"),
    ("s_combined_prune-l1global-r50-randomctl-ptqint8_13a9774e", "onnxruntime", "int8"),
    ("s_combined_prune-l1global-r50-randomctl-ptqint8_13a9774e", "openvino", "int8"),
    ("s_pruning_bnscale_r50_f51fef4c", "onnxruntime", "fp32"),
    ("s_pruning_bnscale_r50_f51fef4c", "openvino", "fp32"),
    ("s_pruning_l1global_r25_4e91ddbc", "onnxruntime", "fp32"),
    ("s_pruning_l1global_r25_4e91ddbc", "openvino", "fp32"),
    ("s_pruning_l1global_r25_randomctl_a658e5d4", "onnxruntime", "fp32"),
    ("s_pruning_l1global_r25_randomctl_a658e5d4", "openvino", "fp32"),
    ("s_pruning_l1global_r50_9419b26f", "onnxruntime", "fp32"),
    ("s_pruning_l1global_r50_9419b26f", "openvino", "fp32"),
    ("s_pruning_l1global_r50_iter4_d8fe066f", "onnxruntime", "fp32"),
    ("s_pruning_l1global_r50_iter4_d8fe066f", "openvino", "fp32"),
    ("s_pruning_l1global_r50_randomctl_d581bd22", "onnxruntime", "fp32"),
    ("s_pruning_l1global_r50_randomctl_d581bd22", "openvino", "fp32"),
    ("s_ptq_int8-noexcl_5bcd71f0", "onnxruntime", "int8"),
    ("s_ptq_int8-noexcl_5bcd71f0", "openvino", "int8"),
    ("s_qat_int8-noexcl_d4d403fa", "onnxruntime", "int8"),
    ("s_qat_int8-noexcl_d4d403fa", "openvino", "int8"),
)

#: Las cuatro tomas de la ventana transitoria del 31/08 (17:17 a 17:25) no fijan suelo: remedidas
#: dan factores de 1,2 a 1,7 que decaen y a partir de las 17:37 vuelven a 0,96-1,04.
_CONTAMINATED_TRANSIENT: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("n_baseline_default_fab9d45c", "pytorch", "fp32"),
        ("n_baseline_default_fab9d45c", "onnxruntime", "fp32"),
        ("n_ptq_int8-noexcl_a78f566a", "openvino", "int8"),
        ("m_baseline_default_dfe22158", "pytorch", "fp32"),
    }
)


def _d095_process_pair() -> ControlPair:
    """Par entre procesos de la toma descartada y la registrada del QAT en ONNX Runtime INT8.

    Son dos volcados del mismo fichero con 5 min 29 s entre arranques. Ninguna toma es una fila del
    registro.
    """
    descartada = f"{_EVIDENCE_DIR}/d095_toma_descartada_qat_onnxruntime_int8.json"
    conservada = f"{_ARCHIVE_DIR}/" + dump_filename(
        "s_qat_int8-noexcl_d4d403fa", "onnxruntime", "int8"
    )
    return ControlPair(
        name="A_entre_procesos",
        kind="entre_procesos",
        rationale=(
            "El mismo fichero medido dos veces, con 5 min 29 s entre los dos "
            "arranques y las dos tomas en reposo (carga por núcleo 0,027 y "
            "0,014). No hay nada que pueda diferir salvo la medida. Las dos "
            "tomas son volcados en disco, no filas del registro: una fila se "
            "reescribe y un volcado archivado no."
        ),
        evidence=(descartada, conservada),
        members=(
            PairMember(label="toma descartada por D-095", dump=descartada),
            PairMember(label="toma conservada, en el archivo previo a D-136", dump=conservada),
        ),
    )


def _d136_boundary_pairs() -> tuple[ControlPair, ...]:
    """Las tres parejas que forman los tres controles de contorno.

    Se declaran todas porque cuál da la diferencia mayor depende del eje.
    """
    resumen = f"{_EVIDENCE_DIR}/{_D136_SUMMARY}"
    rutas = tuple(f"{_EVIDENCE_DIR}/{nombre}" for nombre in _D136_CONTROLS)
    pares = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        pares.append(
            ControlPair(
                name=f"D136_contorno_{i + 1}{j + 1}",
                kind="entre_procesos",
                rationale=(
                    "Dos tomas del mismo fichero ONNX en la misma sesión y bajo "
                    "el mismo régimen declarado, separadas por hora y media o "
                    "por tres horas. Lo que se ejecuta es idéntico byte a byte, "
                    "así que la diferencia observada es error de medida y nada "
                    "más. Se tomaron para vigilar la deriva de la campaña, no "
                    "para fijar este suelo."
                ),
                evidence=(rutas[i], rutas[j], resumen),
                members=(
                    PairMember(label=f"control de contorno {i + 1} (D-136)", dump=rutas[i]),
                    PairMember(label=f"control de contorno {j + 1} (D-136)", dump=rutas[j]),
                ),
            )
        )
    return tuple(pares)


def _cross_session_pairs() -> tuple[ControlPair, ...]:
    """Pares entre sesiones: cada combinación del archivo de agosto contra su remedida archivada.

    Las dos tomas difieren también en el régimen de medida, así que el suelo resultante es una cota
    alta del error entre sesiones. La campaña única no entra en estos pares.
    """
    pares = []
    for clave in _REMEASURED:
        nombre = dump_filename(*clave)
        archivada = f"{_ARCHIVE_DIR}/{nombre}"
        viva = f"{_D136_ARCHIVE_DIR}/{nombre}"
        contaminado = clave in _CONTAMINATED_TRANSIENT
        pares.append(
            ControlPair(
                name="X_%s_%s_%s" % clave,
                kind="entre_sesiones",
                rationale=(
                    "El mismo artefacto medido en dos sesiones distintas: la "
                    "archivada de agosto, sin régimen declarado, y la del 02/09 "
                    "bajo el régimen de D-136. Lo que se ejecuta es el mismo "
                    "fichero, así que la diferencia no viene del modelo."
                ),
                evidence=(archivada, viva),
                members=(
                    PairMember(label=f"toma archivada de {nombre}", dump=archivada),
                    PairMember(
                        label=f"remedida bajo el régimen de D-136, archivada, de {nombre}",
                        dump=viva,
                    ),
                ),
                contaminated=contaminado,
                contamination_reason=(
                    (
                        "Cae en la ventana transitoria del 31/08 (17:17-17:25), "
                        "donde la máquina entera fue entre un 20 % y un 69 % más "
                        "lenta durante unos veinte minutos, con la peor toma a "
                        "carga por núcleo 0,0. La diferencia de este par es esa "
                        "avería, no el error de medir en dos sesiones."
                    )
                    if contaminado
                    else ""
                ),
            )
        )
    return tuple(pares)


def default_control_pairs() -> tuple[ControlPair, ...]:
    """Pares de control del proyecto: primero los de la misma sesión y después los entre
    sesiones."""
    return _d095_process_pair(), *_d136_boundary_pairs(), *_cross_session_pairs()


def verify_declared_pairs_match_disk(project_root: Path) -> tuple[str, ...]:
    """Diferencias entre la lista declarada de remedidas y lo que hay en disco.

    No se llama desde measure_floors; la usan los tests. Se cruzan los dos archivos, no el
    directorio vivo.
    """
    archivo = project_root / _ARCHIVE_DIR
    remedidas = project_root / _D136_ARCHIVE_DIR
    if not archivo.is_dir():
        return (f"No existe el directorio del archivo previo a D-136: {_ARCHIVE_DIR}",)
    if not remedidas.is_dir():
        return (f"No existe el directorio de la remedida archivada de D-136: {_D136_ARCHIVE_DIR}",)

    en_disco = {p.name for p in archivo.glob("*.json") if (remedidas / p.name).is_file()}
    declarados = {dump_filename(*clave) for clave in _REMEASURED}

    problemas = [
        f"Declarada pero sin las dos tomas en disco: {nombre}"
        for nombre in sorted(declarados - en_disco)
    ]
    problemas += [
        f"Remedida en disco y no declarada como par entre sesiones: {nombre}"
        for nombre in sorted(en_disco - declarados)
    ]
    return tuple(problemas)


#: Cómo se saca cada eje de un volcado de benchmark, con su unidad.
_DUMP_ACCESSORS: dict[str, Any] = {
    "latency_p50_ms": lambda d: d["aggregated"]["latency_p50_ms"],
    "latency_p90_ms": lambda d: d["aggregated"]["latency_p90_ms"],
    "latency_p95_ms": lambda d: d["aggregated"]["latency_p95_ms"],
    "latency_p99_ms": lambda d: d["aggregated"]["latency_p99_ms"],
    "latency_mean_ms": lambda d: d["aggregated"]["latency_mean_ms"],
    "peak_host_memory_mb": lambda d: d["memory"]["rss_peak_bytes"] / BYTES_PER_MIB,
}


def _value_from_dump(dump: Mapping[str, Any], axis: str) -> float | None:
    """Valor de un eje en un volcado de benchmark, o None. Falla si el eje no tiene traducción."""
    if axis not in _DUMP_ACCESSORS:
        raise FloorMeasurementError(
            f"No se sabe de dónde sacar '{axis}' de un volcado de benchmark. Añádelo a "
            "_DUMP_ACCESSORS con su unidad."
        )
    try:
        valor = _DUMP_ACCESSORS[axis](dump)
    except (KeyError, TypeError):
        return None
    return None if valor in ("", None) else float(valor)


def _value_from_registry(row: Mapping[str, str], axis: str) -> float | None:
    """Valor de un eje en una fila de despliegue; None si la celda está vacía."""
    bruto = row.get(axis, "")
    bruto = bruto.strip() if isinstance(bruto, str) else bruto
    if bruto in ("", None):
        return None
    return float(bruto)


def _member_values(
    member: PairMember,
    deployment_rows: Sequence[Mapping[str, str]],
    axes: Sequence[str],
    project_root: Path,
) -> tuple[dict[str, float], str]:
    """Valores de una toma para los ejes pedidos y su fecha (``AAAA-MM-DD`` o vacía)."""
    if member.dump is not None:
        ruta = project_root / member.dump
        if not ruta.is_file():
            raise FloorMeasurementError(
                f"La toma '{member.label}' dice estar en {member.dump} y ahí no "
                f"hay nada. Sin ella no se puede medir el suelo de este par."
            )
        volcado = json.loads(ruta.read_text(encoding="utf-8"))
        # Un volcado de benchmark solo tiene ejes de despliegue; el suelo de calidad se añade en
        # measure_floors.
        valores = {eje: _value_from_dump(volcado, eje) for eje in axes if eje in _DUMP_ACCESSORS}
        fecha = str(volcado.get("started_at", ""))[:10]
    else:
        candidatas = [
            fila
            for fila in deployment_rows
            if fila.get("experiment_id") == member.experiment_id
            and fila.get("runtime") == member.runtime
        ]
        if len(candidatas) != 1:
            raise FloorMeasurementError(
                f"La toma '{member.label}' ({member.experiment_id}, {member.runtime}) casa con "
                f"{len(candidatas)} filas de despliegue y debe casar con una."
            )
        valores = {eje: _value_from_registry(candidatas[0], eje) for eje in axes}
        fecha = str(candidatas[0].get("timestamp", ""))[:10]
    return {eje: valor for eje, valor in valores.items() if valor is not None}, fecha


def _check_pair_kind(pair: ControlPair, first_date: str, second_date: str) -> None:
    """Comprueba que las fechas de las dos tomas cuadran con el tipo declarado del par."""
    if not first_date or not second_date:
        return
    misma_fecha = first_date == second_date
    if pair.kind == "entre_procesos" and not misma_fecha:
        raise FloorMeasurementError(
            f"El par '{pair.name}' se declara entre procesos de la misma sesión, pero sus tomas "
            f"son del {first_date} y del {second_date}."
        )
    if pair.kind == "entre_sesiones" and misma_fecha:
        raise FloorMeasurementError(
            f"El par '{pair.name}' se declara entre sesiones, pero sus dos tomas son del mismo día "
            f"({first_date})."
        )


def relative_difference(primero: float, segundo: float) -> float:
    """Diferencia relativa respecto a la menor de las dos medidas, sin signo.

    Dividir por la menor da la cota más alta y no depende del orden.
    """
    if primero <= 0 or segundo <= 0:
        raise FloorMeasurementError(
            f"Una diferencia relativa exige dos magnitudes positivas, y se han "
            f"pasado {primero} y {segundo}."
        )
    return abs(segundo - primero) / min(primero, segundo)


def measure_floors(
    deployment_rows: Sequence[Mapping[str, str]],
    axes: Sequence[str],
    project_root: Path,
    pairs: Sequence[ControlPair] | None = None,
) -> FloorSet:
    """Mide el suelo de cada eje a partir de los pares de control.

    El suelo entre procesos sale de los pares del mismo día; el de entre sesiones, del máximo de
    todos los pares limpios. A map_50_95 se le añade el suelo entre evaluadores; map_50 no tiene
    suelo.
    """
    pares = tuple(default_control_pairs()) if pairs is None else tuple(pairs)

    observaciones: dict[str, dict[str, float]] = {}
    descartados: dict[str, str] = {}
    por_tipo: dict[str, dict[str, list[float]]] = {"entre_procesos": {}, "entre_sesiones": {}}

    for par in pares:
        for rel in par.evidence:
            if not (project_root / rel).exists():
                raise FloorMeasurementError(
                    f"El par de control '{par.name}' depende de {rel}, que no está en disco."
                )

        primera, fecha_1 = _member_values(par.members[0], deployment_rows, axes, project_root)
        segunda, fecha_2 = _member_values(par.members[1], deployment_rows, axes, project_root)
        _check_pair_kind(par, fecha_1, fecha_2)
        comunes = sorted(set(primera) & set(segunda))

        observaciones[par.name] = {
            eje: relative_difference(primera[eje], segunda[eje]) for eje in comunes
        }

        if par.contaminated:
            descartados[par.name] = par.contamination_reason
            logger.info(
                "Par de control '%s' observado pero no usado para fijar suelo: %s",
                par.name,
                par.contamination_reason,
            )
            continue

        for eje in comunes:
            por_tipo[par.kind].setdefault(eje, []).append(observaciones[par.name][eje])

    suelos: dict[str, AxisFloor] = {}
    for eje in axes:
        procesos = por_tipo["entre_procesos"].get(eje, [])
        sesiones = por_tipo["entre_sesiones"].get(eje, [])
        if not procesos and not sesiones:
            continue

        misma = max(procesos) if procesos else None
        # El error entre sesiones no puede ser menor que el de un mismo proceso.
        candidatos_cruce = list(sesiones) + list(procesos)
        cruce = max(candidatos_cruce) if candidatos_cruce else None

        origen = []
        if procesos:
            origen.append("%d par(es) entre procesos" % len(procesos))
        if sesiones:
            origen.append("%d par(es) entre sesiones" % len(sesiones))

        suelos[eje] = AxisFloor(
            axis=eje,
            kind="relativo",
            same_session=misma,
            cross_session=cruce,
            provenance="medido de " + " y ".join(origen) + " (D-095, D-126, D-127, D-136)",
            caveat=(
                "Dos observaciones por par, no una dispersión: es una cota "
                "inferior del ruido. Superarla no prueba que una diferencia sea "
                "real (D-110). El suelo entre sesiones sale de contrastar la "
                "remedida de D-136 contra el archivo previo a ella, así que mezcla el "
                "error de medida con el efecto del régimen: es una cota alta, no "
                "una estimación."
            ),
        )

    if "map_50_95" in axes:
        suelos["map_50_95"] = AxisFloor(
            axis="map_50_95",
            kind="absoluto",
            same_session=EVALUATOR_READABILITY_FLOOR,
            cross_session=EVALUATOR_READABILITY_FLOOR,
            provenance=(
                "D-118: distancia entre los dos evaluadores sobre el mismo "
                "checkpoint. No depende de la sesión de medida, porque no es "
                "una medida de latencia."
            ),
            caveat=(
                "Es la variación del evaluador, no la de la semilla de "
                "entrenamiento, que este estudio no tiene medida (D-110)."
            ),
        )

    if "map_50" in axes:
        suelos["map_50"] = AxisFloor(
            axis="map_50",
            kind="absoluto",
            same_session=None,
            cross_session=None,
            provenance="sin medir",
            caveat=(
                "El 0,004579 de D-118 es sobre mAP@50-95. Trasladarlo a mAP@50 "
                "sería una analogía, no una cota, y D-127 lo rechaza "
                "explícitamente. Mientras no se mida, ninguna separación de "
                "mAP@50 puede declararse significativa."
            ),
        )

    sin_suelo = [eje for eje in axes if eje not in suelos]
    if sin_suelo:
        logger.info(
            "Ejes sin suelo medido, sobre los que no se podrá certificar ninguna " "ventaja: %s",
            ", ".join(sorted(sin_suelo)),
        )

    return FloorSet(floors=suelos, pair_observations=observaciones, skipped_pairs=descartados)
