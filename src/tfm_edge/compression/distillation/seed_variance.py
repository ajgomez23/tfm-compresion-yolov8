"""Lectura de la campaña de semillas del piloto de destilación.

En esta receta la semilla no llega al cargador de datos: solo cambia la inicialización de los
adaptadores. Lo que se mide es la sensibilidad del efecto a esa inicialización con los datos fijos;
los controles con alpha=0 son idénticos y su desviación es cero. La regla se fijó antes de ver
resultados: cada piloto se empareja con el control de su semilla (seed_base del volcado), se usan
diferencias emparejadas con intervalo del 95 % por t de Student (con n=4, t vale 3,182 y no 2), y si
el intervalo cruza el cero se distingue ruido de efecto pequeño con ``sd_delta >= abs(mean_delta)``.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.distillation.reading import EVALUATOR_READABILITY_FLOOR
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR

logger = get_logger(__name__)

#: Prefijo de los volcados de la comprobación previa, que están en el mismo directorio que los
#: resultados pero no son ejecuciones: solo llevan experiment_id y preflight.
PREFLIGHT_DUMP_PREFIX = "preflight_"

#: Valores críticos de la t de Student (dos colas, 95 %) por grados de libertad. Se tabulan para no
#: depender de scipy, que no está entre las dependencias. Fuera de la tabla se falla.
T_CRITICAL_95_TWO_SIDED: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}

#: Mínimo de semillas para calcular una varianza (ddof=1).
MIN_SEEDS = 2

#: Primera predicción, escrita antes de medir: el suelo entre evaluadores sería una cota inferior
#: del ruido entre entrenamientos.
FLOOR_PREDICTION = (
    "reading.py afirma que 0,004579 es cota INFERIOR de la variabilidad entre "
    "entrenamientos. Se cumple si la desviación típica entre las semillas de los "
    "controles alcanza o supera esa cifra."
)

#: Esta campaña no puede contrastar FLOOR_PREDICTION, porque la semilla solo cambia los adaptadores.
FLOOR_PREDICTION_TESTABLE = False

#: Motivo que se muestra en el informe.
FLOOR_PREDICTION_STATUS = (
    "No contrastable con esta campaña (D-151). Con alpha=0 los adaptadores son "
    "inertes y son lo único que la semilla cambia (D-150): los controles son la "
    "misma ejecución, y una desviación típica de cero entre ellos no refuta ni "
    "confirma que 0,004579 sea cota inferior de la variabilidad entre "
    "entrenamientos."
)

#: Segunda predicción: en la semilla 42 el efecto está en la localización (mAP@50-95 sube y mAP@50
#: se queda plano). Si es real, el reparto debería repetirse.
TAIL_PREDICTION = (
    "En la semilla 42 el efecto está entero en la cola de localización: sube "
    "mAP@50-95 y deja plano mAP@50. Se cumple si en todas las semillas "
    "|delta mAP@50| queda por debajo de |delta mAP@50-95|."
)

#: Salvedades comunes a cualquier lectura. La procedencia de cada pareja la calcula
#: provenance_caveats.
CAMPAIGN_CAVEATS: tuple[str, ...] = (
    "Se lee a flujo de datos fijo (D-150, D-151): las parejas recorren los mismos "
    "lotes con la misma augmentación, y la semilla solo cambia la inicialización "
    "de los tres adaptadores. La dispersión viene de esa única fuente de azar, así "
    "que el intervalo es más estrecho que el de una réplica con entrenamientos "
    "independientes, y nada obliga a que la dispersión que falta sea pequeña.",
    "El confusor de D-113 sigue en pie: entre el piloto y el control cambian a la "
    "vez alpha y la tasa de aprendizaje efectiva, así que estas semillas dicen si "
    "el efecto se repite, no de cuál de las dos cosas viene.",
    "Diez épocas no son la configuración del capítulo 5.4: son el piloto que "
    "decide si merece la pena pagar cincuenta. Y con close_mosaic=10 heredado no "
    "sacan un solo lote con mosaico (D-142), así que la receta no es la de los "
    "baselines con los que se compara el punto de partida.",
)

#: Veredictos posibles, fijados antes de medir.
VERDICT_HELPS = "destilar ayuda, y el efecto se repite entre semillas"
VERDICT_HURTS = "destilar perjudica a esta configuración"
VERDICT_NOISE = "el efecto de una sola tirada era ruido de la inicialización del adaptador"
VERDICT_TOO_SMALL = "a lo sumo un efecto menor de lo que estas semillas resuelven"


class SeedCampaignError(TFMEdgeError):
    """La campaña de semillas no se puede leer tal y como está en disco."""

    def __init__(self, problem: str, remedy: str) -> None:
        super().__init__(problem, remedy)


@dataclass(frozen=True)
class DistillationRun:
    """Ejecución de destilación leída de su volcado crudo.

    Las métricas se leen del volcado y no del CSV, que redondea a seis decimales.
    """

    experiment_id: str
    seed_base: int
    seed_branch: int
    seeding_scheme: str
    is_control: bool
    alpha: float
    map_50: float
    map_50_95: float
    config_path: str
    training_source: str
    preflight_recorded: bool
    dump_path: Path
    environment_id: str = ""
    #: Norma de cada adaptador al crearse, como pares ``(módulo, norma)``. Es lo único que cambia la
    #: semilla, y sirve para comprobar que llega.
    adapter_initial_norms: tuple[tuple[str, float], ...] = ()

    @property
    def label(self) -> str:
        """Etiqueta ``control`` o ``piloto`` con su semilla."""
        papel = "control" if self.is_control else "piloto"
        return f"{papel} de la semilla {self.seed_base}"


@dataclass(frozen=True)
class SeedPair:
    """Un piloto y el control de su misma semilla."""

    seed: int
    pilot: DistillationRun
    control: DistillationRun

    @property
    def delta_50_95(self) -> float:
        """Diferencia en mAP@50-95, piloto menos control."""
        return self.pilot.map_50_95 - self.control.map_50_95

    @property
    def delta_50(self) -> float:
        """Diferencia en mAP@50, piloto menos control."""
        return self.pilot.map_50 - self.control.map_50

    @property
    def same_environment(self) -> bool:
        """True solo si los dos environment_id constan y coinciden."""
        return bool(self.pilot.environment_id) and (
            self.pilot.environment_id == self.control.environment_id
        )

    @property
    def environment_state(self) -> str:
        """Estado del entorno de la pareja: ``mismo``, ``DISTINTO`` o ``sin dato``."""
        if not self.pilot.environment_id or not self.control.environment_id:
            return "sin dato"
        return "mismo" if self.same_environment else "DISTINTO"

    @property
    def effect_is_in_the_tail(self) -> bool:
        """True si la diferencia en mAP@50 es menor en magnitud que en mAP@50-95."""
        return abs(self.delta_50) < abs(self.delta_50_95)


@dataclass(frozen=True)
class SeedCampaignReading:
    """Resultado de la campaña de semillas según la regla fijada."""

    verdict: str
    statement: str
    n_seeds: int
    seeds: tuple[int, ...]
    mean_delta: float
    sd_delta: float
    t_critical: float
    ci_low: float
    ci_high: float
    control_mean: float
    control_sd: float
    process_floor: float
    floor_is_lower_bound: bool
    floor_prediction: str
    tail_prediction: str
    tail_prediction_holds: bool
    all_pairs_same_environment: bool
    caveats: tuple[str, ...]
    controls_identical: bool = False
    floor_prediction_testable: bool = FLOOR_PREDICTION_TESTABLE
    floor_prediction_status: str = FLOOR_PREDICTION_STATUS

    @property
    def ci_crosses_zero(self) -> bool:
        """True si el intervalo de confianza incluye el cero."""
        return self.ci_low <= 0.0 <= self.ci_high

    def to_dict(self) -> dict[str, Any]:
        """Serializa la lectura para el volcado crudo."""
        return {
            "verdict": self.verdict,
            "statement": self.statement,
            "n_seeds": self.n_seeds,
            "seeds": list(self.seeds),
            "mean_delta": self.mean_delta,
            "sd_delta": self.sd_delta,
            "t_critical": self.t_critical,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_crosses_zero": self.ci_crosses_zero,
            "control_mean": self.control_mean,
            "control_sd": self.control_sd,
            "controls_identical": self.controls_identical,
            "process_floor": self.process_floor,
            "floor_is_lower_bound": self.floor_is_lower_bound,
            "floor_prediction_testable": self.floor_prediction_testable,
            "floor_prediction_status": self.floor_prediction_status,
            "floor_prediction": self.floor_prediction,
            "tail_prediction": self.tail_prediction,
            "tail_prediction_holds": self.tail_prediction_holds,
            "all_pairs_same_environment": self.all_pairs_same_environment,
            "caveats": list(self.caveats),
        }


def load_distillation_runs(raw_dir: Path) -> list[DistillationRun]:
    """Lee las ejecuciones de destilación de los volcados crudos, descartando los de la comprobación
    previa.
    """
    if not raw_dir.is_dir():
        raise SeedCampaignError(
            f"No existe el directorio de volcados '{raw_dir}'.",
            "Comprueba --raw-dir, o trae los volcados de Drive si la campaña " "corrió en Colab.",
        )

    ejecuciones: list[DistillationRun] = []
    for ruta in sorted(raw_dir.glob("*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        if ruta.name.startswith(PREFLIGHT_DUMP_PREFIX):
            # Se reconoce por su nombre y su forma, no por lo que le falta, para no ocultar un
            # volcado truncado.
            if "preflight" in datos and "metrics" not in datos:
                logger.debug("Volcado de comprobación previa, omitido: %s", ruta.name)
                continue
            raise SeedCampaignError(
                f"El volcado '{ruta.name}' se llama como una comprobación previa "
                f"pero no tiene su forma: trae {sorted(datos)}.",
                "O el nombre está mal, o 20_distill.py ha cambiado lo que vuelca "
                "y este filtro se ha quedado atrás. No se adivina cuál de las dos.",
            )
        faltan = [
            clave
            for clave in ("experiment_id", "seed_base", "is_control", "metrics")
            if clave not in datos
        ]
        if faltan:
            raise SeedCampaignError(
                f"El volcado '{ruta.name}' no trae {faltan}, así que no se puede "
                f"saber de qué ejecución es.",
                "Un volcado sin semilla está truncado o es de un formato antiguo. Sácalo del "
                "directorio o regenéralo.",
            )
        metricas = datos["metrics"] or {}
        if "map_50_95" not in metricas or "map_50" not in metricas:
            # Una ejecución que falló antes de evaluar no tiene métricas y no entra en el análisis.
            continue
        ejecuciones.append(
            DistillationRun(
                experiment_id=datos["experiment_id"],
                seed_base=int(datos["seed_base"]),
                seed_branch=int(datos.get("seed_branch", 0)),
                seeding_scheme=str(datos.get("seeding_scheme", "")),
                is_control=bool(datos["is_control"]),
                alpha=float(datos.get("effective_config", {}).get("alpha", float("nan"))),
                map_50=float(metricas["map_50"]),
                map_50_95=float(metricas["map_50_95"]),
                config_path=str(datos.get("config_path", "")),
                training_source=str(datos.get("training_source", "")),
                preflight_recorded=datos.get("preflight") is not None,
                dump_path=ruta,
                adapter_initial_norms=tuple(
                    sorted(
                        (str(modulo), float(norma))
                        for modulo, norma in (datos.get("adapter_initial_norms") or {}).items()
                    )
                ),
            )
        )
    return sorted(ejecuciones, key=lambda e: (e.seed_base, e.is_control))


def attach_environment_ids(
    runs: Sequence[DistillationRun],
    quality_rows: Iterable[Mapping[str, str]],
    *,
    evaluator: str = CANONICAL_EVALUATOR,
    tolerance: float = 1e-6,
) -> list[DistillationRun]:
    """Añade a cada ejecución el environment_id de su fila de calidad.

    También comprueba que el registro y el volcado dan la misma métrica.
    """
    por_id: dict[str, Mapping[str, str]] = {}
    for fila in quality_rows:
        if (
            fila.get("evaluator") != evaluator
            or fila.get("runtime") != "pytorch"
            or fila.get("status") != "ok"
            or not fila.get("map_50_95")
        ):
            continue
        clave = fila["experiment_id"]
        anterior = por_id.get(clave)
        if anterior is not None and anterior.get("map_50_95") != fila.get("map_50_95"):
            raise SeedCampaignError(
                f"El registro tiene dos filas de '{clave}' con el mismo evaluador y métricas "
                f"distintas ({anterior['map_50_95']} y {fila['map_50_95']}).",
                "Averigua cuál sobra antes de continuar.",
            )
        por_id[clave] = fila

    completadas: list[DistillationRun] = []
    for ejecucion in runs:
        fila = por_id.get(ejecucion.experiment_id)
        if fila is None:
            completadas.append(ejecucion)
            continue
        del_registro = float(fila["map_50_95"])
        if abs(del_registro - ejecucion.map_50_95) > tolerance:
            raise SeedCampaignError(
                f"'{ejecucion.experiment_id}' dice {ejecucion.map_50_95:.6f} en su "
                f"volcado y {del_registro:.6f} en el registro.",
                "Son dos escrituras del mismo número y no coinciden: uno de los "
                "dos se ha reescrito. Averigua cuál antes de usar ninguno.",
            )
        completadas.append(replace(ejecucion, environment_id=fila.get("environment_id", "")))
    return completadas


def pair_by_seed(runs: Sequence[DistillationRun]) -> list[SeedPair]:
    """Empareja cada piloto con el control de su misma semilla.

    Falla si una semilla tiene más de un piloto o más de un control.
    """
    por_semilla: dict[int, dict[str, list[DistillationRun]]] = {}
    for ejecucion in runs:
        papel = "control" if ejecucion.is_control else "pilot"
        por_semilla.setdefault(ejecucion.seed_base, {"pilot": [], "control": []})
        por_semilla[ejecucion.seed_base][papel].append(ejecucion)

    parejas: list[SeedPair] = []
    for semilla in sorted(por_semilla):
        grupos = por_semilla[semilla]
        for papel, ejecuciones in grupos.items():
            identificadores = {e.experiment_id for e in ejecuciones}
            if len(identificadores) > 1:
                detalle = "\n".join(f"  - {i}" for i in sorted(identificadores))
                raise SeedCampaignError(
                    f"La semilla {semilla} tiene {len(identificadores)} ejecuciones "
                    f"en el papel de {papel}:\n{detalle}",
                    "Dos ejecuciones distintas con la misma semilla y el mismo papel. Aparta la "
                    "que sobre del directorio de volcados.",
                )
        if not grupos["pilot"] or not grupos["control"]:
            # Una semilla incompleta se omite; el script lo indica.
            continue
        parejas.append(
            SeedPair(seed=semilla, pilot=grupos["pilot"][0], control=grupos["control"][0])
        )
    return parejas


def find_run(runs: Sequence[DistillationRun], *, seed: int, is_control: bool) -> DistillationRun:
    """Busca la ejecución de una semilla y un papel. La usa el cuaderno de Colab para pasar
    ``--control-experiment-id`` al piloto.
    """
    papel = "control" if is_control else "piloto"
    candidatas = [e for e in runs if e.seed_base == seed and e.is_control == is_control]
    identificadores = {e.experiment_id for e in candidatas}
    if not identificadores:
        vistas = sorted({(e.seed_base, "control" if e.is_control else "piloto") for e in runs})
        raise SeedCampaignError(
            f"No hay ningún {papel} de la semilla {seed} en los volcados. "
            f"Lo que hay: {vistas}.",
            "Comprueba que la ejecución terminó e inscribió su volcado, y que "
            "--raw-dir apunta al directorio donde está.",
        )
    if len(identificadores) > 1:
        detalle = "\n".join(f"  - {i}" for i in sorted(identificadores))
        raise SeedCampaignError(
            f"Hay {len(identificadores)} ejecuciones distintas en el papel de "
            f"{papel} para la semilla {seed}:\n{detalle}",
            "No se elige una: dos ejecuciones con la misma semilla y el mismo "
            "papel dicen que la campaña se lanzó dos veces. Aparta la que sobre.",
        )
    return candidatas[0]


def assert_dispersion_is_not_degenerate(pairs: Sequence[SeedPair]) -> None:
    """Rechaza una campaña cuyas semillas no se distinguen entre sí.

    Con dispersión nula el intervalo colapsa y saldría el veredicto más favorable sin información.
    Comprueba que los pilotos no comparten métrica, que cada semilla da una inicialización de
    adaptadores distinta y que la dispersión de las diferencias no es cero. La métrica de los
    controles no se compara porque con alpha=0 es siempre idéntica.
    """
    por_metrica: dict[float, list[int]] = {}
    for pareja in pairs:
        por_metrica.setdefault(pareja.pilot.map_50_95, []).append(pareja.seed)
    repetidas = {m: s for m, s in por_metrica.items() if len(s) > 1}
    if repetidas:
        detalle = "\n".join(
            f"  - mAP@50-95 = {metrica!r} en las semillas {sorted(semillas)}"
            for metrica, semillas in sorted(repetidas.items())
        )
        raise SeedCampaignError(
            f"Dos semillas distintas dan la MISMA métrica en los pilotos, "
            f"hasta el último decimal:\n{detalle}",
            "Los entrenamientos han producido pesos idénticos. Comprueba que 'seed' llega en "
            "los overrides del Trainer.",
        )

    for papel, ejecuciones in (
        ("pilotos", [(pareja.seed, pareja.pilot) for pareja in pairs]),
        ("controles", [(pareja.seed, pareja.control) for pareja in pairs]),
    ):
        sin_dato = [e.label for _, e in ejecuciones if not e.adapter_initial_norms]
        if sin_dato:
            raise SeedCampaignError(
                f"Los volcados de {sin_dato} no traen 'adapter_initial_norms'.",
                "Sin la inicialización de los adaptadores no se puede comprobar que la semilla "
                "llega. El volcado es anterior a per_branch_v2 o está truncado.",
            )
        por_inicializacion: dict[tuple[tuple[str, float], ...], list[int]] = {}
        for semilla, ejecucion in ejecuciones:
            por_inicializacion.setdefault(ejecucion.adapter_initial_norms, []).append(semilla)
        compartidas = {i: s for i, s in por_inicializacion.items() if len(s) > 1}
        if compartidas:
            detalle = "\n".join(
                f"  - {dict(inicializacion)} en las semillas {sorted(semillas)}"
                for inicializacion, semillas in compartidas.items()
            )
            raise SeedCampaignError(
                f"Dos semillas distintas dan la MISMA inicialización de adaptador "
                f"en los {papel}:\n{detalle}",
                "La semilla no cambia la inicialización de los adaptadores. Comprueba que "
                "'seed' llega al Trainer.",
            )

    diferencias = [pareja.delta_50_95 for pareja in pairs]
    if statistics.stdev(diferencias) == 0.0:
        raise SeedCampaignError(
            "La dispersión de las diferencias entre semillas es exactamente cero.",
            "Con dispersión cero el intervalo colapsaría a un punto. Revisa que cada semilla "
            "entrene de forma distinta.",
        )


def read_seed_campaign(
    pairs: Sequence[SeedPair], *, floor: float = EVALUATOR_READABILITY_FLOOR
) -> SeedCampaignReading:
    """Aplica la regla de lectura fijada antes de ver los resultados.

    ``floor`` es el suelo entre evaluadores; se informa junto a FLOOR_PREDICTION_STATUS.
    """
    if len(pairs) < MIN_SEEDS:
        raise SeedCampaignError(
            f"Hacen falta al menos {MIN_SEEDS} semillas para que haya varianza "
            f"que medir, y hay {len(pairs)}.",
            "Con una sola semilla se usa la regla de compression/distillation/reading.py.",
        )

    # Una campaña degenerada se rechaza antes de calcular nada.
    assert_dispersion_is_not_degenerate(pairs)

    grados = len(pairs) - 1
    if grados not in T_CRITICAL_95_TWO_SIDED:
        raise SeedCampaignError(
            f"No hay valor crítico tabulado para {grados} grados de libertad.",
            "Amplía T_CRITICAL_95_TWO_SIDED con el valor de una tabla, o cambia el "
            "estadístico. No se extrapola.",
        )
    t_critico = T_CRITICAL_95_TWO_SIDED[grados]

    diferencias = [pareja.delta_50_95 for pareja in pairs]
    media = statistics.fmean(diferencias)
    desviacion = statistics.stdev(diferencias)
    margen = t_critico * desviacion / len(pairs) ** 0.5
    ci_bajo, ci_alto = media - margen, media + margen

    controles = [pareja.control.map_50_95 for pareja in pairs]
    control_media = statistics.fmean(controles)
    control_desviacion = statistics.stdev(controles)

    cruza_el_cero = ci_bajo <= 0.0 <= ci_alto
    if not cruza_el_cero and media > 0:
        veredicto = VERDICT_HELPS
        frase = (
            f"Con {len(pairs)} semillas, destilar sube el mAP@50-95 en "
            f"{media:+.6f} de media (IC 95 %: {ci_bajo:+.6f} a {ci_alto:+.6f}), y "
            f"el intervalo no cruza el cero: el efecto se repite entre "
            f"inicializaciones del adaptador, a flujo de datos fijo (D-151)."
        )
    elif not cruza_el_cero:
        veredicto = VERDICT_HURTS
        frase = (
            f"Con {len(pairs)} semillas, destilar BAJA el mAP@50-95 en "
            f"{media:+.6f} de media (IC 95 %: {ci_bajo:+.6f} a {ci_alto:+.6f}). El "
            f"intervalo no cruza el cero: con esta alpha y estas épocas, destilar "
            f"perjudica."
        )
    elif desviacion >= abs(media):
        veredicto = VERDICT_NOISE
        frase = (
            f"La diferencia media es {media:+.6f} y la dispersión entre semillas "
            f"({desviacion:.6f}) la alcanza. El intervalo cruza el cero: lo que "
            f"midió una sola tirada era la variabilidad que introduce la "
            f"inicialización del adaptador, no el efecto de destilar (D-151)."
        )
    else:
        veredicto = VERDICT_TOO_SMALL
        frase = (
            f"La diferencia media es {media:+.6f}, consistente entre semillas "
            f"(dispersión {desviacion:.6f}), pero el intervalo cruza el cero "
            f"({ci_bajo:+.6f} a {ci_alto:+.6f}). Hay a lo sumo un efecto menor de "
            f"lo que {len(pairs)} semillas resuelven. NO es lo mismo que 'no hay "
            f"efecto': es que harían falta más semillas para verlo."
        )

    return SeedCampaignReading(
        verdict=veredicto,
        statement=frase,
        n_seeds=len(pairs),
        seeds=tuple(pareja.seed for pareja in pairs),
        mean_delta=media,
        sd_delta=desviacion,
        t_critical=t_critico,
        ci_low=ci_bajo,
        ci_high=ci_alto,
        control_mean=control_media,
        control_sd=control_desviacion,
        process_floor=floor,
        floor_is_lower_bound=control_desviacion >= floor,
        floor_prediction=FLOOR_PREDICTION,
        tail_prediction=TAIL_PREDICTION,
        tail_prediction_holds=all(pareja.effect_is_in_the_tail for pareja in pairs),
        all_pairs_same_environment=all(pareja.same_environment for pareja in pairs),
        caveats=CAMPAIGN_CAVEATS + provenance_caveats(pairs),
        controls_identical=len(set(controles)) == 1,
    )


def provenance_caveats(pairs: Sequence[SeedPair]) -> tuple[str, ...]:
    """Salvedades de procedencia de cada pareja: ramas reutilizadas con --reuse-run o sin
    comprobación previa. Vacío si todas son limpias.
    """
    avisos: list[str] = []
    for pareja in pairs:
        problemas = [
            f"el {ejecucion.label} "
            + (
                "se completó con --reuse-run"
                if "reuse-run" in ejecucion.training_source
                else "no trae comprobación previa"
            )
            for ejecucion in (pareja.pilot, pareja.control)
            if "reuse-run" in ejecucion.training_source or not ejecucion.preflight_recorded
        ]
        if problemas:
            avisos.append(
                f"La semilla {pareja.seed} no es una pareja limpia: "
                + "; ".join(problemas)
                + ". No consta que sus dos ramas se entrenaran en la misma sesión."
            )
    return tuple(avisos)
