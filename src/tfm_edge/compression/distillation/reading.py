"""Reglas de lectura del piloto de destilación y de la extensión a 50 épocas.

El veredicto se calcula con un umbral fijado antes de ver los resultados y trae la frase que los
datos permiten afirmar. El suelo es la diferencia entre evaluadores sobre el mismo checkpoint de n
(0,330215 frente a 0,325636); entre dispositivos la diferencia es 2,3e-07.
tests/test_distillation_control.py recalcula el suelo desde el registro. Como la pérdida es ``(1 -
alpha)·detección + alpha·destilación``, el control con alpha=0 entrena la detección con el doble de
peso, y esa advertencia acompaña a todos los veredictos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: Diferencia de mAP@50-95 por debajo de la cual dos cifras no se distinguen: 0,330215 - 0,325636
#: sobre el mismo checkpoint de n con los dos evaluadores.
EVALUATOR_READABILITY_FLOOR: float = 0.004579

#: Baseline del que parten piloto y control.
STARTING_BASELINE_ID: str = "n_baseline_default_fab9d45c"


@dataclass
class PilotReading:
    """Veredicto del piloto según la regla fijada de antemano."""

    verdict: str
    statement: str
    delta_vs_control: float
    delta_control_vs_start: float
    floor: float
    readable: bool
    caveat: str
    confound: str

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el veredicto como diccionario serializable."""
        return asdict(self)

    def summary(self) -> str:
        """Resumen legible del veredicto."""
        return (
            f"LECTURA DEL PILOTO (regla fijada antes de medir)\n"
            f"  destilado − control : {self.delta_vs_control:+.6f} mAP@50-95\n"
            f"  control − partida   : {self.delta_control_vs_start:+.6f} mAP@50-95\n"
            f"  suelo de lectura    : {self.floor:.6f}\n"
            f"  veredicto           : {self.verdict}\n"
            f"  SE PUEDE AFIRMAR    : {self.statement}\n"
            f"  Y SIEMPRE           : {self.caveat}\n"
            f"  CONFUSOR DECLARADO  : {self.confound}"
        )


#: Advertencia común: una sola semilla por variante y 10 épocas.
PILOT_CAVEAT: str = (
    "una semilla por variante y 10 épocas: esto decide si merece la pena "
    "extender la campaña, no cuánto rinde la destilación."
)

#: Confusor entre alpha y la escala del gradiente, que acompaña a todos los veredictos.
PILOT_CONFOUND: str = (
    "alpha y la tasa de aprendizaje no son ejes independientes. La pérdida es "
    "(1 − alpha)·detección + alpha·destilación·lote, así que el control (alpha=0) "
    "entrena la detección con el doble de escala de gradiente que el piloto "
    "(alpha=0,5), y el alumno va con SGD, donde escalar un término equivale a "
    "escalar su tasa. La diferencia entre las dos ramas mezcla 'destilar' con 'la "
    "tarea original pesa la mitad', y separarlas pide una rama que esta campaña no "
    "tiene."
)


def read_pilot(
    distilled_map_50_95: float,
    control_map_50_95: float,
    starting_map_50_95: float,
    floor: float = EVALUATOR_READABILITY_FLOOR,
) -> PilotReading:
    """Aplica la regla de lectura del piloto de destilación.

    Las tres cifras deben venir del mismo evaluador. Lanza ValueError si el umbral no es positivo.
    """
    if floor <= 0.0:
        raise ValueError(
            f"El suelo de lectura debe ser positivo y es {floor}. Sin umbral, el "
            "ruido del evaluador se leería como efecto de la técnica."
        )

    delta = distilled_map_50_95 - control_map_50_95
    delta_control = control_map_50_95 - starting_map_50_95

    if delta >= floor:
        return PilotReading(
            verdict="destilar ayuda",
            statement=(
                f"a 10 épocas el alumno destilado supera al control sin destilar en "
                f"{delta:.6f} de mAP@50-95, por encima del suelo de lectura. Es señal "
                f"positiva y sostiene extender la campaña a más épocas."
            ),
            delta_vs_control=delta,
            delta_control_vs_start=delta_control,
            floor=floor,
            readable=True,
            caveat=PILOT_CAVEAT,
            confound=PILOT_CONFOUND,
        )

    if -delta >= floor:
        return PilotReading(
            verdict="destilar perjudica a esta configuración",
            statement=(
                f"a 10 épocas el alumno destilado queda {-delta:.6f} de mAP@50-95 por "
                f"debajo del control. Destilar perjudica CON ESTE alpha Y ESTA TASA, "
                f"que no son ejes independientes (ver el confusor declarado). No "
                f"autoriza a concluir nada sobre destilar en general."
            ),
            delta_vs_control=delta,
            delta_control_vs_start=delta_control,
            floor=floor,
            readable=False,
            caveat=PILOT_CAVEAT,
            confound=PILOT_CONFOUND,
        )

    if delta_control >= floor:
        return PilotReading(
            verdict="las épocas extra ayudaron, destilar no se ve",
            statement=(
                f"destilado y control no se distinguen ({delta:+.6f}, por debajo del "
                f"suelo), y los dos superan al punto de partida en {delta_control:.6f}. "
                f"Lo que ha movido la cifra son las 10 épocas adicionales, no destilar. "
                f"NO se puede concluir que la destilación no funcione: solo que a 10 "
                f"épocas no se ve."
            ),
            delta_vs_control=delta,
            delta_control_vs_start=delta_control,
            floor=floor,
            readable=False,
            caveat=PILOT_CAVEAT,
            confound=PILOT_CONFOUND,
        )

    return PilotReading(
        verdict="10 épocas no bastan para verlo",
        statement=(
            f"las tres cifras caen dentro del suelo de lectura ({delta:+.6f} frente al "
            f"control, {delta_control:+.6f} frente a la partida). La única afirmación "
            f"que sostienen los datos es que 10 épocas no bastan para verlo. NO es "
            f"'la destilación no funciona en este caso': el diseño experimental no "
            f"permite dar esa conclusión."
        ),
        delta_vs_control=delta,
        delta_control_vs_start=delta_control,
        floor=floor,
        readable=False,
        caveat=PILOT_CAVEAT,
        confound=PILOT_CONFOUND,
    )


# Extensión a 50 épocas

#: Épocas del piloto; decide qué regla aplica 20_distill.py.
PILOT_EPOCHS: int = 10

#: Control de 10 épocas (per_branch_v1) con el que se compara el de la extensión. Con alpha=0 la
#: semilla no influye y los cuatro controles v2 dan la misma cifra.
PILOT_CONTROL_ID: str = "n_distillation_l2n_control_alpha0_pilot10_8826269c"

#: Advertencia común a cualquier veredicto de la extensión.
EXTENSION_CAVEAT: str = (
    "una ejecución por brazo. La sensibilidad a la inicialización de los "
    "adaptadores se midió a 10 épocas (D-151) y no a estas; el suelo de lectura "
    "es la diferencia entre dos evaluadores, no la variabilidad entre "
    "entrenamientos. Dice cuánto rinde destilar CON ESTA RECETA (alpha, tasa, "
    "sin mosaico), no cuánto rinde la destilación."
)

#: Advertencia sobre la selección de best.pt, que acompaña a todos los veredictos.
EXTENSION_SELECTION_CAVEAT: str = (
    "cada brazo se evalúa con su best.pt, que Ultralytics elige por la validación "
    "de cada época sobre val, la MISMA partición que luego mide pycocotools. La "
    "cifra de cada brazo es el máximo sobre sus épocas, y a 50 épocas hay cinco "
    "veces más candidatos que a 10. Infla las dos cifras absolutas, y no se "
    "compensa necesariamente en la diferencia: gana más el brazo con la curva de "
    "validación más ruidosa. No está medido; ninguna cifra absoluta de esta "
    "campaña se compara con otra técnica sin medirlo antes (última época, o test)."
)


@dataclass
class ExtensionReading:
    """Veredicto de una variante de la extensión frente a su control.

    Usa los mismos cuatro veredictos que el piloto y añade, sin que decida el veredicto, cómo se
    compara el control con el de 10 épocas.
    """

    epochs: int
    verdict: str
    statement: str
    delta_vs_control: float
    delta_control_vs_start: float
    delta_distilled_vs_start: float
    delta_control_vs_pilot_control: float
    control_state: str
    control_statement: str
    floor: float
    readable: bool
    continue_campaign: bool
    caveat: str
    confound: str
    selection: str

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el veredicto como diccionario serializable."""
        return asdict(self)

    def summary(self) -> str:
        """Resumen legible del veredicto."""
        decision = (
            "se puede seguir con las ablaciones"
            if self.continue_campaign
            else "SE PARA y se decide con el autor antes de gastar otra sesión"
        )
        return (
            f"LECTURA DE LA EXTENSIÓN A {self.epochs} ÉPOCAS (regla fijada antes de medir)\n"
            f"  destilado − control          : {self.delta_vs_control:+.6f} mAP@50-95\n"
            f"  control − partida            : {self.delta_control_vs_start:+.6f} mAP@50-95\n"
            f"  destilado − partida          : {self.delta_distilled_vs_start:+.6f} "
            f"mAP@50-95 (contexto, no es la comparación del 5.4)\n"
            f"  control − control a {PILOT_EPOCHS} ép.  : "
            f"{self.delta_control_vs_pilot_control:+.6f} mAP@50-95\n"
            f"  suelo de lectura             : {self.floor:.6f}\n"
            f"  veredicto                    : {self.verdict}\n"
            f"  SE PUEDE AFIRMAR             : {self.statement}\n"
            f"  EL CONTROL                   : {self.control_state}. {self.control_statement}\n"
            f"  PUNTO DE DECISIÓN            : {decision} (solo aplica tras kd_combined)\n"
            f"  Y SIEMPRE                    : {self.caveat}\n"
            f"  CONFUSOR DECLARADO           : {self.confound}\n"
            f"  SELECCIÓN DE best.pt         : {self.selection}"
        )


def _estado_del_control(
    delta_control_vs_pilot_control: float, epochs: int, floor: float
) -> tuple[str, str]:
    """Clasifica el control de la extensión frente al de 10 épocas. Devuelve ``(etiqueta,
    significado)``.
    """
    extra = epochs - PILOT_EPOCHS
    if delta_control_vs_pilot_control >= floor:
        return (
            "mejora con las épocas extra",
            f"El afinado sin destilar gana {delta_control_vs_pilot_control:.6f} con "
            f"{extra} épocas más. Es lo esperable, y es lo que hace que la comparación "
            f"con el control, y no con el piloto de 10, sea la única que mide destilar.",
        )
    if -delta_control_vs_pilot_control >= floor:
        return (
            "EMPEORA con las épocas extra",
            f"El control queda {-delta_control_vs_pilot_control:.6f} por debajo del de "
            f"{PILOT_EPOCHS} épocas. Alargar la receta no es solo añadir épocas: la tasa "
            f"decae a lo largo de {epochs} y no de {PILOT_EPOCHS}, y el afinado sin "
            f"mosaico dura {epochs}. La comparación destilado − control sigue siendo "
            f"válida, porque comparten receta, pero ninguna de las dos cifras se puede "
            f"presentar como 'el efecto de entrenar más' frente a la campaña de "
            f"{PILOT_EPOCHS} épocas, y el punto de decisión se para para mirarlo.",
        )
    return (
        "no se mueve con las épocas extra",
        f"El control no se distingue del de {PILOT_EPOCHS} épocas "
        f"({delta_control_vs_pilot_control:+.6f}): {extra} épocas más de afinado sin "
        f"destilar no cambian la cifra. Cualquier diferencia del destilado con el "
        f"control es atribuible a destilar dentro de esta receta, no a la duración.",
    )


def read_extension(
    distilled_map_50_95: float,
    control_map_50_95: float,
    starting_map_50_95: float,
    pilot_control_map_50_95: float,
    epochs: int,
    floor: float = EVALUATOR_READABILITY_FLOOR,
) -> ExtensionReading:
    """Aplica la regla de lectura de la extensión a 50 épocas.

    Igual que read_pilot, pero si el control queda por debajo de la partida lo indica, y el estado
    del control se imprime siempre. Falla si ``epochs`` no supera las del piloto.
    """
    if floor <= 0.0:
        raise ValueError(
            f"El suelo de lectura debe ser positivo y es {floor}. Sin umbral, el "
            "ruido del evaluador se leería como efecto de la técnica."
        )
    if epochs <= PILOT_EPOCHS:
        raise ValueError(
            f"Esta regla es para más de {PILOT_EPOCHS} épocas y la ejecución tiene {epochs}; usa "
            "read_pilot."
        )

    delta = distilled_map_50_95 - control_map_50_95
    delta_control = control_map_50_95 - starting_map_50_95
    delta_distilled = distilled_map_50_95 - starting_map_50_95
    delta_pilot = control_map_50_95 - pilot_control_map_50_95
    estado, significado = _estado_del_control(delta_pilot, epochs, floor)
    control_empeora = -delta_pilot >= floor

    comunes: dict[str, Any] = {
        "epochs": epochs,
        "delta_vs_control": delta,
        "delta_control_vs_start": delta_control,
        "delta_distilled_vs_start": delta_distilled,
        "delta_control_vs_pilot_control": delta_pilot,
        "control_state": estado,
        "control_statement": significado,
        "floor": floor,
        "caveat": EXTENSION_CAVEAT,
        "confound": PILOT_CONFOUND,
        "selection": EXTENSION_SELECTION_CAVEAT,
    }

    if delta >= floor:
        return ExtensionReading(
            verdict="destilar ayuda",
            statement=(
                f"a {epochs} épocas esta variante supera al control sin destilar en "
                f"{delta:.6f} de mAP@50-95, por encima del suelo de lectura. Es un "
                f"efecto de destilar dentro de esta receta, con una ejecución por brazo."
            ),
            readable=True,
            continue_campaign=not control_empeora,
            **comunes,
        )

    if -delta >= floor:
        return ExtensionReading(
            verdict="destilar perjudica a esta configuración",
            statement=(
                f"a {epochs} épocas esta variante queda {-delta:.6f} de mAP@50-95 por "
                f"debajo del control. Destilar perjudica CON ESTE alpha Y ESTA TASA, que "
                f"no son ejes independientes (ver el confusor declarado). No autoriza a "
                f"concluir nada sobre destilar en general."
            ),
            readable=False,
            continue_campaign=False,
            **comunes,
        )

    if delta_control >= floor:
        return ExtensionReading(
            verdict="las épocas extra ayudaron, destilar no se ve",
            statement=(
                f"destilado y control no se distinguen ({delta:+.6f}, por debajo del "
                f"suelo), y los dos superan al punto de partida en {delta_control:.6f}. "
                f"Lo que mueve la cifra es el afinado de {epochs} épocas, no destilar: "
                f"la ventaja que se leía a {PILOT_EPOCHS} épocas no se sostiene a "
                f"{epochs} con esta receta. NO se puede concluir que la destilación no "
                f"funcione."
            ),
            readable=False,
            continue_campaign=False,
            **comunes,
        )

    if -delta_control >= floor:
        situacion = (
            f"destilado y control no se distinguen ({delta:+.6f}), y los dos quedan por "
            f"debajo del punto de partida (el control, {-delta_control:.6f}). El "
            f"afinado de {epochs} épocas con esta receta pierde calidad frente a la "
            f"partida, y destilar no lo compensa de forma distinguible"
        )
    else:
        situacion = (
            f"las tres cifras caen dentro del suelo de lectura ({delta:+.6f} frente al "
            f"control, {delta_control:+.6f} frente a la partida)"
        )
    return ExtensionReading(
        verdict=f"{epochs} épocas no bastan para verlo",
        statement=(
            f"{situacion}. La única afirmación que sostienen los datos es que a "
            f"{epochs} épocas, con esta receta, destilar no se distingue de no "
            f"destilar. NO es 'la destilación no funciona en este caso': el diseño "
            f"no permite dar esa conclusión."
        ),
        readable=False,
        continue_campaign=False,
        **comunes,
    )
