"""Textos de la columna ``notes`` de quality_metrics.

Empiezan por ``evaluator=`` e indican si la cifra es comparable con el baseline. Se comparten entre
los scripts que inscriben filas para que el texto no diverja.
"""

from __future__ import annotations

import re

#: Prefijo ``evaluator=`` de notes. El evaluador tiene columna propia; esto sirve para leer las
#: filas antiguas que solo lo traen en la nota.
_EVALUATOR_RE = re.compile(r"^evaluator=([A-Za-z_][A-Za-z0-9_]*)")

#: Evaluador de la ruta canónica. Va a la columna evaluator y al prefijo de notes.
CANONICAL_EVALUATOR = "pycocotools"

#: Evaluador de Ultralytics (paso 05). No da las mismas cifras que pycocotools, por eso cada fila
#: indica el suyo.
ULTRALYTICS_EVALUATOR = "ultralytics"

#: Prefijo obligatorio de toda fila de calidad.
EVALUATOR_PREFIX = f"evaluator={CANONICAL_EVALUATOR}."

#: Nota de la rama QAT: se miden los pesos afinados en FP32, no el modelo INT8.
QAT_FP32_NOTE = (
    "PESOS AFINADOS CON QAT Y MEDIDOS EN FP32, con la simulación de cuantización "
    "desactivada: esta cifra dice lo que cuesta entrenar con el ruido de "
    "cuantización puesto, NO lo que rinde el INT8. El INT8 de esta rama lo "
    "materializa cada runtime (D-018) y se mide en el paso 11."
)

#: Nota del control sin destilar (alpha=0). Recorre el mismo camino que la destilación con el
#: término de destilación a cero, así que equivale a un afinado normal.
#:
#: Marca que identifica al control en el CSV. La busca scripts/20_distill.py::buscar_control.
DISTILLATION_CONTROL_MARKER = "CONTROL SIN DESTILAR"

DISTILLATION_CONTROL_NOTE = (
    f"{DISTILLATION_CONTROL_MARKER} (alpha=0): esta cifra NO es el resultado de destilar. "
    "Es el mismo alumno, el mismo camino y las mismas épocas que la variante "
    "destilada, con el término de destilación multiplicado por cero; existe para "
    "que la diferencia con ella sea atribuible a destilar y no a las épocas "
    "extra ni a la tasa de aprendizaje. No la uses como resultado de la técnica."
)

#: Nota de una destilación real: el alumno no parte de cero, así que se compara con el control.
DISTILLATION_NOTE = (
    "Alumno destilado a partir del baseline ya entrenado, no desde cero. La "
    "comparación que aísla el efecto de destilar es contra el CONTROL alpha=0 "
    "de la misma campaña; contra el baseline de partida mezcla destilación, "
    "épocas adicionales y cambio de tasa (D-110)."
)

#: Notas que se deducen de la técnica. Las que no, en TECHNIQUES_REQUIRING_EXPLICIT_NOTE.
TECHNIQUE_NOTES: dict[str, str] = {
    "qat": QAT_FP32_NOTE,
}


#: Prefijo de la semilla en notes. El registro no tiene columna de semilla; la fuente es seed_base
#: en el volcado crudo y esto solo ayuda a leer el CSV.
DISTILLATION_SEED_PREFIX = "seed_base="


def build_distillation_note(
    alpha: float, *, seed: int | None = None, provenance: str = "", extra: str = ""
) -> str:
    """Compone la nota de una fila de destilación a partir de ``alpha``. Con ``alpha=0`` marca el
    control.

    ``seed=None`` omite la marca de semilla.
    """
    cuerpo = DISTILLATION_CONTROL_NOTE if alpha == 0.0 else DISTILLATION_NOTE
    marca = f"{DISTILLATION_SEED_PREFIX}{seed}." if seed is not None else ""
    partes = [EVALUATOR_PREFIX, cuerpo, marca, provenance, extra]
    return " ".join(parte.strip() for parte in partes if parte and parte.strip())


def read_seed_from_note(notes: str) -> int | None:
    """Extrae la semilla base de una nota, o None si no la lleva."""
    coincidencia = re.search(rf"{re.escape(DISTILLATION_SEED_PREFIX)}(-?\d+)", notes or "")
    return int(coincidencia.group(1)) if coincidencia else None


#: Técnicas que necesitan nota explícita. Un modelo podado sin reajustar no es comparable con el
#: baseline y eso no se ve en el checkpoint.
TECHNIQUES_REQUIRING_EXPLICIT_NOTE: frozenset[str] = frozenset({"pruning", "combined"})


def build_quality_note(technique: str, *, provenance: str = "", extra: str = "") -> str:
    """Compone la nota de una fila de ``quality_metrics``."""
    partes = [EVALUATOR_PREFIX, TECHNIQUE_NOTES.get(technique, ""), provenance, extra]
    return " ".join(parte.strip() for parte in partes if parte and parte.strip())


def parse_evaluator_from_notes(notes: str) -> str:
    """Extrae el evaluador del prefijo ``evaluator=``. Si no hay prefijo devuelve una cadena
    vacía."""
    coincidencia = _EVALUATOR_RE.match((notes or "").strip())
    return coincidencia.group(1) if coincidencia else ""
