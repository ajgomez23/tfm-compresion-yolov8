"""Equivalencia numérica entre el modelo de PyTorch y el artefacto exportado.

La salida de YOLOv8 tiene forma ``(lote, 4 + clases, cajas)``: los canales 0-3 son coordenadas en
píxeles y el resto puntuaciones en [0, 1]. Cada bloque se compara con su propio criterio, porque una
tolerancia global relativa o absoluta falla con uno de los dos. compare_detections compara además
las detecciones resultantes, que es lo que llega al mAP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import numpy as np

logger = get_logger(__name__)

#: Canales de coordenadas; el resto son puntuaciones.
NUM_BOX_CHANNELS = 4

#: Tolerancia relativa de las coordenadas respecto a la resolución. En ONNX se midió 2,0e-06.
BOX_RELATIVE_TOLERANCE = 1e-4

#: Tolerancia absoluta de las puntuaciones. En ONNX se midió 1,3e-07.
SCORE_ABSOLUTE_TOLERANCE = 1e-4


class EquivalenceError(TFMEdgeError):
    """El artefacto exportado no es numéricamente equivalente al original."""


@dataclass
class EquivalenceReport:
    """Resultado de comparar dos salidas."""

    equivalent: bool = False
    num_samples: int = 0
    input_resolution: int = 640
    box_max_abs_diff: float = 0.0
    box_max_rel_diff: float = 0.0
    score_max_abs_diff: float = 0.0
    box_tolerance: float = BOX_RELATIVE_TOLERANCE
    score_tolerance: float = SCORE_ABSOLUTE_TOLERANCE
    shape_matches: bool = True
    failures: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return asdict(self)

    def summary(self) -> str:
        """Genera un resumen legible."""
        veredicto = "EQUIVALENTE" if self.equivalent else "NO EQUIVALENTE"
        lineas = [
            f"Equivalencia numérica: {veredicto} ({self.num_samples} muestras)",
            f"  coordenadas: {self.box_max_abs_diff:.3e} px "
            f"({self.box_max_rel_diff:.3e} relativo, tolerancia {self.box_tolerance:.0e})",
            f"  puntuaciones: {self.score_max_abs_diff:.3e} "
            f"(tolerancia {self.score_tolerance:.0e})",
        ]
        for fallo in self.failures:
            lineas.append(f"  FALLA: {fallo}")
        return "\n".join(lineas)


def compare_outputs(
    reference: "np.ndarray",
    candidate: "np.ndarray",
    input_resolution: int = 640,
    box_tolerance: float = BOX_RELATIVE_TOLERANCE,
    score_tolerance: float = SCORE_ABSOLUTE_TOLERANCE,
) -> EquivalenceReport:
    """Compara dos salidas ``(N, 4+C, cajas)`` separando coordenadas y puntuaciones."""
    import numpy as np

    informe = EquivalenceReport(
        input_resolution=input_resolution,
        box_tolerance=box_tolerance,
        score_tolerance=score_tolerance,
        num_samples=1,
    )

    if reference.shape != candidate.shape:
        informe.shape_matches = False
        informe.failures.append(
            f"las formas no coinciden: referencia {reference.shape}, "
            f"candidato {candidate.shape}. El artefacto no representa al mismo "
            f"modelo."
        )
        return informe

    if reference.ndim != 3 or reference.shape[1] <= NUM_BOX_CHANNELS:
        informe.failures.append(
            f"forma inesperada {reference.shape}: se esperaba " f"(lote, 4 + clases, cajas)."
        )
        return informe

    diferencia = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))

    dif_cajas = float(diferencia[:, :NUM_BOX_CHANNELS, :].max())
    # La escala es la resolución de entrada y no el máximo observado, para que la tolerancia no
    # dependa de la muestra.
    informe.box_max_abs_diff = dif_cajas
    informe.box_max_rel_diff = dif_cajas / float(input_resolution)
    informe.score_max_abs_diff = float(diferencia[:, NUM_BOX_CHANNELS:, :].max())

    if informe.box_max_rel_diff > box_tolerance:
        informe.failures.append(
            f"las coordenadas difieren {informe.box_max_abs_diff:.3e} px "
            f"({informe.box_max_rel_diff:.3e} relativo), por encima de "
            f"{box_tolerance:.0e}"
        )
    if informe.score_max_abs_diff > score_tolerance:
        informe.failures.append(
            f"las puntuaciones difieren {informe.score_max_abs_diff:.3e}, por "
            f"encima de {score_tolerance:.0e}"
        )

    informe.equivalent = not informe.failures
    return informe


def compare_many(
    references: Sequence["np.ndarray"],
    candidates: Sequence["np.ndarray"],
    input_resolution: int = 640,
    box_tolerance: float = BOX_RELATIVE_TOLERANCE,
    score_tolerance: float = SCORE_ABSOLUTE_TOLERANCE,
) -> EquivalenceReport:
    """Compara varias muestras y se queda con el peor caso, para ejercitar más ramas del grafo."""
    if len(references) != len(candidates):
        raise EquivalenceError(
            f"Se han pasado {len(references)} referencias y {len(candidates)} " f"candidatos.",
            "Tienen que ser las mismas entradas en el mismo orden.",
        )
    if not references:
        raise EquivalenceError(
            "No hay ninguna muestra que comparar.",
            "Pasa al menos dos entradas distintas: una sola puede no ejercitar "
            "todas las ramas del grafo.",
        )

    agregado = EquivalenceReport(
        input_resolution=input_resolution,
        box_tolerance=box_tolerance,
        score_tolerance=score_tolerance,
        num_samples=len(references),
        equivalent=True,
    )
    for referencia, candidato in zip(references, candidates):
        parcial = compare_outputs(
            referencia, candidato, input_resolution, box_tolerance, score_tolerance
        )
        agregado.box_max_abs_diff = max(agregado.box_max_abs_diff, parcial.box_max_abs_diff)
        agregado.box_max_rel_diff = max(agregado.box_max_rel_diff, parcial.box_max_rel_diff)
        agregado.score_max_abs_diff = max(agregado.score_max_abs_diff, parcial.score_max_abs_diff)
        agregado.shape_matches = agregado.shape_matches and parcial.shape_matches
        for fallo in parcial.failures:
            if fallo not in agregado.failures:
                agregado.failures.append(fallo)

    agregado.equivalent = not agregado.failures
    if agregado.equivalent:
        logger.info(
            "Equivalencia verificada sobre %d muestras: coordenadas %.3e px, " "puntuaciones %.3e",
            agregado.num_samples,
            agregado.box_max_abs_diff,
            agregado.score_max_abs_diff,
        )
    else:
        logger.error("El artefacto NO es equivalente: %s", "; ".join(agregado.failures))
    return agregado


def compare_detections(
    reference: "np.ndarray",
    candidate: "np.ndarray",
    conf_threshold: float = 0.25,
) -> dict[str, Any]:
    """Compara cuántas cajas superan el umbral de confianza en cada salida."""
    import numpy as np

    def contar(salida: "np.ndarray") -> int:
        """Cuenta cajas por encima del umbral."""
        puntuaciones = salida[:, NUM_BOX_CHANNELS:, :].max(axis=1)
        return int((puntuaciones > conf_threshold).sum())

    n_referencia = contar(reference)
    n_candidato = contar(candidate)
    return {
        "conf_threshold": conf_threshold,
        "reference_boxes": n_referencia,
        "candidate_boxes": n_candidato,
        "match": n_referencia == n_candidato,
        # Cero cajas en las dos no prueba la equivalencia; se indica para no leerlo como correcto.
        "informative": n_referencia > 0 or n_candidato > 0,
    }
