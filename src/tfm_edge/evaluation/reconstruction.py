"""Reconstruye una evaluación ya hecha a partir de sus detecciones guardadas.

Sirve para responder preguntas nuevas sin volver a inferir. Antes de usarla se comprueba que
reproduce el mAP y el punto de operación registrados. Lo usan los pasos 23 y 25; qué directorio de
detecciones corresponde a cada evaluación lo deciden los scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.evaluation import coco_eval

logger = get_logger(__name__)

#: Tolerancia del anclaje, la misma con la que el registro redondea.
ANCHOR_TOLERANCE = 1e-6

#: Motores y precisiones para trocear nombres de directorio desde la derecha, porque el
#: experiment_id puede llevar guiones bajos.
KNOWN_RUNTIMES = ("pytorch", "onnxruntime", "openvino", "tensorrt", "tflite")
KNOWN_PRECISIONS = ("fp32", "fp16", "int8")

#: Criterio de tamaño primario del registro: los dos lados escalados por el mismo factor.
SIZE_CRITERION = "inference_resolution"


class RebuiltEvaluation:
    """Evaluación reconstruida y ya anclada."""

    def __init__(
        self,
        evaluator: Any,
        anchor: dict[str, Any],
        num_detections: int,
        reference: dict[str, Any],
        detections: list[dict[str, Any]],
    ) -> None:
        self.evaluator = evaluator
        self.anchor = anchor
        self.num_detections = num_detections
        self.reference = reference
        self.detections = detections


def invert_categories(detections: list[dict[str, Any]], num_classes: int) -> list[dict[str, Any]]:
    """Pasa las detecciones a la numeración de clases del proyecto.

    Ultralytics numera desde 1 en datasets que no son COCO. Se comprueba que las categorías caen en
    el rango esperado; un desfase lo detectaría también el anclaje.
    """
    observadas = {int(d["category_id"]) for d in detections}
    esperadas = set(range(1, num_classes + 1))
    if not observadas <= esperadas:
        raise TFMEdgeError(
            f"Categorías {sorted(observadas - esperadas)} fuera de la numeración esperada "
            f"({min(esperadas)}-{max(esperadas)}).",
            "El predictions.json no corresponde a la configuración esperada. Revisa "
            "normalize_categories en models/detector.py.",
        )
    traducidas = []
    for deteccion in detections:
        copia = dict(deteccion)
        copia["category_id"] = int(deteccion["category_id"]) - 1
        traducidas.append(copia)
    return traducidas


def anchor_against_recorded(evaluator: Any, recorded_dump: dict[str, Any]) -> dict[str, Any]:
    """Comprueba que la reconstrucción reproduce el mAP@50 y el punto de mejor F1 registrados.

    Lanza TFMEdgeError si alguna diferencia supera ANCHOR_TOLERANCE.
    """
    registrado = recorded_dump["coco"]["by_criterion"][SIZE_CRITERION]
    comparaciones: dict[str, tuple[float, float]] = {
        "map_50": (float(evaluator.stats[1]), float(registrado["map_50"])),
        "map_50_95": (float(evaluator.stats[0]), float(registrado["map_50_95"])),
    }

    punto = coco_eval.operating_point_at_best_f1(evaluator)
    punto_registrado = registrado.get("operating_point", {})
    for clave in ("precision", "recall", "f1", "confidence"):
        actual, previo = punto.get(clave, ""), punto_registrado.get(clave, "")
        if isinstance(actual, (int, float)) and isinstance(previo, (int, float)):
            comparaciones[f"operating_point.{clave}"] = (float(actual), float(previo))

    diferencias = {k: abs(a - b) for k, (a, b) in comparaciones.items()}
    peor = max(diferencias.values())
    if peor > ANCHOR_TOLERANCE:
        detalle = ", ".join(
            f"{k}: reconstruido {a:.6f} frente a registrado {b:.6f}"
            for k, (a, b) in comparaciones.items()
            if diferencias[k] > ANCHOR_TOLERANCE
        )
        raise TFMEdgeError(
            f"La reconstrucción NO reproduce la evaluación registrada ({detalle}).",
            "No se escribe nada: el predictions.json no es de esa pasada, o han cambiado la "
            "partición o el COCO de referencia.",
        )
    return {
        "anchored": True,
        "tolerance": ANCHOR_TOLERANCE,
        "max_difference": peor,
        "compared": {k: {"rebuilt": a, "recorded": b} for k, (a, b) in comparaciones.items()},
    }


def rebuild_evaluation(
    raw_detections: list[dict[str, Any]],
    recorded_dump: dict[str, Any],
    reference: dict[str, Any],
    images_dir: Path,
    resolution: int,
    num_classes: int,
) -> RebuiltEvaluation:
    """Reconstruye una evaluación desde sus detecciones y la ancla al volcado registrado.

    Sigue el mismo camino que la evaluación original: traduce categorías, vuelve a coordenadas
    originales y escala ambos lados por el mismo factor.
    """
    detecciones = invert_categories(raw_detections, num_classes)
    detecciones = coco_eval.map_detections_to_ground_truth(detecciones, reference)
    detecciones, _ = coco_eval.rescale_detections_to_original(detecciones, reference, images_dir)

    referencia_inferencia = coco_eval.scale_ground_truth(reference, resolution)
    detecciones_inferencia = coco_eval.scale_detections(detecciones, reference, resolution)
    evaluador = coco_eval.build_evaluator(
        referencia_inferencia, detecciones_inferencia, SIZE_CRITERION
    )

    anclaje = anchor_against_recorded(evaluador, recorded_dump)
    return RebuiltEvaluation(
        evaluator=evaluador,
        anchor=anclaje,
        num_detections=len(detecciones),
        reference=referencia_inferencia,
        detections=detecciones_inferencia,
    )
