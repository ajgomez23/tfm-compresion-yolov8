"""Precisión y recall a umbral de confianza fijo, sin interpolar ni monotonizar.

Las columnas precision_at_best_f1 y recall_at_best_f1 se miden en el umbral de mejor F1 de cada
modelo, que cambia entre FP32 e INT8. Aquí se miden todos los modelos en la misma rejilla de
umbrales. La precisión de COCOeval.accumulate está monotonizada, así que se cuenta directamente
sobre evalImgs (dtScores, dtMatches, dtIgnore, gtIgnore). Con umbral cero el recall tiene que
coincidir con el que guarda accumulate; se comprueba en cada uso.
"""

from __future__ import annotations

from typing import Any, Sequence

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.evaluation.coco_eval import OPERATING_POINT_IOU, CocoEvalError

logger = get_logger(__name__)

#: Rejilla fija de umbrales, común a todos los modelos. Por encima de 0,90 casi no quedan
#: detecciones y por debajo de 0,05 domina la cola de confianza mínima.
DEFAULT_CONFIDENCES: tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)

#: Umbral por defecto de Ultralytics en predicción. Se usa en las tablas por ser fijo, no óptimo.
REFERENCE_CONFIDENCE = 0.25

#: Tolerancia de la comprobación contra el recall de pycocotools.
RECALL_IDENTITY_TOLERANCE = 1e-9


def _indices_de_evaluacion(evaluator: Any) -> tuple[int, int, int]:
    """Índices de IoU 0,50, área ``all`` y mayor ``maxDets``, los mismos que
    operating_point_at_best_f1.
    """
    import numpy as np

    parametros = evaluator.params
    coincidencias = np.flatnonzero(np.isclose(parametros.iouThrs, OPERATING_POINT_IOU))
    if coincidencias.size == 0:
        raise CocoEvalError(
            f"La evaluación no incluye IoU={OPERATING_POINT_IOU}. Umbrales presentes: "
            f"{list(parametros.iouThrs)}.",
            "Evalúa con la rejilla de IoU por defecto de pycocotools.",
        )
    t = int(coincidencias[0])
    a = list(parametros.areaRngLbl).index("all")
    m = int(np.argmax(parametros.maxDets))
    return t, a, m


def _detecciones_por_clase(evaluator: Any, k: int, a: int, m: int) -> dict[str, Any] | None:
    """Reúne las detecciones de una clase como lo hace ``accumulate``.

    Se truncan a ``maxDet`` por imagen igual que pycocotools. No se reordena globalmente porque el
    recuento no depende del orden. Devuelve None si la clase no tiene anotaciones.
    """
    import numpy as np

    parametros = evaluator.params
    parametros_eval = evaluator._paramsEval
    num_imagenes = len(parametros_eval.imgIds)
    num_areas = len(parametros_eval.areaRng)
    max_det = parametros.maxDets[m]

    desplazamiento = k * num_areas * num_imagenes + a * num_imagenes
    entradas = [evaluator.evalImgs[desplazamiento + i] for i in range(num_imagenes)]
    entradas = [e for e in entradas if e is not None]
    if not entradas:
        return None

    gt_ignoradas = np.concatenate([e["gtIgnore"] for e in entradas])
    num_gt = int(np.count_nonzero(gt_ignoradas == 0))
    if num_gt == 0:
        return None

    scores = np.concatenate([e["dtScores"][0:max_det] for e in entradas])
    emparejadas = np.concatenate([e["dtMatches"][:, 0:max_det] for e in entradas], axis=1)
    ignoradas = np.concatenate([e["dtIgnore"][:, 0:max_det] for e in entradas], axis=1)
    return {
        "scores": np.asarray(scores, dtype=float),
        "matches": emparejadas,
        "ignored": ignoradas,
        "num_gt": num_gt,
    }


def precision_recall_at_confidences(
    evaluator: Any,
    confidences: Sequence[float] = DEFAULT_CONFIDENCES,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Mide precisión y recall sin interpolar en una rejilla de umbrales.

    Se promedia sobre las clases con anotaciones. Si una clase no conserva detecciones, se toma
    precisión 1 y recall 0. Lanza CocoEvalError si el recall reconstruido no coincide con
    pycocotools.
    """
    import numpy as np

    acumulado = getattr(evaluator, "eval", None)
    if not acumulado or "recall" not in acumulado:
        raise CocoEvalError(
            "Se ha pedido la curva de una evaluación sin acumular.",
            "Llama a COCOeval.accumulate() antes que a " "precision_recall_at_confidences().",
        )

    t, a, m = _indices_de_evaluacion(evaluator)
    catIds = list(evaluator.params.catIds)

    clases: dict[int, dict[str, Any]] = {}
    for k, categoria in enumerate(catIds):
        datos = _detecciones_por_clase(evaluator, k, a, m)
        if datos is None:
            continue
        emparejada = datos["matches"][t].astype(bool)
        ignorada = datos["ignored"][t].astype(bool)
        clases[k] = {
            "category_id": int(categoria),
            "scores": datos["scores"],
            "tp": np.logical_and(emparejada, np.logical_not(ignorada)),
            "fp": np.logical_and(np.logical_not(emparejada), np.logical_not(ignorada)),
            "num_gt": datos["num_gt"],
        }

    if not clases:
        raise CocoEvalError(
            "Ninguna clase tiene anotaciones en esta partición.",
            "Sin anotaciones no hay recall que medir. Comprueba la partición.",
        )

    def medir(umbral: float) -> dict[int, dict[str, float]]:
        """Cuenta aciertos y fallos por clase por encima de un umbral."""
        salida: dict[int, dict[str, float]] = {}
        for k, datos in clases.items():
            seleccion = datos["scores"] >= umbral
            verdaderos = int(np.count_nonzero(np.logical_and(datos["tp"], seleccion)))
            falsos = int(np.count_nonzero(np.logical_and(datos["fp"], seleccion)))
            detectadas = verdaderos + falsos
            salida[k] = {
                "true_positives": verdaderos,
                "false_positives": falsos,
                "num_gt": datos["num_gt"],
                # Sin detecciones por encima del umbral: precisión 1 y recall 0 por convenio.
                "precision": (verdaderos / detectadas) if detectadas else 1.0,
                "recall": verdaderos / datos["num_gt"],
            }
        return salida

    # Comprobación de anclaje: con umbral cero el recall debe ser el de self.eval["recall"].
    recall_pycocotools = acumulado["recall"][t][:, a, m]
    en_cero = medir(0.0)
    diferencias = {
        int(clases[k]["category_id"]): abs(v["recall"] - float(recall_pycocotools[k]))
        for k, v in en_cero.items()
    }
    peor = max(diferencias.values())
    if peor > RECALL_IDENTITY_TOLERANCE:
        culpable = max(diferencias, key=lambda c: diferencias[c])
        raise CocoEvalError(
            f"El recall reconstruido no coincide con el de pycocotools: difiere en {peor:.3e} "
            f"(categoría {culpable}), "
            f"por encima de {RECALL_IDENTITY_TOLERANCE:.0e}.",
            "La reconstrucción sobre evalImgs no lee las mismas detecciones que accumulate() "
            "con esta versión de pycocotools.",
        )

    puntos: list[dict[str, Any]] = []
    for umbral in sorted(float(c) for c in confidences):
        por_clase = medir(umbral)
        precision = float(np.mean([v["precision"] for v in por_clase.values()]))
        recall = float(np.mean([v["recall"] for v in por_clase.values()]))
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        detalle = {}
        for k, valores in por_clase.items():
            indice = int(clases[k]["category_id"])
            etiqueta = (class_names or {}).get(indice, str(indice))
            detalle[str(indice)] = {
                "name": etiqueta,
                "precision": round(valores["precision"], 6),
                "recall": round(valores["recall"], 6),
                "true_positives": valores["true_positives"],
                "false_positives": valores["false_positives"],
                "num_gt": valores["num_gt"],
            }
        puntos.append(
            {
                "confidence": round(umbral, 6),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "true_positives": sum(v["true_positives"] for v in por_clase.values()),
                "false_positives": sum(v["false_positives"] for v in por_clase.values()),
                "num_gt": sum(v["num_gt"] for v in por_clase.values()),
                "per_class": detalle,
            }
        )

    logger.info(
        "Curva medida en %d umbrales fijos sobre %d clases con anotaciones "
        "(identidad con pycocotools: peor diferencia %.1e)",
        len(puntos),
        len(clases),
        peor,
    )
    return {
        "iou": OPERATING_POINT_IOU,
        "max_detections": int(evaluator.params.maxDets[m]),
        "num_classes": len(clases),
        "points": puntos,
        "verification": {
            "recall_matches_pycocotools": True,
            "max_difference": peor,
            "tolerance": RECALL_IDENTITY_TOLERANCE,
            "checked_classes": len(diferencias),
        },
    }


def point_at(curve: dict[str, Any], confidence: float) -> dict[str, Any]:
    """Punto de la curva en un umbral concreto. Si el umbral no está en la rejilla falla; no se
    interpola.
    """
    for punto in curve.get("points", []):
        if abs(float(punto["confidence"]) - float(confidence)) < 1e-9:
            return punto
    disponibles = [p["confidence"] for p in curve.get("points", [])]
    raise CocoEvalError(
        f"La curva no incluye el umbral {confidence}. Medidos: {disponibles}.",
        "Vuelve a medir la curva con ese umbral en la rejilla. No se interpola: "
        "un valor interpolado no sería una medición.",
    )
