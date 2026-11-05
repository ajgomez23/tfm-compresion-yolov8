"""Reparte la pérdida de un par de modelos entre emparejamiento perdido y orden roto.

Cada anotación la encuentran los dos modelos, solo uno o ninguno, y el cambio de aciertos a un
umbral se reparte sin resto entre esos estados (se comprueba en cada umbral). Los falsos positivos
se clasifican en duplicado, localización pobre y espurio. La IoU se calcula aquí a partir de las
cajas y se comprueba que reproduce el emparejamiento de pycocotools. ranking_separation mide el
orden sin umbral: la probabilidad de que un acierto puntúe más que un falso positivo.
"""

from __future__ import annotations

from typing import Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.evaluation.coco_eval import OPERATING_POINT_IOU, CocoEvalError

logger = get_logger(__name__)

#: IoU del emparejamiento, la misma del punto de operación y de la curva.
MATCHING_IOU = OPERATING_POINT_IOU

#: Por encima de esta IoU con un objeto de su clase, un falso positivo es un duplicado.
DUPLICATE_IOU = MATCHING_IOU

#: Por debajo de esta IoU, un falso positivo es espurio; entre esta y DUPLICATE_IOU, mala
#: localización.
SPURIOUS_IOU = 0.1

#: Tolerancia de la comprobación de la IoU propia contra el emparejamiento de pycocotools.
IOU_ANCHOR_TOLERANCE = 1e-6

#: Umbrales por defecto. El cero va primero porque distingue más cajas de un tipo de las mismas
#: cajas con más confianza.
DECOMPOSITION_CONFIDENCES: tuple[float, ...] = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90)

FP_KINDS = ("duplicate", "poor_localization", "spurious")


class MatchingAnalysisError(TFMEdgeError):
    """Error propio del reparto entre emparejamiento y orden."""


# Geometría
def box_iou_matrix(a: Any, b: Any) -> Any:
    """IoU de cada caja de ``a`` contra cada caja de ``b``, en formato COCO ``[x, y, ancho,
    alto]``."""
    import numpy as np

    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=float)

    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[None, :, 0], b[None, :, 1]
    bx2, by2 = bx1 + b[None, :, 2], by1 + b[None, :, 3]

    ancho = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    alto = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    interseccion = ancho * alto
    union = (a[:, 2:3] * a[:, 3:4]) + (b[None, :, 2] * b[None, :, 3]) - interseccion
    return np.where(union > 0, interseccion / np.maximum(union, 1e-12), 0.0)


def rank_auc(positive_scores: Any, negative_scores: Any) -> float | None:
    """Probabilidad de que un positivo puntúe por encima de un negativo, con empates a medias.

    Usa la identidad de Mann-Whitney sobre rangos. Devuelve None si falta alguno de los dos lados.
    """
    import numpy as np

    positivos = np.asarray(positive_scores, dtype=float)
    negativos = np.asarray(negative_scores, dtype=float)
    n1, n0 = positivos.size, negativos.size
    if n1 == 0 or n0 == 0:
        return None

    todos = np.concatenate([positivos, negativos])
    orden = np.argsort(todos, kind="mergesort")
    ordenados = todos[orden]
    inicio = np.flatnonzero(np.concatenate(([True], ordenados[1:] != ordenados[:-1])))
    fin = np.concatenate((inicio[1:], [ordenados.size]))
    promedio = (inicio + fin - 1) / 2.0 + 1.0
    rangos = np.empty(ordenados.size, dtype=float)
    rangos[orden] = np.repeat(promedio, fin - inicio)
    return float((rangos[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# Lectura de una evaluación
class EvaluationFacts:
    """Datos de una evaluación necesarios para repartir su pérdida."""

    def __init__(
        self,
        num_gt: int,
        covered: dict[int, float],
        scores: Any,
        is_tp: Any,
        max_iou: Any,
        category: Any,
        matched_iou_min: float,
    ) -> None:
        self.num_gt = num_gt
        self.covered = covered
        self.scores = scores
        self.is_tp = is_tp
        self.max_iou = max_iou
        self.category = category
        self.matched_iou_min = matched_iou_min

    def tp_at(self, confidence: float) -> int:
        """Aciertos por encima de un umbral."""
        import numpy as np

        return int(np.count_nonzero(self.is_tp & (self.scores >= confidence)))

    def fp_at(self, confidence: float) -> int:
        """Falsos positivos por encima de un umbral."""
        import numpy as np

        return int(np.count_nonzero((~self.is_tp) & (self.scores >= confidence)))


def _indices_de_evaluacion(evaluator: Any) -> tuple[int, int]:
    """Índices de la IoU del emparejamiento y del rango de área ``all``."""
    import numpy as np

    coincidencias = np.flatnonzero(np.isclose(evaluator.params.iouThrs, MATCHING_IOU))
    if coincidencias.size == 0:
        raise CocoEvalError(
            f"La evaluación no incluye IoU={MATCHING_IOU}. Umbrales presentes: "
            f"{list(evaluator.params.iouThrs)}.",
            "Evalúa con la rejilla de IoU por defecto de pycocotools.",
        )
    return int(coincidencias[0]), list(evaluator.params.areaRngLbl).index("all")


def evaluation_facts(evaluator: Any) -> EvaluationFacts:
    """Lee de una evaluación la cobertura, el estado de cada detección y su IoU.

    Recorre evalImgs en el rango ``all``. La IoU calculada se comprueba contra la pareja peor
    emparejada por pycocotools.
    """
    import numpy as np

    if not getattr(evaluator, "evalImgs", None):
        raise MatchingAnalysisError(
            "Se ha pedido el reparto de una evaluación sin evaluar.",
            "Llama a COCOeval.evaluate() antes que a evaluation_facts().",
        )

    t, a = _indices_de_evaluacion(evaluator)
    parametros_eval = evaluator._paramsEval
    num_imagenes = len(parametros_eval.imgIds)
    num_areas = len(parametros_eval.areaRng)

    cubiertos: dict[int, float] = {}
    scores: list[Any] = []
    aciertos: list[Any] = []
    ious: list[Any] = []
    categorias: list[Any] = []
    num_gt = 0
    peor_iou_emparejada = 1.0
    parejas = 0

    anotaciones_gt = evaluator.cocoGt.anns
    anotaciones_dt = evaluator.cocoDt.anns

    for k, categoria in enumerate(evaluator.params.catIds):
        desplazamiento = k * num_areas * num_imagenes + a * num_imagenes
        for i in range(num_imagenes):
            entrada = evaluator.evalImgs[desplazamiento + i]
            if entrada is None:
                continue

            gt_ids = list(entrada["gtIds"])
            gt_ignorada = np.asarray(entrada["gtIgnore"], dtype=bool)
            gt_validas = [g for g, ign in zip(gt_ids, gt_ignorada) if not ign]
            num_gt += len(gt_validas)

            # Cobertura: qué anotaciones encuentra el modelo
            emparejadas_gt = np.asarray(entrada["gtMatches"][t], dtype=float)
            for j, anotacion in enumerate(gt_ids):
                if gt_ignorada[j]:
                    continue
                deteccion = int(emparejadas_gt[j])
                if deteccion == 0:
                    continue
                cubiertos[int(anotacion)] = float(anotaciones_dt[deteccion]["score"])
                iou_pareja = float(
                    box_iou_matrix(
                        [anotaciones_dt[deteccion]["bbox"]], [anotaciones_gt[anotacion]["bbox"]]
                    )[0, 0]
                )
                peor_iou_emparejada = min(peor_iou_emparejada, iou_pareja)
                parejas += 1

            # Estado y geometría de cada detección
            dt_ids = list(entrada["dtIds"])
            if not dt_ids:
                continue
            dt_ignorada = np.asarray(entrada["dtIgnore"][t], dtype=bool)
            dt_emparejada = np.asarray(entrada["dtMatches"][t], dtype=float) > 0
            se_cuenta = ~dt_ignorada
            if not np.any(se_cuenta):
                continue

            cajas_dt = np.asarray([anotaciones_dt[d]["bbox"] for d in dt_ids], dtype=float)
            cajas_gt = np.asarray(
                [anotaciones_gt[g]["bbox"] for g in gt_validas], dtype=float
            ).reshape(-1, 4)
            iou_max = box_iou_matrix(cajas_dt, cajas_gt).max(axis=1, initial=0.0)

            scores.append(np.asarray(entrada["dtScores"], dtype=float)[se_cuenta])
            aciertos.append(dt_emparejada[se_cuenta])
            ious.append(iou_max[se_cuenta])
            categorias.append(np.full(int(np.count_nonzero(se_cuenta)), int(categoria)))

    if num_gt == 0:
        raise MatchingAnalysisError(
            "Ninguna anotación de la partición cuenta para el emparejamiento.",
            "Sin anotaciones no hay nada que repartir. Comprueba la partición.",
        )
    if parejas and peor_iou_emparejada < MATCHING_IOU - IOU_ANCHOR_TOLERANCE:
        raise MatchingAnalysisError(
            "La IoU calculada no reproduce el emparejamiento de pycocotools: una pareja "
            f"emparejada a IoU {MATCHING_IOU} da {peor_iou_emparejada:.6f}.",
            "Revisa el formato y la escala de las cajas.",
        )

    vacio = np.zeros(0)
    return EvaluationFacts(
        num_gt=num_gt,
        covered=cubiertos,
        scores=np.concatenate(scores) if scores else vacio,
        is_tp=np.concatenate(aciertos) if aciertos else vacio.astype(bool),
        max_iou=np.concatenate(ious) if ious else vacio,
        category=np.concatenate(categorias).astype(int) if categorias else vacio.astype(int),
        matched_iou_min=float(peor_iou_emparejada) if parejas else 0.0,
    )


def check_against_curve(facts: EvaluationFacts, curve_dump: dict[str, Any]) -> dict[str, Any]:
    """Comprueba que aciertos y falsos positivos coinciden con los de la curva del paso 23 en todos
    los umbrales.
    """
    desajustes: list[str] = []
    comprobados = 0
    for punto in curve_dump["curve"]["points"]:
        confianza = float(punto["confidence"])
        tp, fp = facts.tp_at(confianza), facts.fp_at(confianza)
        if tp != int(punto["true_positives"]) or fp != int(punto["false_positives"]):
            desajustes.append(
                f"conf={confianza:.2f}: aquí TP={tp} FP={fp}, "
                f"en la curva TP={punto['true_positives']} FP={punto['false_positives']}"
            )
        comprobados += 1

    if desajustes:
        raise MatchingAnalysisError(
            "Los recuentos de este módulo no coinciden con la curva ya volcada "
            "de la misma evaluación: " + "; ".join(desajustes[:3]),
            "La curva no corresponde a esta evaluación o una de las dos lecturas es incorrecta.",
        )
    # La procedencia de la curva no se copia porque lleva rutas absolutas.
    return {"matches_operating_curve": True, "confidences_checked": comprobados}


# Reparto
def coverage_split(before: EvaluationFacts, after: EvaluationFacts) -> dict[str, Any]:
    """Reparte las anotaciones según qué modelo las encuentra, sin umbral de confianza."""
    if before.num_gt != after.num_gt:
        raise MatchingAnalysisError(
            f"Las dos evaluaciones no cuentan las mismas anotaciones: "
            f"{before.num_gt} frente a {after.num_gt}.",
            "El reparto compara anotación a anotación, así que las dos tienen "
            "que estar evaluadas sobre la misma partición y la misma referencia.",
        )

    antes, despues = set(before.covered), set(after.covered)
    ambos = antes & despues
    return {
        "num_gt": before.num_gt,
        "covered_before": len(antes),
        "covered_after": len(despues),
        "covered_by_both": len(ambos),
        "matching_lost": len(antes - despues),
        "matching_gained": len(despues - antes),
        "covered_by_neither": before.num_gt - len(antes | despues),
        "coverage_rate_before": len(antes) / before.num_gt,
        "coverage_rate_after": len(despues) / after.num_gt,
    }


def recall_decomposition(
    before: EvaluationFacts, after: EvaluationFacts, confidence: float
) -> dict[str, Any]:
    """Reparte el cambio de aciertos a un umbral entre sus tres causas y comprueba que suman el
    total.
    """
    antes, despues = before.covered, after.covered
    solo_antes = set(antes) - set(despues)
    solo_despues = set(despues) - set(antes)
    ambos = set(antes) & set(despues)

    perdidos = sum(1 for g in solo_antes if antes[g] >= confidence)
    ganados = sum(1 for g in solo_despues if despues[g] >= confidence)
    cruzan_despues = sum(1 for g in ambos if despues[g] >= confidence)
    cruzan_antes = sum(1 for g in ambos if antes[g] >= confidence)

    tp_antes, tp_despues = before.tp_at(confidence), after.tp_at(confidence)
    delta = tp_despues - tp_antes
    reparto = {
        "from_matching_lost": -perdidos,
        "from_matching_gained": ganados,
        "from_score_shift": cruzan_despues - cruzan_antes,
    }
    suma = sum(reparto.values())
    if suma != delta:
        raise MatchingAnalysisError(
            f"El reparto de aciertos en conf={confidence:.2f} no suma: los tres "
            f"sumandos dan {suma} y el cambio medido es {delta}.",
            "Los tres estados deberían cubrir todas las anotaciones.",
        )

    return {
        "confidence": confidence,
        "tp_before": tp_antes,
        "tp_after": tp_despues,
        "delta_tp": delta,
        **reparto,
        "identity_holds": True,
    }


def fp_taxonomy(facts: EvaluationFacts, confidence: float) -> dict[str, Any]:
    """Clasifica los falsos positivos por encima de un umbral según lo que tocan."""
    import numpy as np

    seleccion = (~facts.is_tp) & (facts.scores >= confidence)
    iou = facts.max_iou[seleccion]
    duplicados = int(np.count_nonzero(iou >= DUPLICATE_IOU))
    espurios = int(np.count_nonzero(iou < SPURIOUS_IOU))
    total = int(iou.size)
    return {
        "confidence": confidence,
        "duplicate": duplicados,
        "poor_localization": total - duplicados - espurios,
        "spurious": espurios,
        "total": total,
    }


def ranking_separation(
    facts: EvaluationFacts, class_names: dict[int, str] | None = None
) -> dict[str, Any]:
    """Separación entre aciertos y falsos positivos, por clase y en conjunto.

    Se promedia sobre las clases que tienen de las dos cosas.
    """
    import numpy as np

    por_clase: dict[str, float] = {}
    for categoria in sorted(set(int(c) for c in np.unique(facts.category))):
        de_la_clase = facts.category == categoria
        aciertos = facts.scores[de_la_clase & facts.is_tp]
        fallos = facts.scores[de_la_clase & (~facts.is_tp)]
        valor = rank_auc(aciertos, fallos)
        if valor is None:
            continue
        nombre = (class_names or {}).get(categoria, str(categoria))
        por_clase[f"{categoria:02d}_{nombre}"] = valor

    return {
        "macro": float(np.mean(list(por_clase.values()))) if por_clase else None,
        "pooled": rank_auc(facts.scores[facts.is_tp], facts.scores[~facts.is_tp]),
        "classes_measured": len(por_clase),
        "per_class": por_clase,
    }


def covered_score_shift(before: EvaluationFacts, after: EvaluationFacts) -> dict[str, Any]:
    """Cambio de confianza de las anotaciones que encuentran los dos modelos."""
    import numpy as np

    ambos = sorted(set(before.covered) & set(after.covered))
    if not ambos:
        return {"num_covered_by_both": 0, "median_before": None, "median_after": None}
    antes = np.asarray([before.covered[g] for g in ambos])
    despues = np.asarray([after.covered[g] for g in ambos])
    diferencia = despues - antes
    return {
        "num_covered_by_both": len(ambos),
        "median_before": float(np.median(antes)),
        "median_after": float(np.median(despues)),
        "median_delta": float(np.median(diferencia)),
        "p25_delta": float(np.percentile(diferencia, 25)),
        "p75_delta": float(np.percentile(diferencia, 75)),
        "fraction_increased": float(np.count_nonzero(diferencia > 0) / diferencia.size),
    }


def _falsos_positivos_del_par(
    before: EvaluationFacts, after: EvaluationFacts, confidence: float
) -> dict[str, Any]:
    """Desglose de falsos positivos de los dos lados y su diferencia."""
    antes = fp_taxonomy(before, confidence)
    despues = fp_taxonomy(after, confidence)
    return {
        "confidence": confidence,
        "before": antes,
        "after": despues,
        "delta": {tipo: despues[tipo] - antes[tipo] for tipo in (*FP_KINDS, "total")},
    }


def analyze_pair(
    before: EvaluationFacts,
    after: EvaluationFacts,
    confidences: Sequence[float],
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Reparte la pérdida de un par entre emparejamiento y orden."""
    umbrales = sorted(float(c) for c in confidences)
    return {
        "coverage": coverage_split(before, after),
        "recall_decomposition": [recall_decomposition(before, after, c) for c in umbrales],
        "false_positives": [_falsos_positivos_del_par(before, after, c) for c in umbrales],
        "ranking": {
            "before": ranking_separation(before, class_names),
            "after": ranking_separation(after, class_names),
        },
        "covered_score_shift": covered_score_shift(before, after),
        "thresholds": {
            "matching_iou": MATCHING_IOU,
            "duplicate_iou": DUPLICATE_IOU,
            "spurious_iou": SPURIOUS_IOU,
        },
    }
