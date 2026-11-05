"""Reparto de los errores de detección en las seis columnas error_* del registro.

Parte de matching_analysis.evaluation_facts y añade la IoU de cada detección contra las anotaciones
de las demás clases. Con la IoU contra su clase (propia) y contra las demás (ajena):

duplicate: propia >= 0,5. localization: 0,1 <= propia < 0,5. classification: propia < 0,1 y ajena >=
0,5. both: propia < 0,1 y 0,1 <= ajena < 0,5. background: propia < 0,1 y ajena < 0,1. missed cuenta
las anotaciones que ninguna detección empareja.

Los cortes son los de matching_analysis, así que el reparto refina exactamente el de fp_taxonomy
(classification + both + background = spurious), y se comprueba en cada llamada. Para asegurar que
los dos recorridos están alineados, el segundo recalcula también la IoU contra la propia clase y la
compara elemento a elemento.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.evaluation.matching_analysis import (
    DECOMPOSITION_CONFIDENCES,
    DUPLICATE_IOU,
    IOU_ANCHOR_TOLERANCE,
    MATCHING_IOU,
    SPURIOUS_IOU,
    EvaluationFacts,
    _indices_de_evaluacion,
    box_iou_matrix,
    evaluation_facts,
    fp_taxonomy,
)

logger = get_logger(__name__)

#: Las seis clases en el orden de las columnas del registro; missed va al final porque no cuenta
#: cajas.
ERROR_KINDS: tuple[str, ...] = (
    "classification",
    "localization",
    "both",
    "duplicate",
    "background",
    "missed",
)

#: Las cinco clases de falso positivo; suman el total sin resto.
FALSE_POSITIVE_KINDS: tuple[str, ...] = tuple(k for k in ERROR_KINDS if k != "missed")

#: Clase del reparto y columna del registro. Se escribe completa para que un test la contraste con
#: registry.py.
REGISTRY_COLUMNS: dict[str, str] = {
    "classification": "error_classification",
    "localization": "error_localization",
    "both": "error_both",
    "duplicate": "error_duplicate",
    "background": "error_background",
    "missed": "error_missed",
}


#: Cómo se agregan los recuentos. Convertir missed en recall no da el recall de la curva del paso
#: 23, que es la media por clase (0,737 frente a 0,539 en el baseline).
AGGREGATION = (
    "recuentos agregados sobre todas las anotaciones y detecciones de la "
    "particion, no medias por clase"
)


#: Subdirectorio de results/raw donde escribe el paso 34 y lee el paso 30.
DUMP_SUBDIR = "error_decomposition"


def dump_name(confidence: float) -> str:
    """Nombre del volcado para una confianza, en centésimas."""
    return f"error_decomposition_conf{int(round(confidence * 100)):03d}.json"


class ErrorDecompositionError(TFMEdgeError):
    """Error propio del reparto en las seis clases."""


class DecompositionFacts:
    """Hechos de matching_analysis más la IoU contra las demás clases.

    ``own_iou_anchor_max_gap`` es la mayor diferencia al recalcular la IoU contra la propia clase, y
    prueba el alineamiento.
    """

    def __init__(
        self,
        facts: EvaluationFacts,
        max_iou_other_class: Any,
        own_iou_anchor_max_gap: float,
    ) -> None:
        self.facts = facts
        self.max_iou_other_class = max_iou_other_class
        self.own_iou_anchor_max_gap = own_iou_anchor_max_gap

    def missed_at(self, confidence: float) -> int:
        """Anotaciones que ninguna detección por encima del umbral empareja."""
        encontradas = sum(1 for score in self.facts.covered.values() if score >= confidence)
        return int(self.facts.num_gt - encontradas)


def _indice_de_anotaciones_por_imagen(evaluator: Any) -> dict[int, tuple[Any, Any]]:
    """Anotaciones no ignoradas por imagen con su clase, leídas de evalImgs para usar el mismo
    universo que pycocotools.
    """
    import numpy as np

    _, a = _indices_de_evaluacion(evaluator)
    parametros_eval = evaluator._paramsEval
    num_imagenes = len(parametros_eval.imgIds)
    num_areas = len(parametros_eval.areaRng)
    anotaciones_gt = evaluator.cocoGt.anns

    por_imagen: dict[int, list[tuple[list[float], int]]] = {}
    for k, categoria in enumerate(evaluator.params.catIds):
        desplazamiento = k * num_areas * num_imagenes + a * num_imagenes
        for i, imagen in enumerate(parametros_eval.imgIds):
            entrada = evaluator.evalImgs[desplazamiento + i]
            if entrada is None:
                continue
            ignorada = np.asarray(entrada["gtIgnore"], dtype=bool)
            for anotacion, ign in zip(entrada["gtIds"], ignorada):
                if ign:
                    continue
                por_imagen.setdefault(int(imagen), []).append(
                    (list(anotaciones_gt[int(anotacion)]["bbox"]), int(categoria))
                )

    indice: dict[int, tuple[Any, Any]] = {}
    for imagen, entradas in por_imagen.items():
        cajas = np.asarray([caja for caja, _ in entradas], dtype=float).reshape(-1, 4)
        clases = np.asarray([clase for _, clase in entradas], dtype=int)
        indice[imagen] = (cajas, clases)
    return indice


def decomposition_facts(evaluator: Any) -> DecompositionFacts:
    """Añade a los hechos la IoU contra las demás clases y comprueba el alineamiento con la IoU
    propia.
    """
    import numpy as np

    facts = evaluation_facts(evaluator)
    indice = _indice_de_anotaciones_por_imagen(evaluator)

    t, a = _indices_de_evaluacion(evaluator)
    parametros_eval = evaluator._paramsEval
    num_imagenes = len(parametros_eval.imgIds)
    num_areas = len(parametros_eval.areaRng)
    anotaciones_dt = evaluator.cocoDt.anns

    ajenas: list[Any] = []
    propias: list[Any] = []

    for k, categoria in enumerate(evaluator.params.catIds):
        desplazamiento = k * num_areas * num_imagenes + a * num_imagenes
        for i, imagen in enumerate(parametros_eval.imgIds):
            entrada = evaluator.evalImgs[desplazamiento + i]
            if entrada is None:
                continue
            dt_ids = list(entrada["dtIds"])
            if not dt_ids:
                continue
            se_cuenta = ~np.asarray(entrada["dtIgnore"][t], dtype=bool)
            if not np.any(se_cuenta):
                continue

            cajas_dt = np.asarray([anotaciones_dt[d]["bbox"] for d in dt_ids], dtype=float)
            cajas_gt, clases_gt = indice.get(
                int(imagen), (np.zeros((0, 4)), np.zeros(0, dtype=int))
            )
            iou = box_iou_matrix(cajas_dt, cajas_gt)
            de_su_clase = clases_gt == int(categoria)

            propias.append(iou[:, de_su_clase].max(axis=1, initial=0.0)[se_cuenta])
            ajenas.append(iou[:, ~de_su_clase].max(axis=1, initial=0.0)[se_cuenta])

    vacio = np.zeros(0)
    propia = np.concatenate(propias) if propias else vacio
    ajena = np.concatenate(ajenas) if ajenas else vacio

    if propia.size != facts.max_iou.size:
        raise ErrorDecompositionError(
            f"Los dos recorridos no ven el mismo número de detecciones: {facts.max_iou.size} en "
            f"los hechos y {propia.size} aquí.",
            "El reparto en seis clases depende de que los dos vectores estén "
            "alineados. Sin eso, las cifras saldrían plausibles y equivocadas.",
        )
    hueco = float(np.max(np.abs(propia - facts.max_iou))) if propia.size else 0.0
    if hueco > IOU_ANCHOR_TOLERANCE:
        raise ErrorDecompositionError(
            f"La IoU contra su propia clase recalculada aquí no coincide con la "
            f"de los hechos: diferencia máxima {hueco:.6g}.",
            "Los dos recorridos de evalImgs no están dando el mismo orden de "
            "detecciones. Ningún reparto vale mientras eso no cuadre.",
        )

    return DecompositionFacts(facts=facts, max_iou_other_class=ajena, own_iou_anchor_max_gap=hueco)


def detection_accounting(
    evaluator: Any, detections: Sequence[Mapping[str, Any]], num_raw_detections: int | None
) -> dict[str, Any]:
    """Cuadra las detecciones evaluadas con las que escribió el validador.

    pycocotools trunca a maxDet por imagen y clase, así que se evalúan menos de las producidas; la
    diferencia debe ser exactamente el truncado.
    """
    from collections import Counter

    maximo = int(max(evaluator.params.maxDets))
    por_par = Counter((d["image_id"], d["category_id"]) for d in detections)
    truncadas = sum(max(0, n - maximo) for n in por_par.values())
    evaluadas = int(evaluation_facts(evaluator).scores.size)
    cuadra = len(detections) - truncadas == evaluadas
    return {
        "max_detections_per_image_and_class": maximo,
        "num_raw_detections": num_raw_detections,
        "num_reconstructed_detections": len(detections),
        "num_truncated_by_max_det": truncadas,
        "num_evaluated_detections": evaluadas,
        "pairs_over_max_det": sum(1 for n in por_par.values() if n > maximo),
        "closes_without_remainder": cuadra,
        "matches_recorded_raw_count": (
            num_raw_detections is None or int(num_raw_detections) == len(detections)
        ),
    }


def error_decomposition(facts: DecompositionFacts, confidence: float) -> dict[str, Any]:
    """Reparte en las seis clases los errores por encima de un umbral.

    Comprueba en cada llamada que las cinco de falso positivo suman el total y que al agruparlas se
    reproduce fp_taxonomy.
    """
    import numpy as np

    base = facts.facts
    seleccion = (~base.is_tp) & (base.scores >= confidence)
    propia = base.max_iou[seleccion]
    ajena = facts.max_iou_other_class[seleccion]

    es_duplicado = propia >= DUPLICATE_IOU
    es_localizacion = (~es_duplicado) & (propia >= SPURIOUS_IOU)
    resto = (~es_duplicado) & (~es_localizacion)
    es_clasificacion = resto & (ajena >= MATCHING_IOU)
    es_ambos = resto & (~es_clasificacion) & (ajena >= SPURIOUS_IOU)
    es_fondo = resto & (~es_clasificacion) & (~es_ambos)

    recuentos = {
        "classification": int(np.count_nonzero(es_clasificacion)),
        "localization": int(np.count_nonzero(es_localizacion)),
        "both": int(np.count_nonzero(es_ambos)),
        "duplicate": int(np.count_nonzero(es_duplicado)),
        "background": int(np.count_nonzero(es_fondo)),
        "missed": facts.missed_at(confidence),
    }

    total = int(propia.size)
    suma = sum(recuentos[k] for k in FALSE_POSITIVE_KINDS)
    if suma != total:
        raise ErrorDecompositionError(
            f"El reparto de falsos positivos a confianza {confidence} no cierra: "
            f"{suma} repartidos frente a {total} falsos positivos.",
            "Las cinco clases de falso positivo son excluyentes y exhaustivas "
            "por construcción. Un reparto que no suma no es un reparto.",
        )

    grueso = fp_taxonomy(base, confidence)
    equivalencias = (
        ("duplicate", recuentos["duplicate"], grueso["duplicate"]),
        ("poor_localization", recuentos["localization"], grueso["poor_localization"]),
        (
            "spurious",
            recuentos["classification"] + recuentos["both"] + recuentos["background"],
            grueso["spurious"],
        ),
    )
    for etiqueta, fino, gordo in equivalencias:
        if fino != gordo:
            raise ErrorDecompositionError(
                f"El reparto en seis clases no reproduce el de tres a confianza {confidence}: "
                f"agrupando da {fino} para '{etiqueta}' y fp_taxonomy da {gordo}.",
                "El reparto debe refinar exactamente el de results/raw/matching_ranking/.",
            )

    return {
        "confidence": confidence,
        **recuentos,
        "false_positives_total": total,
        "num_gt": int(base.num_gt),
        "true_positives": base.tp_at(confidence),
        "coarse_grained_matches": True,
    }


def decompose(
    evaluator: Any, confidences: Sequence[float] = DECOMPOSITION_CONFIDENCES
) -> dict[str, Any]:
    """Reparte los errores de una evaluación en cada umbral de la rejilla.

    Por defecto usa DECOMPOSITION_CONFIDENCES de matching_analysis.
    """
    facts = decomposition_facts(evaluator)
    por_umbral = [error_decomposition(facts, float(c)) for c in confidences]
    logger.info(
        "Reparto en seis clases sobre %d detecciones evaluadas y %d anotaciones, "
        "en %d umbrales.",
        int(facts.facts.scores.size),
        int(facts.facts.num_gt),
        len(por_umbral),
    )
    return {
        "kinds": list(ERROR_KINDS),
        "registry_columns": dict(REGISTRY_COLUMNS),
        "thresholds": {
            "matching_iou": MATCHING_IOU,
            "duplicate_iou": DUPLICATE_IOU,
            "spurious_iou": SPURIOUS_IOU,
            "order": (
                "propia>=duplicate_iou -> duplicate; propia>=spurious_iou -> "
                "localization; ajena>=matching_iou -> classification; "
                "ajena>=spurious_iou -> both; resto -> background"
            ),
        },
        "aggregation": AGGREGATION,
        "num_evaluated_detections": int(facts.facts.scores.size),
        "num_gt": int(facts.facts.num_gt),
        "own_iou_anchor_max_gap": facts.own_iou_anchor_max_gap,
        "matched_iou_min": facts.facts.matched_iou_min,
        "by_confidence": por_umbral,
    }


def cross_check_curve(
    decomposition: Mapping[str, Any], curve_points: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Contrasta aciertos y falsos positivos del reparto con los recuentos de la curva del paso 23.

    La curva cuenta clase a clase sobre dtMatches, por otro camino. Se comparan recuentos y no el
    recall, que en la curva es la media por clase. Falla si no hay umbrales en común o si algún
    recuento difiere.
    """
    de_la_curva = {round(float(p["confidence"]), 6): p for p in curve_points}
    comprobados: list[float] = []
    for fila in decomposition["by_confidence"]:
        punto = de_la_curva.get(round(float(fila["confidence"]), 6))
        if punto is None:
            continue
        comprobados.append(float(fila["confidence"]))
        for mio, suyo in (
            ("true_positives", "true_positives"),
            ("false_positives_total", "false_positives"),
            ("num_gt", "num_gt"),
        ):
            if int(fila[mio]) != int(punto[suyo]):
                raise ErrorDecompositionError(
                    f"A confianza {fila['confidence']}, el reparto da {mio}="
                    f"{fila[mio]} y la curva {suyo}={punto[suyo]}.",
                    "Son dos lecturas de las mismas detecciones y deben dar los mismos recuentos.",
                )
    if not comprobados:
        raise ErrorDecompositionError(
            "El reparto y la curva no tienen ningún umbral en común.",
            "Un contraste que no compara nada no se da por pasado (mecanismo 16).",
        )
    return {"confidences_checked": comprobados, "matches": True}
