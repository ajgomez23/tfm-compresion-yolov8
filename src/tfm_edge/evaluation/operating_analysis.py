"""Análisis de las curvas a umbral fijo de results/raw/operating_curve.

Produce tres derivaciones por par de modelos: qué eje cae más a umbral común (axis_delta), cuántas
detecciones cruzan el umbral y de qué tipo (detection_shift) y cuánto se recupera al reajustar el
umbral (retune_recovery). Se dan ΔP y ΔR y no su cociente, que se dispara cuando el recall apenas
cambia. Los pares salen de parent_experiment_id en el registro, y se vuelven a exigir las
comprobaciones de anclaje de cada volcado.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import parse_experiment_id
from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE, point_at

logger = get_logger(__name__)

#: Umbrales en los que se comprueba que el desplazamiento de confianzas mantiene el signo.
SHIFT_CONFIDENCES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

QUANTIZING_TECHNIQUES: tuple[str, ...] = ("ptq", "qat", "combined")

PRUNING_TECHNIQUES: tuple[str, ...] = ("pruning",)

KIND_QUANTIZATION = "cuantizacion"
KIND_PRUNING = "poda"


class OperatingAnalysisError(TFMEdgeError):
    """Error al derivar conclusiones de las curvas a umbral fijo."""


def variant_of(experiment_id: str) -> str:
    """Variante de un experiment_id, por ejemplo ``l1global_r50_iter4``."""
    return parse_experiment_id(experiment_id)["variant"]


def load_curve_dumps(directory: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Lee los volcados de curvas indexados por ``(experiment_id, runtime, precision)``.

    La precisión forma parte de la clave porque un experimento QAT tiene curva en FP32 y en INT8.
    Falla si dos volcados comparten clave o si alguno no pasó el anclaje o la comprobación del
    recall.
    """
    if not directory.is_dir():
        raise OperatingAnalysisError(
            f"No existe el directorio de curvas '{directory}'.",
            "Mide las curvas primero: python scripts/23_operating_curve.py --all",
        )

    volcados: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ruta in sorted(directory.glob("*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        clave = (str(datos["experiment_id"]), str(datos["runtime"]), str(datos["precision"]))
        if clave in volcados:
            raise OperatingAnalysisError(
                f"Dos volcados para {clave} ('{ruta.name}' y otro anterior).",
                "Deja un solo volcado por clave en el directorio.",
            )
        ancla = datos.get("anchor", {})
        if not ancla.get("anchored"):
            raise OperatingAnalysisError(
                f"'{ruta.name}' no está anclado contra lo registrado.",
                "Vuelve a medir esa curva. Una reconstrucción sin ancla produce "
                "números plausibles sin haber pasado por donde pasó la medida.",
            )
        verificacion = datos.get("curve", {}).get("verification", {})
        if not verificacion.get("recall_matches_pycocotools"):
            raise OperatingAnalysisError(
                f"'{ruta.name}' no trae comprobada la identidad del recall.",
                "Vuelve a medir esa curva: el recuento propio sobre evalImgs solo "
                "vale si coincide con el que guarda accumulate.",
            )
        volcados[clave] = datos

    if not volcados:
        raise OperatingAnalysisError(
            f"No hay ningún volcado de curva en '{directory}'.",
            "Mide las curvas primero: python scripts/23_operating_curve.py --all",
        )
    logger.info("Leídos %d volcados de curva de '%s'", len(volcados), directory)
    return volcados


def derive_pairs(
    rows: Sequence[dict[str, str]], curves: dict[tuple[str, str, str], dict[str, Any]]
) -> list[dict[str, str]]:
    """Deriva del registro los pares a comparar.

    Un par existe si la fila declara padre, las dos están en estado ok y ambos lados tienen curva en
    el mismo motor y en la precisión que corresponde (FP32 a INT8 en cuantización, FP32 en los dos
    lados en poda).
    """
    estado = {fila["experiment_id"]: fila.get("status", "") for fila in rows}
    motores = sorted({motor for _, motor, _ in curves})
    pares: list[dict[str, str]] = []

    for fila in rows:
        hijo = fila["experiment_id"]
        padre = fila.get("parent_experiment_id", "")
        if not padre or fila.get("status") != "ok" or estado.get(padre) != "ok":
            continue
        tecnica = fila.get("technique", "")
        if tecnica in QUANTIZING_TECHNIQUES:
            kind = KIND_QUANTIZATION
            # En cuantización el par se etiqueta con el modelo de partida.
            label = variant_of(padre)
            precision_antes, precision_despues = "fp32", "int8"
        elif tecnica in PRUNING_TECHNIQUES:
            kind = KIND_PRUNING
            label = variant_of(hijo)
            precision_antes, precision_despues = "fp32", "fp32"
        else:
            continue

        for motor in motores:
            clave_antes = (padre, motor, precision_antes)
            clave_despues = (hijo, motor, precision_despues)
            if clave_antes not in curves or clave_despues not in curves:
                logger.debug(
                    "Sin par para %s en %s: falta la curva de %s",
                    hijo,
                    motor,
                    clave_antes if clave_antes not in curves else clave_despues,
                )
                continue
            pares.append(
                {
                    "kind": kind,
                    "label": label,
                    "via": tecnica,
                    "before": padre,
                    "before_precision": precision_antes,
                    "after": hijo,
                    "after_precision": precision_despues,
                    "runtime": motor,
                }
            )

    pares.sort(key=lambda p: (p["kind"], p["label"], p["via"], p["runtime"]))
    logger.info(
        "Derivados %d pares del registro (%d de cuantización, %d de poda)",
        len(pares),
        sum(1 for p in pares if p["kind"] == KIND_QUANTIZATION),
        sum(1 for p in pares if p["kind"] == KIND_PRUNING),
    )
    return pares


def unused_curves(
    curves: dict[tuple[str, str, str], dict[str, Any]], pairs: Sequence[dict[str, str]]
) -> list[tuple[str, str, str]]:
    """Curvas medidas que no entran en ningún par, para que se vea lo que no se usa."""
    usadas: set[tuple[str, str, str]] = set()
    for par in pairs:
        usadas.add((par["before"], par["runtime"], par["before_precision"]))
        usadas.add((par["after"], par["runtime"], par["after_precision"]))
    return sorted(set(curves) - usadas)


def axis_delta(
    before: dict[str, Any], after: dict[str, Any], confidence: float = REFERENCE_CONFIDENCE
) -> dict[str, Any]:
    """Precisión y recall de los dos lados a un umbral común y qué eje cae más
    (``dominant_axis``)."""
    antes = point_at(before["curve"], confidence)
    despues = point_at(after["curve"], confidence)
    delta_p = despues["precision"] - antes["precision"]
    delta_r = despues["recall"] - antes["recall"]
    return {
        "confidence": confidence,
        "precision_before": antes["precision"],
        "precision_after": despues["precision"],
        "recall_before": antes["recall"],
        "recall_after": despues["recall"],
        "delta_precision": delta_p,
        "delta_recall": delta_r,
        "dominant_axis": "precision" if abs(delta_p) > abs(delta_r) else "recall",
    }


def detection_shift(
    before: dict[str, Any], after: dict[str, Any], confidence: float = REFERENCE_CONFIDENCE
) -> dict[str, Any]:
    """Detecciones por encima del umbral en cada lado (TP + FP) y su cambio.

    ``relative_change`` es None si el lado de partida no tiene ninguna detección.
    """
    antes = point_at(before["curve"], confidence)
    despues = point_at(after["curve"], confidence)
    total_antes = antes["true_positives"] + antes["false_positives"]
    total_despues = despues["true_positives"] + despues["false_positives"]
    return {
        "confidence": confidence,
        "detections_before": total_antes,
        "detections_after": total_despues,
        "relative_change": ((total_despues - total_antes) / total_antes if total_antes else None),
        "delta_true_positives": despues["true_positives"] - antes["true_positives"],
        "delta_false_positives": despues["false_positives"] - antes["false_positives"],
    }


def best_f1_point(curve_dump: dict[str, Any]) -> dict[str, Any]:
    """Punto de la rejilla con mayor F1. Es el máximo sobre la rejilla, no el óptimo exacto de la
    curva.
    """
    return max(curve_dump["curve"]["points"], key=lambda p: p["f1"])


def retune_recovery(
    before: dict[str, Any], after: dict[str, Any], confidence: float = REFERENCE_CONFIDENCE
) -> dict[str, Any]:
    """Parte de la brecha de F1 a umbral fijo que desaparece si cada modelo usa su umbral óptimo.

    ``recovered_fraction`` es None si no había brecha.
    """
    f1_fijo_antes = point_at(before["curve"], confidence)["f1"]
    f1_fijo_despues = point_at(after["curve"], confidence)["f1"]
    mejor_antes = best_f1_point(before)
    mejor_despues = best_f1_point(after)

    brecha_fija = f1_fijo_despues - f1_fijo_antes
    brecha_reajustada = mejor_despues["f1"] - mejor_antes["f1"]
    conf_antes = mejor_antes["confidence"]
    conf_despues = mejor_despues["confidence"]
    if conf_despues > conf_antes:
        movimiento = "sube"
    elif conf_despues < conf_antes:
        movimiento = "baja"
    else:
        movimiento = "igual"

    return {
        "confidence": confidence,
        "f1_gap_fixed": brecha_fija,
        "f1_gap_retuned": brecha_reajustada,
        "best_confidence_before": conf_antes,
        "best_confidence_after": conf_despues,
        "optimum_moves": movimiento,
        "recovered_fraction": (
            1.0 - brecha_reajustada / brecha_fija if abs(brecha_fija) > 1e-9 else None
        ),
    }


def _resumen_cuantizacion(entradas: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega los pares de cuantización. Lo que no se puede calcular queda en None."""
    if not entradas:
        return {"pairs": 0}

    dominantes = [e["axis"]["dominant_axis"] for e in entradas]
    cambios = [e["shift"]["relative_change"] for e in entradas]
    cambios_validos = [c for c in cambios if c is not None]
    recuperados = [
        e["retune"]["recovered_fraction"]
        for e in entradas
        if e["retune"]["recovered_fraction"] is not None
    ]
    movimientos = [e["retune"]["optimum_moves"] for e in entradas]
    delta_tp = [e["shift"]["delta_true_positives"] for e in entradas]
    delta_fp = [e["shift"]["delta_false_positives"] for e in entradas]

    return {
        "pairs": len(entradas),
        "precision_dominated": dominantes.count("precision"),
        "recall_dominated": dominantes.count("recall"),
        "detections_up": sum(1 for c in cambios_validos if c > 0),
        "detections_compared": len(cambios_validos),
        "relative_change_min": min(cambios_validos) if cambios_validos else None,
        "relative_change_max": max(cambios_validos) if cambios_validos else None,
        "delta_true_positives_min": min(delta_tp),
        "delta_true_positives_max": max(delta_tp),
        "delta_false_positives_min": min(delta_fp),
        "delta_false_positives_max": max(delta_fp),
        "recovered_median": statistics.median(recuperados) if recuperados else None,
        "recovered_min": min(recuperados) if recuperados else None,
        "recovered_max": max(recuperados) if recuperados else None,
        "recovered_compared": len(recuperados),
        "optimum_up": movimientos.count("sube"),
        "optimum_equal": movimientos.count("igual"),
        "optimum_down": movimientos.count("baja"),
    }


def _resumen_poda(entradas: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega los pares de poda: cuántos caen por precisión y cuántos por recall."""
    if not entradas:
        return {"pairs": 0}
    dominantes = [e["axis"]["dominant_axis"] for e in entradas]
    return {
        "pairs": len(entradas),
        "precision_dominated": dominantes.count("precision"),
        "recall_dominated": dominantes.count("recall"),
    }


def analyze(
    curves: dict[tuple[str, str, str], dict[str, Any]],
    pairs: Sequence[dict[str, str]],
    confidence: float = REFERENCE_CONFIDENCE,
    shift_confidences: Sequence[float] = SHIFT_CONFIDENCES,
) -> dict[str, Any]:
    """Calcula las tres derivaciones de cada par, los agregados y las curvas no usadas."""
    entradas: list[dict[str, Any]] = []
    for par in pairs:
        clave_antes = (par["before"], par["runtime"], par["before_precision"])
        clave_despues = (par["after"], par["runtime"], par["after_precision"])
        if clave_antes not in curves or clave_despues not in curves:
            raise OperatingAnalysisError(
                f"El par {par['before']} / {par['after']} en {par['runtime']} referencia una "
                "curva que no está.",
                "Mide las curvas que falten antes de derivar nada.",
            )
        antes, despues = curves[clave_antes], curves[clave_despues]

        entrada: dict[str, Any] = dict(par)
        entrada["axis"] = axis_delta(antes, despues, confidence)
        entrada["shift"] = detection_shift(antes, despues, confidence)
        entrada["shift_across_confidences"] = [
            detection_shift(antes, despues, c) for c in shift_confidences
        ]
        entrada["retune"] = retune_recovery(antes, despues, confidence)
        entradas.append(entrada)

    de_cuantizacion = [e for e in entradas if e["kind"] == KIND_QUANTIZATION]
    de_poda = [e for e in entradas if e["kind"] == KIND_PRUNING]

    sin_usar = unused_curves(curves, pairs)
    if sin_usar:
        logger.info(
            "%d curvas medidas no entran en ningún par: %s",
            len(sin_usar),
            ", ".join("/".join(c) for c in sin_usar),
        )

    return {
        "confidence": confidence,
        "shift_confidences": list(shift_confidences),
        "num_curves": len(curves),
        "pairs": entradas,
        "summary": {
            KIND_QUANTIZATION: _resumen_cuantizacion(de_cuantizacion),
            KIND_PRUNING: _resumen_poda(de_poda),
        },
        "unused_curves": [
            {"experiment_id": e, "runtime": r, "precision": p} for e, r, p in sin_usar
        ],
    }
