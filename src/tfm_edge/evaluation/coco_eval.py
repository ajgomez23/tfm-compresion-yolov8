"""Evaluación COCO con dos criterios de tamaño de objeto.

Es la única fuente de las doce columnas de desglose por tamaño del registro. El criterio primario
mide el tamaño a la resolución de inferencia y el secundario en la imagen original. Las detecciones
se generan una vez en coordenadas originales; para el criterio primario se escalan anotaciones y
detecciones por el mismo factor de letterbox, lo que no cambia el emparejamiento porque la IoU es
invariante al escalado uniforme. Por eso el mAP global y los AR deben coincidir en las dos
evaluaciones, y assert_criteria_consistency lo comprueba.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Criterio primario.
PRIMARY_CRITERION = "inference_resolution"

#: Los dos criterios con el sufijo de sus columnas en el registro.
CRITERIA = {"inference_resolution": "inference_res", "original_resolution": "original_res"}

#: Tolerancia de la coherencia entre criterios; solo cubre el redondeo al escalar.
CONSISTENCY_TOLERANCE = 1e-6

#: IoU de la curva precisión-recall del punto de operación, la misma que usa Ultralytics.
OPERATING_POINT_IOU = 0.5

#: Índices de COCOeval.stats.
STAT_INDICES = {
    "map_50_95": 0,
    "map_50": 1,
    "ap_75": 2,
    "ap_small": 3,
    "ap_medium": 4,
    "ap_large": 5,
    "ar_1": 6,
    "ar_10": 7,
    "ar_100": 8,
    "ar_small": 9,
    "ar_medium": 10,
    "ar_large": 11,
}


class CocoEvalError(TFMEdgeError):
    """Error durante la evaluación COCO."""


@dataclass
class CriterionMetrics:
    """Métricas de un criterio de tamaño. Los AP y AR por tamaño son los de este criterio."""

    criterion: str
    map_50_95: float = 0.0
    map_50: float = 0.0
    ap_75: float = 0.0
    ap_small: float | str = ""
    ap_medium: float | str = ""
    ap_large: float | str = ""
    ar_1: float = 0.0
    ar_10: float = 0.0
    ar_100: float = 0.0
    ar_small: float | str = ""
    ar_medium: float | str = ""
    ar_large: float | str = ""
    objects_per_size: dict[str, int] = field(default_factory=dict)
    precision_at_best_f1: float | str = ""
    recall_at_best_f1: float | str = ""
    f1_best: float | str = ""
    conf_at_best_f1: float | str = ""
    operating_point: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve las métricas como diccionario serializable."""
        return asdict(self)


@dataclass
class CocoEvaluationResult:
    """Resultado de la evaluación con los dos criterios."""

    split: str
    input_resolution: int
    num_images: int
    num_annotations: int
    num_detections: int
    by_criterion: dict[str, CriterionMetrics] = field(default_factory=dict)
    per_class: dict[int, dict[str, float]] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)
    conf_threshold: float = 0.001
    iou_threshold: float = 0.7

    @property
    def primary(self) -> CriterionMetrics:
        """Métricas del criterio primario (resolución de inferencia)."""
        return self.by_criterion[PRIMARY_CRITERION]

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        return {
            "split": self.split,
            "input_resolution": self.input_resolution,
            "num_images": self.num_images,
            "num_annotations": self.num_annotations,
            "num_detections": self.num_detections,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "size_criterion_primary": PRIMARY_CRITERION,
            "by_criterion": {k: v.to_dict() for k, v in self.by_criterion.items()},
            "per_class": self.per_class,
            "consistency": self.consistency,
        }

    def to_registry_columns(self) -> dict[str, Any]:
        """Columnas de quality_metrics.csv. Es el único sitio que rellena las doce columnas por
        tamaño.
        """
        columnas: dict[str, Any] = {
            "map_50": round(self.primary.map_50, 6),
            "map_50_95": round(self.primary.map_50_95, 6),
            "ap_75": round(self.primary.ap_75, 6),
            "size_criterion_primary": PRIMARY_CRITERION,
            "size_criterion_inference_resolution": self.input_resolution,
            "ar_1": round(self.primary.ar_1, 6),
            "ar_10": round(self.primary.ar_10, 6),
            "ar_100": round(self.primary.ar_100, 6),
            "num_images": self.num_images,
            "num_detections": self.num_detections,
        }
        # El punto de operación no depende del criterio (área 'all'); se toma el del primario y la
        # comprobación de coherencia verifica que el otro coincide.
        for nombre in ("precision_at_best_f1", "recall_at_best_f1", "f1_best"):
            valor = getattr(self.primary, nombre)
            columnas[nombre] = round(valor, 6) if isinstance(valor, float) else ""
        for criterio, sufijo in CRITERIA.items():
            metricas = self.by_criterion.get(criterio)
            if metricas is None:
                continue
            for tamano in ("small", "medium", "large"):
                for prefijo in ("ap", "ar"):
                    valor = getattr(metricas, f"{prefijo}_{tamano}")
                    columnas[f"{prefijo}_{tamano}_{sufijo}"] = (
                        round(valor, 6) if isinstance(valor, float) else ""
                    )
        return columnas

    def summary(self) -> str:
        """Resumen legible con los dos criterios en paralelo."""
        lineas = [
            f"EVALUACIÓN COCO: partición '{self.split}' a {self.input_resolution} px",
            f"{self.num_images:,} imágenes, {self.num_annotations:,} anotaciones, "
            f"{self.num_detections:,} detecciones",
            "",
            f"mAP@50    : {self.primary.map_50:.4f}",
            f"mAP@50-95 : {self.primary.map_50_95:.4f}",
            f"AP@75     : {self.primary.ap_75:.4f}",
            "",
            f"Punto de operación de mejor F1 (IoU {OPERATING_POINT_IOU:.2f}, umbral "
            f"único para todas las clases): P={_fmt(self.primary.precision_at_best_f1)} "
            f"R={_fmt(self.primary.recall_at_best_f1)} F1={_fmt(self.primary.f1_best)} "
            f"con conf={_fmt(self.primary.conf_at_best_f1)}",
            "",
            "Desglose por tamaño, con los dos criterios:",
            "",
            f"{'criterio':22s} {'AP peq.':>9s} {'AP med.':>9s} {'AP gr.':>9s}   "
            f"{'objetos peq/med/gr':>22s}",
            "-" * 78,
        ]
        for criterio in ("inference_resolution", "original_resolution"):
            m = self.by_criterion.get(criterio)
            if m is None:
                continue
            marca = " *" if criterio == PRIMARY_CRITERION else "  "
            reparto = m.objects_per_size
            lineas.append(
                f"{criterio + marca:22s} "
                f"{_fmt(m.ap_small):>9} {_fmt(m.ap_medium):>9} {_fmt(m.ap_large):>9}   "
                f"{reparto.get('small', 0):>7,} {reparto.get('medium', 0):>7,} "
                f"{reparto.get('large', 0):>6,}"
            )
        lineas += [
            "",
            "* criterio primario.",
            "",
            f"Coherencia entre criterios: "
            f"{'OK' if self.consistency.get('consistent') else 'FALLA'} "
            f"(mAP@50-95 idéntico, diferencia máxima "
            f"{self.consistency.get('max_difference', '?')})",
        ]
        return "\n".join(lineas)


def _fmt(valor: float | str) -> str:
    """Formatea una métrica que puede estar ausente."""
    return f"{valor:.4f}" if isinstance(valor, float) else "n/d"


# Escalado a coordenadas de inferencia


def letterbox_scale(width: int, height: int, resolution: int) -> float:
    """Factor de escala del letterbox: ``min(R/W, R/H)``, igual en los dos ejes."""
    if width <= 0 or height <= 0:
        raise CocoEvalError(
            f"Dimensiones de imagen inválidas: {width}x{height}.",
            "Revisa el COCO de referencia: alguna imagen tiene dimensiones nulas.",
        )
    return min(resolution / width, resolution / height)


def scale_ground_truth(gt: dict[str, Any], resolution: int) -> dict[str, Any]:
    """Escala cajas y áreas de las anotaciones con el factor de letterbox de cada imagen.

    Las áreas se recalculan desde la caja escalada.
    """
    escalado = copy.deepcopy(gt)
    factores = {
        imagen["id"]: letterbox_scale(imagen["width"], imagen["height"], resolution)
        for imagen in escalado["images"]
    }
    for imagen in escalado["images"]:
        s = factores[imagen["id"]]
        imagen["width"] = imagen["width"] * s
        imagen["height"] = imagen["height"] * s
    for anotacion in escalado["annotations"]:
        s = factores[anotacion["image_id"]]
        x, y, w, h = anotacion["bbox"]
        anotacion["bbox"] = [x * s, y * s, w * s, h * s]
        anotacion["area"] = (w * s) * (h * s)
    return escalado


def scale_detections(
    detections: Sequence[dict[str, Any]], gt: dict[str, Any], resolution: int
) -> list[dict[str, Any]]:
    """Escala las detecciones igual que scale_ground_truth."""
    factores = {
        imagen["id"]: letterbox_scale(imagen["width"], imagen["height"], resolution)
        for imagen in gt["images"]
    }
    escaladas = []
    for deteccion in detections:
        s = factores[deteccion["image_id"]]
        x, y, w, h = deteccion["bbox"]
        copia = dict(deteccion)
        copia["bbox"] = [x * s, y * s, w * s, h * s]
        escaladas.append(copia)
    return escaladas


def count_objects_per_size(gt: dict[str, Any]) -> dict[str, int]:
    """Número de anotaciones en cada rango de tamaño COCO."""
    reparto = {"small": 0, "medium": 0, "large": 0}
    for anotacion in gt["annotations"]:
        area = anotacion["area"]
        if area < 32**2:
            reparto["small"] += 1
        elif area < 96**2:
            reparto["medium"] += 1
        else:
            reparto["large"] += 1
    return reparto


# Evaluación


def _importar_pycocotools() -> tuple[Any, Any]:
    """Importa pycocotools o lanza CocoEvalError. Devuelve ``(COCO, COCOeval)``."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise CocoEvalError(
            "pycocotools no está instalado.",
            "Es obligatorio: la especificación exige métricas COCO. "
            "Instálalo con: pip install pycocotools==2.0.11",
        ) from error
    return COCO, COCOeval


def operating_point_at_best_f1(evaluator: Any) -> dict[str, Any]:
    """Punto de operación de mejor F1 sobre las curvas de pycocotools.

    Es el umbral de confianza único para todas las clases que maximiza el F1 medio, a IoU 0,50, área
    ``all`` y el mayor maxDets. Se usan las matrices ``precision`` y ``scores`` de accumulate para
    recorrer la curva en función de la confianza, y se promedia sobre las clases con anotaciones. No
    es comparable con la cifra de Ultralytics: pycocotools monotoniza la precisión y Ultralytics
    suaviza la curva de F1. Si no se puede calcular, los valores quedan vacíos con el motivo en
    ``reason``.
    """
    import numpy as np

    acumulado = getattr(evaluator, "eval", None)
    if not acumulado or "precision" not in acumulado:
        raise CocoEvalError(
            "Se ha pedido el punto de operación de una evaluación sin acumular.",
            "Llama a COCOeval.accumulate() antes que a operating_point_at_best_f1().",
        )

    parametros = evaluator.params
    coincidencias = np.flatnonzero(np.isclose(parametros.iouThrs, OPERATING_POINT_IOU))
    if coincidencias.size == 0:
        raise CocoEvalError(
            f"La evaluación no incluye IoU={OPERATING_POINT_IOU}. Umbrales presentes: "
            f"{list(parametros.iouThrs)}.",
            "El punto de operación se define sobre la curva P-R a IoU 0,50.",
        )
    t = int(coincidencias[0])
    a = list(parametros.areaRngLbl).index("all")
    m = int(np.argmax(parametros.maxDets))

    precision = acumulado["precision"][t][:, :, a, m]  # (R, K)
    scores = acumulado["scores"][t][:, :, a, m]  # (R, K)

    # pycocotools deja -1 en las clases sin anotaciones; no se promedian.
    con_anotaciones = np.flatnonzero(precision[0] >= 0)
    if con_anotaciones.size == 0:
        return {
            "precision": "",
            "recall": "",
            "f1": "",
            "confidence": "",
            "num_classes": 0,
            "reason": "ninguna clase tiene anotaciones en esta partición",
        }

    prec = precision[:, con_anotaciones].T  # (K, R)
    conf = scores[:, con_anotaciones].T  # (K, R)
    rejilla_recall = np.asarray(parametros.recThrs, dtype=float)

    # Confianza 0 es el relleno de accumulate en los recall no alcanzados; no es un umbral
    # candidato.
    candidatos = np.unique(conf[conf > 0])
    if candidatos.size == 0:
        return {
            "precision": "",
            "recall": "",
            "f1": "",
            "confidence": "",
            "num_classes": int(con_anotaciones.size),
            "reason": "no hay detecciones con confianza positiva",
        }

    # scores no crece con el recall, así que contar los puntos que superan el umbral da el mayor
    # recall alcanzable.
    alcanzables = (conf[None, :, :] >= candidatos[:, None, None]).sum(axis=2) - 1  # (C, K)
    hay_punto = alcanzables >= 0
    indice = np.clip(alcanzables, 0, rejilla_recall.size - 1)
    clases = np.arange(prec.shape[0])[None, :]

    # Sin detecciones por encima del umbral: recall 0 y precisión 1.
    p = np.where(hay_punto, prec[clases, indice], 1.0)
    r = np.where(hay_punto, rejilla_recall[indice], 0.0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)

    # Con candidatos en orden ascendente, argmax desempata a favor del umbral más bajo (más recall),
    # que es lo que conviene en detección de EPI.
    mejor = int(np.argmax(f1.mean(axis=1)))
    return {
        "precision": round(float(p[mejor].mean()), 6),
        "recall": round(float(r[mejor].mean()), 6),
        "f1": round(float(f1[mejor].mean()), 6),
        "confidence": round(float(candidatos[mejor]), 6),
        "num_classes": int(con_anotaciones.size),
        "iou": OPERATING_POINT_IOU,
        "max_detections": int(parametros.maxDets[m]),
        "reason": "",
    }


def build_evaluator(
    gt: dict[str, Any], detections: Sequence[dict[str, Any]], criterio: str = ""
) -> Any:
    """Construye un COCOeval ya evaluado, acumulado y resumido.

    Lo comparten las columnas del registro y operating_curve, para que usen el mismo evaluador.
    """
    import contextlib
    import io

    COCO, COCOeval = _importar_pycocotools()

    coco_gt = COCO()
    coco_gt.dataset = gt
    # pycocotools imprime por stdout al indexar; se silencia.
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    if not detections:
        raise CocoEvalError(
            f"No hay detecciones que evaluar bajo el criterio '{criterio}'.",
            "Un modelo entrenado debería producir miles con conf=0.001.",
        )

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(list(detections))
            evaluador = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluador.evaluate()
            evaluador.accumulate()
            evaluador.summarize()
    except Exception as error:
        raise CocoEvalError(
            f"pycocotools ha fallado en el criterio '{criterio}': "
            f"{type(error).__name__}: {error}",
            "Suele significar que los identificadores de imagen o de categoría de "
            "las detecciones no existen en el COCO de referencia.",
        ) from error
    return evaluador


def _evaluar(
    gt: dict[str, Any], detections: Sequence[dict[str, Any]], criterio: str
) -> CriterionMetrics:
    """Ejecuta una evaluación COCO y devuelve las métricas del criterio."""
    evaluador = build_evaluator(gt, detections, criterio)
    stats = evaluador.stats

    def leer(nombre: str) -> float | str:
        """Lee un estadístico; -1 (no hay objetos de ese tamaño) se convierte en celda vacía."""
        valor = float(stats[STAT_INDICES[nombre]])
        return "" if valor < 0 else round(valor, 6)

    punto = operating_point_at_best_f1(evaluador)

    metricas = CriterionMetrics(
        criterion=criterio,
        map_50_95=float(stats[0]),
        map_50=float(stats[1]),
        ap_75=float(stats[2]),
        ap_small=leer("ap_small"),
        ap_medium=leer("ap_medium"),
        ap_large=leer("ap_large"),
        ar_1=float(stats[6]),
        ar_10=float(stats[7]),
        ar_100=float(stats[8]),
        ar_small=leer("ar_small"),
        ar_medium=leer("ar_medium"),
        ar_large=leer("ar_large"),
        objects_per_size=count_objects_per_size(gt),
        precision_at_best_f1=punto["precision"],
        recall_at_best_f1=punto["recall"],
        f1_best=punto["f1"],
        conf_at_best_f1=punto["confidence"],
        operating_point=punto,
    )
    return metricas


def assert_criteria_consistency(
    inferencia: CriterionMetrics, original: CriterionMetrics
) -> dict[str, Any]:
    """Comprueba que los dos criterios solo difieren en el desglose por tamaño.

    mAP@50-95, mAP@50, AP@75, AR y el punto de operación deben coincidir. Si no, probablemente solo
    se escaló uno de los dos lados.
    """
    invariantes = ["map_50_95", "map_50", "ap_75", "ar_1", "ar_10", "ar_100"]
    # El punto de operación usa el área 'all', así que tampoco puede cambiar.
    for nombre in ("precision_at_best_f1", "recall_at_best_f1", "f1_best", "conf_at_best_f1"):
        if isinstance(getattr(inferencia, nombre), float) and isinstance(
            getattr(original, nombre), float
        ):
            invariantes.append(nombre)

    diferencias = {
        nombre: abs(getattr(inferencia, nombre) - getattr(original, nombre))
        for nombre in invariantes
    }
    peor = max(diferencias.values())
    resultado = {
        "consistent": peor <= CONSISTENCY_TOLERANCE,
        "max_difference": float(f"{peor:.3e}"),
        "tolerance": CONSISTENCY_TOLERANCE,
        "differences": {k: float(f"{v:.3e}") for k, v in diferencias.items()},
        "checked": list(invariantes),
    }
    if not resultado["consistent"]:
        raise CocoEvalError(
            "Los dos criterios de tamaño dan métricas globales distintas (diferencia máxima "
            f"{peor:.3e}, tolerancia {CONSISTENCY_TOLERANCE}): {resultado['differences']}",
            "Probablemente solo se escaló uno de los dos lados. El desglose por tamaño no es "
            "válido.",
        )
    logger.info("Coherencia entre criterios verificada: diferencia máxima %.3e", peor)
    return resultado


def evaluate_per_class(
    gt: dict[str, Any], detections: Sequence[dict[str, Any]]
) -> dict[int, dict[str, float]]:
    """AP por clase: ``{índice: {ap_50, ap_50_95, num_annotations}}``."""
    import contextlib
    import io

    COCO, COCOeval = _importar_pycocotools()

    coco_gt = COCO()
    coco_gt.dataset = gt
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(list(detections))

    por_clase: dict[int, dict[str, float]] = {}
    anotaciones_por_clase: dict[int, int] = {}
    for anotacion in gt["annotations"]:
        clave = int(anotacion["category_id"])
        anotaciones_por_clase[clave] = anotaciones_por_clase.get(clave, 0) + 1

    for categoria in gt["categories"]:
        indice = int(categoria["id"])
        with contextlib.redirect_stdout(io.StringIO()):
            evaluador = COCOeval(coco_gt, coco_dt, iouType="bbox")
            evaluador.params.catIds = [indice]
            evaluador.evaluate()
            evaluador.accumulate()
            evaluador.summarize()
        por_clase[indice] = {
            "name": categoria["name"],
            "ap_50_95": round(float(evaluador.stats[0]), 6),
            "ap_50": round(float(evaluador.stats[1]), 6),
            "num_annotations": anotaciones_por_clase.get(indice, 0),
        }
    return por_clase


def evaluate_coco(
    ground_truth: dict[str, Any] | Path | str,
    detections: Sequence[dict[str, Any]],
    input_resolution: int,
    split: str = "val",
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    with_per_class: bool = True,
) -> CocoEvaluationResult:
    """Evalúa detecciones con los dos criterios de tamaño.

    Referencia y detecciones van en coordenadas originales. ``with_per_class`` se puede desactivar
    en pruebas porque añade una evaluación por clase.
    """
    if isinstance(ground_truth, (str, Path)):
        ruta = Path(ground_truth)
        if not ruta.is_file():
            raise CocoEvalError(
                f"No existe el COCO de referencia '{ruta}'.",
                "Genéralo con scripts/02_prepare_dataset.py.",
            )
        gt = json.loads(ruta.read_text(encoding="utf-8"))
    else:
        gt = ground_truth

    for clave in ("images", "annotations", "categories"):
        if clave not in gt:
            raise CocoEvalError(
                f"El COCO de referencia no tiene la clave '{clave}'.",
                "No es un fichero COCO válido.",
            )

    logger.info(
        "Evaluando %s detecciones sobre %s imágenes a %d px, con los dos criterios",
        f"{len(detections):,}",
        f"{len(gt['images']):,}",
        input_resolution,
    )

    # Criterio secundario: coordenadas originales.
    metricas_original = _evaluar(gt, detections, "original_resolution")

    # Criterio primario: los dos lados escalados por el mismo factor.
    gt_inferencia = scale_ground_truth(gt, input_resolution)
    dt_inferencia = scale_detections(detections, gt, input_resolution)
    metricas_inferencia = _evaluar(gt_inferencia, dt_inferencia, "inference_resolution")

    coherencia = assert_criteria_consistency(metricas_inferencia, metricas_original)

    por_clase: dict[int, dict[str, float]] = {}
    if with_per_class:
        por_clase = evaluate_per_class(gt_inferencia, dt_inferencia)

    return CocoEvaluationResult(
        split=split,
        input_resolution=input_resolution,
        num_images=len(gt["images"]),
        num_annotations=len(gt["annotations"]),
        num_detections=len(detections),
        by_criterion={
            "inference_resolution": metricas_inferencia,
            "original_resolution": metricas_original,
        },
        per_class=por_clase,
        consistency=coherencia,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )


# Preparación de la referencia y de las detecciones


def subset_ground_truth(gt: dict[str, Any], file_names: Sequence[str]) -> dict[str, Any]:
    """Recorta el COCO a las imágenes de una partición."""
    buscados = set(file_names)
    imagenes = [i for i in gt["images"] if i["file_name"] in buscados]
    encontrados = {i["file_name"] for i in imagenes}
    if faltan := buscados - encontrados:
        raise CocoEvalError(
            f"{len(faltan)} imágenes de la partición no están en el COCO de "
            f"referencia. Primeras: {sorted(faltan)[:3]}",
            "El COCO y las particiones tienen que venir de la misma preparación "
            "del dataset. Regenéralos con scripts/02_prepare_dataset.py.",
        )
    ids = {i["id"] for i in imagenes}
    return {
        "info": gt.get("info", {}),
        "licenses": gt.get("licenses", []),
        "categories": gt["categories"],
        "images": imagenes,
        "annotations": [a for a in gt["annotations"] if a["image_id"] in ids],
    }


def rescale_detections_to_original(
    detections: Sequence[dict[str, Any]],
    gt: dict[str, Any],
    images_dir: Path | str,
    tolerance: float = 0.02,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Lleva las detecciones a coordenadas de la imagen original.

    Ultralytics reescala a la imagen en disco, que es la copia de 640 px, mientras que el COCO tiene
    las dimensiones originales; sin esta corrección el mAP sale cero. El factor se mide comparando
    el tamaño real del fichero con el del COCO y se comprueba que es igual en x e y.
    """
    from PIL import Image

    raiz = Path(images_dir)
    if not raiz.is_dir():
        raise CocoEvalError(
            f"No existe el directorio de imágenes '{raiz}'.",
            "Es el directorio sobre el que se generaron las detecciones.",
        )

    factores: dict[Any, float] = {}
    problemas: list[str] = []
    for imagen in gt["images"]:
        ruta = raiz / imagen["file_name"]
        if not ruta.is_file():
            problemas.append(f"falta '{imagen['file_name']}'")
            continue
        with Image.open(ruta) as abierta:
            ancho_disco, alto_disco = abierta.size
        factor_ancho = imagen["width"] / ancho_disco
        factor_alto = imagen["height"] / alto_disco
        if abs(factor_ancho - factor_alto) / max(factor_ancho, factor_alto) > tolerance:
            problemas.append(
                f"'{imagen['file_name']}': factores distintos "
                f"({factor_ancho:.4f} en x, {factor_alto:.4f} en y)"
            )
            continue
        factores[imagen["id"]] = (factor_ancho + factor_alto) / 2.0

    if problemas:
        raise CocoEvalError(
            f"{len(problemas)} imágenes impiden reescalar a coordenadas "
            f"originales. Primeras: {problemas[:3]}",
            "Los factores de x e y difieren: la copia no conserva la relación de aspecto. "
            "Regénerala con scripts/02b_resize_dataset.py.",
        )

    peor = max(factores.values()) if factores else 1.0
    menor = min(factores.values()) if factores else 1.0
    hace_falta = abs(peor - 1.0) > 1e-6 or abs(menor - 1.0) > 1e-6

    informe = {
        "rescaled": hace_falta,
        "images_dir": str(raiz),
        "factor_min": round(menor, 6),
        "factor_max": round(peor, 6),
        "num_images": len(factores),
    }
    if not hace_falta:
        logger.info(
            "Las imágenes evaluadas tienen las dimensiones originales: no hay que " "reescalar."
        )
        return list(detections), informe

    logger.info(
        "Imágenes redimensionadas (factores %.4f a %.4f): se reescalan las detecciones a "
        "coordenadas originales.",
        menor,
        peor,
    )
    reescaladas: list[dict[str, Any]] = []
    for deteccion in detections:
        factor = factores.get(deteccion["image_id"])
        if factor is None:
            continue
        x, y, w, h = deteccion["bbox"]
        copia = dict(deteccion)
        copia["bbox"] = [x * factor, y * factor, w * factor, h * factor]
        reescaladas.append(copia)
    return reescaladas, informe


def map_detections_to_ground_truth(
    detections: Sequence[dict[str, Any]], gt: dict[str, Any]
) -> list[dict[str, Any]]:
    """Traduce el ``image_id`` de las detecciones (nombre de fichero) al entero del COCO."""
    por_nombre = {i["file_name"]: i["id"] for i in gt["images"]}
    categorias = {int(c["id"]) for c in gt["categories"]}

    traducidas: list[dict[str, Any]] = []
    sin_emparejar: set[str] = set()
    categorias_raras: set[int] = set()

    for deteccion in detections:
        nombre = deteccion.get("file_name")
        if nombre not in por_nombre:
            # Las detecciones de otra partición se descartan; solo se avisa si no se empareja
            # ninguna.
            sin_emparejar.add(str(nombre))
            continue
        categoria = int(deteccion["category_id"])
        if categoria not in categorias:
            categorias_raras.add(categoria)
            continue
        copia = dict(deteccion)
        copia["image_id"] = por_nombre[nombre]
        traducidas.append(copia)

    if categorias_raras:
        raise CocoEvalError(
            f"Las detecciones traen categorías que no existen en la referencia: "
            f"{sorted(categorias_raras)}. Las válidas son {sorted(categorias)}.",
            "Revisa el mapeo de clases en configs/dataset/sh17_classes.json.",
        )
    if not traducidas:
        raise CocoEvalError(
            f"Ninguna de las {len(detections):,} detecciones se ha podido "
            f"emparejar con la referencia.",
            f"Nombres de fichero sin emparejar (primeros): {sorted(sin_emparejar)[:3]}. "
            "Comprueba que la partición y el COCO recortado coinciden.",
        )
    if sin_emparejar:
        logger.info(
            "Se descartan detecciones de %d imágenes ajenas a esta partición",
            len(sin_emparejar),
        )
    return traducidas
