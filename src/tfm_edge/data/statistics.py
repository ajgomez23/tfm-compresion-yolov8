"""Estadísticas descriptivas del dataset con dos criterios de tamaño de objeto.

Primario: el área escalada a la resolución de inferencia, que es lo que ve el detector. Secundario:
el área en la resolución original, el estándar de pycocotools, que en SH17 deja el 88 % de los
objetos como grandes. El escalado es letterbox, ``s = min(R/W, R/H)`` en ambas dimensiones, y cada
resultado guarda la resolución con la que se calculó. Incluye la tabla clase por tamaño y la
coocurrencia entre clases.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir

logger = get_logger(__name__)

COCO_SMALL_MAX_AREA = 32**2
COCO_MEDIUM_MAX_AREA = 96**2

SIZE_RANGES = ("small", "medium", "large")

#: Los dos criterios; el primero es el primario por defecto.
SizeCriterion = Literal["inference_res", "original_res"]
SIZE_CRITERIA: tuple[SizeCriterion, ...] = ("inference_res", "original_res")
DEFAULT_PRIMARY_CRITERION: SizeCriterion = "inference_res"

CRITERION_LABELS: dict[str, str] = {
    "inference_res": "resolución de inferencia (lo que ve el modelo)",
    "original_res": "resolución original (estándar pycocotools)",
}


def classify_size(area: float) -> str:
    """Clasifica un área en small, medium o large según los umbrales COCO."""
    if area < 0:
        raise ValueError(f"El área no puede ser negativa, y se recibió {area}.")
    if area < COCO_SMALL_MAX_AREA:
        return "small"
    if area <= COCO_MEDIUM_MAX_AREA:
        return "medium"
    return "large"


def letterbox_scale(width: int, height: int, inference_resolution: int) -> float:
    """Factor de escala lineal del letterbox de Ultralytics."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensiones no válidas: {width}x{height}.")
    if inference_resolution <= 0:
        raise ValueError(f"Resolución de inferencia no válida: {inference_resolution}.")
    return min(inference_resolution / width, inference_resolution / height)


def area_at_inference(
    area_original: float, width: int, height: int, inference_resolution: int
) -> float:
    """Área en la entrada del modelo a partir del área original."""
    escala = letterbox_scale(width, height, inference_resolution)
    return area_original * escala * escala


def classify_size_both(
    area_original: float, width: int, height: int, inference_resolution: int
) -> dict[str, str]:
    """Clasifica una caja con los dos criterios."""
    return {
        "inference_res": classify_size(
            area_at_inference(area_original, width, height, inference_resolution)
        ),
        "original_res": classify_size(area_original),
    }


@dataclass
class DatasetStatistics:
    """Estadísticas del dataset."""

    payload: dict[str, Any]

    @property
    def inference_resolution(self) -> int:
        """Resolución de inferencia con la que se calculó el criterio primario."""
        return int(self.payload["size_criteria"]["inference_resolution"])

    def to_json(self, path: Path | str) -> Path:
        """Vuelca las estadísticas a JSON."""
        destino = ensure_parent_dir(path)
        destino.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Estadísticas escritas en %s", destino)
        return destino

    def to_csv(self, output_dir: Path | str) -> dict[str, Path]:
        """Escribe las tablas principales en CSV, con una tabla clase por tamaño para cada
        criterio."""
        import pandas as pd

        directorio = ensure_dir(output_dir)
        escritos: dict[str, Path] = {}

        por_clase = pd.DataFrame(self.payload["per_class"]).T
        por_clase.index.name = "class_name"
        ruta = directorio / "stats_per_class.csv"
        por_clase.to_csv(ruta, na_rep="")
        escritos["per_class"] = ruta

        for criterio in SIZE_CRITERIA:
            cruzada = pd.DataFrame(self.payload["class_by_size"][criterio]).T
            cruzada.index.name = "class_name"
            ruta = directorio / f"stats_class_by_size_{criterio}.csv"
            cruzada.to_csv(ruta, na_rep="")
            escritos[f"class_by_size_{criterio}"] = ruta

        coocurrencia = pd.DataFrame(self.payload["cooccurrence"])
        ruta = directorio / "stats_cooccurrence.csv"
        coocurrencia.to_csv(ruta, na_rep="")
        escritos["cooccurrence"] = ruta

        resoluciones = pd.DataFrame(
            [{"resolution": k, "count": v} for k, v in self.payload["resolutions"].items()]
        )
        ruta = directorio / "stats_resolutions.csv"
        resoluciones.to_csv(ruta, index=False, na_rep="")
        escritos["resolutions"] = ruta

        for nombre, ruta in escritos.items():
            logger.info("Tabla '%s' escrita en %s", nombre, ruta)
        return escritos


def _percentiles(valores: Sequence[float]) -> dict[str, float]:
    """Mínimo, percentiles y máximo de una lista; vacío si no hay valores."""
    import numpy as np

    if not valores:
        return {}
    arreglo = np.asarray(valores, dtype=float)
    return {
        "min": float(arreglo.min()),
        "p25": float(np.percentile(arreglo, 25)),
        "median": float(np.median(arreglo)),
        "mean": float(arreglo.mean()),
        "p75": float(np.percentile(arreglo, 75)),
        "p95": float(np.percentile(arreglo, 95)),
        "max": float(arreglo.max()),
        "std": float(arreglo.std(ddof=1)) if arreglo.size > 1 else 0.0,
    }


def compute_conditional_cooccurrence(
    coco: Mapping[str, Any], class_names: Sequence[str]
) -> dict[str, Any]:
    """Coocurrencia condicional y dirigida entre clases.

    La celda (A, B) es la fracción de instancias de A que están en una imagen con al menos una
    instancia de B. No es simétrica: se lee de la fila a la columna. A diferencia de
    ``payload["cooccurrence"]``, que cuenta imágenes, aquí se cuentan instancias. La diagonal vale
    1.
    """
    imagenes = {img["id"]: img for img in coco.get("images", [])}
    if not imagenes:
        raise ValueError("El dataset COCO no contiene ninguna imagen.")

    numero_clases = len(class_names)
    anotaciones_por_imagen: dict[int, list[int]] = defaultdict(list)
    for anotacion in coco.get("annotations", []):
        clase = int(anotacion["category_id"])
        if 0 <= clase < numero_clases:
            anotaciones_por_imagen[anotacion["image_id"]].append(clase)

    numerador = [[0] * numero_clases for _ in range(numero_clases)]
    denominador = [0] * numero_clases
    for clases in anotaciones_por_imagen.values():
        presentes = set(clases)
        for fila, cuantas in Counter(clases).items():
            denominador[fila] += cuantas
            for columna in presentes:
                numerador[fila][columna] += cuantas

    matriz: dict[str, dict[str, float | None]] = {}
    for i, fila in enumerate(class_names):
        # Sin instancias no hay fracción: queda vacío.
        matriz[fila] = {
            columna: (numerador[i][j] / denominador[i] if denominador[i] else None)
            for j, columna in enumerate(class_names)
        }

    return {
        "definition": (
            "celda(fila=A, columna=B) = fraccion de las INSTANCIAS de A que estan "
            "en una imagen con al menos una instancia de B. Matriz DIRIGIDA y no "
            "simetrica. Calculada sobre el COCO completo."
        ),
        "class_names": list(class_names),
        "matrix": matriz,
        "numerator": {
            fila: {c: numerador[i][j] for j, c in enumerate(class_names)}
            for i, fila in enumerate(class_names)
        },
        "denominator": {fila: denominador[i] for i, fila in enumerate(class_names)},
    }


def compute_statistics(
    coco: Mapping[str, Any],
    class_names: Sequence[str],
    inference_resolution: int = 640,
    primary_criterion: SizeCriterion = DEFAULT_PRIMARY_CRITERION,
) -> DatasetStatistics:
    """Calcula todas las estadísticas descriptivas a partir del COCO preparado."""
    if primary_criterion not in SIZE_CRITERIA:
        raise ValueError(
            f"Criterio de tamaño desconocido: '{primary_criterion}'. "
            f"Válidos: {', '.join(SIZE_CRITERIA)}."
        )

    imagenes = {img["id"]: img for img in coco.get("images", [])}
    anotaciones = list(coco.get("annotations", []))
    if not imagenes:
        raise ValueError("El dataset COCO no contiene ninguna imagen.")

    areas_originales: list[float] = []
    areas_inferencia: list[float] = []
    areas_relativas: list[float] = []
    relaciones_aspecto: list[float] = []
    tamanos: dict[str, Counter[str]] = {c: Counter() for c in SIZE_CRITERIA}
    clase_por_tamano: dict[str, dict[int, Counter[str]]] = {
        c: defaultdict(Counter) for c in SIZE_CRITERIA
    }
    por_clase: Counter[int] = Counter()
    instancias_por_imagen: Counter[int] = Counter()
    clases_por_imagen: dict[int, set[int]] = defaultdict(set)
    areas_por_clase: dict[int, list[float]] = defaultdict(list)
    areas_rel_por_clase: dict[int, list[float]] = defaultdict(list)

    for anotacion in anotaciones:
        imagen = imagenes.get(anotacion["image_id"])
        if imagen is None:
            continue
        clase = int(anotacion["category_id"])
        _, _, ancho, alto = anotacion["bbox"]
        area = float(anotacion.get("area", ancho * alto))
        ancho_img, alto_img = int(imagen["width"]), int(imagen["height"])
        area_imagen = float(ancho_img) * float(alto_img)

        area_inf = area_at_inference(area, ancho_img, alto_img, inference_resolution)
        areas_originales.append(area)
        areas_inferencia.append(area_inf)
        areas_relativas.append(area / area_imagen if area_imagen > 0 else 0.0)
        if alto > 0:
            relaciones_aspecto.append(float(ancho) / float(alto))

        rangos = {"inference_res": classify_size(area_inf), "original_res": classify_size(area)}
        for criterio, rango in rangos.items():
            tamanos[criterio][rango] += 1
            clase_por_tamano[criterio][clase][rango] += 1

        por_clase[clase] += 1
        instancias_por_imagen[anotacion["image_id"]] += 1
        clases_por_imagen[anotacion["image_id"]].add(clase)
        areas_por_clase[clase].append(area)
        areas_rel_por_clase[clase].append(area / area_imagen if area_imagen > 0 else 0.0)

    # Coocurrencia entre clases
    coocurrencia: dict[str, dict[str, int]] = {n: {m: 0 for m in class_names} for n in class_names}
    for clases in clases_por_imagen.values():
        ordenadas = sorted(c for c in clases if c < len(class_names))
        for i, a in enumerate(ordenadas):
            for b in ordenadas[i:]:
                coocurrencia[class_names[a]][class_names[b]] += 1
                if a != b:
                    coocurrencia[class_names[b]][class_names[a]] += 1

    resoluciones = Counter(f"{img['width']}x{img['height']}" for img in imagenes.values())
    total = len(anotaciones)

    payload: dict[str, Any] = {
        "size_criteria": {
            "primary": primary_criterion,
            "inference_resolution": inference_resolution,
            "labels": CRITERION_LABELS,
            "resize_mode": "letterbox",
            "note": (
                "Se calculan SIEMPRE los dos criterios. El primario es el de "
                "resolucion de inferencia: el detector nunca ve la imagen "
                "original, ve un redimensionado letterbox a la resolucion de "
                "entrada. El de resolucion original se conserva como secundario "
                "por comparabilidad con pycocotools y con la literatura. El "
                "escalado es letterbox (preserva relacion de aspecto), asi que el "
                "area escala con min(R/W, R/H)^2, NO con (R/W)*(R/H)."
            ),
        },
        "summary": {
            "num_images": len(imagenes),
            "num_annotations": total,
            "num_classes": len(class_names),
            "images_without_annotations": len(imagenes) - len(instancias_por_imagen),
            "annotations_per_image_mean": round(total / len(imagenes), 4) if imagenes else 0.0,
            "num_distinct_resolutions": len(resoluciones),
        },
        "size_distribution": {
            criterio: {
                "counts": {r: tamanos[criterio].get(r, 0) for r in SIZE_RANGES},
                "fractions": {
                    r: round(tamanos[criterio].get(r, 0) / total, 4) if total else 0.0
                    for r in SIZE_RANGES
                },
                "label": CRITERION_LABELS[criterio],
            }
            for criterio in SIZE_CRITERIA
        },
        "box_area_px_original_res": _percentiles(areas_originales),
        "box_area_px_inference_res": _percentiles(areas_inferencia),
        "box_area_relative": _percentiles(areas_relativas),
        "box_aspect_ratio": _percentiles(relaciones_aspecto),
        "instances_per_image": _percentiles([float(v) for v in instancias_por_imagen.values()]),
        "resolutions": dict(resoluciones.most_common()),
        "per_class": {},
        "class_by_size": {c: {} for c in SIZE_CRITERIA},
        "cooccurrence": coocurrencia,
    }

    for indice, nombre in enumerate(class_names):
        instancias = por_clase[indice]
        imagenes_con_clase = sum(1 for c in clases_por_imagen.values() if indice in c)
        payload["per_class"][nombre] = {
            "class_index": indice,
            "instances": instancias,
            "fraction_of_instances": round(instancias / total, 6) if total else 0.0,
            "images_containing": imagenes_con_clase,
            "fraction_of_images": round(imagenes_con_clase / len(imagenes), 6),
            "area_px_original_median": (
                round(float(sorted(areas_por_clase[indice])[len(areas_por_clase[indice]) // 2]), 2)
                if areas_por_clase[indice]
                else ""
            ),
            "area_relative_median": (
                round(
                    float(
                        sorted(areas_rel_por_clase[indice])[len(areas_rel_por_clase[indice]) // 2]
                    ),
                    8,
                )
                if areas_rel_por_clase[indice]
                else ""
            ),
        }
        for criterio in SIZE_CRITERIA:
            conteo = clase_por_tamano[criterio][indice]
            payload["class_by_size"][criterio][nombre] = {
                "class_index": indice,
                **{r: conteo.get(r, 0) for r in SIZE_RANGES},
                "total": instancias,
                **{
                    f"{r}_fraction": (
                        round(conteo.get(r, 0) / instancias, 4) if instancias else 0.0
                    )
                    for r in SIZE_RANGES
                },
            }

    return DatasetStatistics(payload=payload)


def sizes_by_image(
    coco: Mapping[str, Any],
    inference_resolution: int = 640,
    criterion: SizeCriterion = DEFAULT_PRIMARY_CRITERION,
) -> dict[str, set[str]]:
    """Rangos de tamaño presentes en cada imagen. Lo usan el subconjunto de humo y la
    calibración."""
    imagenes = {img["id"]: img for img in coco.get("images", [])}
    resultado: dict[str, set[str]] = defaultdict(set)
    for anotacion in coco.get("annotations", []):
        imagen = imagenes.get(anotacion["image_id"])
        if imagen is None:
            continue
        area = float(anotacion.get("area", 0.0))
        if criterion == "inference_res":
            area = area_at_inference(
                area, int(imagen["width"]), int(imagen["height"]), inference_resolution
            )
        resultado[Path(imagen["file_name"]).stem].add(classify_size(area))
    return dict(resultado)


def labels_by_image(coco: Mapping[str, Any]) -> dict[str, set[int]]:
    """Clases de cada imagen, para la estratificación. Las imágenes sin anotaciones salen con
    conjunto vacío.
    """
    por_id = {img["id"]: Path(img["file_name"]).stem for img in coco.get("images", [])}
    resultado: dict[str, set[int]] = {nombre: set() for nombre in por_id.values()}
    for anotacion in coco.get("annotations", []):
        nombre = por_id.get(anotacion["image_id"])
        if nombre is not None:
            resultado[nombre].add(int(anotacion["category_id"]))
    return resultado
