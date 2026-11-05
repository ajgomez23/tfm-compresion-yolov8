"""Conjuntos de calibración para PTQ.

La calibración fija los rangos de activación, así que forma parte del experimento: la lista de
imágenes se guarda en disco, el muestreo está estratificado por clase y tamaño de objeto y se
generan varios tamaños (128, 256 y 512). Se guardan nombres de fichero, no copias. Solo se muestrea
de la partición de entrenamiento.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir

logger = get_logger(__name__)

DEFAULT_CALIBRATION_SIZES = (128, 256, 512)


class CalibrationSetError(TFMEdgeError):
    """Error al construir un conjunto de calibración."""


@dataclass
class CalibrationSet:
    """Conjunto de calibración."""

    size: int
    images: list[str]
    seed: int
    class_coverage: dict[str, int]
    size_coverage: dict[str, int]
    source_split: str

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el conjunto como diccionario serializable."""
        return {
            "requested_size": self.size,
            "actual_size": len(self.images),
            "seed": self.seed,
            "source_split": self.source_split,
            "class_coverage": self.class_coverage,
            "size_coverage": self.size_coverage,
            "images": self.images,
            "note": (
                "Lista de nombres base, no copias de imagenes. La calibracion se "
                "muestrea SOLO del conjunto de entrenamiento para no filtrar "
                "informacion de validacion ni de prueba."
            ),
        }


def build_calibration_set(
    candidates: Sequence[str],
    labels_by_image: Mapping[str, set[int]],
    sizes_by_image: Mapping[str, set[str]],
    class_names: Sequence[str],
    size: int,
    seed: int = 42,
    source_split: str = "train",
) -> CalibrationSet:
    """Construye un conjunto de calibración estratificado.

    Primero cubre las 17 clases y los tres rangos de tamaño eligiendo la imagen que más aporta;
    después rellena con las clases menos representadas.
    """
    if not candidates:
        raise CalibrationSetError(
            "No hay imágenes candidatas para la calibración.",
            "Genera antes las particiones con scripts/02_prepare_dataset.py.",
        )
    if size <= 0:
        raise CalibrationSetError(
            f"El tamaño de calibración debe ser positivo y se pidió {size}.",
            "Revisa quantization.calibration_size en la configuración.",
        )
    if size > len(candidates):
        raise CalibrationSetError(
            f"Se piden {size} imágenes de calibración pero solo hay "
            f"{len(candidates)} en la partición '{source_split}'.",
            "Reduce el tamaño de calibración o revisa el particionado. NO se "
            "muestrea de validación ni de prueba: sería una fuga de información.",
        )

    generador = random.Random(seed)
    disponibles = sorted(candidates)
    generador.shuffle(disponibles)

    def elementos(nombre: str) -> set[str]:
        return {f"c{c}" for c in labels_by_image.get(nombre, set())} | {
            f"s{s}" for s in sizes_by_image.get(nombre, set())
        }

    universo: set[str] = set()
    for nombre in disponibles:
        universo |= elementos(nombre)

    seleccion: list[str] = []
    pendientes = set(universo)
    restantes = list(disponibles)

    # Fase 1: cobertura
    while pendientes and restantes and len(seleccion) < size:
        mejor = max(restantes, key=lambda n: (len(elementos(n) & pendientes), n))
        if not (elementos(mejor) & pendientes):
            break
        seleccion.append(mejor)
        pendientes -= elementos(mejor)
        restantes.remove(mejor)

    if pendientes:
        logger.warning(
            "El conjunto de calibración de %d imágenes no cubre %d elementos "
            "(clases o rangos de tamaño): %s",
            size,
            len(pendientes),
            sorted(pendientes)[:10],
        )

    # Fase 2: relleno con las clases menos representadas
    conteo: Counter[int] = Counter()
    for nombre in seleccion:
        conteo.update(labels_by_image.get(nombre, set()))

    while len(seleccion) < size and restantes:

        def prioridad(nombre: str) -> tuple[int, str]:
            clases = labels_by_image.get(nombre, set())
            if not clases:
                return (10**9, nombre)
            return (min(conteo[c] for c in clases), nombre)

        mejor = min(restantes, key=prioridad)
        seleccion.append(mejor)
        conteo.update(labels_by_image.get(mejor, set()))
        restantes.remove(mejor)

    cobertura_clase: Counter[int] = Counter()
    cobertura_tamano: Counter[str] = Counter()
    for nombre in seleccion:
        cobertura_clase.update(labels_by_image.get(nombre, set()))
        cobertura_tamano.update(sizes_by_image.get(nombre, set()))

    return CalibrationSet(
        size=size,
        images=sorted(seleccion),
        seed=seed,
        class_coverage={class_names[i]: cobertura_clase.get(i, 0) for i in range(len(class_names))},
        size_coverage=dict(cobertura_tamano),
        source_split=source_split,
    )


def build_calibration_sets(
    candidates: Sequence[str],
    labels_by_image: Mapping[str, set[int]],
    sizes_by_image: Mapping[str, set[str]],
    class_names: Sequence[str],
    sizes: Sequence[int] = DEFAULT_CALIBRATION_SIZES,
    seed: int = 42,
    source_split: str = "train",
) -> dict[int, CalibrationSet]:
    """Construye varios conjuntos de calibración anidados (el de 256 contiene al de 128, y así).

    Al estar anidados, la única diferencia entre ellos es el tamaño.
    """
    if not sizes:
        raise CalibrationSetError(
            "No se ha pedido ningún tamaño de conjunto de calibración.",
            "Pasa al menos un tamaño que quepa en la partición de entrenamiento. Con "
            "--smoke-test usa --calibration-sizes con valores menores.",
        )

    conjuntos: dict[int, CalibrationSet] = {}
    mayor = build_calibration_set(
        candidates, labels_by_image, sizes_by_image, class_names, max(sizes), seed, source_split
    )
    for tamano in sorted(sizes):
        if tamano == mayor.size:
            conjuntos[tamano] = mayor
            continue
        # Se toman los primeros del mayor, que conservan el orden de cobertura y la estratificación.
        conjuntos[tamano] = build_calibration_set(
            candidates, labels_by_image, sizes_by_image, class_names, tamano, seed, source_split
        )
    return conjuntos


def write_calibration_sets(
    sets: Mapping[int, CalibrationSet], output_dir: Path | str
) -> dict[int, Path]:
    """Escribe cada conjunto como lista de texto y como JSON con su cobertura."""
    directorio = ensure_dir(output_dir)
    rutas: dict[int, Path] = {}
    for tamano, conjunto in sorted(sets.items()):
        lista = directorio / f"calibration_{tamano}.txt"
        ensure_parent_dir(lista).write_text("\n".join(conjunto.images) + "\n", encoding="utf-8")
        detalle = directorio / f"calibration_{tamano}.json"
        detalle.write_text(
            json.dumps(conjunto.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        rutas[tamano] = lista
        sin_cubrir = [c for c, n in conjunto.class_coverage.items() if n == 0]
        logger.info(
            "Calibración de %d imágenes en %s (%d clases cubiertas de %d%s)",
            tamano,
            lista,
            sum(1 for n in conjunto.class_coverage.values() if n > 0),
            len(conjunto.class_coverage),
            f"; SIN cubrir: {sin_cubrir}" if sin_cubrir else "",
        )
    return rutas


def load_calibration_set(path: Path | str) -> list[str]:
    """Carga la lista de imágenes de un conjunto de calibración."""
    fichero = Path(path)
    if not fichero.is_file():
        raise CalibrationSetError(
            f"No existe el conjunto de calibración '{fichero}'.",
            "Genéralo con scripts/02_prepare_dataset.py.",
        )
    return [l.strip() for l in fichero.read_text(encoding="utf-8").splitlines() if l.strip()]


IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def resolve_calibration_images(
    stems: Sequence[str], images_dir: Path | str, limit: int | None = None
) -> list[Path]:
    """Convierte los nombres base del conjunto en rutas de imagen.

    La usan la sensibilidad y la PTQ para calibrar con las mismas imágenes. Un nombre que no aparece
    es un error; se comprueba la lista entera antes de recortarla a ``limit``.
    """
    raiz = Path(images_dir)
    if not raiz.is_dir():
        raise CalibrationSetError(
            f"No existe el directorio de imágenes {raiz}.",
            "Prepara el dataset con scripts/02_prepare_dataset.py y la copia de "
            "640 px con scripts/02b_resize_dataset.py.",
        )

    por_stem: dict[str, Path] = {}
    for sufijo in IMAGE_SUFFIXES:
        for ruta in raiz.glob(f"*{sufijo}"):
            por_stem.setdefault(ruta.stem, ruta)

    resueltas: list[Path] = []
    faltan: list[str] = []
    for stem in stems:
        ruta = por_stem.get(stem)
        if ruta is None:
            faltan.append(stem)
        else:
            resueltas.append(ruta)

    if faltan:
        raise CalibrationSetError(
            f"{len(faltan)} imágenes del conjunto de calibración no están en "
            f"{raiz}. Primeras: {faltan[:3]}",
            "La lista de calibración y las imágenes no son de la misma preparación del dataset. "
            "Regenéralas con scripts/02_prepare_dataset.py.",
        )

    recortadas = resueltas[:limit] if limit is not None else resueltas
    if len(recortadas) < 2:
        raise CalibrationSetError(
            f"Solo se han resuelto {len(recortadas)} imágenes y hacen falta al menos 2.",
            "Con una sola, tanto los rangos de cuantización como el orden de "
            "sensibilidad dependerían de esa imagen concreta.",
        )
    return recortadas
