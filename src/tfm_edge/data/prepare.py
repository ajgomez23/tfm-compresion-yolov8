"""Validación de procedencia y preparación del dataset SH17.

El paquete oficial trae images/, labels/ (YOLO), voc_labels/ (Pascal VOC), meta-data/ (JSON de
Pexels) y train_files.txt y val_files.txt. validate_provenance aplica ocho criterios que detectan
reexportaciones de terceros (imágenes redimensionadas, duplicadas o con otro orden de clases), y
prepare_dataset la ejecuta siempre. processed no copia las imágenes: guarda las anotaciones
convertidas y un manifiesto que apunta a raw, así que quien lo use debe llamar antes a
verify_raw_available.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.common.config import load_class_names
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir, get_project_paths
from tfm_edge.common.registry import utc_timestamp

logger = get_logger(__name__)

# Cifras oficiales para la validación de procedencia (Ahmad y Rahimi, 2024,
# doi:10.1016/j.jnlssr.2024.09.002, y el propio paquete). No son resultados del proyecto.
SH17_EXPECTED_IMAGES = 8099
SH17_EXPECTED_ANNOTATIONS = 75994
SH17_EXPECTED_CLASSES = 17

#: Tolerancia relativa en el recuento de anotaciones.
ANNOTATION_COUNT_TOLERANCE = 0.05

#: Resolución mínima esperada; una reexportación a 640x640 queda por debajo.
MIN_EXPECTED_MAX_WIDTH = 1920

EXPECTED_SUBDIRS = ("images", "labels")

OPTIONAL_SUBDIRS = ("voc_labels", "meta-data")

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

#: Manifiesto que enlaza processed con el raw del que salió.
MANIFEST_FILENAME = "dataset_manifest.json"

#: Fichero de atribuciones que exige la cláusula BY de la licencia.
ATTRIBUTIONS_FILENAME = "SH17_ATTRIBUTIONS.md"

ANOMALIES_FILENAME = "annotation_anomalies.json"


class ProvenanceError(TFMEdgeError):
    """El dataset de entrada no supera la validación de procedencia."""

    def __init__(self, fallos: Sequence[str]) -> None:
        detalle = "\n".join(f"       - {f}" for f in fallos)
        super().__init__(
            f"El dataset de entrada NO es SH17 original. Criterios incumplidos:\n{detalle}",
            (
                "Casi con seguridad se trata de una reexportación de terceros (por "
                "ejemplo de Roboflow) con preprocesado destructivo: redimensionado, "
                "duplicación de imágenes o reordenación de clases.\n"
                "     Eso NO rompe el entrenamiento: produce métricas plausibles y "
                "silenciosamente inválidas, y arruinaría el análisis por tamaño de "
                "objeto y por clase.\n"
                "     Descarga el paquete OFICIAL desde Kaggle "
                "(mugheesahmad/sh17-dataset-for-ppe-detection) y compara su SHA-256 "
                "con el registrado en el manifiesto de integridad."
            ),
        )


class RawDatasetMissingError(TFMEdgeError):
    """Falta data/raw o no es el que generó processed."""

    def __init__(self, motivo: str, raw_dir: Path) -> None:
        super().__init__(
            f"No se puede usar el dataset preparado: {motivo}",
            (
                f"'{raw_dir}' debe contener el dataset SH17 extraído.\n"
                "     'processed' NO duplica las imágenes por presupuesto de disco "
                "y solo guarda anotaciones y un "
                "manifiesto que apunta a 'raw'.\n"
                "     Vuelve a extraer el dataset con scripts/01_download_dataset.py "
                "y a prepararlo con scripts/02_prepare_dataset.py."
            ),
        )


# Validación de procedencia


@dataclass
class ProvenanceCheck:
    """Resultado de un criterio de procedencia."""

    number: int
    criterion: str
    passed: bool
    evidence: str


@dataclass
class ProvenanceReport:
    """Informe de la validación de procedencia."""

    checks: list[ProvenanceCheck] = field(default_factory=list)
    raw_dir: str = ""
    timestamp: str = ""

    @property
    def passed(self) -> bool:
        """True si se cumplen todos los criterios."""
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[str]:
        """Descripción de los criterios incumplidos."""
        return [f"[{c.number}] {c.criterion}: {c.evidence}" for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return {
            "raw_dir": self.raw_dir,
            "timestamp": self.timestamp,
            "passed": self.passed,
            "checks": [
                {
                    "number": c.number,
                    "criterion": c.criterion,
                    "passed": c.passed,
                    "evidence": c.evidence,
                }
                for c in self.checks
            ],
        }

    def summary(self) -> str:
        """Resumen con una línea por criterio."""
        lineas = ["Validación de procedencia del dataset:"]
        for c in sorted(self.checks, key=lambda c: c.number):
            lineas.append(f"  {c.number}. [{'PASA ' if c.passed else 'FALLA'}] {c.criterion}")
            lineas.append(f"        {c.evidence}")
        lineas.append(
            f"  RESULTADO: {sum(c.passed for c in self.checks)}/{len(self.checks)} criterios"
        )
        return "\n".join(lineas)


def _listar_imagenes(raw_dir: Path) -> list[Path]:
    """Rutas de todas las imágenes del dataset, ordenadas."""
    directorio = raw_dir / "images"
    if not directorio.is_dir():
        return []
    return sorted(p for p in directorio.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def _leer_etiqueta_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Lee un fichero de etiquetas YOLO como tuplas ``(clase, cx, cy, w, h)`` normalizadas."""
    anotaciones: list[tuple[int, float, float, float, float]] = []
    for numero, linea in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        partes = linea.split()
        if not partes:
            continue
        if len(partes) < 5:
            raise ValueError(f"{path.name}:{numero}: se esperaban 5 campos y hay {len(partes)}")
        try:
            anotaciones.append(
                (int(partes[0]), *(float(v) for v in partes[1:5]))  # type: ignore[misc]
            )
        except ValueError as error:
            raise ValueError(f"{path.name}:{numero}: campo no numérico ({error})") from error
    return anotaciones


def _dimensiones_desde_metadatos(raw_dir: Path) -> dict[str, tuple[int, int]]:
    """Dimensiones originales de cada imagen según meta-data/, para no abrir las 8.099 imágenes.

    verify_metadata_dimensions comprueba que coinciden con las reales.
    """
    directorio = raw_dir / "meta-data"
    if not directorio.is_dir():
        return {}
    dimensiones: dict[str, tuple[int, int]] = {}
    for fichero in directorio.glob("*.json"):
        try:
            datos = json.loads(fichero.read_text(encoding="utf-8"))
            ancho, alto = int(datos["width"]), int(datos["height"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            logger.debug("Metadato ilegible en %s: %s", fichero.name, error)
            continue
        if ancho > 0 and alto > 0:
            dimensiones[fichero.stem] = (ancho, alto)
    return dimensiones


def verify_metadata_dimensions(
    raw_dir: Path | str, sample_size: int = 200, seed: int = 42
) -> dict[str, Any]:
    """Abre una muestra de imágenes y comprueba que sus dimensiones coinciden con las de
    meta-data/."""
    from PIL import Image

    raiz = Path(raw_dir)
    dimensiones = _dimensiones_desde_metadatos(raiz)
    imagenes = [p for p in _listar_imagenes(raiz) if p.stem in dimensiones]
    if not imagenes:
        return {"checked": 0, "matches": 0, "mismatches": 0, "details": [], "usable": False}

    generador = random.Random(seed)
    muestra = generador.sample(imagenes, min(sample_size, len(imagenes)))

    coincidencias = 0
    discrepancias: list[dict[str, Any]] = []
    for ruta in muestra:
        try:
            with Image.open(ruta) as imagen:
                real = imagen.size
        except Exception as error:  # pragma: no cover - imagen corrupta
            discrepancias.append({"image": ruta.name, "error": str(error)})
            continue
        declarada = dimensiones[ruta.stem]
        if real == declarada:
            coincidencias += 1
        else:
            discrepancias.append(
                {"image": ruta.name, "metadata": list(declarada), "actual": list(real)}
            )

    resultado = {
        "checked": len(muestra),
        "matches": coincidencias,
        "mismatches": len(discrepancias),
        "details": discrepancias[:20],
        "usable": not discrepancias,
    }
    if discrepancias:
        logger.warning(
            "Las dimensiones de meta-data/ NO son fiables: %d discrepancias de %d "
            "comprobadas. Se leerán las dimensiones de las imágenes.",
            len(discrepancias),
            len(muestra),
        )
    else:
        logger.info(
            "Dimensiones de meta-data/ verificadas: %d de %d coinciden exactamente. "
            "Se pueden usar sin abrir las imágenes.",
            coincidencias,
            len(muestra),
        )
    return resultado


def validate_provenance(
    raw_dir: Path | str, sample_size: int = 250, seed: int = 42
) -> ProvenanceReport:
    """Aplica los ocho criterios de procedencia al dataset.

    No lanza excepción; prepare_dataset aborta si el informe falla.
    """
    from PIL import Image

    raiz = Path(raw_dir)
    informe = ProvenanceReport(raw_dir=str(raiz), timestamp=utc_timestamp())

    def anotar(n: int, criterio: str, ok: bool, evidencia: str) -> None:
        informe.checks.append(ProvenanceCheck(n, criterio, ok, evidencia))

    # 0. Estructura
    faltantes = [d for d in EXPECTED_SUBDIRS if not (raiz / d).is_dir()]
    if faltantes:
        anotar(
            0,
            "Estructura de directorios esperada",
            False,
            f"faltan los subdirectorios {faltantes} en '{raiz}'",
        )
        return informe
    anotar(
        0,
        "Estructura de directorios esperada",
        True,
        f"presentes {list(EXPECTED_SUBDIRS)}; opcionales: "
        f"{[d for d in OPTIONAL_SUBDIRS if (raiz / d).is_dir()]}",
    )

    imagenes = _listar_imagenes(raiz)
    etiquetas = sorted((raiz / "labels").glob("*.txt"))

    # 1 y 2. Resoluciones
    generador = random.Random(seed)
    muestra = generador.sample(imagenes, min(sample_size, len(imagenes))) if imagenes else []
    dimensiones: list[tuple[int, int]] = []
    for ruta in muestra:
        try:
            with Image.open(ruta) as imagen:
                dimensiones.append(imagen.size)
        except Exception as error:  # pragma: no cover - imagen corrupta
            logger.debug("No se pudo leer %s: %s", ruta.name, error)

    if dimensiones:
        distintas = len(set(dimensiones))
        uniforme = distintas == 1
        anotar(
            1,
            "Imágenes NO redimensionadas a una resolución uniforme",
            not uniforme,
            f"{distintas} resoluciones distintas en {len(dimensiones)} muestreadas"
            + (f" (todas {dimensiones[0]})" if uniforme else ""),
        )
        ancho_max = max(w for w, _ in dimensiones)
        anotar(
            2,
            "Resoluciones originales conservadas",
            ancho_max >= MIN_EXPECTED_MAX_WIDTH,
            f"ancho máximo observado {ancho_max} px " f"(se esperan ≥ {MIN_EXPECTED_MAX_WIDTH})",
        )
    else:
        anotar(
            1,
            "Imágenes NO redimensionadas a una resolución uniforme",
            False,
            "sin imágenes legibles",
        )
        anotar(2, "Resoluciones originales conservadas", False, "sin imágenes legibles")

    # 3. Duplicados por nombre base
    bases = Counter(p.stem for p in imagenes)
    repetidos = [k for k, v in bases.items() if v > 1]
    anotar(
        3,
        "Sin imágenes duplicadas por nombre base",
        not repetidos,
        f"{len(repetidos)} nombres base repetidos de {len(bases):,} únicos",
    )

    # 4. Recuento de imágenes y anotaciones
    total_anotaciones = 0
    indices: Counter[int] = Counter()
    ilegibles = 0
    for ruta in etiquetas:
        try:
            filas = _leer_etiqueta_yolo(ruta)
        except ValueError:
            ilegibles += 1
            continue
        total_anotaciones += len(filas)
        indices.update(f[0] for f in filas)

    margen = SH17_EXPECTED_ANNOTATIONS * ANNOTATION_COUNT_TOLERANCE
    anotar(
        4,
        f"Recuento coherente con el original ({SH17_EXPECTED_IMAGES:,} imágenes, "
        f"{SH17_EXPECTED_ANNOTATIONS:,} anotaciones)",
        len(imagenes) == SH17_EXPECTED_IMAGES
        and abs(total_anotaciones - SH17_EXPECTED_ANNOTATIONS) <= margen,
        f"{len(imagenes):,} imágenes y {total_anotaciones:,} anotaciones "
        f"({ilegibles} ficheros de etiqueta ilegibles)",
    )

    # 5. Número de clases
    fuera_rango = [i for i in indices if not 0 <= i < SH17_EXPECTED_CLASSES]
    anotar(
        5,
        f"Exactamente {SH17_EXPECTED_CLASSES} clases, sin categoría espuria",
        len(indices) == SH17_EXPECTED_CLASSES and not fuera_rango,
        f"{len(indices)} índices distintos"
        + (f", rango {min(indices)}–{max(indices)}" if indices else "")
        + (f", FUERA DE RANGO: {sorted(fuera_rango)}" if fuera_rango else ""),
    )

    # 6. Correspondencia entre imagen y etiqueta
    sin_etiqueta = {p.stem for p in imagenes} - {p.stem for p in etiquetas}
    sin_imagen = {p.stem for p in etiquetas} - {p.stem for p in imagenes}
    anotar(
        6,
        "Cada imagen tiene su etiqueta y viceversa",
        not sin_etiqueta and not sin_imagen,
        f"{len(sin_etiqueta)} imágenes sin etiqueta, {len(sin_imagen)} etiquetas sin imagen",
    )

    # 7. Sin partición prefabricada impuesta
    particiones = [d for d in ("train", "valid", "val", "test") if (raiz / d).is_dir()]
    anotar(
        7,
        "Sin partición train/valid/test impuesta como estructura",
        not particiones,
        f"directorios de partición encontrados: {particiones or 'ninguno'}",
    )

    # 8. Sin marcadores de reexportación
    marcadores = [p.name for p in raiz.rglob("*") if p.is_file() and "roboflow" in p.name.lower()]
    anotar(
        8,
        "Sin marcadores de reexportación de terceros",
        not marcadores,
        f"{len(marcadores)} ficheros con 'roboflow' en el nombre",
    )

    return informe


def verify_raw_available(
    processed_dir: Path | str | None = None, raw_dir: Path | str | None = None
) -> Path:
    """Comprueba que el raw que generó processed sigue presente y tiene las mismas imágenes.

    Sin ``raw_dir`` se toma del manifiesto y, si no, data/raw.
    """
    rutas = get_project_paths()
    procesado = Path(processed_dir) if processed_dir is not None else rutas.data_processed
    manifiesto_path = procesado / MANIFEST_FILENAME

    if not manifiesto_path.is_file():
        raise RawDatasetMissingError(
            f"no existe el manifiesto '{manifiesto_path}'",
            Path(raw_dir) if raw_dir is not None else rutas.data_raw,
        )
    manifiesto = json.loads(manifiesto_path.read_text(encoding="utf-8"))

    crudo = (
        Path(raw_dir) if raw_dir is not None else Path(manifiesto.get("raw_dir", rutas.data_raw))
    )
    if not (crudo / "images").is_dir():
        raise RawDatasetMissingError(f"no existe '{crudo / 'images'}'", crudo)

    esperadas = int(manifiesto.get("num_images", 0))
    presentes = len(_listar_imagenes(crudo))
    if esperadas and presentes != esperadas:
        raise RawDatasetMissingError(
            f"'{crudo}' tiene {presentes:,} imágenes y el manifiesto registró "
            f"{esperadas:,}. El dataset crudo no es el que generó 'processed'",
            crudo,
        )
    logger.debug("Dataset crudo verificado en %s (%d imágenes)", crudo, presentes)
    return crudo


# Atribución (cláusula BY de la licencia)


def extract_attributions(raw_dir: Path | str, output_path: Path | str) -> Path:
    """Escribe la atribución de cada imagen (fotógrafo y URL de Pexels) a partir de meta-data/.

    El fichero va dentro del repositorio porque data/ no se versiona.
    """
    raiz = Path(raw_dir)
    directorio = raiz / "meta-data"
    if not directorio.is_dir():
        raise RawDatasetMissingError(f"no existe '{directorio}'", raiz)

    registros: list[dict[str, str]] = []
    for fichero in sorted(directorio.glob("*.json")):
        try:
            datos = json.loads(fichero.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        registros.append(
            {
                "image": fichero.stem,
                "photographer": str(datos.get("photographer", "")).strip(),
                "photographer_url": str(datos.get("photographer_url", "")).strip(),
                "source_url": str(datos.get("url", "")).strip(),
            }
        )

    autores = Counter(r["photographer"] for r in registros if r["photographer"])
    destino = ensure_parent_dir(output_path)

    lineas = [
        "# Atribución de las imágenes de SH17",
        "",
        "Fichero generado automáticamente por `tfm_edge.data.prepare.extract_attributions`.",
        "No editar a mano.",
        "",
        "## Por qué existe este fichero",
        "",
        "El dataset SH17 se distribuye bajo CC BY-NC-SA 4.0. La cláusula BY",
        "obliga a conservar la atribución de autoría. Las imágenes proceden de",
        "[Pexels](https://www.pexels.com/license/), y sus autores constan en los",
        "metadatos que acompañan al dataset.",
        "",
        "Este fichero vive en el repositorio, y no en `data/`, porque `data/` no se",
        "versiona: la obligación de atribución no desaparece por no versionar los datos.",
        "",
        "## Cita del dataset",
        "",
        "```bibtex",
        "@article{AHMAD2024,",
        "    title = {SH17: A dataset for human safety and personal protective "
        "equipment detection in manufacturing industry},",
        "    journal = {Journal of Safety Science and Resilience},",
        "    year = {2024},",
        "    issn = {2666-4496},",
        "    doi = {https://doi.org/10.1016/j.jnlssr.2024.09.002},",
        "    author = {Hafiz Mughees Ahmad and Afshin Rahimi},",
        "}",
        "```",
        "",
        "## Resumen",
        "",
        f"- Imágenes con metadatos de atribución: {len(registros):,}",
        f"- Fotógrafos distintos: {len(autores):,}",
        "",
        "## Fotógrafos, por número de imágenes aportadas",
        "",
        "| Fotógrafo | Imágenes | Perfil |",
        "|---|---:|---|",
    ]
    perfiles = {r["photographer"]: r["photographer_url"] for r in registros if r["photographer"]}
    for autor, cuantas in autores.most_common():
        lineas.append(f"| {autor} | {cuantas} | {perfiles.get(autor, '')} |")

    lineas += [
        "",
        "## Detalle por imagen",
        "",
        "| Imagen | Fotógrafo | Origen |",
        "|---|---|---|",
    ]
    for r in registros:
        lineas.append(f"| `{r['image']}` | {r['photographer']} | {r['source_url']} |")
    lineas.append("")

    destino.write_text("\n".join(lineas), encoding="utf-8")
    logger.info(
        "Atribuciones escritas en %s (%d imágenes, %d fotógrafos)",
        destino,
        len(registros),
        len(autores),
    )
    return destino


# Validación de anotaciones


@dataclass
class AnnotationAnomalies:
    """Anomalías encontradas al validar las anotaciones."""

    out_of_bounds: list[dict[str, Any]] = field(default_factory=list)
    non_positive_area: list[dict[str, Any]] = field(default_factory=list)
    images_without_annotations: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    invalid_class: list[dict[str, Any]] = field(default_factory=list)
    unreadable_labels: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Número total de anomalías detectadas."""
        return (
            len(self.out_of_bounds)
            + len(self.non_positive_area)
            + len(self.images_without_annotations)
            + len(self.missing_images)
            + len(self.invalid_class)
            + len(self.unreadable_labels)
        )

    def to_dict(self) -> dict[str, Any]:
        """Diccionario serializable con las listas recortadas a 100 elementos; los totales se
        conservan.
        """
        return {
            "counts": {
                "out_of_bounds": len(self.out_of_bounds),
                "non_positive_area": len(self.non_positive_area),
                "images_without_annotations": len(self.images_without_annotations),
                "missing_images": len(self.missing_images),
                "invalid_class": len(self.invalid_class),
                "unreadable_labels": len(self.unreadable_labels),
                "total": self.total,
            },
            "samples": {
                "out_of_bounds": self.out_of_bounds[:100],
                "non_positive_area": self.non_positive_area[:100],
                "images_without_annotations": self.images_without_annotations[:100],
                "missing_images": self.missing_images[:100],
                "invalid_class": self.invalid_class[:100],
                "unreadable_labels": self.unreadable_labels[:100],
            },
            "policy": (
                "Las cajas que se salen de los límites se RECORTAN al borde de la "
                "imagen. Las de área no positiva tras el recorte se DESCARTAN. Las "
                "anotaciones con clase fuera de rango se DESCARTAN. Nada se corrige "
                "en silencio: todo queda contado aquí."
            ),
        }


def yolo_to_absolute(
    cx: float, cy: float, w: float, h: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Caja YOLO normalizada a ``(x, y, ancho, alto)`` en píxeles con origen arriba a la
    izquierda."""
    ancho_px = w * width
    alto_px = h * height
    return (cx * width - ancho_px / 2, cy * height - alto_px / 2, ancho_px, alto_px)


def _recortar_a_imagen(
    x: float, y: float, ancho: float, alto: float, width: int, height: int
) -> tuple[float, float, float, float]:
    """Recorta una caja a los límites de la imagen."""
    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + ancho, float(width)))
    y2 = max(0.0, min(y + alto, float(height)))
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


# Conversión a COCO


def build_coco_dataset(
    raw_dir: Path | str,
    class_names: Sequence[str],
    image_stems: Iterable[str] | None = None,
    dimensions: dict[str, tuple[int, int]] | None = None,
) -> tuple[dict[str, Any], AnnotationAnomalies]:
    """Convierte las anotaciones YOLO a COCO y devuelve el COCO y las anomalías.

    Las áreas se calculan sobre la resolución original, como hace pycocotools; las detecciones se
    reescalan a coordenadas originales antes de evaluar. ``image_stems`` permite generar un COCO por
    partición.
    """
    from PIL import Image

    raiz = Path(raw_dir)
    if not (raiz / "images").is_dir() or not (raiz / "labels").is_dir():
        raise RawDatasetMissingError("faltan 'images/' o 'labels/'", raiz)

    dims = dict(dimensions or {})
    anomalias = AnnotationAnomalies()
    seleccion = set(image_stems) if image_stems is not None else None

    imagenes_coco: list[dict[str, Any]] = []
    anotaciones_coco: list[dict[str, Any]] = []
    siguiente_anotacion = 1

    for indice_imagen, ruta_imagen in enumerate(_listar_imagenes(raiz), start=1):
        if seleccion is not None and ruta_imagen.stem not in seleccion:
            continue

        if ruta_imagen.stem in dims:
            ancho, alto = dims[ruta_imagen.stem]
        else:
            try:
                with Image.open(ruta_imagen) as imagen:
                    ancho, alto = imagen.size
            except Exception as error:  # pragma: no cover - imagen corrupta
                anomalias.missing_images.append(f"{ruta_imagen.name}: ilegible ({error})")
                continue

        imagenes_coco.append(
            {
                "id": indice_imagen,
                "file_name": ruta_imagen.name,
                "width": int(ancho),
                "height": int(alto),
            }
        )

        ruta_etiqueta = raiz / "labels" / f"{ruta_imagen.stem}.txt"
        if not ruta_etiqueta.is_file():
            anomalias.images_without_annotations.append(ruta_imagen.name)
            continue

        try:
            filas = _leer_etiqueta_yolo(ruta_etiqueta)
        except ValueError as error:
            anomalias.unreadable_labels.append({"file": ruta_etiqueta.name, "error": str(error)})
            continue

        if not filas:
            anomalias.images_without_annotations.append(ruta_imagen.name)
            continue

        for clase, cx, cy, w, h in filas:
            if not 0 <= clase < len(class_names):
                anomalias.invalid_class.append({"file": ruta_etiqueta.name, "class_index": clase})
                continue

            x, y, ancho_caja, alto_caja = yolo_to_absolute(cx, cy, w, h, ancho, alto)
            se_sale = (
                x < -0.5 or y < -0.5 or x + ancho_caja > ancho + 0.5 or y + alto_caja > alto + 0.5
            )
            if se_sale:
                anomalias.out_of_bounds.append(
                    {
                        "file": ruta_etiqueta.name,
                        "bbox": [round(v, 2) for v in (x, y, ancho_caja, alto_caja)],
                        "image_size": [ancho, alto],
                    }
                )
                x, y, ancho_caja, alto_caja = _recortar_a_imagen(
                    x, y, ancho_caja, alto_caja, ancho, alto
                )

            if ancho_caja <= 0 or alto_caja <= 0:
                anomalias.non_positive_area.append(
                    {"file": ruta_etiqueta.name, "bbox": [x, y, ancho_caja, alto_caja]}
                )
                continue

            anotaciones_coco.append(
                {
                    "id": siguiente_anotacion,
                    "image_id": indice_imagen,
                    "category_id": clase,
                    "bbox": [round(v, 2) for v in (x, y, ancho_caja, alto_caja)],
                    # Área sobre la resolución original.
                    "area": round(ancho_caja * alto_caja, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            siguiente_anotacion += 1

    coco = {
        "info": {
            "description": "SH17 convertido a COCO por tfm-edge-object-detection",
            "license": "CC BY-NC-SA 4.0",
            "date_created": utc_timestamp(),
            "area_reference": (
                "Las areas estan calculadas sobre la RESOLUCION ORIGINAL de cada "
                "imagen, no sobre la resolucion de inferencia. Ver D-008."
            ),
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY-NC-SA 4.0",
                "url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            }
        ],
        # Los ids de categoría son los índices de YOLO desde 0, para no añadir una capa de
        # traducción.
        "categories": [
            {"id": i, "name": nombre, "supercategory": "sh17"}
            for i, nombre in enumerate(class_names)
        ],
        "images": imagenes_coco,
        "annotations": anotaciones_coco,
    }
    return coco, anomalias


def write_manifest(
    processed_dir: Path | str,
    raw_dir: Path | str,
    num_images: int,
    num_annotations: int,
    provenance: ProvenanceReport,
    metadata_check: dict[str, Any],
    dataset_sha256: str = "",
) -> Path:
    """Escribe el manifiesto que enlaza processed con el raw del que salió."""
    destino = ensure_parent_dir(Path(processed_dir) / MANIFEST_FILENAME)
    contenido = {
        "dataset": "SH17",
        "created_at": utc_timestamp(),
        "raw_dir": str(Path(raw_dir).resolve()),
        "dataset_sha256": dataset_sha256,
        "license": "CC BY-NC-SA 4.0",
        "num_images": num_images,
        "num_annotations": num_annotations,
        "provenance": provenance.to_dict(),
        "metadata_dimensions_check": metadata_check,
        "warning": (
            "'processed' NO contiene copias de las imagenes: las referencia desde "
            "'raw_dir'. Toda funcion que consuma 'processed' debe llamar antes a "
            "prepare.verify_raw_available()."
        ),
    }
    destino.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifiesto escrito en %s", destino)
    return destino


def prepare_dataset(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    class_names_file: Path | str | None = None,
    dataset_sha256: str = "",
    metadata_sample: int = 200,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Valida la procedencia y prepara el dataset (paso 02). Si la procedencia falla, aborta."""
    rutas = get_project_paths()
    crudo = Path(raw_dir) if raw_dir is not None else rutas.data_raw
    procesado = ensure_dir(processed_dir if processed_dir is not None else rutas.data_processed)
    fichero_clases = (
        Path(class_names_file)
        if class_names_file is not None
        else rutas.root / "configs" / "dataset" / "sh17_classes.json"
    )
    clases = load_class_names(fichero_clases)

    logger.info("Validando la procedencia del dataset en %s...", crudo)
    procedencia = validate_provenance(crudo)
    for linea in procedencia.summary().splitlines():
        logger.info("%s", linea)
    if not procedencia.passed:
        raise ProvenanceError(procedencia.failures)
    logger.info("Procedencia validada: los %d criterios se cumplen.", len(procedencia.checks))

    logger.info("Verificando las dimensiones declaradas en meta-data/...")
    comprobacion = verify_metadata_dimensions(crudo, sample_size=metadata_sample)
    dimensiones = _dimensiones_desde_metadatos(crudo) if comprobacion.get("usable") else {}

    stems: Iterable[str] | None = None
    if smoke_test:
        todas = [p.stem for p in _listar_imagenes(crudo)]
        generador = random.Random(42)
        stems = generador.sample(todas, min(64, len(todas)))
        logger.info("Modo --smoke-test: se procesan %d imágenes.", len(list(stems)))

    logger.info("Convirtiendo las anotaciones a COCO...")
    coco, anomalias = build_coco_dataset(crudo, clases, image_stems=stems, dimensions=dimensiones)

    sufijo = "_smoke" if smoke_test else ""
    ruta_coco = procesado / f"sh17_coco{sufijo}.json"
    ruta_coco.write_text(json.dumps(coco), encoding="utf-8")
    logger.info(
        "COCO escrito en %s (%d imágenes, %d anotaciones)",
        ruta_coco,
        len(coco["images"]),
        len(coco["annotations"]),
    )

    ruta_anomalias = procesado / ANOMALIES_FILENAME
    ruta_anomalias.write_text(
        json.dumps(anomalias.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if anomalias.total:
        logger.warning(
            "Se han detectado %d anomalías en las anotaciones. Detalle en %s",
            anomalias.total,
            ruta_anomalias,
        )

    ruta_procedencia = procesado / "provenance_report.json"
    ruta_procedencia.write_text(
        json.dumps(procedencia.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ruta_atribuciones = extract_attributions(crudo, rutas.root / "docs" / ATTRIBUTIONS_FILENAME)

    ruta_manifiesto = write_manifest(
        procesado,
        crudo,
        num_images=len(coco["images"]),
        num_annotations=len(coco["annotations"]),
        provenance=procedencia,
        metadata_check=comprobacion,
        dataset_sha256=dataset_sha256,
    )

    return {
        "coco_path": str(ruta_coco),
        "manifest_path": str(ruta_manifiesto),
        "anomalies_path": str(ruta_anomalias),
        "provenance_path": str(ruta_procedencia),
        "attributions_path": str(ruta_atribuciones),
        "num_images": len(coco["images"]),
        "num_annotations": len(coco["annotations"]),
        "num_anomalies": anomalias.total,
        "metadata_dimensions_usable": bool(comprobacion.get("usable")),
        "provenance_passed": procedencia.passed,
    }


# YAML de dataset para Ultralytics

ULTRALYTICS_DATASET_FILENAME = "sh17_ultralytics.yaml"

#: Nombre de cada lista de partición en la raíz del dataset; el prefijo evita chocar con los
#: train_files.txt y val_files.txt de SH17.
ULTRALYTICS_SPLIT_TEMPLATE = "ultralytics_{split}.txt"


def _ruta_portable(nombre_fichero: str) -> str:
    """Ruta relativa de una imagen con prefijo ``./``, como la resuelve Ultralytics."""
    ruta = f"./images/{nombre_fichero}"
    # str.replace sustituye todas las apariciones de './', así que una ruta con más de una se
    # corrompería.
    if ruta.count("./") != 1:
        raise TFMEdgeError(
            f"El nombre de imagen '{nombre_fichero}' produce la ruta '{ruta}', que "
            f"contiene {ruta.count('./')} secuencias './'.",
            "Ultralytics sustituye todas las apariciones de './', no solo el "
            "prefijo, y la ruta quedaria corrupta. Renombra el fichero.",
        )
    return ruta


def verify_dataset_portability(dataset_dir: Path | str) -> dict[str, Any]:
    """Comprueba que un directorio de dataset no contiene rutas absolutas ni otras no portables.

    Se ejecuta al generar el dataset y desde 00_setup_check.py, para detectar también artefactos
    generados antes, como un tar.gz ya subido a Drive.
    """
    import yaml

    raiz = Path(dataset_dir)
    destino = raiz / ULTRALYTICS_DATASET_FILENAME
    if not destino.is_file():
        raise TFMEdgeError(
            f"No existe el YAML de dataset '{destino}'.",
            "Genéralo con scripts/02_prepare_dataset.py o 02b_resize_dataset.py.",
        )

    problemas: list[str] = []
    contenido = yaml.safe_load(destino.read_text(encoding="utf-8")) or {}

    if "path" in contenido:
        problemas.append(
            "el YAML declara la clave 'path', que ata el dataset a una ubicación; "
            "debe omitirse para que la raíz sea el directorio del propio YAML"
        )
    for clave in ("train", "val", "test"):
        valor = contenido.get(clave)
        if valor is None:
            problemas.append(f"el YAML no declara '{clave}'")
            continue
        if Path(str(valor)).is_absolute():
            problemas.append(f"el YAML declara '{clave}: {valor}', que es una ruta absoluta")

    listas_revisadas = 0
    for particion in ("train", "val", "test"):
        lista = raiz / ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)
        if not lista.is_file():
            problemas.append(f"falta la lista '{lista.name}'")
            continue
        listas_revisadas += 1
        lineas = [
            linea.strip()
            for linea in lista.read_text(encoding="utf-8").splitlines()
            if linea.strip()
        ]
        absolutas = [linea for linea in lineas if Path(linea).is_absolute()]
        if absolutas:
            problemas.append(
                f"'{lista.name}' tiene {len(absolutas)} rutas absolutas " f"(p. ej. {absolutas[0]})"
            )
        sin_prefijo = [linea for linea in lineas if not linea.startswith("./")]
        if sin_prefijo and not absolutas:
            problemas.append(
                f"'{lista.name}' tiene {len(sin_prefijo)} rutas sin el prefijo "
                f"'./' (p. ej. {sin_prefijo[0]}), que Ultralytics resolvería "
                "contra el directorio de trabajo del proceso"
            )
        dobles = [linea for linea in lineas if linea.count("./") > 1]
        if dobles:
            problemas.append(
                f"'{lista.name}' tiene {len(dobles)} rutas con más de un './', "
                "que se corrompen al resolverse"
            )

    obsoletos = [d.name for d in raiz.glob("ultralytics_splits") if d.is_dir()]
    if obsoletos:
        problemas.append(
            "queda el directorio 'ultralytics_splits/' del diseño anterior, cuyas "
            "listas llevaban rutas absolutas. Bórralo y regenera el YAML"
        )

    informe = {
        "dataset_dir": str(raiz),
        "portable": not problemas,
        "problems": problemas,
        "splits_checked": listas_revisadas,
    }
    if problemas:
        logger.error("El dataset '%s' NO es portable: %d problema(s)", raiz, len(problemas))
    else:
        logger.info("Dataset '%s' verificado como portable", raiz)
    return informe


def assert_dataset_portable(dataset_dir: Path | str) -> None:
    """Falla si el dataset contiene rutas no portables, listando cada problema."""
    informe = verify_dataset_portability(dataset_dir)
    if not informe["portable"]:
        detalle = "\n".join(f"       - {p}" for p in informe["problems"])
        raise TFMEdgeError(
            f"El dataset '{informe['dataset_dir']}' NO es portable:\n{detalle}",
            "Regenera el YAML y las listas con scripts/02b_resize_dataset.py (o "
            "02_prepare_dataset.py) y vuelve a empaquetar el tar.gz.",
        )


def write_ultralytics_dataset_yaml(
    dataset_dir: Path | str,
    splits_dir: Path | str,
    class_names: Sequence[str],
    image_extensions: Mapping[str, str] | None = None,
) -> Path:
    """Escribe en la raíz del dataset el YAML y las listas de partición que usa Ultralytics.

    Las listas salen de las particiones propias. Todo va con rutas relativas porque el directorio se
    empaqueta y se descomprime en otra ruta en Colab. BaseDataset.get_img_files solo resuelve contra
    el directorio de la lista las rutas que empiezan por ``./``; el resto las usa tal cual,
    relativas al directorio de trabajo. Por eso las listas llevan ``./``, están junto a images/ y no
    usan ``..``. El YAML no lleva la clave ``path``, así que Ultralytics toma como raíz el
    directorio del propio YAML.
    """
    import yaml

    raiz = Path(dataset_dir).resolve()
    if not (raiz / "images").is_dir():
        raise RawDatasetMissingError(f"no existe '{raiz / 'images'}'", raiz)

    extensiones = dict(image_extensions or {})
    if not extensiones:
        extensiones = {p.stem: p.suffix for p in _listar_imagenes(raiz)}

    ensure_dir(raiz)
    particiones: dict[str, str] = {}
    for particion in ("train", "val", "test"):
        origen = Path(splits_dir) / f"{particion}.txt"
        if not origen.is_file():
            raise TFMEdgeError(
                f"Falta la lista de la particion '{particion}' en '{origen}'.",
                "Genera las particiones con scripts/02_prepare_dataset.py.",
            )
        nombres = [
            linea.strip()
            for linea in origen.read_text(encoding="utf-8").splitlines()
            if linea.strip()
        ]
        lineas = []
        for nombre in nombres:
            base = Path(nombre).stem
            extension = Path(nombre).suffix or extensiones.get(base, ".jpg")
            lineas.append(_ruta_portable(f"{base}{extension}"))
        nombre_lista = ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)
        (raiz / nombre_lista).write_text("\n".join(lineas) + "\n", encoding="utf-8")
        # En el YAML también relativa, respecto a su propio directorio.
        particiones[particion] = nombre_lista

    contenido = {
        "train": particiones["train"],
        "val": particiones["val"],
        "test": particiones["test"],
        "names": {i: nombre for i, nombre in enumerate(class_names)},
    }
    cabecera = (
        "# YAML de dataset para Ultralytics, GENERADO por\n"
        "# tfm_edge.data.prepare.write_ultralytics_dataset_yaml. No editar a mano.\n"
        "#\n"
        "# Apunta a las particiones ESTRATIFICADAS del estudio, no a las de los\n"
        "# autores del dataset: entrenar sobre otra particion invalidaria la\n"
        "# comparacion con el resto de variantes.\n"
        "#\n"
        "# PORTABLE A PROPOSITO: no hay ninguna ruta absoluta, ni clave 'path'.\n"
        "# Sin 'path', Ultralytics toma como raiz el directorio de este mismo\n"
        "# fichero, asi que el dataset se puede mover o descomprimir en otra\n"
        "# maquina sin regenerar nada. Las listas usan el prefijo './', que es la\n"
        "# unica forma relativa que Ultralytics resuelve contra el directorio de\n"
        "# la lista y no contra el directorio de trabajo del proceso.\n"
        "#\n"
        "# El orden de 'names' procede de configs/dataset/sh17_classes.json, que a\n"
        "# su vez se verifico contra las anotaciones reales.\n"
    )
    destino = raiz / ULTRALYTICS_DATASET_FILENAME
    destino.write_text(
        cabecera + yaml.safe_dump(contenido, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info("YAML de dataset para Ultralytics escrito en %s", destino)
    return destino
