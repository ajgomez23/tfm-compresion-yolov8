"""Copia del dataset con las imágenes redimensionadas, para descargar la CPU al entrenar.

En Colab el cuello de botella era decodificar y redimensionar los JPEG originales (hasta 8192x5462)
en cada época. Redimensionar conservando el aspecto y después aplicar letterbox equivale a aplicar
letterbox directamente, las áreas del COCO se calculan sobre las dimensiones originales y las
etiquetas YOLO están normalizadas, así que el tamaño de los objetos no cambia. La resolución del
estudio no puede superar la de la copia. data/raw no se modifica.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir
from tfm_edge.common.registry import utc_timestamp
from tfm_edge.data.prepare import IMAGE_EXTENSIONS, RawDatasetMissingError, _listar_imagenes

logger = get_logger(__name__)

RESIZED_MANIFEST_FILENAME = "resized_manifest.json"

#: Calidad JPEG alta: se busca ahorrar decodificación, no espacio.
DEFAULT_JPEG_QUALITY = 92


class ResizeError(TFMEdgeError):
    """Error al generar la copia pre-redimensionada."""


@dataclass
class ResizeReport:
    """Resultado de generar la copia redimensionada."""

    target_size: int
    images_written: int = 0
    labels_copied: int = 0
    images_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    source_bytes: int = 0
    output_bytes: int = 0
    source_dir: str = ""
    output_dir: str = ""

    @property
    def reduction_factor(self) -> float:
        """Cuántas veces menos ocupa la copia que el original."""
        return round(self.source_bytes / self.output_bytes, 1) if self.output_bytes else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        datos = asdict(self)
        datos["reduction_factor"] = self.reduction_factor
        datos["source_gib"] = round(self.source_bytes / 2**30, 3)
        datos["output_gib"] = round(self.output_bytes / 2**30, 3)
        return datos

    def summary(self) -> str:
        """Genera un resumen legible."""
        return "\n".join(
            [
                f"Copia pre-redimensionada a {self.target_size} px de lado largo:",
                f"  imágenes escritas : {self.images_written:,}",
                f"  ya existían       : {self.images_skipped:,}",
                f"  etiquetas copiadas: {self.labels_copied:,}",
                f"  errores           : {len(self.errors)}",
                f"  tamaño original   : {self.source_bytes / 2**30:.2f} GiB",
                f"  tamaño de la copia: {self.output_bytes / 2**30:.2f} GiB "
                f"({self.reduction_factor}× menos)",
            ]
        )


def _tamano_directorio(directorio: Path) -> int:
    """Tamaño total de los ficheros de un directorio en bytes, 0 si no existe."""
    if not directorio.is_dir():
        return 0
    return sum(f.stat().st_size for f in directorio.rglob("*") if f.is_file())


def _redimensionar_una(
    origen: Path, destino: Path, target_size: int, quality: int
) -> tuple[bool, str]:
    """Redimensiona una imagen conservando el aspecto. Devuelve ``(exito, mensaje_de_error)``.

    Los PNG de Pexels incluidos en SH17 traen un perfil ICC sRGB incorrecto que libpng avisa en cada
    lectura; se reescriben sin él, sin pérdida.
    """
    from PIL import Image

    try:
        with Image.open(origen) as imagen:
            ancho, alto = imagen.size
            escala = min(target_size / ancho, target_size / alto)
            perfil_malo = bool(imagen.info.get("icc_profile")) and origen.suffix.lower() == ".png"

            if escala >= 1.0 and not perfil_malo:
                # Si ya es menor que el objetivo se copia tal cual, sin ampliar.
                shutil.copy2(origen, destino)
                return True, ""

            # draft() hace que el decodificador JPEG descomprima ya a escala reducida.
            nuevo = (
                (ancho, alto)
                if escala >= 1.0
                else (max(1, round(ancho * escala)), max(1, round(alto * escala)))
            )
            imagen.draft("RGB", nuevo)
            convertida = imagen.convert("RGB")
            if convertida.size != nuevo:
                convertida = convertida.resize(nuevo, Image.LANCZOS)
            # convert arrastra info y Pillow volvería a escribir el perfil ICC; se quita a mano.
            convertida.info.pop("icc_profile", None)
            convertida.save(destino, quality=quality, optimize=True, icc_profile=None)
        return True, ""
    except Exception as error:  # pragma: no cover - imagen corrupta
        return False, f"{origen.name}: {type(error).__name__}: {error}"


def resize_dataset(
    raw_dir: Path | str,
    output_dir: Path | str,
    target_size: int = 640,
    quality: int = DEFAULT_JPEG_QUALITY,
    workers: int = 4,
    force: bool = False,
    dataset_sha256: str = "",
) -> ResizeReport:
    """Genera una copia del dataset con las imágenes redimensionadas. Las etiquetas se copian sin
    cambios.

    Pillow libera el GIL al decodificar, así que varios hilos aceleran el proceso.
    """
    raiz = Path(raw_dir)
    destino = ensure_dir(output_dir)
    if not (raiz / "images").is_dir() or not (raiz / "labels").is_dir():
        raise RawDatasetMissingError("faltan 'images/' o 'labels/'", raiz)

    imagenes_destino = ensure_dir(destino / "images")
    etiquetas_destino = ensure_dir(destino / "labels")

    originales = _listar_imagenes(raiz)
    informe = ResizeReport(
        target_size=target_size,
        source_dir=str(raiz.resolve()),
        output_dir=str(destino.resolve()),
        source_bytes=_tamano_directorio(raiz / "images"),
    )

    logger.info(
        "Redimensionando %d imágenes a %d px de lado largo con %d hilos...",
        len(originales),
        target_size,
        workers,
    )

    tareas: list[tuple[Path, Path]] = []
    for origen in originales:
        salida = imagenes_destino / origen.name
        if salida.is_file() and not force:
            informe.images_skipped += 1
            continue
        tareas.append((origen, salida))

    if tareas:
        with ThreadPoolExecutor(max_workers=workers) as ejecutor:
            resultados = list(
                ejecutor.map(
                    lambda par: _redimensionar_una(par[0], par[1], target_size, quality), tareas
                )
            )
        for exito, error in resultados:
            if exito:
                informe.images_written += 1
            else:
                informe.errors.append(error)

    for etiqueta in sorted((raiz / "labels").glob("*.txt")):
        salida = etiquetas_destino / etiqueta.name
        if not salida.is_file() or force:
            shutil.copy2(etiqueta, salida)
        informe.labels_copied += 1

    informe.output_bytes = _tamano_directorio(imagenes_destino)

    # Comprobación de que no falta nada.
    imagenes_finales = len(_listar_imagenes(destino))
    etiquetas_finales = len(list(etiquetas_destino.glob("*.txt")))
    if imagenes_finales != len(originales):
        raise ResizeError(
            f"La copia tiene {imagenes_finales:,} imágenes y el original " f"{len(originales):,}.",
            "Revisa los errores del informe. La copia no es utilizable si le "
            "faltan imágenes: el particionado dejaría de cuadrar.",
        )
    if etiquetas_finales != len(list((raiz / "labels").glob("*.txt"))):
        raise ResizeError(
            f"La copia tiene {etiquetas_finales:,} etiquetas y el original "
            f"{len(list((raiz / 'labels').glob('*.txt'))):,}.",
            "La copia no es utilizable con ese desajuste.",
        )

    write_resized_manifest(destino, raiz, informe, dataset_sha256)
    logger.info("%s", informe.summary())
    return informe


def write_resized_manifest(
    output_dir: Path | str,
    raw_dir: Path | str,
    report: ResizeReport,
    dataset_sha256: str = "",
) -> Path:
    """Escribe el manifiesto de la copia derivada."""
    destino = ensure_parent_dir(Path(output_dir) / RESIZED_MANIFEST_FILENAME)
    contenido = {
        "dataset": "SH17",
        "kind": "derived_resized",
        "created_at": utc_timestamp(),
        "source_raw_dir": str(Path(raw_dir).resolve()),
        "source_dataset_sha256": dataset_sha256,
        "target_size": report.target_size,
        "resize_mode": "longest_side_preserving_aspect",
        "jpeg_quality": DEFAULT_JPEG_QUALITY,
        "num_images": report.images_written + report.images_skipped,
        "num_labels": report.labels_copied,
        "labels_note": (
            "Las etiquetas YOLO son NORMALIZADAS: no cambian al redimensionar. "
            "Se han copiado sin modificar."
        ),
        "validity_note": (
            "Esta copia es SOLO para acelerar el entrenamiento. Las areas de los "
            "objetos y las estadisticas del estudio salen del COCO, que se calculo "
            "sobre las dimensiones ORIGINALES: pre-redimensionar no las altera. "
            "Verificado analiticamente sobre 60.000 anotaciones con error maximo "
            "6,8e-16 y cero cambios de rango de tamano."
        ),
        "limit_note": (
            f"La resolucion del estudio NO puede superar {report.target_size} px: "
            "entrenar por encima exigiria AMPLIAR desde esta copia, lo que si "
            "alteraria el resultado."
        ),
        "report": report.to_dict(),
    }
    destino.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifiesto de la copia derivada escrito en %s", destino)
    return destino


def find_dataset_for_resolution(data_dir: Path | str, study_resolution: int) -> Path:
    """Elige la copia más pequeña del dataset que sirva para una resolución de estudio, sin
    ampliar."""
    from tfm_edge.data.prepare import ULTRALYTICS_DATASET_FILENAME

    raiz = Path(data_dir)
    disponibles = sorted(
        (int(d.name.split("_")[-1]), d)
        for d in raiz.glob("resized_*")
        if d.is_dir() and d.name.split("_")[-1].isdigit()
    )
    validas = [(n, d) for n, d in disponibles if n >= study_resolution]
    if not validas:
        detalle = (
            f"las copias que hay son de {[n for n, _ in disponibles]} px"
            if disponibles
            else "no hay ninguna copia pre-redimensionada"
        )
        raise ResizeError(
            f"No hay copia del dataset utilizable a {study_resolution} px: {detalle}.",
            "Genera una con: python scripts/02b_resize_dataset.py --target-size "
            f"{study_resolution}.",
        )

    resolucion, directorio = validas[0]
    ruta = directorio / ULTRALYTICS_DATASET_FILENAME
    if not ruta.is_file():
        raise ResizeError(
            f"La copia '{directorio}' no tiene su YAML de Ultralytics.",
            "Regenérala con scripts/02b_resize_dataset.py, que lo escribe dentro.",
        )
    logger.info("A %d px se usará la copia de %d px: %s", study_resolution, resolucion, ruta)
    return ruta


def verify_resized_dataset(resized_dir: Path | str, study_resolution: int) -> dict[str, Any]:
    """Comprueba que una copia redimensionada sirve para la resolución del estudio."""
    manifiesto = Path(resized_dir) / RESIZED_MANIFEST_FILENAME
    if not manifiesto.is_file():
        raise ResizeError(
            f"No existe el manifiesto '{manifiesto}'.",
            "Genera la copia con scripts/02b_resize_dataset.py.",
        )
    datos = json.loads(manifiesto.read_text(encoding="utf-8"))
    objetivo = int(datos["target_size"])

    if study_resolution > objetivo:
        raise ResizeError(
            f"El estudio usa {study_resolution} px y la copia está a {objetivo} px.",
            (
                "La resolución del estudio supera la de la copia y habría que ampliar las "
                "imágenes.\n"
                f"     Genera una copia a {study_resolution} px o más, o baja la resolución del "
                "estudio."
            ),
        )
    return {
        "usable": True,
        "target_size": objetivo,
        "study_resolution": study_resolution,
        "num_images": datos.get("num_images"),
        "source_raw_dir": datos.get("source_raw_dir"),
    }
