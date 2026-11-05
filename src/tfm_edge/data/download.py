"""Obtención y verificación de integridad del dataset SH17.

El repositorio oficial (https://github.com/ahmadmughees/SH17dataset) remite a Kaggle:
https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection. Kaggle pide cuenta y
credenciales de API, así que la ruta principal es verificar un archivo descargado a mano; la
descarga por API es opcional. SH17 se distribuye con licencia CC BY-NC-SA 4.0.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir, get_project_paths

logger = get_logger(__name__)

SH17_REPO_URL = "https://github.com/ahmadmughees/SH17dataset"

SH17_KAGGLE_DATASET = "mugheesahmad/sh17-dataset-for-ppe-detection"

SH17_LICENSE = "CC BY-NC-SA 4.0"

INTEGRITY_MANIFEST = "sh17_integrity.json"

_CHUNK_SIZE = 1024 * 1024


class DatasetNotAvailableError(TFMEdgeError):
    """El dataset no está disponible localmente y no se puede descargar solo."""

    def __init__(self, expected_path: Path) -> None:
        super().__init__(
            f"No se encuentra el archivo del dataset SH17 en '{expected_path}'.",
            (
                "SH17 no permite descarga automática anónima: hay que bajarlo a mano.\n"
                f"     1. Abre {SH17_KAGGLE_DATASET} en Kaggle (requiere cuenta).\n"
                f"        {SH17_REPO_URL} tiene el enlace y la ficha del dataset.\n"
                f"     2. Descarga el archivo y déjalo en '{expected_path}'.\n"
                "     3. Vuelve a ejecutar scripts/01_download_dataset.py.\n"
                "     Alternativa: si tienes credenciales de la API de Kaggle "
                "configuradas, ejecuta el script con --use-kaggle-api."
            ),
        )


class DatasetIntegrityError(TFMEdgeError):
    """El checksum del archivo no coincide con el esperado."""

    def __init__(self, path: Path, expected: str, actual: str) -> None:
        super().__init__(
            f"El archivo '{path}' no supera la verificación de integridad.\n"
            f"     Checksum esperado: {expected}\n"
            f"     Checksum obtenido: {actual}",
            "La descarga puede estar corrupta o incompleta, o el archivo puede "
            "ser una versión distinta del dataset. Bórralo y vuelve a "
            "descargarlo. Si el dataset se ha actualizado en origen, registra el "
            "nuevo checksum con --record-checksum; un cambio de datos invalida los "
            "resultados anteriores.",
        )


@dataclass
class IntegrityRecord:
    """Registro de integridad de un archivo del dataset."""

    filename: str
    sha256: str
    size_bytes: int
    source: str
    license: str
    verified_at: str

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el registro como diccionario serializable."""
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source": self.source,
            "license": self.license,
            "verified_at": self.verified_at,
        }


def compute_sha256(path: Path | str, progress_every_mb: int = 512) -> str:
    """SHA-256 de un fichero, leído por bloques porque el archivo ocupa varios gigabytes."""
    fichero = Path(path)
    if not fichero.is_file():
        raise FileNotFoundError(f"No existe el fichero a verificar: {fichero}")

    digestor = hashlib.sha256()
    leidos = 0
    siguiente_aviso = progress_every_mb * 1024 * 1024
    with fichero.open("rb") as manejador:
        while bloque := manejador.read(_CHUNK_SIZE):
            digestor.update(bloque)
            leidos += len(bloque)
            if leidos >= siguiente_aviso:
                logger.info("  verificando integridad: %.1f GB leídos", leidos / (1024**3))
                siguiente_aviso += progress_every_mb * 1024 * 1024
    return digestor.hexdigest()


def load_integrity_manifest(manifest_path: Path | str) -> dict[str, IntegrityRecord]:
    """Carga el manifiesto de integridad; vacío si no existe."""
    fichero = Path(manifest_path)
    if not fichero.is_file():
        return {}
    with fichero.open("r", encoding="utf-8") as manejador:
        contenido = json.load(manejador)
    return {
        nombre: IntegrityRecord(**datos) for nombre, datos in contenido.get("files", {}).items()
    }


def save_integrity_manifest(records: dict[str, IntegrityRecord], manifest_path: Path | str) -> Path:
    """Guarda el manifiesto de integridad."""
    destino = ensure_parent_dir(manifest_path)
    contenido = {
        "dataset": "SH17",
        "source_repo": SH17_REPO_URL,
        "kaggle_dataset": SH17_KAGGLE_DATASET,
        "license": SH17_LICENSE,
        "files": {nombre: registro.to_dict() for nombre, registro in records.items()},
    }
    with destino.open("w", encoding="utf-8") as manejador:
        json.dump(contenido, manejador, indent=2, ensure_ascii=False)
    logger.info("Manifiesto de integridad escrito en %s", destino)
    return destino


def verify_archive(
    archive_path: Path | str,
    expected_sha256: str | None = None,
    manifest_path: Path | str | None = None,
) -> IntegrityRecord:
    """Verifica el checksum de un archivo del dataset descargado a mano.

    Si no hay checksum esperado ni en el argumento ni en el manifiesto, se registra el actual como
    referencia para las siguientes ejecuciones.
    """
    from tfm_edge.common.registry import utc_timestamp

    archivo = Path(archive_path)
    if not archivo.is_file():
        raise DatasetNotAvailableError(archivo)

    ruta_manifiesto = (
        Path(manifest_path)
        if manifest_path is not None
        else get_project_paths().data_raw / INTEGRITY_MANIFEST
    )
    manifiesto = load_integrity_manifest(ruta_manifiesto)
    referencia = expected_sha256 or (
        manifiesto[archivo.name].sha256 if archivo.name in manifiesto else None
    )

    logger.info(
        "Calculando SHA-256 de %s (%.2f GB)...", archivo.name, archivo.stat().st_size / (1024**3)
    )
    obtenido = compute_sha256(archivo)

    if referencia is None:
        logger.warning(
            "No había checksum de referencia para '%s'. Se guarda el actual (%s) como "
            "referencia; esta ejecución no ha verificado nada.",
            archivo.name,
            obtenido,
        )
    elif obtenido != referencia:
        raise DatasetIntegrityError(archivo, referencia, obtenido)
    else:
        logger.info("Integridad verificada: %s coincide con la referencia.", archivo.name)

    registro = IntegrityRecord(
        filename=archivo.name,
        sha256=obtenido,
        size_bytes=archivo.stat().st_size,
        source=f"kaggle:{SH17_KAGGLE_DATASET}",
        license=SH17_LICENSE,
        verified_at=utc_timestamp(),
    )
    manifiesto[archivo.name] = registro
    save_integrity_manifest(manifiesto, ruta_manifiesto)
    return registro


def extract_archive(archive_path: Path | str, destination: Path | str, force: bool = False) -> Path:
    """Descomprime el archivo del dataset. Si el destino ya tiene contenido y no se pasa ``force``,
    no hace nada.
    """
    archivo = Path(archive_path)
    destino = ensure_dir(destination)
    if not archivo.is_file():
        raise DatasetNotAvailableError(archivo)

    if any(destino.iterdir()) and not force:
        logger.info(
            "El destino '%s' ya tiene contenido; no se vuelve a descomprimir. "
            "Usa --force para forzarlo.",
            destino,
        )
        return destino

    logger.info("Descomprimiendo %s en %s...", archivo.name, destino)
    if archivo.suffix == ".zip":
        _extraer_zip(archivo, destino)
    elif archivo.suffixes[-2:] in ([".tar", ".gz"], [".tar", ".bz2"]) or archivo.suffix in (
        ".tar",
        ".tgz",
    ):
        _extraer_tar(archivo, destino)
    else:
        raise TFMEdgeError(
            f"Formato de archivo no soportado: '{archivo.name}'.",
            "Se admiten .zip, .tar, .tar.gz, .tgz y .tar.bz2. Si Kaggle te ha "
            "entregado otro formato, descomprímelo a mano en data/raw/.",
        )
    logger.info("Descompresión terminada.")
    return destino


def _ruta_segura(destino: Path, nombre: str) -> None:
    """Comprueba que una entrada del archivo no sale del directorio de destino (rutas con
    ``..``)."""
    resuelta = (destino / nombre).resolve()
    if not str(resuelta).startswith(str(destino.resolve())):
        raise TFMEdgeError(
            f"El archivo contiene una ruta que escapa del destino: '{nombre}'.",
            "Es un archivo potencialmente malicioso. No se descomprime. "
            "Descárgalo de nuevo desde la fuente oficial.",
        )


def _extraer_zip(archivo: Path, destino: Path) -> None:
    """Descomprime un ZIP validando cada ruta."""
    with zipfile.ZipFile(archivo) as zf:
        for nombre in zf.namelist():
            _ruta_segura(destino, nombre)
        zf.extractall(destino)


def _extraer_tar(archivo: Path, destino: Path) -> None:
    """Descomprime un TAR validando cada ruta."""
    with tarfile.open(archivo) as tf:
        for miembro in tf.getmembers():
            _ruta_segura(destino, miembro.name)
        tf.extractall(destino)


def download_via_kaggle_api(destination: Path | str) -> Path:
    """Descarga SH17 con la API de Kaggle. Necesita el paquete ``kaggle`` y credenciales ya
    configuradas.
    """
    destino = ensure_dir(destination)

    if shutil.which("kaggle") is None:
        try:
            import kaggle  # noqa: F401
        except Exception as error:
            raise TFMEdgeError(
                f"No se puede usar la API de Kaggle: {error}",
                "Instala el cliente con 'pip install kaggle' y configura las credenciales "
                "(https://www.kaggle.com/docs/api), o descarga el archivo a mano y ejecuta sin "
                "--use-kaggle-api.",
            ) from error

    logger.info(
        "Descargando '%s' con la API de Kaggle en %s. Recuerda que el dataset "
        "está bajo %s, con cláusula no comercial.",
        SH17_KAGGLE_DATASET,
        destino,
        SH17_LICENSE,
    )
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(SH17_KAGGLE_DATASET, path=str(destino), unzip=False)
    except Exception as error:
        raise TFMEdgeError(
            f"La descarga desde Kaggle ha fallado: {error}",
            "Comprueba que las credenciales de Kaggle son válidas y que has aceptado las "
            "condiciones del dataset, o descárgalo a mano.",
        ) from error

    logger.info("Descarga desde Kaggle terminada.")
    return destino


def find_archive(search_dir: Path | str) -> Path | None:
    """Busca el archivo del dataset en un directorio. Si hay varios, devuelve el más grande."""
    directorio = Path(search_dir)
    if not directorio.is_dir():
        return None
    candidatos = [
        ruta
        for patron in ("*.zip", "*.tar", "*.tar.gz", "*.tgz", "*.tar.bz2")
        for ruta in directorio.glob(patron)
    ]
    if not candidatos:
        return None
    return max(candidatos, key=lambda ruta: ruta.stat().st_size)
