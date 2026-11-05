"""Construcción de detectores y descarga de pesos preentrenados.

Todos los modelos del estudio salen de build_detector, así que baseline y variantes parten de la
misma resolución y número de clases. Los pesos oficiales vienen en FP16 y se promocionan a FP32 al
cargar.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, get_project_paths
from tfm_edge.models.detector import VALID_SIZES, Detector

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    pass

logger = get_logger(__name__)

WEIGHTS_BASE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

#: Margen de disco por encima del tamaño del fichero, en bytes.
DISK_SAFETY_MARGIN_BYTES = 512 * 2**20

#: Tamaño mínimo razonable de un checkpoint; por debajo suele ser una página de error.
MIN_PLAUSIBLE_WEIGHTS_BYTES = 1 * 2**20


class FactoryError(TFMEdgeError):
    """Error al construir o preparar un detector."""


class WeightsDownloadError(TFMEdgeError):
    """Error al descargar o validar unos pesos preentrenados."""


def pretrained_weights_filename(size: str) -> str:
    """Nombre del fichero de pesos oficial para un tamaño, por ejemplo ``yolov8s.pt``."""
    if size not in VALID_SIZES:
        raise FactoryError(
            f"Tamaño de modelo '{size}' no válido.",
            f"Los válidos son {list(VALID_SIZES)}.",
        )
    return f"yolov8{size}.pt"


def weights_directory() -> Path:
    """Directorio de pesos preentrenados, creado si no existe."""
    return ensure_dir(get_project_paths().data / "weights")


def _verificar_fichero_de_pesos(ruta: Path) -> None:
    """Comprueba que un fichero de pesos es plausible y se puede abrir."""
    tamano = ruta.stat().st_size
    if tamano < MIN_PLAUSIBLE_WEIGHTS_BYTES:
        raise WeightsDownloadError(
            f"'{ruta.name}' ocupa solo {tamano:,} bytes, que no es un modelo.",
            "Lo más probable es que la descarga devolviera una página de error. "
            "Borra el fichero y vuelve a intentarlo.",
        )
    try:
        import torch

        torch.load(ruta, map_location="cpu", weights_only=False)
    except Exception as error:
        raise WeightsDownloadError(
            f"'{ruta.name}' no se puede abrir como checkpoint: " f"{type(error).__name__}: {error}",
            "El fichero está corrupto o incompleto. Bórralo y vuelve a descargarlo.",
        ) from error


def download_pretrained_weights(
    size: str, destination: Path | str | None = None, force: bool = False
) -> Path:
    """Descarga los pesos preentrenados de un tamaño si no están ya en disco."""
    import urllib.error
    import urllib.request

    nombre = pretrained_weights_filename(size)
    directorio = ensure_dir(destination) if destination else weights_directory()
    destino = directorio / nombre

    if destino.is_file() and not force:
        try:
            _verificar_fichero_de_pesos(destino)
            logger.info("Pesos ya presentes y válidos: %s", destino)
            return destino
        except WeightsDownloadError as error:
            logger.warning("Los pesos existentes no son válidos (%s). Se rebajan.", error)

    url = f"{WEIGHTS_BASE_URL}/{nombre}"
    libre = shutil.disk_usage(directorio).free
    if libre < DISK_SAFETY_MARGIN_BYTES:
        raise WeightsDownloadError(
            f"Solo quedan {libre / 2**20:.0f} MiB libres en '{directorio}' (margen: "
            f"{DISK_SAFETY_MARGIN_BYTES / 2**20:.0f} MiB).",
            "Libera espacio antes de descargar los pesos.",
        )

    logger.info("Descargando %s ...", url)
    temporal = destino.with_suffix(destino.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, temporal)  # noqa: S310 - URL fija y conocida
    except (urllib.error.URLError, OSError) as error:
        temporal.unlink(missing_ok=True)
        raise WeightsDownloadError(
            f"No se han podido descargar los pesos de '{url}': " f"{type(error).__name__}: {error}",
            "Comprueba la conexión. Si estás sin red, descarga el fichero a mano "
            f"y déjalo en '{destino}'.",
        ) from error

    # Se valida antes de renombrarlo, para no dejar un fichero corrupto en lugar del bueno.
    try:
        _verificar_fichero_de_pesos(temporal)
    except WeightsDownloadError:
        temporal.unlink(missing_ok=True)
        raise
    temporal.replace(destino)
    logger.info("Pesos descargados en %s (%.1f MiB)", destino, destino.stat().st_size / 2**20)
    return destino


def build_detector(
    size: str,
    num_classes: int = 17,
    input_resolution: int = 640,
    pretrained: bool = True,
    weights: Path | str | None = None,
    device: str = "cpu",
    download_if_missing: bool = True,
) -> Detector:
    """Construye un detector en FP32, con o sin pesos preentrenados.

    ``weights`` tiene prioridad sobre ``pretrained`` y permite partir de un modelo ya entrenado.
    """
    if size not in VALID_SIZES:
        raise FactoryError(
            f"Tamaño de modelo '{size}' no válido.",
            f"Los válidos son {list(VALID_SIZES)}.",
        )
    if input_resolution % 32 != 0:
        raise FactoryError(
            f"La resolución {input_resolution} no es múltiplo de 32.",
            "YOLOv8 submuestrea por 32; usa por ejemplo 416, 512 o 640.",
        )
    if num_classes < 1:
        raise FactoryError(f"num_classes debe ser >= 1 y es {num_classes}.", "Usa 17 para SH17.")

    ruta: Path | None = None
    if weights is not None:
        ruta = Path(weights)
        if not ruta.is_file():
            raise FactoryError(
                f"No existe el checkpoint '{ruta}'.",
                "Comprueba la ruta, o entrena el modelo antes de partir de él.",
            )
    elif pretrained:
        ruta = (
            download_pretrained_weights(size)
            if download_if_missing
            else weights_directory() / pretrained_weights_filename(size)
        )
        if not ruta.is_file():
            raise WeightsDownloadError(
                f"Faltan los pesos preentrenados '{ruta}'.",
                "Ejecuta con descarga habilitada, o deja el fichero ahí a mano.",
            )

    if ruta is not None:
        detector = Detector.from_pretrained_file(
            ruta,
            size=size,
            num_classes=num_classes,
            input_resolution=input_resolution,
            map_location=device,
        )
    else:
        detector = Detector.from_architecture(
            size, num_classes=num_classes, input_resolution=input_resolution
        )

    detector.to(device)
    detector.validate_structure()
    # Los pesos oficiales vienen en FP16; Detector.load los pasa a FP32 y aquí se comprueba.
    detector.assert_float32()
    logger.info(
        "Detector yolov8%s construido: %s clases, %d px, %s",
        size,
        num_classes,
        input_resolution,
        "preentrenado" if (pretrained or weights) else "desde cero",
    )
    return detector


def build_teacher_student(
    teacher_size: str = "l",
    student_size: str = "n",
    num_classes: int = 17,
    input_resolution: int = 640,
    teacher_weights: Path | str | None = None,
    device: str = "cpu",
) -> tuple[Detector, Detector]:
    """Construye el par maestro-alumno de la destilación (por defecto yolov8l y yolov8n).

    El maestro se devuelve congelado y en modo evaluación. ``teacher_weights`` es obligatorio y debe
    ser un checkpoint entrenado sobre SH17.
    """
    if teacher_size == student_size:
        raise FactoryError(
            f"Maestro y alumno son del mismo tamaño ('{teacher_size}').",
            "Maestro y alumno deben tener tamaños distintos, por ejemplo l y n.",
        )

    # Es un error y no un aviso: con pesos de COCO el maestro no sabe detectar EPI.
    if teacher_weights is None:
        raise FactoryError(
            "No se ha indicado 'teacher_weights': el maestro se construiría con "
            "los pesos de COCO, no con un modelo afinado sobre SH17.",
            "Entrena antes el baseline del maestro y pasa su checkpoint:\n"
            f"     python scripts/04_train_baselines.py --sizes {teacher_size}\n"
            "     y luego results/runs/"
            f"{teacher_size}_baseline_default/weights/best.pt",
        )

    maestro = build_detector(
        teacher_size,
        num_classes=num_classes,
        input_resolution=input_resolution,
        weights=teacher_weights,
        device=device,
    )
    alumno = build_detector(
        student_size,
        num_classes=num_classes,
        input_resolution=input_resolution,
        device=device,
    )

    maestro.eval()
    for parametro in maestro.model.parameters():
        parametro.requires_grad_(False)

    logger.info(
        "Par de destilación listo: maestro yolov8%s (congelado), alumno yolov8%s",
        teacher_size,
        student_size,
    )
    return maestro, alumno


def describe_available_weights() -> dict[str, Any]:
    """Pesos preentrenados presentes y ausentes en el directorio de pesos."""
    directorio = weights_directory()
    presentes: dict[str, int] = {}
    faltan: list[str] = []
    for size in VALID_SIZES:
        ruta = directorio / pretrained_weights_filename(size)
        if ruta.is_file():
            presentes[ruta.name] = ruta.stat().st_size
        else:
            faltan.append(ruta.name)
    return {
        "directory": str(directorio),
        "present": presentes,
        "missing": faltan,
        "free_bytes": shutil.disk_usage(directorio).free,
    }
