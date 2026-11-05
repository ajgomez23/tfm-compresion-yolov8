"""Logging del proyecto: consola legible y fichero detallado.

Los warnings de Python también se envían al log para que queden en el fichero.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path

from tfm_edge.common.paths import ensure_dir, get_project_paths, slugify

LOGGER_NAME = "tfm_edge"

#: Formato de consola: conciso, pensado para leerse en directo.
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"

#: Formato de fichero, con módulo y línea.
FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

ENV_LOG_LEVEL = "TFM_EDGE_LOG_LEVEL"


def _resolve_level(level: int | str | None) -> int:
    """Convierte un nivel de log a entero. Con None usa TFM_EDGE_LOG_LEVEL o INFO."""
    if level is None:
        level = os.environ.get(ENV_LOG_LEVEL, "INFO")
    if isinstance(level, int):
        return level
    resuelto = logging.getLevelName(str(level).upper())
    return resuelto if isinstance(resuelto, int) else logging.INFO


def setup_logging(
    experiment_name: str,
    level: int | str | None = None,
    log_dir: Path | str | None = None,
    console: bool = True,
    timestamped: bool = True,
) -> Path:
    """Configura el logging con un handler de consola y otro de fichero, y devuelve la ruta del
    fichero.

    Si se llama dos veces con el mismo destino no duplica handlers.
    """
    nivel = _resolve_level(level)
    directorio = Path(log_dir) if log_dir is not None else get_project_paths().results_logs
    ensure_dir(directorio)

    nombre = slugify(experiment_name) or "ejecucion"
    if timestamped:
        nombre = f"{nombre}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    fichero_log = directorio / f"{nombre}.log"

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(nivel)
    # Sin propagar al root para no duplicar líneas si el entorno ya tiene logging configurado.
    logger.propagate = False

    destinos_actuales = {
        Path(h.baseFilename).resolve()
        for h in logger.handlers
        if isinstance(h, logging.FileHandler)
    }
    if fichero_log.resolve() not in destinos_actuales:
        handler_fichero = logging.FileHandler(fichero_log, mode="a", encoding="utf-8")
        handler_fichero.setLevel(nivel)
        handler_fichero.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
        logger.addHandler(handler_fichero)

    if console and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        handler_consola = logging.StreamHandler(stream=sys.stderr)
        handler_consola.setLevel(nivel)
        handler_consola.setFormatter(logging.Formatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT))
        logger.addHandler(handler_consola)

    logging.captureWarnings(True)
    logger_warnings = logging.getLogger("py.warnings")
    logger_warnings.handlers = logger.handlers
    logger_warnings.setLevel(nivel)

    logger.debug("Logging configurado. Fichero de log: %s", fichero_log)
    return fichero_log


def get_logger(name: str | None = None) -> logging.Logger:
    """Logger hijo del logger del proyecto. Con None devuelve el raíz."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def reset_logging() -> None:
    """Quita los handlers del logger del proyecto. Lo usan los tests."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logging.getLogger("py.warnings").handlers = []
