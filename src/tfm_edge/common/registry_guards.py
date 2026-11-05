"""Comprobaciones comunes a los scripts que reescriben el registro.

check_git_clean asegura que la reescritura se puede deshacer con git y read_header permite comparar
la cabecera en disco con el esquema antes de escribir. Ninguna aborta por sí sola.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def check_git_clean(registry_dir: Path) -> str | None:
    """Devuelve None si el registro no tiene cambios sin confirmar en git, o el motivo si los
    tiene."""
    try:
        completado = subprocess.run(
            ["git", "status", "--porcelain", "--", str(registry_dir)],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(registry_dir.parent),
        )
    except OSError as error:  # git no instalado, o directorio inaccesible
        return f"no se ha podido ejecutar git ({error})"
    if completado.returncode != 0:
        return f"git ha fallado: {completado.stderr.strip() or 'sin mensaje'}"
    pendiente = completado.stdout.strip()
    if pendiente:
        return f"hay cambios sin confirmar:\n{pendiente}"
    return None


def read_header(csv_path: Path) -> tuple[str, ...]:
    """Lee la cabecera de un CSV sin cargar el resto del fichero."""
    if not csv_path.is_file():
        return ()
    with csv_path.open("r", encoding="utf-8", newline="") as manejador:
        return tuple(next(csv.reader(manejador), []))
