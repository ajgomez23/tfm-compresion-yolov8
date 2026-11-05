"""Configuración común de la batería de tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

# El paquete puede no estar instalado (pip install -e .) cuando se ejecutan los
# tests desde un clon recién hecho. Se añade src/ al path explícitamente.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.logging_utils import reset_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _logging_limpio() -> Iterator[None]:
    """Deja el logging en un estado limpio antes y después de cada test."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def class_names() -> list[str]:
    """Nombres de clase de prueba, con casos incómodos a propósito."""
    return [
        "Persona",
        "casco de seguridad",
        "Gafas-protectoras",
        "guantes",
        "botas",
        "chaleco reflectante",
        "máscara",
    ]


@pytest.fixture
def registry_dir(tmp_path: Path) -> Path:
    """Directorio temporal para un registro de resultados aislado."""
    directorio = tmp_path / "registry"
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


@pytest.fixture
def configs_dir(tmp_path: Path) -> Path:
    """Directorio temporal donde escribir configuraciones YAML de prueba."""
    directorio = tmp_path / "configs"
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


@pytest.fixture
def escribir_yaml(configs_dir: Path):
    """Devuelve una función que escribe un YAML de prueba y devuelve su ruta."""

    def _escribir(nombre: str, contenido: str) -> Path:
        ruta = configs_dir / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(contenido, encoding="utf-8")
        return ruta

    return _escribir
