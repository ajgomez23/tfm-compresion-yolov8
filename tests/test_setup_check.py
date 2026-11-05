"""Tests de la banda de versiones de Python de ``00_setup_check.py``."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from tfm_edge.common.paths import get_project_paths


def _cargar_script() -> ModuleType:
    """Carga ``scripts/00_setup_check.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "00_setup_check.py"
    especificacion = importlib.util.spec_from_file_location("setup_check_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    # El módulo tiene que estar en sys.modules antes de ejecutarlo, porque ``@dataclass`` lo busca
    # ahí.
    sys.modules["setup_check_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """El script cargado una sola vez."""
    return _cargar_script()


class TestQueLaBandaEstaCerradaPorArriba:
    """Lo que distingue una banda de un suelo."""

    def test_hay_un_maximo_declarado(self, script: ModuleType) -> None:
        """Sin extremo superior esto vuelve a ser el suelo que falló."""
        assert hasattr(script, "PYTHON_SOPORTADO_MAX")

    def test_los_extremos_son_coherentes(self, script: ModuleType) -> None:
        """Una banda invertida rechazaría cualquier versión."""
        assert script.PYTHON_SOPORTADO_MIN <= script.PYTHON_SOPORTADO_MAX

    def test_los_extremos_son_los_del_entorno_de_referencia(self, script: ModuleType) -> None:
        """3.10 por especificación, 3.12 por la última rueda de los pines."""
        assert script.PYTHON_SOPORTADO_MIN == (3, 10)
        assert script.PYTHON_SOPORTADO_MAX == (3, 12)


class TestQueLaComprobacionDistingueLosDosExtremos:
    """La función real, con la banda movida alrededor del intérprete actual."""

    def test_el_interprete_de_los_tests_esta_dentro(self, script: ModuleType) -> None:
        """Si los tests corren fuera de la banda, eso es lo primero que hay que saber."""
        resultado = script.comprobar_python()
        assert resultado.disponible, resultado.detalle

    def test_por_encima_del_maximo_no_esta_disponible(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        actual = sys.version_info[:2]
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MIN", (3, 8))
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MAX", (actual[0], actual[1] - 1))
        resultado = script.comprobar_python()
        assert not resultado.disponible
        assert "por encima" in resultado.detalle

    def test_por_debajo_del_minimo_no_esta_disponible(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """El caso que ya cubría el suelo, para que no se pierda al ensanchar."""
        actual = sys.version_info[:2]
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MIN", (actual[0], actual[1] + 1))
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MAX", (actual[0], actual[1] + 2))
        resultado = script.comprobar_python()
        assert not resultado.disponible
        assert "por debajo" in resultado.detalle

    def test_el_motivo_de_cada_extremo_es_distinto(
        self, script: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Un mensaje común obligaría al usuario a adivinar por qué lado falla."""
        actual = sys.version_info[:2]
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MIN", (3, 8))
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MAX", (actual[0], actual[1] - 1))
        por_encima = script.comprobar_python().detalle
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MIN", (actual[0], actual[1] + 1))
        monkeypatch.setattr(script, "PYTHON_SOPORTADO_MAX", (actual[0], actual[1] + 2))
        por_debajo = script.comprobar_python().detalle
        assert por_encima != por_debajo

    def test_la_comprobacion_sigue_siendo_obligatoria(self, script: ModuleType) -> None:
        """Una versión fuera de banda no puede quedar en aviso."""
        assert script.comprobar_python().obligatorio
