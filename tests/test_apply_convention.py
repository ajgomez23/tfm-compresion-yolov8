"""Comprueba que los scripts que mueven evidencia de Colab solo escriben con ``--apply``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tfm_edge.common.paths import get_project_paths

#: Los scripts que mueven evidencia traída a mano y comparten convención.
HERMANOS = ("14_import_colab_registry.py", "15_normalize_pruned_layout.py")


def _banderas(nombre: str) -> set[str]:
    """Las opciones largas que declara el parser de un script."""
    ruta = get_project_paths().root / "scripts" / nombre
    arbol = ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))
    encontradas: set[str] = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call) or not isinstance(nodo.func, ast.Attribute):
            continue
        if nodo.func.attr != "add_argument":
            continue
        for argumento in nodo.args:
            if isinstance(argumento, ast.Constant) and str(argumento.value).startswith("--"):
                encontradas.add(argumento.value)
    return encontradas


class TestQueLosDosHermanosDicenLoMismo:
    """La convención, comprobada sobre los dos a la vez."""

    @pytest.mark.parametrize("script", HERMANOS)
    def test_declara_apply(self, script: str) -> None:
        assert "--apply" in _banderas(script), (
            f"{script} no declara --apply. Los scripts que mueven evidencia traída "
            f"a mano solo escriben si se pide."
        )

    def test_la_lectura_de_banderas_ve_algo(self) -> None:
        for script in HERMANOS:
            declaradas = _banderas(script)
            assert len(declaradas) >= 4, f"{script}: solo se han visto {declaradas}"
            assert "--log-level" in declaradas

    def test_el_14_sigue_aceptando_la_bandera_vieja(self) -> None:
        """``--dry-run`` se acepta y avisa, en vez de romper una invocación escrita antes."""
        fuente = (get_project_paths().root / "scripts" / "14_import_colab_registry.py").read_text(
            encoding="utf-8"
        )

        assert "--dry-run" in _banderas("14_import_colab_registry.py")
        assert "obsoleta" in fuente
