"""Tests de higiene del repositorio."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tfm_edge.common.paths import get_project_paths

#: Directorios de código o documentación que siempre se versionan.
DIRECTORIOS_QUE_DEBEN_VERSIONARSE: tuple[str, ...] = (
    "src",
    "scripts",
    "tests",
    "configs",
    "docs",
)

#: Extensiones que se consideran artefactos y pueden estar ignoradas legítimamente
#: aunque vivan bajo un directorio de código (por ejemplo, un .pt de prueba).
EXTENSIONES_IGNORABLES: frozenset[str] = frozenset(
    {".pyc", ".pyo", ".pt", ".pth", ".onnx", ".engine", ".tflite"}
)

#: Directorios generados por herramientas dentro del árbol de código, que deben estar ignorados.
#: ``*.egg-info`` lo genera ``pip install -e .`` dentro de ``src/``.
DIRECTORIOS_GENERADOS: tuple[str, ...] = (
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)


def _es_generado(ruta: Path) -> bool:
    """Indica si una ruta pertenece a un artefacto generado y no a código fuente."""
    return any(
        parte in DIRECTORIOS_GENERADOS or parte.endswith(".egg-info") for parte in ruta.parts
    )


def _es_repositorio_git(raiz: Path) -> bool:
    """Comprueba si un directorio es un repositorio git utilizable."""
    if shutil.which("git") is None:
        return False
    resultado = subprocess.run(
        ["git", "-C", str(raiz), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return resultado.returncode == 0 and resultado.stdout.strip() == "true"


@pytest.fixture(scope="module")
def raiz_repositorio() -> Path:
    """Devuelve la raíz del repositorio, o salta los tests si no lo es."""
    raiz = get_project_paths().root
    if not _es_repositorio_git(raiz):
        pytest.skip("El proyecto no es un repositorio git; no se puede comprobar la higiene.")
    return raiz


class TestGitignore:
    """El .gitignore no debe excluir nunca código ni documentación."""

    def test_ningun_fichero_de_codigo_esta_ignorado(self, raiz_repositorio: Path) -> None:
        """Ningún fichero bajo los directorios de código está siendo ignorado."""
        candidatos: list[Path] = []
        for nombre in DIRECTORIOS_QUE_DEBEN_VERSIONARSE:
            directorio = raiz_repositorio / nombre
            if not directorio.is_dir():
                continue
            for ruta in directorio.rglob("*"):
                if not ruta.is_file():
                    continue
                if _es_generado(ruta):
                    continue
                if ruta.suffix.lower() in EXTENSIONES_IGNORABLES:
                    continue
                candidatos.append(ruta)

        assert candidatos, "No se ha encontrado ningún fichero de código que comprobar."

        relativas = [str(r.relative_to(raiz_repositorio)) for r in candidatos]
        # check-ignore devuelve, por su salida estándar, solo las rutas ignoradas.
        resultado = subprocess.run(
            ["git", "-C", str(raiz_repositorio), "check-ignore", "--stdin"],
            input="\n".join(relativas),
            capture_output=True,
            text=True,
            check=False,
        )
        ignorados = [linea for linea in resultado.stdout.splitlines() if linea.strip()]
        assert not ignorados, (
            "El .gitignore está excluyendo ficheros que son código o documentación:\n  "
            + "\n  ".join(ignorados)
            + "\n\nRevisa los patrones. Recuerda que un patrón sin barra inicial "
            "('data/') coincide con cualquier directorio de ese nombre en todo el "
            "árbol, no solo con el de la raíz."
        )

    def test_los_directorios_de_datos_de_la_raiz_si_estan_ignorados(
        self, raiz_repositorio: Path
    ) -> None:
        for relativa in ("data/raw/imagen.jpg", "models/baselines/pesos.pt"):
            resultado = subprocess.run(
                ["git", "-C", str(raiz_repositorio), "check-ignore", "-q", relativa],
                capture_output=True,
                check=False,
            )
            assert resultado.returncode == 0, (
                f"'{relativa}' NO está ignorado y debería estarlo: los datos y los "
                "pesos son grandes y regenerables, y no se versionan."
            )

    def test_ningun_modulo_de_src_esta_ignorado(self, raiz_repositorio: Path) -> None:
        """Ningún módulo de ``src/`` figura como ignorado para git."""
        resultado = subprocess.run(
            [
                "git",
                "-C",
                str(raiz_repositorio),
                "status",
                "--porcelain",
                "--ignored=matching",
                "--",
                "src",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        ignorados = [
            linea[3:].strip()
            for linea in resultado.stdout.splitlines()
            if linea.startswith("!!") and linea.rstrip().endswith(".py")
        ]
        ignorados = [r for r in ignorados if not _es_generado(Path(r))]
        assert not ignorados, (
            "El .gitignore está ignorando módulos de Python bajo src/:\n  "
            + "\n  ".join(ignorados)
            + "\n\nSon código fuente: no pueden quedar fuera del control de versiones."
        )


class TestLasGuardasDelRegistroViverUnaSolaVez:
    """Ningún script se trae su propia copia de las guardas compartidas."""

    #: Las funciones que solo pueden estar definidas en el módulo compartido.
    COMPARTIDAS = ("check_git_clean", "read_header")

    def test_ningun_script_las_define_por_su_cuenta(self, raiz_repositorio: Path) -> None:
        """Ningún fichero de scripts/ define ninguna de las dos."""
        culpables: list[str] = []
        for guion in sorted((raiz_repositorio / "scripts").glob("*.py")):
            texto = guion.read_text(encoding="utf-8")
            for nombre in self.COMPARTIDAS:
                if f"def {nombre}(" in texto:
                    culpables.append(f"{guion.name}: define {nombre}")
        assert not culpables, (
            "Estos scripts se traen su propia copia de una guarda compartida:\n  "
            + "\n  ".join(culpables)
            + "\n\nImpórtala de tfm_edge.common.registry_guards."
        )

    def test_el_modulo_compartido_las_define_todas(self, raiz_repositorio: Path) -> None:
        modulo = raiz_repositorio / "src" / "tfm_edge" / "common" / "registry_guards.py"
        texto = modulo.read_text(encoding="utf-8")
        for nombre in self.COMPARTIDAS:
            assert f"def {nombre}(" in texto, f"{nombre} no está en el módulo compartido"

    def test_quien_las_usa_las_importa_de_alli(self, raiz_repositorio: Path) -> None:
        """Todo script que las llame tiene que importarlas del módulo compartido."""
        for guion in sorted((raiz_repositorio / "scripts").glob("*.py")):
            texto = guion.read_text(encoding="utf-8")
            usa = any(f"{nombre}(" in texto for nombre in self.COMPARTIDAS)
            if usa:
                assert "registry_guards import" in texto, (
                    f"{guion.name} usa una guarda compartida sin importarla de "
                    f"tfm_edge.common.registry_guards"
                )
