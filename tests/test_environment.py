"""Tests de la captura de la huella de entorno."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tfm_edge.common import environment
from tfm_edge.common.environment import (
    ENVIRONMENT_COLUMNS,
    _info_git,
    capture_environment,
    compute_environment_id,
    detect_execution_context,
    python_executable_info,
    summarize_environment,
)


class TestDetectExecutionContext:
    """Detección del tipo de entorno de ejecución."""

    def test_devuelve_una_etiqueta_no_vacia(self) -> None:
        contexto = detect_execution_context()
        assert isinstance(contexto, str)
        assert contexto

    def test_detecta_colab_por_variable_de_entorno(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La presencia de COLAB_GPU marca el contexto como Google Colab."""
        monkeypatch.setenv("COLAB_GPU", "1")
        assert "google_colab" in detect_execution_context()

    def test_detecta_sagemaker_por_variable_de_entorno(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """La presencia de SM_TRAINING_ENV marca el contexto como SageMaker."""
        monkeypatch.setenv("SM_TRAINING_ENV", "{}")
        assert "sagemaker" in detect_execution_context()

    def test_combina_varias_etiquetas(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Varias condiciones simultáneas se concatenan con '+'."""
        monkeypatch.setenv("COLAB_GPU", "1")
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        contexto = detect_execution_context()
        assert "google_colab" in contexto
        assert "kaggle" in contexto


class TestCaptureEnvironment:
    """Captura completa de la huella de hardware y software."""

    @pytest.fixture(scope="class")
    def huella(self) -> dict[str, Any]:
        """Captura la huella una sola vez para toda la clase."""
        return capture_environment(device_id="test_device")

    def test_tiene_exactamente_las_columnas_del_esquema(self, huella: dict[str, Any]) -> None:
        """La huella encaja en el esquema de environments.csv sin sobras ni faltas."""
        assert set(huella) == set(ENVIRONMENT_COLUMNS)

    def test_rellena_los_campos_imprescindibles(self, huella: dict[str, Any]) -> None:
        """Sistema, arquitectura, Python e identificadores nunca quedan vacíos."""
        for columna in (
            "environment_id",
            "device_id",
            "os",
            "architecture",
            "python_version",
            "execution_context",
            "timestamp",
        ):
            assert huella[columna], f"La columna '{columna}' no debería estar vacía"

    def test_el_device_id_se_propaga(self, huella: dict[str, Any]) -> None:
        """El identificador de dispositivo llega tal cual a la huella."""
        assert huella["device_id"] == "test_device"

    def test_las_ausencias_son_cadena_vacia(self, huella: dict[str, Any]) -> None:
        """Nada se codifica como 0, -1 ni el texto 'NaN' (sección 16.6)."""
        for columna, valor in huella.items():
            assert valor is not None, f"'{columna}' es None en vez de cadena vacía"
            if isinstance(valor, str):
                assert valor.strip().lower() not in {"nan", "none", "null"}

    def test_es_estable_entre_capturas(self) -> None:
        """Dos capturas seguidas describen el mismo entorno."""
        primera = capture_environment(device_id="test_device")
        segunda = capture_environment(device_id="test_device")
        assert primera["environment_id"] == segunda["environment_id"]

    def test_el_dispositivo_forma_parte_de_la_identidad(self) -> None:
        """La misma máquina con otro device_id es otro entorno de medida."""
        en_gpu = capture_environment(device_id="workstation_gpu")
        en_cpu = capture_environment(device_id="workstation_cpu")
        assert en_gpu["environment_id"] != en_cpu["environment_id"]

    def test_el_resumen_es_legible(self, huella: dict[str, Any]) -> None:
        """El resumen se genera sin errores y menciona el dispositivo."""
        resumen = summarize_environment(huella)
        assert "test_device" in resumen
        assert "Sistema" in resumen

    def test_el_resumen_tolera_una_huella_incompleta(self) -> None:
        resumen = summarize_environment({"device_id": "x"})
        assert "(no disponible)" in resumen


class TestGitDirty:
    """``git_dirty`` ignora ``results/`` y no confunde «no consultado» con limpio."""

    @pytest.fixture()
    def repo(self, tmp_path: Path) -> Path:
        """Repositorio con un fichero de código y uno de ``results/`` confirmados."""
        if shutil.which("git") is None:
            pytest.skip("git no está instalado")
        for argumentos in (
            ["init", "-q"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "test"],
        ):
            subprocess.run(["git", "-C", str(tmp_path), *argumentos], check=True)
        (tmp_path / "src").mkdir()
        (tmp_path / "results" / "registry").mkdir(parents=True)
        (tmp_path / "src" / "codigo.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "results" / "registry" / "tabla.csv").write_text("a\n", encoding="utf-8")
        (tmp_path / "results_notas.txt").write_text("a\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
        subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True)
        return tmp_path

    def test_arbol_recien_confirmado_esta_limpio(self, repo: Path) -> None:
        """Sin cambios, limpio y con commit."""
        info = _info_git(repo)
        assert info["git_dirty"] == "false"
        assert len(info["git_commit"]) == 40

    def test_escribir_en_results_no_ensucia(self, repo: Path) -> None:
        (repo / "results" / "registry" / "tabla.csv").write_text("b\n", encoding="utf-8")
        (repo / "results" / "nuevo.json").write_text("{}\n", encoding="utf-8")
        assert _info_git(repo)["git_dirty"] == "false"

    @pytest.mark.parametrize("fichero", ["src/codigo.py", "results_notas.txt"])
    def test_cambiar_codigo_si_ensucia(self, repo: Path, fichero: str) -> None:
        """Fuera de ``results/`` sí cuenta, incluido un nombre que solo empieza igual."""
        (repo / fichero).write_text("cambio\n", encoding="utf-8")
        assert _info_git(repo)["git_dirty"] == "true"

    def test_desde_un_subdirectorio_mira_el_repositorio_entero(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Con el proceso dentro de ``results/``, un cambio en ``src/`` se sigue viendo."""
        (repo / "src" / "codigo.py").write_text("x = 2\n", encoding="utf-8")
        monkeypatch.chdir(repo / "results" / "registry")
        assert _info_git()["git_dirty"] == "true"
        assert _info_git(repo / "results" / "registry")["git_dirty"] == "true"

    def test_si_git_status_falla_no_se_declara_limpio(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real = environment._ejecutar_o_none

        def falla_status(comando: list[str]) -> str | None:
            return None if "status" in comando else real(comando)

        monkeypatch.setattr(environment, "_ejecutar_o_none", falla_status)
        info = _info_git(repo)
        assert info["git_dirty"] == ""
        assert info["git_commit"]


class TestComputeEnvironmentId:
    """Estabilidad del identificador de entorno."""

    def test_longitud_configurable(self) -> None:
        """El identificador respeta la longitud pedida."""
        huella = {columna: "" for columna in ENVIRONMENT_COLUMNS}
        assert len(compute_environment_id(huella, length=8)) == 8
        assert len(compute_environment_id(huella, length=16)) == 16

    def test_el_contexto_de_ejecucion_cambia_la_identidad(self) -> None:
        """Ejecutar en Colab o en local no es el mismo entorno."""
        base = {columna: "" for columna in ENVIRONMENT_COLUMNS}
        assert compute_environment_id(
            dict(base, execution_context="local")
        ) != compute_environment_id(dict(base, execution_context="google_colab"))


class TestPythonExecutableInfo:
    """Información del intérprete en ejecución."""

    def test_devuelve_las_claves_esperadas(self) -> None:
        """La información del intérprete está completa."""
        info = python_executable_info()
        assert set(info) == {
            "executable",
            "version",
            "prefix",
            "base_prefix",
            "in_virtualenv",
        }
        assert info["executable"]
        assert info["version"]
