"""Tests de la detección de incompatibilidades binarias con la API C de NumPy."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tfm_edge.common import binary_compat
from tfm_edge.common.binary_compat import (
    NUMPY_ABI_SIGNATURES,
    BinaryCompatResult,
    assert_numpy_binary_compat,
    check_numpy_binary_compat,
    format_binary_compat_report,
)
from tfm_edge.common.exceptions import BinaryIncompatibilityError

#: Mensaje literal observado en la sesión de Colab que motivó este módulo.
MENSAJE_REAL_DE_COLAB = (
    "numpy.dtype size changed, may indicate binary incompatibility. "
    "Expected 96 from C header, got 88 from PyObject"
)


def _proceso(stdout: str = "", stderr: str = "", returncode: int = 0) -> Any:
    """Construye un resultado de subproceso falso."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture
def sonda_falsa(monkeypatch: pytest.MonkeyPatch):
    """Permite dictar qué contesta la sonda sin ejecutar ningún subproceso."""

    def instalar(resultado: Any) -> None:
        monkeypatch.setattr(binary_compat.subprocess, "run", lambda *a, **k: resultado)

    return instalar


class TestFirmas:
    """La identificación de un desajuste de ABI por su mensaje."""

    @pytest.mark.parametrize("firma", NUMPY_ABI_SIGNATURES)
    def test_cada_firma_conocida_se_reconoce(self, firma: str) -> None:
        """Toda firma declarada debe reconocerse dentro de un mensaje real."""
        assert binary_compat._parece_desajuste_de_abi(f"ValueError: {firma} en algún sitio")

    def test_el_mensaje_real_de_colab_se_reconoce(self) -> None:
        """El caso que motivó el módulo tiene que estar cubierto, literalmente."""
        assert binary_compat._parece_desajuste_de_abi(MENSAJE_REAL_DE_COLAB)

    def test_se_reconoce_sin_importar_las_mayusculas(self) -> None:
        """Los mensajes de NumPy 2 mezclan mayúsculas ("NumPy 1.x")."""
        assert binary_compat._parece_desajuste_de_abi(
            "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0"
        )

    def test_un_error_cualquiera_no_se_etiqueta_como_abi(self) -> None:
        """Etiquetar de más daría un remedio equivocado con mucha seguridad."""
        assert not binary_compat._parece_desajuste_de_abi(
            "FileNotFoundError: no existe el fichero 'datos.csv'"
        )


class TestClasificacion:
    """Cómo se traduce la respuesta de la sonda a un diagnóstico."""

    def test_un_desajuste_disfrazado_de_importerror_no_se_confunde_con_ausencia(
        self, sonda_falsa: Any
    ) -> None:
        sonda_falsa(
            _proceso(
                stdout=(
                    '{"estado": "fallo", "clase": "ImportError", '
                    f'"error": "ImportError: {MENSAJE_REAL_DE_COLAB}"}}'
                )
            )
        )
        (resultado,) = check_numpy_binary_compat([("pandas", "pandas", True)])
        assert resultado.installed is True, "un paquete roto está instalado, no ausente"
        assert resultado.abi_mismatch is True
        assert resultado.healthy is False

    def test_la_firma_de_abi_gana_al_tipo_de_excepcion(self, sonda_falsa: Any) -> None:
        """Falla si se comprueba el tipo de excepción antes que el mensaje."""
        sonda_falsa(
            _proceso(
                stdout=(
                    '{"estado": "fallo", "clase": "ModuleNotFoundError", '
                    '"error": "ModuleNotFoundError: numpy.core.multiarray failed to import"}'
                )
            )
        )
        (resultado,) = check_numpy_binary_compat([("pandas", "pandas", True)])
        assert resultado.abi_mismatch is True
        assert resultado.installed is True

    def test_un_modulo_ausente_no_se_reporta_como_incompatibilidad(self, sonda_falsa: Any) -> None:
        """Un paquete que no está no tiene un problema de compatibilidad."""
        sonda_falsa(
            _proceso(
                stdout=(
                    '{"estado": "fallo", "clase": "ModuleNotFoundError", '
                    '"error": "ModuleNotFoundError: No module named \'cv2\'"}'
                )
            )
        )
        (resultado,) = check_numpy_binary_compat([("cv2", "opencv-python", False)])
        assert resultado.installed is False
        assert resultado.abi_mismatch is False

    def test_una_caida_dura_sin_salida_se_detecta(self, sonda_falsa: Any) -> None:
        """Un desajuste de ABI puede abortar el intérprete sin lanzar excepción."""
        sonda_falsa(_proceso(stdout="", stderr=f"Fatal: {MENSAJE_REAL_DE_COLAB}", returncode=-11))
        (resultado,) = check_numpy_binary_compat([("pandas", "pandas", True)])
        assert resultado.healthy is False
        assert resultado.abi_mismatch is True
        assert "abortado el intérprete" in resultado.detail

    def test_una_caida_sin_firma_se_reporta_pero_no_se_etiqueta_como_abi(
        self, sonda_falsa: Any
    ) -> None:
        sonda_falsa(_proceso(stdout="", stderr="Killed", returncode=-9))
        (resultado,) = check_numpy_binary_compat([("pandas", "pandas", True)])
        assert resultado.healthy is False
        assert resultado.abi_mismatch is False

    def test_una_sonda_colgada_no_cuelga_la_comprobacion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        def _colgarse(*_args: Any, **_kwargs: Any) -> Any:
            raise subprocess.TimeoutExpired(cmd="python", timeout=1)

        monkeypatch.setattr(binary_compat.subprocess, "run", _colgarse)
        (resultado,) = check_numpy_binary_compat([("torch", "torch", False)], timeout=1)
        assert resultado.healthy is False
        assert "no ha terminado" in resultado.detail

    def test_una_salida_ilegible_no_se_toma_por_buena(self, sonda_falsa: Any) -> None:
        """Si el JSON no se puede leer, el veredicto es fallo, nunca OK."""
        sonda_falsa(_proceso(stdout="esto no es JSON", returncode=0))
        (resultado,) = check_numpy_binary_compat([("pandas", "pandas", True)])
        assert resultado.healthy is False


class TestSondaReal:
    """Ejecuciones de la sonda de verdad, sin sustituir el subproceso."""

    def test_el_entorno_actual_esta_sano(self) -> None:
        """Guarda viva del entorno de desarrollo."""
        resultados = check_numpy_binary_compat()
        rotos = [r for r in resultados if r.installed and not r.healthy]
        assert not rotos, format_binary_compat_report(resultados)

    def test_la_sonda_ejercita_de_verdad_y_no_solo_importa(self, tmp_path: Path) -> None:
        paquete = tmp_path / "pycocotools"
        paquete.mkdir()
        (paquete / "__init__.py").write_text("__version__ = '0.0-falso'\n", encoding="utf-8")
        (paquete / "mask.py").write_text(
            textwrap.dedent(
                f'''
                """Se importa sin ruido; revienta al tocar la capa compilada."""


                def encode(mascara):
                    raise ValueError("{MENSAJE_REAL_DE_COLAB}")
                '''
            ),
            encoding="utf-8",
        )

        import os

        anterior = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = str(tmp_path) + (os.pathsep + anterior if anterior else "")
        try:
            (resultado,) = check_numpy_binary_compat([("pycocotools", "pycocotools", False)])
        finally:
            if anterior is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = anterior

        assert resultado.installed is True
        assert resultado.healthy is False, (
            "la sonda ha dado por bueno un paquete que se importa pero no funciona: "
            "se ha quedado en el import y no ejercita la capa compilada"
        )
        assert resultado.abi_mismatch is True

    def test_un_paquete_inexistente_se_reporta_como_ausente(self) -> None:
        """Un nombre sin ejercicio definido no puede dar un OK silencioso."""
        (resultado,) = check_numpy_binary_compat(
            [("modulo_que_no_existe", "modulo-que-no-existe", False)]
        )
        assert resultado.healthy is False


class TestInforme:
    """El texto que ve el usuario."""

    def test_el_informe_incluye_el_remedio_cuando_algo_esta_roto(self) -> None:
        """Un diagnóstico sin remedio deja al usuario en el mismo sitio."""
        roto = BinaryCompatResult(
            package="pandas",
            module="pandas",
            installed=True,
            healthy=False,
            abi_mismatch=True,
            detail=MENSAJE_REAL_DE_COLAB,
            critical=True,
        )
        texto = format_binary_compat_report([roto])
        assert "pip install --force-reinstall" in texto
        assert "pandas" in texto
        # En Colab reinstalar sin reiniciar no arregla nada: el NumPy anterior
        # sigue cargado en memoria.
        assert "REINICIAR" in texto

    def test_el_informe_no_habla_de_remedios_si_todo_esta_bien(self) -> None:
        """No hay que sugerir reinstalaciones cuando no hay nada que arreglar."""
        sano = BinaryCompatResult(
            package="pandas",
            module="pandas",
            installed=True,
            healthy=True,
            abi_mismatch=False,
            detail="2.2.2 (sonda superada)",
            critical=True,
        )
        assert "pip install" not in format_binary_compat_report([sano])


class TestAssert:
    """La versión que detiene la ejecución."""

    def test_falla_si_un_paquete_critico_esta_roto(self, sonda_falsa: Any) -> None:
        """Seguir adelante con pandas roto significa no poder leer el registro."""
        sonda_falsa(
            _proceso(
                stdout=(
                    '{"estado": "fallo", "clase": "ValueError", '
                    f'"error": "ValueError: {MENSAJE_REAL_DE_COLAB}"}}'
                )
            )
        )
        with pytest.raises(BinaryIncompatibilityError) as excepcion:
            assert_numpy_binary_compat([("pandas", "pandas", True)])
        # El mensaje tiene que decir qué falla Y qué hacer (sección 3).
        assert "pandas" in str(excepcion.value)
        assert "pip install --force-reinstall" in str(excepcion.value)

    def test_no_falla_por_un_paquete_no_critico(self, sonda_falsa: Any) -> None:
        """TFLite roto no debe impedir entrenar; pandas roto sí."""
        sonda_falsa(
            _proceso(
                stdout=(
                    '{"estado": "fallo", "clase": "ValueError", '
                    f'"error": "ValueError: {MENSAJE_REAL_DE_COLAB}"}}'
                )
            )
        )
        resultados = assert_numpy_binary_compat([("cv2", "opencv-python", False)])
        assert resultados[0].abi_mismatch is True

    def test_no_falla_por_un_paquete_ausente(self, sonda_falsa: Any) -> None:
        sonda_falsa(
            _proceso(
                stdout=(
                    '{"estado": "fallo", "clase": "ModuleNotFoundError", '
                    '"error": "ModuleNotFoundError: No module named \'pandas\'"}'
                )
            )
        )
        assert_numpy_binary_compat([("pandas", "pandas", True)])
