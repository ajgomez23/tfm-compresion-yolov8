"""Tests del régimen de medida declarado."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.benchmark.protocol import BenchmarkResult
from tfm_edge.common.paths import get_project_paths


@pytest.fixture(scope="module")
def script_07() -> ModuleType:
    """Carga ``scripts/07_benchmark.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "07_benchmark.py"
    especificacion = importlib.util.spec_from_file_location("benchmark_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


class TestQueElRegimenLlegaALaColumnaNotes:
    """La condición de medida se escribe donde la memoria la va a leer."""

    def test_el_regimen_va_por_delante_de_los_avisos(self, script_07: ModuleType) -> None:
        """Se lee primero la condición declarada y después lo que salió raro."""
        notas = script_07.componer_notas(
            "encadenada con presaturacion",
            ["medida con carga alta y --force: percentiles no comparables"],
        )
        assert notas.startswith("régimen declarado: encadenada con presaturacion")
        assert "carga alta" in notas
        assert notas.count(";") == 1

    def test_sin_regimen_las_notas_son_las_de_siempre(self, script_07: ModuleType) -> None:
        """El cambio no altera lo que escribían las campañas anteriores."""
        assert script_07.componer_notas("", ["a", "b"]) == "a; b"
        assert script_07.componer_notas("", []) == ""

    def test_un_regimen_sin_avisos_no_arrastra_separadores(self, script_07: ModuleType) -> None:
        assert script_07.componer_notas("saturada", []) == "régimen declarado: saturada"


class TestQueElRegimenNoEsUnAviso:
    """Declarar la condición de medida no es señalar una anomalía."""

    def test_el_argumento_regime_existe_y_por_defecto_esta_vacio(
        self, script_07: ModuleType
    ) -> None:
        """Sin ``--regime`` el comportamiento es el anterior al cambio."""
        args = script_07.construir_parser().parse_args(
            ["--profile", "workstation_cpu", "--checkpoint", "x.pt"]
        )
        assert args.regime == ""

    def test_el_argumento_regime_recoge_el_texto(self, script_07: ModuleType) -> None:
        """El texto llega tal cual, sin normalizar ni truncar."""
        args = script_07.construir_parser().parse_args(
            [
                "--profile",
                "workstation_cpu",
                "--checkpoint",
                "x.pt",
                "--regime",
                "encadenada con presaturacion",
            ]
        )
        assert args.regime == "encadenada con presaturacion"

    def test_el_codigo_de_salida_lo_deciden_los_avisos_no_las_notas(self) -> None:
        """Regresión de la línea que se cambió en ``main``."""
        texto = (get_project_paths().root / "scripts" / "07_benchmark.py").read_text(
            encoding="utf-8"
        )
        assert "return 2 if avisos else 0" in texto
        assert "return 2 if resultado.notes else 0" not in texto


class TestQueElRegimenSobreviveEnElVolcado:
    """El registro se puede reescribir; el volcado crudo es la copia que queda."""

    def test_el_campo_se_serializa(self) -> None:
        """``measurement_regime`` viaja en el JSON del volcado."""
        resultado = BenchmarkResult(
            device_id="workstation_cpu",
            backend="onnxruntime",
            measurement_regime="encadenada con presaturacion",
        )
        datos = resultado.to_dict()
        assert datos["measurement_regime"] == "encadenada con presaturacion"

    def test_por_defecto_esta_vacio_y_no_es_None(self) -> None:
        datos = BenchmarkResult(device_id="d", backend="pytorch").to_dict()
        assert datos["measurement_regime"] == ""


class TestQueLosVolcadosArchivadosSiguenLeyendose:
    """Añadir un campo no puede romper la lectura de la evidencia anterior."""

    def test_un_volcado_sin_el_campo_se_reconstruye(self) -> None:
        """El campo tiene valor por defecto, así que no es obligatorio."""
        resultado = BenchmarkResult(device_id="workstation_cpu", backend="openvino")
        assert resultado.measurement_regime == ""

    def test_el_archivo_del_01_09_no_declara_regimen(self) -> None:
        import json

        directorio = (
            get_project_paths().results
            / "archive"
            / "2026-09-01_latencia_pre_regimen_declarado"
            / "raw_benchmark"
        )
        if not directorio.is_dir():
            pytest.skip("el archivo de la campaña anterior no está en esta copia")
        volcados = sorted(directorio.glob("*.json"))
        assert volcados, "el directorio del archivo está vacío"
        for ruta in volcados:
            datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
            assert "measurement_regime" not in datos, (
                f"{ruta.name} declara un régimen: se ha escrito encima de la "
                f"evidencia archivada"
            )

    def test_el_archivo_del_17_09_declara_todo_el_regimen_de_d136(self) -> None:
        """El simétrico del anterior, para el archivo previo a la campaña única."""
        import json

        directorio = (
            get_project_paths().results
            / "archive"
            / "2026-09-17_latencia_regimen_d136_y_sesiones_declaradas"
            / "raw_benchmark"
        )
        if not directorio.is_dir():
            pytest.skip("el archivo previo a la campaña única no está en esta copia")
        volcados = sorted(directorio.glob("*.json"))
        assert len(volcados) == 50
        for ruta in volcados:
            datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
            assert datos.get("measurement_regime") == (
                "campaña encadenada con presaturación (D-136)"
            ), f"{ruta.name} no declara el régimen de medida"
