"""La rejilla de latencia del 5.5: de dónde sale y qué se niega a hacer."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.paths import get_project_paths


def _cargar_script() -> ModuleType:
    """Carga ``scripts/benchmark_grid.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "benchmark_grid.py"
    especificacion = importlib.util.spec_from_file_location("benchmark_grid_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["benchmark_grid_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def _manifiesto(
    directorio: Path,
    combinado: str,
    padre: str,
    *,
    estado_fp32_onnx: str = "ok",
    estado_int8_ort: str = "ok",
    con_padre: bool = True,
    clave_fp32_onnx: str = "onnx",
) -> Path:
    """Escribe un manifiesto de combinado con sus cuatro artefactos."""
    carpeta = directorio / combinado
    carpeta.mkdir(parents=True, exist_ok=True)

    def _ruta(nombre: str) -> str:
        # Ruta absoluta de otra máquina, como las que graban los manifiestos.
        destino = carpeta / nombre
        destino.write_bytes(b"x")
        return f"/otra/maquina/results/exports/combined/{combinado}/{nombre}"

    datos = {
        "source_kind": "pruned",
        "fp32_exports": {
            clave_fp32_onnx: {"status": estado_fp32_onnx, "path": _ruta(f"{combinado}_fp32.onnx")},
            "openvino": {"status": "ok", "path": _ruta(f"{combinado}_fp32.xml")},
        },
        "int8_artifacts": {
            "onnxruntime": {
                "status": estado_int8_ort,
                "path": _ruta(f"{combinado}_int8_onnxruntime.onnx"),
            },
            "openvino": {"status": "ok", "path": _ruta(f"{combinado}_int8.xml")},
        },
    }
    if con_padre:
        datos["parent_experiment_id"] = padre
    ruta = carpeta / f"{combinado}_combined.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


@pytest.fixture
def exports_dir(tmp_path: Path) -> Path:
    """Dos combinados completos, con sus manifiestos y sus ficheros."""
    directorio = tmp_path / "results" / "exports" / "combined"
    _manifiesto(directorio, "s_combined_a_1111", "s_pruning_a_aaaa")
    _manifiesto(directorio, "s_combined_b_2222", "s_pruning_b_bbbb")
    return directorio


class TestConstruccionDeLaRejilla:
    """De dónde salen las entradas y bajo qué experimento se inscriben."""

    def test_cada_combinado_aporta_cuatro_entradas(self, exports_dir: Path, tmp_path: Path) -> None:
        """Dos FP32 y dos INT8 por combinado, ni una menos."""
        entradas = _cargar_script().build_grid(exports_dir, tmp_path)
        assert len(entradas) == 8
        assert {(e.runtime, e.precision) for e in entradas} == {
            ("onnxruntime", "fp32"),
            ("onnxruntime", "int8"),
            ("openvino", "fp32"),
            ("openvino", "int8"),
        }

    def test_el_fp32_se_inscribe_bajo_el_podado_y_el_int8_bajo_el_combinado(
        self, exports_dir: Path, tmp_path: Path
    ) -> None:
        """El artefacto FP32 vive con el combinado pero mide el podado."""
        entradas = _cargar_script().build_grid(exports_dir, tmp_path)
        fp32 = {e.experiment_id for e in entradas if e.precision == "fp32"}
        int8 = {e.experiment_id for e in entradas if e.precision == "int8"}
        assert fp32 == {"s_pruning_a_aaaa", "s_pruning_b_bbbb"}
        assert int8 == {"s_combined_a_1111", "s_combined_b_2222"}

    def test_una_clave_equivocada_en_el_manifiesto_es_un_error(self, tmp_path: Path) -> None:
        directorio = tmp_path / "exports"
        _manifiesto(directorio, "s_combined_a_1111", "s_pruning_a_aaaa", clave_fp32_onnx="ort")
        with pytest.raises(TFMEdgeError, match="no tiene la clave"):
            _cargar_script().build_grid(directorio, tmp_path)

    def test_una_exportacion_fallida_se_salta_sin_romper(self, tmp_path: Path) -> None:
        """Un ``status`` que no es ``ok`` no tiene fichero que medir."""
        directorio = tmp_path / "exports"
        _manifiesto(
            directorio,
            "s_combined_a_1111",
            "s_pruning_a_aaaa",
            estado_fp32_onnx="fallido",
            estado_int8_ort="fallido",
        )
        entradas = _cargar_script().build_grid(directorio, tmp_path)
        assert len(entradas) == 2
        assert all(e.runtime == "openvino" for e in entradas)

    def test_un_manifiesto_sin_padre_es_un_error(self, tmp_path: Path) -> None:
        """Sin el padre no se sabe bajo qué experimento va el FP32."""
        directorio = tmp_path / "exports"
        _manifiesto(directorio, "s_combined_a_1111", "s_pruning_a_aaaa", con_padre=False)
        with pytest.raises(TFMEdgeError, match="parent_experiment_id"):
            _cargar_script().build_grid(directorio, tmp_path)

    def test_sin_manifiestos_no_se_inventa_una_rejilla(self, tmp_path: Path) -> None:
        directorio = tmp_path / "exports"
        directorio.mkdir()
        with pytest.raises(TFMEdgeError, match="manifiesto"):
            _cargar_script().build_grid(directorio, tmp_path)


class TestReubicacionDeRutas:
    """Los manifiestos guardan rutas absolutas de otra máquina."""

    def test_una_ruta_de_otra_maquina_se_cuelga_de_la_raiz_actual(self, tmp_path: Path) -> None:
        """El tramo desde ``results/`` es lo único que se conserva."""
        script = _cargar_script()
        reubicada = script.resolve_under_root("/otra/maquina/results/exports/x/y.onnx", tmp_path)
        assert reubicada == tmp_path / "results" / "exports" / "x" / "y.onnx"

    def test_una_ruta_fuera_de_results_no_se_reubica_a_ciegas(self, tmp_path: Path) -> None:
        """Adivinar dónde cuelga sería inventarse la procedencia."""
        script = _cargar_script()
        with pytest.raises(TFMEdgeError, match="results/"):
            script.resolve_under_root("/algun/sitio/modelo.onnx", tmp_path)


class TestGuardaContraPisarEvidencia:
    """Antes de medir se comprueba si ya hay una medida."""

    def _entrada(self, script: ModuleType) -> object:
        """Construye una entrada de rejilla mínima."""
        return script.GridEntry(
            experiment_id="s_pruning_a_aaaa",
            runtime="onnxruntime",
            precision="fp32",
            artifact=Path("x.onnx"),
            origin="s_combined_a_1111",
        )

    def test_una_fila_equivalente_cuenta_como_ya_medida(self) -> None:
        script = _cargar_script()
        filas = [
            {
                "experiment_id": "s_pruning_a_aaaa",
                "runtime": "onnxruntime",
                "precision": "fp32",
                "device_id": "workstation_cpu",
                "input_resolution": "640",
            }
        ]
        assert script.already_measured(self._entrada(script), filas, "workstation_cpu", 640)

    @pytest.mark.parametrize(
        "campo, valor",
        [
            ("runtime", "openvino"),
            ("precision", "int8"),
            ("device_id", "raspberry_pi5"),
            ("input_resolution", "320"),
        ],
    )
    def test_una_fila_que_difiere_en_algo_no_cuenta(self, campo: str, valor: str) -> None:
        """Otra resolución u otro motor no son esta medida."""
        script = _cargar_script()
        fila = {
            "experiment_id": "s_pruning_a_aaaa",
            "runtime": "onnxruntime",
            "precision": "fp32",
            "device_id": "workstation_cpu",
            "input_resolution": "640",
        }
        fila[campo] = valor
        assert not script.already_measured(self._entrada(script), [fila], "workstation_cpu", 640)

    def test_la_llamada_al_paso_07_lleva_siempre_on_duplicate_fail(self, tmp_path: Path) -> None:
        """Si el filtro se equivocara, el paso 07 se niega en vez de pisar."""
        script = _cargar_script()
        import argparse

        args = argparse.Namespace(
            profile="workstation_cpu",
            resolution=640,
            seed=42,
            registry_dir=None,
            force=False,
        )
        orden = script.build_command(
            self._entrada(script), args, tmp_path / "07.py", tmp_path / "s.json"
        )
        assert "--on-duplicate" in orden
        assert orden[orden.index("--on-duplicate") + 1] == "fail"
        assert orden[orden.index("--experiment-id") + 1] == "s_pruning_a_aaaa"
        assert orden[orden.index("--precision") + 1] == "fp32"

    def test_el_script_no_escribe_por_su_cuenta_en_el_registro(self) -> None:
        """Todas las filas las escribe el paso 07, que es quien mide."""
        texto = (get_project_paths().root / "scripts" / "benchmark_grid.py").read_text(
            encoding="utf-8"
        )
        assert "append_deployment_metrics" not in texto
        assert "read_rows" in texto


class TestCodigoDeSalidaDelPaso07:
    """El código 2 del paso 07 significa medida con avisos."""

    def test_el_dos_cuenta_como_medida_completada(self) -> None:
        """Tratarlo como fallo descartaría medidas buenas."""
        script = _cargar_script()
        assert script.EXIT_MEDIDO_CON_AVISOS == 2
        assert set(script.EXITOS) == {0, 2}

    def test_el_uno_no_cuenta_como_medida(self) -> None:
        assert 1 not in _cargar_script().EXITOS

    def test_saltar_todo_lo_ya_medido_sale_con_cero(self) -> None:
        """Volver a lanzar la campaña entera no es un aviso, es idempotencia."""
        raiz = get_project_paths().root
        registro = raiz / "results" / "registry" / "deployment_metrics.csv"
        if not registro.is_file():
            pytest.skip("No hay registro en este clon.")
        with registro.open(encoding="utf-8") as manejador:
            filas = list(csv.DictReader(manejador))
        medidas = {
            (f["experiment_id"], f["runtime"], f["precision"])
            for f in filas
            if f["device_id"] == "workstation_cpu"
        }
        exports = raiz / "results" / "exports" / "combined"
        if not exports.is_dir():
            pytest.skip("No hay artefactos combinados en este clon.")
        entradas = [
            e for e in _cargar_script().build_grid(exports, raiz) if "iter4" in e.experiment_id
        ]
        if not entradas or any(
            (e.experiment_id, e.runtime, e.precision) not in medidas for e in entradas
        ):
            pytest.skip("La parte 'iter4' de la rejilla no está medida todavía.")

        antes = registro.read_bytes()
        completado = subprocess.run(
            [
                sys.executable,
                str(raiz / "scripts" / "benchmark_grid.py"),
                "--only",
                "iter4",
                "--log-level",
                "ERROR",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completado.returncode == 0, completado.stdout + completado.stderr
        assert "SALTA" in completado.stdout
        assert "Pendientes: 0" in completado.stdout
        assert registro.read_bytes() == antes, "la campaña ha tocado el registro"


class TestContraLaRejillaDeVerdad:
    """La comprobación que de verdad importa, sobre los artefactos reales."""

    def test_la_rejilla_derivada_coincide_con_la_de_calidad(self) -> None:
        """Ni una entrada de más ni una de menos, y todas con fichero."""
        raiz = get_project_paths().root
        exports = raiz / "results" / "exports" / "combined"
        calidad_csv = raiz / "results" / "registry" / "quality_metrics.csv"
        if not exports.is_dir() or not calidad_csv.is_file():
            pytest.skip("No hay manifiestos combinados ni registro en este árbol.")

        entradas = _cargar_script().build_grid(exports, raiz)
        rejilla = {(e.experiment_id, e.runtime, e.precision) for e in entradas}
        assert len(rejilla) == len(entradas), "hay entradas repetidas"

        with calidad_csv.open(encoding="utf-8") as manejador:
            filas = list(csv.DictReader(manejador))
        calidad = {
            (f["experiment_id"], f["runtime"], f["precision"])
            for f in filas
            if f["runtime"] in ("onnxruntime", "openvino")
            and f["experiment_id"].startswith(("s_combined_", "s_pruning_"))
        }
        assert rejilla == calidad

        # Lo anterior se comprueba en cualquier clon porque los manifiestos y el registro se
        # versionan. Los .onnx y los IR no se versionan, así que lo siguiente solo se comprueba
        # donde existen.
        presentes = [e.artifact for e in entradas if e.artifact.is_file()]
        if not presentes:
            pytest.skip(
                "Los artefactos combinados no están en este árbol: de results/exports/combined/ "
                "solo se versionan los manifiestos, no los .onnx ni los IR."
            )

        faltan = [e.artifact for e in entradas if not e.artifact.is_file()]
        assert not faltan, f"artefactos que no están en disco: {faltan}"
