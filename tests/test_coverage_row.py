"""Un perfil que no puede medir inscribe su ausencia con ``status=no_disponible``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from tfm_edge.benchmark import profiles
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import DEPLOYMENT_COLUMNS, ExperimentRegistry

CLASES = ("person", "helmet")

#: Las columnas que una fila de cobertura puede afirmar. Las demás van vacías
#: porque no hay medida de la que salgan, y rellenarlas con un valor plausible
#: sería inventar el contexto de una medida inexistente.
AFIRMABLES = {"device_id", "timestamp", "status", "notes"}


def _perfil_no_disponible(**campos) -> profiles.BenchmarkProfile:
    """Un perfil declarado no disponible, como los de borde."""
    base = {
        "device_id": "edge_gpu_jetson",
        "description": "Jetson que no existe en esta máquina.",
        "available": False,
        "unavailable_reason": "no se dispone del dispositivo físico",
        "latency_representative": False,
    }
    base.update(campos)
    return profiles.BenchmarkProfile(**base)


class TestLaFilaDeCobertura:
    """Qué afirma y qué calla."""

    def test_el_experiment_id_va_vacio(self) -> None:
        """La decisión, y el único punto que no se puede leer de otra forma."""
        fila = _perfil_no_disponible().unavailable_coverage_row()

        assert fila["experiment_id"] == ""

    def test_solo_afirma_lo_que_puede(self) -> None:
        fila = _perfil_no_disponible().unavailable_coverage_row()

        con_valor = {columna for columna in DEPLOYMENT_COLUMNS if str(fila[columna]).strip()}
        assert con_valor == AFIRMABLES, f"Afirma de más o de menos: {con_valor}"

    def test_el_motivo_viaja_con_la_fila(self) -> None:
        """Una fila que dice que falta algo y no dice qué es peor que ninguna."""
        fila = _perfil_no_disponible().unavailable_coverage_row()

        assert fila["notes"].startswith("COBERTURA, no medida.")
        assert "dispositivo físico" in fila["notes"]

    @pytest.mark.parametrize(
        "campos, clave",
        [
            ({"available": False, "unavailable_reason": ""}, "unavailable_reason"),
            ({"valid_for_measurement": False, "invalid_reason": ""}, "invalid_reason"),
        ],
    )
    def test_un_perfil_sin_motivo_no_llega_a_existir(self, campos, clave) -> None:
        base = {
            "device_id": "edge_gpu_jetson",
            "description": "Jetson que no existe en esta máquina.",
            "latency_representative": False,
            **campos,
        }

        with pytest.raises(ValidationError, match=clave):
            profiles.BenchmarkProfile(**base)

    def test_la_fila_encaja_en_el_esquema(self, tmp_path: Path) -> None:
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=CLASES)

        assert registro.append_deployment_metrics(
            _perfil_no_disponible().unavailable_coverage_row(), on_duplicate="skip"
        )

        filas = registro.read_rows("deployment_metrics")
        assert [f["device_id"] for f in filas] == ["edge_gpu_jetson"]
        assert filas[0]["status"] == "no_disponible"

    def test_inscribirla_dos_veces_no_duplica(self, tmp_path: Path) -> None:
        """Con ``skip``, reintentar la campaña no ensucia el registro."""
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=CLASES)
        perfil = _perfil_no_disponible()

        assert registro.append_deployment_metrics(
            perfil.unavailable_coverage_row(), on_duplicate="skip"
        )
        assert not registro.append_deployment_metrics(
            perfil.unavailable_coverage_row(), on_duplicate="skip"
        )

        assert len(registro.read_rows("deployment_metrics")) == 1


class TestQueLaFilaSaleSolaDesdeElPaso07:
    """Una guarda escrita y no cableada no protege de nada."""

    def test_el_07_la_inscribe_en_el_except(self) -> None:
        ruta = get_project_paths().root / "scripts" / "07_benchmark.py"
        arbol = ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))

        manejadores = [
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.ExceptHandler) and "unavailable_coverage_row" in ast.dump(nodo)
        ]

        assert manejadores, "07_benchmark.py no inscribe la cobertura en ningún except."

    def test_sigue_devolviendo_uno(self) -> None:
        """Que la ausencia quede escrita no convierte en éxito una campaña sin latencias."""
        ruta = get_project_paths().root / "scripts" / "07_benchmark.py"
        arbol = ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))

        manejador = next(
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.ExceptHandler) and "unavailable_coverage_row" in ast.dump(nodo)
        )
        devueltos = [
            nodo.value.value
            for nodo in ast.walk(manejador)
            if isinstance(nodo, ast.Return) and isinstance(nodo.value, ast.Constant)
        ]

        assert devueltos == [1], f"El except devuelve {devueltos}, no 1."


class TestQueLosPerfilesDeVerdadSiguenEncajando:
    """El doble de arriba solo vale mientras se parezca a los perfiles reales."""

    def test_los_perfiles_no_disponibles_del_proyecto_dan_su_fila(self) -> None:
        directorio = get_project_paths().root / "configs" / "benchmark"
        todos = profiles.load_all_profiles(directorio)
        assert len(todos) >= 5, f"Solo se han visto {len(todos)} perfiles."

        vistos = 0
        for perfil in todos.values():
            if perfil.available and perfil.valid_for_measurement:
                continue
            fila = perfil.unavailable_coverage_row()
            assert fila["experiment_id"] == ""
            assert fila["device_id"] == perfil.device_id
            assert fila["notes"].strip() != "COBERTURA, no medida."
            vistos += 1

        assert vistos >= 2, (
            f"Solo {vistos} perfiles no disponibles: si todos pasaran a estarlo, "
            f"este test dejaría de mirar nada."
        )
