"""Tests del filtro de partición del análisis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_edge.analysis import pareto
from tfm_edge.analysis.coherence import recount_from_registry
from tfm_edge.analysis.splits import ANALYSIS_SPLIT, analysis_quality_rows, select_analysis_split
from tfm_edge.common.registry import ExperimentRegistry

RAIZ = Path(__file__).resolve().parent.parent


class _RegistroFalso:
    """Implementa solo ``read_rows`` de ``ExperimentRegistry``, que es lo que usa el filtro."""

    def __init__(self, filas: list[dict[str, str]]) -> None:
        self._filas = filas

    def read_rows(self, table: str) -> list[dict[str, str]]:
        assert table == "quality_metrics"
        return [dict(f) for f in self._filas]


def _fila(experimento: str, split: str, status: str = "ok") -> dict[str, str]:
    return {"experiment_id": experimento, "split": split, "status": status}


class TestElFiltro:
    def test_la_particion_de_analisis_es_val(self) -> None:
        assert ANALYSIS_SPLIT == "val"

    def test_se_queda_solo_con_val_y_conserva_el_orden(self) -> None:
        filas = [_fila("b", "val"), _fila("a", "test"), _fila("a", "val"), _fila("c", "")]
        assert [f["experiment_id"] for f in select_analysis_split(filas)] == ["b", "a"]

    def test_los_descartes_se_cuentan_dentro_de_val(self) -> None:
        """Una fila de test fallida no es un descarte del análisis de val."""
        registro = _RegistroFalso(
            [
                _fila("a", "val"),
                _fila("b", "val"),
                _fila("c", "val", status="fallido"),
                _fila("a", "test"),
                _fila("b", "test", status="fallido"),
                _fila("c", "test", status="fallido"),
            ]
        )
        validas, descartadas = analysis_quality_rows(registro)
        assert [f["experiment_id"] for f in validas] == ["a", "b"]
        assert all(f["split"] == "val" for f in validas)
        assert descartadas == 1


class TestContraElRegistroReal:
    """Comprobaciones sobre el registro versionado, que ya tiene filas de test."""

    @pytest.fixture(scope="class")
    def registro(self) -> ExperimentRegistry:
        return ExperimentRegistry(RAIZ / "results" / "registry")

    def test_el_registro_tiene_filas_de_test(self, registro) -> None:
        """Sin ellas, los dos tests siguientes no demostrarían nada."""
        assert any(f["split"] == "test" for f in registro.read_rows("quality_metrics"))

    def test_sin_filtro_el_cruce_de_pareto_falla_por_empate(self, registro) -> None:
        todas, _ = registro.valid_rows("quality_metrics")
        despliegue, _ = registro.latency_comparable_rows()
        objetivos = pareto.named_views()["plano_map5095_p50"]
        with pytest.raises(pareto.ParetoError):
            pareto.build_points(todas, despliegue, objetivos)
        validas, _ = analysis_quality_rows(registro)
        assert pareto.build_points(validas, despliegue, objetivos)

    def test_los_recuentos_coinciden_con_los_volcados_publicados(self, registro) -> None:
        """Los pasos 26 y 29 declaran cuántas filas de calidad usaron."""
        recontados = recount_from_registry(registro)
        for volcado in (
            RAIZ / "results" / "raw" / "pareto" / "pareto_analysis.json",
            RAIZ / "results" / "raw" / "analysis" / "results_analysis.json",
        ):
            entradas = json.loads(volcado.read_text(encoding="utf-8"))["inputs"]
            assert entradas["quality_rows_used"] == recontados["quality_rows_used"]
            assert (
                entradas["quality_rows_discarded_status"]
                == recontados["quality_rows_discarded_status"]
            )
