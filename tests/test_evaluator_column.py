"""Tests de la columna ``evaluator`` de ``quality_metrics``."""

from __future__ import annotations

import csv

import pytest

from tfm_edge.common.config import load_class_names
from tfm_edge.common.exceptions import RegistrySchemaError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import (
    TABLE_PRIMARY_KEYS,
    VALID_EVALUATORS,
    ExperimentRegistry,
    build_quality_columns,
)
from tfm_edge.evaluation.quality_notes import (
    CANONICAL_EVALUATOR,
    EVALUATOR_PREFIX,
    ULTRALYTICS_EVALUATOR,
    parse_evaluator_from_notes,
)


def _clases() -> list[str]:
    """Nombres de clase canónicos del proyecto."""
    rutas = get_project_paths()
    return load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")


def _fila(clases: list[str], **cambios: str) -> dict:
    """Construye una fila completa de ``quality_metrics`` para los tests."""
    fila = {columna: "" for columna in build_quality_columns(clases)}
    fila.update(
        {
            "experiment_id": "s_baseline_default_a1b2c3d4",
            "runtime": "pytorch",
            "precision": "fp32",
            "device_id": "workstation_cpu",
            "evaluator": CANONICAL_EVALUATOR,
            "split": "val",
            "map_50": "0.5100",
            "map_50_95": "0.3171",
            "status": "ok",
        }
    )
    fila.update(cambios)
    return fila


class TestQueElEsquemaObligaADeclararElEvaluador:
    """El evaluador deja de ser opcional y deja de ser prosa."""

    def test_la_columna_esta_en_el_esquema(self) -> None:
        assert "evaluator" in build_quality_columns(_clases())

    def test_la_columna_esta_en_la_clave_primaria(self) -> None:
        assert "evaluator" in TABLE_PRIMARY_KEYS["quality_metrics"]

    def test_una_fila_sin_evaluador_se_rechaza(self, tmp_path) -> None:
        registro = ExperimentRegistry(tmp_path, _clases())
        with pytest.raises(RegistrySchemaError, match="evaluator"):
            registro.append_quality_metrics(_fila(_clases(), evaluator=""), on_duplicate="fail")

    def test_un_evaluador_desconocido_se_rechaza(self, tmp_path) -> None:
        registro = ExperimentRegistry(tmp_path, _clases())
        with pytest.raises(RegistrySchemaError, match="no es válido"):
            registro.append_quality_metrics(
                _fila(_clases(), evaluator="pycoco"), on_duplicate="fail"
            )

    def test_las_dos_ramas_conviven_sin_ser_duplicado(self, tmp_path) -> None:
        clases = _clases()
        registro = ExperimentRegistry(tmp_path, clases)
        registro.append_quality_metrics(
            _fila(clases, evaluator=CANONICAL_EVALUATOR), on_duplicate="fail"
        )
        registro.append_quality_metrics(
            _fila(clases, evaluator=ULTRALYTICS_EVALUATOR, map_50_95="0.3302"),
            on_duplicate="fail",
        )
        filas = registro.read_rows("quality_metrics")
        assert [f["evaluator"] for f in filas] == [CANONICAL_EVALUATOR, ULTRALYTICS_EVALUATOR]
        assert [f["map_50_95"] for f in filas] == ["0.3171", "0.3302"]


class TestQueElLectorDeNotasNoSeInventaNada:
    """La migración lee el dato; no lo rellena a ojo."""

    def test_lee_el_prefijo(self) -> None:
        assert parse_evaluator_from_notes("evaluator=pycocotools; y más cosas") == "pycocotools"
        assert parse_evaluator_from_notes("evaluator=ultralytics. y más") == "ultralytics"

    def test_una_nota_sin_prefijo_no_devuelve_un_valor_por_defecto(self) -> None:
        """Suponer "pycocotools porque es lo habitual" es justo el error."""
        assert parse_evaluator_from_notes("evaluación COCO con los dos criterios") == ""
        assert parse_evaluator_from_notes("") == ""

    def test_no_lo_lee_si_no_esta_al_principio(self) -> None:
        assert parse_evaluator_from_notes("medido con evaluator=pycocotools") == ""

    def test_el_prefijo_se_deriva_de_la_constante(self) -> None:
        """Columna y prosa salen del mismo valor, así que no pueden divergir."""
        assert parse_evaluator_from_notes(EVALUATOR_PREFIX) == CANONICAL_EVALUATOR
        assert CANONICAL_EVALUATOR in VALID_EVALUATORS
        assert ULTRALYTICS_EVALUATOR in VALID_EVALUATORS


class TestQueElRegistroRealSigueCuadrando:
    """Comprobaciones sobre el CSV del repositorio."""

    @pytest.fixture()
    def filas(self) -> list[dict[str, str]]:
        ruta = get_project_paths().results_registry / "quality_metrics.csv"
        if not ruta.is_file():
            pytest.skip("no hay registro en esta copia del repositorio")
        with ruta.open(encoding="utf-8", newline="") as manejador:
            return list(csv.DictReader(manejador))

    def test_la_columna_existe_en_el_fichero(self, filas: list[dict[str, str]]) -> None:
        if not filas:
            pytest.skip("registro vacío")
        assert "evaluator" in filas[0]

    def test_ninguna_fila_se_queda_sin_evaluador(self, filas: list[dict[str, str]]) -> None:
        huerfanas = [
            f"L{i + 2} ({f.get('experiment_id')})"
            for i, f in enumerate(filas)
            if not f["evaluator"]
        ]
        assert not huerfanas, f"filas sin evaluador: {huerfanas}"

    def test_todos_los_valores_son_conocidos(self, filas: list[dict[str, str]]) -> None:
        desconocidos = {f["evaluator"] for f in filas} - set(VALID_EVALUATORS)
        assert not desconocidos, f"evaluadores fuera de VALID_EVALUATORS: {desconocidos}"

    def test_la_columna_y_la_nota_dicen_lo_mismo(self, filas: list[dict[str, str]]) -> None:
        """Dos fuentes para el mismo dato solo son seguras si se comprueban."""
        discrepantes = [
            f"L{i + 2} ({f.get('experiment_id')}): columna='{f['evaluator']}', "
            f"notes='{parse_evaluator_from_notes(f.get('notes', ''))}'"
            for i, f in enumerate(filas)
            if parse_evaluator_from_notes(f.get("notes", "")) != f["evaluator"]
        ]
        assert not discrepantes, f"columna y notes discrepan en: {discrepantes}"
