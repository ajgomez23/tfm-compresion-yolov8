"""Tests de la guarda de ``buscar_control``."""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from tfm_edge.common.exceptions import AmbiguousDistillationControlError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import EXPERIMENTS_COLUMNS, ExperimentRegistry
from tfm_edge.evaluation.quality_notes import build_distillation_note, read_seed_from_note

BASELINE_N = "n_baseline_default_fab9d45c"

#: Identificadores reales: los dos primeros están en disco y el tercero es el que da
#: ``compute_config_hash`` para ``--seed 43``.
PILOTO_42 = "n_distillation_l2n_combined_t2_pilot10_cf761acd"
CONTROL_42 = "n_distillation_l2n_control_alpha0_pilot10_8826269c"
CONTROL_43 = "n_distillation_l2n_control_alpha0_pilot10_b14f1f0f"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Carga ``scripts/20_distill.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "20_distill.py"
    especificacion = importlib.util.spec_from_file_location("distill_script_control", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _inscribir(
    registro: ExperimentRegistry,
    *,
    experiment_id: str,
    alpha: float,
    seed: int | None,
    model_size: str = "n",
    status: str = "ok",
    timestamp: str = "2025-08-25T08:12:08",
    on_duplicate: str = "fail",
) -> None:
    """Inscribe una fila de ``experiments`` de destilación."""
    fila = {columna: "" for columna in EXPERIMENTS_COLUMNS}
    fila.update(
        {
            "experiment_id": experiment_id,
            "parent_experiment_id": BASELINE_N,
            "timestamp": timestamp,
            "model_family": "yolov8",
            "model_size": model_size,
            "technique": "distillation",
            "technique_variant": "l2n_pilot10",
            "precision": "fp32",
            "status": status,
            "notes": build_distillation_note(alpha, seed=seed),
        }
    )
    registro.append_experiment(fila, on_duplicate=on_duplicate)  # type: ignore[arg-type]


class TestQueElControlSeSigueEncontrando:
    """La guarda no puede estorbar al caso de siempre."""

    def test_un_solo_control_se_encuentra(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=PILOTO_42, alpha=0.5, seed=42)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=42)

        assert script.buscar_control(registro, "n") == CONTROL_42

    def test_sin_control_devuelve_cadena_vacia(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=PILOTO_42, alpha=0.5, seed=42)

        assert script.buscar_control(registro, "n") == ""

    def test_el_piloto_no_se_confunde_con_el_control(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=PILOTO_42, alpha=0.5, seed=42)

        assert script.buscar_control(registro, "n") == ""

    def test_un_control_de_otro_tamano_no_cuenta(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=42)
        _inscribir(
            registro,
            experiment_id="s_distillation_l2s_control_alpha0_deadbeef",
            alpha=0.0,
            seed=43,
            model_size="s",
        )

        assert script.buscar_control(registro, "n") == CONTROL_42

    def test_un_control_fallido_no_cuenta(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        """Una fila con status distinto de ``ok`` no es una medida utilizable."""
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=42)
        _inscribir(registro, experiment_id=CONTROL_43, alpha=0.0, seed=43, status="fallido")

        assert script.buscar_control(registro, "n") == CONTROL_42


class TestQueElControlNoSeEligePorPosicion:
    """El quinto caso del patrón, y el que la campaña de semillas activa."""

    def test_dos_controles_distintos_es_un_error(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=42)
        _inscribir(
            registro,
            experiment_id=CONTROL_43,
            alpha=0.0,
            seed=43,
            timestamp="2026-09-03T10:00:00",
        )

        with pytest.raises(AmbiguousDistillationControlError) as excepcion:
            script.buscar_control(registro, "n")

        # El mensaje tiene que enseñar los dos y decir de qué semilla es cada
        # uno, o no sirve para elegir con --control-experiment-id.
        mensaje = str(excepcion.value)
        assert CONTROL_42 in mensaje
        assert CONTROL_43 in mensaje
        assert "semilla base 42" in mensaje
        assert "semilla base 43" in mensaje
        assert "--control-experiment-id" in mensaje

    def test_los_cuatro_controles_de_la_campana_empatan(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        for indice, (identificador, semilla) in enumerate(
            [
                (CONTROL_42, 42),
                (CONTROL_43, 43),
                ("n_distillation_l2n_control_alpha0_pilot10_d85357f5", 44),
                ("n_distillation_l2n_control_alpha0_pilot10_111dea57", 45),
            ]
        ):
            _inscribir(
                registro,
                experiment_id=identificador,
                alpha=0.0,
                seed=semilla,
                timestamp=f"2026-09-0{indice + 3}T10:00:00",
            )

        with pytest.raises(AmbiguousDistillationControlError) as excepcion:
            script.buscar_control(registro, "n")
        assert str(excepcion.value).count("  - ") == 4

    def test_el_mismo_control_inscrito_dos_veces_no_es_un_empate(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=42)
        _inscribir(
            registro,
            experiment_id=CONTROL_42,
            alpha=0.0,
            seed=42,
            timestamp="2026-09-03T10:00:00",
            on_duplicate="version",
        )

        assert len(registro.read_rows("experiments")) == 2
        assert script.buscar_control(registro, "n") == CONTROL_42

    def test_la_semilla_no_desempata_por_su_cuenta(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        """La otra mitad del arreglo."""
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=42)
        _inscribir(registro, experiment_id=CONTROL_43, alpha=0.0, seed=43)

        with pytest.raises(AmbiguousDistillationControlError):
            script.buscar_control(registro, "n")

    def test_la_marca_de_semilla_es_opcional(
        self, script: ModuleType, class_names: list[str], tmp_path
    ) -> None:
        """Las dos filas de la semilla 42 se inscribieron antes de que existiera."""
        registro = ExperimentRegistry(tmp_path, class_names)
        _inscribir(registro, experiment_id=CONTROL_42, alpha=0.0, seed=None)
        _inscribir(registro, experiment_id=CONTROL_43, alpha=0.0, seed=43)

        with pytest.raises(AmbiguousDistillationControlError) as excepcion:
            script.buscar_control(registro, "n")
        assert "sin marca de semilla" in str(excepcion.value)


class TestLaMarcaDeSemillaEnLaNota:
    """La semilla viaja en ``notes`` porque el registro no tiene columna."""

    def test_ida_y_vuelta(self) -> None:
        for semilla in (42, 43, 44, 45):
            assert read_seed_from_note(build_distillation_note(0.0, seed=semilla)) == semilla

    def test_sin_semilla_no_se_inventa_ninguna(self) -> None:
        assert read_seed_from_note(build_distillation_note(0.0)) is None
        assert read_seed_from_note("") is None

    def test_la_marca_no_estorba_a_la_del_control(self) -> None:
        from tfm_edge.evaluation.quality_notes import DISTILLATION_CONTROL_MARKER

        nota = build_distillation_note(0.0, seed=43)
        assert DISTILLATION_CONTROL_MARKER in nota
        assert read_seed_from_note(nota) == 43
        assert nota.startswith("evaluator=pycocotools.")

    def test_el_piloto_tambien_la_lleva(self) -> None:
        """Sin ella, emparejar piloto y control exigiría recalcular hashes."""
        nota = build_distillation_note(0.5, seed=44)
        assert read_seed_from_note(nota) == 44
