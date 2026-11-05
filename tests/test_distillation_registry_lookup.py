"""Tests de ``leer_metrica_del_registro`` cuando varias filas encajan con el filtro."""

from __future__ import annotations

import importlib.util
from types import ModuleType

import pytest

from tfm_edge.common.config import load_class_names
from tfm_edge.common.exceptions import RegistryAmbiguousRowError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import ExperimentRegistry, build_quality_columns
from tfm_edge.compression.distillation.reading import (
    EVALUATOR_READABILITY_FLOOR,
    STARTING_BASELINE_ID,
)
from tfm_edge.evaluation.quality_notes import CANONICAL_EVALUATOR, ULTRALYTICS_EVALUATOR

BASELINE_N = "n_baseline_default_fab9d45c"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Carga ``scripts/20_distill.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "20_distill.py"
    especificacion = importlib.util.spec_from_file_location("distill_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def clases() -> list[str]:
    """Nombres de clase canónicos del proyecto."""
    rutas = get_project_paths()
    return load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")


def _inscribir(
    registro: ExperimentRegistry,
    clases: list[str],
    *,
    experiment_id: str = BASELINE_N,
    evaluator: str = CANONICAL_EVALUATOR,
    device_id: str = "workstation_cpu",
    map_50_95: str = "0.325636",
    split: str = "val",
) -> None:
    """Inscribe una fila de calidad mínima pero completa."""
    fila = {columna: "" for columna in build_quality_columns(clases)}
    fila.update(
        {
            "experiment_id": experiment_id,
            "runtime": "pytorch",
            "precision": "fp32",
            "device_id": device_id,
            "evaluator": evaluator,
            "split": split,
            "map_50": "0.5100",
            "map_50_95": map_50_95,
            "status": "ok",
            "notes": f"evaluator={evaluator}. fila de test",
        }
    )
    registro.append_quality_metrics(fila, on_duplicate="fail")


class TestQueElEvaluadorManda:
    """El filtro que sustituye al orden del fichero."""

    def test_devuelve_la_del_evaluador_pedido_aunque_vaya_la_ultima(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        """El caso real, con las filas al revés."""
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(
            registro,
            clases,
            evaluator=ULTRALYTICS_EVALUATOR,
            device_id="auto",
            map_50_95="0.330215",
        )
        _inscribir(registro, clases)  # la buena, la segunda del fichero

        metrica = script.leer_metrica_del_registro(registro, BASELINE_N, "val")
        assert metrica["map_50_95"] == pytest.approx(0.325636)
        assert metrica["evaluator"] == CANONICAL_EVALUATOR
        assert metrica["device_id"] == "workstation_cpu"

    def test_la_diferencia_descartada_es_el_suelo_de_lectura(self) -> None:
        """Por qué esto no es un detalle de higiene."""
        assert 0.330215 - 0.325636 == pytest.approx(EVALUATOR_READABILITY_FLOOR, abs=1e-6)
        assert STARTING_BASELINE_ID == BASELINE_N

    def test_si_solo_hay_del_otro_evaluador_no_devuelve_nada(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        """Mejor sin cifra que con la cifra de otra escala."""
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(
            registro,
            clases,
            evaluator=ULTRALYTICS_EVALUATOR,
            device_id="auto",
            map_50_95="0.330215",
        )
        assert script.leer_metrica_del_registro(registro, BASELINE_N, "val") == {}

    def test_se_puede_pedir_el_otro_evaluador_explicitamente(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(
            registro,
            clases,
            evaluator=ULTRALYTICS_EVALUATOR,
            device_id="auto",
            map_50_95="0.330215",
        )
        metrica = script.leer_metrica_del_registro(
            registro, BASELINE_N, "val", evaluator=ULTRALYTICS_EVALUATOR
        )
        assert metrica["map_50_95"] == pytest.approx(0.330215)


class TestQueUnEmpateNoSeResuelveEnSilencio:
    """Dos medidas del mismo evaluador que no coinciden provocan un error."""

    def test_dos_dispositivos_distintos_sin_desempate_es_un_error(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(registro, clases, device_id="workstation_cpu", map_50_95="0.325636")
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50_95="0.326100")

        with pytest.raises(RegistryAmbiguousRowError) as excepcion:
            script.leer_metrica_del_registro(registro, BASELINE_N, "val")
        # El mensaje tiene que enseñar las dos, o no sirve para decidir.
        assert "workstation_cpu" in str(excepcion.value)
        assert "colab_t4_gpu" in str(excepcion.value)

    def test_el_dispositivo_preferido_desempata(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(registro, clases, device_id="workstation_cpu", map_50_95="0.325636")
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50_95="0.326100")

        metrica = script.leer_metrica_del_registro(
            registro, BASELINE_N, "val", preferred_device_id="colab_t4_gpu"
        )
        assert metrica["map_50_95"] == pytest.approx(0.326100)
        assert metrica["device_id"] == "colab_t4_gpu"

    def test_un_dispositivo_preferido_que_no_esta_no_inventa_nada(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(registro, clases, device_id="workstation_cpu", map_50_95="0.325636")
        _inscribir(registro, clases, device_id="colab_t4_gpu", map_50_95="0.326100")

        with pytest.raises(RegistryAmbiguousRowError):
            script.leer_metrica_del_registro(
                registro, BASELINE_N, "val", preferred_device_id="jetson_orin_nano"
            )

    def test_dos_filas_que_dicen_lo_mismo_no_son_una_ambiguedad(
        self, script: ModuleType, clases: list[str], tmp_path
    ) -> None:
        """La misma medida inscrita dos veces no obliga a decidir nada."""
        registro = ExperimentRegistry(tmp_path, clases)
        _inscribir(registro, clases, device_id="workstation_cpu", map_50_95="0.325636")
        _inscribir(registro, clases, device_id="constrained_cpu", map_50_95="0.325636")

        metrica = script.leer_metrica_del_registro(registro, BASELINE_N, "val")
        assert metrica["map_50_95"] == pytest.approx(0.325636)


class TestSobreElRegistroDeVerdad:
    """La consulta que hizo el piloto, contra el CSV del repositorio."""

    def test_el_punto_de_partida_sale_del_evaluador_canonico(
        self, script: ModuleType, clases: list[str]
    ) -> None:
        registro = ExperimentRegistry(get_project_paths().results_registry, clases)
        metrica = script.leer_metrica_del_registro(registro, STARTING_BASELINE_ID, "val")
        if not metrica:
            pytest.skip("este repositorio no trae la fila del baseline 'n'")
        assert metrica["evaluator"] == CANONICAL_EVALUATOR
        assert metrica["map_50_95"] == pytest.approx(0.325636)

    def test_el_punto_de_partida_ya_esta_medido_en_la_t4(
        self, script: ModuleType, clases: list[str]
    ) -> None:
        """Lo que sustituye al centinela, y por qué el centinela no saltó."""
        registro = ExperimentRegistry(get_project_paths().results_registry, clases)
        filas = [
            fila
            for fila in registro.read_rows("quality_metrics")
            if fila["experiment_id"] == STARTING_BASELINE_ID
            and fila["runtime"] == "pytorch"
            and fila["split"] == "val"
            and fila.get("evaluator") == CANONICAL_EVALUATOR
        ]
        if not filas:
            pytest.skip("este repositorio no trae la fila del baseline 'n'")

        dispositivos = {fila["device_id"] for fila in filas}
        assert "colab_t4_gpu" in dispositivos, (
            "falta la fila de T4 del punto de partida: sin ella la línea "
            "'control - partida' del piloto vuelve a mezclar dispositivos"
        )

        # Sin desempate no hay ambigüedad, porque las dos filas dicen lo mismo.
        metrica = script.leer_metrica_del_registro(registro, STARTING_BASELINE_ID, "val")
        assert metrica["map_50_95"] == pytest.approx(0.325636)

        # Con desempate sale la de la T4, que es la que el piloto necesita.
        de_la_t4 = script.leer_metrica_del_registro(
            registro, STARTING_BASELINE_ID, "val", preferred_device_id="colab_t4_gpu"
        )
        assert de_la_t4["device_id"] == "colab_t4_gpu"
        assert de_la_t4["evaluator"] == CANONICAL_EVALUATOR
        assert de_la_t4["map_50_95"] == pytest.approx(metrica["map_50_95"], abs=1e-6)
