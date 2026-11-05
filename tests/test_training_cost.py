"""Tests de la lectura del coste de entrenamiento."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_edge.training.cost import (
    COST_FIELD_DEVICE,
    COST_FIELD_EPOCHS,
    COST_FIELD_GPU_NAME,
    COST_FIELD_SECONDS,
    REGISTRY_COLUMNS,
    cost_dict_to_registry,
    locate_run_file,
    locate_training_cost,
    read_training_cost,
    seconds_per_epoch,
)
from tfm_edge.training.trainer import TrainingCost


class TestLasClavesSonLasDelFicheroReal:
    """Los nombres que se leen tienen que existir en lo que se escribe."""

    def test_las_cuatro_claves_son_campos_de_training_cost(self) -> None:
        campos = set(TrainingCost.__dataclass_fields__)
        leidas = {
            COST_FIELD_EPOCHS,
            COST_FIELD_SECONDS,
            COST_FIELD_DEVICE,
            COST_FIELD_GPU_NAME,
        }
        ausentes = leidas - campos
        assert not ausentes, (
            f"Se leen claves que TrainingCost no escribe: {sorted(ausentes)}. "
            "Una clave que no existe no da error: deja la columna vacía."
        )

    def test_un_volcado_real_se_lee_entero(self, tmp_path: Path) -> None:
        coste = TrainingCost(
            wall_time_s=11652.85,
            gpu_hours=3.2369,
            device="cuda:0",
            gpu_name="Tesla T4",
            epochs_completed=100,
            execution_context="google_colab+container",
        )
        (tmp_path / "training_cost.json").write_text(json.dumps(coste.to_dict()))
        (tmp_path / "best.pt").touch()

        fila = read_training_cost(tmp_path / "best.pt")

        assert fila["training_epochs"] == 100
        assert fila["training_time_s"] == pytest.approx(11652.85)
        assert fila["training_device"] == "Tesla T4"


class TestLoQueNoSeSabeQuedaVacio:
    """Ninguna columna se rellena con un cero de relleno."""

    def test_sin_fichero_las_tres_columnas_quedan_vacias(self, tmp_path: Path) -> None:
        fila = read_training_cost(tmp_path / "best.pt")
        assert set(fila) == set(REGISTRY_COLUMNS)
        assert all(valor == "" for valor in fila.values())

    def test_un_json_ilegible_no_revienta_pero_no_inventa(self, tmp_path: Path) -> None:
        (tmp_path / "training_cost.json").write_text("{esto no es json")
        (tmp_path / "best.pt").touch()

        fila = read_training_cost(tmp_path / "best.pt")

        assert all(valor == "" for valor in fila.values())

    def test_cero_epocas_no_es_una_medida(self) -> None:
        """Un volcado de un entrenamiento que no llegó a correr no es un dato."""
        fila = cost_dict_to_registry(TrainingCost().to_dict())
        assert fila["training_epochs"] == ""
        assert fila["training_time_s"] == ""

    def test_un_coste_a_medias_conserva_lo_que_si_hay(self) -> None:
        fila = cost_dict_to_registry({"epochs_completed": 30, "gpu_name": "Tesla T4"})
        assert fila["training_epochs"] == 30
        assert fila["training_time_s"] == ""
        assert fila["training_device"] == "Tesla T4"


class TestElDispositivoQueSeRegistraIdentificaAlgo:
    """Se registra el nombre de la GPU, no ``cuda:0``."""

    def test_se_prefiere_el_nombre_de_la_gpu(self) -> None:
        fila = cost_dict_to_registry({"device": "cuda:0", "gpu_name": "Tesla T4"})
        assert fila["training_device"] == "Tesla T4"

    def test_sin_nombre_de_gpu_se_cae_al_dispositivo(self) -> None:
        fila = cost_dict_to_registry({"device": "cpu", "gpu_name": ""})
        assert fila["training_device"] == "cpu"


class TestLasDosDisposicionesDeCheckpoint:
    """Los pesos llegan de dos sitios y las dos rutas tienen que valer."""

    def test_junto_al_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "training_cost.json").write_text("{}")
        (tmp_path / "best.pt").touch()
        assert locate_training_cost(tmp_path / "best.pt") == tmp_path / "training_cost.json"

    def test_un_nivel_por_encima_como_lo_deja_ultralytics(self, tmp_path: Path) -> None:
        pesos = tmp_path / "weights"
        pesos.mkdir()
        (pesos / "best.pt").touch()
        (tmp_path / "training_cost.json").write_text("{}")
        assert locate_training_cost(pesos / "best.pt") == tmp_path / "training_cost.json"

    def test_el_localizador_sirve_para_cualquier_fichero_de_la_ejecucion(
        self, tmp_path: Path
    ) -> None:
        pesos = tmp_path / "weights"
        pesos.mkdir()
        (pesos / "best.pt").touch()
        (tmp_path / "args.yaml").touch()
        assert locate_run_file(pesos / "best.pt", "args.yaml") == tmp_path / "args.yaml"
        assert locate_run_file(pesos / "best.pt", "no_existe.yaml") is None


class TestElCostePorEpocaSaleDeUnaMedida:
    """Dimensionar un reajuste sin cifra medida sería inventarla."""

    def test_se_calcula_cuando_estan_los_dos_datos(self) -> None:
        fila = {"training_epochs": 100, "training_time_s": 11652.85}
        assert seconds_per_epoch(fila) == pytest.approx(116.5285)

    @pytest.mark.parametrize(
        "fila",
        [
            {"training_epochs": "", "training_time_s": 11652.85},
            {"training_epochs": 100, "training_time_s": ""},
            {"training_epochs": 0, "training_time_s": 10.0},
            {},
        ],
    )
    def test_sin_los_dos_datos_devuelve_none_y_no_una_estimacion(self, fila: dict) -> None:
        assert seconds_per_epoch(fila) is None
