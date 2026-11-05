"""Evaluar en otra máquina no cambia las rutas registradas del experimento."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm_edge.common.registry import EXPERIMENTS_COLUMNS, ExperimentRegistry
from tfm_edge.common.registry_lookup import preserve_registered_paths

CLASES = ("person", "helmet")
EXPERIMENT_ID = "s_baseline_default_a1b2c3d4"


def _registro(tmp_path: Path) -> ExperimentRegistry:
    """Un registro vacío bajo el temporal."""
    return ExperimentRegistry(registry_dir=tmp_path / "registry", class_names=CLASES)


def _fila(**campos: str) -> dict[str, str]:
    """Una fila de ``experiments`` con lo justo para escribirla."""
    fila = {columna: "" for columna in EXPERIMENTS_COLUMNS}
    fila.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "model_family": "yolov8",
            "model_size": "s",
            "technique": "baseline",
            "technique_variant": "default",
            "config_hash": "a1b2c3d4",
            "precision": "fp32",
            "status": "ok",
            "config_path": "models/baselines/s/args.yaml",
            "weights_path": "models/baselines/s/weights/best.pt",
        }
    )
    fila.update(campos)
    return fila


class TestQueLosDosEvaluadoresLaUsan:
    """Una guarda escrita y no cableada no protege de nada."""

    @pytest.mark.parametrize("script", ["05_evaluate_baselines.py", "06_evaluate_coco.py"])
    def test_el_script_envuelve_su_fila(self, script: str) -> None:
        from tfm_edge.common.paths import get_project_paths

        fuente = (get_project_paths().root / "scripts" / script).read_text(encoding="utf-8")

        assert (
            "preserve_registered_paths(" in fuente
        ), f"{script} reinscribe la fila del experimento sin conservar sus rutas."


class TestQueEvaluarNoReapunta:
    """Lo que la fila ya declara sobre dónde vive su checkpoint se respeta."""

    def test_una_ruta_de_colab_no_pisa_la_relativa(self, tmp_path: Path) -> None:
        """El caso real, y el único que distingue esta guarda de no tenerla."""
        registro = _registro(tmp_path)
        registro.append_experiment(_fila(), on_duplicate="overwrite")

        conservada = preserve_registered_paths(
            registro,
            _fila(
                weights_path="/content/drive/MyDrive/tfm/best.pt",
                config_path="/content/drive/MyDrive/tfm/args.yaml",
            ),
        )

        assert conservada["weights_path"] == "models/baselines/s/weights/best.pt"
        assert conservada["config_path"] == "models/baselines/s/args.yaml"

    def test_el_resto_de_la_fila_sigue_completandose(self, tmp_path: Path) -> None:
        registro = _registro(tmp_path)
        registro.append_experiment(_fila(), on_duplicate="overwrite")

        conservada = preserve_registered_paths(
            registro, _fila(effective_params="11166560", training_time_s="11664")
        )

        assert conservada["effective_params"] == "11166560"
        assert conservada["training_time_s"] == "11664"

    def test_una_celda_vacia_si_se_rellena(self, tmp_path: Path) -> None:
        registro = _registro(tmp_path)
        registro.append_experiment(_fila(weights_path=""), on_duplicate="overwrite")

        conservada = preserve_registered_paths(
            registro, _fila(weights_path="models/baselines/s/weights/best.pt")
        )

        assert conservada["weights_path"] == "models/baselines/s/weights/best.pt"

    def test_un_experimento_que_no_estaba_se_escribe_tal_cual(self, tmp_path: Path) -> None:
        """La primera inscripción no tiene nada que conservar."""
        registro = _registro(tmp_path)
        registro.append_experiment(_fila(experiment_id="n_baseline_default_99999999"))

        conservada = preserve_registered_paths(registro, _fila())

        assert conservada["weights_path"] == "models/baselines/s/weights/best.pt"

    def test_se_avisa_de_la_ruta_que_no_se_escribe(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Conservar en silencio sería el mismo defecto con el signo cambiado."""
        registro = _registro(tmp_path)
        registro.append_experiment(_fila(), on_duplicate="overwrite")

        with caplog.at_level("WARNING"):
            preserve_registered_paths(registro, _fila(weights_path="/content/drive/best.pt"))

        assert "Se conserva" in caplog.text
        assert "/content/drive/best.pt" in caplog.text

    def test_las_columnas_las_dice_el_esquema(self, tmp_path: Path) -> None:
        """La guarda no lleva su propia lista de columnas de ruta."""
        from tfm_edge.common.registry import TABLE_PATH_COLUMNS

        registro = _registro(tmp_path)
        registro.append_experiment(_fila(), on_duplicate="overwrite")
        inventadas = {columna: "/otra/maquina/x" for columna in TABLE_PATH_COLUMNS["experiments"]}

        conservada = preserve_registered_paths(registro, _fila(**inventadas))

        for columna in TABLE_PATH_COLUMNS["experiments"]:
            assert conservada[columna] != "/otra/maquina/x", (
                f"'{columna}' es columna de ruta del esquema y la guarda la dejó " f"reapuntar."
            )
