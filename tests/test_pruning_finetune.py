"""Tests del reajuste posterior a la poda (apartado 9.3.5)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")
pytest.importorskip("torch_pruning", reason="requiere torch-pruning")

import torch  # noqa: E402

from tfm_edge.common.config import PruningConfig  # noqa: E402
from tfm_edge.compression.pruning import finetune as finetune_module  # noqa: E402
from tfm_edge.compression.pruning.dependency import PruningError  # noqa: E402

RESOLUCION = 320


@pytest.fixture(scope="module")
def detector_podado():
    """Un yolov8n realmente podado, con semilla fijada."""
    from tfm_edge.common.seed import set_global_seed
    from tfm_edge.compression.pruning.structured import prune_model
    from tfm_edge.models.detector import Detector

    set_global_seed(42)
    detector = Detector.from_architecture("n", num_classes=17, input_resolution=RESOLUCION)
    # min_channels=4, como en test_pruning_shapes.py, porque sobre un yolov8n sin entrenar el
    # reparto global al 30 % deja model.2.cv0.conv por debajo de 8.
    config = PruningConfig(ratio=0.3, criterion="l1", distribution="global", min_channels=4)
    generador = torch.Generator().manual_seed(0)
    entrada = torch.randn(1, 3, RESOLUCION, RESOLUCION, generator=generador)
    podado, resultado = prune_model(detector, config, example_input=entrada)
    assert resultado.parameters_after < resultado.parameters_before
    return podado


class TestUnaMetricaAusenteNoEsCero:
    """Lo que no se ha medido no puede aparecer como un cero medido."""

    def test_sin_evaluador_las_dos_metricas_quedan_vacias(self) -> None:
        """Sin evaluador no hay número, y la celda queda vacía."""
        informe = finetune_module.FinetuneReport()
        assert informe.metric_before == ""
        assert informe.metric_after == ""
        assert informe.recovery == ""

    def test_la_recuperacion_necesita_las_dos_medidas(self) -> None:
        """Una resta con un operando ausente no vale cero."""
        informe = finetune_module.FinetuneReport(metrics_before={"map_50_95": 0.1})
        assert informe.recovery == ""
        informe = finetune_module.FinetuneReport(metrics_after={"map_50_95": 0.3})
        assert informe.recovery == ""

    def test_la_fraccion_recuperada_sin_baseline_queda_vacia(self) -> None:
        """Sin saber de dónde se cayó no se puede decir cuánto se ha subido."""
        informe = finetune_module.FinetuneReport(
            metrics_before={"map_50_95": 0.10}, metrics_after={"map_50_95": 0.30}
        )
        assert informe.recovery == pytest.approx(0.20)
        assert informe.recovery_fraction == ""

    def test_la_fraccion_recuperada_no_divide_por_cero(self) -> None:
        """Si la poda no degradó nada, el cociente no está definido."""
        informe = finetune_module.FinetuneReport(
            metrics_baseline={"map_50_95": 0.30},
            metrics_before={"map_50_95": 0.30},
            metrics_after={"map_50_95": 0.31},
        )
        assert informe.recovery_fraction == ""

    def test_la_fraccion_recuperada_se_calcula_cuando_hay_las_tres(self) -> None:
        """Con baseline, podado y reajustado sí hay cociente."""
        informe = finetune_module.FinetuneReport(
            metrics_baseline={"map_50_95": 0.40},
            metrics_before={"map_50_95": 0.10},
            metrics_after={"map_50_95": 0.34},
        )
        assert informe.recovery_fraction == pytest.approx(0.8, abs=1e-9)

    def test_el_volcado_conserva_las_ausencias(self) -> None:
        """Al serializar, lo ausente sigue siendo cadena vacía y no cero."""
        datos = finetune_module.FinetuneReport().to_dict()
        for clave in ("metric_before", "metric_after", "recovery", "recovery_fraction"):
            assert datos[clave] == "", f"'{clave}' debería quedar vacío, no {datos[clave]!r}"


class TestLaPodaSobreviveAlDisco:
    """El fichero guardado tiene que seguir podado; el capítulo 5.3 va de eso."""

    def test_el_checkpoint_previo_conserva_los_parametros(
        self, detector_podado, tmp_path: Path
    ) -> None:
        """Guardar y recargar no devuelve los canales originales."""
        esperados = detector_podado.count_parameters()
        _, informe = finetune_module.finetune_pruned(
            detector_podado,
            PruningConfig(finetune_epochs=0),
            data_config=tmp_path / "no_se_usa.yaml",
            output_dir=tmp_path,
        )
        ruta = Path(informe.checkpoint_before)
        assert ruta.is_file()

        from tfm_edge.models.detector import Detector

        recargado = Detector.load(ruta, input_resolution=RESOLUCION)
        assert recargado.count_parameters() == esperados

    def test_un_checkpoint_que_perdio_la_poda_se_detecta(
        self, detector_podado, tmp_path: Path
    ) -> None:
        from tfm_edge.common.seed import set_global_seed
        from tfm_edge.models.detector import Detector

        set_global_seed(42)
        sin_podar = Detector.from_architecture("n", num_classes=17, input_resolution=RESOLUCION)
        ruta = sin_podar.save(tmp_path / "sin_podar.pt")

        with pytest.raises(PruningError, match="parámetros"):
            finetune_module._verificar_checkpoint_conserva_la_poda(
                ruta, detector_podado.count_parameters(), RESOLUCION
            )


class TestLaConfiguracionManda:
    """Las épocas y la tasa de aprendizaje salen del YAML que se serializa."""

    def test_no_se_pueden_redefinir_por_la_puerta_de_atras(
        self, detector_podado, tmp_path: Path
    ) -> None:
        """Si el valor efectivo no fuera el registrado, el registro mentiría."""
        for clave, valor in (("epochs", 5), ("learning_rate", 0.1)):
            with pytest.raises(PruningError, match=clave):
                finetune_module.finetune_pruned(
                    detector_podado,
                    PruningConfig(finetune_epochs=1),
                    data_config=tmp_path / "no_se_usa.yaml",
                    output_dir=tmp_path / clave,
                    trainer_kwargs={clave: valor},
                )

    def test_cero_epocas_no_entrena_y_lo_dice(self, detector_podado, tmp_path: Path) -> None:
        """No reajustar es legítimo, pero el informe tiene que distinguirlo."""
        _, informe = finetune_module.finetune_pruned(
            detector_podado,
            PruningConfig(finetune_epochs=0),
            data_config=tmp_path / "no_se_usa.yaml",
            output_dir=tmp_path,
        )
        assert informe.checkpoint_after == ""
        assert informe.metrics_after == {}
        assert informe.metric_after == ""
        assert "no se ha reajustado" in informe.notes

    def test_el_reajuste_parcial_desactivado_no_devuelve_metrica(self, tmp_path: Path) -> None:
        reajuste = finetune_module.make_partial_finetune_callback(
            PruningConfig(finetune_epochs_per_step=0),
            data_config=tmp_path / "no_se_usa.yaml",
            output_dir=tmp_path,
        )
        assert reajuste(None, 1) is None  # type: ignore[arg-type]


class TestSeMideSobreFicherosYNoSobreLaMemoria:
    """Las dos cifras salen de checkpoints en disco, y de dos distintos."""

    def test_el_evaluador_recibe_ficheros_existentes(
        self, detector_podado, tmp_path: Path, monkeypatch
    ) -> None:
        vistos: list[Path] = []

        def evaluador(checkpoint: Path) -> dict[str, float]:
            assert checkpoint.is_file(), f"'{checkpoint}' no existe en disco"
            vistos.append(checkpoint)
            return {"map_50_95": 0.1 * len(vistos)}

        entrenador_falso = _construir_entrenador_falso(monkeypatch, detector_podado)

        _, informe = finetune_module.finetune_pruned(
            detector_podado,
            PruningConfig(finetune_epochs=1),
            data_config=tmp_path / "no_se_usa.yaml",
            output_dir=tmp_path,
            evaluator=evaluador,
        )
        assert entrenador_falso["llamado"], "el reajuste no llegó a entrenar"
        assert len(vistos) == 2, "hay que medir dos veces: antes y después"
        assert vistos[0] != vistos[1], (
            "el 'antes' y el 'después' se han medido sobre el MISMO fichero; "
            "entonces no miden cosas distintas"
        )
        assert informe.metric_before == pytest.approx(0.1)
        assert informe.metric_after == pytest.approx(0.2)
        assert informe.recovery == pytest.approx(0.1)


class TestElVolcadoCrudoNoSePisaConMenosDeLoQueHabia:
    """Un volcado sin métricas no puede sustituir a uno que las tiene."""

    @staticmethod
    def _cargar_script():
        """Carga scripts/13_prune.py como módulo, sin ejecutar su ``main``."""
        import importlib.util

        from tfm_edge.common.paths import get_project_paths

        ruta = get_project_paths().root / "scripts" / "13_prune.py"
        especificacion = importlib.util.spec_from_file_location("prune_script", ruta)
        assert especificacion is not None and especificacion.loader is not None
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)
        return modulo

    @staticmethod
    def _volcado(destino: Path, con_metricas: bool) -> Path:
        """Escribe un volcado crudo con o sin métricas medidas."""
        import json

        metricas = {"map_50_95": 0.376214} if con_metricas else {}
        destino.write_text(
            json.dumps({"finetune": {"metrics_before": metricas, "metrics_after": {}}}),
            encoding="utf-8",
        )
        return destino

    def test_una_ejecucion_sin_medir_no_pisa_una_medida(self, tmp_path: Path) -> None:
        """Es el caso exacto que destruyó la evidencia."""
        script = self._cargar_script()
        anterior = self._volcado(tmp_path / "exp.json", con_metricas=True)
        informe = finetune_module.FinetuneReport(epochs=0, learning_rate=1e-3)
        assert script._degradaria_la_evidencia(anterior, informe) is True

    def test_una_ejecucion_medida_si_puede_sustituir(self, tmp_path: Path) -> None:
        """Repetir la campaña entera es legítimo y no debe bloquearse."""
        script = self._cargar_script()
        anterior = self._volcado(tmp_path / "exp.json", con_metricas=True)
        informe = finetune_module.FinetuneReport(
            epochs=1, learning_rate=1e-3, metrics_before={"map_50_95": 0.1}
        )
        assert script._degradaria_la_evidencia(anterior, informe) is False

    def test_sin_volcado_previo_no_hay_nada_que_proteger(self, tmp_path: Path) -> None:
        """La primera ejecución de una variante escribe siempre."""
        script = self._cargar_script()
        informe = finetune_module.FinetuneReport(epochs=0, learning_rate=1e-3)
        assert script._degradaria_la_evidencia(tmp_path / "no_existe.json", informe) is False

    def test_un_volcado_anterior_sin_metricas_no_bloquea(self, tmp_path: Path) -> None:
        """Proteger un fichero que tampoco medía nada solo estorbaría."""
        script = self._cargar_script()
        anterior = self._volcado(tmp_path / "exp.json", con_metricas=False)
        informe = finetune_module.FinetuneReport(epochs=0, learning_rate=1e-3)
        assert script._degradaria_la_evidencia(anterior, informe) is False

    def test_un_volcado_ilegible_no_es_evidencia_que_proteger(self, tmp_path: Path) -> None:
        """Y no debe hacer fallar la campaña."""
        script = self._cargar_script()
        roto = tmp_path / "exp.json"
        roto.write_text("{esto no es JSON", encoding="utf-8")
        informe = finetune_module.FinetuneReport(epochs=0, learning_rate=1e-3)
        assert script._degradaria_la_evidencia(roto, informe) is False


def _construir_entrenador_falso(monkeypatch, detector_podado) -> dict:
    """Sustituye el entrenador por un doble que deja un ``best.pt`` de verdad."""
    from tfm_edge.training.trainer import TrainingCost, TrainingResult

    estado = {"llamado": False}

    class TrainerFalso:
        def __init__(self, detector, data_config, output_dir, epochs, learning_rate, **kwargs):
            self.detector = detector
            self.output_dir = Path(output_dir)
            self.epochs = epochs
            self.learning_rate = learning_rate

        def train(self, run_name: str = "train") -> TrainingResult:
            estado["llamado"] = True
            destino = self.output_dir / run_name / "weights" / "best.pt"
            self.detector.save(destino)
            return TrainingResult(
                best_checkpoint=destino,
                last_checkpoint=destino,
                cost=TrainingCost(epochs_completed=self.epochs),
                run_dir=self.output_dir / run_name,
            )

    import tfm_edge.training.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "Trainer", TrainerFalso)
    return estado
