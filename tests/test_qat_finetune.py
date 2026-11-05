"""Tests del camino completo de afinado QAT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.compression.quantization.finetune import (  # noqa: E402
    QatFinetuneReport,
    finetune_qat,
    make_qat_epoch_callbacks,
    observers_have_been_exercised,
    wrap_prepared_model,
)
from tfm_edge.compression.quantization.qat import (  # noqa: E402
    QatConfig,
    QatError,
    prepare_for_qat,
    strip_fake_quantization,
)
from tfm_edge.models.detector import Detector  # noqa: E402
from tfm_edge.models.surgery import unfreeze_all  # noqa: E402
from tfm_edge.training.trainer import Trainer, TrainerError  # noqa: E402

pytestmark = pytest.mark.requires_torch


@pytest.fixture(scope="module")
def detector() -> Detector:
    """Construye un yolov8n de prueba."""
    return Detector.from_architecture(size="n", num_classes=17, pretrained=False)


@pytest.fixture(scope="module")
def preparado(detector: Detector) -> Any:
    """Prepara el detector para QAT una sola vez."""
    modelo, _ = prepare_for_qat(detector, QatConfig())
    return modelo


class TestElGradienteVieneDesactivadoEnLosPesosDelEstudio:
    """El checkpoint del estudio es la EMA, y la EMA llega congelada."""

    def test_se_reactivan_los_parametros_congelados(self, detector: Detector) -> None:
        import copy

        modelo = copy.deepcopy(detector.model)
        for parametro in modelo.parameters():
            parametro.requires_grad_(False)

        reactivados = unfreeze_all(modelo)

        assert reactivados > 0
        assert all(p.requires_grad for p in modelo.parameters() if p.dtype.is_floating_point)

    def test_no_toca_lo_que_ya_estaba_activo(self, detector: Detector) -> None:
        import copy

        modelo = copy.deepcopy(detector.model)
        unfreeze_all(modelo)
        assert unfreeze_all(modelo) == 0

    def test_preparar_un_modelo_congelado_no_revienta(self, detector: Detector) -> None:
        """El fallo exacto que daba el baseline real."""
        import copy

        congelado = Detector(
            model=copy.deepcopy(detector.model),
            size=detector.size,
            num_classes=detector.num_classes,
            input_resolution=detector.input_resolution,
        )
        for parametro in congelado.model.parameters():
            parametro.requires_grad_(False)

        _, informe = prepare_for_qat(congelado, QatConfig())

        assert informe.gradient_flows is True


class TestPrepararNoModificaElModeloDeEntrada:
    """La comprobación de gradiente no puede dejar rastro."""

    def test_las_estadisticas_de_batchnorm_no_se_mueven(self, detector: Detector) -> None:
        antes = {
            nombre: tensor.clone()
            for nombre, tensor in detector.model.state_dict().items()
            if "running_" in nombre
        }
        assert antes, "El modelo no tiene estadísticas de BatchNorm que comprobar"

        prepare_for_qat(detector, QatConfig())

        despues = detector.model.state_dict()
        movidas = [n for n, t in antes.items() if not torch.equal(t, despues[n])]
        assert not movidas, (
            f"prepare_for_qat ha movido {len(movidas)} estadísticas de BatchNorm "
            "del modelo de entrada. Un paso hacia delante en modo entrenamiento "
            "las actualiza, y aquí la entrada es ruido."
        )


class TestRevertirDevuelveElModeloDePartida:
    """Es la afirmación por la que no se mide 'el simulado antes de afinar'."""

    def test_los_pesos_vuelven_identicos(self, detector: Detector, preparado: Any) -> None:
        limpio = strip_fake_quantization(preparado)

        originales = detector.model.state_dict()
        revertidos = limpio.state_dict()

        maxima = 0.0
        comparados = 0
        for clave, tensor in originales.items():
            if clave not in revertidos or not torch.is_floating_point(tensor):
                continue
            comparados += 1
            maxima = max(maxima, (tensor - revertidos[clave]).abs().max().item())

        assert comparados > 0
        assert maxima == 0.0, (
            f"Revertir la simulación no devuelve los pesos de partida: "
            f"diferencia máxima {maxima:.3e}."
        )

    def test_la_salida_es_la_misma(self, detector: Detector, preparado: Any) -> None:
        limpio = strip_fake_quantization(preparado)
        generador = torch.Generator().manual_seed(0)
        entrada = torch.randn(
            1, 3, detector.input_resolution, detector.input_resolution, generator=generador
        )

        with torch.no_grad():
            detector.model.eval()
            limpio.eval()
            esperada = detector.model(entrada)
            obtenida = limpio(entrada)

        a = esperada[0] if isinstance(esperada, (list, tuple)) else esperada
        b = obtenida[0] if isinstance(obtenida, (list, tuple)) else obtenida
        assert (a - b).abs().max().item() == 0.0

    def test_el_numero_de_parametros_no_cambia(self, detector: Detector, preparado: Any) -> None:
        limpio = wrap_prepared_model(strip_fake_quantization(preparado), detector)
        assert limpio.count_parameters() == detector.count_parameters()


class TestUnQatSinObservadoresEjercitadosEsUnAfinadoNormal:
    """El modo de fallo cuyo único síntoma sería una etiqueta falsa."""

    def test_recien_preparado_no_han_visto_nada(self, detector: Detector) -> None:
        modelo, _ = prepare_for_qat(detector, QatConfig())
        assert observers_have_been_exercised(modelo) is False

    def test_tras_un_paso_hacia_delante_si(self, detector: Detector) -> None:
        modelo, _ = prepare_for_qat(detector, QatConfig())
        modelo.train()
        with torch.no_grad():
            modelo(torch.randn(1, 3, detector.input_resolution, detector.input_resolution))
        assert observers_have_been_exercised(modelo) is True

    def test_un_modelo_sin_observadores_no_cuela(self, detector: Detector) -> None:
        assert observers_have_been_exercised(detector.model) is False


class TestCongelarLaBatchNormSinFusionarEsUnError:
    """Pedirlo y que no ocurra nada sería fingir que la opción tiene efecto."""

    def test_falla_al_construir_los_callbacks(self) -> None:
        with pytest.raises(QatError, match="ConvBn2d|fusionar"):
            make_qat_epoch_callbacks(QatConfig(freeze_bn_epoch=5), QatFinetuneReport())

    def test_sin_congelaciones_no_hay_callbacks(self) -> None:
        assert make_qat_epoch_callbacks(QatConfig(), QatFinetuneReport()) == []

    def test_congelar_observadores_registra_un_callback(self) -> None:
        callbacks = make_qat_epoch_callbacks(
            QatConfig(freeze_observers_epoch=2), QatFinetuneReport()
        )
        assert [evento for evento, _ in callbacks] == ["on_train_epoch_start"]


class TestElCallbackCongelaEnLaEpocaQueToca:
    """El callback congela en la época prevista y anota lo que ha ocurrido."""

    class _EntrenadorFalso:
        """Lo mínimo que el callback mira del entrenador de Ultralytics."""

        def __init__(self, modelo: Any, epoca: int) -> None:
            self.model = modelo
            self.epoch = epoca

    def test_antes_de_la_epoca_no_congela(self, detector: Detector, preparado: Any) -> None:
        informe = QatFinetuneReport()
        (_, funcion), *_ = make_qat_epoch_callbacks(QatConfig(freeze_observers_epoch=3), informe)

        funcion(self._EntrenadorFalso(preparado, epoca=1))

        assert informe.frozen == {}

    def test_al_llegar_congela_y_lo_anota(self, detector: Detector, preparado: Any) -> None:
        informe = QatFinetuneReport()
        (_, funcion), *_ = make_qat_epoch_callbacks(QatConfig(freeze_observers_epoch=3), informe)

        funcion(self._EntrenadorFalso(preparado, epoca=3))

        assert informe.frozen["observers_epoch"] == 3
        assert informe.frozen["observers_frozen"] > 0

    def test_no_congela_dos_veces(self, detector: Detector, preparado: Any) -> None:
        informe = QatFinetuneReport()
        (_, funcion), *_ = make_qat_epoch_callbacks(QatConfig(freeze_observers_epoch=3), informe)

        funcion(self._EntrenadorFalso(preparado, epoca=3))
        funcion(self._EntrenadorFalso(preparado, epoca=7))

        assert informe.frozen["observers_epoch"] == 3


class TestLosEventosDeCallbackSeValidan:
    """Un evento de callback inexistente se rechaza al registrarlo."""

    def test_un_evento_inexistente_es_un_error(self, detector: Detector, tmp_path: Path) -> None:
        with pytest.raises(TrainerError, match="no conoce"):
            Trainer(
                detector=detector,
                data_config=tmp_path / "data.yaml",
                output_dir=tmp_path,
                callbacks=[("on_epoch_start_que_no_existe", lambda _t: None)],
            )

    def test_un_evento_real_se_acepta(self, detector: Detector, tmp_path: Path) -> None:
        entrenador = Trainer(
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
            callbacks=[("on_train_epoch_start", lambda _t: None)],
        )
        assert len(entrenador.callbacks) == 1


class TestElInformeNoInventaMetricas:
    """Sin medida, la celda va vacía; nunca a cero."""

    def test_sin_metricas_las_derivadas_quedan_vacias(self) -> None:
        informe = QatFinetuneReport()
        assert informe.metric_baseline == ""
        assert informe.metric_after == ""
        assert informe.fp32_delta == ""

    def test_con_solo_el_baseline_la_diferencia_sigue_vacia(self) -> None:
        informe = QatFinetuneReport(metrics_baseline={"map_50_95": 0.3762})
        assert informe.metric_baseline == pytest.approx(0.3762)
        assert informe.fp32_delta == ""

    def test_con_las_dos_se_calcula(self) -> None:
        informe = QatFinetuneReport(
            metrics_baseline={"map_50_95": 0.3762},
            metrics_after={"map_50_95": 0.3700},
        )
        assert informe.fp32_delta == pytest.approx(-0.0062)

    def test_el_volcado_lleva_las_derivadas(self) -> None:
        informe = QatFinetuneReport(
            metrics_baseline={"map_50_95": 0.3762},
            metrics_after={"map_50_95": 0.3700},
        )
        assert informe.to_dict()["fp32_delta"] == pytest.approx(-0.0062)


class TestLaConfiguracionManda:
    """Las épocas y la tasa de aprendizaje salen del YAML."""

    def test_trainer_kwargs_no_puede_pisar_las_epocas(
        self, detector: Detector, tmp_path: Path
    ) -> None:
        from tfm_edge.common.config import QuantizationConfig

        with pytest.raises(QatError, match="epochs"):
            finetune_qat(
                detector,
                QuantizationConfig(mode="qat"),
                data_config=tmp_path / "data.yaml",
                output_dir=tmp_path,
                trainer_kwargs={"epochs": 3},
            )

    def test_con_cero_epocas_la_metrica_de_despues_queda_ausente(
        self, detector: Detector, tmp_path: Path
    ) -> None:
        from tfm_edge.common.config import QuantizationConfig

        configuracion = QuantizationConfig(mode="qat")
        # qat_epochs tiene gt=0 en el esquema, asi que el caso "no se afina" se
        # construye por copia, igual que hace el script de poda.
        sin_afinar = configuracion.model_copy(update={"qat_epochs": 0})

        devuelto, informe = finetune_qat(
            detector,
            sin_afinar,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
        )

        assert informe.metric_after == ""
        assert "qat_epochs=0" in informe.notes
        assert devuelto.count_parameters() == detector.count_parameters()
