"""Tests del entrenamiento consciente de cuantización."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

import torch  # noqa: E402

from tfm_edge.compression.quantization import qat  # noqa: E402
from tfm_edge.compression.quantization.qat import QatConfig, QatError  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402

pytestmark = pytest.mark.requires_torch


@pytest.fixture(scope="module")
def detector() -> Detector:
    """Construye un yolov8n sin pesos, que no requiere red."""
    return Detector.from_architecture("n", num_classes=17, input_resolution=320)


class TestPreparacion:
    """Preparar para QAT no puede alterar el baseline ni fallar en silencio."""

    def test_se_insertan_observadores(self, detector: Detector) -> None:
        """Sin observadores, la variante 'QAT' sería un afinado normal."""
        _, informe = qat.prepare_for_qat(detector, check_gradient=False)
        assert informe.num_weight_observers > 0
        assert informe.num_activation_observers > 0

    def test_el_gradiente_fluye(self, detector: Detector) -> None:
        """El test que evita entrenar tres horas sin aprender nada."""
        _, informe = qat.prepare_for_qat(detector, check_gradient=True)
        assert informe.gradient_flows

    def test_el_detector_original_no_se_toca(self, detector: Detector) -> None:
        antes = sum(p.numel() for p in detector.model.parameters())
        qat.prepare_for_qat(detector, check_gradient=False)
        assert sum(p.numel() for p in detector.model.parameters()) == antes
        assert not any("Fake" in type(m).__name__ for m in detector.model.modules())

    def test_excluir_la_cabeza_reduce_los_observadores(self, detector: Detector) -> None:
        _, completo = qat.prepare_for_qat(detector, check_gradient=False)
        _, sin_cabeza = qat.prepare_for_qat(
            detector, QatConfig(exclude_head=True), check_gradient=False
        )
        assert sin_cabeza.num_weight_observers < completo.num_weight_observers
        assert len(sin_cabeza.excluded_modules) == len(detector.get_head_conv_names())

    def test_un_motor_inexistente_se_rechaza(self, detector: Detector) -> None:
        with pytest.raises(QatError, match="motor de cuantización"):
            qat.prepare_for_qat(detector, QatConfig(qengine="inventado"))

    def test_una_ruta_excluida_inexistente_se_rechaza(self, detector: Detector) -> None:
        """Si no, se creería estar excluyendo algo que no se excluye."""
        with pytest.raises(QatError, match="no se ha podido excluir|No se ha podido excluir"):
            qat.prepare_for_qat(
                detector, QatConfig(excluded_modules=("model.99.noexiste",)), check_gradient=False
            )


class TestModoEntrenamientoYEvaluacion:
    """La cuantización simulada tiene que funcionar en los dos modos."""

    def test_el_paso_hacia_delante_funciona_entrenando(self, detector: Detector) -> None:
        """Es lo que hace viable QAT pese a que SiLU no tenga kernel cuantizado."""
        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        preparado.train()
        salida = preparado(torch.zeros(2, 3, 320, 320))
        assert salida is not None

    def test_el_paso_hacia_delante_funciona_evaluando(self, detector: Detector) -> None:
        """Hace falta para medir el mAP de la variante QAT."""
        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        preparado.eval()
        with torch.no_grad():
            salida = preparado(torch.zeros(1, 3, 320, 320))
        tensor = salida[0] if isinstance(salida, (list, tuple)) else salida
        assert tensor.shape[1] == 21  # 4 coordenadas + 17 clases


class TestRetiradaDeLaSimulacion:
    """QAT produce pesos FP32, no un modelo cuantizado."""

    def test_quedan_pesos_fp32(self, detector: Detector) -> None:
        """Tienen que recorrer el mismo camino que los del baseline."""
        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        limpio = qat.strip_fake_quantization(preparado)
        assert {p.dtype for p in limpio.parameters()} == {torch.float32}

    def test_el_modelo_limpio_sigue_funcionando(self, detector: Detector) -> None:
        """El paso hacia delante funciona."""
        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        limpio = qat.strip_fake_quantization(preparado).eval()
        with torch.no_grad():
            salida = limpio(torch.zeros(1, 3, 320, 320))
        tensor = salida[0] if isinstance(salida, (list, tuple)) else salida
        assert tensor.shape[1] == 21

    def test_no_queda_ni_un_modulo_de_torch_ao(self, detector: Detector) -> None:
        """Desactivar no es quitar, y lo que rompe la exportación es que estén."""
        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        limpio = qat.strip_fake_quantization(preparado)
        restos = sorted(
            {
                type(modulo).__name__
                for modulo in limpio.modules()
                if type(modulo).__module__.startswith("torch.ao")
            }
        )
        assert restos == [], f"sobreviven módulos de torch.ao: {restos}"

    def test_limpiar_un_modelo_sin_simulacion_se_rechaza(self, detector: Detector) -> None:
        """Si no, un afinado normal pasaría por una variante 'QAT'."""
        with pytest.raises(QatError, match="no tiene cuantización simulada"):
            qat.strip_fake_quantization(detector.model)


class TestQueElModeloDespojadoSeExporta:
    """El modelo sin simulación se puede exportar con ``torch.onnx.export``."""

    @pytest.mark.slow
    def test_la_exportacion_a_onnx_funciona(self, detector: Detector, tmp_path) -> None:
        from tfm_edge.compression.quantization.finetune import wrap_prepared_model
        from tfm_edge.export import exporter

        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        limpio = wrap_prepared_model(qat.strip_fake_quantization(preparado), detector)

        resultado = exporter.export_onnx(limpio, tmp_path, "qat_despojado", num_check_samples=1)
        assert resultado.ok, f"la exportación del modelo despojado ha fallado: {resultado.notes}"

    @pytest.mark.slow
    def test_el_grafo_es_el_mismo_que_el_del_modelo_sin_preparar(
        self, detector: Detector, tmp_path
    ) -> None:
        import collections

        import onnx

        from tfm_edge.compression.quantization.finetune import wrap_prepared_model
        from tfm_edge.export import exporter

        def operadores(ruta: str) -> collections.Counter:
            """Cuenta los tipos de operador de un ONNX."""
            modelo = onnx.load(ruta, load_external_data=False)
            return collections.Counter(nodo.op_type for nodo in modelo.graph.node)

        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        limpio = wrap_prepared_model(qat.strip_fake_quantization(preparado), detector)

        despojado = exporter.export_onnx(limpio, tmp_path, "despojado", num_check_samples=1)
        referencia = exporter.export_onnx(detector, tmp_path, "referencia", num_check_samples=1)
        assert despojado.ok and referencia.ok

        assert operadores(despojado.path) == operadores(referencia.path)


class TestCongelacion:
    """Congelar algo que no existe no puede parecer que funcionó."""

    def test_congelar_observadores_devuelve_cuantos(self, detector: Detector) -> None:
        """El recuento permite comprobar que la operación tuvo efecto."""
        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        assert qat.freeze_observers(preparado) > 0

    def test_congelar_bn_avisa_cuando_no_hay_nada_que_congelar(
        self, detector: Detector, caplog
    ) -> None:
        """En YOLOv8 sin fusionar, conv y BN son módulos separados."""
        import logging

        preparado, _ = qat.prepare_for_qat(detector, check_gradient=False)
        with caplog.at_level(logging.WARNING):
            congelados = qat.freeze_batchnorm_stats(preparado)
        assert congelados == 0
        assert "no tiene efecto" in caplog.text
