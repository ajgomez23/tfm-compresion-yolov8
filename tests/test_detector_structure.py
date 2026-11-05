"""Tests de guarda contra cambios en la estructura interna de Ultralytics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.models.detector import (  # noqa: E402
    ANCHORED_ULTRALYTICS_VERSION,
    BACKBONE_END_INDEX,
    EXPECTED_NUM_LAYERS,
    NECK_END_INDEX,
    Detector,
    DetectorError,
    installed_ultralytics_version,
)

pytestmark = pytest.mark.requires_torch


@pytest.fixture(scope="module")
def detector() -> Detector:
    """Construye un yolov8n sin descargar pesos."""
    return Detector.from_architecture(size="n", num_classes=17, pretrained=False)


class TestVersionAnclada:
    """La versión instalada debe ser la anclada."""

    def test_la_version_instalada_es_la_anclada(self) -> None:
        """Avisa si se está usando otra versión de Ultralytics."""
        instalada = installed_ultralytics_version()
        assert instalada == ANCHORED_ULTRALYTICS_VERSION, (
            f"Ultralytics instalado: {instalada}, anclado: {ANCHORED_ULTRALYTICS_VERSION}. "
            "La estructura interna puede haber cambiado. Revisa "
            "src/tfm_edge/models/detector.py antes de continuar."
        )


class TestEstructuraDelModelo:
    """Los índices y nombres de los que depende la poda y la destilación."""

    def test_numero_de_capas(self, detector: Detector) -> None:
        """YOLOv8 tiene exactamente 23 capas en su secuencia principal."""
        assert len(detector.layers) == EXPECTED_NUM_LAYERS

    def test_la_cabeza_esta_en_el_ultimo_indice(self, detector: Detector) -> None:
        """La cabeza Detect es la última capa, en el índice 22."""
        from ultralytics.nn.modules.head import Detect

        cabeza = detector.get_head_modules()
        assert list(cabeza) == [f"model.{NECK_END_INDEX}"]
        assert isinstance(cabeza[f"model.{NECK_END_INDEX}"], Detect)

    def test_los_tres_bloques_cubren_todas_las_capas(self, detector: Detector) -> None:
        """Backbone, cuello y cabeza particionan la secuencia sin solaparse."""
        backbone = set(detector.get_backbone_modules())
        cuello = set(detector.get_neck_modules())
        cabeza = set(detector.get_head_modules())

        assert not (backbone & cuello)
        assert not (cuello & cabeza)
        assert not (backbone & cabeza)
        assert len(backbone | cuello | cabeza) == EXPECTED_NUM_LAYERS

    def test_tamanos_de_cada_bloque(self, detector: Detector) -> None:
        """Los tres bloques tienen el tamaño esperado."""
        assert len(detector.get_backbone_modules()) == BACKBONE_END_INDEX
        assert len(detector.get_neck_modules()) == NECK_END_INDEX - BACKBONE_END_INDEX
        assert len(detector.get_head_modules()) == EXPECTED_NUM_LAYERS - NECK_END_INDEX

    def test_la_cabeza_tiene_los_submodulos_esperados(self, detector: Detector) -> None:
        """Detect expone cv2, cv3 y dfl, que la poda necesita identificar."""
        cabeza = detector.get_head_modules()[f"model.{NECK_END_INDEX}"]
        submodulos = {nombre for nombre, _ in cabeza.named_children()}
        assert {"cv2", "cv3", "dfl"}.issubset(submodulos)

    def test_las_convoluciones_de_la_cabeza_se_identifican(self, detector: Detector) -> None:
        """Se pueden listar las convoluciones a proteger de la poda."""
        protegidas = detector.get_head_conv_names()
        assert protegidas
        assert all(n.startswith(f"model.{NECK_END_INDEX}.") for n in protegidas)
        # No pueden ser todas: si lo fueran, no quedaría nada que podar.
        assert len(protegidas) < len(detector.get_named_conv_layers())

    def test_hay_convoluciones_y_normalizaciones(self, detector: Detector) -> None:
        """El modelo tiene el número de capas convolucionales esperado."""
        convs = detector.get_named_conv_layers()
        assert len(convs) == 64, f"yolov8n debería tener 64 Conv2d y tiene {len(convs)}"


class TestInterfazDelDetector:
    """La interfaz estable que usa el resto del proyecto."""

    def test_pase_hacia_delante(self, detector: Detector) -> None:
        """El detector produce la salida esperada."""
        detector.eval()
        with torch.no_grad():
            salida = detector.forward(torch.randn(1, 3, 640, 640))
        tensor = salida[0] if isinstance(salida, (list, tuple)) else salida
        # 4 coordenadas + 17 clases = 21 canales, sobre 8400 posiciones a 640 px.
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 4 + 17

    def test_captura_de_caracteristicas_intermedias(self, detector: Detector) -> None:
        """Los hooks capturan activaciones y se retiran al salir del contexto."""
        detector.eval()
        capas = ["model.4", "model.6"]
        with detector.capture_features(capas) as features:
            with torch.no_grad():
                detector.forward(torch.randn(1, 3, 320, 320))
            assert set(features) == set(capas)
            assert all(isinstance(v, torch.Tensor) for v in features.values())

        # Tras salir del contexto no debe quedar ningún hook: dejarlos colgados
        # es una fuga de memoria que en un entrenamiento largo agota la GPU.
        for nombre in capas:
            assert not detector.get_module(nombre)._forward_hooks

    def test_los_hooks_se_retiran_aunque_haya_excepcion(self, detector: Detector) -> None:
        """Una excepción dentro del contexto no deja hooks colgados."""
        with pytest.raises(RuntimeError):
            with detector.capture_features(["model.4"]):
                raise RuntimeError("fallo simulado")
        assert not detector.get_module("model.4")._forward_hooks

    def test_modulo_inexistente(self, detector: Detector) -> None:
        """Pedir un módulo que no existe falla con un mensaje accionable."""
        with pytest.raises(DetectorError, match="No existe el módulo"):
            detector.get_module("model.999")

    def test_canales_por_convolucion(self, detector: Detector) -> None:
        """Se pueden leer los canales de cada convolución."""
        canales = detector.channels_per_conv()
        assert len(canales) == len(detector.get_named_conv_layers())
        assert all(isinstance(v, tuple) and len(v) == 2 for v in canales.values())

    def test_el_detector_nace_en_fp32(self, detector: Detector) -> None:
        """Un detector recién construido está en FP32."""
        assert detector.dtype == torch.float32
        detector.assert_float32()

    def test_assert_float32_detecta_media_precision(self) -> None:
        """La comprobación de precisión caza un modelo en FP16."""
        d = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        d._model = d._model.half()
        with pytest.raises(DetectorError, match="NO son float32"):
            d.assert_float32()


class TestGuardadoYCarga:
    """El checkpoint del proyecto se guarda y se recarga en FP32."""

    def test_ida_y_vuelta_en_fp32(self, detector: Detector, tmp_path) -> None:
        """Guardar y recargar conserva la precisión y la estructura."""
        destino = detector.save(tmp_path / "modelo.pt")
        assert destino.is_file()

        recargado = Detector.load(destino)
        recargado.assert_float32()
        assert recargado.count_parameters() == detector.count_parameters()
        assert recargado.channels_per_conv() == detector.channels_per_conv()

    def test_el_checkpoint_declara_su_precision(self, detector: Detector, tmp_path) -> None:
        """El metadato del checkpoint deja constancia de que es FP32."""
        destino = detector.save(tmp_path / "modelo.pt")
        contenido = torch.load(destino, map_location="cpu", weights_only=False)
        assert contenido["tfm_edge"]["saved_dtype"] == "float32"

    def test_guardar_convierte_a_fp32_un_modelo_en_media_precision(self, tmp_path) -> None:
        """Aunque el modelo esté en FP16, el checkpoint sale en FP32."""
        d = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        d._model = d._model.half()
        destino = d.save(tmp_path / "modelo.pt")
        Detector.load(destino).assert_float32()
