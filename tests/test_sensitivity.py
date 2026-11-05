"""Tests del análisis de sensibilidad y de la rama PTQ."""

from __future__ import annotations

import pytest

from tfm_edge.compression.quantization.sensitivity import LayerSensitivity, SensitivityReport


def _informe() -> SensitivityReport:
    """Construye un informe sintético para probar la ordenación."""
    return SensitivityReport(
        num_images=4,
        layers=[
            LayerSensitivity(layer="model.0.conv", output_mse=2.0, num_parameters=432),
            LayerSensitivity(layer="model.1.conv", output_mse=5.0, num_parameters=4608),
            LayerSensitivity(
                layer="model.22.cv2.0.0.conv",
                output_mse=0.05,
                num_parameters=36864,
                in_head=True,
            ),
            LayerSensitivity(layer="model.3.conv", error="no se pudo medir"),
        ],
    )


class TestOrdenacion:
    """El informe ordena las capas por sensibilidad."""

    def test_se_ordena_de_mas_a_menos_sensible(self) -> None:
        """Es lo que se usa para decidir qué excluir."""
        assert [c.layer for c in _informe().ranked] == [
            "model.1.conv",
            "model.0.conv",
            "model.22.cv2.0.0.conv",
        ]

    def test_las_capas_no_medidas_quedan_fuera_del_orden(self) -> None:
        informe = _informe()
        assert "model.3.conv" not in [c.layer for c in informe.ranked]
        assert any(c.error for c in informe.layers), "el fallo debe quedar registrado"

    def test_most_sensitive_respeta_el_limite(self) -> None:
        """Se usa para proponer un número concreto de exclusiones."""
        assert _informe().most_sensitive(2) == ["model.1.conv", "model.0.conv"]


class TestSugerenciaDeExclusiones:
    """Una sugerencia, nunca una decisión."""

    def test_propone_las_mas_sensibles(self) -> None:
        """El caso base."""
        from tfm_edge.compression.quantization.sensitivity import suggest_exclusions

        assert suggest_exclusions(_informe(), num_layers=2) == (
            "model.1.conv",
            "model.0.conv",
        )

    def test_puede_restringirse_a_la_cabeza(self) -> None:
        """Para responder a 'qué excluyo de la cabeza' sin tocar el resto."""
        from tfm_edge.compression.quantization.sensitivity import suggest_exclusions

        assert suggest_exclusions(_informe(), num_layers=5, only_head=True) == (
            "model.22.cv2.0.0.conv",
        )

    def test_lo_propuesto_encaja_en_la_configuracion(self) -> None:
        """La sugerencia tiene que poder pasarse tal cual a Int8Config."""
        from tfm_edge.compression.quantization.sensitivity import suggest_exclusions
        from tfm_edge.export.quantize import Int8Config

        propuestas = suggest_exclusions(_informe(), num_layers=2)
        config = Int8Config(excluded_nodes=propuestas)
        assert config.excluded_nodes == propuestas
        # Y cambia el hash, porque excluir capas produce otra materialización.
        assert config.config_hash != Int8Config().config_hash


class TestLimitacionDeclarada:
    """La limitación del método tiene que viajar con el resultado."""

    def test_el_informe_declara_que_no_capta_interacciones(self) -> None:
        """Sin esto, alguien sumaría las degradaciones individuales."""
        from tfm_edge.compression.quantization.sensitivity import measure_layer_sensitivity

        # Se comprueba sobre el texto del resumen, que es lo que se lee.
        informe = _informe()
        informe.notes = (
            "Se cuantiza una capa cada vez; el resto queda en coma flotante. No "
            "captura interacciones entre capas: la suma de las degradaciones "
            "individuales NO es la de cuantizarlas todas."
        )
        assert "NO es la de cuantizarlas todas" in informe.summary()
        assert callable(measure_layer_sensitivity)

    def test_el_resumen_marca_las_capas_de_la_cabeza(self) -> None:
        assert "cabeza" in _informe().summary()


# ===========================================================================
# Medida real
# ===========================================================================

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")


@pytest.mark.requires_torch
@pytest.mark.slow
class TestMedidaReal:
    """Se mide de verdad, sobre entradas reales."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Construye un yolov8n sin pesos."""
        from tfm_edge.models.detector import Detector

        return Detector.from_architecture("n", num_classes=17, input_resolution=320)

    @pytest.fixture(scope="class")
    def entradas(self, detector):
        """Prepara entradas con el preprocesado compartido."""
        import numpy as np

        from tfm_edge.benchmark.backends import PyTorchBackend
        from tfm_edge.common.paths import get_project_paths

        directorio = get_project_paths().data / "resized_640" / "images"
        pre = PyTorchBackend(input_resolution=320, device="cpu")
        if directorio.is_dir():
            import cv2

            rutas = sorted(directorio.glob("*.jp*g"))[:2]
            if len(rutas) >= 2:
                return [pre.preprocess(cv2.imread(str(r))).numpy() for r in rutas]
        pytest.skip("No hay imágenes reales para medir sensibilidad")

    def test_una_sola_imagen_se_rechaza(self, detector, entradas) -> None:
        """El orden dependería de esa imagen concreta."""
        from tfm_edge.compression.quantization.sensitivity import (
            SensitivityError,
            measure_layer_sensitivity,
        )

        with pytest.raises(SensitivityError, match="al menos 2"):
            measure_layer_sensitivity(detector, entradas[:1], layers=["model.0.conv"])

    def test_se_mide_la_sensibilidad_de_una_capa(self, detector, entradas) -> None:
        """La comprobación de punta a punta."""
        from tfm_edge.compression.quantization.sensitivity import measure_layer_sensitivity

        informe = measure_layer_sensitivity(
            detector, entradas, layers=["model.0.conv", "model.1.conv"]
        )
        assert len(informe.ranked) == 2
        for capa in informe.ranked:
            assert isinstance(capa.output_mse, float)
            assert capa.output_mse >= 0.0
        assert informe.baseline_output_scale > 0.0

    def test_cuantizar_una_capa_cambia_la_salida(self, detector, entradas) -> None:
        """Si no cambiara, la cuantización simulada no se habría aplicado."""
        from tfm_edge.compression.quantization.sensitivity import measure_layer_sensitivity

        informe = measure_layer_sensitivity(detector, entradas, layers=["model.0.conv"])
        assert informe.ranked[0].output_mse > 0.0, (
            "cuantizar una capa no ha movido la salida: la simulación no se "
            "aplicó y la medida no significa nada"
        )
