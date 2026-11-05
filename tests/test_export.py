"""Tests de la exportación y de la comprobación de equivalencia numérica."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tfm_edge.export import equivalence
from tfm_edge.export.equivalence import EquivalenceError


def _salida(cajas: float = 300.0, puntuacion: float = 0.5, num_cajas: int = 100) -> np.ndarray:
    """Construye una salida sintética con la forma de YOLOv8 en exportación."""
    salida = np.zeros((1, 21, num_cajas), dtype=np.float32)
    salida[:, :4, :] = cajas
    salida[:, 4:, :] = puntuacion
    return salida


class TestSeparacionDeMagnitudes:
    """Coordenadas y puntuaciones no se pueden comparar con el mismo criterio."""

    def test_dos_salidas_identicas_son_equivalentes(self) -> None:
        """El caso base."""
        ref = _salida()
        informe = equivalence.compare_outputs(ref, ref.copy(), input_resolution=640)
        assert informe.equivalent
        assert informe.box_max_abs_diff == 0.0
        assert informe.score_max_abs_diff == 0.0

    def test_ruido_de_coma_flotante_en_coordenadas_pasa(self) -> None:
        ref = _salida()
        candidato = ref.copy()
        candidato[:, :4, :] += 1.3e-3
        informe = equivalence.compare_outputs(ref, candidato, input_resolution=640)
        assert informe.equivalent, informe.failures

    def test_un_desplazamiento_real_de_caja_no_pasa(self) -> None:
        """Cinco píxeles sí mueven una detección."""
        ref = _salida()
        candidato = ref.copy()
        candidato[:, 0, 0] += 5.0
        informe = equivalence.compare_outputs(ref, candidato, input_resolution=640)
        assert not informe.equivalent
        assert any("coordenadas" in f for f in informe.failures)

    def test_una_puntuacion_movida_una_centesima_no_pasa(self) -> None:
        ref = _salida()
        candidato = ref.copy()
        candidato[:, 5, 0] += 0.01
        informe = equivalence.compare_outputs(ref, candidato, input_resolution=640)
        assert not informe.equivalent
        assert any("puntuaciones" in f for f in informe.failures)

    def test_la_misma_diferencia_absoluta_se_juzga_distinto(self) -> None:
        """El test que justifica el diseño del módulo."""
        ref = _salida()
        en_cajas = ref.copy()
        en_cajas[:, 0, 0] += 1e-3
        en_puntuaciones = ref.copy()
        en_puntuaciones[:, 5, 0] += 1e-5
        assert equivalence.compare_outputs(ref, en_cajas, input_resolution=640).equivalent
        assert equivalence.compare_outputs(ref, en_puntuaciones, input_resolution=640).equivalent

        grave = ref.copy()
        grave[:, 0, 0] += 5.0
        assert not equivalence.compare_outputs(ref, grave, input_resolution=640).equivalent

    def test_la_tolerancia_de_cajas_escala_con_la_resolucion(self) -> None:
        """A 320 px, la misma diferencia absoluta pesa el doble."""
        ref = _salida()
        candidato = ref.copy()
        candidato[:, 0, 0] += 0.05
        a640 = equivalence.compare_outputs(ref, candidato, input_resolution=640)
        a320 = equivalence.compare_outputs(ref, candidato, input_resolution=320)
        assert a320.box_max_rel_diff == pytest.approx(2 * a640.box_max_rel_diff)


class TestDeteccionDeArtefactosIncorrectos:
    """Lo que no puede pasar la comprobación."""

    def test_formas_distintas_se_detectan(self) -> None:
        """Un artefacto con otra forma de salida no representa al mismo modelo."""
        informe = equivalence.compare_outputs(_salida(), _salida(num_cajas=50))
        assert not informe.equivalent
        assert not informe.shape_matches

    def test_una_forma_inesperada_se_detecta(self) -> None:
        """La salida tiene que ser (lote, 4 + clases, cajas)."""
        plana = np.zeros((1, 100), dtype=np.float32)
        informe = equivalence.compare_outputs(plana, plana.copy())
        assert not informe.equivalent
        assert any("forma inesperada" in f for f in informe.failures)


class TestComparacionDeVariasMuestras:
    """Una sola entrada podría no ejercitar todas las ramas del grafo."""

    def test_se_agrega_el_peor_caso(self) -> None:
        ref = [_salida(), _salida()]
        mal = [_salida(), _salida()]
        mal[1][:, 0, 0] += 5.0
        informe = equivalence.compare_many(ref, mal, input_resolution=640)
        assert not informe.equivalent
        assert informe.num_samples == 2
        assert informe.box_max_abs_diff == pytest.approx(5.0)

    def test_longitudes_distintas_se_rechazan(self) -> None:
        """Tienen que ser las mismas entradas en el mismo orden."""
        with pytest.raises(EquivalenceError, match="referencias"):
            equivalence.compare_many([_salida()], [_salida(), _salida()])

    def test_sin_muestras_se_rechaza(self) -> None:
        """Comparar cero muestras no demuestra nada."""
        with pytest.raises(EquivalenceError, match="ninguna muestra"):
            equivalence.compare_many([], [])


class TestComparacionDeDetecciones:
    """Lo que de verdad importa es si cambian las detecciones."""

    def test_recuentos_iguales_coinciden(self) -> None:
        """Dos salidas con la misma puntuación dan las mismas cajas."""
        ref = _salida(puntuacion=0.9)
        resultado = equivalence.compare_detections(ref, ref.copy(), conf_threshold=0.25)
        assert resultado["match"]
        assert resultado["informative"]

    def test_un_recuento_de_cero_se_declara_NO_informativo(self) -> None:
        """Cero cajas en ambas no demuestra equivalencia, solo que nada pasó."""
        ref = _salida(puntuacion=0.01)
        resultado = equivalence.compare_detections(ref, ref.copy(), conf_threshold=0.25)
        assert resultado["match"]
        assert not resultado["informative"]


# ===========================================================================
# Exportación (necesita los runtimes)
# ===========================================================================

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.export import exporter  # noqa: E402
from tfm_edge.export.exporter import ExportError  # noqa: E402


class TestDisponibilidadDeFormatos:
    """La ausencia de un runtime se registra."""

    def test_un_formato_desconocido_se_rechaza(self) -> None:
        """Solo se contemplan los formatos declarados."""
        with pytest.raises(ExportError, match="no contemplado"):
            exporter.check_format_available("formato_inventado")

    def test_todos_los_formatos_contemplados_responden(self) -> None:
        """Cada formato dice si está y, si no, por qué."""
        for formato in exporter.EXPORT_FORMATS:
            hay, motivo = exporter.check_format_available(formato)
            assert isinstance(hay, bool)
            assert hay or motivo, f"'{formato}' no está y no dice por qué"

    def test_un_formato_ausente_produce_no_disponible_con_motivo(self) -> None:
        resultado = exporter.export_unavailable("tensorrt")
        assert resultado.status == "no_disponible"
        assert resultado.notes
        assert not resultado.ok

    def test_tflite_declara_que_su_latencia_no_representa(self) -> None:
        resultado = exporter.export_unavailable("tflite")
        assert not resultado.latency_representative
        assert "ARM" in resultado.notes


@pytest.mark.requires_torch
class TestExportacionReal:
    """Se exporta de verdad y se comprueba que los números coinciden."""

    @pytest.fixture(scope="class")
    def detector(self):
        """Construye un yolov8n sin pesos, que no requiere red."""
        from tfm_edge.models.detector import Detector

        return Detector.from_architecture("n", num_classes=17, input_resolution=320)

    def test_el_modo_exportacion_se_restaura(self, detector) -> None:
        """Dejar el modelo en modo exportación rompería un entrenamiento posterior."""
        from ultralytics.nn.modules import Detect

        cabezas = [m for m in detector.model.modules() if isinstance(m, Detect)]
        antes = [m.export for m in cabezas]
        with detector.export_mode():
            assert all(m.export for m in cabezas)
        assert [m.export for m in cabezas] == antes

    def test_el_modo_exportacion_se_restaura_tambien_con_excepcion(self, detector) -> None:
        """Es lo que evita dejar el modelo en un estado que nadie esperaría."""
        from ultralytics.nn.modules import Detect

        cabezas = [m for m in detector.model.modules() if isinstance(m, Detect)]
        with pytest.raises(RuntimeError):
            with detector.export_mode():
                raise RuntimeError("fallo simulado")
        assert not any(m.export for m in cabezas)

    def test_el_modo_exportacion_cambia_la_salida(self, detector) -> None:
        salida = detector.reference_output()
        assert salida.ndim == 3
        # (lote, 4 coordenadas + 17 clases, anclas)
        assert salida.shape[1] == 21

    def test_onnx_se_exporta_y_es_equivalente(self, detector, tmp_path) -> None:
        """La prueba de punta a punta del formato de referencia."""
        pytest.importorskip("onnx", reason="requiere onnx")
        pytest.importorskip("onnxruntime", reason="requiere onnxruntime")

        resultado = exporter.export_onnx(detector, tmp_path, "prueba", num_check_samples=2)
        assert resultado.status == "ok", resultado.notes
        assert resultado.equivalence["equivalent"]
        assert Path(resultado.path).is_file()
        # El opset registrado tiene que ser el del artefacto, no el pedido.
        import onnx

        modelo = onnx.load(resultado.path)
        real = {op.domain or "ai.onnx": op.version for op in modelo.opset_import}
        assert real["ai.onnx"] == resultado.metadata["opset"]

    def test_openvino_se_guarda_en_FP32(self, detector, tmp_path) -> None:
        """ov.save_model comprime a FP16 por defecto."""
        pytest.importorskip("onnx", reason="requiere onnx")
        pytest.importorskip("openvino", reason="requiere openvino")

        onnx_res = exporter.export_onnx(detector, tmp_path, "prueba", num_check_samples=1)
        assert onnx_res.ok
        resultado = exporter.export_openvino(
            detector, tmp_path, "prueba", onnx_path=onnx_res.path, num_check_samples=1
        )
        assert resultado.status == "ok", resultado.notes
        assert resultado.metadata["compress_to_fp16"] is False
        assert resultado.metadata["weights_dtype"] == "float32"

    def test_verify_openvino_precision_detecta_fp16(self, detector, tmp_path) -> None:
        """La comprobación caza el artefacto comprimido, que es el defecto."""
        pytest.importorskip("onnx", reason="requiere onnx")
        ov = pytest.importorskip("openvino", reason="requiere openvino")

        onnx_res = exporter.export_onnx(detector, tmp_path, "prueba", num_check_samples=1)
        modelo = ov.convert_model(onnx_res.path)
        comprimido = tmp_path / "fp16" / "m.xml"
        comprimido.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(modelo, str(comprimido), compress_to_fp16=True)

        precision = exporter.verify_openvino_precision(comprimido)
        assert not precision["is_fp32"]
        assert precision["dominant_dtype"] == "float16"
