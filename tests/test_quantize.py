"""Tests de la materialización INT8 y de la comprobación de que PTQ y QAT siguen el mismo camino."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm_edge.export import quantize
from tfm_edge.export.quantize import (
    Int8Config,
    MaterializationMismatchError,
    QuantizationError,
    QuantizationResult,
)


def _resultado(
    config: Int8Config,
    calibracion: str,
    source_kind: str = "baseline",
    runtime: str = "onnxruntime",
    num_imagenes: int = 300,
) -> QuantizationResult:
    """Construye un resultado sintético para probar la comparación."""
    return QuantizationResult(
        runtime=runtime,
        status="ok",
        source_kind=source_kind,
        config=config.to_dict(),
        config_hash=config.config_hash,
        calibration_hash=calibracion,
        num_calibration_images=num_imagenes,
        source_path=f"/ruta/{source_kind}.onnx",
    )


class TestInt8Config:
    """La configuración es lo que hace comparables dos materializaciones."""

    def test_el_hash_es_estable(self) -> None:
        """Dos configuraciones iguales dan el mismo hash, siempre."""
        assert Int8Config().config_hash == Int8Config().config_hash

    def test_el_hash_no_depende_del_orden_de_construccion(self) -> None:
        uno = Int8Config(per_channel=True, calibration_size=128)
        otro = Int8Config(calibration_size=128, per_channel=True)
        assert uno.config_hash == otro.config_hash

    def test_cambiar_un_parametro_cambia_el_hash(self) -> None:
        """Si no cambiara, dos caminos distintos parecerían el mismo."""
        base = Int8Config()
        for cambio in (
            {"per_channel": not base.per_channel},
            {"calibration_size": base.calibration_size + 1},
            {"symmetric_weights": not base.symmetric_weights},
            {"calibration_method": "Entropy"},
            {"excluded_nodes": ("/model.22/cv2.0/Conv",)},
            # Cambia qué se cuantiza, así que no puede compartir hash.
            {"onnx_exclude_output_quantization": ()},
        ):
            assert Int8Config(**cambio).config_hash != base.config_hash, cambio

    def test_la_salida_de_concat_va_excluida_por_defecto(self) -> None:
        """Sin ella el modelo INT8 de ONNX Runtime no detecta nada."""
        assert "Concat" in Int8Config().onnx_exclude_output_quantization

    def test_la_configuracion_es_inmutable(self) -> None:
        config = Int8Config()
        with pytest.raises(Exception):
            config.per_channel = False  # type: ignore[misc]


class TestHashDeCalibracion:
    """Los rangos dependen de qué imágenes calibran, y en qué orden."""

    def test_el_mismo_conjunto_da_el_mismo_hash(self) -> None:
        """El caso base."""
        imagenes = ["a.jpg", "b.jpg", "c.jpg"]
        assert quantize.calibration_hash(imagenes) == quantize.calibration_hash(list(imagenes))

    def test_un_conjunto_distinto_da_otro_hash(self) -> None:
        """Calibrar con otras imágenes es calibrar de otra manera."""
        assert quantize.calibration_hash(["a.jpg"]) != quantize.calibration_hash(["b.jpg"])

    def test_otro_orden_da_otro_hash(self) -> None:
        assert quantize.calibration_hash(["a.jpg", "b.jpg"]) != quantize.calibration_hash(
            ["b.jpg", "a.jpg"]
        )

    def test_solo_cuenta_el_nombre_no_la_ruta(self) -> None:
        """Así el hash sobrevive a mover el dataset de sitio."""
        assert quantize.calibration_hash([Path("/aqui/a.jpg")]) == quantize.calibration_hash(
            [Path("/alli/a.jpg")]
        )

    def test_un_conjunto_vacio_se_rechaza(self) -> None:
        """La cuantización estática necesita imágenes reales."""
        with pytest.raises(QuantizationError, match="vacío"):
            quantize.calibration_hash([])


class TestD018:
    """Comprobación de que PTQ y QAT comparten el camino de materialización."""

    def test_el_mismo_camino_se_acepta(self) -> None:
        """Misma configuración, misma calibración, distinto punto de partida."""
        config = Int8Config()
        informe = quantize.assert_same_materialization_path(
            _resultado(config, "cal123", "baseline"),
            _resultado(config, "cal123", "qat"),
        )
        assert informe["same_path"]
        assert informe["config_hash"] == config.config_hash

    def test_parametros_distintos_se_rechazan(self) -> None:
        with pytest.raises(MaterializationMismatchError, match="parámetros distintos"):
            quantize.assert_same_materialization_path(
                _resultado(Int8Config(), "cal123", "baseline"),
                _resultado(Int8Config(per_channel=False), "cal123", "qat"),
            )

    def test_calibraciones_distintas_se_rechazan(self) -> None:
        """Calibrar con conjuntos distintos produce rangos distintos."""
        config = Int8Config()
        with pytest.raises(MaterializationMismatchError, match="calibración"):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "baseline"),
                _resultado(config, "cal999", "qat"),
            )

    def test_runtimes_distintos_se_rechazan(self) -> None:
        """Solo tiene sentido comparar dentro del mismo runtime."""
        config = Int8Config()
        with pytest.raises(MaterializationMismatchError, match="runtimes distintos"):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "baseline", runtime="onnxruntime"),
                _resultado(config, "cal123", "qat", runtime="openvino"),
            )

    def test_comparar_una_rama_consigo_misma_se_rechaza(self) -> None:
        config = Int8Config()
        with pytest.raises(
            MaterializationMismatchError, match="la comparación admitida es exactamente"
        ):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "baseline"),
                _resultado(config, "cal123", "baseline"),
            )

    def test_la_rama_combinada_no_se_cuela_como_comparacion_de_d018(self) -> None:
        """Con tres procedencias, 'que difieran' dejó de bastar."""
        config = Int8Config()
        with pytest.raises(
            MaterializationMismatchError, match="la comparación admitida es exactamente"
        ):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "pruned"),
                _resultado(config, "cal123", "qat"),
            )


class TestParejaAutorizada:
    """La pareja (pruned, qat) solo pasa si se autoriza explícitamente."""

    def test_la_pareja_de_la_poda_se_acepta_si_se_autoriza(self) -> None:
        config = Int8Config()
        informe = quantize.assert_same_materialization_path(
            _resultado(config, "cal123", "pruned"),
            _resultado(config, "cal123", "qat"),
            admitted_pair=quantize.PRUNED_QAT_SOURCE_KINDS,
        )
        assert informe["same_path"]

    def test_autorizada_la_poda_la_pareja_del_baseline_ya_no_pasa(self) -> None:
        config = Int8Config()
        with pytest.raises(
            MaterializationMismatchError, match="la comparación admitida es exactamente"
        ):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "baseline"),
                _resultado(config, "cal123", "qat"),
                admitted_pair=quantize.PRUNED_QAT_SOURCE_KINDS,
            )

    def test_autorizar_la_poda_no_relaja_los_hashes(self) -> None:
        """Cambiar la pareja no toca las comprobaciones de camino."""
        config = Int8Config()
        with pytest.raises(MaterializationMismatchError, match="calibración"):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "pruned"),
                _resultado(config, "cal999", "qat"),
                admitted_pair=quantize.PRUNED_QAT_SOURCE_KINDS,
            )

    @pytest.mark.parametrize(
        "pareja",
        [frozenset({"baseline", "pruned"}), frozenset({"qat"}), frozenset({"qat", "otra"})],
        ids=["baseline-pruned", "solo-qat", "inventada"],
    )
    def test_una_pareja_no_admisible_se_rechaza(self, pareja: frozenset[str]) -> None:
        """El parámetro no puede convertir la guarda en un sí a todo."""
        config = Int8Config()
        with pytest.raises(QuantizationError, match="no reconocida"):
            quantize.assert_same_materialization_path(
                _resultado(config, "cal123", "baseline"),
                _resultado(config, "cal123", "pruned"),
                admitted_pair=pareja,
            )

    def test_la_pareja_por_defecto_sigue_siendo_la_de_d018(self) -> None:
        """Nada que no la pida recibe la pareja de la poda."""
        assert quantize.D018_SOURCE_KINDS == frozenset({"baseline", "qat"})
        assert quantize.PRUNED_QAT_SOURCE_KINDS != quantize.D018_SOURCE_KINDS
        assert quantize.PRUNED_QAT_SOURCE_KINDS <= set(quantize.VALID_SOURCE_KINDS)
        assert set(quantize.ADMISSIBLE_SOURCE_PAIRS) == {
            quantize.D018_SOURCE_KINDS,
            quantize.PRUNED_QAT_SOURCE_KINDS,
        }


class TestValidacionDeEntradas:
    """Lo que el punto de entrada único rechaza."""

    def test_un_runtime_sin_implementar_se_rechaza(self) -> None:
        """No se inventa un camino para un runtime que no lo tiene."""
        with pytest.raises(QuantizationError, match="sin materialización INT8"):
            quantize.materialize_int8("m.onnx", "o.onnx", ["a.jpg"], runtime="tensorrt")

    def test_un_source_kind_inventado_se_rechaza(self) -> None:
        """La procedencia tiene que quedar declarada, y la lista es cerrada."""
        with pytest.raises(QuantizationError, match="source_kind"):
            quantize.materialize_int8(
                "m.onnx", "o.onnx", ["a.jpg"], runtime="onnxruntime", source_kind="otra_cosa"
            )

    def test_las_procedencias_admitidas_son_las_tres_del_estudio(self) -> None:
        assert quantize.VALID_SOURCE_KINDS == ("baseline", "qat", "pruned")
        assert quantize.D018_SOURCE_KINDS <= set(quantize.VALID_SOURCE_KINDS)


# ===========================================================================
# Materialización real
# ===========================================================================

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")
pytest.importorskip("onnx", reason="requiere onnx")
pytest.importorskip("onnxruntime", reason="requiere onnxruntime")


@pytest.mark.requires_torch
@pytest.mark.slow
class TestMaterializacionReal:
    """Se cuantiza de verdad y se comprueba lo que sale."""

    @pytest.fixture(scope="class")
    def imagenes(self) -> list[Path]:
        """Devuelve imágenes reales de calibración, u omite los tests."""
        from tfm_edge.common.paths import get_project_paths

        directorio = get_project_paths().data / "resized_640" / "images"
        if not directorio.is_dir():
            pytest.skip("No hay dataset preparado")
        encontradas = sorted(directorio.glob("*.jp*g"))[:8]
        if len(encontradas) < 4:
            pytest.skip("No hay imágenes suficientes")
        return encontradas

    @pytest.fixture(scope="class")
    def onnx_fp32(self, tmp_path_factory) -> str:
        """Exporta un ONNX FP32 pequeño del que partir."""
        from tfm_edge.export import exporter
        from tfm_edge.models.detector import Detector

        detector = Detector.from_architecture("n", num_classes=17, input_resolution=320)
        destino = tmp_path_factory.mktemp("fp32")
        resultado = exporter.export_onnx(detector, destino, "m", num_check_samples=1)
        if not resultado.ok:
            pytest.skip(f"No se ha podido exportar: {resultado.notes}")
        return resultado.path

    def test_el_artefacto_int8_es_mas_pequeno(
        self, onnx_fp32: str, imagenes: list[Path], tmp_path: Path
    ) -> None:
        """Si no encogiera, la cuantización no habría llegado a los pesos."""
        config = Int8Config(calibration_size=len(imagenes))
        resultado = quantize.materialize_int8(
            onnx_fp32,
            tmp_path / "int8.onnx",
            imagenes,
            runtime="onnxruntime",
            config=config,
            input_resolution=320,
        )
        assert resultado.ok, resultado.notes
        assert resultado.size_reduction > 2.0, (
            f"solo {resultado.size_reduction}x menor: INT8 debería reducir unas "
            "cuatro veces respecto a FP32"
        )
        assert resultado.num_calibration_images == len(imagenes)
        assert resultado.config_hash == config.config_hash

    def test_las_dos_ramas_producen_el_mismo_hash_de_camino(
        self, onnx_fp32: str, imagenes: list[Path], tmp_path: Path
    ) -> None:
        config = Int8Config(calibration_size=len(imagenes))
        comun = {
            "calibration_images": imagenes,
            "runtime": "onnxruntime",
            "config": config,
            "input_resolution": 320,
        }
        ptq = quantize.materialize_int8(
            onnx_fp32, tmp_path / "ptq.onnx", source_kind="baseline", **comun
        )
        qat = quantize.materialize_int8(
            onnx_fp32, tmp_path / "qat.onnx", source_kind="qat", **comun
        )
        assert ptq.ok and qat.ok
        informe = quantize.assert_same_materialization_path(ptq, qat)
        assert informe["same_path"]

    def test_la_rama_podada_recorre_el_mismo_camino_y_lo_declara(
        self, onnx_fp32: str, imagenes: list[Path], tmp_path: Path
    ) -> None:
        """La rama del apartado 5.5 materializa por el camino único."""
        config = Int8Config(calibration_size=len(imagenes))
        comun = {
            "calibration_images": imagenes,
            "runtime": "onnxruntime",
            "config": config,
            "input_resolution": 320,
        }
        ptq = quantize.materialize_int8(
            onnx_fp32, tmp_path / "ptq.onnx", source_kind="baseline", **comun
        )
        podado = quantize.materialize_int8(
            onnx_fp32, tmp_path / "podado.onnx", source_kind="pruned", **comun
        )
        assert ptq.ok and podado.ok, podado.notes
        assert podado.source_kind == "pruned"
        assert podado.config_hash == ptq.config_hash
        assert podado.calibration_hash == ptq.calibration_hash

    def test_un_tipo_de_cuantizacion_invalido_falla_con_mensaje_util(
        self, onnx_fp32: str, imagenes: list[Path], tmp_path: Path
    ) -> None:
        """Construir el nombre del enum por concatenación daba 'QUint88'."""
        resultado = quantize.materialize_int8(
            onnx_fp32,
            tmp_path / "malo.onnx",
            imagenes,
            runtime="onnxruntime",
            config=Int8Config(activation_type="float8", calibration_size=len(imagenes)),
            input_resolution=320,
        )
        assert resultado.status == "fallido"
        assert "activation_type" in resultado.notes


class TestComprobacionFuncional:
    """Un artefacto que existe no es un artefacto que funciona."""

    @staticmethod
    def _motor_falso(salida):
        """Construye una fábrica de motores que devuelve una salida fija."""
        import torch

        class _Motor:
            def load(self, artefacto):
                self.artefacto = artefacto

            def preprocess(self, imagen):
                return torch.zeros(1, 3, 32, 32)

            def infer(self, tensor):
                return torch.from_numpy(salida)

            def unload(self):
                pass

        return lambda *_a, **_k: _Motor()

    def _comprobar(self, monkeypatch, tmp_path, salida):
        """Ejecuta la comprobación funcional contra un motor falso."""
        import cv2
        import numpy as np

        from tfm_edge.benchmark import backends

        monkeypatch.setattr(backends, "build_backend", self._motor_falso(salida))
        imagen = tmp_path / "a.png"
        cv2.imwrite(str(imagen), np.zeros((32, 32, 3), dtype=np.uint8))
        return quantize.assert_artifact_is_functional(
            tmp_path / "modelo.onnx", "onnxruntime", [imagen], input_resolution=32
        )

    def test_una_salida_sana_pasa(self, monkeypatch, tmp_path) -> None:
        import numpy as np

        salida = np.zeros((1, 21, 100), dtype=np.float32)
        salida[0, :4] = 300.0  # coordenadas, de rango grande
        salida[0, 4:] = np.linspace(0.0, 0.9, 17 * 100).reshape(17, 100)
        informe = self._comprobar(monkeypatch, tmp_path, salida)
        assert informe["passed"] is True
        assert informe["unique_scores"] > 1

    def test_las_puntuaciones_aplastadas_a_cero_se_cazan(self, monkeypatch, tmp_path) -> None:
        import numpy as np

        salida = np.zeros((1, 21, 100), dtype=np.float32)
        # Las coordenadas se mantienen, como en el fallo real, y el artefacto parece sano.
        salida[0, :4] = 300.0
        informe = self._comprobar(monkeypatch, tmp_path, salida)
        assert informe["passed"] is False
        assert informe["unique_scores"] == 1
        assert "NO detecta nada" in informe["reason"]

    def test_una_degradacion_fuerte_pero_legitima_no_se_confunde_con_el_fallo(
        self, monkeypatch, tmp_path
    ) -> None:
        import numpy as np

        salida = np.zeros((1, 21, 100), dtype=np.float32)
        salida[0, :4] = 300.0
        salida[0, 4:] = np.linspace(0.0, 0.0007, 17 * 100).reshape(17, 100)
        informe = self._comprobar(monkeypatch, tmp_path, salida)
        assert informe["passed"] is True, "una degradación fuerte no es una salida degenerada"
        assert informe["max_score"] < 0.05, "el caso tiene que estar por debajo del umbral viejo"

    def test_si_no_se_puede_comprobar_no_se_declara_fallido(self, monkeypatch, tmp_path) -> None:
        from tfm_edge.benchmark import backends

        def _revienta(*_a, **_k):
            raise RuntimeError("motor no disponible")

        monkeypatch.setattr(backends, "build_backend", _revienta)
        informe = quantize.assert_artifact_is_functional(
            tmp_path / "modelo.onnx", "onnxruntime", [tmp_path / "a.png"], input_resolution=32
        )
        assert informe["passed"] == ""
        assert "no se ha podido comprobar" in informe["reason"]

    def test_sin_imagenes_no_se_inventa_un_veredicto(self, tmp_path) -> None:
        """Sin datos con los que comprobar, el veredicto queda vacío."""
        informe = quantize.assert_artifact_is_functional(
            tmp_path / "modelo.onnx", "onnxruntime", []
        )
        assert informe["passed"] == ""
