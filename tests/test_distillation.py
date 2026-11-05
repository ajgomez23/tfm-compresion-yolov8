"""Tests de las piezas de la destilación (9.4), sin necesitar pesos entrenados."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
import torch.nn as nn  # noqa: E402

from tfm_edge.common.config import DistillationConfig  # noqa: E402
from tfm_edge.compression.distillation.adapters import AdapterError, FeatureAdapters  # noqa: E402
from tfm_edge.compression.distillation.feature_hooks import (  # noqa: E402
    FeatureCaptureError,
    FeatureCollector,
)
from tfm_edge.compression.distillation.losses import (  # noqa: E402
    DistillationLoss,
    DistillationLossError,
    HeadGeometry,
    box_mask_grid,
    build_mask,
    class_kl_bernoulli,
    dfl_kl,
    feature_mse,
    ground_truth_box_mask,
    head_geometry,
    masked_mean,
    split_head_output,
)
from tfm_edge.compression.distillation.trainer import (  # noqa: E402
    ADAPTER_CHANGE_TOLERANCE,
    DistillationResult,
)
from tfm_edge.models.detector import drop_forward_hooks  # noqa: E402

pytestmark = [pytest.mark.requires_torch]

GEOMETRIA = HeadGeometry(num_classes=17, reg_max=16, strides=(8, 16, 32))
RESOLUCION = 320
#: Anclajes a 320 px: 40² + 20² + 10².
ANCLAJES = 1600 + 400 + 100


def _lote(cajas: list[list[float]], indices: list[int], lote: int = 2) -> dict:
    """Construye un lote mínimo en formato Ultralytics."""
    return {
        "img": torch.zeros(lote, 3, RESOLUCION, RESOLUCION),
        "bboxes": torch.tensor(cajas, dtype=torch.float32).reshape(-1, 4),
        "cls": torch.zeros(len(indices), 1),
        "batch_idx": torch.tensor(indices, dtype=torch.float32),
    }


class TestMediaEnmascarada:
    """La normalización por la máscara, que es donde aparecen los ``nan``."""

    def test_mascara_vacia_da_cero_y_no_nan(self) -> None:
        """Un lote sin objetos existe, y ahí la máscara se queda vacía."""
        valores = torch.rand(2, ANCLAJES)
        vacia = torch.zeros(2, ANCLAJES, dtype=torch.bool)
        resultado = masked_mean(valores, vacia)
        assert float(resultado) == 0.0
        assert not torch.isnan(resultado)

    def test_control_la_media_es_la_de_los_marcados(self) -> None:
        valores = torch.tensor([[1.0, 3.0, 100.0]])
        mascara = torch.tensor([[True, True, False]])
        assert float(masked_mean(valores, mascara)) == pytest.approx(2.0)

    def test_la_media_no_depende_del_numero_de_anclajes_no_marcados(self) -> None:
        """Normalizar por el total en vez de por los marcados diluiría la señal."""
        valores = torch.cat([torch.full((1, 3), 5.0), torch.zeros(1, 997)], dim=1)
        mascara = torch.cat(
            [torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 997, dtype=torch.bool)], dim=1
        )
        assert float(masked_mean(valores, mascara)) == pytest.approx(5.0)


class TestDivergencias:
    """Las dos divergencias, cada una con su forma de valer cero."""

    def test_clase_identica_da_cero(self) -> None:
        """Sin diferencia entre maestro y alumno no hay nada que destilar."""
        logits = torch.randn(2, 17, 64)
        assert float(class_kl_bernoulli(logits, logits.clone(), 2.0).abs().max()) < 1e-5

    def test_clase_distinta_da_positivo(self) -> None:
        maestro = torch.randn(2, 17, 64)
        alumno = maestro + 1.5
        assert float(class_kl_bernoulli(maestro, alumno, 2.0).min()) > 0.0

    def test_la_kl_de_clase_es_de_bernoulli_y_no_invariante_al_desplazamiento(self) -> None:
        """El detalle que distingue esta pérdida de una copiada de clasificación."""
        maestro = torch.randn(1, 17, 8)
        desplazado = maestro + 3.0
        assert float(class_kl_bernoulli(maestro, desplazado, 2.0).min()) > 0.1

    def test_caja_identica_da_cero(self) -> None:
        """La KL categórica del DFL vale cero contra sí misma."""
        dist = torch.randn(2, 64, 32)
        assert float(dfl_kl(dist, dist.clone(), GEOMETRIA, 2.0).abs().max()) < 1e-5

    def test_caja_distinta_da_positivo(self) -> None:
        """Mutación del anterior."""
        maestro = torch.randn(2, 64, 32)
        assert float(dfl_kl(maestro, maestro * 0.3, GEOMETRIA, 2.0).min()) > 0.0

    def test_la_kl_de_caja_si_es_invariante_al_desplazamiento(self) -> None:
        maestro = torch.randn(1, 64, 8)
        desplazado = maestro + 5.0
        assert float(dfl_kl(maestro, desplazado, GEOMETRIA, 2.0).abs().max()) < 1e-4


class TestMascaras:
    """Que la máscara marque lo que dice marcar."""

    def test_una_caja_marca_su_zona_y_solo_su_zona(self) -> None:
        """Una caja en el cuadrante superior izquierdo no marca el resto."""
        lote = _lote([[0.25, 0.25, 0.5, 0.5]], [0], lote=1)
        mascara = box_mask_grid(lote, 4, 4)
        assert mascara.shape == (1, 4, 4)
        assert bool(mascara[0, 0, 0]) and bool(mascara[0, 1, 1])
        assert not bool(mascara[0, 3, 3])

    def test_lote_sin_cajas_no_marca_nada(self) -> None:
        """El caso que obliga al ``clamp`` de la media enmascarada."""
        lote = _lote([], [], lote=2)
        assert int(box_mask_grid(lote, 8, 8).sum()) == 0

    def test_los_anclajes_aplanados_son_los_de_los_tres_niveles(self) -> None:
        """El orden y el número tienen que coincidir con ``split_head_output``."""
        lote = _lote([[0.5, 0.5, 0.4, 0.4]], [0], lote=1)
        mascara = ground_truth_box_mask(lote, GEOMETRIA, RESOLUCION)
        assert mascara.shape == (1, ANCLAJES)

    def test_la_interseccion_esta_contenida_en_las_dos(self) -> None:
        lote = _lote([[0.5, 0.5, 0.6, 0.6]], [0], lote=1)
        logits = torch.randn(1, 17, ANCLAJES)
        cajas = build_mask("gt_boxes", lote, logits, GEOMETRIA, RESOLUCION, 0.05)
        maestro = build_mask("teacher_attention", lote, logits, GEOMETRIA, RESOLUCION, 0.05)
        interseccion = build_mask("gt_boxes_and_teacher", lote, logits, GEOMETRIA, RESOLUCION, 0.05)
        assert bool((interseccion & ~cajas).sum() == 0)
        assert bool((interseccion & ~maestro).sum() == 0)
        assert int(interseccion.sum()) <= min(int(cajas.sum()), int(maestro.sum()))

    def test_sin_mascara_entran_todos(self) -> None:
        lote = _lote([], [], lote=1)
        logits = torch.randn(1, 17, ANCLAJES)
        assert int(build_mask("none", lote, logits, GEOMETRIA, RESOLUCION, 0.05).sum()) == ANCLAJES

    def test_estrategia_desconocida_es_un_error(self) -> None:
        """Una errata en el YAML no puede acabar en "no enmascarar"."""
        lote = _lote([], [], lote=1)
        with pytest.raises(DistillationLossError, match="desconocida"):
            build_mask("gt_box", lote, torch.randn(1, 17, ANCLAJES), GEOMETRIA, RESOLUCION, 0.05)


class TestPerdidaDeCaracteristicas:
    """La MSE de características y, sobre todo, su máscara."""

    def test_mapas_identicos_dan_cero(self) -> None:
        """Con normalización por canal, dos mapas iguales siguen dando cero."""
        mapa = torch.randn(2, 8, 10, 10)
        assert float(feature_mse(mapa, mapa.clone())) < 1e-8

    def test_la_mascara_ignora_lo_que_queda_fuera(self) -> None:
        """La comprobación que separa una máscara real de una decorativa."""
        alumno = torch.zeros(1, 4, 6, 6)
        maestro = torch.zeros(1, 4, 6, 6)
        maestro[0, :, 5, 5] = 10.0  # la diferencia está solo en la esquina
        mascara = torch.zeros(1, 6, 6, dtype=torch.bool)
        mascara[0, 0, 0] = True
        fuera = feature_mse(alumno, maestro, mascara, normalize_per_channel=False)
        mascara_dentro = torch.zeros(1, 6, 6, dtype=torch.bool)
        mascara_dentro[0, 5, 5] = True
        dentro = feature_mse(alumno, maestro, mascara_dentro, normalize_per_channel=False)
        assert float(fuera) == pytest.approx(0.0)
        assert float(dentro) > 1.0

    def test_la_normalizacion_por_canal_quita_la_escala(self) -> None:
        """Multiplicar un mapa por una constante no puede cambiar la pérdida."""
        alumno = torch.randn(1, 4, 8, 8)
        maestro = torch.randn(1, 4, 8, 8)
        base = feature_mse(alumno, maestro, normalize_per_channel=True)
        escalado = feature_mse(alumno, maestro * 7.0, normalize_per_channel=True)
        assert float(base) == pytest.approx(float(escalado), rel=1e-4)

    def test_formas_distintas_son_un_error_explicito(self) -> None:
        """Sin adaptador, las formas no cuadran y hay que decirlo."""
        with pytest.raises(DistillationLossError, match="no coinciden"):
            feature_mse(torch.randn(1, 4, 8, 8), torch.randn(1, 16, 8, 8))


class TestGeometriaDeLaCabeza:
    """La geometría se lee del modelo; suponerla es lo que se quiere evitar."""

    def test_canales_por_mapa(self) -> None:
        """4*reg_max + nc, que es lo que reparte la pérdida nativa."""
        assert GEOMETRIA.channels_per_map == 81

    def test_reparto_de_canales(self) -> None:
        """Los mapas crudos se parten en distribución y clases."""
        mapas = [torch.randn(2, 81, 40, 40), torch.randn(2, 81, 20, 20), torch.randn(2, 81, 10, 10)]
        dist, cls = split_head_output(mapas, GEOMETRIA)
        assert dist.shape == (2, 64, ANCLAJES)
        assert cls.shape == (2, 17, ANCLAJES)

    def test_una_salida_de_evaluacion_es_un_error_con_mensaje(self) -> None:
        with pytest.raises(DistillationLossError, match="modo entrenamiento"):
            split_head_output([torch.randn(2, 21, 40, 40)], GEOMETRIA)

    def test_una_cabeza_sin_los_atributos_es_un_error(self) -> None:
        falso = nn.Module()
        falso.model = nn.Sequential(nn.Conv2d(3, 3, 1))
        with pytest.raises(DistillationLossError, match="no expone"):
            head_geometry(falso)


class TestCombinacion:
    """El reparto entre detección y destilación, y las unidades del desglose."""

    def _perdida(self, alpha: float = 0.5) -> DistillationLoss:
        return DistillationLoss(
            geometry=GEOMETRIA,
            resolution=RESOLUCION,
            alpha=alpha,
            class_weight=2.0,
            box_weight=3.0,
            feature_weight=4.0,
        )

    def test_el_total_es_el_que_se_retropropaga(self) -> None:
        """Registrar un total distinto del optimizado sería un número falso."""
        componentes = self._perdida().combine(
            torch.tensor(10.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor(1.0),
            marked_anchors=5,
            batch_size=4,
        )
        # (1-0.5)*10 + 0.5*(2+3+4)*4 = 5 + 18
        assert float(componentes.total) == pytest.approx(23.0)
        assert float(componentes.response_cls) == pytest.approx(2.0)

    def test_alpha_cero_deja_solo_la_deteccion(self) -> None:
        componentes = self._perdida(alpha=0.0).combine(
            torch.tensor(10.0), torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0), 5, 4
        )
        assert float(componentes.total) == pytest.approx(10.0)

    def test_alpha_fuera_de_rango_es_un_error(self) -> None:
        """Un alpha mayor que uno restaría detección."""
        with pytest.raises(DistillationLossError, match="alpha"):
            DistillationLoss(geometry=GEOMETRIA, resolution=RESOLUCION, alpha=1.5)

    def test_temperatura_no_positiva_es_un_error(self) -> None:
        """Dividir por cero daría ``inf`` y envenenaría el primer paso."""
        with pytest.raises(DistillationLossError, match="temperatura"):
            DistillationLoss(geometry=GEOMETRIA, resolution=RESOLUCION, temperature=0.0)


class _ModeloDePrueba(nn.Module):
    """Modelo mínimo con nombres de capa, para los tests de captura."""

    def __init__(self) -> None:
        super().__init__()
        self.uno = nn.Conv2d(3, 4, 3, padding=1)
        self.dos = nn.Conv2d(4, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pase hacia delante."""
        return self.dos(self.uno(x))


class _DetectorDePrueba:
    """Implementa lo mínimo que :class:`FeatureCollector` necesita de un detector."""

    def __init__(self) -> None:
        self.model = _ModeloDePrueba()

    def get_module(self, nombre: str) -> nn.Module:
        """Devuelve un submódulo por nombre."""
        return dict(self.model.named_modules())[nombre]


class TestColectorDeCaracteristicas:
    """Captura estricta de características."""

    def test_captura_y_retirada(self) -> None:
        detector = _DetectorDePrueba()
        colector = FeatureCollector(owner="prueba")
        colector.attach(detector, ["uno"])
        detector.model(torch.randn(1, 3, 8, 8))
        assert colector.features["uno"].shape == (1, 4, 8, 8)
        assert colector.detach() == 1
        assert colector.features == {}
        assert not detector.get_module("uno")._forward_hooks

    def test_dos_pases_sin_vaciar_son_un_error(self) -> None:
        """Sin esto, la pérdida compararía dos pases distintos entre sí."""
        detector = _DetectorDePrueba()
        colector = FeatureCollector(owner="prueba")
        colector.attach(detector, ["uno"])
        detector.model(torch.randn(1, 3, 8, 8))
        with pytest.raises(FeatureCaptureError, match="dos veces"):
            detector.model(torch.randn(1, 3, 8, 8))
        colector.detach()

    def test_control_tras_vaciar_el_segundo_pase_es_valido(self) -> None:
        detector = _DetectorDePrueba()
        colector = FeatureCollector(owner="prueba")
        colector.attach(detector, ["uno"])
        detector.model(torch.randn(1, 3, 8, 8))
        colector.clear()
        detector.model(torch.randn(1, 3, 8, 8))
        assert "uno" in colector.features
        colector.detach()

    def test_pedir_una_capa_no_capturada_es_un_error(self) -> None:
        """``require`` no devuelve un diccionario a medias."""
        colector = FeatureCollector(owner="prueba")
        with pytest.raises(FeatureCaptureError, match="No se han capturado"):
            colector.require(["uno"])

    def test_la_copia_profunda_va_vacia(self) -> None:
        """La EMA se reconstruye copiando; el colector no puede viajar lleno."""
        detector = _DetectorDePrueba()
        colector = FeatureCollector(owner="prueba")
        colector.attach(detector, ["uno"])
        detector.model(torch.randn(1, 3, 8, 8))
        copia = copy.deepcopy(colector)
        assert copia.features == {}
        assert copia.attached == 0
        colector.detach()

    def test_soltar_los_hooks_de_una_copia(self) -> None:
        """Lo que hace posible escribir el checkpoint."""
        detector = _DetectorDePrueba()
        colector = FeatureCollector(owner="prueba")
        colector.attach(detector, ["uno", "dos"])
        copia = copy.deepcopy(detector.model)
        assert drop_forward_hooks(copia) == 2
        assert sum(len(m._forward_hooks) for m in copia.modules()) == 0
        # El modelo vivo conserva los suyos: se limpia la copia, no el original.
        assert sum(len(m._forward_hooks) for m in detector.model.modules()) == 2
        colector.detach()


class TestAdaptadores:
    """Los adaptadores 1x1 y la comprobación de que aprenden."""

    def test_proyectan_a_los_canales_del_maestro(self) -> None:
        """La proyección va del alumno al maestro, que es la que conserva."""
        adaptadores = FeatureAdapters({"model.15": (64, 256)})
        salida = adaptadores({"model.15": torch.randn(2, 64, 10, 10)})
        assert salida["model.15"].shape == (2, 256, 10, 10)

    def test_una_capa_sin_adaptador_es_un_error(self) -> None:
        """Mejor fallar que proyectar con el adaptador de otra capa."""
        adaptadores = FeatureAdapters({"model.15": (64, 256)})
        with pytest.raises(AdapterError, match="No hay adaptador"):
            adaptadores({"model.18": torch.randn(2, 128, 10, 10)})

    def test_sin_emparejamientos_es_un_error(self) -> None:
        """Unos adaptadores vacíos harían que el término valiese siempre cero."""
        with pytest.raises(AdapterError, match="emparejamientos"):
            FeatureAdapters({})

    def test_los_adaptadores_se_mueven_con_un_optimizador(self) -> None:
        """La mitad afirmativa de la guarda del entrenador."""
        adaptadores = FeatureAdapters({"model.15": (8, 16)})
        inicial = adaptadores.weight_signature()["model.15"]
        optimizador = torch.optim.Adam(adaptadores.trainable_parameters(), lr=1e-3)
        objetivo = torch.randn(1, 16, 4, 4)
        for _ in range(5):
            salida = adaptadores({"model.15": torch.randn(1, 8, 4, 4)})["model.15"]
            ((salida - objetivo) ** 2).mean().backward()
            optimizador.step()
            optimizador.zero_grad(set_to_none=True)
        final = adaptadores.weight_signature()["model.15"]
        assert abs(final - inicial) > ADAPTER_CHANGE_TOLERANCE * inicial

    def test_control_sin_optimizador_no_se_mueven(self) -> None:
        adaptadores = FeatureAdapters({"model.15": (8, 16)})
        inicial = adaptadores.weight_signature()["model.15"]
        objetivo = torch.randn(1, 16, 4, 4)
        for _ in range(5):
            salida = adaptadores({"model.15": torch.randn(1, 8, 4, 4)})["model.15"]
            ((salida - objetivo) ** 2).mean().backward()
        assert adaptadores.weight_signature()["model.15"] == pytest.approx(inicial)


class TestComprobacionDeAprendizaje:
    """La guarda ``adapters_learned``, con el caso histórico que la engañaba."""

    def _resultado(self, inicial: float, final: float) -> DistillationResult:
        return DistillationResult(
            training=None,  # type: ignore[arg-type]
            adapter_initial_norms={"model.15": inicial},
            adapter_final_norms={"model.15": final},
        )

    def test_un_cambio_real_cuenta(self) -> None:
        """Lo que produce el optimizador correcto."""
        assert self._resultado(9.245894, 9.271070).adapters_learned

    def test_el_cambio_que_engañaba_a_la_guarda_no_cuenta(self) -> None:
        """Los números son los medidos con el optimizador equivocado."""
        assert not self._resultado(9.245894, 9.245898).adapters_learned

    def test_sin_adaptadores_no_se_afirma_que_hayan_aprendido(self) -> None:
        """Con destilación solo por respuesta no hay adaptadores que mover."""
        assert not DistillationResult(training=None).adapters_learned  # type: ignore[arg-type]


class TestConfiguracion:
    """Los valores por defecto son los medidos."""

    def _config(self, **extra) -> DistillationConfig:
        base = {
            "feature_pairs": [{"teacher": "model.15", "student": "model.15"}],
        }
        base.update(extra)
        return DistillationConfig(**base)

    def test_la_temperatura_por_defecto_es_la_medida(self) -> None:
        assert self._config().temperature == 2.0

    def test_la_mascara_de_respuesta_por_defecto_es_la_interseccion(self) -> None:
        assert self._config().region_masking == "gt_boxes_and_teacher"
        assert self._config().feature_masking == "gt_boxes"

    def test_caracteristicas_sin_emparejamientos_es_un_error(self) -> None:
        """Un ``kd_type`` con características y sin capas no destilaría nada."""
        with pytest.raises(ValueError, match="feature_pairs"):
            DistillationConfig(kd_type="combined", feature_pairs=[])

    def test_emparejamientos_que_se_ignorarian_son_un_error(self) -> None:
        with pytest.raises(ValueError, match="ignorarían"):
            self._config(kd_type="logit")

    def test_un_par_mal_escrito_es_un_error(self) -> None:
        """Una errata en las claves dejaría el emparejamiento sin efecto."""
        with pytest.raises(ValueError, match="claves"):
            DistillationConfig(
                kd_type="feature", feature_pairs=[{"maestro": "model.15", "student": "model.15"}]
            )
