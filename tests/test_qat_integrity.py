"""Tests de las guardas de invariante del afinado QAT."""

from __future__ import annotations

import copy
import io
from typing import Any

import pytest
import torch

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.compression.quantization import qat
from tfm_edge.compression.quantization.finetune import (
    QatFinetuneReport,
    count_fake_quant_modules,
    finetune_qat,
    make_qat_integrity_callbacks,
    observers_have_been_exercised,
)
from tfm_edge.models.detector import Detector, drop_quantization_config


@pytest.fixture(scope="module")
def detector() -> Detector:
    """Un yolov8n pequeño, suficiente para todo lo de este módulo."""
    return Detector.from_architecture("n", num_classes=17, input_resolution=320)


@pytest.fixture(scope="module")
def preparado(detector: Detector) -> Any:
    """El mismo modelo, ya con la simulación insertada."""
    modelo, _ = qat.prepare_for_qat(detector, check_gradient=False)
    return modelo


class _EmaDeJuguete:
    """Reproduce lo que hace ``ModelEMA`` y basta para las guardas."""

    def __init__(self, model: Any) -> None:
        self.ema = copy.deepcopy(model)
        self.enabled = True
        self.updates = 0


class _TrainerDeJuguete:
    """Lo mínimo que las guardas miran de un entrenador de Ultralytics."""

    def __init__(self, model: Any, amp: bool = False) -> None:
        self.model = model
        self.ema = _EmaDeJuguete(model)
        # Ultralytics lo deja resuelto a bool en '_setup_train', antes de
        # 'on_train_start' (engine/trainer.py:304 y :371).
        self.amp = amp


def _guardas(informe: QatFinetuneReport) -> dict[str, Any]:
    """Devuelve las guardas indexadas por su nombre."""
    return {funcion.__name__: funcion for _, funcion in make_qat_integrity_callbacks(informe)}


class TestContarLaSimulacion:
    """``count_fake_quant_modules`` es lo que el recuento de parámetros no ve."""

    def test_un_modelo_normal_no_tiene_ninguno(self, detector: Detector) -> None:
        assert count_fake_quant_modules(detector.model) == 0

    def test_uno_preparado_tiene_varios(self, preparado: Any) -> None:
        assert count_fake_quant_modules(preparado) > 0

    def test_el_recuento_de_parametros_NO_distingue_los_dos(
        self, detector: Detector, preparado: Any
    ) -> None:
        """El motivo de que esta guarda exista."""
        normales = sum(p.numel() for p in detector.model.parameters())
        con_qat = sum(p.numel() for p in preparado.parameters())
        assert normales == con_qat


class TestLaGuardaDeArranque:
    """Aborta antes de gastar GPU si lo que va a entrenarse no es QAT."""

    def test_pasa_con_un_modelo_preparado(self, preparado: Any) -> None:
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_entrena_el_modelo_preparado"]
        guarda(_TrainerDeJuguete(preparado))
        assert informe.integrity["fake_quant_at_train_start"] > 0

    def test_aborta_con_un_modelo_sin_preparar(self, detector: Detector) -> None:
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_entrena_el_modelo_preparado"]
        with pytest.raises(TFMEdgeError, match="cuantización simulada"):
            guarda(_TrainerDeJuguete(detector.model))

    def test_desactiva_la_ema_para_el_primer_lote(self, preparado: Any) -> None:
        """Lo que evita el choque de formas en el primer paso del optimizador."""
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_entrena_el_modelo_preparado"]
        entrenador = _TrainerDeJuguete(preparado)
        assert entrenador.ema.enabled is True
        guarda(entrenador)
        assert entrenador.ema.enabled is False


class TestLaReconstruccionDeLaEma:
    """Y la vuelve a poner en cuanto las formas se han asentado."""

    def test_reconstruye_una_vez_y_solo_una(self, preparado: Any) -> None:
        informe = QatFinetuneReport()
        guardas = _guardas(informe)
        arranque = guardas["comprobar_que_entrena_el_modelo_preparado"]
        rehacer = guardas["rehacer_la_ema_tras_el_primer_lote"]
        entrenador = _TrainerDeJuguete(preparado)
        arranque(entrenador)
        primera = entrenador.ema

        rehacer(entrenador)
        assert entrenador.ema is not primera
        assert entrenador.ema.enabled is True
        assert informe.integrity["ema_rebuilt_after_first_batch"] is True

        segunda = entrenador.ema
        rehacer(entrenador)
        assert entrenador.ema is segunda, "no debe rehacerse en cada lote"

    def test_la_ema_reconstruida_conserva_la_simulacion(self, preparado: Any) -> None:
        informe = QatFinetuneReport()
        guardas = _guardas(informe)
        entrenador = _TrainerDeJuguete(preparado)
        guardas["comprobar_que_entrena_el_modelo_preparado"](entrenador)
        guardas["rehacer_la_ema_tras_el_primer_lote"](entrenador)
        assert informe.integrity["fake_quant_in_ema"] == count_fake_quant_modules(preparado)


class TestLaGuardaDeFp32:
    """AMP y cuantización simulada no conviven."""

    def test_pasa_sin_precision_mixta(self, preparado: Any) -> None:
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_el_afinado_corre_en_fp32"]
        guarda(_TrainerDeJuguete(preparado, amp=False))
        assert informe.integrity["amp"] is False

    def test_aborta_con_amp(self, preparado: Any) -> None:
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_el_afinado_corre_en_fp32"]
        with pytest.raises(TFMEdgeError, match="precisión mixta"):
            guarda(_TrainerDeJuguete(preparado, amp=True))

    def test_mira_el_estado_real_y_no_lo_que_se_pidio(self, preparado: Any) -> None:
        """Por qué la guarda lee ``trainer.amp`` y no la configuración."""
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_el_afinado_corre_en_fp32"]
        entrenador = _TrainerDeJuguete(preparado, amp=False)
        entrenador.amp = True  # como si Ultralytics lo hubiera resuelto a True
        with pytest.raises(TFMEdgeError):
            guarda(entrenador)


class TestQueElChoqueConAmpEsReal:
    """El choque con AMP se reproduce con el kernel de PyTorch."""

    def test_el_nucleo_de_la_simulacion_rechaza_media_precision(self) -> None:
        """Exactamente el error que mató la campaña de Colab."""
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize

        simulador = FusedMovingAvgObsFakeQuantize()
        simulador(torch.randn(4, 8))  # asienta los búferes en FP32

        with pytest.raises(RuntimeError, match="expected scalar type Float but found Half"):
            simulador(torch.randn(4, 8, dtype=torch.float16))

    def test_autocast_NO_protege_al_nucleo(self) -> None:
        """Por qué no vale «excluir la simulación del autocast»."""
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize

        simulador = FusedMovingAvgObsFakeQuantize()
        entrada = torch.randn(4, 8)
        with torch.autocast("cpu", dtype=torch.float16):
            salida = simulador(entrada)
        assert salida.dtype is torch.float32

    def test_un_modelo_preparado_muere_dentro_de_autocast(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))  # un paso limpio primero

        with pytest.raises(RuntimeError, match="expected scalar type Float but found Half"):
            with torch.autocast("cpu", dtype=torch.float16):
                modelo(torch.randn(1, 3, 320, 320))


#: Convolución del ``DFL``, la única cuantizada que no se ejecuta durante el entrenamiento.
RUTA_DFL = "model.22.dfl.conv"

#: Una convolución cualquiera del cuerpo, que sí se ejecuta en cada lote.
RUTA_NORMAL = "model.0.conv"


def _desasentar(modelo: Any, ruta: str) -> Any:
    """Devuelve un observador de peso al estado de recién preparado."""
    conv = dict(modelo.named_modules())[ruta]
    observador = conv.weight_fake_quant.activation_post_process
    with torch.no_grad():
        observador.min_val.resize_(0)
        observador.max_val.resize_(0)
    return observador


def _desasentar_el_dfl(modelo: Any) -> Any:
    """Deja el observador del ``DFL`` sin forma, como antes de asentarlo."""
    return _desasentar(modelo, RUTA_DFL)


class TestQueElDflNoSeEjecutaEntrenando:
    """La propiedad de Ultralytics de la que salía el fallo."""

    def test_es_el_unico_modulo_cuantizado_que_no_corre_entrenando(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        conv = dict(modelo.named_modules())["model.22.dfl.conv"]
        # El [1] del mensaje de error sale de aquí: el DFL proyecta a un canal.
        assert conv.out_channels == 1

        veces = {"n": 0}
        conv.register_forward_hook(lambda *_: veces.__setitem__("n", veces["n"] + 1))

        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))
        assert veces["n"] == 0, "Detect.forward vuelve antes de llegar al DFL"

        modelo.eval()
        with torch.no_grad():
            modelo(torch.randn(1, 3, 320, 320))
        assert veces["n"] > 0, "en evaluación sí se ejecuta, y ahí asienta su forma"

    def test_y_por_eso_su_observador_se_quedaba_sin_forma(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        observador = _desasentar_el_dfl(modelo)

        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))
        assert observador.min_val.shape == (0,), "entrenar no le llega"

        modelo.eval()
        with torch.no_grad():
            modelo(torch.randn(1, 3, 320, 320))
        assert observador.min_val.shape == (1,), "validar sí"


class TestQueElChoqueDeFormasEsReal:
    """El choque de formas se reproduce con el ``ModelEMA`` de Ultralytics."""

    def test_una_forma_sin_asentar_mata_la_actualizacion_de_la_ema(self, preparado: Any) -> None:
        from ultralytics.utils.torch_utils import ModelEMA

        modelo = copy.deepcopy(preparado)
        _desasentar(modelo, RUTA_NORMAL)
        ema = ModelEMA(modelo)

        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))

        with pytest.raises(RuntimeError, match=r"must match the size of tensor"):
            ema.update(modelo)

    def test_la_secuencia_EXACTA_que_mato_la_campana(self, preparado: Any) -> None:
        """Entrenar, validar sobre la EMA, y arrancar la época siguiente."""
        from ultralytics.utils.torch_utils import ModelEMA

        modelo = copy.deepcopy(preparado)
        _desasentar_el_dfl(modelo)

        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))  # época 1
        ema = ModelEMA(modelo)  # la reconstrucción
        ema.ema.eval()
        with torch.no_grad():
            ema.ema(torch.randn(1, 3, 320, 320))  # validación de la época 1

        with pytest.raises(
            RuntimeError, match=r"output with shape \[1\] doesn't match the broadcast shape \[0\]"
        ):
            ema.update(modelo)  # primer paso del optimizador de la época 2

    def test_y_que_asentarlas_al_preparar_lo_evita(self, preparado: Any) -> None:
        """La misma secuencia sobre lo que ``prepare_for_qat`` devuelve hoy."""
        from ultralytics.utils.torch_utils import ModelEMA

        modelo = copy.deepcopy(preparado)  # sin desasentar nada
        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))
        ema = ModelEMA(modelo)
        ema.ema.eval()
        with torch.no_grad():
            ema.ema(torch.randn(1, 3, 320, 320))

        ema.update(modelo)  # no debe lanzar nada

        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))
        ema.update(modelo)  # ni en la vuelta siguiente


class TestQueNoQuedanFormasPorDecidir:
    """El invariante que sustituye a buscar el ``DFL``."""

    def test_preparar_no_deja_ningun_bufer_sin_forma(self, preparado: Any) -> None:
        assert qat.count_empty_buffers(preparado) == 0

    def test_asentar_los_pesos_es_lo_que_lo_consigue(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        _desasentar_el_dfl(modelo)
        assert qat.count_empty_buffers(modelo) == 2  # min_val y max_val del DFL
        assert qat.settle_weight_observers(modelo) > 0
        assert qat.count_empty_buffers(modelo) == 0

    def test_la_guarda_pasa_con_las_formas_asentadas(self, preparado: Any) -> None:
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_ninguna_forma_esta_por_decidir"]
        guarda(_TrainerDeJuguete(preparado))
        assert informe.integrity["empty_buffers_at_train_start"] == 0

    def test_la_guarda_aborta_si_queda_alguna(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        _desasentar_el_dfl(modelo)
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_ninguna_forma_esta_por_decidir"]
        with pytest.raises(TFMEdgeError, match="sin forma real"):
            guarda(_TrainerDeJuguete(modelo))

    def test_la_guarda_tambien_mira_la_ema_y_no_solo_el_modelo(self, preparado: Any) -> None:
        """El fallo era justo que uno de los dos tuviera lo que el otro no."""
        entrenador = _TrainerDeJuguete(copy.deepcopy(preparado))
        _desasentar_el_dfl(entrenador.ema.ema)
        informe = QatFinetuneReport()
        guarda = _guardas(informe)["comprobar_que_ninguna_forma_esta_por_decidir"]
        with pytest.raises(TFMEdgeError, match="sin forma real"):
            guarda(entrenador)


class TestQueAsentarLosPesosNoTapaLaComprobacionDeDatos:
    """Asentar los observadores de peso no hace pasar la comprobación de datos."""

    def test_recien_preparado_dice_que_NO_han_visto_datos(self, preparado: Any) -> None:
        assert observers_have_been_exercised(copy.deepcopy(preparado)) is False

    def test_y_los_pesos_ya_estan_asentados_aun_asi(self, preparado: Any) -> None:
        conv = dict(preparado.named_modules())["model.22.dfl.conv"]
        rango = conv.weight_fake_quant.activation_post_process
        assert rango.min_val.numel() > 0
        assert bool(torch.isfinite(rango.min_val).all())

    def test_tras_un_paso_con_datos_dice_que_SI(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        modelo.train()
        modelo(torch.randn(1, 3, 320, 320))
        assert observers_have_been_exercised(modelo) is True


class TestQueElCheckpointSePuedeEscribir:
    """``torch.save`` es quien rompía, así que es quien tiene que ejecutarse."""

    def test_un_modelo_preparado_NO_se_puede_serializar_tal_cual(self, preparado: Any) -> None:
        """El fallo que dejaba un afinado entero sin escribir nada a disco."""
        modelo = copy.deepcopy(preparado)
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            torch.save(modelo, io.BytesIO())

    def test_y_sin_el_qconfig_si(self, preparado: Any) -> None:
        modelo = copy.deepcopy(preparado)
        quitados = drop_quantization_config(modelo)
        assert quitados > 0
        torch.save(modelo, io.BytesIO())  # no debe lanzar nada

    def test_quitarlo_NO_se_lleva_por_delante_la_simulacion(self, preparado: Any) -> None:
        """Lo que guarda los rangos aprendidos tiene que sobrevivir al guardado."""
        modelo = copy.deepcopy(preparado)
        antes = count_fake_quant_modules(modelo)
        drop_quantization_config(modelo)
        assert count_fake_quant_modules(modelo) == antes

    def test_sobre_un_modelo_normal_no_hace_nada(self, detector: Detector) -> None:
        assert drop_quantization_config(copy.deepcopy(detector.model)) == 0


class TestQueAmpNoSeCuelaPorLaConfiguracion:
    """La guarda de ``on_train_start`` es la red; esto falla antes, al construir."""

    def test_rechaza_amp_en_trainer_kwargs(self, detector: Detector, tmp_path: Any) -> None:
        with pytest.raises(TFMEdgeError, match="amp"):
            finetune_qat(
                detector,
                None,  # type: ignore[arg-type]
                data_config=tmp_path / "sh17.yaml",
                output_dir=tmp_path,
                trainer_kwargs={"amp": True},
            )

    def test_rechaza_amp_en_extra_args(self, detector: Detector, tmp_path: Any) -> None:
        with pytest.raises(TFMEdgeError, match="FP32"):
            finetune_qat(
                detector,
                None,  # type: ignore[arg-type]
                data_config=tmp_path / "sh17.yaml",
                output_dir=tmp_path,
                trainer_kwargs={"extra_args": {"amp": True}},
            )
