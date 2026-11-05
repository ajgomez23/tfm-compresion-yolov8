"""El optimizador y la tasa de aprendizaje configurados llegan de verdad."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from torch import nn  # noqa: E402

from tfm_edge.common.config import TrainingConfig  # noqa: E402
from tfm_edge.training.trainer import OPTIMIZADORES_ADMITIDOS, Trainer, TrainerError  # noqa: E402

pytestmark = pytest.mark.requires_torch

#: Cifras reales del estudio, de ``data/resized_640/ultralytics_train.txt`` y de
#: ``configs/default.yaml``. No son un ejemplo: son las que deciden de qué lado
#: del umbral cae cada campaña.
IMAGENES_TRAIN = 5670
LOTE = 16
NBS = 16
NUM_CLASES = 17


def _modelo_minimo() -> nn.Module:
    """Un módulo con convolución, normalización y sesgo."""
    return nn.Sequential(nn.Conv2d(3, 8, 3), nn.BatchNorm2d(8), nn.SiLU())


def _construir(nombre: str, lr: float, epocas: int):
    """Deja que Ultralytics construya el optimizador, con los números del estudio."""
    from ultralytics.engine.trainer import BaseTrainer

    stub = SimpleNamespace(
        args=SimpleNamespace(lr0=lr, momentum=0.937, optimizer=nombre),
        data={"nc": NUM_CLASES},
    )
    iteraciones = math.ceil(IMAGENES_TRAIN / max(LOTE, NBS)) * epocas
    return BaseTrainer.build_optimizer(
        stub,
        model=_modelo_minimo(),
        name=nombre,
        lr=lr,
        momentum=0.937,
        decay=0.0005,
        iterations=iteraciones,
    )


class TestUltralyticsDescartaLaTasaConAuto:
    """Ultralytics descarta la tasa configurada con ``optimizer=auto``."""

    @pytest.mark.parametrize("epocas", [10, 28, 29, 30, 100])
    def test_auto_ignora_la_tasa_configurada(self, epocas: int) -> None:
        pedido = 0.001
        optimizador = _construir("auto", pedido, epocas)
        assert optimizador.param_groups[0]["lr"] != pytest.approx(pedido)

    def test_el_umbral_esta_entre_28_y_29_epocas(self) -> None:
        a_28 = _construir("auto", 0.001, 28)
        a_29 = _construir("auto", 0.001, 29)

        assert type(a_28).__name__ == "AdamW"
        assert type(a_29).__name__ == "SGD"
        assert a_28.param_groups[0]["lr"] == pytest.approx(0.000476)
        assert a_29.param_groups[0]["lr"] == pytest.approx(0.01)

    @pytest.mark.parametrize("nombre", ["SGD", "Adam", "AdamW"])
    @pytest.mark.parametrize("epocas", [10, 30])
    def test_nombrarlo_hace_que_la_tasa_llegue(self, nombre: str, epocas: int) -> None:
        """Con el optimizador nombrado, el ``lr`` pedido llega intacto."""
        pedido = 0.0001
        optimizador = _construir(nombre, pedido, epocas)
        assert type(optimizador).__name__ == nombre
        assert optimizador.param_groups[0]["lr"] == pytest.approx(pedido)


class TestElTrainerLoPasa:
    """Que la biblioteca lo respete no sirve si el proyecto no se lo dice."""

    def test_los_overrides_llevan_el_optimizador(self, tmp_path) -> None:
        """``_construir_overrides`` incluye ``optimizer``."""
        from tfm_edge.models.detector import Detector

        detector = Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False)
        entrenador = Trainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
            learning_rate=0.0001,
            optimizer="AdamW",
        )
        overrides = entrenador._construir_overrides("train")

        assert overrides["optimizer"] == "AdamW"
        assert overrides["lr0"] == pytest.approx(0.0001)

    def test_por_defecto_no_es_auto(self, tmp_path) -> None:
        from tfm_edge.models.detector import Detector

        detector = Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False)
        entrenador = Trainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
        )
        assert entrenador.optimizer == "SGD"
        assert entrenador._construir_overrides("train")["optimizer"] == "SGD"

    def test_un_optimizador_desconocido_falla_al_construir(self, tmp_path) -> None:
        from tfm_edge.models.detector import Detector

        detector = Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False)
        with pytest.raises(TrainerError, match="sgd"):
            Trainer(
                detector=detector,
                data_config=tmp_path / "data.yaml",
                output_dir=tmp_path,
                optimizer="sgd",
            )

    def test_la_lista_admitida_coincide_con_el_esquema(self) -> None:
        """``OPTIMIZADORES_ADMITIDOS`` y ``TrainingConfig.optimizer`` no se separan."""
        import typing

        anotacion = TrainingConfig.model_fields["optimizer"].annotation
        del_esquema = frozenset(typing.get_args(anotacion))
        assert del_esquema == OPTIMIZADORES_ADMITIDOS


class TestLaGuardaDeTiempoDeEjecucion:
    """La comprobación que se ejecuta en cada entrenamiento."""

    def _guarda(self, tmp_path, **kwargs):
        """Construye un Trainer y devuelve su callback de verificación."""
        from tfm_edge.models.detector import Detector

        detector = Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False)
        entrenador = Trainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
            **kwargs,
        )
        evento, funcion = entrenador._callback_de_verificacion_del_optimizador()
        return evento, funcion

    def test_el_evento_es_posterior_a_la_construccion_del_optimizador(self, tmp_path) -> None:
        """``on_train_start`` se dispara después de ``_setup_train``."""
        import inspect

        from ultralytics.engine.trainer import BaseTrainer

        evento, _ = self._guarda(tmp_path)
        assert evento == "on_train_start"

        fuente = inspect.getsource(BaseTrainer._do_train)
        assert fuente.index("_setup_train()") < fuente.index('run_callbacks("on_train_start")')

    def test_acepta_el_optimizador_correcto(self, tmp_path) -> None:
        """Con lo pedido montado, la guarda deja pasar."""
        _, verificar = self._guarda(tmp_path, learning_rate=0.001, optimizer="SGD")
        montado = torch.optim.SGD(_modelo_minimo().parameters(), lr=0.001)
        verificar(SimpleNamespace(optimizer=montado))

    def test_rechaza_un_optimizador_distinto(self, tmp_path) -> None:
        _, verificar = self._guarda(tmp_path, learning_rate=0.001, optimizer="SGD")
        montado = torch.optim.AdamW(_modelo_minimo().parameters(), lr=0.000476)
        with pytest.raises(TrainerError, match="AdamW"):
            verificar(SimpleNamespace(optimizer=montado))

    def test_rechaza_una_tasa_distinta(self, tmp_path) -> None:
        _, verificar = self._guarda(tmp_path, learning_rate=0.0001, optimizer="SGD")
        montado = torch.optim.SGD(_modelo_minimo().parameters(), lr=0.01)
        with pytest.raises(TrainerError, match="0.01"):
            verificar(SimpleNamespace(optimizer=montado))

    def test_con_auto_avisa_pero_no_falla(self, tmp_path) -> None:
        _, verificar = self._guarda(tmp_path, learning_rate=0.001, optimizer="auto")
        montado = torch.optim.AdamW(_modelo_minimo().parameters(), lr=0.000476)
        verificar(SimpleNamespace(optimizer=montado))
