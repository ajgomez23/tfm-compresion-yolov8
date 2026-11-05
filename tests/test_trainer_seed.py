"""La semilla configurada llega de verdad al entrenamiento."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.common.seed import derive_seed  # noqa: E402
from tfm_edge.training.trainer import Trainer, TrainerError  # noqa: E402

pytestmark = pytest.mark.requires_torch

NUM_CLASES = 17

#: Semillas de rama reales de las dos ejecuciones que salieron bit a bit iguales.
SEMILLA_42 = 1184247825
SEMILLA_43 = 478468298


def _detector():
    """Un detector mínimo, sin pesos preentrenados."""
    from tfm_edge.models.detector import Detector

    return Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False)


def _entrenador(tmp_path, **kwargs):
    """Construye un ``Trainer`` con lo mínimo y los ajustes que pida el test."""
    return Trainer(
        detector=_detector(),
        data_config=tmp_path / "data.yaml",
        output_dir=tmp_path,
        **kwargs,
    )


class TestElDefectoSigueEnLaBiblioteca:
    """El valor por defecto de la semilla en Ultralytics."""

    def test_el_valor_por_defecto_de_ultralytics_es_cero(self) -> None:
        from ultralytics.cfg import get_cfg

        cfg = get_cfg(overrides={"model": "yolov8n.yaml"})

        assert cfg.seed == 0
        # 'deterministic' es la otra mitad: sin él el resultado variaría de todos
        # modos y el fallo se habría visto como ruido, no como igualdad exacta.
        assert cfg.deterministic is True

    def test_el_trainer_siembra_en_su_constructor(self) -> None:
        import inspect

        from ultralytics.engine.trainer import BaseTrainer

        fuente = inspect.getsource(BaseTrainer.__init__)

        assert "init_seeds(self.args.seed + 1 + RANK" in fuente

    def test_dos_semillas_distintas_dan_flujos_distintos(self) -> None:
        from ultralytics.utils.torch_utils import init_seeds

        def flujo(semilla: int) -> list[float]:
            init_seeds(semilla + 1)
            return [random.random() for _ in range(5)]

        primero = flujo(SEMILLA_42)
        segundo = flujo(SEMILLA_43)
        repetido = flujo(SEMILLA_42)

        assert primero == repetido, "la misma semilla debe repetir el flujo"
        assert primero != segundo, "dos semillas distintas no pueden dar el mismo flujo"

    def test_sin_la_clave_las_dos_semillas_del_incidente_colapsan(self) -> None:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils.torch_utils import init_seeds

        from tfm_edge.common.seed import set_global_seed

        def flujo_como_antes(semilla_del_proyecto: int) -> list[float]:
            set_global_seed(semilla_del_proyecto)
            # Lo que hacía BaseTrainer con unos overrides sin 'seed'.
            cfg = get_cfg(overrides={"model": "yolov8n.yaml"})
            init_seeds(cfg.seed + 1, deterministic=cfg.deterministic)
            return [random.random() for _ in range(5)]

        assert flujo_como_antes(SEMILLA_42) == flujo_como_antes(SEMILLA_43)


class TestElTrainerLaPasa:
    """Que la biblioteca la respete no sirve si el proyecto no se la dice."""

    def test_ultralytics_resuelve_la_semilla_que_se_le_pide(self, tmp_path) -> None:
        """El eslabón que faltaba, comprobado donde importa."""
        from ultralytics.cfg import get_cfg

        entrenador = _entrenador(tmp_path, seed=SEMILLA_43)
        cfg = get_cfg(overrides=entrenador._construir_overrides("train"))

        assert cfg.seed == SEMILLA_43

    def test_sin_semilla_falla_en_vez_de_heredar_la_de_ultralytics(self, tmp_path) -> None:
        entrenador = _entrenador(tmp_path)

        with pytest.raises(TrainerError, match="semilla"):
            entrenador._construir_overrides("train")

    def test_extra_args_no_puede_colar_otra_semilla_sin_que_se_note(self, tmp_path) -> None:
        from ultralytics.cfg import get_cfg

        entrenador = _entrenador(tmp_path, seed=SEMILLA_43, extra_args={"seed": SEMILLA_42})
        cfg = get_cfg(overrides=entrenador._construir_overrides("train"))

        assert cfg.seed == SEMILLA_42, "extra_args gana: por eso la guarda no es opcional"

        _, verificar = entrenador._callback_de_verificacion_de_la_semilla()
        from types import SimpleNamespace

        with pytest.raises(TrainerError, match="no ha llegado"):
            verificar(SimpleNamespace(args=cfg))


class TestLaGuardaDeEjecucion:
    """La comprobación que se ejecuta en cada entrenamiento."""

    def _verificar(self, tmp_path, seed: int):
        """Devuelve la función de la guarda, ya construida."""
        _, funcion = _entrenador(tmp_path, seed=seed)._callback_de_verificacion_de_la_semilla()
        return funcion

    def test_pasa_cuando_la_semilla_ha_llegado(self, tmp_path) -> None:
        """El caso bueno no debe fallar."""
        from types import SimpleNamespace

        verificar = self._verificar(tmp_path, SEMILLA_43)
        verificar(SimpleNamespace(args=SimpleNamespace(seed=SEMILLA_43)))

    def test_falla_con_el_cero_de_ultralytics(self, tmp_path) -> None:
        from types import SimpleNamespace

        verificar = self._verificar(tmp_path, SEMILLA_43)
        with pytest.raises(TrainerError, match="seed=0"):
            verificar(SimpleNamespace(args=SimpleNamespace(seed=0)))

    def test_falla_si_ultralytics_no_trae_la_semilla(self, tmp_path) -> None:
        from types import SimpleNamespace

        verificar = self._verificar(tmp_path, SEMILLA_43)
        with pytest.raises(TrainerError, match="args.seed"):
            verificar(SimpleNamespace(args=SimpleNamespace()))


class TestLasRamasNoCompartenSemilla:
    """Cada rama tiene su propia semilla."""

    def test_las_semillas_del_incidente_son_las_que_se_esperaban(self) -> None:
        assert derive_seed(42, "distillation", "combined") == SEMILLA_42
        assert derive_seed(43, "distillation", "combined") == SEMILLA_43

    @pytest.mark.parametrize("base", [42, 43, 44, 45])
    def test_cada_rama_de_una_misma_base_recibe_semilla_propia(self, base: int) -> None:
        """Piloto y control no pueden compartir semilla dentro de una misma base."""
        derivadas = {
            derive_seed(base, "distillation", tipo) for tipo in ("combined", "response", "feature")
        }

        assert len(derivadas) == 3
        assert all(0 <= s < 2**31 for s in derivadas), "fuera del rango de manual_seed"
