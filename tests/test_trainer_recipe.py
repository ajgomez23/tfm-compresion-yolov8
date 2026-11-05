"""Tests de la receta que el ``Trainer`` pasa a Ultralytics."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.common.config import TrainingConfig, load_config  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.training.trainer import (  # noqa: E402
    CLAVES_DE_RECETA_GESTIONADAS,
    CLOSE_MOSAIC_POR_DEFECTO,
    COS_LR_POR_PLANIFICADOR,
    Trainer,
    TrainerError,
    mosaic_regime,
)

pytestmark = pytest.mark.requires_torch

NUM_CLASES = 17


def _detector():
    """Un detector mínimo, sin pesos preentrenados."""
    from tfm_edge.models.detector import Detector

    return Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False)


def _entrenador(tmp_path, **kwargs):
    """Construye un ``Trainer`` con semilla y los ajustes que pida el test."""
    kwargs.setdefault("seed", 1)
    return Trainer(
        detector=_detector(),
        data_config=tmp_path / "data.yaml",
        output_dir=tmp_path,
        **kwargs,
    )


def _resuelto(entrenador):
    """Pasa los overrides por el ``get_cfg`` de Ultralytics."""
    from ultralytics.cfg import get_cfg

    return get_cfg(overrides=entrenador._construir_overrides("train"))


class TestNadaCambiaParaLoYaMedido:
    """Los defectos del Trainer son los que heredaron todas las campañas."""

    def test_los_defectos_coinciden_con_los_de_ultralytics(self, tmp_path) -> None:
        from ultralytics.cfg import get_cfg

        de_la_biblioteca = get_cfg(overrides={"model": "yolov8n.yaml"})
        del_trainer = _resuelto(_entrenador(tmp_path, epochs=100))

        for clave in sorted(CLAVES_DE_RECETA_GESTIONADAS):
            assert getattr(del_trainer, clave) == getattr(de_la_biblioteca, clave), clave

    def test_el_defecto_es_lineal_porque_es_lo_que_corrio(self, tmp_path) -> None:
        assert _resuelto(_entrenador(tmp_path, epochs=100)).cos_lr is False

    def test_close_mosaic_por_defecto_es_el_de_la_biblioteca(self) -> None:
        from ultralytics.cfg import get_cfg

        assert get_cfg(overrides={"model": "yolov8n.yaml"}).close_mosaic == CLOSE_MOSAIC_POR_DEFECTO


class TestLaRecetaLlega:
    """Lo que se pide es lo que Ultralytics resuelve."""

    def test_coseno_llega_como_cos_lr(self, tmp_path) -> None:
        assert _resuelto(_entrenador(tmp_path, epochs=100, scheduler="cosine")).cos_lr is True

    def test_lrf_momentum_y_weight_decay_llegan(self, tmp_path) -> None:
        cfg = _resuelto(
            _entrenador(
                tmp_path,
                epochs=100,
                final_learning_rate_factor=0.02,
                momentum=0.9,
                weight_decay=0.001,
            )
        )
        assert cfg.lrf == 0.02
        assert cfg.momentum == 0.9
        assert cfg.weight_decay == 0.001

    def test_step_se_rechaza_en_vez_de_caer_en_lineal(self, tmp_path) -> None:
        """El esquema lo admite; Ultralytics no lo ofrece."""
        with pytest.raises(TrainerError, match="step"):
            _entrenador(tmp_path, scheduler="step")

    def test_extra_args_no_puede_pisar_la_receta(self, tmp_path) -> None:
        for clave in sorted(CLAVES_DE_RECETA_GESTIONADAS):
            with pytest.raises(TrainerError, match=clave):
                _entrenador(tmp_path, extra_args={clave: 1})

    def test_el_esquema_y_la_traduccion_no_se_separan(self) -> None:
        """Todo planificador traducible tiene que ser admisible en el esquema."""
        admitidos = set(TrainingConfig.model_fields["scheduler"].annotation.__args__)
        assert set(COS_LR_POR_PLANIFICADOR) <= admitidos


class TestLaGuardaDeEjecucionDeLaReceta:
    """La comprobación de ``on_train_start`` sobre los argumentos resueltos."""

    def test_pasa_si_coincide(self, tmp_path) -> None:
        entrenador = _entrenador(tmp_path, epochs=100, scheduler="cosine")
        _, verificar = entrenador._callback_de_verificacion_de_la_receta()
        verificar(SimpleNamespace(args=_resuelto(entrenador)))

    def test_falla_si_algo_no_llego(self, tmp_path) -> None:
        entrenador = _entrenador(tmp_path, epochs=100, momentum=0.9)
        cfg = _resuelto(entrenador)
        cfg.momentum = 0.937  # lo que habría quedado si la clave no llegara
        _, verificar = entrenador._callback_de_verificacion_de_la_receta()
        with pytest.raises(TrainerError, match="momentum"):
            verificar(SimpleNamespace(args=cfg))


class TestElRegimenDeMosaico:
    """Guarda del régimen de mosaico en el punto común de los cuatro caminos."""

    def test_el_disparador_de_la_biblioteca_sigue_siendo_una_igualdad(self) -> None:
        from ultralytics.engine.trainer import BaseTrainer

        fuente = inspect.getsource(BaseTrainer._do_train)
        assert "if epoch == (self.epochs - self.args.close_mosaic):" in fuente

    @pytest.mark.parametrize(
        ("epocas", "cierre", "regimen"),
        [
            (100, 10, "cierra"),  # baselines
            (30, 10, "cierra"),  # reajuste de poda de una etapa
            (5, 10, "cierre_inalcanzable"),  # pasos parciales de poda iterativa
            (10, 10, "sin_mosaico"),  # QAT y piloto de destilación
            (10, 0, "sin_cierre"),
        ],
    )
    def test_los_regimenes_de_d142(self, epocas: int, cierre: int, regimen: str) -> None:
        assert mosaic_regime(epocas, cierre)[0] == regimen

    def test_sin_mosaico_se_rechaza_si_no_se_pide(self, tmp_path) -> None:
        entrenador = _entrenador(tmp_path, epochs=10)
        with pytest.raises(TrainerError, match="NI UN LOTE"):
            entrenador._construir_overrides("train")

    def test_sin_mosaico_pedido_a_sabiendas_se_admite(self, tmp_path) -> None:
        """La receta ya medida sigue siendo reproducible, pidiéndola."""
        cfg = _resuelto(_entrenador(tmp_path, epochs=10, allow_mosaic_free=True))
        assert cfg.close_mosaic == cfg.epochs == 10

    def test_se_evalua_sobre_los_argumentos_ya_fusionados(self, tmp_path) -> None:
        entrenador = _entrenador(tmp_path, epochs=100, extra_args={"epochs": 10})
        with pytest.raises(TrainerError, match="NI UN LOTE"):
            entrenador._construir_overrides("train")

    def test_el_cierre_inalcanzable_no_se_rechaza(self, tmp_path) -> None:
        _entrenador(tmp_path, epochs=5)._construir_overrides("train")


class TestElCalentamientoSePasaSiempre:
    """``warmup_epochs`` se pasa siempre y llega a la configuración resuelta por Ultralytics."""

    def test_el_valor_pedido_llega_a_ultralytics(self, tmp_path) -> None:
        cfg = _resuelto(_entrenador(tmp_path, warmup_epochs=1.5))
        assert cfg.warmup_epochs == 1.5

    def test_el_defecto_es_el_que_corrieron_todas_las_campanas(self, tmp_path) -> None:
        """3.0, que es lo que heredaban del defecto de Ultralytics."""
        cfg = _resuelto(_entrenador(tmp_path))
        assert cfg.warmup_epochs == 3.0

    def test_cero_llega_tal_cual(self, tmp_path) -> None:
        """Es la receta del afinado QAT y del piloto de destilación."""
        cfg = _resuelto(_entrenador(tmp_path, warmup_epochs=0.0, epochs=10, allow_mosaic_free=True))
        assert cfg.warmup_epochs == 0.0

    def test_por_extra_args_se_rechaza(self, tmp_path) -> None:
        """Se fusiona después, así que pisaría la receta en silencio."""
        with pytest.raises(TrainerError, match="warmup_epochs"):
            _entrenador(tmp_path, extra_args={"warmup_epochs": 0.0})


class TestElModoDeterministaSePasaSiempre:
    """``deterministic`` se pasa siempre a Ultralytics."""

    def test_el_defecto_es_el_que_corrieron_todas_las_campanas(self, tmp_path) -> None:
        cfg = _resuelto(_entrenador(tmp_path))
        assert cfg.deterministic is True

    def test_pedirlo_en_falso_llega_a_ultralytics(self, tmp_path) -> None:
        """La comprobación que distingue «llega» de «coincide por casualidad»."""
        cfg = _resuelto(_entrenador(tmp_path, deterministic=False))
        assert cfg.deterministic is False

    def test_por_extra_args_se_rechaza(self, tmp_path) -> None:
        with pytest.raises(TrainerError, match="deterministic"):
            _entrenador(tmp_path, extra_args={"deterministic": False})


class TestLaConfiguracionDeclaraLoQueCorre:
    """El esquema y el YAML base dicen lo que se entrenó."""

    def test_el_yaml_base_declara_lineal_y_dos_cargadores(self) -> None:
        config = load_config(get_project_paths().root / "configs" / "default.yaml")
        assert config.training.scheduler == "linear"
        assert config.training.workers == 2

    def test_los_argumentos_de_receta_los_acepta_el_trainer(self, tmp_path) -> None:
        """La traducción del esquema al Trainer vive en un sitio y encaja."""
        receta = TrainingConfig().trainer_recipe_kwargs()
        entrenador = _entrenador(tmp_path, epochs=100, **receta)
        assert entrenador.workers == 2
        assert entrenador.scheduler == "linear"

    def test_workers_entra_en_la_identidad(self) -> None:
        from tfm_edge.common.config import compute_config_hash

        ruta = get_project_paths().root / "configs" / "default.yaml"
        dos = load_config(ruta)
        cero = load_config(ruta, overrides=["training.workers=0"])
        assert compute_config_hash(dos.model_dump(mode="json")) != compute_config_hash(
            cero.model_dump(mode="json")
        )
