"""Comprueba que la clasificación de claves heredadas cubre lo que el código hereda de Ultralytics.
"""

from __future__ import annotations

import importlib.util
from copy import copy
from pathlib import Path

import pytest
import yaml

from tfm_edge.training.inherited_cfg import INHERITED_ON_PURPOSE, IRRELEVANT_WHEN_TRAINING, classify

pytestmark = pytest.mark.requires_torch

NUM_CLASES = 17


def _claves_de_ultralytics() -> dict[str, object]:
    """Lee el ``cfg/default.yaml`` del paquete instalado sin importarlo."""
    spec = importlib.util.find_spec("ultralytics")
    assert spec is not None and spec.origin is not None, "ultralytics no está instalado"
    destino = Path(spec.origin).parent / "cfg" / "default.yaml"
    return yaml.safe_load(destino.read_text(encoding="utf-8"))


def _overrides_reales(tmp_path: Path) -> dict[str, object]:
    """Construye un ``Trainer`` de verdad y le pide sus overrides."""
    from tfm_edge.models.detector import Detector
    from tfm_edge.training.trainer import Trainer

    entrenador = Trainer(
        detector=Detector.from_architecture(size="n", num_classes=NUM_CLASES, pretrained=False),
        data_config=tmp_path / "data.yaml",
        output_dir=tmp_path,
        seed=1,
    )
    return entrenador._construir_overrides("train")


class TestLaParticionEsExhaustiva:
    """Ninguna clave heredada se queda sin mirar."""

    def test_toda_clave_heredada_esta_clasificada(self, tmp_path) -> None:
        todas = set(_claves_de_ultralytics())
        pasadas = set(_overrides_reales(tmp_path))
        heredadas = todas - pasadas

        _, _, sin_clasificar = classify(heredadas)

        assert not sin_clasificar, (
            "Estas claves de Ultralytics no las pasa el Trainer y no están en "
            "ninguno de los dos cubos de inherited_cfg.py. Clasifícalas antes de "
            f"entrenar nada: {sorted(sin_clasificar)}"
        )

    def test_no_se_clasifica_lo_que_si_se_pasa(self, tmp_path) -> None:
        """Una clave clasificada como heredada que en realidad se pasa es ruido."""
        pasadas = set(_overrides_reales(tmp_path))
        declaradas = set(IRRELEVANT_WHEN_TRAINING) | set(INHERITED_ON_PURPOSE)

        solapan = pasadas & declaradas

        assert not solapan, (
            "Estas claves SÍ las pasa el Trainer, así que no se heredan y sobran "
            f"en inherited_cfg.py: {sorted(solapan)}"
        )

    def test_no_se_clasifica_lo_que_ultralytics_no_declara(self) -> None:
        """Una clave inventada o retirada aguas arriba también es un fallo."""
        todas = set(_claves_de_ultralytics())
        declaradas = set(IRRELEVANT_WHEN_TRAINING) | set(INHERITED_ON_PURPOSE)

        fantasmas = declaradas - todas

        assert not fantasmas, (
            "Estas claves están clasificadas y Ultralytics no las declara: " f"{sorted(fantasmas)}"
        )

    def test_los_dos_cubos_son_disjuntos(self) -> None:
        """Una clave en los dos cubos significa que no se decidió cuál era."""
        solapan = set(IRRELEVANT_WHEN_TRAINING) & set(INHERITED_ON_PURPOSE)
        assert not solapan, f"Clasificadas dos veces: {sorted(solapan)}"

    def test_toda_clave_lleva_motivo_escrito(self) -> None:
        for cubo in (IRRELEVANT_WHEN_TRAINING, INHERITED_ON_PURPOSE):
            for clave, motivo in cubo.items():
                assert motivo.strip(), f"'{clave}' no tiene motivo escrito"
                assert len(motivo) > 25, f"'{clave}' tiene un motivo demasiado corto: {motivo!r}"


class TestElModoDeterministaNoPuedeColarse:
    """``deterministic`` no entra en la identidad, así que con False no se construye."""

    def test_con_true_el_payload_sale_normal(self) -> None:
        from tfm_edge.common.config import experiment_identity_payload, load_config
        from tfm_edge.common.paths import get_project_paths

        config = load_config(get_project_paths().root / "configs" / "default.yaml")

        assert config.deterministic is True
        assert "training" in experiment_identity_payload(config)

    def test_con_false_se_niega_a_construir_la_identidad(self) -> None:
        from tfm_edge.common.config import experiment_identity_payload, load_config
        from tfm_edge.common.exceptions import ConfigSchemaError
        from tfm_edge.common.paths import get_project_paths

        config = load_config(get_project_paths().root / "configs" / "default.yaml")
        con_falso = config.model_copy(update={"deterministic": False})

        with pytest.raises(ConfigSchemaError, match="deterministic"):
            experiment_identity_payload(con_falso)

    def test_el_mensaje_dice_que_hacer_para_usarlo(self) -> None:
        """Una excepción que no dice qué puede hacer el usuario es media excepción."""
        from tfm_edge.common.config import experiment_identity_payload, load_config
        from tfm_edge.common.exceptions import ConfigSchemaError
        from tfm_edge.common.paths import get_project_paths

        config = load_config(get_project_paths().root / "configs" / "default.yaml")

        with pytest.raises(ConfigSchemaError) as error:
            experiment_identity_payload(config.model_copy(update={"deterministic": False}))

        assert "EXPERIMENT_IDENTITY_SECTIONS" in str(error.value)


class TestLoQueLaAuditoriaAfirmaDeUltralytics:
    """Las afirmaciones del módulo sobre Ultralytics, comprobadas contra Ultralytics."""

    def test_pretrained_es_inerte_con_un_modelo_ya_construido(self, tmp_path) -> None:
        import inspect

        from ultralytics.engine.trainer import BaseTrainer

        fuente = inspect.getsource(BaseTrainer.setup_model)

        assert "isinstance(self.model, torch.nn.Module)" in fuente, (
            "setup_model no sale antes ante un nn.Module; 'pretrained' podría dejar de ser "
            "inerte. Revisa inherited_cfg.py."
        )
        assert "isinstance(self.args.pretrained, (str, Path))" in fuente, (
            "setup_model no exige que 'pretrained' sea una ruta para usarla: "
            "revisa el motivo declarado en inherited_cfg.py."
        )

    def test_la_aptitud_que_elige_best_pt_es_solo_el_map(self) -> None:
        """Justifica por qué las claves de validación importan para best.pt."""
        import numpy as np
        from ultralytics.utils.metrics import Metric

        metrica = Metric()
        # mean_results() devuelve [P, R, mAP@0.5, mAP@0.5:0.95]; la aptitud las
        # pondera. Se sustituye por un vector conocido para leer los pesos sin
        # depender de ninguna evaluación real.
        metrica.mean_results = lambda: [0.1, 0.2, 0.3, 0.4]  # type: ignore[method-assign]

        assert np.isclose(metrica.fitness(), 0.4), (
            "La aptitud de Ultralytics no es solo el mAP@0.5:0.95. Los "
            "motivos de conf/iou/max_det en inherited_cfg.py dependen de eso."
        )

    def test_el_validador_resuelve_conf_a_la_milesima(self) -> None:
        """El motivo de ``conf`` afirma que None se resuelve a 0.001."""
        import inspect

        from ultralytics.engine.validator import BaseValidator

        fuente = inspect.getsource(BaseValidator.__init__)
        assert "self.args.conf = 0.01 if self.args.task ==" in fuente and "0.001" in fuente, (
            "El validador no resuelve conf=None a 0.001: revisa el motivo "
            "declarado para 'conf' en inherited_cfg.py."
        )

    def test_el_validador_de_entrenamiento_hereda_los_args_del_entrenador(self) -> None:
        """Sin esto, las claves de validación no llegarían al validador de cada época."""
        import inspect

        from ultralytics.models.yolo.detect.train import DetectionTrainer

        fuente = inspect.getsource(DetectionTrainer.get_validator)
        assert "args=copy(self.args)" in fuente, (
            "El validador de entrenamiento no recibe una copia de los args del entrenador: "
            "conf, iou, max_det y split podrían venir de otro sitio. Revisa inherited_cfg.py."
        )
        assert copy is not None  # el import del módulo documenta de dónde sale la copia
