"""Tests de guarda del diseño del Trainer."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.models.detector import Detector, drop_cached_loss  # noqa: E402
from tfm_edge.training.trainer import (  # noqa: E402
    STUDY_MODEL,
    PrecisionError,
    Trainer,
    TrainerError,
    verify_checkpoint_is_fp32,
)

pytestmark = pytest.mark.requires_torch


class TestSuposicionesSobreUltralytics:
    """Las dos suposiciones del diseño, verificadas contra el código instalado."""

    def test_setup_model_respeta_un_modulo_ya_construido(self) -> None:
        """``BaseTrainer.setup_model`` devuelve pronto si el modelo ya existe."""
        import ast
        import textwrap

        from ultralytics.engine.trainer import BaseTrainer

        fuente = inspect.getsource(BaseTrainer.setup_model)
        arbol = ast.parse(textwrap.dedent(fuente))
        funcion = arbol.body[0]
        assert isinstance(funcion, ast.FunctionDef)

        # Se descarta el docstring para quedarse con la primera sentencia real.
        cuerpo = list(funcion.body)
        if (
            cuerpo
            and isinstance(cuerpo[0], ast.Expr)
            and isinstance(getattr(cuerpo[0], "value", None), ast.Constant)
            and isinstance(cuerpo[0].value.value, str)
        ):
            cuerpo = cuerpo[1:]

        mensaje = (
            "BaseTrainer.setup_model ya NO devuelve pronto cuando el modelo es un "
            "nn.Module ya construido.\n"
            "El Trainer del proyecto depende de ese comportamiento para que una "
            "poda sobreviva al reajuste. Si esto cambia, el modelo se "
            "reconstruiría desde el YAML y la poda se perdería EN SILENCIO.\n"
            "Revisa src/tfm_edge/training/trainer.py antes de entrenar nada.\n\n"
            f"Código actual:\n{fuente}"
        )
        assert cuerpo, mensaje
        primera = cuerpo[0]
        assert isinstance(primera, ast.If), mensaje

        # La condición debe ser isinstance(self.model, ...nn.Module).
        condicion = ast.dump(primera.test)
        assert "isinstance" in condicion, mensaje
        assert "'model'" in condicion or "attr='model'" in condicion, mensaje
        assert "Module" in condicion, mensaje

        # Y el cuerpo del if, un return sin valor: devuelve sin tocar el modelo.
        assert any(isinstance(s, ast.Return) and s.value is None for s in primera.body), mensaje

    def test_model_train_si_reconstruye_el_modelo(self) -> None:
        from ultralytics.engine.model import Model

        fuente = inspect.getsource(Model.train)
        assert "self.trainer.model = self.trainer.get_model(" in fuente, (
            "Model.train no reconstruye el modelo con get_model(). "
            "Revisa si el Trainer puede volver a delegar en la API alta."
        )
        assert (
            "cfg=self.model.yaml" in fuente
        ), "Model.train no reconstruye desde el YAML de arquitectura."

    def test_save_model_de_ultralytics_guarda_en_media_precision(self) -> None:
        from ultralytics.engine.trainer import BaseTrainer

        fuente = inspect.getsource(BaseTrainer.save_model)
        assert ".half()" in fuente, (
            "BaseTrainer.save_model no guarda en media precisión. Revisa si la copia FP32 del "
            "proyecto sigue haciendo falta."
        )

    def test_la_reanudacion_depende_del_ckpt_que_setup_model_no_devuelve(self) -> None:
        """El motivo de que este entrenador no pueda reanudar."""
        from ultralytics.engine.trainer import BaseTrainer

        setup = inspect.getsource(BaseTrainer._setup_train)
        assert "ckpt = self.setup_model()" in setup, (
            "_setup_train no toma el checkpoint de setup_model(). "
            "Revisa si la reanudación ha pasado a ser posible."
        )
        assert (
            "self.resume_training(ckpt)" in setup
        ), "_setup_train no alimenta resume_training con ese checkpoint."

        reanudar = inspect.getsource(BaseTrainer.resume_training)
        assert "if ckpt is None" in reanudar, (
            "resume_training no sale pronto con ckpt=None. "
            "Puede que ahora se pueda reanudar conservando el modelo."
        )

    def test_setup_model_devuelve_none_con_un_modelo_ya_asignado(self) -> None:
        """Lo anterior, pero ejecutado en lugar de leído."""
        from ultralytics.models.yolo.detect import DetectionTrainer

        from tfm_edge.common.paths import get_project_paths

        # Con el dataset del proyecto y no con 'coco8.yaml', que se descarga al construir el
        # entrenador.
        datos = get_project_paths().data_raw / "sh17_ultralytics.yaml"
        if not datos.is_file():
            pytest.skip(
                f"No existe '{datos}'. Prepara el dataset con scripts/02_prepare_dataset.py."
            )

        entrenador = DetectionTrainer(
            overrides={"model": "yolov8n.yaml", "data": str(datos), "epochs": 1}
        )
        entrenador.model = torch.nn.Conv2d(3, 4, 3)

        assert (
            entrenador.setup_model() is None
        ), "setup_model devuelve un checkpoint con el modelo preasignado: podría reanudarse."


class TestLaReanudacionFallaEnVezDeEmpezarDeCero:
    """Pedir reanudar cuando hay algo que reanudar es un error."""

    @pytest.fixture
    def detector(self) -> Detector:
        """Construye un yolov8n de prueba."""
        return Detector.from_architecture(size="n", num_classes=17, pretrained=False)

    def test_falla_si_hay_un_last_pt_de_una_ejecucion_anterior(
        self, detector: Detector, tmp_path: Path
    ) -> None:
        pesos = tmp_path / "train" / "weights"
        pesos.mkdir(parents=True)
        (pesos / "last.pt").touch()

        entrenador = Trainer(
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
            resume_if_possible=True,
        )
        with pytest.raises(TrainerError, match="no puede reanudar|NO"):
            entrenador.train(run_name="train")

    def test_sin_ejecucion_anterior_no_estorba(self, detector: Detector, tmp_path: Path) -> None:
        """La guarda no puede impedir la primera ejecución."""
        entrenador = Trainer(
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
            resume_if_possible=True,
        )
        # No llega a entrenar porque el YAML de datos no existe, pero pasa la guarda: el error no
        # puede ser el de reanudación.
        with pytest.raises(Exception) as excinfo:
            entrenador.train(run_name="train")
        assert "reanudar" not in str(excinfo.value)

    def test_con_no_resume_no_se_comprueba_nada(self, detector: Detector, tmp_path: Path) -> None:
        pesos = tmp_path / "train" / "weights"
        pesos.mkdir(parents=True)
        (pesos / "last.pt").touch()

        entrenador = Trainer(
            detector=detector,
            data_config=tmp_path / "data.yaml",
            output_dir=tmp_path,
            resume_if_possible=False,
        )
        with pytest.raises(Exception) as excinfo:
            entrenador.train(run_name="train")
        assert "reanudar" not in str(excinfo.value)


class TestTrainerConfiguracion:
    """Configuración del Trainer, sin llegar a entrenar."""

    @pytest.fixture
    def detector(self) -> Detector:
        """Construye un yolov8n de prueba."""
        return Detector.from_architecture(size="n", num_classes=17, pretrained=False)

    def test_lote_efectivo(self, detector: Detector, tmp_path: Path) -> None:
        """El lote efectivo es el producto del lote por la acumulación."""
        t = Trainer(
            detector,
            tmp_path / "data.yaml",
            tmp_path,
            batch_size=4,
            gradient_accumulation_steps=8,
        )
        assert t.effective_batch_size == 32

    def test_rechaza_acumulacion_invalida(self, detector: Detector, tmp_path: Path) -> None:
        """Una acumulación menor que 1 no tiene sentido."""
        with pytest.raises(TrainerError, match="gradient_accumulation_steps"):
            Trainer(detector, tmp_path / "d.yaml", tmp_path, gradient_accumulation_steps=0)

    def test_los_overrides_traducen_la_configuracion(
        self, detector: Detector, tmp_path: Path
    ) -> None:
        """La traducción a argumentos de Ultralytics vive solo en el Trainer."""
        t = Trainer(
            detector,
            tmp_path / "data.yaml",
            tmp_path,
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            epochs=7,
            batch_size=4,
            gradient_accumulation_steps=4,
            amp=False,
        )
        overrides = t._construir_overrides("prueba")
        assert overrides["epochs"] == 7
        assert overrides["batch"] == 4
        assert overrides["amp"] is False
        # nbs es como Ultralytics expresa la acumulación de gradientes.
        assert overrides["nbs"] == 16
        assert overrides["imgsz"] == detector.input_resolution

    def test_la_resolucion_no_esta_hardcodeada(self, tmp_path: Path) -> None:
        """La resolución viaja por configuración, nunca fijada a 640."""
        d = Detector.from_architecture(
            size="n", num_classes=17, input_resolution=416, pretrained=False
        )
        t = Trainer(d, tmp_path / "data.yaml", tmp_path, seed=42)
        assert t._construir_overrides("p")["imgsz"] == 416


class TestVerificacionDeEstructura:
    """La poda debe sobrevivir al entrenamiento, y se comprueba."""

    def test_detecta_un_cambio_en_el_numero_de_parametros(self, tmp_path: Path) -> None:
        """Si el modelo se reconstruye, el número de parámetros cambia."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        t = Trainer(detector, tmp_path / "data.yaml", tmp_path)

        canales = detector.channels_per_conv()
        params = detector.count_parameters()

        # Se simula la reconstrucción: otro modelo, con los canales del YAML.
        t.detector = Detector.from_architecture(size="s", num_classes=17, pretrained=False)
        with pytest.raises(TrainerError, match="número de parámetros ha cambiado"):
            t._verificar_estructura_conservada(canales, params)

    def test_acepta_una_estructura_intacta(self, tmp_path: Path) -> None:
        """Si nada cambia, la comprobación pasa sin ruido."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        t = Trainer(detector, tmp_path / "data.yaml", tmp_path)
        t._verificar_estructura_conservada(
            detector.channels_per_conv(), detector.count_parameters()
        )

    def test_detecta_canales_restaurados_con_igual_numero_de_parametros(
        self, tmp_path: Path
    ) -> None:
        """Un cambio de canales se detecta aunque el total de parámetros no varíe."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        t = Trainer(detector, tmp_path / "data.yaml", tmp_path)
        canales = dict(detector.channels_per_conv())
        primera = next(iter(canales))
        canales[primera] = (canales[primera][0], canales[primera][1] + 8)
        with pytest.raises(TrainerError, match="cambiado de canales"):
            t._verificar_estructura_conservada(canales, detector.count_parameters())


class TestPrecisionDeLosCheckpoints:
    """Los checkpoints del estudio están en FP32, comprobado por dtype."""

    def test_verifica_un_checkpoint_fp32(self, tmp_path: Path) -> None:
        """Un checkpoint correcto pasa la verificación."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        ruta = detector.save(tmp_path / "ok.pt")
        informe = verify_checkpoint_is_fp32(ruta)
        assert informe["dtypes"]["model"] == {
            "torch.float32": len(list(detector.model.named_parameters()))
        }

    def test_rechaza_un_checkpoint_en_media_precision(self, tmp_path: Path) -> None:
        """Un checkpoint en FP16 se rechaza con un mensaje explicativo."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        ruta = tmp_path / "fp16.pt"
        torch.save({"model": detector.model.half()}, ruta)
        with pytest.raises(PrecisionError, match="no está en FP32"):
            verify_checkpoint_is_fp32(ruta)

    def test_checkpoint_inexistente(self, tmp_path: Path) -> None:
        """Verificar un fichero que no existe falla con un mensaje claro."""
        with pytest.raises(TrainerError, match="No existe el checkpoint"):
            verify_checkpoint_is_fp32(tmp_path / "no_existe.pt")

    def test_el_modelo_del_estudio_esta_declarado(self) -> None:
        """Está declarado cuál de los dos modelos es el del estudio."""
        assert STUDY_MODEL in {"ema", "model"}


# La pérdida en caché.


def _lote_minimo(lado: int = 128) -> dict:
    """Construye el lote más pequeño con el que Ultralytics calcula la pérdida."""
    return {
        "img": torch.rand(2, 3, lado, lado),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[1.0], [2.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
    }


def _pegar_la_perdida(detector: Detector) -> None:
    """Hace que el modelo cachee su pérdida, igual que en el primer lote real."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    modelo = detector.model
    modelo.args = get_cfg(DEFAULT_CFG)
    modelo.train()
    modelo.loss(_lote_minimo())
    assert getattr(modelo, "criterion", None) is not None


class TestLaPerdidaCacheadaNoViajaConElModelo:
    """La función de pérdida se queda en el entrenamiento que la construyó."""

    def test_el_to_del_modelo_no_alcanza_a_la_perdida(self) -> None:
        """La reproducción del fallo, sin necesidad de GPU."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        _pegar_la_perdida(detector)
        modelo = detector.model

        modelo.criterion.proj = modelo.criterion.proj.to("meta")
        modelo.to("cpu")
        assert modelo.criterion.proj.device.type == "meta", "el .to() no arrastra la pérdida"

        with pytest.raises(RuntimeError, match="device"):
            modelo.loss(_lote_minimo())

        assert drop_cached_loss(modelo) is True
        modelo.loss(_lote_minimo())  # se reconstruye sobre el dispositivo correcto
        assert modelo.criterion.proj.device == next(modelo.parameters()).device

    def test_soltar_una_perdida_que_no_hay_no_falla(self) -> None:
        """Un modelo recién construido no tiene nada que soltar."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        assert drop_cached_loss(detector.model) is False

    def test_save_no_la_serializa(self, tmp_path: Path) -> None:
        """El checkpoint que escribe el proyecto no se lleva la pérdida dentro."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        _pegar_la_perdida(detector)
        ruta = detector.save(tmp_path / "sin_perdida.pt")

        contenido = torch.load(ruta, map_location="cpu", weights_only=False)
        assert getattr(contenido["model"], "criterion", None) is None

    def test_load_suelta_la_que_traiga_el_fichero(self, tmp_path: Path) -> None:
        """Los checkpoints ya escritos también quedan cubiertos."""
        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        _pegar_la_perdida(detector)
        ruta = tmp_path / "heredada.pt"
        # Con torch.save y no con Detector.save, para que el fichero la lleve, como los checkpoints
        # existentes.
        torch.save({"model": detector.model, "tfm_edge": {"size": "n", "num_classes": 17}}, ruta)
        assert getattr(torch.load(ruta, weights_only=False)["model"], "criterion", None) is not None

        recargado = Detector.load(ruta)
        assert getattr(recargado.model, "criterion", None) is None

    def test_el_trainer_no_entrega_el_modelo_con_la_perdida_pegada(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ultralytics.models.yolo.detect as detect_module

        capturado: dict = {}

        class _TrainerFalso:
            """Se queda con el modelo que le entregan y para ahí."""

            def __init__(self, overrides: dict) -> None:
                self.model = None
                self.overrides = overrides

            def add_callback(self, evento: str, funcion: object) -> None:
                pass

            def final_eval(self) -> None:
                pass

            def train(self) -> None:
                capturado["model"] = self.model
                raise RuntimeError("parada deliberada del test")

        monkeypatch.setattr(detect_module, "DetectionTrainer", _TrainerFalso)

        detector = Detector.from_architecture(size="n", num_classes=17, pretrained=False)
        _pegar_la_perdida(detector)
        entrenador = Trainer(
            detector=detector,
            data_config=tmp_path / "datos.yaml",
            output_dir=tmp_path / "salida",
            resume_if_possible=False,
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
        )
        with pytest.raises(TrainerError, match="parada deliberada"):
            entrenador.train()

        assert capturado["model"] is detector.model
        assert getattr(capturado["model"], "criterion", None) is None
