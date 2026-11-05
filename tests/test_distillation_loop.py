"""Test de integración del bucle de destilación (9.4.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.common.config import DistillationConfig  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.compression.distillation import DistillationTrainer  # noqa: E402
from tfm_edge.compression.distillation.trainer import DistillationError  # noqa: E402
from tfm_edge.data.smoke_subset import prepare_smoke_subset  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402

pytestmark = [
    pytest.mark.requires_torch,
    pytest.mark.requires_weights,
    pytest.mark.requires_dataset,
    pytest.mark.slow,
]

#: Resolución reducida: lo que se comprueba es el mecanismo, no la calidad.
RESOLUCION = 320
EPOCAS = 2
IMAGENES_TRAIN = 8
IMAGENES_VAL = 4
LOTE = 2
PARES = [
    {"teacher": "model.15", "student": "model.15"},
    {"teacher": "model.18", "student": "model.18"},
    {"teacher": "model.21", "student": "model.21"},
]


def _ruta_baseline(size: str) -> Path:
    """Devuelve el checkpoint de un baseline, u omite el test si no está."""
    ruta = get_project_paths().models_baselines / f"{size}_baseline_default" / "best.pt"
    if not ruta.is_file():
        pytest.skip(f"No existe '{ruta}'. Entrena con scripts/04_train_baselines.py.")
    return ruta


def _dataset_reducido(destino: Path) -> Path:
    """Prepara un subconjunto mínimo del dataset."""
    rutas = get_project_paths()
    datos = rutas.data / "resized_640" / "sh17_ultralytics.yaml"
    if not datos.is_file():
        pytest.skip(f"No existe '{datos}'. Prepara el dataset con scripts/02_prepare_data.py.")
    yaml_reducido, _ = prepare_smoke_subset(
        datos, rutas.data / "splits", "train", IMAGENES_TRAIN, destino
    )
    prepare_smoke_subset(datos, rutas.data / "splits", "val", IMAGENES_VAL, destino)
    return yaml_reducido


def _config(**extra) -> DistillationConfig:
    """Configuración de destilación para el test."""
    base = {
        "teacher_size": "l",
        "kd_type": "combined",
        "epochs": EPOCAS,
        "learning_rate": 0.001,
        "feature_pairs": PARES,
    }
    base.update(extra)
    return DistillationConfig(**base)


@pytest.fixture(scope="module")
def destilado(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Ejecuta una destilación corta y devuelve todo lo que hay que mirar."""
    trabajo = tmp_path_factory.mktemp("destilacion")
    yaml_reducido = _dataset_reducido(trabajo / "ds")
    alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
    maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)

    destilador = DistillationTrainer(
        seed=42,  # explícita: el Trainer no tiene semilla por defecto
        student=alumno,
        teacher=maestro,
        data_config=yaml_reducido,
        output_dir=trabajo / "run",
        config=_config(),
        resolution=RESOLUCION,
        batch_size=LOTE,
        workers=0,
        device="cpu",
        patience=10,
        extra_args={"val": True, "plots": False, "save_period": -1},
    )
    resultado = destilador.train(run_name="test")
    return {"resultado": resultado, "alumno": alumno, "maestro": maestro}


@pytest.fixture(scope="module")
def control(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Ejecuta el control sin destilar (alpha=0) por el mismo camino."""
    trabajo = tmp_path_factory.mktemp("control")
    yaml_reducido = _dataset_reducido(trabajo / "ds")
    alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
    maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)

    destilador = DistillationTrainer(
        seed=42,  # explícita: el Trainer no tiene semilla por defecto
        student=alumno,
        teacher=maestro,
        data_config=yaml_reducido,
        output_dir=trabajo / "run",
        config=_config(alpha=0.0),
        resolution=RESOLUCION,
        batch_size=LOTE,
        workers=0,
        device="cpu",
        patience=10,
        extra_args={"val": True, "plots": False, "save_period": -1},
    )
    return {"resultado": destilador.train(run_name="control"), "alumno": alumno}


class TestElControlSinDestilar:
    """Lo que hace que el control valga como control, comprobado ejecutándolo."""

    def test_el_gradiente_a_los_adaptadores_es_cero_en_todos_los_pasos(self, control: dict) -> None:
        """Si llegara gradiente, el control no estaría aislado de la destilación."""
        normas = control["resultado"].adapter_grad_norms
        assert normas, "no se ha registrado ninguna norma de gradiente"
        assert control["resultado"].adapter_grad_was_always_zero, normas

    def test_los_adaptadores_no_se_han_movido(self, control: dict) -> None:
        """Adam con gradiente nulo no mueve nada; si se mueven, algo más los entrena."""
        assert not control["resultado"].adapters_learned

    def test_el_total_optimizado_es_exactamente_la_deteccion(self, control: dict) -> None:
        """La comprobación de que alpha=0 es un afinado normal, sobre datos reales."""
        for fila in control["resultado"].epoch_components:
            assert fila["total"] == pytest.approx(fila["detection"], rel=1e-12)

    def test_el_desglose_registra_la_destilacion_que_NO_se_aplico(self, control: dict) -> None:
        for fila in control["resultado"].epoch_components:
            assert fila["response_cls"] > 0.0
            assert fila["response_box"] > 0.0
            assert fila["features"] > 0.0

    def test_el_checkpoint_del_control_existe(self, control: dict) -> None:
        assert control["resultado"].training.best_checkpoint.is_file()


class TestElBucleDeDestilacion:
    """Lo que solo se ve ejecutando el bucle entero."""

    def test_el_criterio_se_llama_en_todos_los_pasos(self, destilado: dict) -> None:
        """Instalado antes de ``train()``, el entrenador lo soltaría."""
        esperados = (IMAGENES_TRAIN // LOTE) * EPOCAS
        assert destilado["resultado"].steps == esperados

    def test_hay_un_desglose_por_epoca(self, destilado: dict) -> None:
        """El registro desglosado que pide la sección 9.4.4."""
        resultado = destilado["resultado"]
        assert len(resultado.epoch_components) == EPOCAS
        assert resultado.components_csv is not None
        lineas = resultado.components_csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(lineas) == EPOCAS + 1

    def test_los_tres_terminos_son_no_nulos(self, destilado: dict) -> None:
        """Un término a cero significaría que esa mitad no está haciendo nada."""
        primera = destilado["resultado"].epoch_components[0]
        assert primera["response_cls"] > 0.0
        assert primera["response_box"] > 0.0
        assert primera["features"] > 0.0
        assert primera["marked_anchors"] > 0.0

    def test_el_total_registrado_es_el_que_se_optimiza(self, destilado: dict) -> None:
        fila = destilado["resultado"].epoch_components[0]
        alfa = _config().alpha
        esperado = (1.0 - alfa) * fila["detection"] + alfa * LOTE * (
            fila["response_cls"] + fila["response_box"] + fila["features"]
        )
        assert fila["total"] == pytest.approx(esperado, rel=1e-3)

    def test_los_adaptadores_han_aprendido(self, destilado: dict) -> None:
        """El fallo silencioso que motiva el optimizador propio."""
        resultado = destilado["resultado"]
        assert resultado.adapters_learned
        assert set(resultado.adapter_initial_norms) == {"model.15", "model.18", "model.21"}

    def test_el_gradiente_llega_a_los_adaptadores(self, destilado: dict) -> None:
        """La telemetría que delataría un término de características desconectado."""
        assert destilado["resultado"].epoch_components[0]["adapter_grad_norm"] > 0.0

    def test_no_quedan_hooks_colgados(self, destilado: dict) -> None:
        """Dejarlos es una fuga que en un entrenamiento largo agota la memoria."""
        for nombre in ("model.15", "model.18", "model.21"):
            assert not destilado["alumno"].get_module(nombre)._forward_hooks
            assert not destilado["maestro"].get_module(nombre)._forward_hooks

    def test_el_maestro_sigue_congelado(self, destilado: dict) -> None:
        """Los tres invariantes, comprobados también al terminar."""
        maestro = destilado["maestro"]
        assert not maestro.model.training
        assert not any(p.requires_grad for p in maestro.model.parameters())

    def test_el_checkpoint_se_escribe_y_no_lleva_dentro_ni_maestro_ni_hooks(
        self, destilado: dict
    ) -> None:
        """Sin soltar los hooks de la copia, este fichero no llega a existir."""
        ruta = destilado["resultado"].training.best_checkpoint
        assert ruta.is_file()
        contenido = torch.load(ruta, map_location="cpu", weights_only=False)
        modelo = contenido.get("ema") or contenido.get("model")
        assert modelo is not None
        assert sum(len(m._forward_hooks) for m in modelo.modules()) == 0
        criterio = getattr(modelo, "criterion", None)
        assert criterio is None or getattr(criterio, "teacher", None) is None

    def test_los_adaptadores_se_descartan(self, destilado: dict) -> None:
        """No forman parte del alumno desplegado (9.4.2)."""
        assert destilado["resultado"].adapter_final_norms
        # El alumno no ha ganado ningún submódulo por el camino.
        assert not any("proyecciones" in n for n, _ in destilado["alumno"].model.named_modules())


class TestLaGuardaDeLosAdaptadores:
    """Verificación por mutación de la guarda que protege del fallo silencioso."""

    def test_sin_avanzar_el_optimizador_la_destilacion_falla(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yaml_reducido = _dataset_reducido(tmp_path / "ds")
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
        destilador = DistillationTrainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            student=alumno,
            teacher=maestro,
            data_config=yaml_reducido,
            output_dir=tmp_path / "run",
            config=_config(epochs=1),
            resolution=RESOLUCION,
            batch_size=LOTE,
            workers=0,
            device="cpu",
            patience=10,
            extra_args={"val": False, "plots": False, "save_period": -1},
        )
        monkeypatch.setattr(DistillationTrainer, "_paso_de_adaptadores", lambda self, trainer: None)
        with pytest.raises(DistillationError, match="no han cambiado"):
            destilador.train(run_name="mutante")


class TestGuardasDelEntrenador:
    """Las guardas de invariante, cada una con su control negativo."""

    def test_amp_es_un_error_explicito(self, tmp_path: Path) -> None:
        """La destilación corre sin AMP, y pedirlo no puede ignorarse en silencio."""
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
        with pytest.raises(DistillationError, match="AMP"):
            DistillationTrainer(
                seed=42,  # explícita: el Trainer no tiene semilla por defecto
                student=alumno,
                teacher=maestro,
                data_config=tmp_path / "no-importa.yaml",
                output_dir=tmp_path / "run",
                config=_config(),
                amp=True,
            )

    def test_control_sin_amp_se_construye(self, tmp_path: Path) -> None:
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
        destilador = DistillationTrainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            student=alumno,
            teacher=maestro,
            data_config=tmp_path / "no-importa.yaml",
            output_dir=tmp_path / "run",
            config=_config(),
        )
        assert destilador.loss.temperature == 2.0

    def test_un_maestro_en_modo_entrenamiento_es_un_error(self, tmp_path: Path) -> None:
        """Actualizaría sus BatchNorm con los lotes del alumno."""
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
        destilador = DistillationTrainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            student=alumno,
            teacher=maestro,
            data_config=tmp_path / "no-importa.yaml",
            output_dir=tmp_path / "run",
            config=_config(),
        )
        maestro.train()
        with pytest.raises(DistillationError, match="modo entrenamiento"):
            destilador._verificar_maestro(torch.device("cpu"))

    def test_un_maestro_con_gradiente_es_un_error(self, tmp_path: Path) -> None:
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
        destilador = DistillationTrainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            student=alumno,
            teacher=maestro,
            data_config=tmp_path / "no-importa.yaml",
            output_dir=tmp_path / "run",
            config=_config(),
        )
        maestro.eval()
        for parametro in maestro.model.parameters():
            parametro.requires_grad_(True)
        with pytest.raises(DistillationError, match="gradiente activo"):
            destilador._verificar_maestro(torch.device("cpu"))

    def test_control_un_maestro_congelado_pasa_las_tres_guardas(self, tmp_path: Path) -> None:
        """Control negativo de las dos anteriores."""
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        maestro = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
        destilador = DistillationTrainer(
            seed=42,  # explícita: el Trainer no tiene semilla por defecto
            student=alumno,
            teacher=maestro,
            data_config=tmp_path / "no-importa.yaml",
            output_dir=tmp_path / "run",
            config=_config(),
        )
        destilador._congelar_maestro(torch.device("cpu"))
        destilador._verificar_maestro(torch.device("cpu"))

    def test_maestro_y_alumno_con_cabezas_distintas_es_un_error(self, tmp_path: Path) -> None:
        """Compararía distribuciones sobre soportes distintos."""
        alumno = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        otro = Detector.from_architecture(size="s", num_classes=5, pretrained=False)
        with pytest.raises(DistillationError, match="geometría"):
            DistillationTrainer(
                seed=42,  # explícita: el Trainer no tiene semilla por defecto
                student=alumno,
                teacher=otro,
                data_config=tmp_path / "no-importa.yaml",
                output_dir=tmp_path / "run",
                config=_config(),
            )
