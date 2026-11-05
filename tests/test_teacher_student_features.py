"""Tests de guarda del emparejamiento maestro-alumno para la destilación."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.models.detector import Detector, DetectorError  # noqa: E402
from tfm_edge.models.surgery import unfreeze_all  # noqa: E402

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_weights]

#: Resolución de las pruebas. Basta con un múltiplo de 32 menor que 640, porque se comprueban
#: proporciones y canales.
RESOLUCION = 320

#: Emparejamiento maestro-alumno medido sobre los checkpoints: salidas P3/P4/P5 del cuello. ``l``
#: satura en 512 canales, así que las razones son 4x, 4x y 2x.
EMPAREJAMIENTO = {
    "model.15": {"maestro": 256, "alumno": 64, "divisor": 8},
    "model.18": {"maestro": 512, "alumno": 128, "divisor": 16},
    "model.21": {"maestro": 512, "alumno": 256, "divisor": 32},
}

#: ``reg_max`` de la cabeza de YOLOv8 y clases del estudio. La cabeza en modo
#: entrenamiento saca ``4 * reg_max + nc`` canales, que es como parte los mapas
#: ``v8DetectionLoss``. La destilación por respuesta tiene que partirlos igual.
REG_MAX = 16
NUM_CLASSES = 17


def _ruta_baseline(size: str) -> Path:
    """Devuelve el checkpoint de un baseline, u omite el test si no está."""
    ruta = get_project_paths().models_baselines / f"{size}_baseline_default" / "best.pt"
    if not ruta.is_file():
        pytest.skip(f"No existe '{ruta}'. Entrena con scripts/04_train_baselines.py.")
    return ruta


@pytest.fixture(scope="module")
def maestro() -> Detector:
    """Carga el maestro de destilación, el ``yolov8l`` entrenado."""
    detector = Detector.load(_ruta_baseline("l"), input_resolution=RESOLUCION)
    detector.eval()
    return detector


@pytest.fixture(scope="module")
def alumno() -> Detector:
    """Carga el alumno de destilación, el ``yolov8n`` entrenado."""
    detector = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
    detector.eval()
    return detector


@pytest.fixture(scope="module")
def entrada() -> "torch.Tensor":
    """Lote de una imagen del tamaño de prueba."""
    return torch.randn(1, 3, RESOLUCION, RESOLUCION)


def _capturar(detector: Detector, capas: list[str], x: "torch.Tensor") -> dict[str, object]:
    """Captura las activaciones de unas capas en un pase sin gradiente."""
    with detector.capture_features(capas) as feats:
        with torch.no_grad():
            detector.forward(x)
        return dict(feats)


class TestElMaestroYElAlumnoSonLosQueSeCreen:
    """Antes de emparejar nada, que los ficheros sean los que dicen ser."""

    def test_los_dos_tienen_la_estructura_esperada(
        self, maestro: Detector, alumno: Detector
    ) -> None:
        """Misma topología de 23 capas en los dos, que es lo que permite emparejar."""
        maestro.validate_structure()
        alumno.validate_structure()
        assert maestro.size == "l"
        assert alumno.size == "n"
        assert len(maestro.layers) == len(alumno.layers)

    def test_el_maestro_es_mucho_mayor_que_el_alumno(
        self, maestro: Detector, alumno: Detector
    ) -> None:
        """La razón entre ambos es lo que da sentido a destilar."""
        assert maestro.count_parameters() > 10 * alumno.count_parameters()


class TestLaTablaDeEmparejamiento:
    """Las tres capas emparejadas, sobre los pesos reales."""

    @pytest.mark.parametrize("capa", sorted(EMPAREJAMIENTO))
    def test_los_canales_son_los_medidos(
        self, maestro: Detector, alumno: Detector, entrada: "torch.Tensor", capa: str
    ) -> None:
        """Cada lado saca los canales con los que se dimensionan las adaptaciones."""
        esperado = EMPAREJAMIENTO[capa]
        del_maestro = _capturar(maestro, [capa], entrada)[capa]
        del_alumno = _capturar(alumno, [capa], entrada)[capa]

        assert del_maestro.shape[1] == esperado["maestro"]
        assert del_alumno.shape[1] == esperado["alumno"]

    @pytest.mark.parametrize("capa", sorted(EMPAREJAMIENTO))
    def test_las_dimensiones_espaciales_coinciden(
        self, maestro: Detector, alumno: Detector, entrada: "torch.Tensor", capa: str
    ) -> None:
        """Es lo que permite adaptar con 1x1 y sin remuestrear."""
        esperado = EMPAREJAMIENTO[capa]
        del_maestro = _capturar(maestro, [capa], entrada)[capa]
        del_alumno = _capturar(alumno, [capa], entrada)[capa]
        lado = RESOLUCION // int(esperado["divisor"])

        assert del_maestro.shape[2:] == del_alumno.shape[2:]
        assert tuple(del_maestro.shape[2:]) == (lado, lado)


class TestElMecanismoDeCapturaSobreElCheckpointReal:
    """``capture_features`` solo estaba probado sobre un modelo sintético."""

    def test_captura_todas_las_capas_del_maestro(
        self, maestro: Detector, entrada: "torch.Tensor"
    ) -> None:
        """Las 23 capas del maestro entrenado, en un solo pase."""
        capas = [f"model.{i}" for i in range(len(maestro.layers))]
        capturadas = _capturar(maestro, capas, entrada)
        assert sorted(capturadas) == sorted(capas)

    def test_los_hooks_se_retiran_y_el_diccionario_se_vacia(
        self, maestro: Detector, entrada: "torch.Tensor"
    ) -> None:
        """Dejar hooks colgados agota la memoria en un entrenamiento largo."""
        modulo = maestro.get_module("model.15")
        assert len(modulo._forward_hooks) == 0

        with maestro.capture_features(["model.15"]) as feats:
            assert len(modulo._forward_hooks) == 1
            with torch.no_grad():
                maestro.forward(entrada)
            assert len(feats) == 1

        assert len(modulo._forward_hooks) == 0
        assert len(feats) == 0

    def test_una_capa_inexistente_no_deja_instrumentadas_las_validas(
        self, maestro: Detector
    ) -> None:
        """El fallo ocurre a mitad de instrumentar, y hay que deshacer lo hecho."""
        with pytest.raises(DetectorError):
            with maestro.capture_features(["model.15", "model.999"]):
                pass  # pragma: no cover - no se llega nunca

        assert len(maestro.get_module("model.15")._forward_hooks) == 0


class TestElGradienteNoFluyeHastaQueSeDescongela:
    """El hallazgo del sondeo, con su remedio, en el mismo test."""

    def test_el_checkpoint_llega_congelado(self, alumno: Detector) -> None:
        """Ni un solo parámetro entrenable, tal como sale del disco."""
        parametros = list(alumno.model.parameters())
        assert parametros
        assert not any(p.requires_grad for p in parametros)

    def test_sin_descongelar_la_perdida_por_caracteristicas_no_retropropaga(
        self, entrada: "torch.Tensor"
    ) -> None:
        recien_cargado = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        recien_cargado.train()
        with recien_cargado.capture_features(["model.15"]) as feats:
            recien_cargado.forward(entrada)
            activacion = feats["model.15"]
            assert not activacion.requires_grad
            assert activacion.grad_fn is None
            with pytest.raises(RuntimeError, match="does not require grad"):
                activacion.float().pow(2).mean().backward()

    def test_tras_descongelar_el_gradiente_llega_a_los_pesos(self, entrada: "torch.Tensor") -> None:
        detector = Detector.load(_ruta_baseline("n"), input_resolution=RESOLUCION)
        reactivados = unfreeze_all(detector.model)
        assert reactivados > 0
        detector.train()

        capas = sorted(EMPAREJAMIENTO)
        with detector.capture_features(capas) as feats:
            detector.forward(entrada)
            for nombre in capas:
                assert feats[nombre].grad_fn is not None, nombre
            perdida = sum(t.float().pow(2).mean() for t in feats.values())
            perdida.backward()

        con_gradiente = {n for n, p in detector.model.named_parameters() if p.grad is not None}
        assert "model.0.conv.weight" in con_gradiente, "el troncal tiene que recibir gradiente"
        # La cabeza es posterior a las capas capturadas y una pérdida solo de características no la
        # alcanza; por eso 9.4 la combina con la de detección.
        assert not any(n.startswith("model.22.") for n in con_gradiente)


class TestLaSalidaDeLaCabezaSegunElModo:
    """El contrato que la destilación por respuesta tiene que respetar."""

    def test_en_entrenamiento_son_tres_mapas_de_reg_max_por_cuatro_mas_clases(
        self, maestro: Detector, entrada: "torch.Tensor"
    ) -> None:
        maestro.train()
        try:
            with torch.no_grad():
                salida = maestro.forward(entrada)
        finally:
            maestro.eval()

        assert len(salida) == len(EMPAREJAMIENTO)
        for mapa, divisor in zip(salida, [8, 16, 32]):
            assert mapa.shape[1] == 4 * REG_MAX + NUM_CLASSES
            assert tuple(mapa.shape[2:]) == (RESOLUCION // divisor, RESOLUCION // divisor)

    def test_en_evaluacion_la_salida_es_otra_cosa(
        self, maestro: Detector, entrada: "torch.Tensor"
    ) -> None:
        """En ``eval()`` la cabeza decodifica, y lo decodificado no sirve para destilar."""
        maestro.eval()
        with torch.no_grad():
            salida = maestro.forward(entrada)

        assert isinstance(salida, tuple) and len(salida) == 2
        # (batch, 4 + nc, num_anclas): cajas ya decodificadas, sin la
        # distribución del DFL que la destilación de caja necesita.
        assert salida[0].shape[1] == 4 + NUM_CLASSES
