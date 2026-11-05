"""Tests de guarda: el tamaño de un modelo sale del modelo, no de un valor por defecto."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.models.detector import (  # noqa: E402
    VALID_SIZES,
    Detector,
    DetectorError,
    infer_model_size,
)

pytestmark = pytest.mark.requires_torch

#: Tamaños con baseline entrenado y su número de parámetros medido con ``count_parameters``.
BASELINES_ESPERADOS = {
    "n": 3_014_163,
    "s": 11_142_179,
    "m": 25_866_163,
    "l": 43_642_947,
}


@pytest.fixture(scope="module")
def detector_n() -> Detector:
    """Construye un yolov8n desde su YAML, sin descargar nada."""
    return Detector.from_architecture(size="n", num_classes=17, pretrained=False)


def _guardar_a_mano(
    modelo: torch.nn.Module, destino: Path, metadatos: dict[str, object] | None = None
) -> Path:
    """Escribe un checkpoint con los metadatos exactos que pide el test."""
    contenido: dict[str, object] = {"model": modelo}
    if metadatos is not None:
        contenido["tfm_edge"] = metadatos
    torch.save(contenido, destino)
    return destino


class TestDeduccionDesdeLaArquitectura:
    """``infer_model_size`` lee el modelo, y dice ``None`` cuando no puede."""

    @pytest.mark.parametrize("size", ["n", "s"])
    def test_deduce_la_escala_de_un_modelo_recien_construido(self, size: str) -> None:
        """Un modelo construido desde ``yolov8{size}.yaml`` declara su escala."""
        detector = Detector.from_architecture(size=size, num_classes=17, pretrained=False)
        assert infer_model_size(detector.model) == size

    def test_sin_ninguna_pista_devuelve_none(self, detector_n: Detector) -> None:
        """Un modelo sin ``yaml`` utilizable no se adivina."""
        modelo = detector_n.model
        original = modelo.yaml
        try:
            modelo.yaml = {}
            assert infer_model_size(modelo) is None
        finally:
            modelo.yaml = original

    def test_deduce_por_los_multiplicadores(self, detector_n: Detector) -> None:
        """Sin ``scale`` ni ``yaml_file``, sirven ``depth`` y ``width``."""
        modelo = detector_n.model
        original = modelo.yaml
        try:
            modelo.yaml = {"depth_multiple": 0.33, "width_multiple": 0.25}
            assert infer_model_size(modelo) == "n"
        finally:
            modelo.yaml = original

    def test_unos_multiplicadores_desconocidos_no_se_redondean(self, detector_n: Detector) -> None:
        modelo = detector_n.model
        original = modelo.yaml
        try:
            modelo.yaml = {"depth_multiple": 0.42, "width_multiple": 0.31}
            assert infer_model_size(modelo) is None
        finally:
            modelo.yaml = original


class TestCargaDeCheckpoints:
    """``Detector.load`` resuelve el tamaño o se niega, pero no lo supone."""

    def test_un_checkpoint_sin_metadatos_se_carga_con_su_tamano_real(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        ruta = _guardar_a_mano(
            detector_n.model, tmp_path / "sin_size.pt", metadatos={"saved_dtype": "float32"}
        )
        assert Detector.load(ruta).size == "n"

    def test_un_checkpoint_sin_bloque_de_metadatos_tambien(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """Ni siquiera hace falta que haya bloque ``tfm_edge``."""
        ruta = _guardar_a_mano(detector_n.model, tmp_path / "pelado.pt")
        assert Detector.load(ruta).size == "n"

    def test_una_etiqueta_que_contradice_a_la_arquitectura_se_rechaza(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        ruta = _guardar_a_mano(
            detector_n.model, tmp_path / "mal_etiquetado.pt", metadatos={"size": "s"}
        )
        with pytest.raises(DetectorError, match="arquitectura es la de yolov8n"):
            Detector.load(ruta)

    def test_un_size_declarado_que_no_coincide_se_rechaza(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """Lo mismo cuando la afirmación llega por el argumento."""
        ruta = _guardar_a_mano(detector_n.model, tmp_path / "modelo.pt")
        with pytest.raises(DetectorError, match="arquitectura es la de yolov8n"):
            Detector.load(ruta, size="l")

    def test_un_checkpoint_mudo_falla_en_vez_de_suponer(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """Sin pista ninguna, cargar es un error explícito."""
        modelo = detector_n.model
        original = modelo.yaml
        try:
            modelo.yaml = {}
            ruta = _guardar_a_mano(modelo, tmp_path / "mudo.pt")
        finally:
            modelo.yaml = original
        with pytest.raises(DetectorError, match="No se puede saber de qué tamaño"):
            Detector.load(ruta)

    def test_un_checkpoint_mudo_se_puede_cargar_declarando_el_tamano(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """El argumento ``size`` es la salida para un fichero sin metadatos."""
        modelo = detector_n.model
        original = modelo.yaml
        try:
            modelo.yaml = {}
            ruta = _guardar_a_mano(modelo, tmp_path / "mudo.pt")
        finally:
            modelo.yaml = original
        assert Detector.load(ruta, size="n").size == "n"

    def test_una_escala_fuera_del_estudio_se_rechaza(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """``x`` es una escala válida de Ultralytics, pero no de este TFM."""
        assert "x" not in VALID_SIZES
        modelo = detector_n.model
        original = modelo.yaml
        try:
            modelo.yaml = {**original, "scale": "x"}
            ruta = _guardar_a_mano(modelo, tmp_path / "escala_x.pt")
        finally:
            modelo.yaml = original
        with pytest.raises(DetectorError, match="solo trabaja con"):
            Detector.load(ruta)

    def test_ida_y_vuelta_por_save_conserva_el_tamano(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """Un checkpoint escrito por el propio proyecto se recarga igual."""
        destino = detector_n.save(tmp_path / "ida_y_vuelta.pt")
        assert Detector.load(destino).size == detector_n.size == "n"


class TestPesosPreentrenados:
    """``from_pretrained_file`` cumple el ``Raises`` que su docstring prometía."""

    def test_un_tamano_que_no_cuadra_con_el_fichero_se_rechaza(
        self, detector_n: Detector, tmp_path: Path
    ) -> None:
        """Pedir ``size='m'`` sobre un fichero que es ``n`` es un error."""
        ruta = _guardar_a_mano(detector_n.model, tmp_path / "pesos.pt")
        with pytest.raises(DetectorError, match="arquitectura es la de yolov8n"):
            Detector.from_pretrained_file(ruta, size="m", num_classes=17)


@pytest.mark.requires_weights
class TestContraLosCheckpointsReales:
    """Comprobaciones sobre los cuatro checkpoints del estudio; se omiten si no están."""

    @pytest.fixture(scope="class")
    def directorio_baselines(self) -> Path:
        """Devuelve el directorio de baselines, u omite la clase si falta."""
        directorio = get_project_paths().models_baselines
        if not directorio.is_dir():
            pytest.skip(
                f"No existe '{directorio}'. Entrena los baselines con "
                "scripts/04_train_baselines.py."
            )
        return directorio

    @pytest.mark.parametrize("size", sorted(BASELINES_ESPERADOS))
    def test_cada_baseline_declara_su_propio_tamano(
        self, directorio_baselines: Path, size: str
    ) -> None:
        """Carga el baseline de un tamaño y comprueba que dice ser ese."""
        ruta = directorio_baselines / f"{size}_baseline_default" / "best.pt"
        if not ruta.is_file():
            pytest.skip(f"No existe '{ruta}'.")
        detector = Detector.load(ruta)
        assert detector.size == size
        # El recuento de parámetros es la prueba independiente de que el fichero
        # es el que se cree: la etiqueta y la arquitectura tienen que contar lo
        # mismo. Sin esto, cuatro copias del mismo modelo pasarían el test.
        assert detector.count_parameters() == BASELINES_ESPERADOS[size]

    def test_los_cuatro_no_dicen_todos_lo_mismo(self, directorio_baselines: Path) -> None:
        """Los cuatro baselines cargados dan cuatro tamaños distintos."""
        tamanos = []
        for size in sorted(BASELINES_ESPERADOS):
            ruta = directorio_baselines / f"{size}_baseline_default" / "best.pt"
            if not ruta.is_file():
                pytest.skip(f"No existe '{ruta}'.")
            tamanos.append(Detector.load(ruta).size)
        assert sorted(tamanos) == sorted(BASELINES_ESPERADOS)

    @pytest.mark.parametrize("size", ["n", "s"])
    def test_los_pesos_oficiales_se_deducen_por_los_multiplicadores(self, size: str) -> None:
        """Un fichero real que de verdad no trae ``scale`` ni ``yaml_file``."""
        ruta = get_project_paths().data / "weights" / f"yolov8{size}.pt"
        if not ruta.is_file():
            pytest.skip(f"No existe '{ruta}'.")
        contenido = torch.load(ruta, map_location="cpu", weights_only=False)
        modelo = contenido.get("model") or contenido.get("ema")
        assert "scale" not in getattr(modelo, "yaml", {}), (
            "Estos pesos ya traen 'scale': el test ha dejado de ejercitar la "
            "deducción por multiplicadores y hay que buscar otro caso."
        )
        assert infer_model_size(modelo) == size
        assert Detector.load(ruta).size == size
