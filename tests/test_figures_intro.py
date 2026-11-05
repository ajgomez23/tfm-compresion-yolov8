"""Las figuras del capítulo 1, salvo el resultado principal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tfm_edge.common.registry import ExperimentRegistry
from tfm_edge.reporting import figures_intro as fi
from tfm_edge.reporting import style

RAIZ = Path(__file__).resolve().parent.parent
COCO = RAIZ / "data" / "processed" / "sh17_coco.json"
IMAGENES = RAIZ / "data" / "raw" / "images"
ATRIBUCIONES = RAIZ / "docs" / "SH17_ATTRIBUTIONS.md"


def _experimentos() -> list[dict[str, str]]:
    """Filas válidas de ``experiments.csv`` de la campaña real."""
    return ExperimentRegistry(RAIZ / "results" / "registry").valid_rows("experiments")[0]


@pytest.fixture
def capturar(monkeypatch, tmp_path: Path) -> dict[str, Any]:
    """Guarda cada figura en ``tmp_path`` y se queda con ella antes de cerrarla."""
    original = style.save_figure
    capturadas: dict[str, Any] = {}

    def guardar(figura, output_dir, capitulo, apartado, descripcion, close=True):
        rutas = original(figura, tmp_path, capitulo, apartado, descripcion, close=False)
        capturadas[descripcion] = {"figure": figura, "files": rutas, "placement": (capitulo, apartado)}
        return rutas

    monkeypatch.setattr(fi.style, "save_figure", guardar)
    yield capturadas
    import matplotlib.pyplot as plt

    plt.close("all")


def _medir(figura: Any) -> tuple[float, list[Any], Any]:
    """Cuerpo mínimo de texto, textos visibles y caja del contenido de una figura."""
    from matplotlib.text import Text

    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    textos = [t for t in figura.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(t.get_fontsize() for t in textos), textos, figura.get_tightbbox(dibujo)


def _ancho_png_cm(ruta: Path) -> float:
    """Ancho real del PNG guardado, en centímetros."""
    from PIL import Image

    with Image.open(ruta) as imagen:
        return imagen.size[0] / imagen.info.get("dpi", (300, 300))[0] * 2.54


class TestLoQueLosDiagramasAnuncian:
    """El flujo y las técnicas se contrastan con el registro."""

    def test_el_registro_real_tiene_los_cuatro_tamanos_y_las_cinco_tecnicas(self) -> None:
        fi.verify_study_scope(_experimentos())

    @pytest.mark.parametrize("tamano", fi.STUDY_MODEL_SIZES)
    def test_sin_el_baseline_de_un_tamano_el_flujo_no_sale(
        self, tamano: str, tmp_path: Path
    ) -> None:
        filas = [
            f
            for f in _experimentos()
            if not (f["technique"] == "baseline" and f["model_size"] == tamano)
        ]
        with pytest.raises(fi.IntroFigureError, match="baselines"):
            fi.plot_study_flow(tmp_path, filas)

    @pytest.mark.parametrize("tecnica", fi.STUDY_TECHNIQUES)
    def test_sin_una_tecnica_el_flujo_no_sale(self, tecnica: str, tmp_path: Path) -> None:
        filas = [f for f in _experimentos() if f["technique"] != tecnica]
        with pytest.raises(fi.IntroFigureError, match="técnicas"):
            fi.plot_study_flow(tmp_path, filas)


class TestDiagramas:
    """Los tres diagramas: 16 cm, 7 pt y sin textos superpuestos."""

    @pytest.mark.parametrize(
        ("nombre", "dibujar"),
        [
            ("problema_epi_en_el_borde", lambda d: fi.plot_problem(d)),
            ("tecnicas_de_compresion", lambda d: fi.plot_techniques(d)),
            ("flujo_del_estudio", lambda d: fi.plot_study_flow(d, _experimentos())),
        ],
    )
    def test_caben_en_16_cm_a_7_pt_sin_textos_pisados(
        self, capturar, tmp_path: Path, nombre: str, dibujar
    ) -> None:
        dibujar(tmp_path)
        figura = capturar[nombre]["figure"]
        minimo, textos, contenido = _medir(figura)
        assert capturar[nombre]["placement"] == (1, fi.PLACEMENT[nombre])
        assert figura.get_figwidth() == pytest.approx(16.0 / 2.54)
        assert contenido.x0 >= 0.0 and contenido.x1 <= figura.get_figwidth() + 1e-6
        assert _ancho_png_cm(capturar[nombre]["files"]["png"]) <= 16.0 + 1e-6
        assert minimo >= 7.0
        dibujo = figura.canvas.get_renderer()
        cajas = [(t.get_text(), t.get_window_extent(dibujo)) for t in textos]
        for i, (texto_a, a) in enumerate(cajas):
            for texto_b, b in cajas[i + 1 :]:
                assert not a.overlaps(b), (nombre, texto_a, texto_b)


class TestEspacioDeDecision:
    """El esquema es sintético y elige lo que dice que elige."""

    def test_los_puntos_quedan_bajo_la_frontera(self) -> None:
        puntos, _ = fi.decision_space_points()
        techo = fi.decision_space_ceiling(puntos[:, 0], puntos[:, 1])
        assert (puntos[:, 2] < techo).all()

    def test_la_elegida_es_la_mejor_de_la_frontera_dentro_de_la_restriccion(self) -> None:
        import numpy as np

        _, elegida = fi.decision_space_points()
        limite_l, limite_m = fi.DECISION_SPACE_LIMITS
        malla_l, malla_m = np.meshgrid(
            np.linspace(0.0, limite_l, 50), np.linspace(0.0, limite_m, 50)
        )
        assert elegida[2] == pytest.approx(float(fi.decision_space_ceiling(malla_l, malla_m).max()))
        assert (elegida[0], elegida[1]) == (limite_l, limite_m)

    def test_sin_marcas_en_los_ejes_a_7_pt_y_dentro_de_16_cm(self, capturar, tmp_path) -> None:
        fi.plot_decision_space(tmp_path)
        figura = capturar["espacio_de_decision"]["figure"]
        minimo, _, _ = _medir(figura)
        eje = figura.axes[0]
        assert not list(eje.get_xticks()) and not list(eje.get_yticks())
        assert not list(eje.get_zticks())
        assert minimo >= 7.0
        assert _ancho_png_cm(capturar["espacio_de_decision"]["files"]["png"]) <= 16.0 + 1e-6


class TestMuestraSH17:
    """La muestra: imágenes sin difuminar y la atribución de cada una."""

    def test_el_difuminado_cambia_solo_el_interior_de_la_caja(self) -> None:
        import numpy as np
        from PIL import Image

        azar = np.random.default_rng(0)
        imagen = Image.fromarray(azar.integers(0, 256, (120, 160, 3), dtype=np.uint8))
        borrosa = np.asarray(fi.blur_boxes(imagen, [(40.0, 30.0, 50.0, 40.0)])).astype(float)
        original = np.asarray(imagen).astype(float)
        dentro = (slice(32, 68), slice(42, 88))
        assert borrosa[dentro].std() < 0.5 * original[dentro].std()
        fuera = np.ones(original.shape[:2], dtype=bool)
        fuera[30:71, 40:91] = False
        assert (borrosa[fuera] == original[fuera]).all()

    def test_las_cuatro_imagenes_tienen_atribucion_y_licencia(self) -> None:
        entradas = fi.sample_attributions(fi.load_attributions(ATRIBUCIONES))
        assert [e["image"] for e in entradas] == list(fi.SAMPLE_IMAGES)
        for entrada in entradas:
            assert entrada["photographer"] and entrada["source"].startswith("https://www.pexels.com/")
            assert entrada["license_dataset"] == "CC BY-NC-SA 4.0 (SH17)"

    def test_una_imagen_sin_atribucion_no_entra(self) -> None:
        with pytest.raises(fi.IntroFigureError, match="sin atribución"):
            fi.sample_attributions(fi.load_attributions(ATRIBUCIONES), ["no-existe.jpeg"])

    def test_por_defecto_no_difumina_y_dibuja_la_imagen_original(
        self, capturar, monkeypatch, tmp_path
    ) -> None:
        import numpy as np
        from PIL import Image

        if not (COCO.is_file() and IMAGENES.is_dir()):
            pytest.skip("Sin data/processed/sh17_coco.json o sin data/raw/images en este árbol.")
        assert fi.SAMPLE_BLUR is False
        coco = json.loads(COCO.read_text(encoding="utf-8"))
        llamadas: list[Any] = []
        monkeypatch.setattr(fi, "blur_boxes", lambda *a, **k: llamadas.append(a))

        _, resumen = fi.plot_sh17_sample(coco, IMAGENES, fi.load_attributions(ATRIBUCIONES), tmp_path)

        assert not llamadas
        assert all(e["blurred"] is False and e["blurred_boxes"] == 0 for e in resumen)
        figura = capturar["muestra_sh17"]["figure"]
        for panel, fichero in zip(figura.axes, fi.SAMPLE_IMAGES):
            with Image.open(IMAGENES / fichero) as original:
                original.draft("RGB", (1400, 1400))
                esperada = np.asarray(original.convert("RGB"))
            dibujada = np.asarray(panel.get_images()[0].get_array())
            assert np.array_equal(dibujada, esperada), fichero
        minimo, _, _ = _medir(figura)
        assert minimo >= 7.0
        assert figura.get_figwidth() == pytest.approx(16.0 / 2.54)

    def test_con_difuminado_cubre_todas_las_cajas_de_cabeza_y_cara(
        self, capturar, monkeypatch, tmp_path
    ) -> None:
        if not (COCO.is_file() and IMAGENES.is_dir()):
            pytest.skip("Sin data/processed/sh17_coco.json o sin data/raw/images en este árbol.")
        coco = json.loads(COCO.read_text(encoding="utf-8"))
        nombres = {c["id"]: c["name"] for c in coco["categories"]}
        por_nombre = {i["file_name"]: i for i in coco["images"]}
        esperadas: dict[str, list[tuple[float, ...]]] = {}
        for fichero in fi.SAMPLE_IMAGES:
            info = por_nombre[fichero]
            esperadas[fichero] = sorted(
                tuple(v / info["width"] for v in a["bbox"])
                for a in coco["annotations"]
                if a["image_id"] == info["id"] and nombres[a["category_id"]] in fi.BLURRED_CLASSES
            )

        vistas: list[tuple[int, list[tuple[float, ...]]]] = []
        original = fi.blur_boxes

        def espia(imagen, cajas):
            vistas.append((imagen.width, [tuple(c) for c in cajas]))
            return original(imagen, cajas)

        monkeypatch.setattr(fi, "blur_boxes", espia)
        _, resumen = fi.plot_sh17_sample(
            coco, IMAGENES, fi.load_attributions(ATRIBUCIONES), tmp_path, blur=True
        )

        assert len(vistas) == len(fi.SAMPLE_IMAGES)
        for fichero, (ancho, cajas), entrada in zip(fi.SAMPLE_IMAGES, vistas, resumen):
            normalizadas = sorted(tuple(v / ancho for v in c) for c in cajas)
            assert normalizadas == pytest.approx(esperadas[fichero]), fichero
            assert entrada["blurred"] is True
            assert entrada["blurred_boxes"] == len(esperadas[fichero]) > 0

    def test_el_manifiesto_versionado_trae_la_atribucion(self) -> None:
        ruta = RAIZ / "results" / "raw" / "intro" / "intro_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 35 no se ha ejecutado en este árbol.")
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        assert set(manifiesto["figures"]) | set(manifiesto["skipped"]) == set(fi.PLACEMENT)
        for nombre, entrada in manifiesto["figures"].items():
            for formato in style.FIGURE_FORMATS:
                assert (RAIZ / entrada["files"][formato]).is_file(), nombre
        muestra = manifiesto["figures"].get("muestra_sh17")
        if muestra is not None:
            assert [i["image"] for i in muestra["images"]] == list(fi.SAMPLE_IMAGES)
            assert all(i["photographer"] for i in muestra["images"])
            # El manifiesto dice lo que tiene la figura versionada: hoy, sin difuminar.
            assert all(i["blurred"] is fi.SAMPLE_BLUR for i in muestra["images"])
