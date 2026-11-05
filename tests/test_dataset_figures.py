"""Tests de las figuras del capítulo 3 añadidas para los apartados 3.3 y 3.4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tfm_edge.data.eda import (  # noqa: E402
    plot_conditional_cooccurrence,
    plot_split_class_distribution,
    select_letterbox_example,
)
from tfm_edge.data.statistics import compute_conditional_cooccurrence  # noqa: E402
from tfm_edge.reporting.style import SPLIT_STYLES, apply_style, split_style_for  # noqa: E402


def _coco(imagenes: list[list[int]], nombres: list[str]) -> dict[str, Any]:
    """Construye un COCO mínimo a partir de las clases de cada imagen."""
    anotaciones = []
    for indice_imagen, clases in enumerate(imagenes):
        for clase in clases:
            anotaciones.append(
                {
                    "id": len(anotaciones),
                    "image_id": indice_imagen,
                    "category_id": clase,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "area": 100.0,
                }
            )
    return {
        "images": [
            {"id": i, "file_name": f"img_{i}.jpg", "width": 100, "height": 100}
            for i in range(len(imagenes))
        ],
        "annotations": anotaciones,
        "categories": [{"id": i, "name": n} for i, n in enumerate(nombres)],
    }


class TestCoocurrenciaCondicional:
    """La matriz dirigida del apartado 3.4."""

    def test_no_es_simetrica(self) -> None:
        """El caso que motiva la figura, en pequeño."""
        nombres = ["comun", "raro"]
        # Nueve imágenes solo con 'comun' y una con las dos.
        coco = _coco([[0]] * 9 + [[0, 1]], nombres)
        resultado = compute_conditional_cooccurrence(coco, nombres)
        matriz = resultado["matrix"]

        assert matriz["raro"]["comun"] == pytest.approx(1.0)
        assert matriz["comun"]["raro"] == pytest.approx(0.1)
        assert matriz["raro"]["comun"] != matriz["comun"]["raro"]

    def test_pondera_por_instancia_y_no_por_imagen(self) -> None:
        """Una imagen con tres cajas de A aporta tres al denominador de A."""
        nombres = ["a", "b"]
        # Imagen 0: tres 'a' junto a una 'b'. Imagen 1: una 'a' sola.
        coco = _coco([[0, 0, 0, 1], [0]], nombres)
        resultado = compute_conditional_cooccurrence(coco, nombres)

        assert resultado["denominator"]["a"] == 4
        # 3 de las 4 instancias de 'a' comparten imagen con alguna 'b'.
        assert resultado["matrix"]["a"]["b"] == pytest.approx(0.75)
        # Las dos instancias de 'b'... solo hay una, y sí ve a 'a'.
        assert resultado["matrix"]["b"]["a"] == pytest.approx(1.0)

    def test_la_diagonal_vale_uno(self) -> None:
        """Toda instancia de A está en una imagen que contiene A."""
        nombres = ["a", "b"]
        resultado = compute_conditional_cooccurrence(_coco([[0, 1], [0]], nombres), nombres)
        for nombre in nombres:
            assert resultado["matrix"][nombre][nombre] == pytest.approx(1.0)

    def test_clase_sin_instancias_queda_vacia_no_a_cero(self) -> None:
        """Un cero afirmaría «nunca coocurren», y aquí no se ha medido nada."""
        nombres = ["a", "ausente"]
        resultado = compute_conditional_cooccurrence(_coco([[0]], nombres), nombres)
        assert resultado["denominator"]["ausente"] == 0
        assert resultado["matrix"]["ausente"]["a"] is None

    def test_la_definicion_viaja_con_los_numeros(self) -> None:
        """Sin la definición, la matriz no dice en qué dirección se lee."""
        nombres = ["a", "b"]
        resultado = compute_conditional_cooccurrence(_coco([[0, 1]], nombres), nombres)
        assert "INSTANCIAS" in resultado["definition"]
        assert "simetrica" in resultado["definition"]


class TestLetterbox:
    """El envoltorio del letterbox real de Ultralytics."""

    @staticmethod
    def _letterbox() -> Any:
        """Importa el envoltorio, saltando el test si falta Ultralytics."""
        pytest.importorskip("ultralytics", reason="El letterbox real necesita Ultralytics.")
        from tfm_edge.models.detector import letterbox_image

        return letterbox_image

    def test_el_lienzo_sale_cuadrado(self) -> None:
        """Con ``auto=False`` el relleno llega hasta el cuadrado completo."""
        letterbox_image = self._letterbox()
        imagen = np.full((100, 400, 3), 200, dtype=np.uint8)
        lienzo, _ = letterbox_image(imagen, resolution=640)
        assert lienzo.shape == (640, 640, 3)

    def test_la_caja_de_toda_la_imagen_cae_sobre_el_contenido(self) -> None:
        """La geometría declarada coincide con dónde acabaron los píxeles."""
        letterbox_image = self._letterbox()
        alto, ancho = 100, 400
        imagen = np.full((alto, ancho, 3), 200, dtype=np.uint8)
        lienzo, transformacion = letterbox_image(imagen, resolution=640)

        contenido = np.any(lienzo != 114, axis=2)
        filas = np.flatnonzero(contenido.any(axis=1))
        columnas = np.flatnonzero(contenido.any(axis=0))

        x, y, w, h = transformacion.map_box(0, 0, ancho, alto)
        assert x == pytest.approx(columnas[0], abs=1)
        assert y == pytest.approx(filas[0], abs=1)
        assert w == pytest.approx(columnas[-1] - columnas[0] + 1, abs=1)
        assert h == pytest.approx(filas[-1] - filas[0] + 1, abs=1)

    def test_conserva_la_relacion_de_aspecto(self) -> None:
        """Es la afirmación que sostiene la figura del apartado 3.3.3."""
        letterbox_image = self._letterbox()
        imagen = np.full((1625, 3962, 3), 200, dtype=np.uint8)
        _, transformacion = letterbox_image(imagen, resolution=640)
        _, _, w, h = transformacion.map_box(0, 0, 2053, 1355)
        assert w / h == pytest.approx(2053 / 1355, rel=1e-6)

    def test_una_imagen_cuadrada_no_lleva_bandas(self) -> None:
        """Sin bandas que dibujar, el letterbox es un escalado a secas."""
        letterbox_image = self._letterbox()
        _, transformacion = letterbox_image(np.zeros((320, 320, 3), np.uint8), resolution=640)
        assert (transformacion.pad_x, transformacion.pad_y) == (0, 0)
        assert transformacion.scale == pytest.approx(2.0)


class TestEstiloDeParticion:
    """La paleta compartida de train/val/test."""

    def test_las_tres_particiones_tienen_estilo_propio(self) -> None:
        colores = {SPLIT_STYLES[s].color for s in ("train", "val", "test")}
        marcadores = {SPLIT_STYLES[s].marker for s in ("train", "val", "test")}
        assert len(colores) == 3
        assert len(marcadores) == 3

    def test_una_particion_desconocida_es_un_error(self) -> None:
        """Devolver un color por defecto pintaría dos particiones iguales."""
        with pytest.raises(KeyError, match="SPLIT_STYLES"):
            split_style_for("validacion")


class TestPrefijosDeFigura:
    """Capítulo y apartado en el nombre de fichero."""

    def test_la_figura_del_particionado_va_al_apartado_3_3(self, tmp_path: Path) -> None:
        apply_style()
        clases = ["a", "b"]
        conteos = {
            "train": {"a": 10, "b": 4},
            "val": {"a": 3, "b": 1},
            "test": {"a": 3, "b": 1},
        }
        rutas = plot_split_class_distribution(conteos, clases, tmp_path)
        assert rutas["pdf"].name == "fig_3_3_distribucion_clases_particion.pdf"
        assert rutas["png"].name == "fig_3_3_distribucion_clases_particion.png"
        assert rutas["pdf"].is_file() and rutas["png"].is_file()

    def test_la_coocurrencia_condicional_va_al_apartado_3_4(self, tmp_path: Path) -> None:
        """La coocurrencia es análisis exploratorio, y el EDA ahora es 3.4."""
        apply_style()
        nombres = ["a", "b"]
        resultado = compute_conditional_cooccurrence(_coco([[0, 1], [0]], nombres), nombres)
        rutas = plot_conditional_cooccurrence(resultado, tmp_path)
        # 'condicional' distingue esta figura de fig_3_4_coocurrencia_de_clases, que es la
        # simétrica.
        assert rutas["pdf"].name == "fig_3_4_coocurrencia_condicional.pdf"
        assert rutas["png"].is_file()


class TestSeleccionDeEjemploDeLetterbox:
    """Las cajas del ejemplo se buscan en las anotaciones."""

    def test_devuelve_todas_las_cajas(self) -> None:
        coco = _coco([[0, 1]], ["grande", "pequena"])
        coco["annotations"][0]["bbox"] = [0.0, 0.0, 80.0, 60.0]
        coco["annotations"][1]["bbox"] = [5.0, 5.0, 4.0, 4.0]
        nombre, cajas, etiqueta = select_letterbox_example(coco, ["grande", "pequena"], "img_0.jpg")
        assert nombre is not None
        assert cajas == [(0.0, 0.0, 80.0, 60.0), (5.0, 5.0, 4.0, 4.0)]
        # La etiqueta es la de la mayor, que es la única que se rotula.
        assert etiqueta == "grande"

    def test_la_etiqueta_es_la_de_la_mayor_aunque_no_sea_la_primera(self) -> None:
        """El orden del COCO no decide cuál se destaca; el área sí."""
        coco = _coco([[0, 1]], ["pequena", "grande"])
        coco["annotations"][0]["bbox"] = [5.0, 5.0, 4.0, 4.0]
        coco["annotations"][1]["bbox"] = [0.0, 0.0, 80.0, 60.0]
        _, cajas, etiqueta = select_letterbox_example(coco, ["pequena", "grande"], "img_0.jpg")
        assert len(cajas) == 2
        assert etiqueta == "grande"

    def test_devuelve_vacio_si_la_imagen_no_esta(self) -> None:
        """Se omite la figura en vez de dibujar otra imagen cualquiera."""
        coco = _coco([[0]], ["a"])
        assert select_letterbox_example(coco, ["a"], "no_existe.jpg") == (None, [], None)
