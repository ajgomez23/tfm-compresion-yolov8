"""La reconstrucción anclada de una evaluación, y sus dos guardas."""

from __future__ import annotations

from typing import Any

import pytest

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.evaluation import reconstruction
from tfm_edge.evaluation.coco_eval import build_evaluator, operating_point_at_best_f1

pytest.importorskip("pycocotools", reason="La reconstrucción se ancla con pycocotools.")

CAJA = [0.0, 0.0, 100.0, 100.0]
LEJOS = [500.0, 500.0, 20.0, 20.0]


def _referencia() -> dict[str, Any]:
    """COCO de referencia mínimo."""
    return {
        "categories": [{"id": 0, "name": "helmet"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 640}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": list(CAJA),
                "area": 10000.0,
                "iscrowd": 0,
            }
        ],
    }


def _detecciones() -> list[dict[str, Any]]:
    """Un acierto y un falso positivo."""
    return [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": list(LEJOS), "score": 0.80},
    ]


def _volcado(evaluador: Any, desvio: float = 0.0) -> dict[str, Any]:
    """Construye un volcado registrado a partir de un evaluador."""
    return {
        "coco": {
            "by_criterion": {
                reconstruction.SIZE_CRITERION: {
                    "map_50": float(evaluador.stats[1]) + desvio,
                    "map_50_95": float(evaluador.stats[0]),
                    "operating_point": (
                        operating_point_at_best_f1(evaluador) if desvio == 0.0 else {}
                    ),
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# Traducción de categorías
# ---------------------------------------------------------------------------
def test_las_categorias_se_invierten_desplazando_uno() -> None:
    """Ultralytics numera desde 1 cuando el dataset no es COCO."""
    invertidas = reconstruction.invert_categories(
        [{"category_id": 1, "score": 0.5}, {"category_id": 17, "score": 0.4}], 17
    )
    assert [d["category_id"] for d in invertidas] == [0, 16]


def test_una_categoria_fuera_de_rango_para_el_paso() -> None:
    with pytest.raises(TFMEdgeError, match="fuera de la numeración"):
        reconstruction.invert_categories([{"category_id": 0}, {"category_id": 3}], 17)


def test_la_traduccion_no_toca_las_detecciones_originales() -> None:
    originales = [{"category_id": 1, "score": 0.5}]
    reconstruction.invert_categories(originales, 17)
    assert originales[0]["category_id"] == 1


# ---------------------------------------------------------------------------
# Anclaje
# ---------------------------------------------------------------------------
def test_el_anclaje_rechaza_una_reconstruccion_que_no_reproduce() -> None:
    evaluador = build_evaluator(_referencia(), _detecciones(), "servilleta")
    with pytest.raises(TFMEdgeError, match="NO reproduce"):
        reconstruction.anchor_against_recorded(evaluador, _volcado(evaluador, desvio=0.01))


def test_el_anclaje_acepta_la_reconstruccion_correcta() -> None:
    """Y con las cifras que de verdad salieron, pasa y deja constancia."""
    evaluador = build_evaluator(_referencia(), _detecciones(), "servilleta")
    resultado = reconstruction.anchor_against_recorded(evaluador, _volcado(evaluador))
    assert resultado["anchored"] is True
    assert resultado["max_difference"] <= reconstruction.ANCHOR_TOLERANCE
    assert "map_50" in resultado["compared"]


def test_el_anclaje_compara_tambien_el_punto_de_operacion() -> None:
    evaluador = build_evaluator(_referencia(), _detecciones(), "servilleta")
    resultado = reconstruction.anchor_against_recorded(evaluador, _volcado(evaluador))
    assert "operating_point.precision" in resultado["compared"]
    assert "operating_point.confidence" in resultado["compared"]


# ---------------------------------------------------------------------------
# El paso 23 no se ha quedado con una copia propia
# ---------------------------------------------------------------------------
def test_el_paso_23_usa_la_reconstruccion_compartida() -> None:
    """Dos copias del mismo camino son dos cosas que pueden desviarse."""
    from tfm_edge.common.paths import get_project_paths

    texto = (get_project_paths().root / "scripts" / "23_operating_curve.py").read_text(
        encoding="utf-8"
    )
    assert "reconstruction.rebuild_evaluation" in texto
    assert "def invertir_categorias" not in texto
    assert "def anclar" not in texto
