"""El reparto en las seis clases del registro, contra una cuenta hecha a mano."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tfm_edge.common import registry as registry_mod
from tfm_edge.evaluation import error_decomposition as ed
from tfm_edge.evaluation import matching_analysis
from tfm_edge.evaluation.coco_eval import build_evaluator

pytest.importorskip("pycocotools", reason="El reparto se mide sobre pycocotools.")
np = pytest.importorskip("numpy")


# Caso calculado a mano, con dos clases.
#
# Dos imágenes de 640x640 y dos clases: 0 = casco, 1 = chaleco.
#
# Anotaciones:
#   imagen 1: casco   en [0, 0, 100, 100]        (la llamamos A)
#             chaleco en [200, 200, 100, 100]    (la llamamos C)
#   imagen 2: casco   en [0, 0, 100, 100]        (la llamamos B)
#
# Detecciones (todas de clase 0, que es lo que hace visible el error de clase)
#   imagen 1  caja exacta sobre A        conf 0,90: acierto
#             caja exacta sobre C        conf 0,70: clasificación
#             caja 70 px a la derecha de C  conf 0,60: ambos
#             caja 60 px a la derecha de A  conf 0,50: localización
#             caja 10 px en diagonal de A   conf 0,40: duplicado
#   imagen 2  caja exacta sobre B        conf 0,30: acierto
#             caja lejos                 conf 0,20: fondo
#
# Las IoU, a mano (todas las cajas son de 100x100, área 10.000):
#   exacta: 1,0
#   10 px en diagonal: 90*90 / (20.000 - 8.100)   = 8.100/11.900 = 0,6807
#   60 px de lado: 40*100 / (20.000 - 4.000)  = 4.000/16.000 = 0,2500
#   70 px de lado: 30*100 / (20.000 - 3.000)  = 3.000/17.000 = 0,1765
#   lejos: 0
#
# De ahí salen las seis casillas, sin margen de interpretación:
#   conf 0,00  clasificación 1, localización 1, ambos 1, duplicado 1, fondo 1,
#              perdidas 1 (C, que ninguna detección de clase 1 empareja)
#   conf 0,25  el fondo (0,20) se queda fuera: fondo 0
#   conf 0,50  el duplicado (0,40) se queda fuera: duplicado 0, y el
#              acierto de B (0,30) también, así que perdidas pasa a 2
#   conf 0,75  solo queda el acierto de A: los cinco a 0

CAJA = [0.0, 0.0, 100.0, 100.0]
DIAGONAL_10 = [10.0, 10.0, 100.0, 100.0]
LADO_60 = [60.0, 0.0, 100.0, 100.0]
CHALECO = [200.0, 200.0, 100.0, 100.0]
CHALECO_LADO_70 = [270.0, 200.0, 100.0, 100.0]
LEJOS = [500.0, 500.0, 20.0, 20.0]

#: Lo que tiene que dar el reparto en cada umbral de la rejilla del proyecto.
ESPERADO: dict[float, dict[str, int]] = {
    0.00: {
        "classification": 1,
        "localization": 1,
        "both": 1,
        "duplicate": 1,
        "background": 1,
        "missed": 1,
    },
    0.10: {
        "classification": 1,
        "localization": 1,
        "both": 1,
        "duplicate": 1,
        "background": 1,
        "missed": 1,
    },
    0.25: {
        "classification": 1,
        "localization": 1,
        "both": 1,
        "duplicate": 1,
        "background": 0,
        "missed": 1,
    },
    0.50: {
        "classification": 1,
        "localization": 1,
        "both": 1,
        "duplicate": 0,
        "background": 0,
        "missed": 2,
    },
    0.75: {
        "classification": 0,
        "localization": 0,
        "both": 0,
        "duplicate": 0,
        "background": 0,
        "missed": 2,
    },
    0.90: {
        "classification": 0,
        "localization": 0,
        "both": 0,
        "duplicate": 0,
        "background": 0,
        "missed": 2,
    },
}


def _anotacion(ident: int, imagen: int, categoria: int, caja: list[float]) -> dict[str, Any]:
    """Una anotación COCO del caso."""
    return {
        "id": ident,
        "image_id": imagen,
        "category_id": categoria,
        "bbox": list(caja),
        "area": float(caja[2] * caja[3]),
        "iscrowd": 0,
    }


def _referencia_dos_clases() -> dict[str, Any]:
    """COCO de referencia con dos clases y un chaleco que nadie detecta."""
    return {
        "categories": [{"id": 0, "name": "helmet"}, {"id": 1, "name": "vest"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 640},
            {"id": 2, "file_name": "b.jpg", "width": 640, "height": 640},
        ],
        "annotations": [
            _anotacion(1, 1, 0, CAJA),
            _anotacion(2, 1, 1, CHALECO),
            _anotacion(3, 2, 0, CAJA),
        ],
    }


def _detecciones_de_las_seis() -> list[dict[str, Any]]:
    """Las siete detecciones que producen las seis clases de error."""
    return [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": list(CHALECO), "score": 0.70},
        {"image_id": 1, "category_id": 0, "bbox": list(CHALECO_LADO_70), "score": 0.60},
        {"image_id": 1, "category_id": 0, "bbox": list(LADO_60), "score": 0.50},
        {"image_id": 1, "category_id": 0, "bbox": list(DIAGONAL_10), "score": 0.40},
        {"image_id": 2, "category_id": 0, "bbox": list(CAJA), "score": 0.30},
        {"image_id": 2, "category_id": 0, "bbox": list(LEJOS), "score": 0.20},
    ]


def _referencia_una_clase() -> dict[str, Any]:
    """COCO de referencia con una sola clase, para el caso degenerado."""
    return {
        "categories": [{"id": 0, "name": "helmet"}],
        "images": [{"id": 1, "file_name": "a.jpg", "width": 640, "height": 640}],
        "annotations": [_anotacion(1, 1, 0, CAJA)],
    }


@pytest.fixture(scope="module")
def hechos() -> ed.DecompositionFacts:
    """Hechos ampliados del caso de las seis clases."""
    return ed.decomposition_facts(
        build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    )


@pytest.fixture(scope="module")
def volcado() -> dict[str, Any]:
    """Reparto completo del caso, en la rejilla por defecto."""
    return ed.decompose(
        build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    )


# ---------------------------------------------------------------------------
# La IoU contra las demás clases, y el alineamiento
# ---------------------------------------------------------------------------
def test_la_iou_ajena_sale_de_las_anotaciones_de_las_otras_clases(
    hechos: ed.DecompositionFacts,
) -> None:
    """Las siete detecciones tienen la IoU ajena que se calcula a mano."""
    assert hechos.facts.scores.size == 7
    esperado = [0.0, 1.0, 3000.0 / 17000.0, 0.0, 0.0, 0.0, 0.0]
    assert hechos.max_iou_other_class == pytest.approx(esperado, abs=1e-4)


def test_la_iou_propia_recalculada_coincide_en_todas_las_posiciones(
    hechos: ed.DecompositionFacts,
) -> None:
    """El ancla del alineamiento entre los dos recorridos cierra en cero."""
    assert hechos.own_iou_anchor_max_gap == pytest.approx(0.0, abs=1e-12)


def test_un_recorrido_desalineado_no_se_reparte(monkeypatch: Any) -> None:
    """Si los dos recorridos no coinciden, se levanta el error en vez de repartir."""
    original = ed.evaluation_facts

    def con_una_iou_movida(evaluador: Any) -> matching_analysis.EvaluationFacts:
        """Los hechos de verdad, con una IoU cambiada."""
        facts = original(evaluador)
        movida = np.array(facts.max_iou, dtype=float)
        movida[0] = 0.99
        facts.max_iou = movida
        return facts

    monkeypatch.setattr(ed, "evaluation_facts", con_una_iou_movida)
    evaluador = build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    with pytest.raises(ed.ErrorDecompositionError, match="no coincide"):
        ed.decomposition_facts(evaluador)


def test_un_recorrido_con_otro_numero_de_detecciones_no_se_reparte(monkeypatch: Any) -> None:
    """Si los dos recorridos no ven las mismas detecciones, se para."""
    original = ed.evaluation_facts

    def con_una_menos(evaluador: Any) -> matching_analysis.EvaluationFacts:
        """Los hechos de verdad, sin la última detección."""
        facts = original(evaluador)
        facts.max_iou = np.array(facts.max_iou[:-1], dtype=float)
        return facts

    monkeypatch.setattr(ed, "evaluation_facts", con_una_menos)
    evaluador = build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    with pytest.raises(ed.ErrorDecompositionError, match="mismo número de"):
        ed.decomposition_facts(evaluador)


# ---------------------------------------------------------------------------
# Las seis casillas, una a una
# ---------------------------------------------------------------------------
def test_las_seis_clases_aparecen_una_vez_cada_una(hechos: ed.DecompositionFacts) -> None:
    """A confianza cero el caso produce exactamente un error de cada tipo."""
    reparto = ed.error_decomposition(hechos, 0.0)
    for clase in ed.ERROR_KINDS:
        assert reparto[clase] == 1, f"'{clase}' debería tener exactamente un caso"


@pytest.mark.parametrize("confianza", sorted(ESPERADO))
def test_el_reparto_coincide_con_la_cuenta_a_mano(
    hechos: ed.DecompositionFacts, confianza: float
) -> None:
    """En cada umbral de la rejilla, las seis casillas son las de la servilleta."""
    reparto = ed.error_decomposition(hechos, confianza)
    obtenido = {clase: reparto[clase] for clase in ed.ERROR_KINDS}
    assert obtenido == ESPERADO[confianza]


def test_las_cinco_de_falso_positivo_agotan_el_total(hechos: ed.DecompositionFacts) -> None:
    """El reparto de falsos positivos suma el total, sin resto, en toda la rejilla."""
    for confianza in matching_analysis.DECOMPOSITION_CONFIDENCES:
        reparto = ed.error_decomposition(hechos, confianza)
        suma = sum(reparto[clase] for clase in ed.FALSE_POSITIVE_KINDS)
        assert suma == reparto["false_positives_total"] == hechos.facts.fp_at(confianza)


def test_las_perdidas_dependen_del_umbral(hechos: ed.DecompositionFacts) -> None:
    """Una anotación encontrada solo por una caja de poca confianza cuenta como perdida."""
    assert hechos.missed_at(0.0) == 1
    assert hechos.missed_at(0.30) == 1
    assert hechos.missed_at(0.50) == 2
    assert hechos.missed_at(0.95) == 3
    assert hechos.facts.num_gt == 3


def test_sin_anotaciones_de_otras_clases_no_hay_error_de_clase() -> None:
    """Con una sola clase, ``classification`` y ``both`` son cero por construcción."""
    detecciones = [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": list(DIAGONAL_10), "score": 0.40},
        {"image_id": 1, "category_id": 0, "bbox": list(LEJOS), "score": 0.20},
    ]
    hechos = ed.decomposition_facts(
        build_evaluator(_referencia_una_clase(), detecciones, "una clase")
    )
    reparto = ed.error_decomposition(hechos, 0.0)
    assert reparto["classification"] == 0
    assert reparto["both"] == 0
    assert reparto["duplicate"] == 1
    assert reparto["background"] == 1


# ---------------------------------------------------------------------------
# La relación con el reparto de tres clases que ya estaba volcado
# ---------------------------------------------------------------------------
def test_el_reparto_fino_reproduce_el_de_tres_clases(hechos: ed.DecompositionFacts) -> None:
    """Agrupando las casillas finas sale exactamente ``fp_taxonomy``."""
    for confianza in matching_analysis.DECOMPOSITION_CONFIDENCES:
        fino = ed.error_decomposition(hechos, confianza)
        grueso = matching_analysis.fp_taxonomy(hechos.facts, confianza)
        assert fino["duplicate"] == grueso["duplicate"]
        assert fino["localization"] == grueso["poor_localization"]
        assert fino["classification"] + fino["both"] + fino["background"] == grueso["spurious"]


def test_el_reparto_de_tres_clases_no_sirve_para_rellenar_las_columnas(
    hechos: ed.DecompositionFacts,
) -> None:
    """La casilla ``spurious`` contiene aquí tres errores distintos."""
    grueso = matching_analysis.fp_taxonomy(hechos.facts, 0.0)
    fino = ed.error_decomposition(hechos, 0.0)
    assert grueso["spurious"] == 3
    assert fino["background"] == 1
    assert fino["classification"] == 1
    assert fino["both"] == 1


def test_un_reparto_que_no_reproduce_el_de_tres_no_se_devuelve(
    hechos: ed.DecompositionFacts, monkeypatch: Any
) -> None:
    """Si la equivalencia con ``fp_taxonomy`` se rompiera, se para en vez de volcar."""
    monkeypatch.setattr(ed, "DUPLICATE_IOU", 0.9)
    with pytest.raises(ed.ErrorDecompositionError, match="no reproduce el de tres"):
        ed.error_decomposition(hechos, 0.0)


# ---------------------------------------------------------------------------
# El cuadre contra las detecciones que el validador produjo
# ---------------------------------------------------------------------------
def test_sin_truncado_las_evaluadas_son_todas() -> None:
    """Con siete detecciones no hay nada que truncar y el cuadre es exacto."""
    evaluador = build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    cuentas = ed.detection_accounting(evaluador, _detecciones_de_las_seis(), 7)
    assert cuentas["num_truncated_by_max_det"] == 0
    assert cuentas["num_evaluated_detections"] == 7
    assert cuentas["pairs_over_max_det"] == 0
    assert cuentas["closes_without_remainder"] is True
    assert cuentas["matches_recorded_raw_count"] is True


def test_con_truncado_el_hueco_queda_explicado() -> None:
    """El hueco entre lo producido y lo evaluado es exactamente lo truncado."""
    detecciones = [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": list(DIAGONAL_10), "score": 0.80},
        {"image_id": 1, "category_id": 0, "bbox": list(LADO_60), "score": 0.70},
        {"image_id": 1, "category_id": 0, "bbox": list(LEJOS), "score": 0.60},
    ]
    evaluador = build_evaluator(_referencia_una_clase(), detecciones, "una clase")
    evaluador.params.maxDets = [1, 2]
    evaluador.evaluate()

    cuentas = ed.detection_accounting(evaluador, detecciones, len(detecciones))
    assert cuentas["max_detections_per_image_and_class"] == 2
    assert cuentas["num_truncated_by_max_det"] == 2
    assert cuentas["num_evaluated_detections"] == 2
    assert cuentas["pairs_over_max_det"] == 1
    assert cuentas["closes_without_remainder"] is True


def test_un_recuento_crudo_que_no_es_el_del_volcado_se_declara() -> None:
    """Si las reconstruidas no son las que el volcado registró, se dice."""
    evaluador = build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    cuentas = ed.detection_accounting(evaluador, _detecciones_de_las_seis(), 999)
    assert cuentas["matches_recorded_raw_count"] is False
    assert cuentas["num_reconstructed_detections"] == 7
    assert cuentas["num_raw_detections"] == 999


def test_los_aciertos_y_los_falsos_positivos_agotan_las_evaluadas(
    hechos: ed.DecompositionFacts,
) -> None:
    """A confianza cero, TP + FP es el total de detecciones evaluadas."""
    reparto = ed.error_decomposition(hechos, 0.0)
    assert reparto["true_positives"] + reparto["false_positives_total"] == hechos.facts.scores.size


# ---------------------------------------------------------------------------
# El contraste contra la curva del paso 23, que valida lo que agosto no cubre
# ---------------------------------------------------------------------------
def _curva(detecciones: list[dict[str, Any]], confianzas: tuple[float, ...]) -> list[Any]:
    """Los puntos de la curva del paso 23 para unas detecciones del caso."""
    from tfm_edge.evaluation import operating_curve

    evaluador = build_evaluator(_referencia_dos_clases(), detecciones, "servilleta")
    return operating_curve.precision_recall_at_confidences(evaluador, confidences=confianzas)[
        "points"
    ]


def test_los_recuentos_coinciden_con_los_de_la_curva(volcado: dict[str, Any]) -> None:
    """Aciertos, falsos positivos y anotaciones son los mismos por los dos caminos."""
    puntos = _curva(_detecciones_de_las_seis(), (0.10, 0.25, 0.50, 0.75, 0.90))
    resultado = ed.cross_check_curve(volcado, puntos)
    assert resultado["matches"] is True
    assert resultado["confidences_checked"] == [0.10, 0.25, 0.50, 0.75, 0.90]


def test_una_curva_de_otras_detecciones_no_pasa(volcado: dict[str, Any]) -> None:
    otras = _detecciones_de_las_seis()[:-2]  # sin el acierto de B ni el fondo
    puntos = _curva(otras, (0.10, 0.25))
    with pytest.raises(ed.ErrorDecompositionError, match="la curva"):
        ed.cross_check_curve(volcado, puntos)


def test_sin_umbrales_en_comun_no_se_da_por_pasado(volcado: dict[str, Any]) -> None:
    """Un contraste que no compara nada no puede pasar (mecanismo 16)."""
    puntos = _curva(_detecciones_de_las_seis(), (0.33,))
    with pytest.raises(ed.ErrorDecompositionError, match="ningún umbral en común"):
        ed.cross_check_curve(volcado, puntos)


def test_el_volcado_declara_que_los_recuentos_son_agregados(volcado: dict[str, Any]) -> None:
    """El ``recall`` de la curva es la media por clase y el reparto cuenta agregado."""
    assert "agregados" in volcado["aggregation"]
    assert "no medias por clase" in volcado["aggregation"]


# ---------------------------------------------------------------------------
# El volcado y las columnas
# ---------------------------------------------------------------------------
def test_las_seis_clases_son_las_seis_columnas_del_registro() -> None:
    """La tabla de columnas coincide con lo que el registro declara, en los dos sentidos."""
    del_registro = {
        columna
        for columna in registry_mod.QUALITY_COLUMNS_AFTER_CLASSES
        if columna.startswith("error_")
    }
    assert set(ed.REGISTRY_COLUMNS.values()) == del_registro
    assert set(ed.REGISTRY_COLUMNS) == set(ed.ERROR_KINDS)
    assert len(ed.ERROR_KINDS) == len(set(ed.ERROR_KINDS)) == 6


def test_el_volcado_declara_los_cortes_y_el_orden(volcado: dict[str, Any]) -> None:
    """Los cortes son convencionales, así que van escritos junto a los recuentos."""
    umbrales = volcado["thresholds"]
    assert umbrales["matching_iou"] == matching_analysis.MATCHING_IOU
    assert umbrales["duplicate_iou"] == matching_analysis.DUPLICATE_IOU
    assert umbrales["spurious_iou"] == matching_analysis.SPURIOUS_IOU
    assert "background" in umbrales["order"]
    assert volcado["num_evaluated_detections"] == 7
    assert volcado["num_gt"] == 3


def test_el_volcado_recorre_la_rejilla_del_proyecto(volcado: dict[str, Any]) -> None:
    medidos = [fila["confidence"] for fila in volcado["by_confidence"]]
    assert medidos == list(matching_analysis.DECOMPOSITION_CONFIDENCES)


def test_el_volcado_lleva_la_prueba_del_alineamiento(volcado: dict[str, Any]) -> None:
    assert volcado["own_iou_anchor_max_gap"] == pytest.approx(0.0, abs=1e-12)
    assert volcado["matched_iou_min"] >= matching_analysis.MATCHING_IOU


def test_el_reparto_es_determinista(volcado: dict[str, Any]) -> None:
    """Dos ejecuciones sobre las mismas detecciones dan el mismo reparto."""
    otra_vez = ed.decompose(
        build_evaluator(_referencia_dos_clases(), _detecciones_de_las_seis(), "servilleta")
    )
    assert otra_vez["by_confidence"] == volcado["by_confidence"]


def test_el_modulo_no_escribe_en_el_registro() -> None:
    """El paso 1 de C no toca ninguna de las ochenta filas."""
    fuente = Path(ed.__file__).read_text(encoding="utf-8")
    assert "tfm_edge.common.registry" not in fuente
    assert "common import registry" not in fuente
    for prohibido in ("ExperimentRegistry", "update_cells", "_write_rows_atomic", "to_csv"):
        assert prohibido not in fuente, f"el módulo no debería nombrar '{prohibido}'"
