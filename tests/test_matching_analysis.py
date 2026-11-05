"""El reparto entre emparejamiento y orden, contra números hechos a mano."""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.evaluation import matching_analysis, operating_curve
from tfm_edge.evaluation.coco_eval import build_evaluator

pytest.importorskip("pycocotools", reason="El reparto se mide sobre pycocotools.")
np = pytest.importorskip("numpy")


# Caso calculado a mano.
#
# Dos imágenes de 640x640, una sola clase, una anotación por imagen, las dos en
# [0, 0, 100, 100]. Las llamamos A (imagen 1) y B (imagen 2).
#
# Modelo de antes:
#   imagen 1: caja exacta sobre A          conf 0,90: acierto
#             caja desplazada 10 px        conf 0,40: FP duplicado (IoU 0,681)
#   imagen 2: caja exacta sobre B          conf 0,30: acierto
#             caja lejos                   conf 0,20: FP espurio (IoU 0)
#
# Modelo de después:
#   imagen 1: caja exacta sobre A          conf 0,95: acierto, con más confianza
#             caja desplazada 60 px        conf 0,50: FP localización (IoU 0,250)
#   imagen 2: caja desplazada 70 px        conf 0,80: FP localización (IoU 0,176)
#             caja lejos                   conf 0,10: FP espurio
#
# De ahí sale, sin margen de interpretación:
#   cobertura      antes {A, B}, después {A}: 1 perdida, 0 ganada, 1 en ambos
#   ΔTP conf 0,25  antes 2, después 1: -1, todo de emparejamiento perdido
#   ΔTP conf 0,50  antes 1, después 1: 0
#   ΔTP conf 0,95  antes 0, después 1: +1, todo de orden
#   FP conf 0,25   antes {1 duplicado}, después {2 localización}

CAJA = [0.0, 0.0, 100.0, 100.0]
DESPLAZADA_10 = [10.0, 10.0, 100.0, 100.0]
DESPLAZADA_60 = [60.0, 0.0, 100.0, 100.0]
DESPLAZADA_70 = [70.0, 0.0, 100.0, 100.0]
LEJOS = [500.0, 500.0, 20.0, 20.0]


def _referencia() -> dict[str, Any]:
    """COCO de referencia del caso de servilleta."""
    return {
        "categories": [{"id": 0, "name": "helmet"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 640, "height": 640},
            {"id": 2, "file_name": "b.jpg", "width": 640, "height": 640},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": list(CAJA),
                "area": 10000.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 0,
                "bbox": list(CAJA),
                "area": 10000.0,
                "iscrowd": 0,
            },
        ],
    }


def _detecciones_antes() -> list[dict[str, Any]]:
    """Detecciones del modelo de antes."""
    return [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": list(DESPLAZADA_10), "score": 0.40},
        {"image_id": 2, "category_id": 0, "bbox": list(CAJA), "score": 0.30},
        {"image_id": 2, "category_id": 0, "bbox": list(LEJOS), "score": 0.20},
    ]


def _detecciones_despues() -> list[dict[str, Any]]:
    """Detecciones del modelo de después."""
    return [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.95},
        {"image_id": 1, "category_id": 0, "bbox": list(DESPLAZADA_60), "score": 0.50},
        {"image_id": 2, "category_id": 0, "bbox": list(DESPLAZADA_70), "score": 0.80},
        {"image_id": 2, "category_id": 0, "bbox": list(LEJOS), "score": 0.10},
    ]


@pytest.fixture(scope="module")
def antes() -> matching_analysis.EvaluationFacts:
    """Hechos del modelo de antes."""
    return matching_analysis.evaluation_facts(
        build_evaluator(_referencia(), _detecciones_antes(), "servilleta")
    )


@pytest.fixture(scope="module")
def despues() -> matching_analysis.EvaluationFacts:
    """Hechos del modelo de después."""
    return matching_analysis.evaluation_facts(
        build_evaluator(_referencia(), _detecciones_despues(), "servilleta")
    )


# ---------------------------------------------------------------------------
# Geometría
# ---------------------------------------------------------------------------
def test_la_iou_de_dos_cajas_iguales_es_uno() -> None:
    """Una caja consigo misma tiene IoU 1."""
    assert matching_analysis.box_iou_matrix([CAJA], [CAJA])[0, 0] == pytest.approx(1.0)


def test_la_iou_de_dos_cajas_disjuntas_es_cero() -> None:
    """Dos cajas que no se tocan tienen IoU 0."""
    assert matching_analysis.box_iou_matrix([CAJA], [LEJOS])[0, 0] == pytest.approx(0.0)


def test_la_iou_coincide_con_la_cuenta_a_mano() -> None:
    """Las tres IoU del caso salen de una servilleta."""
    fila = matching_analysis.box_iou_matrix([DESPLAZADA_10, DESPLAZADA_60, DESPLAZADA_70], [CAJA])[
        :, 0
    ]
    assert fila[0] == pytest.approx(8100 / 11900)
    assert fila[1] == pytest.approx(0.25)
    assert fila[2] == pytest.approx(3000 / 17000)


def test_la_iou_con_un_lado_vacio_devuelve_la_forma_correcta() -> None:
    """Una imagen sin anotaciones de esa clase no rompe el cálculo."""
    matriz = matching_analysis.box_iou_matrix([CAJA, LEJOS], [])
    assert matriz.shape == (2, 0)


# ---------------------------------------------------------------------------
# El estadístico de orden
# ---------------------------------------------------------------------------
def test_el_orden_perfecto_da_uno() -> None:
    """Si todos los aciertos puntúan por encima de todos los fallos, sale 1."""
    assert matching_analysis.rank_auc([0.8, 0.9], [0.1, 0.2]) == pytest.approx(1.0)


def test_el_orden_invertido_da_cero() -> None:
    """Si todos los fallos puntúan por encima, sale 0."""
    assert matching_analysis.rank_auc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)


def test_los_empates_cuentan_medio() -> None:
    """Un empate no es ni un acierto ni un fallo del orden."""
    assert matching_analysis.rank_auc([0.5], [0.5]) == pytest.approx(0.5)


def test_el_orden_del_caso_de_servilleta_sale_de_contar_parejas() -> None:
    """Con aciertos 0,90 y 0,30 y fallos 0,40 y 0,20, tres de cuatro parejas van bien."""
    assert matching_analysis.rank_auc([0.90, 0.30], [0.40, 0.20]) == pytest.approx(0.75)


def test_sin_uno_de_los_dos_lados_no_hay_medida() -> None:
    """Sin falsos positivos la pregunta no tiene respuesta, y no se inventa un 0."""
    assert matching_analysis.rank_auc([0.9], []) is None
    assert matching_analysis.rank_auc([], [0.9]) is None


# ---------------------------------------------------------------------------
# Lectura de la evaluación
# ---------------------------------------------------------------------------
def test_la_cobertura_no_depende_del_umbral(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """Las dos anotaciones están cubiertas, aunque una lo esté con conf 0,30."""
    assert antes.num_gt == 2
    assert sorted(antes.covered) == [1, 2]
    assert antes.covered[1] == pytest.approx(0.90)
    assert antes.covered[2] == pytest.approx(0.30)


def test_el_modelo_de_despues_pierde_una_anotacion(
    despues: matching_analysis.EvaluationFacts,
) -> None:
    assert sorted(despues.covered) == [1]


def test_la_iou_calculada_ancla_contra_el_emparejamiento(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """Las parejas que pycocotools empareja dan aquí IoU de al menos 0,5."""
    assert antes.matched_iou_min >= matching_analysis.MATCHING_IOU


def test_los_recuentos_de_detecciones_son_los_del_caso(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """Cuatro detecciones, dos emparejadas."""
    assert antes.scores.size == 4
    assert int(np.count_nonzero(antes.is_tp)) == 2


def test_una_evaluacion_sin_evaluar_no_se_reparte() -> None:
    """Pedir el reparto de algo sin ``evaluate()`` es un error explícito."""

    class Vacia:
        """Un evaluador al que todavía no se le ha llamado a evaluate."""

        evalImgs: list[Any] = []

    with pytest.raises(matching_analysis.MatchingAnalysisError, match="sin evaluar"):
        matching_analysis.evaluation_facts(Vacia())


# ---------------------------------------------------------------------------
# El contraste contra la curva ya volcada
# ---------------------------------------------------------------------------
def test_los_recuentos_coinciden_con_la_curva_del_paso_23(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """Las dos lecturas de las mismas detecciones dan los mismos recuentos."""
    evaluador = build_evaluator(_referencia(), _detecciones_antes(), "servilleta")
    curva = operating_curve.precision_recall_at_confidences(
        evaluador, confidences=(0.05, 0.25, 0.50, 0.95)
    )
    resultado = matching_analysis.check_against_curve(antes, {"curve": curva})
    assert resultado["matches_operating_curve"] is True
    assert resultado["confidences_checked"] == 4


def test_una_curva_de_otra_evaluacion_no_pasa_el_contraste(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """Si la curva es de otro modelo, el contraste falla en vez de seguir."""
    otra = operating_curve.precision_recall_at_confidences(
        build_evaluator(_referencia(), _detecciones_despues(), "servilleta"),
        confidences=(0.25,),
    )
    with pytest.raises(matching_analysis.MatchingAnalysisError, match="no coinciden"):
        matching_analysis.check_against_curve(antes, {"curve": otra})


# ---------------------------------------------------------------------------
# El reparto
# ---------------------------------------------------------------------------
def test_la_cobertura_se_reparte_en_cuatro_estados(
    antes: matching_analysis.EvaluationFacts, despues: matching_analysis.EvaluationFacts
) -> None:
    """A la cubren los dos, B solo el de antes, y no hay ninguna ganada."""
    reparto = matching_analysis.coverage_split(antes, despues)
    assert reparto["num_gt"] == 2
    assert reparto["covered_before"] == 2
    assert reparto["covered_after"] == 1
    assert reparto["covered_by_both"] == 1
    assert reparto["matching_lost"] == 1
    assert reparto["matching_gained"] == 0
    assert reparto["covered_by_neither"] == 0


def test_dos_evaluaciones_con_distintas_anotaciones_no_se_comparan(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """Comparar sobre particiones distintas no es comparar."""
    recortada = dict(_referencia())
    recortada["images"] = recortada["images"][:1]
    recortada["annotations"] = recortada["annotations"][:1]
    otra = matching_analysis.evaluation_facts(
        build_evaluator(recortada, _detecciones_antes()[:2], "recortada")
    )
    with pytest.raises(matching_analysis.MatchingAnalysisError, match="mismas anotaciones"):
        matching_analysis.coverage_split(antes, otra)


@pytest.mark.parametrize(
    "confianza, delta, emparejamiento, orden",
    [
        # conf 0,25: de los dos aciertos de antes queda uno; se pierde B.
        (0.25, -1, -1, 0),
        # conf 0,50: B no contaba antes (conf 0,30), así que no cambia nada.
        (0.50, 0, 0, 0),
        # conf 0,95: A cruza el umbral solo después. Puro orden.
        (0.95, 1, 0, 1),
    ],
)
def test_el_reparto_de_aciertos_sale_de_la_servilleta(
    antes: matching_analysis.EvaluationFacts,
    despues: matching_analysis.EvaluationFacts,
    confianza: float,
    delta: int,
    emparejamiento: int,
    orden: int,
) -> None:
    """Los tres sumandos coinciden con la cuenta hecha a mano en tres umbrales."""
    reparto = matching_analysis.recall_decomposition(antes, despues, confianza)
    assert reparto["delta_tp"] == delta
    assert reparto["from_matching_lost"] + reparto["from_matching_gained"] == emparejamiento
    assert reparto["from_score_shift"] == orden
    assert reparto["identity_holds"] is True


def test_un_reparto_que_no_suma_no_se_devuelve() -> None:
    """Si la cobertura y los recuentos no describen la misma evaluación, falla."""
    coherente = matching_analysis.EvaluationFacts(
        num_gt=1,
        covered={1: 0.9},
        scores=np.array([0.9]),
        is_tp=np.array([True]),
        max_iou=np.array([1.0]),
        category=np.array([0]),
        matched_iou_min=1.0,
    )
    incoherente = matching_analysis.EvaluationFacts(
        num_gt=1,
        covered={1: 0.9},
        scores=np.array([0.9]),
        is_tp=np.array([False]),
        max_iou=np.array([0.0]),
        category=np.array([0]),
        matched_iou_min=1.0,
    )
    with pytest.raises(matching_analysis.MatchingAnalysisError, match="no suma"):
        matching_analysis.recall_decomposition(coherente, incoherente, 0.25)


# ---------------------------------------------------------------------------
# El desglose de falsos positivos
# ---------------------------------------------------------------------------
def test_los_falsos_positivos_de_antes_son_un_duplicado(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """En conf 0,25 solo queda la caja desplazada 10 px, que solapa 0,681."""
    desglose = matching_analysis.fp_taxonomy(antes, 0.25)
    assert desglose == {
        "confidence": 0.25,
        "duplicate": 1,
        "poor_localization": 0,
        "spurious": 0,
        "total": 1,
    }


def test_bajando_el_umbral_aparece_el_espurio(
    antes: matching_analysis.EvaluationFacts,
) -> None:
    """La caja lejana entra con conf 0,20 y no toca nada de su clase."""
    desglose = matching_analysis.fp_taxonomy(antes, 0.10)
    assert desglose["spurious"] == 1
    assert desglose["total"] == 2


def test_los_falsos_positivos_de_despues_son_localizacion_pobre(
    despues: matching_analysis.EvaluationFacts,
) -> None:
    """Las dos cajas desplazadas tocan un objeto de su clase sin llegar a 0,5."""
    desglose = matching_analysis.fp_taxonomy(despues, 0.25)
    assert desglose == {
        "confidence": 0.25,
        "duplicate": 0,
        "poor_localization": 2,
        "spurious": 0,
        "total": 2,
    }


def test_los_tres_tipos_agotan_el_total(
    antes: matching_analysis.EvaluationFacts, despues: matching_analysis.EvaluationFacts
) -> None:
    """El desglose reparte todos los falsos positivos, sin resto."""
    for hechos in (antes, despues):
        for umbral in (0.0, 0.10, 0.25, 0.50, 0.95):
            desglose = matching_analysis.fp_taxonomy(hechos, umbral)
            assert sum(desglose[tipo] for tipo in matching_analysis.FP_KINDS) == desglose["total"]


# ---------------------------------------------------------------------------
# El desplazamiento de confianza sobre lo que los dos encuentran
# ---------------------------------------------------------------------------
def test_el_desplazamiento_se_mide_solo_sobre_lo_cubierto_por_ambos(
    antes: matching_analysis.EvaluationFacts, despues: matching_analysis.EvaluationFacts
) -> None:
    """Solo A está en los dos, y su confianza sube de 0,90 a 0,95."""
    desplazamiento = matching_analysis.covered_score_shift(antes, despues)
    assert desplazamiento["num_covered_by_both"] == 1
    assert desplazamiento["median_before"] == pytest.approx(0.90)
    assert desplazamiento["median_after"] == pytest.approx(0.95)
    assert desplazamiento["median_delta"] == pytest.approx(0.05)
    assert desplazamiento["fraction_increased"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# El reparto completo
# ---------------------------------------------------------------------------
def test_el_reparto_completo_es_determinista(
    antes: matching_analysis.EvaluationFacts, despues: matching_analysis.EvaluationFacts
) -> None:
    """Con las mismas entradas sale exactamente lo mismo."""
    umbrales = (0.10, 0.25, 0.50)
    primero = matching_analysis.analyze_pair(antes, despues, umbrales)
    segundo = matching_analysis.analyze_pair(antes, despues, umbrales)
    assert primero == segundo


def test_el_reparto_completo_declara_los_umbrales_que_ha_usado(
    antes: matching_analysis.EvaluationFacts, despues: matching_analysis.EvaluationFacts
) -> None:
    """Los cortes de IoU son convencionales, así que viajan con los recuentos."""
    analisis = matching_analysis.analyze_pair(antes, despues, (0.25,))
    assert analisis["thresholds"]["matching_iou"] == matching_analysis.MATCHING_IOU
    assert analisis["thresholds"]["spurious_iou"] == matching_analysis.SPURIOUS_IOU


def test_el_orden_del_par_se_mide_en_los_dos_lados(
    antes: matching_analysis.EvaluationFacts, despues: matching_analysis.EvaluationFacts
) -> None:
    """El de antes ordena 0,75; el de después no se equivoca en ninguna pareja."""
    analisis = matching_analysis.analyze_pair(antes, despues, (0.25,))
    assert analisis["ranking"]["before"]["macro"] == pytest.approx(0.75)
    assert analisis["ranking"]["after"]["macro"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# El script
# ---------------------------------------------------------------------------
def _cargar_script() -> ModuleType:
    """Carga el paso 25 como módulo."""
    ruta = get_project_paths().root / "scripts" / "25_matching_ranking.py"
    spec = importlib.util.spec_from_file_location("paso_25", ruta)
    assert spec is not None and spec.loader is not None
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def test_el_nombre_del_volcado_lleva_la_confianza() -> None:
    """Dos confianzas de referencia no comparten fichero."""
    script = _cargar_script()
    assert script.nombre_volcado(0.25) == "matching_ranking_conf025.json"
    assert script.nombre_volcado(0.5) == "matching_ranking_conf050.json"


def test_el_paso_25_no_escribe_en_el_registro() -> None:
    """El reparto se deriva del registro y no lo modifica."""
    ruta = get_project_paths().root / "scripts" / "25_matching_ranking.py"
    texto = ruta.read_text(encoding="utf-8")
    for prohibido in ("append_row", "append_quality", "write_rows", "upsert"):
        assert prohibido not in texto, f"el paso 25 llama a {prohibido}"
    assert "read_rows" in texto


def test_las_etiquetas_de_la_tabla_distinguen_la_tecnica() -> None:
    """PTQ y QAT comparten etiqueta, así que la tabla tiene que llevar la técnica."""
    script = _cargar_script()
    ptq = script._etiqueta({"label": "default", "via": "ptq"})
    qat = script._etiqueta({"label": "default", "via": "qat"})
    assert ptq != qat
    assert "ptq" in ptq and "qat" in qat


def test_una_procedencia_que_no_existe_no_se_adivina(tmp_path: Any) -> None:
    """Si ni la ruta grabada ni el nombre aparecen, se para en vez de inventar."""
    script = _cargar_script()
    curva = {
        "experiment_id": "s_baseline_default_00000000",
        "runtime": "onnxruntime",
        "precision": "fp32",
        "source": {
            "predictions": "/no/existe/validator_x/predictions.json",
            "recorded_dump": "/no/existe/x.json",
        },
    }
    with pytest.raises(TFMEdgeError, match="No se encuentran"):
        script.resolver_procedencia(curva, tmp_path, tmp_path)
