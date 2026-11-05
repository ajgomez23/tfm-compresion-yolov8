"""La curva de precisión y recall a umbral fijo, contra números hechos a mano."""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.paths import get_project_paths
from tfm_edge.evaluation import operating_curve
from tfm_edge.evaluation.coco_eval import CocoEvalError, build_evaluator

pytest.importorskip("pycocotools", reason="La curva se mide con pycocotools.")


# Caso calculado a mano.
#
# Dos imágenes de 640x640, una sola clase, una anotación por imagen.
#
#   imagen 1: anotación en [0, 0, 100, 100]
#             detección exacta        conf 0,90: acierto
#             detección lejos         conf 0,80: falso positivo
#   imagen 2: anotación en [0, 0, 100, 100]
#             detección exacta        conf 0,30: acierto
#             detección lejos         conf 0,10: falso positivo
#
# De ahí salen los cuatro puntos que se comprueban:
#
#   conf 0,95: no queda ninguna: P = 1 (convenio), R = 0
#   conf 0,50: 0,90 y 0,80:      TP=1 FP=1: P = 1/2,  R = 1/2
#   conf 0,25: y además 0,30:    TP=2 FP=1: P = 2/3,  R = 1
#   conf 0,05: todas:            TP=2 FP=2: P = 1/2,  R = 1

CAJA = [0.0, 0.0, 100.0, 100.0]
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


def _detecciones() -> list[dict[str, Any]]:
    """Detecciones del caso de servilleta."""
    return [
        {"image_id": 1, "category_id": 0, "bbox": list(CAJA), "score": 0.90},
        {"image_id": 1, "category_id": 0, "bbox": list(LEJOS), "score": 0.80},
        {"image_id": 2, "category_id": 0, "bbox": list(CAJA), "score": 0.30},
        {"image_id": 2, "category_id": 0, "bbox": list(LEJOS), "score": 0.10},
    ]


@pytest.fixture(scope="module")
def curva() -> dict[str, Any]:
    """Mide la curva del caso de servilleta una sola vez."""
    evaluador = build_evaluator(_referencia(), _detecciones(), "servilleta")
    return operating_curve.precision_recall_at_confidences(
        evaluador, confidences=(0.05, 0.25, 0.50, 0.95), class_names={0: "helmet"}
    )


@pytest.mark.parametrize(
    ("umbral", "precision", "recall", "tp", "fp"),
    [
        (0.95, 1.0, 0.0, 0, 0),
        (0.50, 0.5, 0.5, 1, 1),
        (0.25, 2 / 3, 1.0, 2, 1),
        (0.05, 0.5, 1.0, 2, 2),
    ],
)
def test_precision_y_recall_coinciden_con_la_cuenta_a_mano(
    curva: dict[str, Any], umbral: float, precision: float, recall: float, tp: int, fp: int
) -> None:
    """Los cuatro puntos salen exactamente como en la servilleta."""
    punto = operating_curve.point_at(curva, umbral)
    assert punto["precision"] == pytest.approx(precision, abs=1e-6)
    assert punto["recall"] == pytest.approx(recall, abs=1e-6)
    assert punto["true_positives"] == tp
    assert punto["false_positives"] == fp
    assert punto["num_gt"] == 2


def test_sin_detecciones_por_encima_del_umbral_la_precision_es_uno(
    curva: dict[str, Any],
) -> None:
    """El convenio de precisión 1 y recall 0 se aplica, y el F1 sale cero."""
    punto = operating_curve.point_at(curva, 0.95)
    assert punto["precision"] == 1.0
    assert punto["recall"] == 0.0
    assert punto["f1"] == 0.0


def test_el_recall_no_crece_al_subir_el_umbral(curva: dict[str, Any]) -> None:
    """Subir la confianza solo puede quitar detecciones, nunca añadirlas."""
    recalls = [p["recall"] for p in curva["points"]]
    assert recalls == sorted(recalls, reverse=True)
    confianzas = [p["confidence"] for p in curva["points"]]
    assert confianzas == sorted(confianzas)


def test_la_identidad_con_pycocotools_se_comprueba_y_se_reporta(
    curva: dict[str, Any],
) -> None:
    """La verificación corre siempre y deja constancia de qué comparó."""
    verificacion = curva["verification"]
    assert verificacion["recall_matches_pycocotools"] is True
    assert verificacion["max_difference"] <= operating_curve.RECALL_IDENTITY_TOLERANCE
    assert verificacion["checked_classes"] == 1


def test_el_desglose_por_clase_lleva_el_nombre_y_los_recuentos(
    curva: dict[str, Any],
) -> None:
    detalle = operating_curve.point_at(curva, 0.50)["per_class"]["0"]
    assert detalle["name"] == "helmet"
    assert detalle["true_positives"] == 1
    assert detalle["false_positives"] == 1
    assert detalle["num_gt"] == 2


def test_una_clase_sin_anotaciones_no_entra_en_la_media() -> None:
    """Promediar sobre clases que la partición no contiene falsearía la media."""
    referencia = _referencia()
    referencia["categories"].append({"id": 1, "name": "vest"})
    evaluador = build_evaluator(referencia, _detecciones(), "servilleta")
    curva = operating_curve.precision_recall_at_confidences(evaluador, confidences=(0.25,))
    assert curva["num_classes"] == 1
    assert operating_curve.point_at(curva, 0.25)["recall"] == pytest.approx(1.0)


def test_pedir_un_umbral_que_no_se_midio_falla(curva: dict[str, Any]) -> None:
    with pytest.raises(CocoEvalError, match="no incluye el umbral"):
        operating_curve.point_at(curva, 0.42)


def test_una_evaluacion_sin_acumular_falla() -> None:
    """Sin ``accumulate`` no hay identidad contra la que anclarse."""

    class Falso:
        """Evaluador a medio hacer."""

        eval: dict[str, Any] = {}

    with pytest.raises(CocoEvalError, match="sin acumular"):
        operating_curve.precision_recall_at_confidences(Falso())


def test_la_identidad_falla_si_el_recuento_se_desalinea() -> None:
    """Si la reconstrucción dejara de leer lo mismo, la guarda tiene que saltar."""
    evaluador = build_evaluator(_referencia(), _detecciones(), "servilleta")
    evaluador.eval["recall"] = evaluador.eval["recall"] * 0.5
    with pytest.raises(CocoEvalError, match="no coincide con el de pycocotools"):
        operating_curve.precision_recall_at_confidences(evaluador, confidences=(0.25,))


# ---------------------------------------------------------------------------
# El script: lo que decide antes de medir
# ---------------------------------------------------------------------------


def _cargar_script() -> ModuleType:
    """Carga ``scripts/23_operating_curve.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "23_operating_curve.py"
    especificacion = importlib.util.spec_from_file_location("operating_curve_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.mark.parametrize(
    ("nombre", "esperado"),
    [
        (
            "validator_s_pruning_l1global_r50_iter4_d8fe066f_onnxruntime_fp32",
            ("s_pruning_l1global_r50_iter4_d8fe066f", "onnxruntime", "fp32"),
        ),
        (
            "validator_s_combined_prune-l1global-r50-ptqint8_9067dba2_openvino_int8",
            ("s_combined_prune-l1global-r50-ptqint8_9067dba2", "openvino", "int8"),
        ),
    ],
)
def test_el_troceo_respeta_los_guiones_bajos_del_experiment_id(
    nombre: str, esperado: tuple[str, str, str]
) -> None:
    assert _cargar_script().trocear_directorio(nombre) == esperado


@pytest.mark.parametrize(
    "nombre",
    [
        "validator_onnxruntime_fp32",  # sin experiment_id
        "validator_s_qat_int8-noexcl_d4d403fa_onnxruntime_int8_smoke",
        "otra_cosa",
    ],
)
def test_el_troceo_rechaza_lo_que_no_tiene_esa_forma(nombre: str) -> None:
    assert _cargar_script().trocear_directorio(nombre) is None


# Los tests de la traducción de categorías y del anclaje están en tests/test_reconstruction.py.


def test_el_paso_no_escribe_en_el_registro() -> None:
    """La curva no cabe en una fila, y el paso no lo intenta."""
    fuente = (get_project_paths().root / "scripts" / "23_operating_curve.py").read_text(
        encoding="utf-8"
    )
    assert "ExperimentRegistry" not in fuente
    assert "append_row" not in fuente
