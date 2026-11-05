"""Tests de la derivación de las tablas del punto de operación."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.paths import get_project_paths
from tfm_edge.evaluation.coco_eval import CocoEvalError
from tfm_edge.evaluation.operating_analysis import (
    KIND_PRUNING,
    KIND_QUANTIZATION,
    OperatingAnalysisError,
    analyze,
    axis_delta,
    best_f1_point,
    derive_pairs,
    detection_shift,
    load_curve_dumps,
    retune_recovery,
    unused_curves,
    variant_of,
)

# Fixtures con números elegidos para que las cuentas se puedan hacer a mano:
#
#   antes,  conf 0,25:  TP=60  FP=20: P=60/80=0,75   R=60/100=0,60
#   despues,conf 0,25:  TP=50  FP=50: P=50/100=0,50  R=50/100=0,50
#
#   dP = -0,25   dR = -0,10: cae mas la precision
#   detecciones: de 80 a 100, es decir +25 %;  dTP = -10, dFP = +30


def _punto(conf: float, tp: int, fp: int, num_gt: int = 100) -> dict[str, Any]:
    """Construye un punto de curva con precisión, recall y F1 coherentes."""
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "confidence": conf,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "num_gt": num_gt,
    }


def _curva(
    experiment_id: str,
    runtime: str,
    precision: str,
    puntos: list[dict[str, Any]],
    anclada: bool = True,
    recall_verificado: bool = True,
) -> dict[str, Any]:
    """Construye un volcado de curva con la forma que escribe el paso 23."""
    return {
        "experiment_id": experiment_id,
        "runtime": runtime,
        "precision": precision,
        "anchor": {"anchored": anclada, "max_difference": 0.0},
        "curve": {
            "points": puntos,
            "verification": {"recall_matches_pycocotools": recall_verificado},
        },
    }


def _par_de_prueba() -> tuple[dict[str, Any], dict[str, Any]]:
    """Devuelve las dos curvas de la cuenta hecha a mano."""
    antes = _curva(
        "s_baseline_default_00000000",
        "onnxruntime",
        "fp32",
        [_punto(0.10, 70, 60), _punto(0.25, 60, 20), _punto(0.50, 30, 5)],
    )
    despues = _curva(
        "s_ptq_int8-noexcl_11111111",
        "onnxruntime",
        "int8",
        [_punto(0.10, 65, 120), _punto(0.25, 50, 50), _punto(0.50, 40, 20)],
    )
    return antes, despues


def _volcar(directorio: Path, curva: dict[str, Any]) -> Path:
    """Escribe un volcado de curva en un directorio."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{curva['experiment_id']}_{curva['runtime']}_{curva['precision']}.json"
    ruta.write_text(json.dumps(curva), encoding="utf-8")
    return ruta


# --- La variante se trocea posicionalmente ----------------------------------


@pytest.mark.parametrize(
    ("experiment_id", "esperado"),
    [
        ("s_baseline_default_a1b2c3d4", "default"),
        ("s_pruning_l1global_r50_9f8e7d6c", "l1global_r50"),
        ("s_pruning_l1global_r50_iter4_d8fe066f", "l1global_r50_iter4"),
        ("s_combined_prune50-ptqint8_11aa22bb", "prune50-ptqint8"),
    ],
)
def test_la_variante_se_lee_aunque_lleve_guiones_bajos(experiment_id: str, esperado: str) -> None:
    assert variant_of(experiment_id) == esperado


# --- Las guardas de los volcados de entrada ---------------------------------


def test_una_curva_sin_anclar_no_se_deriva(tmp_path: Path) -> None:
    """Derivar de un volcado sin ancla lo blanquearía."""
    _volcar(
        tmp_path,
        _curva(
            "s_baseline_default_00000000",
            "onnxruntime",
            "fp32",
            [_punto(0.25, 1, 1)],
            anclada=False,
        ),
    )
    with pytest.raises(OperatingAnalysisError, match="no está anclado"):
        load_curve_dumps(tmp_path)


def test_una_curva_sin_la_identidad_del_recall_no_se_deriva(tmp_path: Path) -> None:
    """El recuento propio sobre evalImgs solo vale si coincide con accumulate."""
    _volcar(
        tmp_path,
        _curva(
            "s_baseline_default_00000000",
            "onnxruntime",
            "fp32",
            [_punto(0.25, 1, 1)],
            recall_verificado=False,
        ),
    )
    with pytest.raises(OperatingAnalysisError, match="identidad del recall"):
        load_curve_dumps(tmp_path)


def test_un_directorio_vacio_es_un_error_y_no_una_tabla_vacia(tmp_path: Path) -> None:
    """Sin curvas no hay derivación; devolver nada en silencio sería peor."""
    tmp_path.joinpath("curvas").mkdir()
    with pytest.raises(OperatingAnalysisError, match="No hay ningún volcado"):
        load_curve_dumps(tmp_path / "curvas")


def test_las_dos_precisiones_de_un_experimento_conviven(tmp_path: Path) -> None:
    _volcar(
        tmp_path,
        _curva("s_qat_int8-noexcl_22222222", "onnxruntime", "fp32", [_punto(0.25, 60, 20)]),
    )
    _volcar(
        tmp_path,
        _curva("s_qat_int8-noexcl_22222222", "onnxruntime", "int8", [_punto(0.25, 50, 50)]),
    )
    curvas = load_curve_dumps(tmp_path)
    assert len(curvas) == 2
    assert (
        curvas[("s_qat_int8-noexcl_22222222", "onnxruntime", "fp32")]["curve"]["points"][0][
            "true_positives"
        ]
        == 60
    )
    assert (
        curvas[("s_qat_int8-noexcl_22222222", "onnxruntime", "int8")]["curve"]["points"][0][
            "true_positives"
        ]
        == 50
    )


# --- Qué se compara con qué sale del registro -------------------------------


def _filas_de_registro() -> list[dict[str, str]]:
    """Devuelve filas de registro con las tres clases de parentesco."""
    return [
        {
            "experiment_id": "s_baseline_default_00000000",
            "technique": "baseline",
            "parent_experiment_id": "",
            "status": "ok",
        },
        {
            "experiment_id": "s_qat_int8-noexcl_22222222",
            "technique": "qat",
            "parent_experiment_id": "s_baseline_default_00000000",
            "status": "ok",
        },
        {
            "experiment_id": "s_pruning_l1global_r50_33333333",
            "technique": "pruning",
            "parent_experiment_id": "s_baseline_default_00000000",
            "status": "ok",
        },
        {
            "experiment_id": "s_ptq_int8-noexcl_44444444",
            "technique": "ptq",
            "parent_experiment_id": "s_baseline_default_00000000",
            "status": "fallido",
        },
    ]


def _curvas_de_registro() -> dict[tuple[str, str, str], dict[str, Any]]:
    """Índice de curvas coherente con :func:`_filas_de_registro`, incluida la FP32 del QAT."""
    curvas = {}
    for experimento, precision in (
        ("s_baseline_default_00000000", "fp32"),
        ("s_qat_int8-noexcl_22222222", "fp32"),
        ("s_qat_int8-noexcl_22222222", "int8"),
        ("s_pruning_l1global_r50_33333333", "fp32"),
    ):
        clave = (experimento, "onnxruntime", precision)
        curvas[clave] = _curva(experimento, "onnxruntime", precision, [_punto(0.25, 60, 20)])
    return curvas


def test_un_par_de_cuantizacion_coge_el_int8_del_hijo_y_no_su_fp32() -> None:
    pares = derive_pairs(_filas_de_registro(), _curvas_de_registro())
    cuantizacion = [p for p in pares if p["kind"] == KIND_QUANTIZATION]
    assert len(cuantizacion) == 1
    par = cuantizacion[0]
    assert par["before"] == "s_baseline_default_00000000"
    assert par["before_precision"] == "fp32"
    assert par["after"] == "s_qat_int8-noexcl_22222222"
    assert par["after_precision"] == "int8"


def test_un_par_de_poda_se_queda_en_fp32_por_los_dos_lados() -> None:
    """Poda y cuantización se miden por separado o no se atribuyen a nada."""
    pares = derive_pairs(_filas_de_registro(), _curvas_de_registro())
    poda = [p for p in pares if p["kind"] == KIND_PRUNING]
    assert len(poda) == 1
    assert poda[0]["before_precision"] == poda[0]["after_precision"] == "fp32"
    assert poda[0]["label"] == "l1global_r50"


def test_una_fila_fallida_no_produce_par() -> None:
    """Un experimento en estado fallido no entra en ninguna comparación."""
    curvas = _curvas_de_registro()
    curvas[("s_ptq_int8-noexcl_44444444", "onnxruntime", "int8")] = _curva(
        "s_ptq_int8-noexcl_44444444", "onnxruntime", "int8", [_punto(0.25, 1, 1)]
    )
    pares = derive_pairs(_filas_de_registro(), curvas)
    assert all("44444444" not in p["after"] for p in pares)


def test_las_curvas_que_no_entran_en_ningun_par_se_declaran() -> None:
    """Que una medición no se use es legítimo; que no se note, no."""
    curvas = _curvas_de_registro()
    pares = derive_pairs(_filas_de_registro(), curvas)
    assert unused_curves(curvas, pares) == [("s_qat_int8-noexcl_22222222", "onnxruntime", "fp32")]


# --- La aritmética, contra una cuenta hecha a mano --------------------------


def test_el_delta_de_ejes_sale_a_la_cuenta_hecha_a_mano() -> None:
    antes, despues = _par_de_prueba()
    eje = axis_delta(antes, despues, 0.25)
    assert eje["precision_before"] == pytest.approx(0.75)
    assert eje["precision_after"] == pytest.approx(0.50)
    assert eje["delta_precision"] == pytest.approx(-0.25)
    assert eje["delta_recall"] == pytest.approx(-0.10)
    assert eje["dominant_axis"] == "precision"


def test_el_desplazamiento_cuenta_detecciones_y_las_separa() -> None:
    antes, despues = _par_de_prueba()
    shift = detection_shift(antes, despues, 0.25)
    assert shift["detections_before"] == 80
    assert shift["detections_after"] == 100
    assert shift["relative_change"] == pytest.approx(0.25)
    assert shift["delta_true_positives"] == -10
    assert shift["delta_false_positives"] == 30


def test_sin_detecciones_de_partida_el_cambio_relativo_queda_vacio() -> None:
    antes = _curva("a_baseline_x_00000000", "onnxruntime", "fp32", [_punto(0.25, 0, 0)])
    despues = _curva("a_ptq_x_11111111", "onnxruntime", "int8", [_punto(0.25, 5, 5)])
    assert detection_shift(antes, despues, 0.25)["relative_change"] is None


def test_el_optimo_es_el_maximo_de_la_rejilla_medida() -> None:
    antes, _ = _par_de_prueba()
    mejor = best_f1_point(antes)
    esperados = [p["f1"] for p in antes["curve"]["points"]]
    assert mejor["f1"] == max(esperados)


def test_el_reajuste_mide_la_brecha_en_los_dos_optimos() -> None:
    """La fracción recuperada es cuánto de la brecha fija desaparece al reajustar."""
    antes, despues = _par_de_prueba()
    r = retune_recovery(antes, despues, 0.25)
    brecha_fija = despues["curve"]["points"][1]["f1"] - antes["curve"]["points"][1]["f1"]
    brecha_reaj = best_f1_point(despues)["f1"] - best_f1_point(antes)["f1"]
    assert r["f1_gap_fixed"] == pytest.approx(brecha_fija)
    assert r["f1_gap_retuned"] == pytest.approx(brecha_reaj)
    assert r["recovered_fraction"] == pytest.approx(1 - brecha_reaj / brecha_fija)
    assert r["optimum_moves"] in ("sube", "baja", "igual")


def test_sin_brecha_al_umbral_fijo_no_hay_fraccion_recuperada() -> None:
    """Dividir por algo que tiende a cero no da un porcentaje, da ruido."""
    antes = _curva("a_baseline_x_00000000", "onnxruntime", "fp32", [_punto(0.25, 60, 20)])
    despues = _curva("a_ptq_x_11111111", "onnxruntime", "int8", [_punto(0.25, 60, 20)])
    assert retune_recovery(antes, despues, 0.25)["recovered_fraction"] is None


def test_un_umbral_no_medido_se_rechaza_en_vez_de_interpolarse() -> None:
    antes, despues = _par_de_prueba()
    with pytest.raises(CocoEvalError, match="no incluye el umbral"):
        axis_delta(antes, despues, 0.33)


# --- La derivación completa -------------------------------------------------


def test_el_analisis_completo_resume_lo_que_ha_derivado() -> None:
    curvas = _curvas_de_registro()
    pares = derive_pairs(_filas_de_registro(), curvas)
    analisis = analyze(curvas, pares, 0.25, (0.25,))
    assert analisis["summary"][KIND_QUANTIZATION]["pairs"] == 1
    assert analisis["summary"][KIND_PRUNING]["pairs"] == 1
    assert analisis["unused_curves"] == [
        {
            "experiment_id": "s_qat_int8-noexcl_22222222",
            "runtime": "onnxruntime",
            "precision": "fp32",
        }
    ]


def test_un_par_que_referencia_una_curva_ausente_para_la_derivacion() -> None:
    curvas = _curvas_de_registro()
    pares = derive_pairs(_filas_de_registro(), curvas)
    del curvas[("s_baseline_default_00000000", "onnxruntime", "fp32")]
    with pytest.raises(OperatingAnalysisError, match="referencia una curva que no está"):
        analyze(curvas, pares, 0.25, (0.25,))


def test_la_derivacion_es_determinista() -> None:
    """Con las mismas curvas, el volcado sale byte a byte igual."""
    curvas = _curvas_de_registro()
    pares = derive_pairs(_filas_de_registro(), curvas)
    primera = json.dumps(analyze(curvas, pares, 0.25, (0.25,)), indent=2, ensure_ascii=False)
    segunda = json.dumps(analyze(curvas, pares, 0.25, (0.25,)), indent=2, ensure_ascii=False)
    assert primera == segunda


# --- El script --------------------------------------------------------------


def _cargar_script() -> ModuleType:
    """Carga ``scripts/24_operating_analysis.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "24_operating_analysis.py"
    especificacion = importlib.util.spec_from_file_location("operating_analysis_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.mark.parametrize(
    ("confianza", "esperado"),
    [(0.25, "operating_analysis_conf025.json"), (0.5, "operating_analysis_conf050.json")],
)
def test_el_umbral_va_en_el_nombre_del_volcado(confianza: float, esperado: str) -> None:
    assert _cargar_script().nombre_volcado(confianza) == esperado


def test_el_script_no_escribe_en_el_registro() -> None:
    ruta = get_project_paths().root / "scripts" / "24_operating_analysis.py"
    texto = ruta.read_text(encoding="utf-8")
    for prohibido in ("append_row", "append_quality", "write_rows", "upsert"):
        assert prohibido not in texto, f"el paso 24 no escribe en el registro ({prohibido})"
    assert "read_rows" in texto
