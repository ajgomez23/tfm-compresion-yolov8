"""El paso 34: a qué fila va cada reparto, y qué se niega a adivinar."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import TABLE_PRIMARY_KEYS, is_absolute_path_text
from tfm_edge.evaluation import error_decomposition

RAIZ = get_project_paths().root
GUION = RAIZ / "scripts" / "34_register_error_decomposition.py"


def _cargar_script() -> ModuleType:
    """Carga el paso 34 como módulo."""
    especificacion = importlib.util.spec_from_file_location("paso34", GUION)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _coco_de_sustitucion(directorio: Path) -> Path:
    """Un COCO vacío para los caminos del paso que no llegan a leerlo."""
    ruta = directorio / "coco_que_no_se_lee.json"
    ruta.write_text("{}", encoding="utf-8")
    return ruta


@pytest.fixture(scope="module")
def paso() -> ModuleType:
    """El paso 34 cargado."""
    return _cargar_script()


def _volcado_eval(
    experimento: str,
    motor: str,
    precision: str,
    smoke: bool,
    detecciones: int,
    validador: str,
    split: str = "val",
) -> dict[str, Any]:
    """Un volcado de evaluación mínimo, como los de ``artifact_eval``."""
    return {
        "experiment_id": experimento,
        "runtime": motor,
        "precision": precision,
        "smoke_test": smoke,
        "num_raw_detections": detecciones,
        "validator_dir": validador,
        "coco": {"split": split},
    }


def _fila(
    experimento: str = "s_baseline_default_745e3d08",
    motor: str = "onnxruntime",
    precision: str = "fp32",
    device: str = "workstation_cpu",
    evaluador: str = "pycocotools",
    split: str = "val",
    notas: str = "evaluator=pycocotools",
) -> dict[str, str]:
    """Una fila de ``quality_metrics`` con lo que el paso 34 mira."""
    fila = {
        "experiment_id": experimento,
        "runtime": motor,
        "precision": precision,
        "device_id": device,
        "evaluator": evaluador,
        "split": split,
        "notes": notas,
    }
    for columna in error_decomposition.REGISTRY_COLUMNS.values():
        fila[columna] = ""
    return fila


# ---------------------------------------------------------------------------
# Los volcados de humo, que están ahí para tropezar
# ---------------------------------------------------------------------------
def test_el_volcado_de_humo_se_descarta(paso: ModuleType, tmp_path: Path) -> None:
    """Con un volcado real y su versión de humo, se queda el real."""
    real = _volcado_eval("s_qat_x_1234abcd", "onnxruntime", "int8", False, 116748, "validator_x")
    humo = _volcado_eval("s_qat_x_1234abcd", "onnxruntime", "int8", True, 6279, "validator_x_smoke")
    (tmp_path / "a_val_640.json").write_text(json.dumps(real), encoding="utf-8")
    (tmp_path / "a_val_640_smoke.json").write_text(json.dumps(humo), encoding="utf-8")

    indice = paso.load_eval_sources(tmp_path, "val")
    assert len(indice) == 1
    elegido = indice[("s_qat_x_1234abcd", "onnxruntime", "int8")]
    assert elegido["num_raw_detections"] == 116748
    assert elegido["validator_dir"] == "validator_x"


def test_el_humo_no_se_descarta_por_el_nombre_del_fichero(paso: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "sin_pistas.json").write_text(
        json.dumps(_volcado_eval("s_x_1111aaaa", "openvino", "int8", True, 10, "v")),
        encoding="utf-8",
    )
    assert paso.load_eval_sources(tmp_path, "val") == {}


def test_dos_volcados_reales_con_la_misma_clave_abortan(paso: ModuleType, tmp_path: Path) -> None:
    """Si después de quitar el humo queda más de uno, se para en vez de elegir."""
    for nombre, validador in (("uno.json", "validator_a"), ("dos.json", "validator_b")):
        (tmp_path / nombre).write_text(
            json.dumps(_volcado_eval("s_x_2222bbbb", "openvino", "int8", False, 99, validador)),
            encoding="utf-8",
        )
    with pytest.raises(TFMEdgeError, match="Dos volcados de evaluación"):
        paso.load_eval_sources(tmp_path, "val")


def test_el_volcado_de_test_no_compite_con_el_de_val(paso: ModuleType, tmp_path: Path) -> None:
    val = _volcado_eval("s_x_5555eeee", "openvino", "int8", False, 100, "validator_v")
    test = _volcado_eval("s_x_5555eeee", "openvino", "int8", False, 90, "validator_t", "test")
    del test["coco"]
    test["split"] = "test"
    (tmp_path / "v.json").write_text(json.dumps(val), encoding="utf-8")
    (tmp_path / "t.json").write_text(json.dumps(test), encoding="utf-8")
    clave = ("s_x_5555eeee", "openvino", "int8")
    assert paso.load_eval_sources(tmp_path, "val")[clave]["validator_dir"] == "validator_v"
    assert paso.load_eval_sources(tmp_path, "test")[clave]["validator_dir"] == "validator_t"


def test_un_volcado_sin_particion_no_se_asigna(paso: ModuleType, tmp_path: Path) -> None:
    """Sin partición declarada no se sabe de qué detecciones se habla."""
    volcado = _volcado_eval("s_x_6666ffff", "openvino", "int8", False, 100, "validator_v")
    del volcado["coco"]
    (tmp_path / "v.json").write_text(json.dumps(volcado), encoding="utf-8")
    with pytest.raises(TFMEdgeError, match="no declara su partición"):
        paso.load_eval_sources(tmp_path, "val")


def test_dos_curvas_de_la_misma_evaluacion_abortan(paso: ModuleType, tmp_path: Path) -> None:
    curva = {
        "experiment_id": "s_x_3333cccc",
        "runtime": "onnxruntime",
        "precision": "fp32",
        "source": {"predictions": "results/raw/artifact_eval/validator_x/predictions.json"},
    }
    (tmp_path / "una.json").write_text(json.dumps(curva), encoding="utf-8")
    (tmp_path / "otra.json").write_text(json.dumps(curva), encoding="utf-8")
    with pytest.raises(TFMEdgeError, match="Dos volcados de curva"):
        paso.load_curve_sources(tmp_path)


def test_una_curva_sin_procedencia_no_entra(paso: ModuleType, tmp_path: Path) -> None:
    """Una curva que no graba de dónde salen sus detecciones no sirve de procedencia."""
    (tmp_path / "sin.json").write_text(
        json.dumps(
            {
                "experiment_id": "s_x_4444dddd",
                "runtime": "openvino",
                "precision": "fp32",
                "source": {},
            }
        ),
        encoding="utf-8",
    )
    assert paso.load_curve_sources(tmp_path) == {}


# ---------------------------------------------------------------------------
# A qué fila va cada reparto
# ---------------------------------------------------------------------------
def test_la_curva_manda_sobre_el_volcado(paso: ModuleType) -> None:
    """Con las dos procedencias disponibles se usa la que trae más verificación."""
    clave = ("s_baseline_default_745e3d08", "onnxruntime", "fp32")
    curvas = {clave: {"source": {"predictions": "x/predictions.json"}}}
    volcados = {clave: _volcado_eval(*clave, False, 10, "validator_x")}
    plan, incidencias = paso.plan_rows([_fila()], curvas, volcados, "val", None)
    assert plan[0]["route"] == "curva"
    assert incidencias == []


def test_sin_curva_se_usa_el_volcado(paso: ModuleType) -> None:
    """Sin curva, el ``validator_dir`` del volcado propio vale, con un ancla menos."""
    clave = ("s_baseline_default_745e3d08", "onnxruntime", "fp32")
    volcados = {clave: _volcado_eval(*clave, False, 10, "validator_x")}
    plan, _ = paso.plan_rows([_fila()], {}, volcados, "val", None)
    assert plan[0]["route"] == "volcado"


def test_un_volcado_sin_validator_dir_no_vale(paso: ModuleType) -> None:
    """Si el volcado no dice dónde quedaron las detecciones, la fila se queda vacía."""
    clave = ("s_baseline_default_745e3d08", "onnxruntime", "fp32")
    volcados = {clave: _volcado_eval(*clave, False, 10, "")}
    plan, incidencias = paso.plan_rows([_fila()], {}, volcados, "val", None)
    assert plan[0]["route"] is None
    assert plan[0]["reason"] == "sin_detecciones"
    assert len(incidencias) == 1


@pytest.mark.parametrize(
    "cambio, motivo",
    [
        ({"evaluador": "ultralytics"}, "otro_evaluador"),
        ({"split": "test"}, "otro_split"),
    ],
)
def test_las_filas_que_no_aplican_llevan_su_motivo(
    paso: ModuleType, cambio: dict[str, str], motivo: str
) -> None:
    clave = ("s_baseline_default_745e3d08", "onnxruntime", "fp32")
    curvas = {clave: {"source": {"predictions": "x/predictions.json"}}}
    plan, _ = paso.plan_rows([_fila(**cambio)], curvas, {}, "val", None)
    assert plan[0]["route"] is None
    assert plan[0]["reason"] == motivo


def test_dos_filas_rellenables_con_la_misma_procedencia_abortan(paso: ModuleType) -> None:
    """Falla ante el empate en vez de elegir."""
    clave = ("s_baseline_default_745e3d08", "onnxruntime", "fp32")
    curvas = {clave: {"source": {"predictions": "x/predictions.json"}}}
    filas = [_fila(device="workstation_cpu"), _fila(device="colab_t4_gpu")]
    with pytest.raises(TFMEdgeError, match="apunta a 2 filas rellenables"):
        paso.plan_rows(filas, curvas, {}, "val", None)


def test_el_empate_solo_cuenta_entre_rellenables(paso: ModuleType) -> None:
    """Dos filas con la misma procedencia no chocan si solo una es rellenable."""
    clave = ("s_baseline_default_745e3d08", "pytorch", "fp32")
    curvas = {clave: {"source": {"predictions": "x/predictions.json"}}}
    filas = [
        _fila(motor="pytorch", evaluador="pycocotools"),
        _fila(motor="pytorch", device="auto", evaluador="ultralytics"),
    ]
    plan, _ = paso.plan_rows(filas, curvas, {}, "val", None)
    assert [e["route"] for e in plan] == ["curva", None]


def test_only_deja_fuera_lo_que_no_nombra(paso: ModuleType) -> None:
    """``--only`` limita el trabajo sin marcar las demás como incidencia."""
    clave = ("s_baseline_default_745e3d08", "onnxruntime", "fp32")
    curvas = {clave: {"source": {"predictions": "x/predictions.json"}}}
    plan, incidencias = paso.plan_rows([_fila()], curvas, {}, "val", ["otro_experimento"])
    assert plan[0]["route"] is None
    assert incidencias == []
    assert plan[0].get("skip_note") is False


# ---------------------------------------------------------------------------
# Lo que se escribe en notes
# ---------------------------------------------------------------------------
def test_la_nota_dice_la_ruta_y_el_umbral(paso: ModuleType) -> None:
    """Un recuento sin su umbral no se puede leer, así que el umbral va en la nota."""
    de_curva = paso.note_suffix("curva", None, 0.25)
    de_volcado = paso.note_suffix("volcado", None, 0.25)
    assert "0.25" in de_curva and "0.25" in de_volcado
    assert "contrastada contra la curva" in de_curva
    assert "SIN contraste contra curva" in de_volcado
    assert de_curva != de_volcado


def test_la_nota_de_las_vacias_dice_por_que(paso: ModuleType) -> None:
    """Una celda vacía sin motivo no distingue «no se pudo» de «nadie mira»."""
    for motivo in paso.REASONS:
        texto = paso.note_suffix(None, motivo, 0.25)
        assert texto == paso.REASONS[motivo]
        assert len(texto) > 40


def test_la_nota_no_promete_un_contraste_que_el_codigo_no_hace(paso: ModuleType) -> None:
    """Si la nota de la ruta de curva dice «contrastada», el paso tiene que contrastar."""
    fuente = GUION.read_text(encoding="utf-8")
    if "contrastada contra la curva" in paso.ROUTES["curva"]:
        assert "error_decomposition.cross_check_curve(" in fuente
        assert '"curve_cross_check": contraste' in fuente
    assert "SIN contraste" in paso.ROUTES["volcado"]


def test_un_cuadre_que_no_cierra_no_se_escribe() -> None:
    """El cuadre de detecciones es un informe, y el paso decide sobre él."""
    fuente = GUION.read_text(encoding="utf-8")
    assert 'cuadre["closes_without_remainder"]' in fuente
    assert 'cuadre["matches_recorded_raw_count"]' in fuente


def test_lo_que_se_anade_no_rompe_a_quien_ya_lee_notes(paso: ModuleType) -> None:
    """Otro módulo parsea ``notes``, y este paso la modifica en las ochenta filas."""
    from tfm_edge.evaluation.quality_notes import parse_evaluator_from_notes

    previa = "evaluator=pycocotools; brecha frente al checkpoint map_50_95=-0.00756."
    for sufijo in (
        paso.note_suffix("curva", None, 0.25),
        paso.note_suffix("volcado", None, 0.25),
        paso.note_suffix(None, "sin_detecciones", 0.25),
    ):
        ampliada = f"{previa}; {sufijo}"
        assert parse_evaluator_from_notes(ampliada) == "pycocotools"
        assert previa in ampliada


def test_ningun_texto_de_notes_lleva_una_ruta_absoluta(paso: ModuleType) -> None:
    """``notes`` es la única columna de texto que el validador no mira."""
    for texto in list(paso.ROUTES.values()) + list(paso.REASONS.values()):
        assert not is_absolute_path_text(texto), texto
        assert ";" not in texto, "el punto y coma es el separador de notes"


def test_los_motivos_cubren_todo_lo_que_el_plan_puede_decir(paso: ModuleType) -> None:
    fuente = GUION.read_text(encoding="utf-8")
    en_el_codigo = {
        motivo for motivo in paso.REASONS if f'entrada["reason"] = "{motivo}"' in fuente
    }
    assert en_el_codigo == set(paso.REASONS)


# ---------------------------------------------------------------------------
# Las guardas del paso
# ---------------------------------------------------------------------------
def test_sin_apply_no_se_escribe_nada(tmp_path: Path) -> None:
    """El paso enseña el plan y no toca el registro ni el volcado."""
    registro = tmp_path / "registro"
    registro.mkdir()
    for csv_path in (get_project_paths().results_registry).glob("*.csv"):
        (registro / csv_path.name).write_bytes(csv_path.read_bytes())
    antes = {p.name: p.read_bytes() for p in registro.glob("*.csv")}
    salida = tmp_path / "salida"

    completado = subprocess.run(
        [
            sys.executable,
            str(GUION),
            "--registry-dir",
            str(registro),
            "--output-dir",
            str(salida),
            "--coco",
            str(_coco_de_sustitucion(tmp_path)),
            "--only",
            "no_existe_este_experimento",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(RAIZ),
    )
    assert completado.returncode == 0, completado.stderr
    assert {p.name: p.read_bytes() for p in registro.glob("*.csv")} == antes
    assert not salida.exists()


def test_con_el_registro_sucio_se_para_antes_de_reconstruir(tmp_path: Path) -> None:
    registro = tmp_path / "registro"
    registro.mkdir()
    for csv_path in get_project_paths().results_registry.glob("*.csv"):
        (registro / csv_path.name).write_bytes(csv_path.read_bytes())
    antes = {p.name: p.read_bytes() for p in registro.glob("*.csv")}

    completado = subprocess.run(
        [
            sys.executable,
            str(GUION),
            "--registry-dir",
            str(registro),
            "--output-dir",
            str(tmp_path / "salida"),
            "--coco",
            str(_coco_de_sustitucion(tmp_path)),
            "--apply",
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(RAIZ),
        timeout=60,
    )
    assert completado.returncode == 2, completado.stderr
    assert "no está limpio en git" in completado.stderr
    assert {p.name: p.read_bytes() for p in registro.glob("*.csv")} == antes
    assert not (tmp_path / "salida").exists()


def test_el_paso_escribe_por_update_cells_y_no_por_su_cuenta() -> None:
    """La escritura pasa por la guarda del registro, que valida y normaliza."""
    fuente = GUION.read_text(encoding="utf-8")
    assert fuente.count("update_cells(") >= 2
    assert fuente.count("_write_rows_atomic(TABLE, previas)") == 2
    assert fuente.count("_write_rows_atomic") == 2
    assert "registro.append(" not in fuente


def test_las_notas_se_anaden_y_nunca_sustituyen() -> None:
    """El valor que se escribe en ``notes`` lleva dentro el que había."""
    fuente = GUION.read_text(encoding="utf-8")
    assert 'f"{actual}; {sufijo}" if actual else sufijo' in fuente


def test_un_registro_fuera_del_arbol_no_revienta_el_volcado(
    paso: ModuleType, tmp_path: Path
) -> None:
    parser = paso.build_parser()
    args = parser.parse_args(["--confidence", "0.25"])
    fuera = tmp_path / "registro_desechable" / "quality_metrics.csv"
    texto = paso.build_dump(
        args,
        [0.0, 0.25],
        [_fila()],
        [],
        RAIZ / "results/raw/operating_curve",
        RAIZ / "results/raw/artifact_eval",
        fuera,
        RAIZ / "data/processed/sh17_coco.json",
    )
    volcado = json.loads(texto)
    assert volcado["source"]["registry"].endswith("registro_desechable/quality_metrics.csv")
    assert volcado["source"]["curve_dir"] == "results/raw/operating_curve"
    assert volcado["coverage"]["quality_rows"] == 1
    assert volcado["coverage"]["with_decomposition"] == 0


def test_el_volcado_se_serializa_antes_de_escribir_en_el_registro() -> None:
    """Todo lo que puede fallar se hace antes de tocar una celda."""
    fuente = GUION.read_text(encoding="utf-8")
    cuerpo = fuente[fuente.index("def main(") :]
    assert cuerpo.index("build_dump(") < cuerpo.index("registro.update_cells(")
    despues = cuerpo[cuerpo.index("destino_volcado.write_text") :]
    assert "_write_rows_atomic" in despues, "un volcado que falla tiene que restaurar"


def test_la_confianza_del_registro_es_la_de_referencia_del_proyecto(paso: ModuleType) -> None:
    """No se declara un punto de operación nuevo para esto."""
    from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE

    parser = paso.build_parser()
    valores = {accion.dest: accion.default for accion in parser._actions}  # noqa: SLF001
    assert valores["confidence"] == REFERENCE_CONFIDENCE
    assert valores["apply"] is False
    assert valores["on_conflict"] == "fail"


def test_el_nombre_del_volcado_lleva_la_confianza(paso: ModuleType) -> None:
    """Dos confianzas distintas no se pisan el volcado."""
    from tfm_edge.evaluation import error_decomposition

    assert error_decomposition.dump_name(0.25) == "error_decomposition_conf025.json"
    assert error_decomposition.dump_name(0.5) == "error_decomposition_conf050.json"
    assert error_decomposition.dump_name(0.0) == "error_decomposition_conf000.json"
    assert not hasattr(paso, "dump_name"), "el paso 34 vuelve a tener su propia copia"


def test_la_clave_que_usa_el_paso_es_la_del_registro(paso: ModuleType) -> None:
    fuente = GUION.read_text(encoding="utf-8")
    assert "TABLE_PRIMARY_KEYS[TABLE]" in fuente
    assert len(TABLE_PRIMARY_KEYS["quality_metrics"]) == 6


# ---------------------------------------------------------------------------
# Lo que quedó escrito en el registro
# ---------------------------------------------------------------------------
def test_el_registro_declara_la_ruta_de_cada_cifra() -> None:
    """Toda fila con las seis columnas puestas dice por qué ruta entró."""
    csv_path = get_project_paths().results_registry / "quality_metrics.csv"
    filas = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    columnas = list(error_decomposition.REGISTRY_COLUMNS.values())
    if not any(all((f[c] or "").strip() for c in columnas) for f in filas):
        pytest.skip("El paso 34 no se ha ejecutado sobre este registro todavía.")

    for fila in filas:
        llena = all((fila[c] or "").strip() for c in columnas)
        vacia = not any((fila[c] or "").strip() for c in columnas)
        if fila["split"] != "val":
            # Las filas de test se añadieron después de la última ejecución del paso 34, así que no
            # llevan reparto ni motivo.
            assert vacia, "el paso 34 no ha reconstruido nunca otra partición que val"
            continue
        assert llena or vacia, "una fila a medias no dice a qué umbral pertenece"
        notas = fila["notes"] or ""
        if llena:
            assert "descomposicion de errores (paso 34)" in notas
            assert ("contrastada contra la curva" in notas) or (
                "SIN contraste contra curva" in notas
            )
        else:
            assert "sin descomposicion de errores" in notas
