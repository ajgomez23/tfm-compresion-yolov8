"""Tests de que los scripts llaman a métodos del registro que existen."""

from __future__ import annotations

import ast
import csv
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.environment import ENVIRONMENT_COLUMNS
from tfm_edge.common.exceptions import RegistrySchemaError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import (
    DEPLOYMENT_COLUMNS,
    EXPERIMENTS_COLUMNS,
    VALID_DUPLICATE_POLICIES,
    ExperimentRegistry,
    build_quality_columns,
)
from tfm_edge.evaluation.coco_eval import (
    CRITERIA,
    PRIMARY_CRITERION,
    CocoEvaluationResult,
    CriterionMetrics,
)

#: Nombre de la clase cuyos métodos se comprueban. Se resuelve por anotación y
#: por asignación, no por el nombre de la variable: llamar ``registro`` a la
#: variable es una convención del proyecto, no algo en lo que apoyarse.
CLASE_REGISTRO = "ExperimentRegistry"

#: Clases de mentira para los tests. El registro acepta cualquier lista de
#: nombres; usar las de SH17 ataría este fichero al dataset sin ganar nada.
CLASES: tuple[str, ...] = ("person", "helmet", "gloves")


def _scripts() -> list[Path]:
    """Devuelve los scripts de CLI del proyecto."""
    return sorted(get_project_paths().root.glob("scripts/*.py"))


def _nombres_del_registro(arbol: ast.Module) -> set[str]:
    """Localiza las variables que contienen un ``ExperimentRegistry``."""
    nombres: set[str] = set()

    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Assign):
            valor = nodo.value
            if (
                isinstance(valor, ast.Call)
                and isinstance(valor.func, ast.Name)
                and valor.func.id == CLASE_REGISTRO
            ):
                for destino in nodo.targets:
                    if isinstance(destino, ast.Name):
                        nombres.add(destino.id)
        elif isinstance(nodo, ast.arg) and nodo.annotation is not None:
            anotacion = nodo.annotation
            # Vale tanto `x: ExperimentRegistry` como `x: "ExperimentRegistry"`.
            if isinstance(anotacion, ast.Name) and anotacion.id == CLASE_REGISTRO:
                nombres.add(nodo.arg)
            elif isinstance(anotacion, ast.Constant) and anotacion.value == CLASE_REGISTRO:
                nombres.add(nodo.arg)

    return nombres


def _metodos_invocados(arbol: ast.Module, nombres: set[str]) -> set[tuple[str, int]]:
    """Recoge los métodos que se invocan sobre un registro."""
    invocados: set[tuple[str, int]] = set()
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        funcion = nodo.func
        if (
            isinstance(funcion, ast.Attribute)
            and isinstance(funcion.value, ast.Name)
            and funcion.value.id in nombres
        ):
            invocados.add((funcion.attr, funcion.lineno))
    return invocados


class TestQueLosScriptsLlamanAMetodosQueExisten:
    """Ningún script puede llamar a un método que el registro no tiene."""

    def test_ningun_script_llama_a_un_metodo_inexistente(self) -> None:
        """El invariante, sobre todos los scripts a la vez."""
        disponibles = set(dir(ExperimentRegistry))
        fallos: list[str] = []

        for script in _scripts():
            arbol = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            nombres = _nombres_del_registro(arbol)
            if not nombres:
                continue
            for metodo, linea in sorted(_metodos_invocados(arbol, nombres)):
                if metodo not in disponibles:
                    fallos.append(f"{script.name}:{linea} llama a '.{metodo}()', que no existe")

        assert not fallos, "Métodos del registro que no existen:\n  " + "\n  ".join(fallos)

    def test_la_comprobacion_ve_de_verdad_los_scripts_que_escriben(self) -> None:
        analizados: dict[str, set[str]] = {}
        for script in _scripts():
            arbol = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            nombres = _nombres_del_registro(arbol)
            if nombres:
                analizados[script.name] = {
                    metodo for metodo, _ in _metodos_invocados(arbol, nombres)
                }

        assert "16_qat.py" in analizados, "El script del fallo no se está analizando."
        assert (
            "append_quality_metrics" in analizados["16_qat.py"]
        ), "16_qat.py no inscribe la fila de calidad, o la llamada no es visible aquí."

    def test_y_detecta_un_metodo_inventado(self) -> None:
        codigo = (
            "from tfm_edge.common.registry import ExperimentRegistry\n"
            "registro = ExperimentRegistry(registry_dir=None, class_names=[])\n"
            "registro.append_quality({}, on_duplicate='fail')\n"
        )
        arbol = ast.parse(codigo)
        nombres = _nombres_del_registro(arbol)
        assert nombres == {"registro"}
        metodos = {metodo for metodo, _ in _metodos_invocados(arbol, nombres)}
        assert metodos == {"append_quality"}
        assert "append_quality" not in set(dir(ExperimentRegistry))


#: Métodos del registro que reciben una fila, y la tabla en la que la escriben.
METODO_A_TABLA: dict[str, str] = {
    "append_experiment": "experiments",
    "append_quality_metrics": "quality_metrics",
    "append_deployment_metrics": "deployment_metrics",
    "append_environment": "environments",
}


def _columnas_de(tabla: str) -> set[str]:
    """Devuelve el esquema de una tabla sin construir un registro."""
    if tabla == "experiments":
        return set(EXPERIMENTS_COLUMNS)
    if tabla == "deployment_metrics":
        return set(DEPLOYMENT_COLUMNS)
    if tabla == "environments":
        return set(ENVIRONMENT_COLUMNS)
    return set(build_quality_columns(CLASES))


def _diccionarios_asignados(arbol: ast.Module) -> dict[str, ast.Dict]:
    """Mapea cada variable asignada a un diccionario literal con ese literal."""
    asignados: dict[str, ast.Dict] = {}
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Assign) or not isinstance(nodo.value, ast.Dict):
            continue
        for destino in nodo.targets:
            if isinstance(destino, ast.Name):
                asignados[destino.id] = nodo.value
    return asignados


def _literal_de(argumento: ast.expr, asignados: dict[str, ast.Dict]) -> ast.Dict | None:
    """Encuentra el diccionario literal detrás del primer argumento."""
    if isinstance(argumento, ast.Dict):
        return argumento
    if isinstance(argumento, ast.Name):
        return asignados.get(argumento.id)
    if isinstance(argumento, ast.Call):
        for anidado in argumento.args:
            encontrado = _literal_de(anidado, asignados)
            if encontrado is not None:
                return encontrado
    return None


def _filas_literales(arbol: ast.Module, nombres: set[str]) -> list[tuple[str, int, set[str]]]:
    """Recoge las filas que se pueden resolver leyendo el AST."""
    asignados = _diccionarios_asignados(arbol)
    encontradas: list[tuple[str, int, set[str]]] = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call) or not isinstance(nodo.func, ast.Attribute):
            continue
        if not isinstance(nodo.func.value, ast.Name) or nodo.func.value.id not in nombres:
            continue
        tabla = METODO_A_TABLA.get(nodo.func.attr)
        if tabla is None or not nodo.args:
            continue
        literal = _literal_de(nodo.args[0], asignados)
        if literal is None:
            continue
        claves = {
            clave.value
            for clave in literal.keys
            if isinstance(clave, ast.Constant) and isinstance(clave.value, str)
        }
        encontradas.append((tabla, nodo.lineno, claves))
    return encontradas


class TestQueLasColumnasQueEscribenLosScriptsExisten:
    """Las columnas que escriben los scripts existen en el esquema."""

    @staticmethod
    def _analizar() -> list[tuple[str, str, int, set[str]]]:
        """Recoge las filas literales de todos los scripts."""
        recogidas: list[tuple[str, str, int, set[str]]] = []
        for script in _scripts():
            arbol = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            nombres = _nombres_del_registro(arbol)
            for tabla, linea, claves in _filas_literales(arbol, nombres):
                recogidas.append((script.name, tabla, linea, claves))
        return recogidas

    def test_ninguna_columna_literal_es_desconocida(self) -> None:
        """El invariante, sobre todos los scripts a la vez."""
        fallos: list[str] = []
        for script, tabla, linea, claves in self._analizar():
            sobran = claves - _columnas_de(tabla)
            # Las columnas de AP por clase dependen del dataset y nadie las escribe a mano, así que
            # se comprueba que no aparecen.
            sobran = {columna for columna in sobran if not columna.startswith("ap_class_")}
            if sobran:
                fallos.append(
                    f"{script}:{linea} escribe en '{tabla}' columnas que no existen: "
                    f"{', '.join(sorted(sobran))}"
                )
        assert not fallos, "Columnas desconocidas:\n  " + "\n  ".join(fallos)

    def test_la_comprobacion_ve_filas_de_verdad(self) -> None:
        analizadas = self._analizar()
        assert len(analizadas) >= 4, f"Solo se han visto {len(analizadas)} filas literales."
        scripts = {script for script, _, _, _ in analizadas}
        assert "06_evaluate_coco.py" in scripts
        assert any(len(claves) > 10 for _, _, _, claves in analizadas)

    def test_y_detecta_una_columna_inventada(self) -> None:
        codigo = (
            "from tfm_edge.common.registry import ExperimentRegistry\n"
            "registro = ExperimentRegistry(registry_dir=None, class_names=[])\n"
            "registro.append_experiment({'experiment_id': 'x', 'map': 0.1})\n"
        )
        arbol = ast.parse(codigo)
        filas = _filas_literales(arbol, _nombres_del_registro(arbol))
        assert len(filas) == 1
        tabla, _, claves = filas[0]
        assert claves - _columnas_de(tabla) == {"map"}


def _cargar_16_qat() -> ModuleType:
    """Carga ``scripts/16_qat.py`` como módulo, sin ejecutar su ``main``."""
    ruta = get_project_paths().root / "scripts" / "16_qat.py"
    especificacion = importlib.util.spec_from_file_location("qat_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _metricas(criterio: str, desplazamiento: float = 0.0) -> CriterionMetrics:
    """Construye unas métricas de un criterio con valores reconocibles."""
    return CriterionMetrics(
        criterion=criterio,
        map_50_95=0.371 + desplazamiento,
        map_50=0.564 + desplazamiento,
        ap_75=0.392 + desplazamiento,
        ap_small=0.101 + desplazamiento,
        ap_medium=0.302 + desplazamiento,
        ap_large=0.503 + desplazamiento,
        ar_1=0.201 + desplazamiento,
        ar_10=0.402 + desplazamiento,
        ar_100=0.451 + desplazamiento,
        ar_small=0.151 + desplazamiento,
        ar_medium=0.352 + desplazamiento,
        ar_large=0.553 + desplazamiento,
        objects_per_size={"small": 10, "medium": 20, "large": 30},
        precision_at_best_f1=0.62,
        recall_at_best_f1=0.55,
        f1_best=0.58,
        conf_at_best_f1=0.31,
    )


def _informe_eval() -> dict[str, Any]:
    """Construye el informe que devuelve ``evaluate_checkpoint``."""
    resultado = CocoEvaluationResult(
        split="val",
        input_resolution=640,
        num_images=1191,
        num_annotations=12345,
        num_detections=45678,
        by_criterion={
            PRIMARY_CRITERION: _metricas(PRIMARY_CRITERION),
            "original_resolution": _metricas("original_resolution", desplazamiento=0.001),
        },
        per_class={indice: {"ap_50_95": 0.3 + indice / 100} for indice in range(len(CLASES))},
        consistency={"max_abs_diff": 0.0},
    )
    return {
        "result": resultado,
        "registry_columns": {
            **resultado.to_registry_columns(),
            "input_resolution": 640,
            "conf_threshold": 0.001,
            "iou_threshold": 0.7,
            "split": "val",
            "eval_time_s": 123.45,
        },
        "num_matched_detections": 45678,
    }


class TestQueLaFilaDeCalidadDeQatSeEscribe:
    """La última línea de ``16_qat.py``, recorrida de verdad."""

    @staticmethod
    def _fila(script: ModuleType, tmp_path: Path) -> dict[str, Any]:
        """Construye la fila de calidad con el código real del script."""
        return script.construir_fila_calidad(
            experiment_id="s_qat_int8-noexcl_43d64af3",
            informe_eval=_informe_eval(),
            clases=list(CLASES),
            device_id="colab_t4",
            environment_id="0123456789abcdef",
            ruta_detalle=tmp_path / "detalle.json",
        )

    def test_la_fila_se_escribe_y_se_lee_de_vuelta(self, tmp_path: Path) -> None:
        script = _cargar_16_qat()
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=list(CLASES))

        escrita = registro.append_quality_metrics(self._fila(script, tmp_path), on_duplicate="fail")
        assert escrita is True

        csv_calidad = tmp_path / "quality_metrics.csv"
        assert csv_calidad.is_file(), "No se ha creado quality_metrics.csv."
        with csv_calidad.open(encoding="utf-8", newline="") as manejador:
            filas = list(csv.DictReader(manejador))

        assert len(filas) == 1
        fila = filas[0]
        assert fila["experiment_id"] == "s_qat_int8-noexcl_43d64af3"
        assert fila["runtime"] == "pytorch"
        assert fila["precision"] == "fp32"
        assert float(fila["map_50_95"]) == pytest.approx(0.371)
        assert fila["status"] == "ok"

    def test_el_ap_por_clase_llega_a_sus_columnas(self, tmp_path: Path) -> None:
        """Las columnas por clase no pueden quedarse en el camino."""
        script = _cargar_16_qat()
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=list(CLASES))
        registro.append_quality_metrics(self._fila(script, tmp_path), on_duplicate="fail")

        with (tmp_path / "quality_metrics.csv").open(encoding="utf-8", newline="") as manejador:
            fila = next(iter(csv.DictReader(manejador)))

        for indice, nombre in enumerate(CLASES):
            columna = f"ap_class_{indice:02d}_{nombre}"
            assert columna in fila, f"Falta la columna '{columna}'."
            assert float(fila[columna]) == pytest.approx(0.3 + indice / 100)

    def test_el_desglose_por_tamano_lleva_los_dos_criterios(self, tmp_path: Path) -> None:
        script = _cargar_16_qat()
        fila = self._fila(script, tmp_path)

        for sufijo in CRITERIA.values():
            for tamano in ("small", "medium", "large"):
                for prefijo in ("ap", "ar"):
                    columna = f"{prefijo}_{tamano}_{sufijo}"
                    assert columna in fila, f"Falta la columna '{columna}'."
                    assert fila[columna] != "", f"'{columna}' se ha quedado vacía."

    def test_la_nota_dice_que_esto_no_es_el_int8(self, tmp_path: Path) -> None:
        script = _cargar_16_qat()
        fila = self._fila(script, tmp_path)
        assert "NO lo que rinde el INT8" in fila["notes"]
        assert fila["precision"] == "fp32"


def _politicas_ofrecidas(arbol: ast.Module) -> list[tuple[int, list[str] | None]]:
    """Recoge los ``choices`` del argumento ``--on-duplicate`` de un script."""
    encontradas: list[tuple[int, list[str] | None]] = []
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call) or not isinstance(nodo.func, ast.Attribute):
            continue
        if nodo.func.attr != "add_argument":
            continue
        if not nodo.args or not isinstance(nodo.args[0], ast.Constant):
            continue
        if nodo.args[0].value != "--on-duplicate":
            continue
        for clave in nodo.keywords:
            if clave.arg != "choices":
                continue
            valor = clave.value
            if isinstance(valor, (ast.List, ast.Tuple)):
                encontradas.append(
                    (
                        nodo.lineno,
                        [
                            elemento.value
                            for elemento in valor.elts
                            if isinstance(elemento, ast.Constant)
                        ],
                    )
                )
            else:
                encontradas.append((nodo.lineno, None))
    return encontradas


class TestQueLasPoliticasQueOfrecenLosScriptsExisten:
    """Los ``choices`` de política de duplicados coinciden con las del registro."""

    @staticmethod
    def _analizar() -> list[tuple[str, int, list[str] | None]]:
        """Recoge los ``choices`` de ``--on-duplicate`` de todos los scripts."""
        recogidas: list[tuple[str, int, list[str] | None]] = []
        for script in _scripts():
            arbol = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
            for linea, opciones in _politicas_ofrecidas(arbol):
                recogidas.append((script.name, linea, opciones))
        return recogidas

    def test_ninguna_politica_ofrecida_es_desconocida(self) -> None:
        """El invariante, sobre todos los scripts a la vez."""
        fallos: list[str] = []
        for script, linea, opciones in self._analizar():
            if opciones is None:
                continue
            sobran = set(opciones) - set(VALID_DUPLICATE_POLICIES)
            if sobran:
                fallos.append(
                    f"{script}:{linea} ofrece políticas que el registro rechaza: "
                    f"{', '.join(sorted(sobran))}"
                )
        assert not fallos, "Políticas inexistentes en --on-duplicate:\n  " + "\n  ".join(fallos)

    def test_la_comprobacion_ve_los_scripts_que_la_ofrecen(self) -> None:
        analizadas = self._analizar()
        scripts = {script for script, _, _ in analizadas}
        assert len(scripts) >= 8, f"Solo se han visto {len(scripts)} scripts con --on-duplicate."
        assert {
            "07_benchmark.py",
            "10_quantize_ptq.py",
        } <= scripts, "Los dos scripts afectados no están entre los analizados."

    def test_los_dos_scripts_del_fallo_derivan_sus_opciones(self) -> None:
        for script, _, opciones in self._analizar():
            if script in {"07_benchmark.py", "10_quantize_ptq.py"}:
                assert opciones is None, (
                    f"{script} vuelve a escribir los choices a mano: {opciones}. "
                    "Tienen que salir de VALID_DUPLICATE_POLICIES."
                )

    def test_y_detecta_una_politica_inventada(self) -> None:
        codigo = (
            "import argparse\n"
            "parser = argparse.ArgumentParser()\n"
            "parser.add_argument('--on-duplicate', choices=['overwrite', 'skip', 'error'])\n"
        )
        encontradas = _politicas_ofrecidas(ast.parse(codigo))
        assert len(encontradas) == 1
        _, opciones = encontradas[0]
        assert opciones is not None
        assert set(opciones) - set(VALID_DUPLICATE_POLICIES) == {"error"}

    def test_el_registro_acepta_de_verdad_todas_las_politicas(self, tmp_path: Path) -> None:
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=list(CLASES))
        for politica in VALID_DUPLICATE_POLICIES:
            with pytest.raises(RegistrySchemaError):
                registro.append("experiments", {}, on_duplicate=politica)

    def test_y_el_registro_sigue_rechazando_la_que_no_existe(self, tmp_path: Path) -> None:
        registro = ExperimentRegistry(registry_dir=tmp_path, class_names=list(CLASES))
        with pytest.raises(ValueError, match="Política de duplicados desconocida"):
            registro.append("experiments", {}, on_duplicate="error")
