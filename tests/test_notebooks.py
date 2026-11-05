"""Tests de los cuadernos de Colab: sintaxis, copia del registro, venv y pines."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tfm_edge.common.paths import get_project_paths

# Cuaderno: (fragmento que identifica la celda de campaña, función que copia el registro).
CUADERNOS_DE_CAMPANA: dict[str, str] = {
    "01_colab_prune_finetune.ipynb": "scripts/13_prune.py",
    "02_colab_qat.ipynb": "scripts/16_qat.py",
    "03_colab_distill.ipynb": "scripts/20_distill.py",
}

FUNCION_DE_SALVADO = "salvar_registro"

#: Opciones con las que una invocación no escribe en el registro y no necesita copia.
BANDERAS_QUE_NO_ESCRIBEN: tuple[str, ...] = ("--dry-run", "--preflight-only")


def _celdas_de_campana(cuaderno: Path, lanzador: str) -> list[tuple[int, str]]:
    """Localiza las celdas que lanzan la campaña, sin las que usan
    :data:`BANDERAS_QUE_NO_ESCRIBEN`."""
    return [
        (indice, fuente)
        for indice, fuente in _celdas_de_codigo(cuaderno)
        if lanzador in fuente
        and "subprocess.run" in fuente
        and not any(bandera in fuente for bandera in BANDERAS_QUE_NO_ESCRIBEN)
    ]


def _cuadernos() -> list[Path]:
    """Todos los cuadernos del repositorio."""
    return sorted((get_project_paths().root / "notebooks").glob("*.ipynb"))


def _celdas_de_codigo(cuaderno: Path) -> list[tuple[int, str]]:
    """Devuelve las celdas de código de un cuaderno."""
    contenido = json.loads(cuaderno.read_text(encoding="utf-8"))
    return [
        (indice, "".join(celda["source"]))
        for indice, celda in enumerate(contenido["cells"])
        if celda["cell_type"] == "code"
    ]


def _sin_magias(fuente: str) -> str:
    """Quita las líneas de magia de IPython, que no son Python."""
    return "\n".join(
        linea for linea in fuente.split("\n") if not linea.lstrip().startswith(("!", "%"))
    )


def _lineas_de_llamada(fuente: str, funcion: str) -> list[int]:
    """Líneas en las que se llama a ``funcion``, no en las que se define."""
    arbol = ast.parse(fuente)
    return [
        nodo.lineno
        for nodo in ast.walk(arbol)
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Name)
        and nodo.func.id == funcion
    ]


class TestQueLosCuadernosSonPythonValido:
    """Un error de sintaxis en un cuaderno se paga en una sesión de GPU."""

    @pytest.mark.parametrize("cuaderno", _cuadernos(), ids=lambda ruta: ruta.name)
    def test_todas_las_celdas_de_codigo_parsean(self, cuaderno: Path) -> None:
        """Cada celda de código, quitadas las magias, tiene que parsear."""
        for indice, fuente in _celdas_de_codigo(cuaderno):
            try:
                ast.parse(_sin_magias(fuente))
            except SyntaxError as error:  # pragma: no cover - solo si algo se rompe
                pytest.fail(f"{cuaderno.name}, celda {indice}: {error}")

    def test_hay_cuadernos_que_revisar(self) -> None:
        assert len(_cuadernos()) >= 3


class TestQueElRegistroSeSalvaEnLaCeldaDeLaCampana:
    """La copia del registro a Drive está en la propia celda de la campaña."""

    @pytest.mark.parametrize("nombre,lanzador", sorted(CUADERNOS_DE_CAMPANA.items()))
    def test_la_celda_que_lanza_la_campana_copia_el_registro(
        self, nombre: str, lanzador: str
    ) -> None:
        """Copiar no puede depender de que alguien ejecute una celda posterior."""
        cuaderno = get_project_paths().root / "notebooks" / nombre
        celdas = _celdas_de_campana(cuaderno, lanzador)
        assert celdas, f"{nombre}: no se encuentra la celda que lanza {lanzador}"
        for indice, fuente in celdas:
            assert _lineas_de_llamada(fuente, FUNCION_DE_SALVADO), (
                f"{nombre}, celda {indice}: lanza la campaña pero no llama a "
                f"{FUNCION_DE_SALVADO}(); la copia a Drive dependería de una celda posterior."
            )

    @pytest.mark.parametrize("nombre,lanzador", sorted(CUADERNOS_DE_CAMPANA.items()))
    def test_se_copia_antes_de_cualquier_raise(self, nombre: str, lanzador: str) -> None:
        cuaderno = get_project_paths().root / "notebooks" / nombre
        for indice, fuente in _celdas_de_campana(cuaderno, lanzador):
            llamadas = _lineas_de_llamada(fuente, FUNCION_DE_SALVADO)
            assert llamadas, f"{nombre}, celda {indice}: sin llamada que ordenar"
            arbol = ast.parse(fuente)
            relanzamientos = [
                nodo.lineno for nodo in ast.walk(arbol) if isinstance(nodo, ast.Raise)
            ]
            if relanzamientos:
                assert min(llamadas) < min(relanzamientos), (
                    f"{nombre}, celda {indice}: hay un 'raise' en la línea "
                    f"{min(relanzamientos)}, antes de la llamada a "
                    f"{FUNCION_DE_SALVADO}() de la línea {min(llamadas)}. Si salta, "
                    "no se copia nada."
                )


#: Cuadernos que ejecutan el proyecto en un venv propio. La lista es explícita y los tests
#: comprueban que coincide con los cuadernos que definen ``VENV_PY``. El 00 y el 01 no usan venv.
CUADERNOS_CON_VENV: frozenset[str] = frozenset({"02_colab_qat.ipynb", "03_colab_distill.ipynb"})

#: Cuántas invocaciones de script tiene cada cuaderno migrado. Es el control del
#: reconocimiento de más abajo: si una invocación deja de tener forma de ``argv``,
#: el test que la vigila dejaría de verla y pasaría sin comprobar nada.
INVOCACIONES_ESPERADAS: dict[str, int] = {"02_colab_qat.ipynb": 3, "03_colab_distill.ipynb": 3}

#: Cuántas llamadas a subprocess.run hay en las celdas que cruzan al venv. Mismo
#: papel que la constante de arriba: si una llamada deja de reconocerse, el test que
#: exige el entorno explícito pasaría sin mirar ninguna.
SUBPROCESOS_ESPERADOS: dict[str, int] = {"02_colab_qat.ipynb": 5, "03_colab_distill.ipynb": 5}

#: Cuántas instalaciones de uv hay en cada cuaderno migrado. Tercer control del
#: mismo tipo que los dos de arriba: sin él, cambiar la forma de la llamada dejaría
#: al test sin nada que mirar y pasaría igual.
INSTALACIONES_DE_UV: dict[str, int] = {"02_colab_qat.ipynb": 1, "03_colab_distill.ipynb": 1}

#: Nombre de la constante del cuaderno que lleva el pin de uv.
CONSTANTE_PIN_UV = "UV_PIN"

#: Módulos que solo existen dentro del venv. Importarlos en el kernel no daría un
#: error claro: daría el paquete equivocado, o el de Colab, o ninguno.
MODULOS_DEL_VENV: tuple[str, ...] = ("tfm_edge", "torch", "ultralytics")


def _fragmentos_del_venv(cuaderno: Path) -> list[tuple[int, str, str]]:
    """Los ``CODIGO_*`` que se envían al intérprete del venv con ``python -c``."""
    encontrados: list[tuple[int, str, str]] = []
    for indice, fuente in _celdas_de_codigo(cuaderno):
        for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
            if not isinstance(nodo, ast.Assign):
                continue
            if not isinstance(nodo.value, ast.Constant) or not isinstance(nodo.value.value, str):
                continue
            for destino in nodo.targets:
                if isinstance(destino, ast.Name) and destino.id.startswith("CODIGO_"):
                    encontrados.append((indice, destino.id, nodo.value.value))
    return encontrados


def _menciona(nodo: ast.AST, nombre: str) -> bool:
    """Si un trozo de árbol usa ese identificador."""
    return any(isinstance(hijo, ast.Name) and hijo.id == nombre for hijo in ast.walk(nodo))


def _usa_sys_executable(nodo: ast.AST) -> bool:
    """Si un trozo de árbol usa ``sys.executable``, el intérprete del kernel."""
    return any(
        isinstance(hijo, ast.Attribute)
        and hijo.attr == "executable"
        and isinstance(hijo.value, ast.Name)
        and hijo.value.id == "sys"
        for hijo in ast.walk(nodo)
    )


def _invocaciones_de_script(cuaderno: Path) -> list[tuple[int, str, ast.expr]]:
    """Listas de argumentos que lanzan un ``scripts/*.py`` (intérprete y script al principio)."""
    salida: list[tuple[int, str, ast.expr]] = []
    for indice, fuente in _celdas_de_codigo(cuaderno):
        for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
            if not isinstance(nodo, ast.List) or len(nodo.elts) < 2:
                continue
            segundo = nodo.elts[1]
            if not isinstance(segundo, ast.Constant) or not isinstance(segundo.value, str):
                continue
            if segundo.value.startswith("scripts/") and segundo.value.endswith(".py"):
                salida.append((indice, segundo.value, nodo.elts[0]))
    return salida


def _llamadas_a_subprocess_run(fuente: str) -> list[ast.Call]:
    """Las llamadas a subprocess.run de una celda."""
    return [
        nodo
        for nodo in ast.walk(ast.parse(_sin_magias(fuente)))
        if isinstance(nodo, ast.Call)
        and isinstance(nodo.func, ast.Attribute)
        and nodo.func.attr == "run"
        and isinstance(nodo.func.value, ast.Name)
        and nodo.func.value.id == "subprocess"
    ]


def _argumentos_de_pip_install(fuente: str) -> list[list[ast.expr]]:
    """Los ``argv`` de una celda que son una instalación con pip."""
    salida: list[list[ast.expr]] = []
    for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
        if not isinstance(nodo, ast.List):
            continue
        literales = {
            elemento.value
            for elemento in nodo.elts
            if isinstance(elemento, ast.Constant) and isinstance(elemento.value, str)
        }
        if {"pip", "install"} <= literales:
            salida.append(nodo.elts)
    return salida


def _texto_inicial(expr: ast.expr) -> str:
    """El texto literal con el que empieza un argumento, sea o no un f-string."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr) and expr.values:
        primero = expr.values[0]
        if isinstance(primero, ast.Constant) and isinstance(primero.value, str):
            return primero.value
    return ""


def _importes_del_venv(fuente: str) -> list[str]:
    """Módulos de :data:`MODULOS_DEL_VENV` importados en la propia celda, sin contar los
    fragmentos."""
    nombres: list[str] = []
    for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
        if isinstance(nodo, ast.Import):
            nombres += [alias.name for alias in nodo.names]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            nombres.append(nodo.module)
    return [nombre for nombre in nombres if nombre.split(".")[0] in MODULOS_DEL_VENV]


class TestQueLosFragmentosEnviadosAlVenvSonPythonValido:
    """Un fragmento roto cuesta lo mismo que una celda rota, y no se ve venir."""

    @pytest.mark.parametrize("cuaderno", _cuadernos(), ids=lambda ruta: ruta.name)
    def test_cada_fragmento_parsea(self, cuaderno: Path) -> None:
        """Cada ``CODIGO_*`` tiene que ser Python válido por sí mismo."""
        for indice, nombre, fragmento in _fragmentos_del_venv(cuaderno):
            try:
                ast.parse(fragmento)
            except SyntaxError as error:  # pragma: no cover - solo si algo se rompe
                pytest.fail(f"{cuaderno.name}, celda {indice}, {nombre}: {error}")

    def test_hay_fragmentos_que_revisar(self) -> None:
        total = sum(len(_fragmentos_del_venv(c)) for c in _cuadernos())
        assert total >= 8, f"solo se han encontrado {total} fragmentos CODIGO_*"


class TestQueLaCampanaNoCorreEnElKernelDeColab:
    """La campaña se ejecuta en el venv, no en el kernel de Colab."""

    def test_los_cuadernos_con_venv_son_los_declarados(self) -> None:
        con_venv = {
            cuaderno.name
            for cuaderno in _cuadernos()
            if any(
                _menciona(ast.parse(_sin_magias(fuente)), "VENV_PY")
                for _, fuente in _celdas_de_codigo(cuaderno)
            )
        }
        assert con_venv == set(CUADERNOS_CON_VENV), (
            f"cuadernos que usan VENV_PY: {sorted(con_venv)}; declarados: "
            f"{sorted(CUADERNOS_CON_VENV)}. Si has migrado uno, decláralo aquí."
        )

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_VENV))
    def test_se_reconocen_todas_las_invocaciones(self, nombre: str) -> None:
        cuaderno = get_project_paths().root / "notebooks" / nombre
        invocaciones = _invocaciones_de_script(cuaderno)
        assert len(invocaciones) == INVOCACIONES_ESPERADAS[nombre], (
            f"{nombre}: se reconocen {len(invocaciones)} invocaciones y se esperaban "
            f"{INVOCACIONES_ESPERADAS[nombre]}: "
            f"{[(i, s) for i, s, _ in invocaciones]}"
        )

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_VENV))
    def test_los_scripts_se_lanzan_con_el_interprete_del_venv(self, nombre: str) -> None:
        """``sys.executable`` es el 3.13 del kernel, que no tiene ni torch."""
        cuaderno = get_project_paths().root / "notebooks" / nombre
        for indice, script, primero in _invocaciones_de_script(cuaderno):
            assert not _usa_sys_executable(primero), (
                f"{nombre}, celda {indice}: lanza {script} con sys.executable, que es "
                "el intérprete del kernel. Ahí no está instalado el proyecto."
            )
            assert _menciona(
                primero, "VENV_PY"
            ), f"{nombre}, celda {indice}: lanza {script} sin el intérprete del venv."

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_VENV))
    def test_el_kernel_no_importa_lo_que_vive_en_el_venv(self, nombre: str) -> None:
        cuaderno = get_project_paths().root / "notebooks" / nombre
        for indice, fuente in _celdas_de_codigo(cuaderno):
            importados = _importes_del_venv(fuente)
            assert not importados, (
                f"{nombre}, celda {indice}: el kernel importa {importados}. Eso vive "
                "en el venv: pásalo a un fragmento CODIGO_* y ejecútalo con en_venv()."
            )

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_VENV))
    def test_los_subprocesos_del_venv_llevan_entorno_explicito(self, nombre: str) -> None:
        """El entorno del kernel describe al kernel, y deja de ser verdad en el venv."""
        cuaderno = get_project_paths().root / "notebooks" / nombre
        vistas = 0
        for indice, fuente in _celdas_de_codigo(cuaderno):
            if not _menciona(ast.parse(_sin_magias(fuente)), "VENV_PY"):
                continue
            for llamada in _llamadas_a_subprocess_run(fuente):
                vistas += 1
                assert "env" in {clave.arg for clave in llamada.keywords}, (
                    f"{nombre}, celda {indice}: subprocess.run sin 'env'. Heredaría "
                    "el entorno del kernel, MPLBACKEND y PYTHONPATH incluidos. Pasa "
                    "env=ENTORNO_VENV."
                )
        assert vistas == SUBPROCESOS_ESPERADOS[nombre], (
            f"{nombre}: se han reconocido {vistas} llamadas a subprocess.run en las "
            f"celdas del venv y se esperaban {SUBPROCESOS_ESPERADOS[nombre]}."
        )


class TestQueElProvisionadorDelInterpreteVaAnclado:
    """``uv`` es la única dependencia que no puede anclarse en ``requirements.txt``."""

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_VENV))
    def test_uv_se_instala_con_version_exacta(self, nombre: str) -> None:
        """Un ``uv`` pelado instala la de hoy, que no es la que se midió."""
        cuaderno = get_project_paths().root / "notebooks" / nombre
        vistas = 0
        for indice, fuente in _celdas_de_codigo(cuaderno):
            for argv in _argumentos_de_pip_install(fuente):
                for elemento in argv:
                    inicial = _texto_inicial(elemento)
                    if inicial != "uv" and not inicial.startswith("uv="):
                        continue
                    vistas += 1
                    assert inicial.startswith(
                        "uv=="
                    ), f"{nombre}, celda {indice}: se instala {inicial!r} sin versión fija."
        assert vistas == INSTALACIONES_DE_UV[nombre], (
            f"{nombre}: se han reconocido {vistas} instalaciones de uv y se esperaban "
            f"{INSTALACIONES_DE_UV[nombre]}."
        )

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_VENV))
    def test_el_pin_es_un_numero_de_version_literal(self, nombre: str) -> None:
        """El pin tiene que poder leerse sin ejecutar el cuaderno."""
        cuaderno = get_project_paths().root / "notebooks" / nombre
        valores: list[str] = []
        for _, fuente in _celdas_de_codigo(cuaderno):
            for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
                if not isinstance(nodo, ast.Assign):
                    continue
                nombres = {d.id for d in nodo.targets if isinstance(d, ast.Name)}
                if CONSTANTE_PIN_UV not in nombres:
                    continue
                assert isinstance(nodo.value, ast.Constant) and isinstance(
                    nodo.value.value, str
                ), f"{nombre}: {CONSTANTE_PIN_UV} no es una cadena literal."
                valores.append(nodo.value.value)

        assert len(valores) == 1, (
            f"{nombre}: {CONSTANTE_PIN_UV} está asignada {len(valores)} veces; con dos "
            "asignaciones el pin que manda depende del orden de ejecución."
        )
        trozos = valores[0].split(".")
        assert len(trozos) == 3 and all(t.isdigit() for t in trozos), (
            f"{nombre}: {CONSTANTE_PIN_UV} vale {valores[0]!r} y se esperaba una "
            "versión exacta del tipo 0.12.5."
        )


#: Cuadernos que piden a uv una versión exacta de CPython. El 02 sigue pidiendo "3.12".
CUADERNOS_CON_PYTHON_EXACTO: frozenset[str] = frozenset({"03_colab_distill.ipynb"})

#: Nombre de la constante que lleva la versión exacta del intérprete.
CONSTANTE_PIN_PYTHON = "PYTHON_PIN"


def _asignaciones_literales(cuaderno: Path, constante: str) -> list[ast.expr]:
    """Los valores asignados a una constante en todo el cuaderno."""
    valores: list[ast.expr] = []
    for _, fuente in _celdas_de_codigo(cuaderno):
        for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
            if isinstance(nodo, ast.Assign) and any(
                isinstance(d, ast.Name) and d.id == constante for d in nodo.targets
            ):
                valores.append(nodo.value)
    return valores


def _creaciones_de_venv(cuaderno: Path) -> list[list[ast.expr]]:
    """Los ``argv`` literales que crean el venv con uv (``uv venv ...``)."""
    salida: list[list[ast.expr]] = []
    for _, fuente in _celdas_de_codigo(cuaderno):
        for nodo in ast.walk(ast.parse(_sin_magias(fuente))):
            if not isinstance(nodo, ast.List):
                continue
            textos = [_texto_inicial(e) for e in nodo.elts]
            if "uv" in textos and "venv" in textos[textos.index("uv") :]:
                salida.append(nodo.elts)
    return salida


class TestQueElInterpreteVaAnclado:
    """La versión exacta del CPython del venv, y que sea el gestionado por uv."""

    def test_los_cuadernos_con_version_exacta_son_los_declarados(self) -> None:
        con_pin = {
            cuaderno.name
            for cuaderno in _cuadernos()
            if _asignaciones_literales(cuaderno, CONSTANTE_PIN_PYTHON)
        }
        assert con_pin == set(CUADERNOS_CON_PYTHON_EXACTO)

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_PYTHON_EXACTO))
    def test_la_version_es_un_literal_exacto(self, nombre: str) -> None:
        cuaderno = get_project_paths().root / "notebooks" / nombre
        valores = _asignaciones_literales(cuaderno, CONSTANTE_PIN_PYTHON)

        assert len(valores) == 1, f"{nombre}: {CONSTANTE_PIN_PYTHON} asignada {len(valores)} veces"
        assert isinstance(valores[0], ast.Constant) and isinstance(valores[0].value, str)
        trozos = valores[0].value.split(".")
        assert len(trozos) == 3 and all(t.isdigit() for t in trozos), (
            f"{nombre}: {CONSTANTE_PIN_PYTHON} vale {valores[0].value!r} y se esperaba una "
            "versión exacta del tipo 3.12.14."
        )

    @pytest.mark.parametrize("nombre", sorted(CUADERNOS_CON_PYTHON_EXACTO))
    def test_el_venv_se_crea_con_esa_version_y_gestionado(self, nombre: str) -> None:
        """Sin --managed-python, uv puede tomar un 3.12.14 de la imagen con otra compilación."""
        cuaderno = get_project_paths().root / "notebooks" / nombre
        creaciones = _creaciones_de_venv(cuaderno)

        assert len(creaciones) == 1, f"{nombre}: {len(creaciones)} creaciones de venv con uv"
        argv = creaciones[0]
        textos = [_texto_inicial(e) for e in argv]
        assert "--managed-python" in textos, f"{nombre}: el venv no exige el Python gestionado"
        indice = textos.index("--python")
        assert isinstance(argv[indice + 1], ast.Name)
        assert (
            argv[indice + 1].id == CONSTANTE_PIN_PYTHON
        ), f"{nombre}: --python no recibe {CONSTANTE_PIN_PYTHON}, sino otra cosa"
