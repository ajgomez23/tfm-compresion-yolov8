"""Tests de la normalización de rutas ya registradas (scripts/19)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from tfm_edge.common.paths import ProjectPaths, get_project_paths
from tfm_edge.common.registry import ExperimentRegistry

CLASES = ("person", "helmet")


def cargar_script():
    """Carga scripts/19_normalize_registry_paths.py como módulo."""
    ruta = get_project_paths().root / "scripts" / "19_normalize_registry_paths.py"
    especificacion = importlib.util.spec_from_file_location("normalize_paths_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["normalize_paths_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def script():
    """El módulo del script, cargado una vez."""
    return cargar_script()


@pytest.fixture()
def entorno(tmp_path: Path):
    """Un árbol de proyecto de mentira con su registro vacío."""
    rutas = ProjectPaths(
        root=tmp_path,
        configs=tmp_path / "configs",
        data=tmp_path / "data",
        models=tmp_path / "models",
        results=tmp_path / "results",
    )
    registro = ExperimentRegistry(
        registry_dir=tmp_path / "results" / "registry", class_names=CLASES
    )
    return rutas, registro


def crear_fichero(destino: Path) -> Path:
    """Crea un fichero vacío con sus directorios padre."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("{}", encoding="utf-8")
    return destino


def escribir_fila(registro: ExperimentRegistry, **campos: str) -> None:
    """Escribe una fila en ``experiments`` saltándose la normalización de rutas."""
    registro._write_rows_atomic("experiments", [campos])  # noqa: SLF001


class TestReescrituraCorrecta:
    """Ruta absoluta dentro del árbol del proyecto."""

    def test_convierte_y_verifica(self, script, entorno):
        rutas, registro = entorno
        objetivo = crear_fichero(rutas.results / "raw" / "pruning" / "exp.json")
        escribir_fila(
            registro,
            experiment_id="s_pruning_l1global_r50_8db60bc4",
            config_path=str(objetivo),
            notes="una nota cualquiera",
        )

        _, cambios, incidencias = script.plan_table(registro, "experiments", rutas)

        assert incidencias == []
        assert len(cambios) == 1
        assert cambios[0].column == "config_path"
        assert cambios[0].proposed == "results/raw/pruning/exp.json"

    def test_aplica_sin_tocar_las_demas_celdas(self, script, entorno):
        rutas, registro = entorno
        objetivo = crear_fichero(rutas.results / "raw" / "pruning" / "exp.json")
        escribir_fila(
            registro,
            experiment_id="s_pruning_l1global_r50_8db60bc4",
            config_path=str(objetivo),
            notes="una nota cualquiera",
        )
        previas, cambios, _ = script.plan_table(registro, "experiments", rutas)

        fallos = script.apply_table(registro, "experiments", previas, cambios)

        assert fallos == []
        fila = registro.read_rows("experiments")[0]
        assert fila["config_path"] == "results/raw/pruning/exp.json"
        assert fila["experiment_id"] == "s_pruning_l1global_r50_8db60bc4"
        assert fila["notes"] == "una nota cualquiera"

    def test_es_idempotente(self, script, entorno):
        rutas, registro = entorno
        objetivo = crear_fichero(rutas.results / "raw" / "pruning" / "exp.json")
        escribir_fila(
            registro, experiment_id="s_baseline_default_745e3d08", config_path=str(objetivo)
        )
        previas, cambios, _ = script.plan_table(registro, "experiments", rutas)
        script.apply_table(registro, "experiments", previas, cambios)

        _, segundos, incidencias = script.plan_table(registro, "experiments", rutas)

        assert segundos == []
        assert incidencias == []


class TestCeldasQueNoSeTocan:
    """Los cuatro casos en los que la celda se deja como está."""

    def test_fichero_inexistente(self, script, entorno):
        rutas, registro = entorno
        ausente = rutas.results / "raw" / "pruning" / "no_existe.json"
        escribir_fila(
            registro, experiment_id="s_baseline_default_745e3d08", config_path=str(ausente)
        )

        _, cambios, incidencias = script.plan_table(registro, "experiments", rutas)

        assert cambios == []
        assert len(incidencias) == 1
        assert "no existe" in incidencias[0]

    def test_fuera_del_arbol(self, script, entorno, tmp_path_factory):
        rutas, registro = entorno
        fuera = crear_fichero(tmp_path_factory.mktemp("drive") / "MyDrive" / "best.pt")
        escribir_fila(
            registro, experiment_id="s_baseline_default_745e3d08", weights_path=str(fuera)
        )

        _, cambios, incidencias = script.plan_table(registro, "experiments", rutas)

        assert cambios == []
        assert len(incidencias) == 1
        assert "fuera del árbol" in incidencias[0]

    def test_cabecera_distinta_del_esquema(self, script, entorno):
        rutas, registro = entorno
        objetivo = crear_fichero(rutas.results / "raw" / "pruning" / "exp.json")
        destino = registro.path_for("experiments")
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8", newline="") as manejador:
            escritor = csv.DictWriter(manejador, fieldnames=["experiment_id", "config_path"])
            escritor.writeheader()
            escritor.writerow(
                {"experiment_id": "s_baseline_default_745e3d08", "config_path": str(objetivo)}
            )

        _, cambios, incidencias = script.plan_table(registro, "experiments", rutas)

        assert cambios == []
        assert len(incidencias) == 1
        assert "cabecera" in incidencias[0]

    def test_una_ruta_absoluta_en_una_columna_que_no_es_de_ruta(self, script, entorno):
        rutas, registro = entorno
        objetivo = crear_fichero(rutas.results / "raw" / "pruning" / "exp.json")
        escribir_fila(
            registro, experiment_id="s_baseline_default_745e3d08", notes=f"medido en {objetivo}"
        )

        _, cambios, incidencias = script.plan_table(registro, "experiments", rutas)

        assert cambios == []
        assert incidencias == []
        assert registro.read_rows("experiments")[0]["notes"] == f"medido en {objetivo}"


class TestRestauracionAnteFallo:
    """Si lo escrito no es lo planificado, vuelve la imagen previa."""

    def test_restaura_si_el_fichero_releido_no_coincide(self, script, entorno, monkeypatch):
        rutas, registro = entorno
        objetivo = crear_fichero(rutas.results / "raw" / "pruning" / "exp.json")
        escribir_fila(
            registro,
            experiment_id="s_baseline_default_745e3d08",
            config_path=str(objetivo),
            notes="la nota original",
        )
        previas, cambios, _ = script.plan_table(registro, "experiments", rutas)
        original = registro.read_rows("experiments")

        escritura_real = registro._write_rows_atomic
        llamadas = {"n": 0}

        def escritura_saboteada(table, rows):
            """Estropea la primera escritura y deja pasar la restauración."""
            llamadas["n"] += 1
            if llamadas["n"] == 1:
                rows = [dict(fila) for fila in rows]
                rows[0]["notes"] = "una nota que nadie ha pedido"
            return escritura_real(table, rows)

        monkeypatch.setattr(registro, "_write_rows_atomic", escritura_saboteada)

        fallos = script.apply_table(registro, "experiments", previas, cambios)

        assert len(fallos) == 1
        assert "restaurado" in fallos[0]
        assert registro.read_rows("experiments") == original


class TestGuardaDeGit:
    """El registro tiene que estar limpio en git para poder deshacer."""

    def test_fuera_de_un_repositorio_no_se_da_por_limpio(self, script, tmp_path):
        motivo = script.check_git_clean(tmp_path)

        assert motivo is not None

    def test_el_registro_real_del_proyecto_se_puede_comprobar(self, script):
        """Que el motivo sea ``None`` o un cambio pendiente, pero nunca un error de git."""
        motivo = script.check_git_clean(get_project_paths().results_registry)

        assert motivo is None or "cambios sin confirmar" in motivo
