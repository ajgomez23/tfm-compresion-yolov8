"""El tramo 26–33 sobre directorios desechables: quién escribe, y dónde."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.paths import get_project_paths
from tfm_edge.reporting import build_report as br

RAIZ = get_project_paths().root
RESULTS = RAIZ / "results"
REGISTRO = RESULTS / "registry"
VOLCADO_26 = RESULTS / "raw" / "pareto"
VOLCADO_29 = RESULTS / "raw" / "analysis" / "results_analysis.json"


def _huella_de_results() -> dict[str, tuple[int, int]]:
    """Tamaño y fecha de modificación de todo lo que hay bajo ``results/``."""
    huellas: dict[str, tuple[int, int]] = {}
    for ruta in RESULTS.rglob("*"):
        if not ruta.is_file() or ruta.suffix == ".log":
            continue
        estado = ruta.stat()
        huellas[str(ruta.relative_to(RESULTS))] = (estado.st_size, estado.st_mtime_ns)
    return huellas


def _cargar_paso_30() -> ModuleType:
    """Carga ``scripts/30_build_results_figures.py`` como módulo."""
    ruta = RAIZ / "scripts" / "30_build_results_figures.py"
    especificacion = importlib.util.spec_from_file_location("paso_30_desechable", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _ordenes_de(monkeypatch: pytest.MonkeyPatch, **directorios: Path) -> list[list[str]]:
    """Ejecuta ``run_figures`` con un espía en ``subprocess.run``."""
    ordenes: list[list[str]] = []

    def espia(orden: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        ordenes.append(orden)
        return subprocess.CompletedProcess(orden, 0)

    monkeypatch.setattr(br.subprocess, "run", espia)
    br.run_figures(RAIZ, "skip", **directorios)
    return ordenes


class TestElPaso28LlevaSusDirectoriosALosHijos:
    """Lo que el padre recibe tiene que llegar a los dos que escriben."""

    @pytest.fixture()
    def ordenes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
        """Las dos órdenes, con los cuatro directorios en un temporal."""
        return _ordenes_de(
            monkeypatch,
            registry_dir=tmp_path / "registro",
            pareto_analysis_dir=tmp_path / "pareto",
            results_analysis_dir=tmp_path / "analisis",
            figures_dir=tmp_path / "figuras",
        )

    def test_se_lanzan_los_dos_pasos(self, ordenes: list[list[str]]) -> None:
        """Si algún día se añade un tercero, esto obliga a mirar el resto."""
        guiones = [Path(orden[1]).name for orden in ordenes]
        assert guiones == [paso.script for paso in br.FIGURE_STEPS]

    def test_ninguna_ruta_de_la_orden_apunta_al_arbol_de_verdad(
        self, ordenes: list[list[str]], tmp_path: Path
    ) -> None:
        for orden in ordenes:
            for argumento in orden[2:]:
                if "/" not in argumento and "\\" not in argumento:
                    continue
                ruta = Path(argumento)
                assert ruta.is_relative_to(tmp_path), f"{argumento} no cuelga del temporal"

    def test_cada_hijo_recibe_su_volcado_y_no_el_del_otro(
        self, ordenes: list[list[str]], tmp_path: Path
    ) -> None:
        """Los dos llaman ``--analysis-dir`` a su opción y no piden lo mismo."""
        por_guion = {Path(orden[1]).name: orden for orden in ordenes}
        volcados = {
            guion: Path(orden[orden.index("--analysis-dir") + 1])
            for guion, orden in por_guion.items()
        }

        assert volcados["27_build_pareto_figures.py"] == tmp_path / "pareto"
        assert volcados["30_build_results_figures.py"] == tmp_path / "analisis"

    def test_las_figuras_de_los_dos_van_al_mismo_destino(
        self, ordenes: list[list[str]], tmp_path: Path
    ) -> None:
        """El ``--figures-dir`` del padre es el ``--output-dir`` de los hijos."""
        for orden in ordenes:
            assert Path(orden[orden.index("--output-dir") + 1]) == tmp_path / "figuras"

    def test_sin_el_volcado_del_29_no_se_lanza_nada(self, tmp_path: Path) -> None:
        tablas = tmp_path / "tablas"
        with pytest.raises(ValueError, match="results_analysis_dir"):
            br.build(
                RAIZ,
                REGISTRO,
                VOLCADO_26,
                tablas,
                tmp_path / "figuras",
                "workstation_cpu",
            )
        assert not tablas.exists(), "ha escrito las tablas antes de fallar"

    def test_con_skip_figures_no_hace_falta(self, tmp_path: Path) -> None:
        manifiesto = br.build(
            RAIZ,
            REGISTRO,
            VOLCADO_26,
            tmp_path / "tablas",
            tmp_path / "figuras",
            "workstation_cpu",
            skip_figures=True,
        )
        assert manifiesto["figures"] == {"skipped": True}


class TestElPaso30EscribeSoloDondeSeLeDice:
    """Este sí se ejecuta de verdad, con matplotlib dibujando."""

    @pytest.fixture(scope="class")
    def ejecucion(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        """Ejecuta el paso 30 entero contra directorios desechables."""
        if not VOLCADO_29.is_file():
            pytest.skip("No está el volcado del paso 29 en este árbol.")

        temporal = tmp_path_factory.mktemp("paso30")
        volcado = temporal / "analisis"
        volcado.mkdir()
        (volcado / VOLCADO_29.name).write_bytes(VOLCADO_29.read_bytes())
        figuras = temporal / "figuras"
        figuras.mkdir()

        paso = _cargar_paso_30()
        argumentos = [
            "30_build_results_figures.py",
            "--registry-dir",
            str(REGISTRO),
            "--analysis-dir",
            str(volcado),
            "--output-dir",
            str(figuras),
            "--on-existing",
            "overwrite",
        ]
        antes = _huella_de_results()
        anterior = sys.argv
        sys.argv = argumentos
        try:
            codigo = paso.main()
        finally:
            sys.argv = anterior
        return {
            "codigo": codigo,
            "antes": antes,
            "despues": _huella_de_results(),
            "volcado": volcado,
            "figuras": figuras,
        }

    def test_termina_bien_y_dibuja(self, ejecucion: dict[str, Any]) -> None:
        """Si no hubiera dibujado nada, lo de abajo pasaría por vacío."""
        assert ejecucion["codigo"] == 0
        assert sorted(p.name for p in ejecucion["figuras"].glob("*.pdf"))

    def test_el_manifiesto_va_al_volcado_que_se_le_dio(self, ejecucion: dict[str, Any]) -> None:
        destino = ejecucion["volcado"] / "results_figures_manifest.json"

        assert destino.is_file(), "el manifiesto no está junto al volcado que se leyó"
        manifiesto = json.loads(destino.read_text(encoding="utf-8"))
        assert manifiesto["figures"], "el manifiesto no declara ninguna figura"

    def test_no_ha_tocado_nada_bajo_results(self, ejecucion: dict[str, Any]) -> None:
        antes, despues = ejecucion["antes"], ejecucion["despues"]

        assert sorted(despues) == sorted(antes), (
            "han aparecido o desaparecido ficheros bajo results/: "
            f"{sorted(set(antes) ^ set(despues))}"
        )
        cambiados = [ruta for ruta in antes if antes[ruta] != despues[ruta]]
        assert not cambiados, f"reescritos bajo results/: {cambiados}"


#: Argumentos con los que cada paso que no lanza procesos hijos escribe solo en el temporal. Los
#: pasos 28 y 30 tienen su propia clase. Un test de abajo compara esta tabla con los scripts en
#: disco.
TRAMO_SIN_HIJOS: dict[str, dict[str, str]] = {
    "26_analyze_pareto.py": {
        "--registry-dir": "registro",
        "--output-dir": "salida",
        "--on-existing": "overwrite",
    },
    "27_build_pareto_figures.py": {
        "--registry-dir": "registro",
        "--output-dir": "salida",
        # Su manifiesto cuelga de aquí, no de --output-dir: aislar solo la
        # salida deja que reescriba results/raw/pareto/figures_manifest.json.
        "--analysis-dir": "volcado_26",
        "--on-existing": "overwrite",
    },
    "29_analyze_results.py": {
        "--registry-dir": "registro",
        "--output-dir": "salida",
        "--on-existing": "overwrite",
    },
    "31_register_export_cost.py": {
        # El único que escribe en el registro, así que se le da una copia.
        "--registry-dir": "registro_escribible",
        "--output-dir": "salida",
        "--apply": "",
        "--on-conflict": "overwrite",
    },
    "32_distill_seed_variance.py": {
        "--registry-dir": "registro",
        "--output-dir": "salida",
    },
    "33_build_design_figures.py": {
        "--registry-dir": "registro",
        "--output-dir": "salida",
    },
}

#: Los pasos del tramo 26–33 que ya tienen clase propia arriba.
TRAMO_CON_CLASE_PROPIA = {"28_build_report.py", "30_build_results_figures.py"}


class TestElRestoDelTramo26a33EscribeSoloDondeSeLeDice:
    """La auditoría de destinos del tramo entero, ejecutada y no leída."""

    @pytest.fixture(scope="class")
    def ejecuciones(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
        """Ejecuta los seis pasos, cada uno con todos sus destinos en temporales."""
        base = tmp_path_factory.mktemp("tramo")
        registro_escribible = base / "registro_escribible"
        shutil.copytree(REGISTRO, registro_escribible)
        volcado_26 = base / "volcado_26"
        volcado_26.mkdir()
        for fichero in VOLCADO_26.glob("*.json"):
            (volcado_26 / fichero.name).write_bytes(fichero.read_bytes())

        resultados: dict[str, dict[str, Any]] = {}
        for guion, argumentos in TRAMO_SIN_HIJOS.items():
            salida = base / guion.split("_")[0] / "salida"
            salida.mkdir(parents=True)
            sitios = {
                "registro": REGISTRO,
                "registro_escribible": registro_escribible,
                "volcado_26": volcado_26,
                "salida": salida,
            }
            orden = [sys.executable, str(RAIZ / "scripts" / guion)]
            for bandera, valor in argumentos.items():
                orden.append(bandera)
                if valor:
                    orden.append(str(sitios.get(valor, valor)))

            antes = _huella_de_results()
            completado = subprocess.run(orden, capture_output=True, text=True, check=False)
            resultados[guion] = {
                "codigo": completado.returncode,
                "salida": salida,
                "texto": completado.stdout + completado.stderr,
                "antes": antes,
                "despues": _huella_de_results(),
            }
        return resultados

    @pytest.mark.parametrize("guion", sorted(TRAMO_SIN_HIJOS))
    def test_termina_bien_y_escribe_algo(
        self, guion: str, ejecuciones: dict[str, dict[str, Any]]
    ) -> None:
        """Sin esto, «no ha tocado results/» lo cumpliría un paso que no hace nada."""
        ejecucion = ejecuciones[guion]

        assert ejecucion["codigo"] == 0, ejecucion["texto"][-1500:]
        escritos = [ruta for ruta in ejecucion["salida"].rglob("*") if ruta.is_file()]
        assert escritos, f"{guion} terminó con 0 y no escribió nada en su destino"

    @pytest.mark.parametrize("guion", sorted(TRAMO_SIN_HIJOS))
    def test_no_ha_tocado_nada_bajo_results(
        self, guion: str, ejecuciones: dict[str, dict[str, Any]]
    ) -> None:
        ejecucion = ejecuciones[guion]
        antes, despues = ejecucion["antes"], ejecucion["despues"]

        assert sorted(despues) == sorted(antes), (
            f"{guion}: han aparecido o desaparecido ficheros bajo results/: "
            f"{sorted(set(antes) ^ set(despues))}"
        )
        cambiados = [ruta for ruta in antes if antes[ruta] != despues[ruta]]
        assert not cambiados, f"{guion} reescribió bajo results/: {cambiados}"

    def test_la_tabla_cubre_el_tramo_entero(self) -> None:
        """Un paso nuevo en el tramo no puede quedarse fuera en silencio."""
        en_disco = {
            ruta.name
            for ruta in (RAIZ / "scripts").glob("*.py")
            if ruta.name[:2].isdigit() and 26 <= int(ruta.name[:2]) <= 33
        }

        assert en_disco == set(TRAMO_SIN_HIJOS) | TRAMO_CON_CLASE_PROPIA, (
            "la tabla de la auditoría y los guiones del tramo 26–33 no coinciden: "
            f"{sorted(en_disco ^ (set(TRAMO_SIN_HIJOS) | TRAMO_CON_CLASE_PROPIA))}"
        )


class TestLosDosManifiestosDicenDondeVan:
    """La misma regla en dos sitios se separa sola."""

    @pytest.mark.parametrize("guion", ["27_build_pareto_figures.py", "30_build_results_figures.py"])
    def test_el_help_avisa_de_que_la_salida_no_basta(self, guion: str) -> None:
        """Se lee la ayuda de verdad, que es lo que el usuario ve."""
        completado = subprocess.run(
            [sys.executable, str(RAIZ / "scripts" / guion), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        ayuda = " ".join(completado.stdout.split())

        assert (
            "El manifiesto de este paso se escribe aquí" in ayuda
        ), f"{guion} no dice en --help que su manifiesto va a --analysis-dir"
        assert (
            "copiar el volcado a un temporal" in ayuda
        ), f"{guion} no dice qué hay que hacer para aislarlo de verdad"
