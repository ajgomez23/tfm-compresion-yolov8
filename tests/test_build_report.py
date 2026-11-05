"""El paso que arma el informe: de dónde salen los números y qué dice que falta."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.analysis import pareto
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import is_absolute_path_text
from tfm_edge.reporting import build_report as br

RAIZ = get_project_paths().root
ANALISIS = RAIZ / "results" / "raw" / "pareto"
REGISTRO = RAIZ / "results" / "registry"
DISPOSITIVO = "workstation_cpu"


def _cargar_script() -> ModuleType:
    """Carga ``scripts/28_build_report.py`` como módulo."""
    ruta = RAIZ / "scripts" / "28_build_report.py"
    especificacion = importlib.util.spec_from_file_location("build_report_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["build_report_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def analisis() -> dict:
    """El volcado real del paso 26."""
    if not (ANALISIS / br.ANALYSIS_NAME).is_file():
        pytest.skip("no está el volcado del paso 26")
    return br.load_analysis(ANALISIS)


class TestElCatalogoDiceLoQueFalta:
    """Terminar sin error no puede leerse como «el informe está completo»."""

    def test_toda_pieza_pendiente_dice_quien_la_hara(self) -> None:
        """Una pieza pendiente sin dueño no es información, es una queja."""
        for pieza in br.pending_items():
            assert pieza.producer, f"la pieza '{pieza.key}' no dice quién la producirá"
            assert pieza.note, f"la pieza '{pieza.key}' no dice qué falta"
        assert br.pending_items() == tuple(p for p in br.CATALOGUE if not p.available)

    def test_sin_pendientes_el_resumen_lo_dice(self) -> None:
        """Una lista vacía sin rótulo se lee igual que una sección que no se imprimió."""
        texto = "\n".join(br.summarize({"tables": [], "figures": {"skipped": True}, "pending": []}))
        assert "Pendiente de la memoria: ninguna pieza del catálogo" in texto

    def test_la_figura_de_tipos_de_error_la_produce_el_paso_30(self) -> None:
        import re

        from tfm_edge.reporting import figures_results as fr

        pieza = next(p for p in br.CATALOGUE if p.key == "tipos_de_error")
        assert pieza.available
        assert pieza.producer == "scripts/30_build_results_figures.py"
        assert (pieza.chapter, pieza.section) == fr.PLACEMENT["tipos_de_error"]
        assert "faltan los datos" not in pieza.note
        assert not re.search(r"\d+\s+filas", pieza.note), pieza.note

    def test_la_perdida_por_compresion_es_del_6_2_y_se_nombra_asi(self) -> None:
        from tfm_edge.reporting import figures_results as fr
        from tfm_edge.reporting import style

        pieza = next(p for p in br.CATALOGUE if p.key == "perdida_por_compresion")
        assert pieza.available
        assert pieza.producer == "scripts/30_build_results_figures.py"
        assert (pieza.chapter, pieza.section) == fr.PLACEMENT["perdida_map_por_compresion"]
        nombre = style.figure_filename(
            *fr.PLACEMENT["perdida_map_por_compresion"], "perdida_map_por_compresion"
        )
        assert nombre.startswith("fig_6_2_")

    def test_la_validacion_frente_a_prueba_es_del_6_7_y_se_nombra_asi(self) -> None:
        """La produce el paso 30 y va en el 6.7, como dice su nombre."""
        from tfm_edge.reporting import figures_results as fr
        from tfm_edge.reporting import style

        pieza = next(p for p in br.CATALOGUE if p.key == "validacion_frente_a_prueba")
        assert pieza.available
        assert pieza.producer == "scripts/30_build_results_figures.py"
        assert (pieza.chapter, pieza.section) == fr.PLACEMENT["validacion_frente_a_prueba"]
        nombre = style.figure_filename(
            *fr.PLACEMENT["validacion_frente_a_prueba"], "validacion_frente_a_prueba"
        )
        assert nombre.startswith("fig_6_7_")

    def test_el_catalogo_no_repite_claves(self) -> None:
        """Dos piezas con la misma clave se pisarían en el manifiesto."""
        claves = [pieza.key for pieza in br.CATALOGUE]
        assert len(claves) == len(set(claves))

    def test_el_resumen_enumera_lo_pendiente(self) -> None:
        manifiesto = {
            "tables": [],
            "figures": {"skipped": True},
            "pending": [
                {
                    "key": pieza.key,
                    "kind": pieza.kind,
                    "chapter": pieza.chapter,
                    "section": pieza.section,
                    "description": pieza.description,
                    "producer": pieza.producer,
                    "note": pieza.note,
                }
                for pieza in br.pending_items()
            ],
        }
        texto = "\n".join(br.summarize(manifiesto))
        assert "Pendiente de la memoria" in texto
        for pieza in br.pending_items():
            assert pieza.description in texto


class TestLosSuelosSalenDelVolcado:
    """Los umbrales se leen del volcado donde se midieron."""

    def test_el_suelo_relativo_de_latencia_llega_a_la_columna(self, analisis) -> None:
        argumentos = br._floor_kwargs(analisis, "plano_map5095_p50", "latency_p50_ms")
        assert "floor_relative" in argumentos
        assert argumentos["floor_relative"] > 0
        assert "floor" not in argumentos, "un suelo relativo no puede llegar también como absoluto"
        assert "D-095" in argumentos["floor_note"], (
            "la nota tiene que arrastrar la procedencia que midió el paso 26, no un "
            f"texto escrito aquí: {argumentos['floor_note']}"
        )

    def test_la_columna_recibe_los_dos_suelos_y_el_base_es_el_entre_sesiones(
        self, analisis
    ) -> None:
        suelo = br.floor_of(analisis, "spec_3d_map5095_p95_memoria", "latency_p95_ms")
        argumentos = br._floor_kwargs(analisis, "spec_3d_map5095_p95_memoria", "latency_p95_ms")
        assert argumentos["floor_relative"] == pytest.approx(suelo["cross_session"])
        assert argumentos["floor_same_session"] == pytest.approx(suelo["same_session"])
        assert suelo["cross_session"] >= suelo["same_session"]

    def test_map50_no_tiene_suelo_y_la_columna_se_queda_sin_el(self, analisis) -> None:
        assert br.floor_of(analisis, "plano_map50_p50", "map_50") is None
        assert br._floor_kwargs(analisis, "plano_map50_p50", "map_50") == {}

    def test_la_tabla_de_medidas_lo_advierte(self, analisis) -> None:
        """Si una columna no certifica nada, la tabla tiene que decirlo."""
        tablas = br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)
        medidas = tablas[0]
        assert any("mAP@50 no tiene suelo" in nota for nota in medidas.notes), medidas.notes


class TestNoPuedeContarOtraCampana:
    """Una tabla que incluya filas que la frontera excluye pone dos recuentos."""

    @pytest.mark.parametrize(
        "clave, alterar",
        [
            ("deployment_rows_used", lambda v: v + 1),
            # Los tres contadores que antes no se comprobaban.
            ("quality_rows_used", lambda v: v - 2),
            ("quality_rows_discarded_status", lambda v: v + 3),
            ("deployment_rows_discarded", lambda v: {**v, "status_no_ok": v["status_no_ok"] + 1}),
        ],
    )
    def test_aborta_si_el_registro_y_el_volcado_no_coinciden(
        self, tmp_path, analisis, clave, alterar
    ) -> None:
        """Es el desajuste que nadie vería leyendo la tabla."""
        alterado = json.loads(json.dumps(analisis))
        alterado["inputs"][clave] = alterar(alterado["inputs"][clave])
        (tmp_path / br.ANALYSIS_NAME).write_text(
            json.dumps(alterado, ensure_ascii=False), encoding="utf-8"
        )
        with pytest.raises(ValueError, match=clave):
            br.build_tables(REGISTRO, tmp_path, DISPOSITIVO)

    def test_los_cuatro_desajustes_se_dicen_de_una_vez(self, tmp_path, analisis) -> None:
        alterado = json.loads(json.dumps(analisis))
        alterado["inputs"]["deployment_rows_used"] += 1
        alterado["inputs"]["quality_rows_used"] -= 2
        (tmp_path / br.ANALYSIS_NAME).write_text(
            json.dumps(alterado, ensure_ascii=False), encoding="utf-8"
        )

        with pytest.raises(ValueError) as error:
            br.build_tables(REGISTRO, tmp_path, DISPOSITIVO)

        assert "deployment_rows_used" in str(error.value)
        assert "quality_rows_used" in str(error.value)

    def test_aborta_si_los_puntos_no_son_los_del_volcado(self, analisis) -> None:
        """Un punto de más o de menos y la tabla y la figura dejan de hablar de lo mismo."""
        vista = "plano_map5095_p50"
        objetivos = pareto.named_views()[vista]
        puntos = [
            punto
            for punto in pareto.build_points(*_filas(), objetivos)
            if punto.device_id == DISPOSITIVO
        ]
        with pytest.raises(ValueError, match="no coinciden"):
            br.build_frontier_table(puntos[:-1], analisis, vista, DISPOSITIVO)

    def test_aborta_si_el_volcado_no_tiene_ese_dispositivo(self, analisis) -> None:
        """Antes que inventarse una frontera vacía."""
        with pytest.raises(ValueError, match="no tiene la vista"):
            br.build_frontier_table([], analisis, "plano_map5095_p50", "un_dispositivo_que_no")

    def test_sin_volcado_no_se_construye_nada(self, tmp_path) -> None:
        """Sin él, los suelos saldrían de otro sitio y podrían contradecir a la figura."""
        with pytest.raises(FileNotFoundError, match="26_analyze_pareto"):
            br.build_tables(REGISTRO, tmp_path, DISPOSITIVO)


def _filas() -> tuple[list, list]:
    """Las filas del registro con los mismos filtros que el análisis."""
    from tfm_edge.common.registry import ExperimentRegistry

    registro = ExperimentRegistry(REGISTRO)
    calidad, _ = analysis_quality_rows(registro)
    despliegue, _ = registro.latency_comparable_rows()
    return calidad, despliegue


class TestLaTablaDeLaFronteraCoincideConLaFigura:
    """Las dos salen del mismo volcado y tienen que decir lo mismo."""

    def test_los_puntos_de_la_frontera_estricta_son_los_del_volcado(self, analisis) -> None:
        vista = "plano_map5095_p50"
        tabla = next(
            t
            for t in br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)
            if t.name == f"frontera_{vista}_{DISPOSITIVO}"
        )
        en_tabla = sum(1 for fila in tabla.rows if fila["frontier"] == "estricta")
        grupo = analisis["views"][vista]["groupings"]["device_id"][DISPOSITIVO]
        assert en_tabla == len(grupo["frontier_strict"])

    def test_los_tres_estados_son_los_de_la_leyenda_de_la_figura(self, analisis) -> None:
        """Si la tabla inventara un cuarto estado, el lector no sabría a qué se refiere."""
        tabla = next(
            t for t in br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO) if "frontera" in t.name
        )
        estados = {fila["frontier"] for fila in tabla.rows}
        assert estados <= {"estricta", "solo con suelo", "dominado"}

    def test_todas_van_apaisadas(self) -> None:
        """En vertical se salían del papel; las de la frontera, justo por su columna."""
        tablas = br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)
        assert tablas
        assert all(t.latex_layout == "landscape" for t in tablas)


class TestLasNegritasDicenLoMismoQueLaFrontera:
    """La negrita y la frontera de una misma tabla usan el mismo suelo."""

    def test_cada_negrita_es_lo_que_pareto_compare_dice_de_esa_pareja(self, analisis) -> None:
        from tfm_edge.analysis import floors as mf

        calidad, despliegue = _filas()
        tablas = {t.name: t for t in br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)}
        comprobadas = 0
        for vista, objetivos in pareto.named_views().items():
            tabla = tablas[f"frontera_{vista}_{DISPOSITIVO}"]
            marcas, _ = br.tables.best_marks(tabla)
            puntos = {
                (p.experiment_id, p.runtime, p.precision): p
                for p in pareto.build_points(calidad, despliegue, objetivos)
                if p.device_id == DISPOSITIVO
            }
            filas = [puntos[(f["experiment_id"], f["runtime"], f["precision"])] for f in tabla.rows]
            for objetivo in objetivos:
                suelo = br.floor_of(analisis, vista, objetivo.column)
                if suelo is None:
                    continue
                suelos = mf.FloorSet(
                    floors={
                        objetivo.column: mf.AxisFloor(
                            axis=objetivo.column,
                            kind=suelo["kind"],
                            same_session=suelo["same_session"],
                            cross_session=suelo["cross_session"],
                            provenance=suelo["provenance"],
                        )
                    }
                )
                valores = [p.values[objetivo.column] for p in filas]
                mejor = max(valores) if objetivo.direction == "max" else min(valores)
                mejores = [p for p in filas if p.values[objetivo.column] == mejor]
                indistinguibles = {
                    indice
                    for indice, p in enumerate(filas)
                    if any(
                        pareto.compare(m, p, objetivo, suelos, pareto.same_session(m, p)).verdict
                        == "indistinguible"
                        for m in mejores
                    )
                }
                if objetivo.column not in marcas:
                    # Columna entera indistinguible: no se marca nada, y se dice al pie.
                    assert indistinguibles == set(range(len(filas))), (vista, objetivo.column)
                else:
                    assert marcas[objetivo.column] == indistinguibles, (vista, objetivo.column)
                comprobadas += 1
        assert comprobadas >= 5, "la comprobación tiene que haber recorrido columnas con suelo"


class TestLaTablaDeMedidasPartida:
    """La 6.1 se parte en calidad y despliegue sin perder nada."""

    def test_las_dos_mitades_llevan_las_mismas_filas_en_el_mismo_orden(self) -> None:
        """Si no, la fila 12 de una y la 12 de la otra hablarían de cosas distintas."""
        calidad, despliegue = br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)[:2]
        assert calidad.name == f"medidas_calidad_{DISPOSITIVO}"
        assert despliegue.name == f"medidas_despliegue_{DISPOSITIVO}"
        clave = ("experiment_id", "runtime")
        assert [tuple(f[c] for c in clave) for f in calidad.rows] == [
            tuple(f[c] for c in clave) for f in despliegue.rows
        ]

    def test_entre_las_dos_estan_las_once_columnas(self) -> None:
        """Partirla no puede dejar ninguna medida fuera."""
        calidad, despliegue = br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)[:2]
        identificacion = ["experiment_id", "technique", "runtime", "precision"]
        assert [c.key for c in calidad.columns][:4] == identificacion
        assert [c.key for c in despliegue.columns][:4] == identificacion
        medidas = {c.key for c in calidad.columns} | {c.key for c in despliegue.columns}
        assert len(medidas) == 11, sorted(medidas)


class TestElScript:
    """Lo que decide el script antes de construir nada."""

    def test_un_dispositivo_que_no_esta_se_rechaza(self) -> None:
        """Mejor que una tabla vacía titulada con ese dispositivo."""
        script = _cargar_script()
        with pytest.raises(ValueError, match="no está en el registro"):
            script.elegir_dispositivo(REGISTRO, "una_raspberry_que_no_hay")

    def test_con_uno_solo_no_hace_falta_elegir(self) -> None:
        """Hoy la campaña tiene un dispositivo; pedirlo a mano sería ruido."""
        script = _cargar_script()
        assert script.elegir_dispositivo(REGISTRO, None) == DISPOSITIVO

    def test_las_tablas_se_escriben_en_los_tres_formatos(self, tmp_path) -> None:
        manifiesto = br.build(
            RAIZ,
            REGISTRO,
            ANALISIS,
            tmp_path / "tablas",
            RAIZ / "results" / "figures",
            DISPOSITIVO,
            skip_figures=True,
        )
        assert manifiesto["tables"]
        for tabla in manifiesto["tables"]:
            assert set(tabla["files"]) == {"tex", "xlsx", "csv"}
            for ruta in tabla["files"].values():
                # Se resuelve contra RAIZ y no contra el directorio de trabajo, porque el manifiesto
                # guarda rutas relativas.
                assert (RAIZ / ruta).is_file()
        assert manifiesto["figures"] == {"skipped": True}

    def test_dry_run_no_escribe_nada(self, tmp_path) -> None:
        """La comprobación que más veces ha hecho falta en este proyecto."""
        destino = tmp_path / "tablas"
        manifiesto = br.build(
            RAIZ,
            REGISTRO,
            ANALISIS,
            destino,
            RAIZ / "results" / "figures",
            DISPOSITIVO,
            skip_figures=True,
            dry_run=True,
        )
        assert manifiesto["tables"], "se ha construido, aunque no se escriba"
        assert all("files" not in tabla for tabla in manifiesto["tables"])
        assert not destino.exists(), "--dry-run ha creado el directorio de destino"


class TestElManifiestoVersionadoEsElDeHoy:
    """El manifiesto versionado coincide con el que se genera ahora."""

    @pytest.fixture()
    def manifiesto(self) -> dict:
        """El manifiesto versionado, tal cual está en disco."""
        ruta = ANALISIS / br.MANIFEST_NAME
        if not ruta.is_file():
            pytest.skip("El paso 28 no se ha ejecutado en este árbol.")
        return json.loads(ruta.read_text(encoding="utf-8"))

    def test_declara_las_figuras_que_hay_hoy(self, manifiesto: dict) -> None:
        """Si entra una figura y nadie relanza el 28, falla aquí y no en la memoria."""
        en_disco = sorted(ruta.name for ruta in (RAIZ / "results" / "figures").glob("*.pdf"))

        assert sorted(manifiesto["figures_on_disk"]) == en_disco, (
            "El manifiesto del informe y results/figures/ no dicen lo mismo. "
            "Regenéralo con 'python scripts/28_build_report.py --on-existing "
            "overwrite --skip-figures'."
        )

    def test_el_reparto_por_capitulos_no_pierde_ninguna(self, manifiesto: dict) -> None:
        """Una figura en disco que no cae en ningún capítulo no la ve el test de arriba."""
        repartidas = sorted(
            figura for figuras in manifiesto["figures_by_chapter"].values() for figura in figuras
        )

        assert repartidas == sorted(manifiesto["figures_on_disk"]), (
            "Hay figuras en el manifiesto que no están en ningún capítulo, o al revés: "
            f"{sorted(set(repartidas) ^ set(manifiesto['figures_on_disk']))}"
        )

    def test_no_lleva_ninguna_ruta_absoluta(self, manifiesto: dict) -> None:
        """Las 15 que tenía se fueron al regenerarlo, y no deben volver."""
        rutas = [
            valor
            for valor in re.findall(r'"([^"]+)"', json.dumps(manifiesto, ensure_ascii=False))
            if is_absolute_path_text(valor)
        ]

        assert not rutas, f"el manifiesto vuelve a llevar rutas absolutas: {rutas[:4]}"


class TestLoQueElPasoNoHaComprobado:
    """«Terminó sin error» se lee como «está comprobado» si nadie dice lo contrario."""

    def test_la_compilacion_del_tex_sigue_pendiente(self) -> None:
        """Salir bien formado no es que pdflatex lo acepte."""
        avisos = " ".join(br.PENDING_VERIFICATIONS).lower()

        assert br.PENDING_VERIFICATIONS
        assert "pdflatex" in avisos
        assert "compil" in avisos

    def test_los_avisos_salen_en_el_resumen(self) -> None:
        """De nada sirven si solo viven en una constante."""
        manifiesto = {
            "tables": [],
            "figures": {"skipped": True},
            "pending": [],
            "pending_verifications": list(br.PENDING_VERIFICATIONS),
        }

        lineas = "\n".join(br.summarize(manifiesto))

        assert "Sin comprobar" in lineas
        for aviso in br.PENDING_VERIFICATIONS:
            assert aviso in lineas


class TestLaTablaLlevaLaFronteraDeSuMotor:
    """La columna por motor, del volcado y no recalculada (commit de tab_6_4 por motor)."""

    def test_cada_fila_dice_lo_que_dice_el_grupo_de_su_motor(self, analisis) -> None:
        """Contado desde la tabla, contra el grupo ``device_id+runtime`` del volcado."""
        tablas = {t.name: t for t in br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO)}
        comprobadas = 0
        for vista in pareto.named_views():
            tabla = tablas[f"frontera_{vista}_{DISPOSITIVO}"]
            grupos = analisis["views"][vista]["groupings"]["device_id+runtime"]
            for fila in tabla.rows:
                grupo = grupos[f"{DISPOSITIVO}/{fila['runtime']}"]
                etiqueta = f"{fila['experiment_id']} [{fila['runtime']}/{fila['precision']}]"
                esperado = (
                    "estricta"
                    if etiqueta in grupo["frontier_strict"]
                    else "solo con suelo" if etiqueta in grupo["frontier_robust"] else "dominado"
                )
                assert fila["frontier_runtime"] == esperado, (vista, etiqueta)
                comprobadas += 1
        assert comprobadas

    def test_la_nota_de_dominado_es_la_regla_del_codigo(self) -> None:
        tabla = next(
            t for t in br.build_tables(REGISTRO, ANALISIS, DISPOSITIVO) if "frontera" in t.name
        )
        nota = " ".join(tabla.notes)
        assert "mejor en todos los ejes" not in nota
        assert "no es peor en ningún eje" in nota
        assert "por encima del suelo" in nota

    def test_sin_el_grupo_de_un_motor_no_hay_tabla(self, analisis) -> None:
        """Una columna por motor vacía se leería como «dominado» en todas las filas."""
        import copy

        vista = "plano_map5095_p50"
        alterado = copy.deepcopy(analisis)
        alterado["views"][vista]["groupings"]["device_id+runtime"].pop(
            f"{DISPOSITIVO}/onnxruntime"
        )
        puntos = [
            punto
            for punto in pareto.build_points(*_filas(), pareto.named_views()[vista])
            if punto.device_id == DISPOSITIVO
        ]
        with pytest.raises(ValueError, match="no tiene el grupo"):
            br.build_frontier_table(puntos, alterado, vista, DISPOSITIVO)
