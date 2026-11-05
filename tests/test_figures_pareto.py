"""Las figuras de la frontera, y los dos fallos que solo se ven mirando el PDF."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tfm_edge.analysis import floors as mf
from tfm_edge.analysis import pareto as mp
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry
from tfm_edge.reporting import figures_pareto as fp
from tfm_edge.reporting import style

RAIZ = Path(__file__).resolve().parent.parent

OBJETIVOS_2D = (
    mp.Objective("mAP@50", "map_50", "quality", "max"),
    mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
)


@pytest.fixture(autouse=True, scope="module")
def _estilo() -> None:
    """Aplica el estilo del proyecto antes de dibujar nada."""
    style.apply_style()


def _punto(
    experimento: str,
    motor: str,
    entorno: str,
    **valores: float,
) -> mp.Point:
    """Construye un punto de prueba con un ``experiment_id`` válido."""
    return mp.Point(
        experiment_id=experimento,
        runtime=motor,
        precision="fp32",
        device_id="dev",
        environment_id=entorno,
        timestamp="2026-08-27T00:00:00",
        values=dict(valores),
    )


def _suelos_de_juguete() -> mf.FloorSet:
    """Un conjunto de suelos con un eje relativo y otro sin medir."""
    return mf.FloorSet(
        floors={
            "latency_p50_ms": mf.AxisFloor(
                axis="latency_p50_ms",
                kind="relativo",
                same_session=0.10,
                cross_session=0.14,
                provenance="de juguete",
            ),
        }
    )


class TestEjesCompartidos:
    """Fijar límites de un eje compartido los congela y deja un panel vacío."""

    def test_todos_los_paneles_abarcan_el_punto_mas_lento(self) -> None:
        """El eje compartido tiene que llegar al punto más lento de cualquier motor."""
        rapido = [
            _punto(
                "s_baseline_default_aaaaaaaa", "onnxruntime", "e1", map_50=0.5, latency_p50_ms=40.0
            ),
            _punto("s_ptq_int8_bbbbbbbb", "onnxruntime", "e1", map_50=0.45, latency_p50_ms=50.0),
        ]
        lento = [
            _punto(
                "s_baseline_default_aaaaaaaa", "pytorch", "e1", map_50=0.5, latency_p50_ms=222.0
            ),
        ]
        todos = [*rapido, *lento]
        suelos = _suelos_de_juguete()
        por_motor = {"onnxruntime": rapido, "pytorch": lento}
        fronteras = {
            motor: (
                mp.frontier(lista, OBJETIVOS_2D, suelos, mp.MODE_STRICT),
                mp.frontier(lista, OBJETIVOS_2D, suelos, mp.MODE_ROBUST),
            )
            for motor, lista in por_motor.items()
        }

        figura, _ = fp.figure_frontier_by_device(
            por_motor,
            fronteras,
            todos,
            (
                mp.frontier(todos, OBJETIVOS_2D, suelos, mp.MODE_STRICT),
                mp.frontier(todos, OBJETIVOS_2D, suelos, mp.MODE_ROBUST),
            ),
            OBJETIVOS_2D,
            suelos,
            "dev",
        )
        try:
            paneles = [ejes for ejes in figura.axes if ejes.get_xlabel()]
            assert paneles, "no se ha dibujado ningún panel con datos"
            for ejes in paneles:
                assert ejes.get_xlim()[1] >= 222.0, (
                    f"el panel '{ejes.get_title()}' recorta en "
                    f"{ejes.get_xlim()[1]:.1f} ms y deja fuera el punto de 222 ms"
                )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)


class TestSesionDeReferencia:
    """La misma medida no puede salir marcada en un panel y sin marcar en otro."""

    def test_es_la_mayoritaria(self) -> None:
        puntos = [
            _punto("s_baseline_default_aaaaaaaa", "a", "vieja", map_50=0.5, latency_p50_ms=40.0),
            _punto("s_ptq_int8_bbbbbbbb", "a", "nueva", map_50=0.5, latency_p50_ms=41.0),
            _punto("s_qat_int8_cccccccc", "a", "nueva", map_50=0.5, latency_p50_ms=42.0),
        ]
        assert fp.reference_environment(puntos) == "nueva"
        assert fp.cross_session_labels(puntos, "nueva") == {"s_baseline_default_aaaaaaaa [a/fp32]"}

    def test_el_empate_avisa_y_es_determinista(self, caplog) -> None:
        """Con un empate, «la sesión» deja de significar algo y hay que decirlo."""
        puntos = [
            _punto("s_baseline_default_aaaaaaaa", "a", "bbb", map_50=0.5, latency_p50_ms=40.0),
            _punto("s_ptq_int8_bbbbbbbb", "a", "aaa", map_50=0.5, latency_p50_ms=41.0),
        ]
        with caplog.at_level("WARNING"):
            primero = fp.reference_environment(puntos)
        assert primero == "aaa"
        assert fp.reference_environment(list(reversed(puntos))) == "aaa"
        assert "empatados" in caplog.text

    def test_un_motor_de_otra_campana_no_declara_su_propia_referencia(self) -> None:
        nuevos = [
            _punto(
                "s_baseline_default_aaaaaaaa",
                "onnxruntime",
                "nueva",
                map_50=0.5,
                latency_p50_ms=40.0,
            ),
            _punto("s_ptq_int8_bbbbbbbb", "onnxruntime", "nueva", map_50=0.45, latency_p50_ms=50.0),
        ]
        viejos = [
            _punto(
                "s_baseline_default_aaaaaaaa", "pytorch", "vieja", map_50=0.5, latency_p50_ms=222.0
            ),
        ]
        todos = [*nuevos, *viejos]
        suelos = _suelos_de_juguete()
        por_motor = {"onnxruntime": nuevos, "pytorch": viejos}
        fronteras = {
            motor: (
                mp.frontier(lista, OBJETIVOS_2D, suelos, mp.MODE_STRICT),
                mp.frontier(lista, OBJETIVOS_2D, suelos, mp.MODE_ROBUST),
            )
            for motor, lista in por_motor.items()
        }

        figura, resumen = fp.figure_frontier_by_device(
            por_motor,
            fronteras,
            todos,
            (
                mp.frontier(todos, OBJETIVOS_2D, suelos, mp.MODE_STRICT),
                mp.frontier(todos, OBJETIVOS_2D, suelos, mp.MODE_ROBUST),
            ),
            OBJETIVOS_2D,
            suelos,
            "dev",
        )
        try:
            referencias = {datos["reference_environment"] for datos in resumen.values()}
            assert referencias == {"nueva"}, "cada panel ha elegido su propia sesión"
            etiqueta = "s_baseline_default_aaaaaaaa [pytorch/fp32]"
            assert etiqueta in resumen["pytorch"]["cross_session"]
            assert etiqueta in resumen["__dispositivo__"]["cross_session"]
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)


class TestLaEnvolventeSoloCuandoSignifica:
    """Una escalera sobre la proyección de una frontera de tres ejes no es nada."""

    def test_la_rejilla_rechaza_tres_objetivos(self) -> None:
        """Dibuja envolvente, así que exige que la frontera sea de esos dos ejes."""
        suelos = _suelos_de_juguete()
        puntos = [_punto("s_baseline_default_aaaaaaaa", "a", "e1", map_50=0.5, latency_p50_ms=40.0)]
        vacia = mp.frontier(puntos, OBJETIVOS_2D, suelos, mp.MODE_STRICT)
        with pytest.raises(ValueError, match="los dos ejes que dibuja"):
            fp.figure_frontier_by_device(
                {"a": puntos},
                {"a": (vacia, vacia)},
                puntos,
                (vacia, vacia),
                mp.default_objectives(),
                suelos,
                "dev",
            )

    def test_la_de_memoria_rechaza_dos_objetivos(self) -> None:
        suelos = _suelos_de_juguete()
        puntos = [_punto("s_baseline_default_aaaaaaaa", "a", "e1", map_50=0.5, latency_p50_ms=40.0)]
        vacia = mp.frontier(puntos, OBJETIVOS_2D, suelos, mp.MODE_STRICT)
        with pytest.raises(ValueError, match="exactamente tres"):
            fp.figure_frontier_with_memory(puntos, vacia, vacia, OBJETIVOS_2D, suelos, "dev")


class TestElPieDeclaraElSuelo:
    """Lo que la figura dice del suelo es la mitad de su razón de ser."""

    def test_un_eje_sin_suelo_lo_dice(self) -> None:
        """Callarlo dejaría leer separaciones pequeñas como diferencias."""
        objetivo = mp.Objective("mAP@50", "map_50", "quality", "max")
        texto = fp._floor_caption(None, objetivo, 0.0, cross_session_applies=True)
        assert "sin suelo medido" in texto
        assert "no se certifica ninguna ventaja" in texto

    def test_dos_suelos_iguales_se_dicen_una_vez(self) -> None:
        """Repetir la misma cifra con dos rótulos sugiere una diferencia inexistente."""
        suelo = mf.AxisFloor(
            axis="latency_p95_ms",
            kind="relativo",
            same_session=0.1472,
            cross_session=0.1472,
            provenance="de juguete",
        )
        objetivo = mp.Objective("latencia p95", "latency_p95_ms", "deployment", "min")
        texto = fp._floor_caption(suelo, objetivo, 100.0, cross_session_applies=True)
        assert texto.count("14,7 %") == 1
        assert "igual dentro y entre sesiones" in texto
        assert "14,72 ms aquí" in texto

    def test_dos_suelos_distintos_se_dicen_los_dos(self) -> None:
        """Cuando difieren, la diferencia es información."""
        objetivo = mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min")
        texto = fp._floor_caption(
            _suelos_de_juguete().get("latency_p50_ms"),
            objetivo,
            100.0,
            cross_session_applies=True,
        )
        assert "misma sesión 10,0 %" in texto
        assert "entre sesiones 14,0 %" in texto

    def test_con_una_sola_sesion_el_suelo_entre_sesiones_no_se_da_como_los_demas(self) -> None:
        """Un suelo que no gobierna ninguna comparación no puede presentarse igual."""
        objetivo = mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min")
        texto = fp._floor_caption(
            _suelos_de_juguete().get("latency_p50_ms"),
            objetivo,
            100.0,
            cross_session_applies=False,
        )
        assert "10,0 %" in texto
        assert "el que gobierna todas las comparaciones de esta figura" in texto
        assert "14,0 %" in texto, "el suelo entre sesiones sigue teniendo que aparecer"
        assert "no se dibuja" in texto
        assert "ningún punto de otra sesión" in texto

    def test_el_suelo_relativo_no_se_da_en_unidades_sin_anclaje(self) -> None:
        """Sin anclaje, «10 ms» sería un número inventado."""
        objetivo = mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min")
        texto = fp._floor_caption(
            _suelos_de_juguete().get("latency_p50_ms"),
            objetivo,
            0.0,
            cross_session_applies=True,
        )
        assert "aquí" not in texto


class TestFormato:
    """El texto de las figuras va en español."""

    def test_coma_decimal(self) -> None:
        """Y sin depender del locale de la máquina, que no es reproducible."""
        assert fp.format_number(14.72, 2) == "14,72"
        assert fp.format_number(0.004579, 4) == "0,0046"

    def test_la_tecnica_sale_del_identificador(self) -> None:
        """El color de la familia no se elige a ojo."""
        punto = _punto(
            "s_combined_prune-l1global-r50-ptqint8_9067dba2",
            "openvino",
            "e1",
            map_50=0.5,
            latency_p50_ms=40.0,
        )
        assert fp.technique_of(punto) == "combined"


class TestLoQueLaFiguraDiceDeSuAlcance:
    """Rótulos, alcance de la campaña y advertencia del tamaño del marcador."""

    def _campana(self):
        """Monta una campaña con un par indistinguible y un motor sin cobertura."""
        suelos = _suelos_de_juguete()
        # Mismo mAP y latencias que difieren un 5 %, por debajo del suelo del
        # 10 %: el suelo no los separa en ningún eje.
        rapido = [
            _punto(
                "s_baseline_default_aaaaaaaa", "onnxruntime", "e1", map_50=0.50, latency_p50_ms=40.0
            ),
            _punto("s_ptq_int8_bbbbbbbb", "onnxruntime", "e1", map_50=0.50, latency_p50_ms=42.0),
            _punto(
                "s_pruning_l1global_r50_cccccccc",
                "onnxruntime",
                "e1",
                map_50=0.46,
                latency_p50_ms=30.0,
            ),
        ]
        # PyTorch, como en la campaña real, solo tiene medido el baseline.
        lento = [
            _punto(
                "s_baseline_default_aaaaaaaa", "pytorch", "e1", map_50=0.50, latency_p50_ms=222.0
            ),
        ]
        return {"onnxruntime": rapido, "pytorch": lento}, [*rapido, *lento], suelos

    def _figura(self):
        """Dibuja la figura por dispositivo de la campaña de prueba."""
        por_motor, todos, suelos = self._campana()
        fronteras = {
            motor: (
                mp.frontier(lista, OBJETIVOS_2D, suelos, mp.MODE_STRICT),
                mp.frontier(lista, OBJETIVOS_2D, suelos, mp.MODE_ROBUST),
            )
            for motor, lista in por_motor.items()
        }
        return fp.figure_frontier_by_device(
            por_motor,
            fronteras,
            todos,
            (
                mp.frontier(todos, OBJETIVOS_2D, suelos, mp.MODE_STRICT),
                mp.frontier(todos, OBJETIVOS_2D, suelos, mp.MODE_ROBUST),
            ),
            OBJETIVOS_2D,
            suelos,
            "dev",
        )

    @staticmethod
    def _pie(figura) -> str:
        """Devuelve el texto de la franja inferior."""
        franjas = [ejes for ejes in figura.axes if not ejes.get_xlabel()]
        return "\n".join(t.get_text() for ejes in franjas for t in ejes.texts)

    def test_ningun_panel_lleva_el_rotulo_flotante(self) -> None:
        """Regresión del solapamiento con el título del panel."""
        figura, resumen = self._figura()
        try:
            assert resumen["onnxruntime"]["indistinguishable_drawn"] >= 1, (
                "el caso de prueba no tiene ningún par indistinguible, así que no " "prueba nada"
            )
            paneles = [ejes for ejes in figura.axes if ejes.get_xlabel()]
            for ejes in paneles:
                textos = [t.get_text() for t in ejes.texts]
                assert "indistinguibles" not in textos, (
                    f"el panel '{ejes.get_title()}' sigue llevando el rótulo "
                    f"flotante, que es el que se solapaba con el título"
                )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)

    def test_el_pie_cuenta_los_pares_de_cada_panel(self) -> None:
        """Lo que se quita de los ejes tiene que aparecer al pie."""
        figura, resumen = self._figura()
        try:
            pie = self._pie(figura)
            assert "trazo punteado" in pie, f"el pie no menciona los pares: {pie!r}"
            cuantos = resumen["onnxruntime"]["indistinguishable_drawn"]
            assert (
                f"{cuantos} en onnxruntime" in pie
            ), f"el pie no da el recuento del panel de onnxruntime ({cuantos}): {pie!r}"
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)

    def test_el_pie_declara_el_motor_sin_cobertura(self) -> None:
        """Un panel casi vacío no puede parecer un fallo de dibujo."""
        figura, _ = self._figura()
        try:
            pie = self._pie(figura)
            assert "cobertura desigual" in pie, f"el pie no declara la cobertura: {pie!r}"
            assert "pytorch" in pie, f"el pie no nombra el motor incompleto: {pie!r}"
            assert "1 de 4 puntos" in pie, f"el pie no da el recuento de puntos: {pie!r}"
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)

    def test_cada_miembro_de_un_par_indistinguible_lleva_su_propia_marca(self) -> None:
        from matplotlib.colors import to_rgba

        figura, resumen = self._figura()
        try:
            assert resumen["onnxruntime"]["indistinguishable_drawn"] == 1, (
                "el caso de prueba debe tener exactamente un par indistinguible "
                "para que el recuento de marcas sea comprobable"
            )
            (panel,) = [ejes for ejes in figura.axes if ejes.get_title() == "onnxruntime"]
            esperado = to_rgba(fp.FLOOR_COLOR)
            marcas = [
                coleccion
                for coleccion in panel.collections
                if len(coleccion.get_edgecolor())
                and tuple(coleccion.get_edgecolor()[0]) == esperado
            ]
            assert (
                len(marcas) == 2
            ), f"cada miembro del par tiene que llevar su marca; hay {len(marcas)}"
            for marca in marcas:
                assert (
                    len(marca.get_facecolor()) == 0
                ), "la marca es un contorno hueco: rellena taparía el punto que rodea"
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)

    def test_con_una_sola_sesion_no_se_dibuja_la_barra_de_entre_sesiones(self) -> None:
        """Una barra que domina la figura y no gobierna ninguna comparación engaña."""
        figura, _ = self._figura()
        try:
            paneles = [ejes for ejes in figura.axes if ejes.get_xlabel()]
            assert paneles, "la figura no tiene ningún panel con datos"
            for panel in paneles:
                barras = [linea for linea in panel.lines if linea.get_color() == fp.FLOOR_COLOR]
                trazos = {linea.get_linestyle() for linea in barras}
                assert (
                    "-" in trazos
                ), f"el panel '{panel.get_title()}' no declara el suelo de la misma sesión"
                assert "--" not in trazos, (
                    f"el panel '{panel.get_title()}' dibuja la barra de entre sesiones "
                    f"sin tener ningún punto de otra sesión"
                )
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)

    def test_sin_motores_incompletos_no_hay_nota(self) -> None:
        suelos = _suelos_de_juguete()
        resumen = {
            "a": {"families": ["baseline", "ptq"], "num_points": 2},
            "b": {"families": ["baseline", "ptq"], "num_points": 2},
        }
        assert fp._coverage_note(resumen, ["a", "b"], 4) is None
        assert suelos is not None

    def test_el_resumen_lleva_el_recuento_de_puntos(self) -> None:
        """El manifiesto tiene que permitir contar sin abrir el PDF."""
        figura, resumen = self._figura()
        try:
            assert resumen["pytorch"]["num_points"] == 1
            assert resumen["onnxruntime"]["num_points"] == 3
            assert resumen["__dispositivo__"]["num_points"] == 4
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)

    def test_la_advertencia_del_area_va_en_la_leyenda_de_tamanos(self) -> None:
        """La aclaración que hace falta para leer los tamaños, junto a los tamaños."""
        suelos = _suelos_de_juguete()
        objetivos = (
            mp.Objective("mAP@50", "map_50", "quality", "max"),
            mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
            mp.Objective("memoria pico", "peak_memory_mib", "deployment", "min"),
        )
        puntos = [
            _punto(
                "s_baseline_default_aaaaaaaa",
                "onnxruntime",
                "e1",
                map_50=0.50,
                latency_p50_ms=40.0,
                peak_memory_mib=500.0,
            ),
            _punto(
                "s_ptq_int8_bbbbbbbb",
                "onnxruntime",
                "e1",
                map_50=0.46,
                latency_p50_ms=30.0,
                peak_memory_mib=430.0,
            ),
        ]
        estricta = mp.frontier(puntos, objetivos, suelos, mp.MODE_STRICT)
        robusta = mp.frontier(puntos, objetivos, suelos, mp.MODE_ROBUST)
        figura, _ = fp.figure_frontier_with_memory(
            puntos, estricta, robusta, objetivos, suelos, "dev"
        )
        try:
            titulos = [
                leyenda.get_title().get_text()
                for ejes in figura.axes
                for leyenda in ejes.findobj(match=lambda o: o.__class__.__name__ == "Legend")
            ]
            memoria = [t for t in titulos if "Memoria pico" in t]
            assert memoria, f"no hay leyenda de tamaños: {titulos!r}"
            assert (
                "no proporcional al valor" in memoria[0]
            ), f"la leyenda de tamaños no lleva la advertencia: {memoria[0]!r}"
            pie = self._pie(figura)
            assert (
                "no proporcional al valor" not in pie
            ), "la advertencia sigue duplicada al pie, que es de donde se movía"
        finally:
            import matplotlib.pyplot as plt

            plt.close(figura)


class TestContraElRegistroReal:
    """Lo que la figura afirma sobre la campaña, comprobado con el registro."""

    @pytest.fixture(scope="class")
    def manifiesto(self) -> dict:
        """El manifiesto que dejó el paso 27."""
        destino = RAIZ / "results" / "raw" / "pareto" / "figures_manifest.json"
        if not destino.is_file():
            pytest.skip("no hay manifiesto: ejecuta scripts/27_build_pareto_figures.py")
        return json.loads(destino.read_text(encoding="utf-8"))

    #: Puntos medidos fuera de la sesión de referencia que están declarados. Está vacía desde que la
    #: campaña única volvió a medir las 50 filas en una sola sesión.
    SESIONES_DECLARADAS: dict[str, str] = {}

    def test_solo_quedan_puntos_de_otra_sesion_declarados(self, manifiesto) -> None:
        declarados = set(self.SESIONES_DECLARADAS)
        for vista, dispositivos in manifiesto["views"].items():
            for dispositivo, datos in dispositivos.items():
                marcados = set(datos["panels"]["__dispositivo__"]["cross_session"])
                assert marcados - declarados == set(), (
                    f"la vista '{vista}' marca puntos de otra sesión no declarados en "
                        f"'{dispositivo}': {sorted(marcados - declarados)}."
                )
                assert declarados - marcados == set(), (
                    f"la vista '{vista}' ya no marca {sorted(declarados - marcados)} como de "
                    "otra sesión; si se volvieron a medir, quítalos de SESIONES_DECLARADAS."
                )


    def test_la_figura_coincide_con_el_volcado_del_paso_26(self, manifiesto) -> None:
        """Si la figura y el análisis registrado discrepan, manda el análisis."""
        assert manifiesto["checked_against_step_26"] is True

        volcado = json.loads(
            (RAIZ / "results" / "raw" / "pareto" / "pareto_analysis.json").read_text(
                encoding="utf-8"
            )
        )
        for vista, dispositivos in manifiesto["views"].items():
            grupo = volcado["views"][vista]["groupings"]["device_id"]
            for dispositivo, datos in dispositivos.items():
                assert datos["frontier_strict"] == sorted(grupo[dispositivo]["frontier_strict"])
                assert datos["frontier_robust"] == sorted(grupo[dispositivo]["frontier_robust"])

    def test_ptq_y_qat_salen_como_par_indistinguible(self, manifiesto) -> None:
        """El resultado que la figura de tres objetivos tiene que contar."""
        analisis = json.loads(
            (RAIZ / "results" / "raw" / "pareto" / "pareto_analysis.json").read_text(
                encoding="utf-8"
            )
        )
        vista = analisis["views"]["spec_3d_map5095_p95_memoria"]
        pares = {
            tuple(sorted(par))
            for par in vista["groupings"]["device_id"]["workstation_cpu"]["indistinguishable_pairs"]
        }

        assert (
            "s_ptq_int8-noexcl_5bcd71f0 [openvino/int8]",
            "s_qat_int8-noexcl_d4d403fa [openvino/int8]",
        ) in pares

        manifiesto_vista = manifiesto["views"]["spec_3d_map5095_p95_memoria"]["workstation_cpu"]
        assert manifiesto_vista["panels"]["__dispositivo__"]["indistinguishable_drawn"] == len(
            pares
        )
        perdidas = {
            (d["dominator"], d["dominated"])
            for d in vista["groupings"]["device_id"]["workstation_cpu"]["dominations_lost_to_floor"]
        }
        assert perdidas == {
            (
                "s_qat_int8-noexcl_4d491880 [onnxruntime/int8]",
                "s_combined_prune-l1global-r25-ptqint8_4a1d5adc [onnxruntime/int8]",
            )
        }
        assert manifiesto_vista["dominations_lost_to_floor"] == len(perdidas)

    def test_map50_pierde_mas_dominaciones_que_map5095(self, manifiesto) -> None:
        """El precio de que `map_50` no tenga suelo medido, en la propia figura."""
        con_suelo = manifiesto["views"]["plano_map5095_p50"]["workstation_cpu"]
        sin_suelo = manifiesto["views"]["plano_map50_p50"]["workstation_cpu"]
        assert sin_suelo["dominations_lost_to_floor"] > con_suelo["dominations_lost_to_floor"]
        assert sin_suelo["dominations_lost_to_floor"] == 33
        assert con_suelo["dominations_lost_to_floor"] == 6

    def test_los_ficheros_de_figura_existen_en_los_dos_formatos(self, manifiesto) -> None:
        """El PDF es lo que se inserta en la memoria; el PNG, para revisar."""
        figuras = RAIZ / "results" / "figures"
        for dispositivos in manifiesto["views"].values():
            for datos in dispositivos.values():
                base = style.figure_filename(fp.CHAPTER, fp.SECTION, datos["figure"])
                for formato in style.FIGURE_FORMATS:
                    assert (figuras / f"{base}.{formato}").is_file()

    def test_el_dibujo_es_reproducible(self) -> None:
        """Dos renderizados del mismo dato dan los mismos bytes."""
        import io
        import os

        os.environ.setdefault("SOURCE_DATE_EPOCH", "1767225600")

        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        objetivos = mp.default_objectives()
        suelos = mf.measure_floors(despliegue, [o.column for o in objetivos], RAIZ)
        puntos = mp.build_points(calidad, despliegue, objetivos)
        estricta = mp.frontier(puntos, objetivos, suelos, mp.MODE_STRICT)
        robusta = mp.frontier(puntos, objetivos, suelos, mp.MODE_ROBUST)

        import matplotlib.pyplot as plt

        bytes_por_pasada = []
        for _ in range(2):
            figura, _ = fp.figure_frontier_with_memory(
                puntos, estricta, robusta, objetivos, suelos, "workstation_cpu"
            )
            buffer = io.BytesIO()
            figura.savefig(buffer, format="png")
            plt.close(figura)
            bytes_por_pasada.append(buffer.getvalue())

        assert bytes_por_pasada[0] == bytes_por_pasada[1]


def _cargar_paso_27() -> Any:
    """Carga ``scripts/27_build_pareto_figures.py`` como módulo."""
    import importlib.util
    import sys

    ruta = RAIZ / "scripts" / "27_build_pareto_figures.py"
    especificacion = importlib.util.spec_from_file_location("pareto_figures_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["pareto_figures_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def _suelos_de_plano() -> mf.FloorSet:
    """Suelos de prueba para el plano mAP@50-95 / p50, absoluto y relativo."""
    return mf.FloorSet(
        floors={
            "map_50_95": mf.AxisFloor(
                axis="map_50_95",
                kind="absoluto",
                same_session=0.01,
                cross_session=0.01,
                provenance="de juguete",
            ),
            "latency_p50_ms": mf.AxisFloor(
                axis="latency_p50_ms",
                kind="relativo",
                same_session=0.10,
                cross_session=0.20,
                provenance="de juguete",
            ),
        }
    )


OBJETIVOS_PLANO = (
    mp.Objective("mAP@50-95", "map_50_95", "quality", "max"),
    mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
)


class TestLaRegionDominadaEsLaDeDominates:
    """La región sombreada de la figura de nano frente a small coincide con
    :func:`pareto.dominates`.
    """

    def test_dentro_si_y_solo_si_dominado(self) -> None:
        from matplotlib.path import Path as Trayecto

        suelos = _suelos_de_plano()
        referencia = _punto("n_baseline_default_aaaaaaaa", "ort", "E", map_50_95=0.30,
                            latency_p50_ms=50.0)
        region = Trayecto(fp.dominated_region(referencia, OBJETIVOS_PLANO, suelos))
        comprobados = 0
        for latencia in (40.0, 49.0, 50.5, 54.0, 56.0, 80.0):
            for calidad in (0.32, 0.301, 0.299, 0.295, 0.285, 0.20):
                punto = _punto(
                    "s_ptq_int8-noexcl_bbbbbbbb",
                    "ort",
                    "E",
                    map_50_95=calidad,
                    latency_p50_ms=latencia,
                )
                dominado = mp.dominates(referencia, punto, OBJETIVOS_PLANO, suelos) is not None
                assert region.contains_point((latencia, calidad)) == dominado, (latencia, calidad)
                comprobados += dominado
        assert comprobados > 0

    def test_sin_suelo_de_algun_eje_no_hay_region(self) -> None:
        suelos = mf.FloorSet(floors={})
        referencia = _punto("n_baseline_default_aaaaaaaa", "ort", "E", map_50_95=0.30,
                            latency_p50_ms=50.0)
        with pytest.raises(ValueError, match="suelos"):
            fp.dominated_region(referencia, OBJETIVOS_PLANO, suelos)


class TestBloqueadoBajoElSuelo:
    """El anillo de «no dominado solo por un déficit bajo el suelo»."""

    def test_un_deficit_bajo_el_suelo_que_bloquea_se_detecta(self) -> None:
        suelos = _suelos_de_plano()
        n = _punto("n_ptq_int8-noexcl_aaaaaaaa", "ort", "E", map_50_95=0.2477,
                   latency_p50_ms=30.0)
        s = _punto("s_qat_int8-noexcl_bbbbbbbb", "ort", "E", map_50_95=0.24772,
                   latency_p50_ms=45.0)

        deficits = fp.blocked_below_floor(n, s, OBJETIVOS_PLANO, suelos)

        assert deficits is not None
        assert [c.column for c in deficits] == ["map_50_95"]
        assert -deficits[0].advantage == pytest.approx(0.00002)

    def test_un_deficit_por_encima_del_suelo_no_es_este_caso(self) -> None:
        suelos = _suelos_de_plano()
        n = _punto("n_ptq_int8-noexcl_aaaaaaaa", "ort", "E", map_50_95=0.25,
                   latency_p50_ms=30.0)
        s = _punto("s_qat_int8-noexcl_bbbbbbbb", "ort", "E", map_50_95=0.27,
                   latency_p50_ms=45.0)

        assert fp.blocked_below_floor(n, s, OBJETIVOS_PLANO, suelos) is None

    def test_si_lo_domina_no_es_este_caso(self) -> None:
        suelos = _suelos_de_plano()
        n = _punto("n_ptq_int8-noexcl_aaaaaaaa", "ort", "E", map_50_95=0.25,
                   latency_p50_ms=30.0)
        s = _punto("s_qat_int8-noexcl_bbbbbbbb", "ort", "E", map_50_95=0.24,
                   latency_p50_ms=45.0)

        assert fp.blocked_below_floor(n, s, OBJETIVOS_PLANO, suelos) is None


class TestElPaso27ContrastaCadaMotor:
    """Un panel por motor que no coincide con el volcado del 26 impide la figura."""

    @staticmethod
    def _caso() -> tuple[Any, dict, dict]:
        paso = _cargar_paso_27()
        suelos = _suelos_de_plano()
        puntos = [
            _punto("n_baseline_default_aaaaaaaa", "ort", "E", map_50_95=0.32, latency_p50_ms=50.0),
            _punto("s_baseline_default_bbbbbbbb", "ort", "E", map_50_95=0.37, latency_p50_ms=140.0),
            _punto("s_ptq_int8-noexcl_cccccccc", "ort", "E", map_50_95=0.28, latency_p50_ms=70.0),
        ]
        estricta = mp.frontier(puntos, OBJETIVOS_PLANO, suelos, mp.MODE_STRICT)
        robusta = mp.frontier(puntos, OBJETIVOS_PLANO, suelos, mp.MODE_ROBUST)
        grupo = {
            "frontier_strict": [p.label for p in estricta.nondominated],
            "frontier_robust": [p.label for p in robusta.nondominated],
            "num_dominations_strict": len(estricta.dominations),
            "num_dominations_robust": len(robusta.dominations),
            "dominations_lost_to_floor": [
                d.to_dict() for d in mp.dominations_lost_to_floor(estricta, robusta)
            ],
        }
        analisis = {"views": {"v": {"groupings": {"device_id+runtime": {"dev/ort": grupo}}}}}
        return paso, {"ort": (estricta, robusta)}, analisis

    def test_si_coincide_no_dice_nada(self) -> None:
        paso, fronteras, analisis = self._caso()
        paso._contrastar_motores("v", "dev", fronteras, analisis)

    def test_una_frontera_distinta_aborta(self) -> None:
        paso, fronteras, analisis = self._caso()
        analisis["views"]["v"]["groupings"]["device_id+runtime"]["dev/ort"][
            "frontier_robust"
        ].pop()
        with pytest.raises(mp.ParetoError, match="no dicen lo mismo"):
            paso._contrastar_motores("v", "dev", fronteras, analisis)

    def test_un_recuento_distinto_aborta(self) -> None:
        paso, fronteras, analisis = self._caso()
        analisis["views"]["v"]["groupings"]["device_id+runtime"]["dev/ort"][
            "num_dominations_strict"
        ] += 1
        with pytest.raises(mp.ParetoError, match="num_dominations_strict"):
            paso._contrastar_motores("v", "dev", fronteras, analisis)

    def test_un_grupo_que_falta_aborta(self) -> None:
        paso, fronteras, analisis = self._caso()
        analisis["views"]["v"]["groupings"]["device_id+runtime"] = {}
        with pytest.raises(mp.ParetoError, match="no tiene el grupo"):
            paso._contrastar_motores("v", "dev", fronteras, analisis)


@pytest.fixture(scope="module")
def generado_27() -> tuple[list[tuple[str, Any]], dict]:
    """Todas las figuras del paso 27 dibujadas sobre el registro y el volcado reales."""
    import matplotlib.pyplot as plt

    volcado = RAIZ / "results" / "raw" / "pareto"
    if not (volcado / "pareto_analysis.json").is_file():
        pytest.skip("no hay volcado del paso 26")
    paso = _cargar_paso_27()
    figuras, manifiesto = paso.generar(
        RAIZ / "results" / "registry", RAIZ, volcado, sorted(paso.FIGURES)
    )
    yield figuras, manifiesto
    for _, figura in figuras:
        plt.close(figura)


@pytest.fixture(scope="module")
def figuras_por_motor(generado_27) -> dict[str, tuple[Any, dict]]:
    """Las figuras por motor, con su entrada de manifiesto."""
    figuras, manifiesto = generado_27
    por_motor = manifiesto["engine_figures"]
    return {d: (f, por_motor[d]) for d, f in figuras if d in por_motor}


def _comprobar_etiquetas_sin_cruces(figura: Any, nombre: str) -> None:
    """Ningún texto de etiqueta lo cruza un borde de región, una escalera ni un trazo."""
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.text import Annotation, Text

    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    for eje in figura.axes:
        etiquetas = [a for a in eje.texts if isinstance(a, Annotation) and a.get_text().strip()]
        colocadas = [a for a in etiquetas if a.arrow_patch is not None or a.get_bbox_patch()]
        assert colocadas, nombre
        regiones = [
            p.get_path().transformed(p.get_transform())
            for p in eje.patches
            if isinstance(p, (Polygon, Rectangle))
        ]
        tramos = []
        for linea in eje.get_lines():
            if linea.get_drawstyle() != "steps-post":
                continue
            xs, ys = linea.get_xdata(), linea.get_ydata()
            for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:]):
                esquina = eje.transData.transform((x1, y0))
                tramos.append((*eje.transData.transform((x0, y0)), *esquina))
                tramos.append((*esquina, *eje.transData.transform((x1, y1))))
        for a in colocadas:
            caja = Text.get_window_extent(a, dibujo)
            rectangulo = (caja.x0, caja.x1, caja.y0, caja.y1)
            texto = a.get_text()
            for region in regiones:
                assert not region.intersects_bbox(caja, filled=True), (nombre, texto)
            for tramo in tramos:
                assert not fp._segment_hits_box(tramo, rectangulo), (nombre, texto)
            for otra in colocadas:
                if otra is a or otra.arrow_patch is None:
                    continue
                trazo = otra.arrow_patch.get_path().transformed(otra.arrow_patch.get_transform())
                assert not trazo.intersects_bbox(caja, filled=False), (
                    nombre,
                    texto,
                    otra.get_text(),
                )


class TestFigurasPorMotorSobreLaCampanaReal:
    """Las cinco figuras por motor del 6.4, dibujadas sobre los datos de verdad."""

    NOMBRES = (
        "frontera_map5095_p50_por_motor_workstation_cpu",
        "frontera_3obj_por_motor_workstation_cpu",
        "n_frente_a_compresiones_s_workstation_cpu",
        "dominaciones_perdidas_por_suelo_workstation_cpu",
        "miembros_por_tecnica_workstation_cpu",
    )

    @staticmethod
    def _volcado() -> dict:
        return json.loads(
            (RAIZ / "results" / "raw" / "pareto" / "pareto_analysis.json").read_text(
                encoding="utf-8"
            )
        )

    def test_salen_las_cinco(self, figuras_por_motor) -> None:
        assert set(figuras_por_motor) == set(self.NOMBRES)

    def test_caben_en_16_cm_sin_pie_y_ningun_texto_baja_de_6_5_pt(
        self, figuras_por_motor
    ) -> None:
        from matplotlib.text import Text

        for nombre, (figura, _) in figuras_por_motor.items():
            assert figura.get_figwidth() == pytest.approx(16.0 / 2.54), nombre
            figura.canvas.draw()
            contenido = figura.get_tightbbox(figura.canvas.get_renderer())
            assert contenido.x0 >= 0.0, nombre
            assert contenido.x1 <= figura.get_figwidth() + 1e-6, nombre
            assert not figura.texts, f"{nombre}: el pie va en el texto, no en la imagen"
            textos = [t for t in figura.findobj(Text) if t.get_visible() and t.get_text().strip()]
            assert min(t.get_fontsize() for t in textos) >= 6.5, nombre

    def test_cada_panel_por_motor_es_el_del_volcado(self, figuras_por_motor) -> None:
        """Los miembros que la figura rotula son la frontera robusta de ese motor."""
        volcado = self._volcado()
        vistas = {
            "frontera_map5095_p50_por_motor_workstation_cpu": "plano_map5095_p50",
            "frontera_3obj_por_motor_workstation_cpu": "spec_3d_map5095_p95_memoria",
        }
        for nombre, vista in vistas.items():
            resumen = figuras_por_motor[nombre][1]["summary"]
            grupos = volcado["views"][vista]["groupings"]["device_id+runtime"]
            for motor, panel in resumen["runtimes"].items():
                esperado = sorted(grupos[f"workstation_cpu/{motor}"]["frontier_robust"])
                assert sorted(m["point"] for m in panel["members"]) == esperado, (nombre, motor)

    def test_los_miembros_llevan_su_nombre_en_la_figura(self, figuras_por_motor) -> None:
        figura, entrada = figuras_por_motor["frontera_map5095_p50_por_motor_workstation_cpu"]
        textos = {t.get_text() for eje in figura.axes for t in eje.texts}
        for panel in entrada["summary"]["runtimes"].values():
            for miembro in panel["members"]:
                assert miembro["label"] in textos

    def test_los_controles_van_huecos_y_poda_mas_qat_con_su_estilo(
        self, figuras_por_motor
    ) -> None:
        """Mismos códigos que la figura del 6.2."""
        import numpy as np

        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos = {f["experiment_id"]: f for f in registro.valid_rows("experiments")[0]}
        figura, _ = figuras_por_motor["frontera_map5095_p50_por_motor_workstation_cpu"]
        eje = figura.axes[0]
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        puntos = [
            p
            for p in mp.build_points(calidad, despliegue, OBJETIVOS_PLANO)
            if p.runtime == "onnxruntime"
        ]
        identidades = fp.point_identities(puntos, experimentos)
        controles = [p for p in puntos if identidades[p.label]["control"]]
        assert controles, "la campaña tiene controles aleatorios"
        for punto in controles:
            caras = [
                c.get_facecolor()[0]
                for c in eje.collections
                if len(c.get_offsets()) == 1
                and np.allclose(
                    c.get_offsets()[0],
                    (punto.values["latency_p50_ms"], punto.values["map_50_95"]),
                )
            ]
            assert caras and all(np.allclose(c[:3], (1.0, 1.0, 1.0)) for c in caras)
        poda_qat = [p for p in puntos if identidades[p.label]["category"] == "pruning_qat"]
        assert poda_qat
        assert fp.category_style("pruning_qat") is style.PRUNING_QAT_STYLE
        for punto in poda_qat:
            cadena = fp.experiment_chain(punto.experiment_id, experimentos)
            assert experimentos[cadena[0]]["technique"] == "qat"
            assert any(experimentos[e]["technique"] == "pruning" for e in cadena[1:])

    def test_las_regiones_de_nano_dicen_lo_que_dice_dominates(self, figuras_por_motor) -> None:
        """Qué domina cada referencia de nano, recalculado aparte de la figura."""
        _, entrada = figuras_por_motor["n_frente_a_compresiones_s_workstation_cpu"]
        volcado = self._volcado()
        for motor, panel in entrada["summary"]["runtimes"].items():
            dominaciones = volcado["views"]["plano_map5095_p50"]["groupings"]["device_id+runtime"][
                f"workstation_cpu/{motor}"
            ]["dominations_robust"]
            assert set(panel["references"]) == {"n FP32", "n PTQ"}
            for referencia, dominados in panel["dominated_by"].items():
                prefijo = "n_baseline" if referencia == "n FP32" else "n_ptq"
                esperados = {
                    d["dominated"]
                    for d in dominaciones
                    if d["dominator"].startswith(prefijo) and d["dominated"].startswith("s_")
                }
                assert len(dominados) == len(esperados), (motor, referencia)

    def test_l1_r50_qat_en_ort_es_el_caso_bajo_el_suelo(self, figuras_por_motor) -> None:
        """El caso que la figura tiene que enseñar sin llamarlo «más calidad»."""
        _, entrada = figuras_por_motor["n_frente_a_compresiones_s_workstation_cpu"]
        casos = entrada["summary"]["runtimes"]["onnxruntime"]["blocked_below_floor"]
        caso = next(
            c for c in casos if c["point"] == "L1 r50 + QAT" and c["reference"] == "n PTQ"
        )
        assert caso["column"] == "map_50_95"
        assert caso["deficit"] == pytest.approx(0.000015, abs=1e-6)
        assert caso["deficit"] < caso["threshold"]

    def test_en_nano_las_etiquetas_no_se_tocan_ni_pisan_marcadores_ni_anillos(
        self, figuras_por_motor
    ) -> None:
        import math

        from matplotlib.text import Annotation, Text

        figura, _ = figuras_por_motor["n_frente_a_compresiones_s_workstation_cpu"]
        figura.canvas.draw()
        dibujo = figura.canvas.get_renderer()
        ppp = figura.dpi
        radio_marcador = math.sqrt(fp.ENGINE_MARKER_AREA) / 2.0 * ppp / 72.0
        for eje in figura.axes:
            # La caja del texto solo: la de la anotación incluye su trazo de unión,
            # que por construcción llega hasta el marcador.
            cajas = [
                Text.get_window_extent(a, dibujo)
                for a in eje.texts
                if isinstance(a, Annotation) and a.get_text()
            ]
            assert cajas
            for i, a in enumerate(cajas):
                for b in cajas[i + 1 :]:
                    assert not a.overlaps(b), (a, b)
            for coleccion in eje.collections:
                tamanos = coleccion.get_sizes()
                if not len(tamanos) or not len(coleccion.get_offsets()):
                    continue
                radio = (
                    math.sqrt(float(tamanos[0])) / 2.0 * ppp / 72.0
                    if float(tamanos[0]) > fp.ENGINE_MARKER_AREA
                    else radio_marcador
                )
                for x, y in coleccion.get_offsets():
                    px, py = eje.transData.transform((x, y))
                    for caja in cajas:
                        cerca_x = max(caja.x0, min(px, caja.x1))
                        cerca_y = max(caja.y0, min(py, caja.y1))
                        assert math.hypot(px - cerca_x, py - cerca_y) >= radio - 1e-6

    @pytest.mark.parametrize(
        "nombre",
        [
            "n_frente_a_compresiones_s_workstation_cpu",
            "frontera_map5095_p50_por_motor_workstation_cpu",
        ],
    )
    def test_ninguna_etiqueta_la_cruza_un_borde_una_escalera_ni_un_trazo(
        self, figuras_por_motor, nombre: str
    ) -> None:
        figura, _ = figuras_por_motor[nombre]
        _comprobar_etiquetas_sin_cruces(figura, nombre)

    def test_en_nano_la_diferencia_bajo_el_suelo_no_se_escribe_en_el_panel(
        self, figuras_por_motor
    ) -> None:
        """Las dos frases rojas las explica el texto; los anillos se quedan."""
        figura, entrada = figuras_por_motor["n_frente_a_compresiones_s_workstation_cpu"]
        textos = [t.get_text() for eje in figura.axes for t in eje.texts]
        assert not any("no domina a" in t for t in textos)
        assert entrada["summary"]["runtimes"]["onnxruntime"]["blocked_below_floor"]
        anillos = [
            c
            for eje in figura.axes
            for c in eje.collections
            if len(c.get_sizes())
            and float(c.get_sizes()[0]) == fp.ENGINE_MARKER_AREA * fp.RING_AREA_FACTOR
        ]
        assert anillos

    def test_los_recuentos_son_los_del_volcado(self, figuras_por_motor) -> None:
        _, entrada = figuras_por_motor["dominaciones_perdidas_por_suelo_workstation_cpu"]
        volcado = self._volcado()
        for fila in entrada["summary"]["counts"]:
            agrupaciones = volcado["views"][fila["view"]]["groupings"]
            grupo = (
                agrupaciones["device_id"]["workstation_cpu"]
                if fila["group"] == "dispositivo"
                else agrupaciones["device_id+runtime"][f"workstation_cpu/{fila['group']}"]
            )
            assert fila["strict"] == grupo["num_dominations_strict"]
            assert fila["robust"] == grupo["num_dominations_robust"]
            assert fila["lost"] == len(grupo["dominations_lost_to_floor"])

    def test_la_matriz_de_pertenencia_es_la_del_volcado(self, figuras_por_motor) -> None:
        _, entrada = figuras_por_motor["miembros_por_tecnica_workstation_cpu"]
        volcado = self._volcado()
        for columna in entrada["summary"]["columns"]:
            vista, motor = columna.split("/")
            grupo = volcado["views"][vista]["groupings"]["device_id+runtime"][
                f"workstation_cpu/{motor}"
            ]
            robustos = sum(
                1 for f in entrada["summary"]["rows"] if f["states"][columna] != "dominado"
            )
            estrictos = sum(
                1 for f in entrada["summary"]["rows"] if f["states"][columna] == "estricta"
            )
            assert robustos == len(grupo["frontier_robust"]), columna
            assert estrictos == len(grupo["frontier_strict"]), columna

    def test_el_manifiesto_versionado_las_trae_y_estan_en_disco(self) -> None:
        destino = RAIZ / "results" / "raw" / "pareto" / "figures_manifest.json"
        if not destino.is_file():
            pytest.skip("no hay manifiesto: ejecuta scripts/27_build_pareto_figures.py")
        manifiesto = json.loads(destino.read_text(encoding="utf-8"))
        assert set(manifiesto["engine_figures"]) == set(self.NOMBRES)
        for nombre in self.NOMBRES:
            base = style.figure_filename(fp.CHAPTER, fp.SECTION, nombre)
            for formato in style.FIGURE_FORMATS:
                assert (RAIZ / "results" / "figures" / f"{base}.{formato}").is_file()


def _entradas_resultado_principal() -> tuple[Any, ...]:
    """Lo que el paso 27 pasa a la figura del resultado principal, sobre la campaña real."""
    paso = _cargar_paso_27()
    objetivos = paso.FIGURES["plano_map5095_p50"]["objectives"]
    puntos, suelos = paso._cargar_puntos(RAIZ / "results" / "registry", RAIZ, objetivos)
    por_motor = {
        clave[1]: lista
        for clave, lista in mp.group_points(puntos, ("device_id", "runtime")).items()
        if clave[0] == "workstation_cpu"
    }
    fronteras = {m: paso._fronteras(lista, objetivos, suelos) for m, lista in por_motor.items()}
    experimentos, _ = ExperimentRegistry(RAIZ / "results" / "registry").valid_rows("experiments")
    identidades = fp.point_identities(
        list(puntos), {f["experiment_id"]: f for f in experimentos}
    )
    return por_motor, fronteras, objetivos, suelos, identidades


class TestResultadoPrincipalDelCapitulo1:
    """Figura del resultado principal para la introducción."""

    @staticmethod
    def _figura(generado_27) -> tuple[Any, dict]:
        figuras, manifiesto = generado_27
        intro = manifiesto["intro_figures"]
        assert len(intro) == 1
        nombre = next(iter(intro))
        return dict(figuras)[nombre], intro[nombre]

    def test_va_al_capitulo_1_aunque_la_dibuje_el_paso_27(self) -> None:
        nombre = f"{fp.HEADLINE_DESCRIPTION}_workstation_cpu"
        assert fp.figure_placement(nombre) == (1, 2)
        assert fp.figure_placement("frontera_map5095_p50_por_motor_workstation_cpu") == (6, 4)

    def test_rotula_sus_seis_miembros_y_la_zona_en_los_dos_motores(self, generado_27) -> None:
        figura, entrada = self._figura(generado_27)
        for eje, (motor, panel) in zip(figura.axes, entrada["summary"]["runtimes"].items()):
            textos = {t.get_text() for t in eje.texts}
            assert set(fp.HEADLINE_LABELS) <= textos, motor
            assert fp.HEADLINE_ZONE_LABEL in textos, motor
            assert set(fp.HEADLINE_LABELS) <= set(panel["members"]), motor
            assert panel["zone"], motor
            assert all("r50" in e for e in panel["zone"]), motor

    def test_cabe_en_16_cm_sin_pie_y_ningun_texto_baja_de_7_pt(self, generado_27) -> None:
        from matplotlib.text import Text

        figura, _ = self._figura(generado_27)
        figura.canvas.draw()
        contenido = figura.get_tightbbox(figura.canvas.get_renderer())
        textos = [t for t in figura.findobj(Text) if t.get_visible() and t.get_text().strip()]
        assert figura.get_figwidth() == pytest.approx(16.0 / 2.54)
        assert contenido.x0 >= 0.0 and contenido.x1 <= figura.get_figwidth() + 1e-6
        assert not figura.texts
        assert min(t.get_fontsize() for t in textos) >= 7.0

    def test_ninguna_etiqueta_queda_cruzada(self, generado_27) -> None:
        figura, _ = self._figura(generado_27)
        _comprobar_etiquetas_sin_cruces(figura, fp.HEADLINE_DESCRIPTION)

    def test_falla_si_uno_de_sus_seis_miembros_deja_la_frontera(self) -> None:
        import dataclasses

        import matplotlib.pyplot as plt

        if not (RAIZ / "results" / "raw" / "pareto" / "pareto_analysis.json").is_file():
            pytest.skip("no hay volcado del paso 26")
        por_motor, fronteras, objetivos, suelos, identidades = _entradas_resultado_principal()
        estricta, robusta = fronteras["onnxruntime"]
        sin_l1 = tuple(
            p for p in robusta.nondominated if identidades[p.label]["label"] != "L1 r25"
        )
        assert len(sin_l1) == len(robusta.nondominated) - 1
        fronteras["onnxruntime"] = (estricta, dataclasses.replace(robusta, nondominated=sin_l1))
        try:
            with pytest.raises(ValueError, match="no son miembros de la frontera robusta"):
                fp.figure_headline_frontier(
                    por_motor, fronteras, objetivos, suelos, identidades, "workstation_cpu"
                )
        finally:
            plt.close("all")

    def test_falla_si_la_zona_intermedia_se_queda_vacia(self, monkeypatch) -> None:
        import matplotlib.pyplot as plt

        if not (RAIZ / "results" / "raw" / "pareto" / "pareto_analysis.json").is_file():
            pytest.skip("no hay volcado del paso 26")
        monkeypatch.setattr(fp, "HEADLINE_ZONE_RATIO", "r90")
        try:
            with pytest.raises(ValueError, match="zona intermedia quedaría vacía"):
                fp.figure_headline_frontier(*_entradas_resultado_principal(), "workstation_cpu")
        finally:
            plt.close("all")

    def test_el_manifiesto_versionado_la_trae_y_esta_en_disco(self) -> None:
        destino = RAIZ / "results" / "raw" / "pareto" / "figures_manifest.json"
        if not destino.is_file():
            pytest.skip("no hay manifiesto: ejecuta scripts/27_build_pareto_figures.py")
        manifiesto = json.loads(destino.read_text(encoding="utf-8"))
        assert list(manifiesto.get("intro_figures", {})) == [
            f"{fp.HEADLINE_DESCRIPTION}_workstation_cpu"
        ]
        base = style.figure_filename(1, 2, f"{fp.HEADLINE_DESCRIPTION}_workstation_cpu")
        for formato in style.FIGURE_FORMATS:
            assert (RAIZ / "results" / "figures" / f"{base}.{formato}").is_file()
