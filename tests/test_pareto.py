"""La frontera de Pareto, y sobre todo lo que se niega a afirmar."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_edge.analysis import floors as mf
from tfm_edge.analysis import pareto as mp
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry

RAIZ = Path(__file__).resolve().parent.parent


def _punto(nombre: str, entorno: str = "e1", **valores: float) -> mp.Point:
    """Construye un punto de prueba."""
    return mp.Point(
        experiment_id=nombre,
        runtime="onnxruntime",
        precision="fp32",
        device_id="workstation_cpu",
        environment_id=entorno,
        timestamp="2026-08-27T00:00:00+00:00",
        values=dict(valores),
    )


def _suelos(**ejes: mf.AxisFloor) -> mf.FloorSet:
    """Empaqueta suelos sueltos en un conjunto."""
    return mf.FloorSet(floors=dict(ejes))


def _suelo_relativo(columna: str, valor: float) -> mf.AxisFloor:
    """Suelo relativo igual dentro y fuera de sesión."""
    return mf.AxisFloor(
        axis=columna,
        kind="relativo",
        same_session=valor,
        cross_session=valor,
        provenance="sintético, para test",
    )


LATENCIA = mp.Objective("latencia", "lat", "deployment", "min")
CALIDAD = mp.Objective("calidad", "map", "quality", "max")


class TestSemanticaDelSuelo:
    """Qué certifica y qué no certifica una diferencia."""

    def test_ventaja_por_debajo_del_suelo_es_indistinguible(self) -> None:
        """Ganar por menos de lo que el eje resuelve no es ganar."""
        a = _punto("a", lat=100.0)
        b = _punto("b", lat=105.0)
        suelos = _suelos(lat=_suelo_relativo("lat", 0.10))

        c = mp.compare(a, b, LATENCIA, suelos, misma_sesion=True)

        assert c.verdict == "indistinguible"
        assert c.certain is True
        assert c.threshold == pytest.approx(10.0)

    def test_ventaja_por_encima_del_suelo_se_certifica(self) -> None:
        """Ganar por más de lo que el eje resuelve sí es ganar."""
        a = _punto("a", lat=100.0)
        b = _punto("b", lat=120.0)
        suelos = _suelos(lat=_suelo_relativo("lat", 0.10))

        c = mp.compare(a, b, LATENCIA, suelos, misma_sesion=True)

        assert c.verdict == "mejor"
        assert c.certain is True

    def test_sin_suelo_medido_la_ventaja_no_se_certifica(self) -> None:
        """Sin suelo no se puede decir que una diferencia signifique algo."""
        a = _punto("a", map=0.50)
        b = _punto("b", map=0.40)
        suelos = _suelos()  # ningún suelo declarado

        c = mp.compare(a, b, CALIDAD, suelos, misma_sesion=True)

        assert c.verdict == "mejor"
        assert c.certain is False
        assert c.threshold is None

    def test_empate_exacto_es_indistinguible_aunque_no_haya_suelo(self) -> None:
        """Cero es menor que cualquier suelo, incluido uno desconocido."""
        a = _punto("a", map=0.50)
        b = _punto("b", map=0.50)
        suelos = _suelos()

        c = mp.compare(a, b, CALIDAD, suelos, misma_sesion=True)

        assert c.verdict == "indistinguible"
        assert c.certain is True

    def test_el_suelo_entre_sesiones_es_el_que_se_aplica_a_puntos_de_sesiones_distintas(
        self,
    ) -> None:
        """Dos puntos de entornos distintos usan el suelo grande."""
        a = _punto("a", entorno="e1", lat=100.0)
        b = _punto("b", entorno="e2", lat=112.0)
        suelo = mf.AxisFloor(
            axis="lat",
            kind="relativo",
            same_session=0.05,
            cross_session=0.15,
            provenance="sintético",
        )
        suelos = _suelos(lat=suelo)

        assert mp.same_session(a, b) is False
        c = mp.compare(a, b, LATENCIA, suelos, misma_sesion=False)
        assert c.verdict == "indistinguible"

        # Los mismos números, dentro de una sesión, sí se distinguen.
        dentro = mp.compare(a, b, LATENCIA, suelos, misma_sesion=True)
        assert dentro.verdict == "mejor"


class TestAsimetriaDeLaDominacion:
    """El suelo gobierna la ventaja y nunca la desventaja."""

    def test_una_desventaja_dentro_del_suelo_sigue_bloqueando(self) -> None:
        """Ser peor por poco no autoriza a decir que no se es peor."""
        # 'a' gana en calidad de sobra y pierde en latencia por menos del suelo.
        a = _punto("a", lat=101.0, map=0.90)
        b = _punto("b", lat=100.0, map=0.10)
        suelos = _suelos(
            lat=_suelo_relativo("lat", 0.10),
            map=mf.AxisFloor("map", "absoluto", 0.01, 0.01, "sintético"),
        )

        assert mp.dominates(a, b, (LATENCIA, CALIDAD), suelos, mp.MODE_ROBUST) is None
        assert mp.dominates(a, b, (LATENCIA, CALIDAD), suelos, mp.MODE_STRICT) is None

    def test_una_ventaja_sin_certificar_no_establece_dominacion(self) -> None:
        """Ganar en un eje sin suelo medido no basta para dominar."""
        a = _punto("a", lat=100.0, map=0.90)
        b = _punto("b", lat=100.0, map=0.10)
        suelos = _suelos(lat=_suelo_relativo("lat", 0.10))  # 'map' sin suelo

        assert mp.dominates(a, b, (LATENCIA, CALIDAD), suelos, mp.MODE_ROBUST) is None
        # En crudo sí domina: es exactamente la afirmación que el suelo retira.
        assert mp.dominates(a, b, (LATENCIA, CALIDAD), suelos, mp.MODE_STRICT) is not None

    def test_la_frontera_robusta_contiene_a_la_estricta(self) -> None:
        puntos = [
            _punto("a", lat=100.0, map=0.50),
            _punto("b", lat=101.0, map=0.49),
            _punto("c", lat=140.0, map=0.60),
            _punto("d", lat=200.0, map=0.30),
        ]
        suelos = _suelos(
            lat=_suelo_relativo("lat", 0.10),
            map=mf.AxisFloor("map", "absoluto", 0.005, 0.005, "sintético"),
        )
        objetivos = (LATENCIA, CALIDAD)

        estricta = mp.frontier(puntos, objetivos, suelos, mp.MODE_STRICT)
        robusta = mp.frontier(puntos, objetivos, suelos, mp.MODE_ROBUST)

        etiquetas_e = {p.label for p in estricta.nondominated}
        etiquetas_r = {p.label for p in robusta.nondominated}
        assert etiquetas_e <= etiquetas_r
        assert len(robusta.dominations) <= len(estricta.dominations)

    def test_las_dominaciones_perdidas_son_las_que_el_suelo_retira(self) -> None:
        """La lista de bajas es la diferencia entre las dos fronteras."""
        puntos = [
            _punto("rapido", lat=100.0, map=0.50),
            _punto("casi_igual", lat=101.0, map=0.50),
        ]
        suelos = _suelos(
            lat=_suelo_relativo("lat", 0.10),
            map=mf.AxisFloor("map", "absoluto", 0.005, 0.005, "sintético"),
        )
        objetivos = (LATENCIA, CALIDAD)

        estricta = mp.frontier(puntos, objetivos, suelos, mp.MODE_STRICT)
        robusta = mp.frontier(puntos, objetivos, suelos, mp.MODE_ROBUST)
        perdidas = mp.dominations_lost_to_floor(estricta, robusta)

        assert len(estricta.dominations) == 1
        assert len(robusta.dominations) == 0
        assert len(perdidas) == 1
        assert perdidas[0].dominator.startswith("rapido")
        assert (
            robusta.indistinguishable == (("casi_igual", "rapido"),)
            or len(robusta.indistinguishable) == 1
        )


class TestIndicadoresAgregados:
    """Hipervolumen y distancia al ideal, contra respuestas conocidas."""

    def test_hipervolumen_de_un_punto_optimo_es_uno(self) -> None:
        """Un punto en la esquina buena domina toda la caja."""
        puntos = [_punto("a", lat=10.0, map=1.0)]
        limites = {"lat": (10.0, 20.0), "map": (0.0, 1.0)}

        assert mp.hypervolume(puntos, (LATENCIA, CALIDAD), limites) == pytest.approx(1.0)

    def test_hipervolumen_de_dos_puntos_en_los_ejes(self) -> None:
        # Normalizados: (0, 0.5) y (0.5, 0) en coordenadas de minimización.
        puntos = [
            _punto("a", lat=10.0, map=0.5),
            _punto("b", lat=15.0, map=1.0),
        ]
        limites = {"lat": (10.0, 20.0), "map": (0.0, 1.0)}

        assert mp.hypervolume(puntos, (LATENCIA, CALIDAD), limites) == pytest.approx(0.75)

    def test_distancia_al_ideal_es_cero_en_el_optimo(self) -> None:
        """El ideal es el mejor valor observado en cada eje a la vez."""
        p = _punto("a", lat=10.0, map=1.0)
        limites = {"lat": (10.0, 20.0), "map": (0.0, 1.0)}

        assert mp.distance_to_ideal(p, (LATENCIA, CALIDAD), limites) == pytest.approx(0.0)

    def test_un_eje_sin_rango_no_rompe_la_normalizacion(self) -> None:
        """Si todos los puntos empatan en un eje, ese eje no aporta volumen."""
        limites = {"lat": (10.0, 10.0), "map": (0.0, 1.0)}
        p = _punto("a", lat=10.0, map=1.0)

        assert mp.distance_to_ideal(p, (LATENCIA, CALIDAD), limites) == pytest.approx(0.0)


class TestConstruccionDePuntos:
    """El cruce de calidad y despliegue, que es donde se pierden filas."""

    def _filas(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Filas mínimas de las dos tablas."""
        base = {
            "experiment_id": "x",
            "runtime": "onnxruntime",
            "precision": "fp32",
            "device_id": "workstation_cpu",
        }
        calidad = [{**base, "map_50_95": "0.30"}]
        despliegue = [
            {
                **base,
                "latency_p95_ms": "100.0",
                "peak_host_memory_mb": "400.0",
                "environment_id": "e1",
                "timestamp": "2026-08-27T00:00:00+00:00",
            }
        ]
        return calidad, despliegue

    def test_una_fila_de_despliegue_sin_calidad_se_descarta(self) -> None:
        calidad, despliegue = self._filas()
        puntos = mp.build_points([], despliegue, mp.default_objectives())
        assert puntos == ()

    def test_una_celda_vacia_descarta_el_punto_y_no_vale_cero(self) -> None:
        """Ausente es ausente."""
        calidad, despliegue = self._filas()
        despliegue[0]["peak_host_memory_mb"] = ""

        puntos = mp.build_points(calidad, despliegue, mp.default_objectives())

        assert puntos == ()

    def test_dos_filas_de_calidad_para_la_misma_configuracion_es_un_error(self) -> None:
        calidad, despliegue = self._filas()
        calidad.append(dict(calidad[0]))

        with pytest.raises(mp.ParetoError, match="filas de calidad"):
            mp.build_points(calidad, despliegue, mp.default_objectives())

    def test_el_cruce_bueno_produce_un_punto_completo(self) -> None:
        """El caso normal, para que los anteriores signifiquen algo."""
        calidad, despliegue = self._filas()

        puntos = mp.build_points(calidad, despliegue, mp.default_objectives())

        assert len(puntos) == 1
        assert puntos[0].values["map_50_95"] == pytest.approx(0.30)
        assert puntos[0].environment_id == "e1"


class TestMedicionDelSuelo:
    """Los pares de control, y las guardas que impiden inventarse un suelo."""

    def test_un_par_sin_su_evidencia_en_disco_falla(self, tmp_path: Path) -> None:
        """Sin la evidencia, que las dos tomas sean comparables es una suposición."""
        par = mf.ControlPair(
            name="inventado",
            kind="entre_procesos",
            rationale="da igual",
            evidence=("docs/evidencia/no_existe.txt",),
            members=(
                mf.PairMember("uno", experiment_id="a", runtime="onnxruntime"),
                mf.PairMember("dos", experiment_id="b", runtime="onnxruntime"),
            ),
        )

        with pytest.raises(mf.FloorMeasurementError, match="no_existe"):
            mf.measure_floors([], ["latency_p50_ms"], tmp_path, pairs=[par])

    def test_una_toma_debe_identificarse_de_una_forma_y_solo_una(self) -> None:
        """O registro o volcado; las dos cosas a la vez es ambigüedad."""
        with pytest.raises(mf.FloorMeasurementError):
            mf.PairMember("ambigua", experiment_id="a", runtime="r", dump="x.json")
        with pytest.raises(mf.FloorMeasurementError):
            mf.PairMember("sin nada")

    def test_la_diferencia_relativa_exige_magnitudes_positivas(self) -> None:
        """Una diferencia relativa contra cero no significa nada."""
        with pytest.raises(mf.FloorMeasurementError):
            mf.relative_difference(0.0, 10.0)

    def test_la_diferencia_relativa_no_depende_del_orden(self) -> None:
        """Se divide por la menor, que es la cota alta y es simétrica."""
        assert mf.relative_difference(100.0, 110.0) == pytest.approx(
            mf.relative_difference(110.0, 100.0)
        )
        assert mf.relative_difference(100.0, 110.0) == pytest.approx(0.10)

    def test_map_50_se_declara_sin_suelo_en_vez_de_heredar_el_de_map_50_95(
        self, tmp_path: Path
    ) -> None:
        suelos = mf.measure_floors([], ["map_50", "map_50_95"], tmp_path, pairs=[])

        assert suelos.floors["map_50"].is_measured() is False
        assert suelos.floors["map_50_95"].is_measured() is True
        assert suelos.floors["map_50_95"].same_session == pytest.approx(
            mf.EVALUATOR_READABILITY_FLOOR
        )


class TestContraElRegistroReal:
    """Comprobaciones sobre el registro y la evidencia de la campaña."""

    @pytest.fixture(scope="class")
    def contexto(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Filas utilizables del registro real."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        return calidad, despliegue

    def test_el_suelo_de_p95_es_peor_que_el_de_p50(self, contexto) -> None:
        """Un percentil de cola resuelve peor que una mediana."""
        _, despliegue = contexto
        suelos = mf.measure_floors(despliegue, ["latency_p50_ms", "latency_p95_ms"], RAIZ)

        p50 = suelos.floors["latency_p50_ms"]
        p95 = suelos.floors["latency_p95_ms"]

        assert p95.same_session > p50.same_session
        assert p50.same_session == pytest.approx(0.0926, abs=1e-4)
        assert p95.same_session == pytest.approx(0.1472, abs=1e-4)

    def test_el_par_contaminado_se_observa_pero_no_fija_suelo(self, contexto) -> None:
        """La ventana transitoria del 31/08 se deja anotada, y no acota nada."""
        _, despliegue = contexto
        suelos = mf.measure_floors(despliegue, ["latency_p50_ms"], RAIZ)

        contaminados = sorted(suelos.skipped_pairs)
        assert contaminados == [
            "X_m_baseline_default_dfe22158_pytorch_fp32",
            "X_n_baseline_default_fab9d45c_onnxruntime_fp32",
            "X_n_baseline_default_fab9d45c_pytorch_fp32",
            "X_n_ptq_int8-noexcl_a78f566a_openvino_int8",
        ]

        # Los cuatro son peores que el suelo; si se usaran, el suelo de p50 pasaría del 12,7 % al 69
        # %.
        suelo = suelos.floors["latency_p50_ms"].cross_session
        for nombre in contaminados:
            assert suelos.pair_observations[nombre]["latency_p50_ms"] > suelo

    def test_el_tipo_declarado_de_cada_par_se_comprueba_contra_las_fechas(self, contexto) -> None:
        """Un par «entre procesos» cuyas tomas distan dieciséis días ya ha pasado."""
        _, despliegue = contexto
        mf.measure_floors(despliegue, ["latency_p50_ms"], RAIZ)  # no lanza

        mentiroso = mf.ControlPair(
            name="dice_entre_procesos_y_no_lo_es",
            kind="entre_procesos",
            rationale="da igual",
            evidence=(),
            members=(
                mf.PairMember(
                    label="archivada",
                    dump=mf._ARCHIVE_DIR
                    + "/"
                    + mf.dump_filename("s_baseline_default_745e3d08", "onnxruntime", "fp32"),
                ),
                mf.PairMember(
                    label="viva",
                    dump=mf._LIVE_DIR
                    + "/"
                    + mf.dump_filename("s_baseline_default_745e3d08", "onnxruntime", "fp32"),
                ),
            ),
        )

        with pytest.raises(mf.FloorMeasurementError, match="entre procesos"):
            mf.measure_floors(despliegue, ["latency_p50_ms"], RAIZ, pairs=[mentiroso])

    def test_la_lista_declarada_de_remedidas_coincide_con_el_disco(self) -> None:
        """Un par que aparece solo porque alguien dejó un fichero no es un control."""
        assert mf.verify_declared_pairs_match_disk(RAIZ) == ()

    def test_los_pares_entre_procesos_no_dependen_de_ninguna_fila_del_registro(self) -> None:
        for par in mf.default_control_pairs():
            for toma in par.members:
                assert toma.dump is not None, (
                    f"La toma '{toma.label}' del par '{par.name}' se identifica "
                    f"por una fila del registro, que es reescribible."
                )

    def test_ningun_par_lee_del_directorio_vivo_de_volcados(self) -> None:
        for par in mf.default_control_pairs():
            for ruta in (*par.evidence, *(toma.dump or "" for toma in par.members)):
                assert not ruta.startswith(mf._LIVE_DIR), (
                    f"El par '{par.name}' lee {ruta}, del directorio vivo, que la "
                    f"próxima remedida pisará."
                )

    def test_la_dominacion_de_bnscale_por_el_azar_no_sobrevive_al_suelo(self, contexto) -> None:
        calidad, despliegue = contexto
        objetivos = (
            mp.Objective("mAP@50", "map_50", "quality", "max"),
            mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
        )
        suelos = mf.measure_floors(despliegue, [o.column for o in objetivos], RAIZ)
        puntos = mp.build_points(calidad, despliegue, objetivos)

        estricta = mp.frontier(puntos, objetivos, suelos, mp.MODE_STRICT)
        robusta = mp.frontier(puntos, objetivos, suelos, mp.MODE_ROBUST)
        perdidas = mp.dominations_lost_to_floor(estricta, robusta)

        pares = {(d.dominator, d.dominated) for d in perdidas}
        esperado = (
            "s_combined_prune-l1global-r50-randomctl-ptqint8_13a9774e " "[openvino/int8]",
            "s_combined_prune-bnscale-r50-ptqint8_7712cb36 [openvino/int8]",
        )
        assert esperado in pares

    def test_la_frontera_robusta_contiene_a_la_estricta_sobre_datos_reales(self, contexto) -> None:
        """El invariante, sobre las 31 medidas de verdad y en las tres vistas."""
        calidad, despliegue = contexto
        vistas = [
            mp.default_objectives(),
            (
                mp.Objective("mAP@50", "map_50", "quality", "max"),
                mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min"),
            ),
        ]
        for objetivos in vistas:
            suelos = mf.measure_floors(despliegue, [o.column for o in objetivos], RAIZ)
            puntos = mp.build_points(calidad, despliegue, objetivos)
            estricta = mp.frontier(puntos, objetivos, suelos, mp.MODE_STRICT)
            robusta = mp.frontier(puntos, objetivos, suelos, mp.MODE_ROBUST)

            assert {p.label for p in estricta.nondominated} <= {
                p.label for p in robusta.nondominated
            }

    def test_el_analisis_completo_es_serializable_y_estable(self, contexto) -> None:
        """El volcado tiene que poder escribirse, y dos cálculos dar lo mismo."""
        calidad, despliegue = contexto
        objetivos = mp.default_objectives()
        suelos = mf.measure_floors(despliegue, [o.column for o in objetivos], RAIZ)
        puntos = mp.build_points(calidad, despliegue, objetivos)

        uno = mp.analyze(puntos, objetivos, suelos, [("device_id", "runtime")])
        otro = mp.analyze(puntos, objetivos, suelos, [("device_id", "runtime")])

        assert json.dumps(uno, sort_keys=True) == json.dumps(otro, sort_keys=True)
