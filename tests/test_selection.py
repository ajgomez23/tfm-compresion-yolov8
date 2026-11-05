"""La selección por restricciones, y los dos silencios que evita."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tfm_edge.analysis import floors as fl
from tfm_edge.analysis import pareto as pa
from tfm_edge.analysis import selection as se
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry

RAIZ = Path(__file__).resolve().parent.parent


def _punto(nombre: str, **valores: float) -> pa.Point:
    """Punto de prueba en el dispositivo de trabajo."""
    return pa.Point(
        experiment_id=nombre,
        runtime="onnxruntime",
        precision="fp32",
        device_id="workstation_cpu",
        environment_id="e1",
        timestamp="2026-08-27T00:00:00+00:00",
        values=dict(valores),
    )


def _suelos(**ejes: float) -> fl.FloorSet:
    """Conjunto de suelos relativos."""
    return fl.FloorSet(
        floors={
            eje: fl.AxisFloor(
                axis=eje,
                kind="relativo",
                same_session=valor,
                cross_session=valor,
                provenance="par de control de prueba",
            )
            for eje, valor in ejes.items()
        }
    )


class TestElBordeDeUnaRestriccion:
    """Una restricción tiene tres estados."""

    def test_lo_que_se_queda_a_un_pelo_se_marca_y_se_devuelve(self) -> None:
        """Descartarlo afirmaría una diferencia que la medida no ve."""
        punto = _punto("s_pruning_a_11111111", latency_p95_ms=100.7)
        restriccion = se.Constraint("latencia", "latency_p95_ms", 100.0, "max")

        comprobacion = se.check_constraint(punto, restriccion, _suelos(latency_p95_ms=0.147))

        assert comprobacion.verdict == se.VERDICT_BORDERLINE

    def test_lo_que_esta_claramente_fuera_se_descarta(self) -> None:
        punto = _punto("s_pruning_a_11111111", latency_p95_ms=291.5)
        restriccion = se.Constraint("latencia", "latency_p95_ms", 100.0, "max")

        comprobacion = se.check_constraint(punto, restriccion, _suelos(latency_p95_ms=0.147))

        assert comprobacion.verdict == se.VERDICT_FAILS

    def test_lo_que_cumple_con_holgura_cumple(self) -> None:
        """Y el tercer estado, para que los tres estén ejercitados."""
        punto = _punto("s_pruning_a_11111111", latency_p95_ms=41.3)
        restriccion = se.Constraint("latencia", "latency_p95_ms", 100.0, "max")

        comprobacion = se.check_constraint(punto, restriccion, _suelos(latency_p95_ms=0.147))

        assert comprobacion.verdict == se.VERDICT_MEETS

    def test_sin_suelo_medido_el_limite_es_una_linea_exacta(self) -> None:
        """Y la comprobación lo dice, en vez de aplicar un margen inventado."""
        punto = _punto("s_pruning_a_11111111", latency_p95_ms=100.7)
        restriccion = se.Constraint("latencia", "latency_p95_ms", 100.0, "max")

        comprobacion = se.check_constraint(punto, restriccion, _suelos())

        assert comprobacion.verdict == se.VERDICT_FAILS
        assert "línea exacta" in comprobacion.explanation

    def test_un_minimo_se_evalua_al_reves(self) -> None:
        punto = _punto("s_pruning_a_11111111", map_50_95=0.25)
        restriccion = se.Constraint("mAP mínimo", "map_50_95", 0.30, "min")

        comprobacion = se.check_constraint(punto, restriccion, _suelos())

        assert comprobacion.verdict == se.VERDICT_FAILS

    def test_sin_valor_medido_no_se_da_por_cumplida(self) -> None:
        """Suponerlo colaría la configuración en una lista de viables."""
        punto = _punto("s_pruning_a_11111111", latency_p95_ms=50.0)
        restriccion = se.Constraint("memoria", "peak_host_memory_mb", 500.0, "max")

        comprobacion = se.check_constraint(punto, restriccion, _suelos())

        assert comprobacion.verdict == se.VERDICT_FAILS
        assert comprobacion.value is None

    def test_una_direccion_desconocida_es_un_error(self) -> None:
        """Una restricción que no dice si es techo o suelo no se puede evaluar."""
        with pytest.raises(se.SelectionError, match="dirección"):
            se.Constraint("rara", "latency_p95_ms", 100.0, "por_ahi")


class TestElOrdenTieneEmpates:
    """Coronar a un ganador dentro del suelo es afirmar de más."""

    def test_los_indistinguibles_comparten_rango(self) -> None:
        """71,0 y 75,0 con un suelo del 14,7 % son la misma latencia."""
        puntos = [
            _punto("s_a_11111111", latency_p95_ms=71.0),
            _punto("s_b_22222222", latency_p95_ms=75.0),
            _punto("s_c_33333333", latency_p95_ms=291.5),
        ]

        resultado = se.select(
            puntos, (), _suelos(latency_p95_ms=0.147), "workstation_cpu", "latency_p95_ms"
        )
        rangos = {c.point.experiment_id: c.rank for c in resultado.candidates}

        assert rangos["s_a_11111111"] == rangos["s_b_22222222"] == 1
        assert rangos["s_c_33333333"] == 2

    def test_sin_suelo_no_hay_empates(self) -> None:
        puntos = [
            _punto("s_a_11111111", latency_p95_ms=71.0),
            _punto("s_b_22222222", latency_p95_ms=75.0),
        ]

        resultado = se.select(puntos, (), _suelos(), "workstation_cpu", "latency_p95_ms")
        rangos = [c.rank for c in resultado.candidates]

        assert rangos == [1, 2]

    def test_el_agrupamiento_no_encadena(self) -> None:
        """Tres valores en escalera no caen todos en el mismo rango."""
        puntos = [
            _punto("s_a_11111111", latency_p95_ms=100.0),
            _punto("s_b_22222222", latency_p95_ms=109.0),
            _punto("s_c_33333333", latency_p95_ms=119.0),
        ]

        resultado = se.select(
            puntos, (), _suelos(latency_p95_ms=0.10), "workstation_cpu", "latency_p95_ms"
        )
        rangos = {c.point.experiment_id: c.rank for c in resultado.candidates}

        assert rangos["s_a_11111111"] == rangos["s_b_22222222"] == 1
        assert rangos["s_c_33333333"] == 2

    def test_ordenar_por_un_eje_sin_medir_es_un_error(self) -> None:
        """Ponerlas al final sería inventarse su posición."""
        puntos = [_punto("s_a_11111111", latency_p95_ms=71.0)]

        with pytest.raises(se.SelectionError, match="No se puede ordenar"):
            se.select(puntos, (), _suelos(), "workstation_cpu", "map_50_95")


def _suelos_de_la_campana() -> fl.FloorSet:
    """El suelo de p95 con sus dos valores distintos, como en la campaña."""
    return fl.FloorSet(
        floors={
            "latency_p95_ms": fl.AxisFloor(
                axis="latency_p95_ms",
                kind="relativo",
                same_session=0.1472,
                cross_session=0.4729,
                provenance="valores de la campaña, para la prueba",
            )
        }
    )


class TestElSueloDeCadaComparacionLoDecideLaSesion:
    """Los rangos comparan dos medidas; el límite, una medida y un criterio."""

    def test_en_la_misma_sesion_un_20_por_ciento_separa_rangos(self) -> None:
        """130,9 frente a 155,2 supera el 14,7 %, que es el suelo que toca."""
        puntos = [
            _punto("s_a_11111111", latency_p95_ms=130.9),
            _punto("s_b_22222222", latency_p95_ms=155.15),
        ]

        resultado = se.select(
            puntos, (), _suelos_de_la_campana(), "workstation_cpu", "latency_p95_ms"
        )

        assert [c.rank for c in resultado.candidates] == [1, 2]

    def test_entre_sesiones_los_mismos_valores_empatan(self) -> None:
        otro = dataclasses.replace(
            _punto("s_b_22222222", latency_p95_ms=155.15), environment_id="e2"
        )
        puntos = [_punto("s_a_11111111", latency_p95_ms=130.9), otro]

        resultado = se.select(
            puntos, (), _suelos_de_la_campana(), "workstation_cpu", "latency_p95_ms"
        )

        assert [c.rank for c in resultado.candidates] == [1, 1]

    def test_el_limite_usa_el_suelo_entre_sesiones_a_proposito(self) -> None:
        punto = _punto("s_pruning_a_11111111", latency_p95_ms=130.0)
        restriccion = se.Constraint("latencia", "latency_p95_ms", 100.0, "max")

        comprobacion = se.check_constraint(punto, restriccion, _suelos_de_la_campana())

        assert comprobacion.verdict == se.VERDICT_BORDERLINE


class TestLaListaCompleta:
    """Lo que se devuelve y lo que se aparta."""

    def test_las_descartadas_van_aparte_no_desaparecen(self) -> None:
        """Quien consulta tiene derecho a saber qué se quedó fuera y por qué."""
        puntos = [
            _punto("s_a_11111111", latency_p95_ms=41.0, map_50_95=0.35),
            _punto("s_b_22222222", latency_p95_ms=291.5, map_50_95=0.38),
        ]
        restricciones = (se.Constraint("latencia", "latency_p95_ms", 100.0, "max"),)

        resultado = se.select(
            puntos,
            restricciones,
            _suelos(latency_p95_ms=0.147),
            "workstation_cpu",
            "latency_p95_ms",
        )

        assert [c.point.experiment_id for c in resultado.candidates] == ["s_a_11111111"]
        assert [c.point.experiment_id for c in resultado.excluded] == ["s_b_22222222"]
        assert resultado.excluded[0].checks[0].verdict == se.VERDICT_FAILS

    def test_sin_puntos_del_dispositivo_es_un_error(self) -> None:
        """Una lista vacía se confundiría con «ninguna cumple», que es otra cosa."""
        puntos = [_punto("s_a_11111111", latency_p95_ms=41.0)]

        with pytest.raises(se.SelectionError, match="No hay ningún punto medido"):
            se.select(puntos, (), _suelos(), "raspberry_pi5", "latency_p95_ms")

    def test_el_resultado_declara_los_ejes_sin_suelo(self) -> None:
        """Sobre esos ejes el borde no está protegido, y hay que decirlo."""
        puntos = [_punto("s_a_11111111", latency_p95_ms=41.0, map_50=0.5)]
        restricciones = (se.Constraint("mAP mínimo", "map_50", 0.3, "min"),)

        resultado = se.select(
            puntos,
            restricciones,
            _suelos(latency_p95_ms=0.147),
            "workstation_cpu",
            "latency_p95_ms",
        )

        assert any("map_50" in nota for nota in resultado.notes)


class TestSobreElRegistroReal:
    """Comprobaciones sobre el registro de la campaña."""

    @pytest.fixture(scope="class")
    def resultado(self) -> se.SelectionResult:
        """Una consulta de selección sobre el registro del proyecto."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        objetivos = (
            pa.Objective("mAP@50-95", "map_50_95", "quality", "max"),
            pa.Objective("latencia p95", "latency_p95_ms", "deployment", "min"),
            pa.Objective("memoria pico", "peak_host_memory_mb", "deployment", "min"),
        )
        puntos = pa.build_points(calidad, despliegue, objetivos)
        suelos = fl.measure_floors(
            despliegue, ("latency_p95_ms", "peak_host_memory_mb", "map_50_95"), RAIZ
        )
        restricciones = (
            se.Constraint("latencia máxima", "latency_p95_ms", 100.0, "max"),
            se.Constraint("mAP mínimo", "map_50_95", 0.30, "min"),
        )
        return se.select(puntos, restricciones, suelos, "workstation_cpu", "latency_p95_ms")

    def test_la_consulta_devuelve_configuraciones_y_descartes(
        self, resultado: se.SelectionResult
    ) -> None:
        assert resultado.candidates
        assert resultado.excluded

    def test_hay_configuraciones_en_el_margen(self, resultado: se.SelectionResult) -> None:
        """Si no las hubiera, el tratamiento del borde no se estaría ejercitando."""
        assert any(c.borderline for c in resultado.candidates)

    def test_hay_empates_en_el_orden(self, resultado: se.SelectionResult) -> None:
        """Con el suelo del p95 al 14,7 %, coronar un ganador sería afirmar de más."""
        rangos = [c.rank for c in resultado.candidates]
        assert len(set(rangos)) < len(rangos)

    def test_el_baseline_no_cumple_la_latencia(self, resultado: se.SelectionResult) -> None:
        descartados = {c.point.experiment_id for c in resultado.excluded}
        assert any(e.startswith("s_baseline_default") for e in descartados)
