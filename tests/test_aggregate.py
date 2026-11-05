"""Tests del informe de cobertura y de los huecos que no debe señalar."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm_edge.analysis import aggregate as ag
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry

RAIZ = Path(__file__).resolve().parent.parent


def _experimento(experiment_id: str, technique: str, **extra: str) -> dict[str, str]:
    """Construye una fila de experimento de prueba."""
    fila = {
        "experiment_id": experiment_id,
        "technique": technique,
        "parent_experiment_id": "",
        "status": "ok",
    }
    fila.update(extra)
    return fila


def _calidad(
    experiment_id: str,
    runtime: str = "onnxruntime",
    precision: str = "fp32",
    evaluator: str = "pycocotools",
    device_id: str = "workstation_cpu",
    **extra: str,
) -> dict[str, str]:
    """Construye una fila de calidad de prueba."""
    fila = {
        "experiment_id": experiment_id,
        "runtime": runtime,
        "precision": precision,
        "evaluator": evaluator,
        "device_id": device_id,
        "map_50": "0.5",
        "map_50_95": "0.3",
    }
    fila.update(extra)
    return fila


def _despliegue(
    experiment_id: str,
    runtime: str = "onnxruntime",
    precision: str = "fp32",
    device_id: str = "workstation_cpu",
) -> dict[str, str]:
    """Construye una fila de despliegue de prueba."""
    return {
        "experiment_id": experiment_id,
        "runtime": runtime,
        "precision": precision,
        "device_id": device_id,
    }


class TestLaReglaDeExpectativa:
    """De dónde sale lo que se espera, y hasta dónde llega."""

    def test_lo_que_alcanza_un_miembro_se_espera_de_toda_la_familia(self) -> None:
        """Un motor alcanzado por uno se exige a sus hermanos de familia."""
        experimentos = [
            _experimento("s_pruning_a_11111111", "pruning"),
            _experimento("s_pruning_b_22222222", "pruning"),
        ]
        calidad = [_calidad("s_pruning_a_11111111"), _calidad("s_pruning_b_22222222")]
        despliegue = [_despliegue("s_pruning_a_11111111")]

        informe = ag.coverage_report(experimentos, calidad, despliegue)
        huecos = informe.gaps_by_experiment()

        assert "s_pruning_b_22222222" in huecos
        assert [c.table for c in huecos["s_pruning_b_22222222"]] == ["deployment"]

    def test_no_se_inventa_un_entorno_que_nadie_alcanzo(self) -> None:
        """Sin ninguna medida en un motor, ese motor no genera huecos."""
        experimentos = [_experimento("s_ptq_x_11111111", "ptq")]
        calidad = [_calidad("s_ptq_x_11111111", runtime="openvino", precision="int8")]
        despliegue = [_despliegue("s_ptq_x_11111111", runtime="openvino", precision="int8")]

        informe = ag.coverage_report(experimentos, calidad, despliegue)

        motores = {c.context.as_dict()["runtime"] for c in informe.cells}
        assert motores == {"openvino"}
        assert informe.gaps == ()

    def test_una_familia_no_arrastra_a_otra(self) -> None:
        """Lo que alcanza la poda no se le exige a la cuantización."""
        experimentos = [
            _experimento("s_pruning_a_11111111", "pruning"),
            _experimento("s_ptq_x_22222222", "ptq"),
        ]
        calidad = [
            _calidad("s_pruning_a_11111111"),
            _calidad("s_ptq_x_22222222", precision="int8"),
        ]

        informe = ag.coverage_report(experimentos, calidad, [])

        assert informe.gaps == ()

    def test_la_regla_va_escrita_en_el_informe(self) -> None:
        """El informe dice respecto a qué mide la cobertura y qué no ve."""
        informe = ag.coverage_report(
            [_experimento("s_pruning_a_11111111", "pruning")],
            [_calidad("s_pruning_a_11111111")],
            [],
        )

        # Un informe de cobertura que no declara su propio límite hace creer que
        # ha comprobado lo que no ha mirado.
        assert "familia" in informe.rule
        assert "no genera ningún hueco" in informe.rule


class TestLoQueElInformeNoDebeInventarse:
    """Los huecos falsos entierran a los verdaderos."""

    def test_el_dispositivo_no_genera_huecos_de_calidad(self) -> None:
        """Evaluar un modelo en otra máquina no obliga a evaluarlos todos allí."""
        experimentos = [
            _experimento("n_baseline_default_11111111", "baseline"),
            _experimento("s_baseline_default_22222222", "baseline"),
        ]
        calidad = [
            _calidad("n_baseline_default_11111111", runtime="pytorch"),
            _calidad("n_baseline_default_11111111", runtime="pytorch", device_id="colab_t4_gpu"),
            _calidad("s_baseline_default_22222222", runtime="pytorch"),
        ]

        informe = ag.coverage_report(experimentos, calidad, [])

        assert informe.gaps == ()

    def test_el_dispositivo_si_genera_huecos_de_despliegue(self) -> None:
        """La latencia sí es una propiedad de la máquina, y ahí el hueco es real."""
        experimentos = [
            _experimento("s_a_11111111", "pruning"),
            _experimento("s_b_22222222", "pruning"),
        ]
        despliegue = [
            _despliegue("s_a_11111111", device_id="workstation_cpu"),
            _despliegue("s_a_11111111", device_id="raspberry_pi5"),
            _despliegue("s_b_22222222", device_id="workstation_cpu"),
        ]

        informe = ag.coverage_report(experimentos, [], despliegue)
        huecos = informe.gaps_by_experiment()

        assert list(huecos) == ["s_b_22222222"]
        assert huecos["s_b_22222222"][0].context.as_dict()["device_id"] == "raspberry_pi5"

    def test_dos_evaluadores_no_son_un_duplicado(self) -> None:
        experimentos = [_experimento("s_baseline_default_11111111", "baseline")]
        calidad = [
            _calidad("s_baseline_default_11111111", runtime="pytorch"),
            _calidad(
                "s_baseline_default_11111111",
                runtime="pytorch",
                evaluator="ultralytics",
                device_id="auto",
            ),
        ]

        informe = ag.coverage_report(experimentos, calidad, [])

        assert informe.inconsistencies == ()
        assert len(informe.cells) == 2


class TestLaPremisaSeMideNoSeSupone:
    """El mAP no depende de la máquina."""

    def test_dos_dispositivos_con_el_mismo_numero_dan_dispersion_cero(self) -> None:
        """La premisa se sostiene y el informe lo dice con un número."""
        calidad = [
            _calidad("n_baseline_default_11111111", runtime="pytorch", map_50="0.524158"),
            _calidad(
                "n_baseline_default_11111111",
                runtime="pytorch",
                device_id="colab_t4_gpu",
                map_50="0.524158",
            ),
        ]

        observaciones = ag.device_independence(calidad)

        assert len(observaciones) == 1
        assert observaciones[0].spread["map_50"] == 0.0
        assert observaciones[0].devices == ("colab_t4_gpu", "workstation_cpu")

    def test_si_la_premisa_se_rompe_el_informe_lo_dice(self) -> None:
        """Una diferencia entre máquinas invalida el eje, y hay que enterarse."""
        experimentos = [_experimento("n_baseline_default_11111111", "baseline")]
        calidad = [
            _calidad("n_baseline_default_11111111", runtime="pytorch", map_50="0.52"),
            _calidad(
                "n_baseline_default_11111111",
                runtime="pytorch",
                device_id="colab_t4_gpu",
                map_50="0.61",
            ),
        ]

        informe = ag.coverage_report(experimentos, calidad, [])

        assert informe.device_dependence_detected
        assert informe.to_dict()["device_independence"]["dependence_detected"] is True

    def test_sin_configuraciones_repetidas_no_hay_comprobacion(self) -> None:
        """Que no haya observaciones no es que la premisa se cumpla."""
        observaciones = ag.device_independence([_calidad("s_a_11111111")])

        assert observaciones == ()


class TestIncoherenciasDelRegistro:
    """Lo que no es un hueco de campaña sino un registro mal escrito."""

    def test_metrica_de_un_experimento_que_no_existe(self) -> None:
        """Una fila huérfana no puede decir de qué configuración es."""
        informe = ag.coverage_report([], [_calidad("s_fantasma_x_11111111")], [])

        tipos = {i.kind for i in informe.inconsistencies}
        assert "metrica_sin_experimento" in tipos

    def test_fila_repetida_en_el_mismo_entorno(self) -> None:
        """Dos filas idénticas dejan la configuración sin punto en la frontera."""
        experimentos = [_experimento("s_a_11111111", "pruning")]
        calidad = [_calidad("s_a_11111111"), _calidad("s_a_11111111")]

        informe = ag.coverage_report(experimentos, calidad, [])

        tipos = {i.kind for i in informe.inconsistencies}
        assert "fila_duplicada" in tipos

    def test_experimento_sin_ninguna_metrica(self) -> None:
        """Un experimento registrado y nunca evaluado se señala."""
        informe = ag.coverage_report([_experimento("s_a_11111111", "pruning")], [], [])

        tipos = {i.kind for i in informe.inconsistencies}
        assert "experimento_sin_metricas" in tipos

    def test_una_fila_sin_entorno_es_un_error_no_una_cobertura(self) -> None:
        """Sin motor no se sabe dónde se midió, y no vale darla por buena."""
        fila = _calidad("s_a_11111111")
        fila["runtime"] = ""

        with pytest.raises(ag.AggregateError, match="no declara"):
            ag.coverage_report([_experimento("s_a_11111111", "pruning")], [fila], [])


class TestSobreElRegistroReal:
    """Comprobaciones sobre el registro de la campaña."""

    @pytest.fixture(scope="class")
    def informe(self) -> ag.CoverageReport:
        """Informe de cobertura del registro del proyecto."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        return ag.coverage_report(experimentos, calidad, despliegue)

    def test_la_premisa_del_dispositivo_se_sostiene_en_esta_campana(
        self, informe: ag.CoverageReport
    ) -> None:
        """Hay al menos una configuración medida en dos máquinas, y coinciden."""
        assert informe.device_spreads, (
            "Sin ninguna configuración evaluada en dos dispositivos, la premisa sobre "
            "la que se agrupa la calidad no se está contrastando con nada."
        )
        assert not informe.device_dependence_detected

    def test_los_baselines_grandes_ya_estan_desplegados(self, informe: ag.CoverageReport) -> None:
        """Era el hueco más caro de la campaña, y está cerrado."""
        huecos = informe.gaps_by_experiment()
        for tamano in ("n", "m", "l"):
            con_hueco = [
                e
                for e in huecos
                if e.startswith(f"{tamano}_baseline")
                and any(c.table == "deployment" for c in huecos[e])
            ]
            assert not con_hueco, (
                f"El baseline '{tamano}' ha vuelto a quedarse sin medida de "
                f"despliegue: {con_hueco}."
            )

    def test_el_registro_no_tiene_incoherencias(self, informe: ag.CoverageReport) -> None:
        """Filas huérfanas o duplicadas romperían el análisis aguas abajo."""
        assert informe.inconsistencies == (), [i.detail for i in informe.inconsistencies]
