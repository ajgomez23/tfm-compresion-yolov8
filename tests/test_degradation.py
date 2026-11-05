"""La degradación, y las tres cosas que se niega a calcular."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm_edge.analysis import degradation as dg
from tfm_edge.analysis import floors as fl
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry

RAIZ = Path(__file__).resolve().parent.parent


def _suelo_relativo(eje: str, valor: float) -> fl.AxisFloor:
    """Suelo relativo medido para un eje."""
    return fl.AxisFloor(
        axis=eje,
        kind="relativo",
        same_session=valor,
        cross_session=valor,
        provenance="par de control de prueba",
    )


def _conjunto(**ejes: fl.AxisFloor) -> fl.FloorSet:
    """Empaqueta suelos sueltos."""
    return fl.FloorSet(floors=dict(ejes))


def _experimento(experiment_id: str, padre: str = "", **extra: str) -> dict[str, str]:
    """Fila de experimento de prueba."""
    fila = {
        "experiment_id": experiment_id,
        "parent_experiment_id": padre,
        "technique": "pruning",
        "status": "ok",
    }
    fila.update(extra)
    return fila


def _calidad(experiment_id: str, runtime: str, **valores: str) -> dict[str, str]:
    """Fila de calidad de prueba."""
    fila = {
        "experiment_id": experiment_id,
        "runtime": runtime,
        "precision": "fp32",
        "device_id": "workstation_cpu",
        "evaluator": "pycocotools",
        "map_50": "0.5",
        "map_50_95": "0.3",
    }
    fila.update(valores)
    return fila


class TestLaCadenaDeReferencia:
    """Contra qué se degrada una variante."""

    def test_sube_hasta_la_raiz_no_hasta_el_padre(self) -> None:
        experimentos = {
            "s_baseline_default_11111111": _experimento("s_baseline_default_11111111"),
            "s_pruning_a_22222222": _experimento(
                "s_pruning_a_22222222", "s_baseline_default_11111111"
            ),
            "s_combined_b_33333333": _experimento("s_combined_b_33333333", "s_pruning_a_22222222"),
        }

        raiz, cadena = dg.resolve_reference("s_combined_b_33333333", experimentos)

        assert raiz == "s_baseline_default_11111111"
        assert cadena == (
            "s_combined_b_33333333",
            "s_pruning_a_22222222",
            "s_baseline_default_11111111",
        )

    def test_una_cadena_rota_es_un_error(self) -> None:
        """Sin la fila del padre no se sabe respecto a qué se degrada."""
        experimentos = {
            "s_pruning_a_22222222": _experimento("s_pruning_a_22222222", "s_fantasma_x_9")
        }

        with pytest.raises(dg.DegradationError, match="no está en experiments.csv"):
            dg.resolve_reference("s_pruning_a_22222222", experimentos)

    def test_una_cadena_ciclica_es_un_error(self) -> None:
        """Un ciclo colgaría el recorrido en vez de dar una referencia."""
        experimentos = {
            "a_baseline_x_11111111": _experimento("a_baseline_x_11111111", "b_baseline_y_2"),
            "b_baseline_y_2": _experimento("b_baseline_y_2", "a_baseline_x_11111111"),
        }

        with pytest.raises(dg.DegradationError, match="cíclica"):
            dg.resolve_reference("a_baseline_x_11111111", experimentos)


class TestNoSeCruzaDeMotor:
    """Comparar entre motores mide dos cambios y los atribuye a uno."""

    def test_sin_referencia_en_el_mismo_motor_no_se_compara(self) -> None:
        experimentos = [
            _experimento("s_baseline_default_11111111", technique="baseline"),
            _experimento("s_pruning_a_22222222", "s_baseline_default_11111111"),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", "pytorch"),
            _calidad("s_pruning_a_22222222", "openvino"),
        ]

        filas = dg.degradation_rows(experimentos, calidad, [], _conjunto())

        assert filas == ()

    def test_con_referencia_en_el_mismo_motor_si_se_compara(self) -> None:
        """El caso simétrico, para que el test anterior no pase por no calcular nunca."""
        experimentos = [
            _experimento("s_baseline_default_11111111", technique="baseline"),
            _experimento("s_pruning_a_22222222", "s_baseline_default_11111111"),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", "openvino", map_50_95="0.40"),
            _calidad("s_pruning_a_22222222", "openvino", map_50_95="0.33"),
        ]

        filas = dg.degradation_rows(experimentos, calidad, [], _conjunto())

        assert len(filas) == 1
        assert filas[0].reference.runtime == "openvino"
        assert filas[0].overall["map_50_95"].absolute == pytest.approx(-0.07)


class TestLaFilaDiceSiHuboReajuste:
    """La misma poda antes y después del reajuste son dos filas del registro."""

    def _campana(self, epocas: str) -> tuple[list, list]:
        """Baseline reajustado y una poda con las épocas indicadas."""
        experimentos = [
            _experimento(
                "s_baseline_default_11111111", technique="baseline", training_epochs="100"
            ),
            _experimento(
                "s_pruning_a_22222222", "s_baseline_default_11111111", training_epochs=epocas
            ),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", "pytorch", map_50_95="0.376"),
            _calidad("s_pruning_a_22222222", "pytorch", map_50_95="0.0"),
        ]
        return experimentos, calidad

    def test_sin_epocas_la_fila_sale_sin_reajuste(self) -> None:
        experimentos, calidad = self._campana("")

        filas = dg.degradation_rows(experimentos, calidad, [], _conjunto())

        assert filas[0].stage == dg.STAGE_RAW
        assert filas[0].to_dict()["stage"] == dg.STAGE_RAW

    def test_con_epocas_la_fila_sale_reajustada(self) -> None:
        """El simétrico, para que el anterior no pase por decir siempre lo mismo."""
        experimentos, calidad = self._campana("30")

        filas = dg.degradation_rows(experimentos, calidad, [], _conjunto())

        assert filas[0].stage == dg.STAGE_FINETUNED


class TestNoSeCruzaDeEvaluador:
    """El evaluador mueve el número tanto como el motor."""

    def _campana(self) -> tuple[list, list]:
        """Baseline con dos evaluadores y una variante con uno."""
        experimentos = [
            _experimento("s_baseline_default_11111111", technique="baseline"),
            _experimento("s_pruning_a_22222222", "s_baseline_default_11111111"),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", "pytorch", map_50_95="0.376214"),
            _calidad(
                "s_baseline_default_11111111",
                "pytorch",
                evaluator="ultralytics",
                device_id="auto",
                map_50_95="0.383774",
            ),
            _calidad("s_pruning_a_22222222", "pytorch", map_50_95="0.316322"),
        ]
        return experimentos, calidad

    def test_la_referencia_es_la_del_mismo_evaluador(self) -> None:
        """La variante es de pycocotools, así que su referencia también."""
        experimentos, calidad = self._campana()

        filas = dg.degradation_rows(experimentos, calidad, [], _conjunto())

        assert len(filas) == 1
        assert filas[0].overall["map_50_95"].reference == pytest.approx(0.376214)

    def test_varias_candidatas_que_discrepan_es_un_error(self) -> None:
        """Elegir una sería decidir a cara o cruz contra qué se degrada."""
        candidatas = [
            _calidad("s_baseline_default_11111111", "pytorch", map_50_95="0.376"),
            _calidad(
                "s_baseline_default_11111111",
                "pytorch",
                device_id="colab_t4_gpu",
                map_50_95="0.401",
            ),
        ]

        with pytest.raises(dg.DegradationError, match="filas que no coinciden"):
            dg._pick_reference(candidatas, "otro_dispositivo", ("map_50_95",))

    def test_varias_candidatas_que_coinciden_valen(self) -> None:
        """El mAP no depende de la máquina, y aquí se comprueba en vez de suponerlo."""
        candidatas = [
            _calidad("s_baseline_default_11111111", "pytorch", map_50_95="0.376"),
            _calidad(
                "s_baseline_default_11111111",
                "pytorch",
                device_id="colab_t4_gpu",
                map_50_95="0.376",
            ),
        ]

        elegida = dg._pick_reference(candidatas, "otro_dispositivo", ("map_50_95",))

        assert elegida is not None
        assert elegida["device_id"] == "colab_t4_gpu"


class TestLaEficienciaNoDivideEntreRuido:
    """El denominador tiene que estar certificado antes de dividir."""

    def test_una_reduccion_por_debajo_del_suelo_no_se_reparte(self) -> None:
        """Un 5 % de mejora con un suelo del 15 % no es una reducción."""
        eficiencia = dg.compression_efficiency(
            quality_loss=0.05,
            quality_column="map_50_95",
            reference_value=100.0,
            value=95.0,
            axis="latency_p95_ms",
            floors=_conjunto(latency_p95_ms=_suelo_relativo("latency_p95_ms", 0.147)),
            same_session=True,
        )

        assert eficiencia.points_per_factor is None
        assert not eficiencia.certified
        assert "ruido amplificado" in eficiencia.reason

    def test_una_reduccion_por_encima_del_suelo_si_se_reparte(self) -> None:
        eficiencia = dg.compression_efficiency(
            quality_loss=0.06,
            quality_column="map_50_95",
            reference_value=200.0,
            value=100.0,
            axis="latency_p95_ms",
            floors=_conjunto(latency_p95_ms=_suelo_relativo("latency_p95_ms", 0.147)),
            same_session=True,
        )

        assert eficiencia.certified
        assert eficiencia.factor == pytest.approx(2.0)
        assert eficiencia.points_per_factor == pytest.approx(0.06)

    def test_un_eje_sin_suelo_medido_no_certifica_nada(self) -> None:
        """Sin suelo no se puede afirmar que la reducción sea real."""
        eficiencia = dg.compression_efficiency(
            quality_loss=0.06,
            quality_column="map_50_95",
            reference_value=200.0,
            value=100.0,
            axis="map_50",
            floors=_conjunto(),
            same_session=True,
        )

        assert eficiencia.points_per_factor is None
        assert "no tiene suelo de medida medido" in eficiencia.reason

    def test_el_suelo_absoluto_se_aplica_en_sus_unidades(self) -> None:
        """Un suelo absoluto no se multiplica por el valor."""
        suelo = fl.AxisFloor(
            axis="map_50_95",
            kind="absoluto",
            same_session=0.004579,
            cross_session=0.004579,
            provenance="suelo del evaluador",
        )

        eficiencia = dg.compression_efficiency(
            quality_loss=0.01,
            quality_column="map_50",
            reference_value=0.400,
            value=0.398,
            axis="map_50_95",
            floors=_conjunto(map_50_95=suelo),
            same_session=True,
        )

        assert eficiencia.points_per_factor is None
        assert not eficiencia.certified

    def test_un_cero_en_el_eje_es_una_medida_rota(self) -> None:
        """Dividir por cero daría un factor infinito con aspecto de récord."""
        with pytest.raises(dg.DegradationError, match="medida rota"):
            dg.compression_efficiency(
                quality_loss=0.01,
                quality_column="map_50_95",
                reference_value=100.0,
                value=0.0,
                axis="latency_p95_ms",
                floors=_conjunto(),
                same_session=True,
            )


def _suelos_de_la_campana() -> fl.FloorSet:
    """Los dos suelos de p95 tal como salen de la campaña."""
    return _conjunto(
        latency_p95_ms=fl.AxisFloor(
            axis="latency_p95_ms",
            kind="relativo",
            same_session=0.1472,
            cross_session=0.4729,
            provenance="valores de la campaña, para la prueba",
        )
    )


def _despliegue(experiment_id: str, runtime: str, p95: str, entorno: str) -> dict[str, str]:
    """Fila de despliegue de prueba."""
    return {
        "experiment_id": experiment_id,
        "runtime": runtime,
        "precision": "fp32",
        "device_id": "workstation_cpu",
        "environment_id": entorno,
        "latency_p95_ms": p95,
    }


class TestElSueloDependeDeLaSesion:
    """El suelo depende de la huella de las dos filas."""

    #: Las cifras de la poda L1 r25 en ONNX Runtime: 24,2 ms de diferencia, por
    #: encima del margen de la misma sesión (19,3) y muy por debajo del otro (61,9).
    REFERENCIA = 155.15
    VARIANTE = 130.901

    def test_en_la_misma_sesion_certifica(self) -> None:
        """Con el suelo de la misma sesión el factor 1,185 se distingue de 1."""
        eficiencia = dg.compression_efficiency(
            quality_loss=0.018,
            quality_column="map_50_95",
            reference_value=self.REFERENCIA,
            value=self.VARIANTE,
            axis="latency_p95_ms",
            floors=_suelos_de_la_campana(),
            same_session=True,
        )

        assert eficiencia.certified
        assert eficiencia.same_session
        assert eficiencia.factor == pytest.approx(1.185, abs=1e-3)

    def test_entre_sesiones_no_certifica(self) -> None:
        eficiencia = dg.compression_efficiency(
            quality_loss=0.018,
            quality_column="map_50_95",
            reference_value=self.REFERENCIA,
            value=self.VARIANTE,
            axis="latency_p95_ms",
            floors=_suelos_de_la_campana(),
            same_session=False,
        )

        assert not eficiencia.certified
        assert not eficiencia.same_session
        assert "entre sesiones" in eficiencia.reason

    @pytest.mark.parametrize(
        ("entorno_referencia", "entorno_variante", "misma"),
        [
            ("af4650a76eb8", "af4650a76eb8", True),
            ("af4650a76eb8", "6484b9f04e71", False),
            ("", "", False),
        ],
        ids=["misma_huella", "huellas_distintas", "sin_huella"],
    )
    def test_degradation_rows_decide_la_sesion_por_las_dos_filas(
        self, entorno_referencia: str, entorno_variante: str, misma: bool
    ) -> None:
        """Igual huella, suelo pequeño; distinta o vacía, el grande."""
        experimentos = [
            _experimento("s_baseline_default_11111111", technique="baseline"),
            _experimento("s_pruning_a_22222222", "s_baseline_default_11111111"),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", "onnxruntime", map_50_95="0.40"),
            _calidad("s_pruning_a_22222222", "onnxruntime", map_50_95="0.38"),
        ]
        despliegue = [
            _despliegue(
                "s_baseline_default_11111111",
                "onnxruntime",
                str(self.REFERENCIA),
                entorno_referencia,
            ),
            _despliegue(
                "s_pruning_a_22222222", "onnxruntime", str(self.VARIANTE), entorno_variante
            ),
        ]

        (fila,) = dg.degradation_rows(experimentos, calidad, despliegue, _suelos_de_la_campana())
        eficiencia = fila.efficiency["latency_p95_ms"]

        assert eficiencia.same_session is misma
        assert eficiencia.certified is misma
        assert eficiencia.to_dict()["same_session"] is misma


class TestElPuntoDeRuptura:
    """Detectar aceleración exige dos curvaturas, y eso son cuatro niveles."""

    def test_con_tres_niveles_no_es_determinable(self) -> None:
        """Es exactamente lo que tiene esta campaña, y la respuesta es que no."""
        resultado = dg.find_breakpoint([(0.0, 0.40), (0.25, 0.37), (0.5, 0.31)])

        assert not resultado.determinable
        assert resultado.level is None
        assert "al menos 4" in resultado.reason

    def test_con_cinco_niveles_encuentra_donde_se_acelera(self) -> None:
        # Plana hasta 0,50 y desplomándose a partir de ahí.
        niveles = [(0.0, 0.40), (0.25, 0.39), (0.50, 0.38), (0.75, 0.20), (0.90, 0.02)]

        resultado = dg.find_breakpoint(niveles)

        assert resultado.determinable
        assert resultado.level == 0.50
        assert len(resultado.second_differences) == 3

    def test_la_respuesta_declara_que_no_es_un_contraste(self) -> None:
        """Sin dispersión medida, el nivel es descriptivo y hay que decirlo."""
        resultado = dg.find_breakpoint([(0.0, 0.40), (0.25, 0.39), (0.50, 0.38), (0.75, 0.20)])

        assert "no debe citarse como significativa" in resultado.reason

    def test_dos_calidades_para_el_mismo_nivel_es_un_error(self) -> None:
        """Promediarlas escondería que son dos configuraciones distintas."""
        with pytest.raises(dg.DegradationError, match="dos calidades distintas"):
            dg.find_breakpoint([(0.5, 0.30), (0.5, 0.10), (0.25, 0.35), (0.0, 0.40)])


class TestLosDesgloses:
    """Por clase y por tamaño, con los nombres que dice el registro."""

    def test_el_nombre_de_clase_sale_de_la_columna(self) -> None:
        """Las etiquetas del mapa de calor no se inventan."""
        assert dg.class_label("ap_class_02_ear_mufs") == "ear mufs"

    def test_una_columna_con_otro_formato_es_un_error(self) -> None:
        """Deducir un nombre pondría una etiqueta falsa en un eje."""
        with pytest.raises(dg.DegradationError, match="no sigue el formato"):
            dg.class_label("ap_class_sin_indice")

    def test_las_columnas_de_tamano_siguen_el_criterio(self) -> None:
        """El registro guarda los dos criterios y hay que elegir cuál se lee."""
        assert dg.size_columns("original_res")["small"] == "ap_small_original_res"

    def test_una_celda_ilegible_no_se_trata_como_ausente(self) -> None:
        """Borraría la diferencia entre «no se midió» y «se escribió mal»."""
        referencia = _calidad("s_baseline_default_11111111", "pytorch")
        variante = _calidad("s_pruning_a_22222222", "pytorch", map_50="no_es_un_numero")

        with pytest.raises(dg.DegradationError, match="no es un número"):
            dg.delta(referencia, variante, "map_50")

    def test_una_celda_vacia_no_produce_diferencia(self) -> None:
        """Sin medida no hay diferencia que calcular, y no se rellena con cero."""
        referencia = _calidad("s_baseline_default_11111111", "pytorch")
        variante = _calidad("s_pruning_a_22222222", "pytorch", map_50="")

        assert dg.delta(referencia, variante, "map_50") is None


class TestSobreElRegistroReal:
    """Comprobaciones sobre el registro de la campaña."""

    @pytest.fixture(scope="class")
    def filas(self) -> tuple[dg.DegradationRow, ...]:
        """Degradación calculada sobre el registro del proyecto."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        suelos = fl.measure_floors(
            despliegue,
            ("latency_p95_ms", "artifact_size_bytes", "map_50", "map_50_95"),
            RAIZ,
        )
        return dg.degradation_rows(experimentos, calidad, despliegue, suelos)

    def test_toda_fila_se_compara_dentro_de_su_motor(
        self, filas: tuple[dg.DegradationRow, ...]
    ) -> None:
        """Ninguna comparación cruza de motor, que es la regla del módulo."""
        assert filas
        for fila in filas:
            assert fila.reference.runtime == fila.runtime

    def test_ninguna_variante_se_compara_consigo_misma(
        self, filas: tuple[dg.DegradationRow, ...]
    ) -> None:
        """La raíz no es una variante degradada de nada."""
        for fila in filas:
            assert fila.reference.experiment_id != fila.experiment_id

    def test_las_diecisiete_clases_tienen_desglose(
        self, filas: tuple[dg.DegradationRow, ...]
    ) -> None:
        """El mapa de calor por clase necesita las diecisiete de SH17."""
        assert all(len(fila.per_class) == 17 for fila in filas)

    def test_ninguna_fila_se_queda_sin_desglose_por_tamano(
        self, filas: tuple[dg.DegradationRow, ...]
    ) -> None:
        sin_tamano = [f.experiment_id for f in filas if not f.per_size]
        assert sin_tamano == []

    def test_toda_referencia_usa_el_evaluador_de_la_variante(
        self, filas: tuple[dg.DegradationRow, ...]
    ) -> None:
        """Con el evaluador cruzado, cada diferencia lleva 8 milésimas de más."""
        assert all(fila.reference.experiment_id for fila in filas)
        assert len({f.runtime for f in filas}) >= 2

    def test_en_una_sola_sesion_ninguna_aceleracion_queda_sin_certificar(
        self, filas: tuple[dg.DegradationRow, ...]
    ) -> None:
        """Con el registro en una sola sesión, todas las p95 usan el suelo pequeño."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        despliegue, _ = registro.latency_comparable_rows()
        huellas = {f.get("environment_id", "") for f in despliegue}
        assert len(huellas) == 1, (
            f"El registro de latencia ya no es de una sola sesión ({sorted(huellas)}): "
            f"este test supone una campaña única y hay que revisarlo."
        )

        p95 = [f.efficiency["latency_p95_ms"] for f in filas if "latency_p95_ms" in f.efficiency]
        assert len(p95) == 38
        assert all(e.same_session for e in p95)
        sin_certificar = [e for e in p95 if not e.certified]
        assert sin_certificar == []


class TestElCosteDeLaCadenaCuandoUnEslabonNoEntrena:
    """Un eslabón de PTQ no da un paso de gradiente, y eso no es una ausencia."""

    @pytest.fixture()
    def experimentos(self) -> dict[str, dict[str, str]]:
        """Baseline, poda con reajuste y combinación sin entrenamiento."""
        return {
            "s_baseline_default_00000000": _experimento(
                "s_baseline_default_00000000", technique="baseline", training_time_s="11652.85"
            ),
            "s_pruning_x_11111111": _experimento(
                "s_pruning_x_11111111",
                padre="s_baseline_default_00000000",
                training_time_s="2832.03",
            ),
            "s_combined_x_22222222": _experimento(
                "s_combined_x_22222222",
                padre="s_pruning_x_11111111",
                technique="combined",
                training_time_s="",
            ),
        }

    def test_sin_el_coste_de_exportar_la_cadena_no_tiene_total(
        self, experimentos: dict[str, dict[str, str]]
    ) -> None:
        """Es el estado en el que estaba la campaña, y era correcto declararlo."""
        coste = dg.chain_cost("s_combined_x_22222222", experimentos)

        assert coste.seconds is None
        assert "s_combined_x_22222222" in coste.missing

    def test_con_el_coste_de_exportar_la_cadena_se_cierra(
        self, experimentos: dict[str, dict[str, str]]
    ) -> None:
        """Producir el combinado costó reajustar la poda y cuantizarla."""
        coste = dg.chain_cost(
            "s_combined_x_22222222",
            experimentos,
            extra_costs={"s_combined_x_22222222": 20.83},
        )

        assert coste.seconds == pytest.approx(2852.86)
        assert coste.missing == ()

    def test_el_coste_de_exportar_se_suma_al_de_entrenar_en_el_mismo_eslabon(
        self, experimentos: dict[str, dict[str, str]]
    ) -> None:
        coste = dg.chain_cost(
            "s_pruning_x_11111111",
            experimentos,
            extra_costs={"s_pruning_x_11111111": 1.08},
        )

        assert coste.seconds == pytest.approx(2833.11)

    def test_no_se_rellena_con_cero_un_eslabon_sin_coste(
        self, experimentos: dict[str, dict[str, str]]
    ) -> None:
        """Un cero afirmaría que fue gratis; lo que pasa es que no se apuntó."""
        coste = dg.chain_cost(
            "s_combined_x_22222222",
            experimentos,
            extra_costs={"s_pruning_x_11111111": 1.08},
        )

        assert coste.seconds is None
        assert coste.missing == ("s_combined_x_22222222",)

    def test_la_raiz_sigue_fuera_de_la_suma(self, experimentos: dict[str, dict[str, str]]) -> None:
        """El entrenamiento del baseline se paga una vez y no es comprimir."""
        coste = dg.chain_cost(
            "s_combined_x_22222222",
            experimentos,
            extra_costs={"s_combined_x_22222222": 20.83},
        )

        assert coste.seconds is not None
        assert coste.seconds < 11652.85
        assert all(eslabon != "s_baseline_default_00000000" for eslabon, _ in coste.contributors)
