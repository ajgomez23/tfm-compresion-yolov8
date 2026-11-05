"""Tests de la regla de lectura de la campaña de semillas."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tfm_edge.analysis.splits import select_analysis_split
from tfm_edge.common.paths import get_project_paths
from tfm_edge.compression.distillation.reading import EVALUATOR_READABILITY_FLOOR
from tfm_edge.compression.distillation.seed_variance import (
    MIN_SEEDS,
    T_CRITICAL_95_TWO_SIDED,
    VERDICT_HELPS,
    VERDICT_HURTS,
    VERDICT_NOISE,
    VERDICT_TOO_SMALL,
    DistillationRun,
    SeedCampaignError,
    SeedPair,
    assert_dispersion_is_not_degenerate,
    attach_environment_ids,
    find_run,
    load_distillation_runs,
    pair_by_seed,
    provenance_caveats,
    read_seed_campaign,
)

PILOTO_42 = "n_distillation_l2n_combined_t2_pilot10_cf761acd"
CONTROL_42 = "n_distillation_l2n_control_alpha0_pilot10_8826269c"

#: Subdirectorio donde se apartaron los volcados de ``per_branch_v1``, para que el directorio de la
#: campaña solo tenga la vigente.
ARCHIVO_V1 = "per_branch_v1"

#: Las dos cifras que hay en disco, con la precisión del volcado y no la del CSV.
MAP_50_95_PILOTO_42 = 0.3171713365109344
MAP_50_95_CONTROL_42 = 0.31141300557794604
MAP_50_PILOTO_42 = 0.5100515314941519
MAP_50_CONTROL_42 = 0.510145719945786


def _ejecucion(
    *,
    seed: int,
    is_control: bool,
    map_50_95: float,
    map_50: float = 0.51,
    environment_id: str = "22303afccc93",
    experiment_id: str | None = None,
) -> DistillationRun:
    """Fabrica una ejecución sintética."""
    papel = "control" if is_control else "pilot"
    # Como en los volcados reales, la inicialización depende de la semilla y es la misma para piloto
    # y control.
    inicializacion = (
        ("model.15", 9.2 + seed * 0.001),
        ("model.18", 13.0 + seed * 0.001),
        ("model.21", 13.05 + seed * 0.001),
    )
    return DistillationRun(
        experiment_id=experiment_id or f"n_distillation_{papel}_{seed}",
        seed_base=seed,
        seed_branch=1000 + seed,
        seeding_scheme="per_branch_v1",
        is_control=is_control,
        alpha=0.0 if is_control else 0.5,
        map_50=map_50,
        map_50_95=map_50_95,
        config_path="configs/compression/distillation/kd_pilot_10ep.yaml",
        training_source="entrenado en esta ejecución",
        preflight_recorded=True,
        dump_path=Path(f"/tmp/{papel}_{seed}.json"),
        environment_id=environment_id,
        adapter_initial_norms=inicializacion,
    )


def _parejas_con_diferencias(
    diferencias: list[float], *, base_control: float = 0.311413
) -> list[SeedPair]:
    """Construye parejas cuyas diferencias emparejadas son las pedidas."""
    parejas = []
    for indice, diferencia in enumerate(diferencias):
        semilla = 42 + indice
        desplazado = base_control + indice * 0.0001
        control = _ejecucion(seed=semilla, is_control=True, map_50_95=desplazado)
        piloto = _ejecucion(seed=semilla, is_control=False, map_50_95=desplazado + diferencia)
        parejas.append(SeedPair(seed=semilla, pilot=piloto, control=control))
    return parejas


class TestSobreLosVolcadosReales:
    """Que el lector consuma los ficheros que de verdad hay, con su estructura."""

    @pytest.fixture(scope="class")
    def ejecuciones(self) -> list[DistillationRun]:
        """Carga los volcados reales del repositorio."""
        raw = get_project_paths().results / "raw" / "distillation" / ARCHIVO_V1
        return load_distillation_runs(raw)

    def test_encuentra_las_dos_ejecuciones_de_la_semilla_42(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        identificadores = {e.experiment_id for e in ejecuciones}
        assert PILOTO_42 in identificadores
        assert CONTROL_42 in identificadores

    def test_las_semillas_salen_del_volcado_y_no_del_nombre(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        assert {e.seed_base for e in ejecuciones} == {42, 43}
        for ejecucion in ejecuciones:
            assert ejecucion.seeding_scheme == "per_branch_v1"

    def test_el_piloto_y_el_control_comparten_semilla_de_rama(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        """Dentro de una semilla, la única diferencia entre las dos ramas es ``alpha``."""
        for semilla in (42, 43):
            de_esta_semilla = [e for e in ejecuciones if e.seed_base == semilla]
            assert len(de_esta_semilla) == 2
            assert len({e.seed_branch for e in de_esta_semilla}) == 1
            assert {e.alpha for e in de_esta_semilla} == {0.0, 0.5}

        assert len({e.seed_branch for e in ejecuciones}) == 2

    def test_las_metricas_son_las_del_volcado_no_las_redondeadas(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        """El CSV redondea a seis decimales y el efecto está en el quinto."""
        piloto = find_run(ejecuciones, seed=42, is_control=False)
        assert piloto.map_50_95 == pytest.approx(MAP_50_95_PILOTO_42, abs=1e-15)

    def test_empareja_las_dos_semillas_que_hay(self, ejecuciones: list[DistillationRun]) -> None:
        """Y la pareja se busca por su semilla, nunca por su posición."""
        parejas = {pareja.seed: pareja for pareja in pair_by_seed(ejecuciones)}

        assert set(parejas) == {42, 43}
        assert parejas[42].delta_50_95 == pytest.approx(0.005758, abs=1e-6)
        assert parejas[42].delta_50 == pytest.approx(-0.000094, abs=1e-6)

    def test_la_campana_de_agosto_dispara_la_guarda_de_dispersion(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        parejas = pair_by_seed(ejecuciones)
        assert len(parejas) >= MIN_SEEDS, "el motivo del fallo no puede ser la falta de semillas"

        with pytest.raises(SeedCampaignError) as excepcion:
            read_seed_campaign(parejas)

        mensaje = str(excepcion.value)
        assert "MISMA métrica" in mensaje
        assert repr(MAP_50_95_PILOTO_42) in mensaje, "el mensaje tiene que traer la cifra repetida"
        assert "[42, 43]" in mensaje, "y decir qué semillas la comparten"
        assert "'seed' llega" in mensaje, "y qué comprobar antes de gastar más GPU"

    def test_el_efecto_de_la_semilla_42_esta_en_la_cola(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        """La observación de la que sale la segunda predicción fechada."""
        pareja = next(p for p in pair_by_seed(ejecuciones) if p.seed == 42)
        assert pareja.delta_50_95 > 0
        assert pareja.delta_50 < 0
        assert pareja.effect_is_in_the_tail

    def test_la_procedencia_asimetrica_de_la_semilla_42_consta(
        self, ejecuciones: list[DistillationRun]
    ) -> None:
        """El control se completó con --reuse-run y no tiene comprobación previa."""
        control = find_run(ejecuciones, seed=42, is_control=True)
        piloto = find_run(ejecuciones, seed=42, is_control=False)
        assert "reuse-run" in control.training_source
        assert not control.preflight_recorded
        assert piloto.preflight_recorded


class TestElDirectorioTraeMasQueEjecuciones:
    """El directorio de volcados contiene también las comprobaciones previas de memoria."""

    @pytest.fixture
    def directorio(self, tmp_path: Path) -> Path:
        """Reproduce el directorio tal y como queda tras una sesión de Colab."""
        real = get_project_paths().results / "raw" / "distillation" / ARCHIVO_V1
        for nombre in (f"{PILOTO_42}.json", f"{CONTROL_42}.json"):
            shutil.copy(real / nombre, tmp_path / nombre)
        (tmp_path / f"preflight_{CONTROL_42}_2026-09-07T093619Z.json").write_text(
            json.dumps({"experiment_id": CONTROL_42, "preflight": {"completed": True}}),
            encoding="utf-8",
        )
        return tmp_path

    def test_las_comprobaciones_previas_no_son_ejecuciones(self, directorio: Path) -> None:
        ejecuciones = load_distillation_runs(directorio)
        assert {e.experiment_id for e in ejecuciones} == {PILOTO_42, CONTROL_42}

    def test_resolver_el_control_con_previas_delante(self, directorio: Path) -> None:
        """Es exactamente lo que hace la celda 40 del cuaderno, y lo que falló."""
        control = find_run(load_distillation_runs(directorio), seed=42, is_control=True)
        assert control.experiment_id == CONTROL_42

    def test_un_volcado_de_ejecucion_truncado_sigue_siendo_un_error(self, directorio: Path) -> None:
        """El arreglo simétrico habría silenciado esto."""
        (directorio / "n_distillation_l2n_roto_00000000.json").write_text(
            json.dumps({"experiment_id": "n_distillation_l2n_roto_00000000"}),
            encoding="utf-8",
        )
        with pytest.raises(SeedCampaignError, match="seed_base"):
            load_distillation_runs(directorio)

    def test_una_previa_sin_forma_de_previa_no_se_descarta_en_silencio(
        self, directorio: Path
    ) -> None:
        """Si ``20_distill.py`` cambiara lo que vuelca, el filtro se enteraría."""
        (directorio / "preflight_de_otra_cosa.json").write_text(
            json.dumps({"experiment_id": "x", "metrics": {"map_50": 0.1}}),
            encoding="utf-8",
        )
        with pytest.raises(SeedCampaignError, match="no tiene su forma"):
            load_distillation_runs(directorio)


class TestQueLasParejasNoSeCrucen:
    """El piloto de una semilla, con el control de esa semilla."""

    def test_cada_piloto_va_con_su_control(self) -> None:
        ejecuciones = [
            _ejecucion(seed=42, is_control=True, map_50_95=0.311),
            _ejecucion(seed=43, is_control=True, map_50_95=0.312),
            _ejecucion(seed=42, is_control=False, map_50_95=0.317),
            _ejecucion(seed=43, is_control=False, map_50_95=0.318),
        ]
        parejas = pair_by_seed(ejecuciones)
        assert [p.seed for p in parejas] == [42, 43]
        for pareja in parejas:
            assert pareja.pilot.seed_base == pareja.control.seed_base

    def test_una_semilla_a_medias_se_omite_sin_fallar(self) -> None:
        ejecuciones = [
            _ejecucion(seed=42, is_control=True, map_50_95=0.311),
            _ejecucion(seed=42, is_control=False, map_50_95=0.317),
            _ejecucion(seed=43, is_control=True, map_50_95=0.312),
        ]
        parejas = pair_by_seed(ejecuciones)
        assert [p.seed for p in parejas] == [42]

    def test_dos_pilotos_de_la_misma_semilla_es_un_error(self) -> None:
        ejecuciones = [
            _ejecucion(seed=42, is_control=True, map_50_95=0.311),
            _ejecucion(seed=42, is_control=False, map_50_95=0.317, experiment_id="piloto_a"),
            _ejecucion(seed=42, is_control=False, map_50_95=0.319, experiment_id="piloto_b"),
        ]
        with pytest.raises(SeedCampaignError) as excepcion:
            pair_by_seed(ejecuciones)
        assert "piloto_a" in str(excepcion.value)
        assert "piloto_b" in str(excepcion.value)

    def test_el_mismo_volcado_dos_veces_no_es_un_empate(self) -> None:
        ejecuciones = [
            _ejecucion(seed=42, is_control=True, map_50_95=0.311),
            _ejecucion(seed=42, is_control=False, map_50_95=0.317),
            _ejecucion(seed=42, is_control=False, map_50_95=0.317),
        ]
        assert len(pair_by_seed(ejecuciones)) == 1

    def test_find_run_no_elige_entre_dos(self) -> None:
        ejecuciones = [
            _ejecucion(seed=42, is_control=True, map_50_95=0.311, experiment_id="a"),
            _ejecucion(seed=42, is_control=True, map_50_95=0.312, experiment_id="b"),
        ]
        with pytest.raises(SeedCampaignError):
            find_run(ejecuciones, seed=42, is_control=True)

    def test_find_run_dice_lo_que_hay_cuando_no_encuentra(self) -> None:
        ejecuciones = [_ejecucion(seed=42, is_control=True, map_50_95=0.311)]
        with pytest.raises(SeedCampaignError) as excepcion:
            find_run(ejecuciones, seed=43, is_control=True)
        assert "42" in str(excepcion.value)


class TestElCruceContraElRegistro:
    """Volcado y registro son dos escrituras del mismo número."""

    @staticmethod
    def _fila(experiment_id: str, map_50_95: str, entorno: str = "22303afccc93") -> dict:
        return {
            "experiment_id": experiment_id,
            "runtime": "pytorch",
            "evaluator": "pycocotools",
            "status": "ok",
            "environment_id": entorno,
            "map_50_95": map_50_95,
        }

    def test_rellena_el_entorno(self) -> None:
        ejecucion = replace(
            _ejecucion(seed=42, is_control=True, map_50_95=0.311413), environment_id=""
        )
        completadas = attach_environment_ids(
            [ejecucion], [self._fila(ejecucion.experiment_id, "0.311413")]
        )
        assert completadas[0].environment_id == "22303afccc93"

    def test_una_discrepancia_de_metrica_es_un_error(self) -> None:
        """Si divergen, alguien ha reescrito uno de los dos."""
        ejecucion = _ejecucion(seed=42, is_control=True, map_50_95=0.311413)
        with pytest.raises(SeedCampaignError) as excepcion:
            attach_environment_ids([ejecucion], [self._fila(ejecucion.experiment_id, "0.350000")])
        assert "volcado" in str(excepcion.value)

    def test_el_redondeo_del_csv_no_es_una_discrepancia(self) -> None:
        """El CSV guarda seis decimales; el volcado, todos."""
        ejecucion = _ejecucion(seed=42, is_control=False, map_50_95=MAP_50_95_PILOTO_42)
        completadas = attach_environment_ids(
            [ejecucion], [self._fila(ejecucion.experiment_id, "0.317171")]
        )
        assert completadas[0].map_50_95 == pytest.approx(MAP_50_95_PILOTO_42, abs=1e-15)

    def test_una_fila_de_otro_evaluador_no_cuenta(self) -> None:
        """Mezclar evaluadores mete la diferencia que el umbral existe para descartar."""
        ejecucion = replace(
            _ejecucion(seed=42, is_control=True, map_50_95=0.311413), environment_id=""
        )
        fila = self._fila(ejecucion.experiment_id, "0.330215")
        fila["evaluator"] = "ultralytics"
        completadas = attach_environment_ids([ejecucion], [fila])
        assert completadas[0].environment_id == ""

    def test_la_fila_de_test_del_mismo_experimento_se_filtra_antes(self) -> None:
        """Val y test del mismo experimento dan cifras distintas."""
        ejecucion = _ejecucion(seed=42, is_control=True, map_50_95=0.311413)
        val = {**self._fila(ejecucion.experiment_id, "0.311413"), "split": "val"}
        test = {**self._fila(ejecucion.experiment_id, "0.298000", "d76308ac7935"), "split": "test"}
        with pytest.raises(SeedCampaignError):
            attach_environment_ids([ejecucion], [val, test])
        completadas = attach_environment_ids([ejecucion], select_analysis_split([val, test]))
        assert completadas[0].environment_id == "22303afccc93"


class TestLosCuatroVeredictos:
    """Fijados antes de ver los resultados, uno por caso."""

    def test_efecto_consistente_y_positivo(self) -> None:
        lectura = read_seed_campaign(_parejas_con_diferencias([0.010, 0.011, 0.0105, 0.0102]))
        assert lectura.verdict == VERDICT_HELPS
        assert not lectura.ci_crosses_zero
        assert lectura.ci_low > 0

    def test_efecto_consistente_y_negativo(self) -> None:
        lectura = read_seed_campaign(_parejas_con_diferencias([-0.010, -0.011, -0.0105, -0.0102]))
        assert lectura.verdict == VERDICT_HURTS
        assert lectura.ci_high < 0

    def test_dispersion_que_alcanza_al_efecto_es_ruido(self) -> None:
        """El desenlace que la campaña existe para poder declarar."""
        lectura = read_seed_campaign(_parejas_con_diferencias([0.006, -0.004, 0.002, -0.003]))
        assert lectura.verdict == VERDICT_NOISE
        assert lectura.ci_crosses_zero
        assert lectura.sd_delta >= abs(lectura.mean_delta)

    def test_efecto_pequeno_no_es_ausencia_de_efecto(self) -> None:
        """La distinción que evita leer un resultado ambiguo como confirmación."""
        lectura = read_seed_campaign(_parejas_con_diferencias([0.0002, 0.0012, 0.0012, 0.0022]))
        assert lectura.verdict == VERDICT_TOO_SMALL
        assert lectura.ci_crosses_zero
        assert lectura.sd_delta < abs(lectura.mean_delta)
        assert "no hay efecto" in lectura.statement


class TestElUmbralEntreRuidoYEfectoPequeno:
    """El umbral entre ruido y efecto pequeño es una comparación explícita."""

    def test_la_frontera_es_sd_mayor_o_igual_que_la_media(self) -> None:
        """Los dos veredictos que cruzan el cero se reparten por ``sd >= |media|``."""
        ruido = read_seed_campaign(_parejas_con_diferencias([0.006, -0.004, 0.002, -0.003]))
        pequeno = read_seed_campaign(_parejas_con_diferencias([0.0002, 0.0012, 0.0012, 0.0022]))
        assert ruido.ci_crosses_zero and pequeno.ci_crosses_zero
        assert ruido.sd_delta >= abs(ruido.mean_delta)
        assert pequeno.sd_delta < abs(pequeno.mean_delta)
        assert ruido.verdict != pequeno.verdict


class TestElEstadistico:
    """Con n=4 se usa la t de Student."""

    def test_usa_t3_y_no_dos(self) -> None:
        lectura = read_seed_campaign(_parejas_con_diferencias([0.004, 0.005, 0.006, 0.007]))
        assert lectura.n_seeds == 4
        assert lectura.t_critical == T_CRITICAL_95_TWO_SIDED[3] == 3.182

    def test_el_atajo_de_dos_errores_tipicos_daria_otro_veredicto(self) -> None:
        """Por qué la distinción no es cosmética."""
        parejas = _parejas_con_diferencias([0.0002, 0.0012, 0.0012, 0.0022])
        lectura = read_seed_campaign(parejas)

        error_tipico = lectura.sd_delta / lectura.n_seeds**0.5
        assert lectura.mean_delta - 2 * error_tipico > 0  # el atajo declararía efecto
        assert lectura.ci_low < 0  # el estadístico correcto no
        assert lectura.ci_crosses_zero
        assert lectura.verdict == VERDICT_TOO_SMALL

    def test_fuera_de_la_tabla_se_falla_en_vez_de_extrapolar(self) -> None:
        # Doce diferencias distintas: con doce iguales saltaría antes la guarda de
        # dispersión degenerada y este test dejaría de comprobar lo que dice.
        parejas = _parejas_con_diferencias([0.001 + i * 0.0001 for i in range(12)])
        with pytest.raises(SeedCampaignError) as excepcion:
            read_seed_campaign(parejas)
        assert "grados de libertad" in str(excepcion.value)

    def test_la_tabla_trae_el_valor_que_la_campana_necesita(self) -> None:
        assert T_CRITICAL_95_TWO_SIDED[3] == 3.182


class TestLasPrediccionesFechadas:
    """Las dos afirmaciones que esta campaña puede refutar."""

    def test_el_suelo_se_declara_cota_inferior_o_no(self) -> None:
        """La predicción de ``reading.py``, evaluada con la dispersión medida."""
        dispersos = [
            SeedPair(
                seed=42 + i,
                pilot=_ejecucion(
                    seed=42 + i, is_control=False, map_50_95=0.317 + i * 0.01 + i * 0.0003
                ),
                control=_ejecucion(seed=42 + i, is_control=True, map_50_95=0.311 + i * 0.01),
            )
            for i in range(4)
        ]
        lectura = read_seed_campaign(dispersos)
        assert lectura.control_sd > EVALUATOR_READABILITY_FLOOR
        assert lectura.floor_is_lower_bound

    def test_una_dispersion_menor_que_el_suelo_refuta_a_reading(self) -> None:
        apretados = [
            SeedPair(
                seed=42 + i,
                pilot=_ejecucion(
                    seed=42 + i, is_control=False, map_50_95=0.317 + i * 0.00001 + i * 0.000003
                ),
                control=_ejecucion(seed=42 + i, is_control=True, map_50_95=0.311 + i * 0.00001),
            )
            for i in range(4)
        ]
        lectura = read_seed_campaign(apretados)
        assert lectura.control_sd < EVALUATOR_READABILITY_FLOOR
        assert not lectura.floor_is_lower_bound

    def test_la_prediccion_de_la_cola_se_evalua_en_todas_las_semillas(self) -> None:
        """Basta con que una semilla la incumpla para que no se cumpla."""
        parejas = _parejas_con_diferencias([0.005, 0.006, 0.004, 0.005])
        assert read_seed_campaign(parejas).tail_prediction_holds

        rebelde = parejas[-1]
        parejas[-1] = SeedPair(
            seed=rebelde.seed,
            pilot=replace(rebelde.pilot, map_50=0.60),
            control=rebelde.control,
        )
        assert not read_seed_campaign(parejas).tail_prediction_holds


class TestElInformeSeSerializa:
    """Lo que se vuelca tiene que poder volver a leerse."""

    def test_to_dict_es_json(self) -> None:
        lectura = read_seed_campaign(_parejas_con_diferencias([0.004, 0.005, 0.006, 0.007]))
        texto = json.dumps(lectura.to_dict(), ensure_ascii=False)
        vuelta = json.loads(texto)
        assert vuelta["n_seeds"] == 4
        assert vuelta["verdict"] == lectura.verdict
        assert vuelta["seeds"] == [42, 43, 44, 45]
        # Los avisos viajan con el veredicto: un veredicto sin ellos se lee como
        # más concluyente de lo que es.
        assert len(vuelta["caveats"]) >= 3

    def test_el_entorno_se_reporta_no_se_supone(self) -> None:
        parejas = _parejas_con_diferencias([0.004, 0.005, 0.006, 0.007])
        assert read_seed_campaign(parejas).all_pairs_same_environment

        primera = parejas[0]
        parejas[0] = SeedPair(
            seed=primera.seed,
            pilot=replace(primera.pilot, environment_id="otro"),
            control=primera.control,
        )
        assert not read_seed_campaign(parejas).all_pairs_same_environment

    def test_sin_dato_no_es_lo_mismo_que_distinto(self) -> None:
        """Aplastar los dos estados es hacer que el informe afirme lo que no sabe."""
        pareja = _parejas_con_diferencias([0.005])[0]
        assert pareja.environment_state == "mismo"

        distinta = SeedPair(
            seed=pareja.seed,
            pilot=replace(pareja.pilot, environment_id="otro"),
            control=pareja.control,
        )
        assert distinta.environment_state == "DISTINTO"
        assert not distinta.same_environment

        sin_dato = SeedPair(
            seed=pareja.seed,
            pilot=replace(pareja.pilot, environment_id=""),
            control=pareja.control,
        )
        assert sin_dato.environment_state == "sin dato"
        # Y sigue siendo False para decidir: sin dato no se da por bueno.
        assert not sin_dato.same_environment


class TestLaDispersionDegenerada:
    """El caso con menos información posible no puede dar el veredicto más rotundo."""

    def _pareja(self, semilla: int, piloto: float, control: float) -> SeedPair:
        """Una pareja con las métricas pedidas."""
        return SeedPair(
            seed=semilla,
            pilot=_ejecucion(seed=semilla, is_control=False, map_50_95=piloto),
            control=_ejecucion(seed=semilla, is_control=True, map_50_95=control),
        )

    def test_el_incidente_real_se_rechaza(self) -> None:
        """Con las cifras exactas de disco, la lectura falla en vez de resumir."""
        repetidas = [
            self._pareja(42, MAP_50_95_PILOTO_42, MAP_50_95_CONTROL_42),
            self._pareja(43, MAP_50_95_PILOTO_42, MAP_50_95_CONTROL_42),
        ]

        with pytest.raises(SeedCampaignError) as excepcion:
            read_seed_campaign(repetidas)

        mensaje = str(excepcion.value)
        assert "MISMA métrica" in mensaje
        assert "'seed' llega" in mensaje

    def test_sin_la_guarda_el_veredicto_seria_el_mas_seguro(self) -> None:
        import statistics

        diferencias = [MAP_50_95_PILOTO_42 - MAP_50_95_CONTROL_42] * 2
        desviacion = statistics.stdev(diferencias)
        margen = T_CRITICAL_95_TWO_SIDED[1] * desviacion / len(diferencias) ** 0.5

        assert desviacion == 0.0
        assert margen == 0.0
        media = statistics.fmean(diferencias)
        assert media > 0
        assert not (media - margen <= 0.0 <= media + margen), "no cruzaría el cero"

    def test_controles_identicos_con_la_semilla_llegando_se_leen(self) -> None:
        parejas = [
            self._pareja(42, 0.3171, MAP_50_95_CONTROL_42),
            self._pareja(43, 0.3185, MAP_50_95_CONTROL_42),
            self._pareja(44, 0.3160, MAP_50_95_CONTROL_42),
        ]

        lectura = read_seed_campaign(parejas)
        assert lectura.controls_identical
        assert lectura.control_sd == 0.0

    def test_la_inicializacion_compartida_se_rechaza_en_los_controles(self) -> None:
        parejas = [
            self._pareja(42, 0.3171, MAP_50_95_CONTROL_42),
            self._pareja(43, 0.3185, MAP_50_95_CONTROL_42),
        ]
        misma = parejas[0].control.adapter_initial_norms
        parejas[1] = SeedPair(
            seed=43,
            pilot=parejas[1].pilot,
            control=replace(parejas[1].control, adapter_initial_norms=misma),
        )

        with pytest.raises(SeedCampaignError) as excepcion:
            read_seed_campaign(parejas)
        mensaje = str(excepcion.value)
        assert "MISMA inicialización" in mensaje
        assert "controles" in mensaje
        assert "[42, 43]" in mensaje
        assert "'seed' llega" in mensaje

    def test_la_inicializacion_compartida_se_rechaza_en_los_pilotos(self) -> None:
        """Aunque la métrica de los pilotos difiera, la semilla no ha llegado."""
        parejas = [self._pareja(42, 0.3171, 0.3114), self._pareja(43, 0.3185, 0.3121)]
        misma = parejas[0].pilot.adapter_initial_norms
        parejas[1] = SeedPair(
            seed=43,
            pilot=replace(parejas[1].pilot, adapter_initial_norms=misma),
            control=parejas[1].control,
        )

        with pytest.raises(
            SeedCampaignError, match="MISMA inicialización de adaptador en los pilotos"
        ):
            read_seed_campaign(parejas)

    def test_sin_inicializacion_no_se_lee_a_ciegas(self) -> None:
        """Un volcado sin el observable no puede pasar la comprobación por omisión."""
        parejas = [self._pareja(42, 0.3171, 0.3114), self._pareja(43, 0.3185, 0.3121)]
        parejas[1] = SeedPair(
            seed=43,
            pilot=parejas[1].pilot,
            control=replace(parejas[1].control, adapter_initial_norms=()),
        )

        with pytest.raises(SeedCampaignError, match="adapter_initial_norms"):
            read_seed_campaign(parejas)

    def test_una_campana_con_variacion_real_pasa(self) -> None:
        """La guarda no puede rechazar lo que sí es una campaña."""
        parejas = [
            self._pareja(42, 0.3171, 0.3114),
            self._pareja(43, 0.3185, 0.3121),
            self._pareja(44, 0.3160, 0.3109),
        ]

        assert_dispersion_is_not_degenerate(parejas)
        lectura = read_seed_campaign(parejas)
        assert lectura.n_seeds == 3

    def test_diferencias_identicas_con_metricas_distintas_tambien_se_rechazan(self) -> None:
        parejas = [
            self._pareja(42, 0.3171, 0.3114),
            self._pareja(43, 0.3181, 0.3124),
        ]

        with pytest.raises(SeedCampaignError, match="exactamente cero"):
            read_seed_campaign(parejas)


class TestLoQueElInformeDice:
    """Lo que el informe afirma tiene que ser lo que la campaña mide."""

    def test_ninguna_frase_atribuye_nada_a_entre_entrenamientos(self) -> None:
        """La campaña no ve esa fuente de ruido; ningún veredicto puede nombrarla."""
        casos = {
            VERDICT_HELPS: [0.010, 0.011, 0.0105, 0.0102],
            VERDICT_HURTS: [-0.010, -0.011, -0.0105, -0.0102],
            VERDICT_NOISE: [0.006, -0.004, 0.002, -0.003],
            VERDICT_TOO_SMALL: [0.0002, 0.0012, 0.0012, 0.0022],
        }
        for veredicto, diferencias in casos.items():
            lectura = read_seed_campaign(_parejas_con_diferencias(diferencias))
            assert lectura.verdict == veredicto
            assert "entre entrenamientos" not in lectura.verdict
            assert "entre entrenamientos" not in lectura.statement

    def test_ayuda_se_lee_a_flujo_de_datos_fijo(self) -> None:
        lectura = read_seed_campaign(_parejas_con_diferencias([0.010, 0.011, 0.0105, 0.0102]))
        assert "a flujo de datos fijo" in lectura.statement

    def test_la_prediccion_del_suelo_se_declara_no_contrastable(self) -> None:
        """La regla se evalúa igual; lo que no se hace es presentarla como prueba."""
        lectura = read_seed_campaign(_parejas_con_diferencias([0.004, 0.005, 0.006, 0.007]))
        assert not lectura.floor_prediction_testable
        assert "No contrastable" in lectura.floor_prediction_status
        vuelta = lectura.to_dict()
        assert vuelta["floor_prediction_testable"] is False
        assert "floor_is_lower_bound" in vuelta, "el resultado de la regla no se borra"

    def test_las_salvedades_de_procedencia_salen_de_los_datos(self) -> None:
        """Una pareja limpia no arrastra aviso; una reutilizada, sí."""
        limpias = _parejas_con_diferencias([0.004, 0.005, 0.006])
        assert provenance_caveats(limpias) == ()

        reutilizada = SeedPair(
            seed=limpias[0].seed,
            pilot=limpias[0].pilot,
            control=replace(
                limpias[0].control,
                training_source="reutilizado con --reuse-run",
                preflight_recorded=False,
            ),
        )
        avisos = provenance_caveats([reutilizada, *limpias[1:]])
        assert len(avisos) == 1
        assert "La semilla 42 no es una pareja limpia" in avisos[0]
        assert "--reuse-run" in avisos[0]
        lectura = read_seed_campaign([reutilizada, *limpias[1:]])
        assert avisos[0] in lectura.caveats


class TestSobreLaCampanaV2:
    """Comprobaciones sobre los ocho volcados per_branch_v2."""

    @pytest.fixture(scope="class")
    def parejas(self) -> list[SeedPair]:
        """Empareja los volcados reales del directorio que lee la campaña."""
        raw = get_project_paths().results / "raw" / "distillation"
        return pair_by_seed(load_distillation_runs(raw))

    def test_son_las_cuatro_semillas_del_esquema_v2(self, parejas: list[SeedPair]) -> None:
        assert [p.seed for p in parejas] == [42, 43, 44, 45]
        for pareja in parejas:
            assert pareja.pilot.seeding_scheme == "per_branch_v2"
            assert pareja.control.seeding_scheme == "per_branch_v2"

    def test_la_guarda_nueva_pasa_y_los_controles_son_la_misma_ejecucion(
        self, parejas: list[SeedPair]
    ) -> None:
        assert_dispersion_is_not_degenerate(parejas)
        lectura = read_seed_campaign(parejas)
        assert lectura.controls_identical
        assert lectura.control_sd == 0.0

    def test_cada_semilla_da_su_inicializacion_y_la_comparten_sus_dos_ramas(
        self, parejas: list[SeedPair]
    ) -> None:
        inicializaciones = [p.pilot.adapter_initial_norms for p in parejas]
        assert all(inicializaciones)
        assert len(set(inicializaciones)) == len(parejas)
        for pareja in parejas:
            assert pareja.pilot.adapter_initial_norms == pareja.control.adapter_initial_norms

    def test_las_cuatro_parejas_son_limpias(self, parejas: list[SeedPair]) -> None:
        """Por eso la salvedad escrita de la semilla 42 tenía que desaparecer."""
        assert provenance_caveats(parejas) == ()


class TestLaSalidaDelScript:
    """Lo que ``32_distill_seed_variance.py`` imprime sobre la campaña real."""

    @pytest.fixture(scope="class")
    def salida(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        """Ejecuta el script con ``--dry-run`` y captura lo impreso."""
        import contextlib
        import importlib.util
        import io

        ruta = get_project_paths().root / "scripts" / "32_distill_seed_variance.py"
        especificacion = importlib.util.spec_from_file_location("seed_variance_script", ruta)
        assert especificacion is not None and especificacion.loader is not None
        modulo = importlib.util.module_from_spec(especificacion)
        especificacion.loader.exec_module(modulo)

        modulo.setup_logging = lambda *args, **kwargs: None
        salida_vacia = tmp_path_factory.mktemp("informe")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            codigo = modulo.main(["--dry-run", "--output-dir", str(salida_vacia)])
        assert codigo == 0
        assert not any(salida_vacia.iterdir()), "--dry-run no puede escribir el informe"
        return buffer.getvalue()

    def test_no_declara_falsa_una_prediccion_no_contrastada(self, salida: str) -> None:
        assert "FALSA" not in salida
        assert "NO CONTRASTABLE" in salida

    def test_no_rotula_a_los_controles_como_varianza_entre_entrenamientos(
        self, salida: str
    ) -> None:
        assert "ENTRE ENTRENAMIENTOS" not in salida
        assert "entre entrenamientos." not in salida.split("--- Predicciones")[0]
        assert "misma ejecución" in salida

    def test_la_salvedad_de_la_semilla_42_ya_no_sale(self, salida: str) -> None:
        assert "no es una pareja perfectamente limpia" not in salida
        assert "a flujo de datos fijo" in salida
