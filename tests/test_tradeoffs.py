"""Las curvas de compromiso, y lo que se niegan a unir en una sola línea."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm_edge.analysis import tradeoffs as td
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry

RAIZ = Path(__file__).resolve().parent.parent


def _experimento(
    experiment_id: str,
    variante: str,
    ratio: str = "",
    padre: str = "",
    epocas: str = "",
    tecnica: str = "pruning",
) -> dict[str, str]:
    """Fila de experimento de prueba."""
    return {
        "experiment_id": experiment_id,
        "technique": tecnica,
        "technique_variant": variante,
        "compression_ratio": ratio,
        "parent_experiment_id": padre,
        "training_epochs": epocas,
        "status": "ok",
    }


def _calidad(experiment_id: str, runtime: str = "pytorch", **valores: str) -> dict[str, str]:
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


class TestElNivelSaleDelNombreYDeLaColumna:
    """Y los dos tienen que decir lo mismo."""

    @pytest.mark.parametrize(
        ("variante", "metodo", "nivel"),
        [
            ("l1global_r50", "l1global", 0.5),
            ("l1global_r25_randomctl", "l1global_randomctl", 0.25),
            ("l1global_r50_iter4", "l1global_iter4", 0.5),
            ("bnscale_r50", "bnscale", 0.5),
            ("prune-l1global-r50-ptqint8", "prune-l1global-ptqint8", 0.5),
            ("int8-noexcl", "int8-noexcl", None),
        ],
    )
    def test_el_metodo_es_el_nombre_sin_su_nivel(
        self, variante: str, metodo: str, nivel: float | None
    ) -> None:
        """El troceo aguanta guiones, guiones bajos y tokens en medio."""
        assert td.split_variant(variante) == (metodo, nivel)

    def test_dos_niveles_en_el_nombre_es_un_error(self) -> None:
        """Con dos no se sabe cuál manda."""
        with pytest.raises(td.TradeoffError, match="2 niveles"):
            td.split_variant("l1global_r25_r50")

    def test_el_nombre_y_la_columna_tienen_que_coincidir(self) -> None:
        """Si discrepan, elegir uno pondría el punto donde no va."""
        fila = _experimento("s_pruning_a_11111111", "l1global_r50", ratio="0.25")

        with pytest.raises(td.TradeoffError, match="pero declara compression_ratio"):
            td.level_of(fila)

    def test_cuando_coinciden_manda_la_columna(self) -> None:
        """El valor declarado es el que se usa, y el nombre solo da el método."""
        fila = _experimento("s_pruning_a_11111111", "l1global_r50", ratio="0.5")

        assert td.level_of(fila) == ("l1global", 0.5)


class TestLaEtapaSeparaDosProcedimientos:
    """Podar y podar-y-reajustar no son dos puntos de la misma curva."""

    def test_la_poda_en_crudo_y_la_reajustada_no_se_mezclan(self) -> None:
        experimentos = [
            _experimento("s_baseline_default_11111111", "default", tecnica="baseline"),
            _experimento(
                "s_pruning_a_22222222",
                "bnscale_r50",
                ratio="0.5",
                padre="s_baseline_default_11111111",
            ),
            _experimento(
                "s_pruning_b_33333333",
                "bnscale_r50",
                ratio="0.5",
                padre="s_baseline_default_11111111",
                epocas="30",
            ),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", map_50_95="0.376"),
            _calidad("s_pruning_a_22222222", map_50_95="0.0"),
            _calidad("s_pruning_b_33333333", map_50_95="0.33"),
        ]

        curvas = td.family_curves(experimentos, calidad, [])

        etapas = {c.stage for c in curvas}
        assert etapas == {td.STAGE_RAW, td.STAGE_FINETUNED}
        for curva in curvas:
            assert len({p.experiment_id for p in curva.points if not p.is_anchor}) == 1

    def test_la_etapa_mira_la_cadena_no_solo_al_experimento(self) -> None:
        """Cuantizar no entrena, pero el podado del que sale sí se reajustó."""
        experimentos = {
            "s_baseline_default_11111111": _experimento(
                "s_baseline_default_11111111", "default", tecnica="baseline", epocas="100"
            ),
            "s_pruning_a_22222222": _experimento(
                "s_pruning_a_22222222",
                "l1global_r50",
                ratio="0.5",
                padre="s_baseline_default_11111111",
                epocas="30",
            ),
            "s_combined_b_33333333": _experimento(
                "s_combined_b_33333333",
                "prune-l1global-r50-ptqint8",
                ratio="0.5",
                padre="s_pruning_a_22222222",
                tecnica="combined",
            ),
        }

        assert td.training_stage("s_combined_b_33333333", experimentos) == td.STAGE_FINETUNED

    def test_las_epocas_del_baseline_no_reajustan_nada(self) -> None:
        """Contarlas haría que todo descendiente saliera reajustado."""
        experimentos = {
            "s_baseline_default_11111111": _experimento(
                "s_baseline_default_11111111", "default", tecnica="baseline", epocas="100"
            ),
            "s_pruning_a_22222222": _experimento(
                "s_pruning_a_22222222",
                "l1global_r50",
                ratio="0.5",
                padre="s_baseline_default_11111111",
            ),
        }

        assert td.training_stage("s_pruning_a_22222222", experimentos) == td.STAGE_RAW


class TestLaFormaDeclaraCuantosPuntosLaSostienen:
    """Un punto no es un segmento y un segmento no es una curva."""

    def _campana(self, niveles: list[str]) -> tuple[list, list]:
        """Monta una campaña con un método a varios niveles."""
        experimentos = [
            _experimento("s_baseline_default_11111111", "default", tecnica="baseline", epocas="100")
        ]
        calidad = [_calidad("s_baseline_default_11111111")]
        for indice, ratio in enumerate(niveles):
            identificador = f"s_pruning_v{indice}_{indice}0000000"
            entero = int(round(float(ratio) * 100))
            experimentos.append(
                _experimento(
                    identificador,
                    f"l1global_r{entero}",
                    ratio=ratio,
                    padre="s_baseline_default_11111111",
                    epocas="30",
                )
            )
            calidad.append(_calidad(identificador))
        return experimentos, calidad

    def test_un_solo_nivel_con_ancla_es_un_segmento(self) -> None:
        """El ancla aporta el nivel cero, así que ya son dos."""
        experimentos, calidad = self._campana(["0.5"])

        curvas = td.family_curves(experimentos, calidad, [])

        assert [c.shape for c in curvas] == ["segmento"]

    def test_dos_niveles_con_ancla_son_una_curva(self) -> None:
        experimentos, calidad = self._campana(["0.25", "0.5"])

        curvas = td.family_curves(experimentos, calidad, [])

        assert [c.shape for c in curvas] == ["curva"]
        assert curvas[0].levels == (0.0, 0.25, 0.5)

    def test_dos_experimentos_en_el_mismo_nivel_es_un_error(self) -> None:
        """Dos puntos en la misma abscisa no forman una curva."""
        experimentos, calidad = self._campana(["0.5"])
        experimentos.append(
            _experimento(
                "s_pruning_otro_99999999",
                "l1global_r50",
                ratio="0.5",
                padre="s_baseline_default_11111111",
                epocas="30",
            )
        )
        calidad.append(_calidad("s_pruning_otro_99999999"))

        with pytest.raises(td.TradeoffError, match="dos experimentos en el nivel"):
            td.family_curves(experimentos, calidad, [])


class TestElAncla:
    """El nivel cero solo vale si es de la misma precisión."""

    def test_un_baseline_fp32_no_ancla_una_curva_int8(self) -> None:
        """Entre los dos hay una cuantización, que es otra técnica."""
        experimentos = [
            _experimento(
                "s_baseline_default_11111111", "default", tecnica="baseline", epocas="100"
            ),
            _experimento(
                "s_combined_a_22222222",
                "prune-l1global-r50-ptqint8",
                ratio="0.5",
                padre="s_baseline_default_11111111",
                tecnica="combined",
            ),
        ]
        calidad = [
            _calidad("s_baseline_default_11111111", "onnxruntime"),
            _calidad("s_combined_a_22222222", "onnxruntime", precision="int8"),
        ]

        curvas = td.family_curves(experimentos, calidad, [])

        assert len(curvas) == 1
        assert all(not p.is_anchor for p in curvas[0].points)
        assert "otra técnica" in curvas[0].anchor_note


class TestSobreElRegistroReal:
    """Comprobaciones sobre el registro de la campaña."""

    @pytest.fixture(scope="class")
    def curvas(self) -> tuple[td.FamilyCurve, ...]:
        """Curvas construidas sobre el registro del proyecto."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()
        return td.family_curves(experimentos, calidad, despliegue)

    def test_hay_curvas_de_verdad_y_tambien_puntos_sueltos(
        self, curvas: tuple[td.FamilyCurve, ...]
    ) -> None:
        """Las dos formas conviven, que es lo que la figura tiene que reflejar."""
        formas = {c.shape for c in curvas}
        assert "curva" in formas
        assert "punto" in formas

    def test_ninguna_curva_mezcla_dos_metodos(self, curvas: tuple[td.FamilyCurve, ...]) -> None:
        """Cada curva describe un procedimiento que alguien puede recorrer."""
        for curva in curvas:
            assert curva.method
            assert len(curva.points) == len(curva.levels)

    def test_hay_un_metodo_que_aparece_en_las_dos_etapas(
        self, curvas: tuple[td.FamilyCurve, ...]
    ) -> None:
        """Es la conflación que la comprobación levantó sobre estos datos."""
        por_etapa: dict[str, set[tuple[str, str]]] = {}
        for curva in curvas:
            por_etapa.setdefault(curva.stage, set()).add((curva.method, curva.runtime))

        compartidos = por_etapa.get(td.STAGE_RAW, set()) & por_etapa.get(td.STAGE_FINETUNED, set())
        assert compartidos, (
            "Ningún método aparece a la vez en crudo y reajustado: la separación por "
            "etapa no se está ejercitando con estos datos."
        )

    def test_ninguna_curva_mezcla_las_dos_etapas(self, curvas: tuple[td.FamilyCurve, ...]) -> None:
        """Una curva en crudo no lleva puntos reajustados ni al revés."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")
        indice = {f["experiment_id"]: f for f in experimentos}

        for curva in curvas:
            for punto in curva.points:
                if punto.is_anchor:
                    continue
                assert td.training_stage(punto.experiment_id, indice) == curva.stage
