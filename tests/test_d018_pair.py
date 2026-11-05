"""Tests de qué pareja de ramas PTQ se admite comparar con un QAT."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import ExperimentRegistry
from tfm_edge.compression.quantization.ptq import (
    Int8BranchResult,
    PtqError,
    compare_with_qat,
    resolve_d018_pair,
)
from tfm_edge.export import quantize
from tfm_edge.export.quantize import MaterializationMismatchError

PODADO = "s_pruning_l1global_r50_9419b26f"
OTRO_PODADO = "s_pruning_bnscale_r50_f51fef4c"
BASELINE = "s_baseline_default_745e3d08"
QAT_PODADO = "s_qat_int8-noexcl_43300d34"
QAT_BASELINE = "s_qat_int8-noexcl_d4d403fa"


def _fila(experiment_id: str, technique: str, parent: str = "") -> dict[str, str]:
    """Fila mínima de ``experiments.csv``."""
    return {"experiment_id": experiment_id, "technique": technique, "parent_experiment_id": parent}


@pytest.fixture
def experimentos() -> list[dict[str, str]]:
    """Un registro sintético con las dos ramas y un segundo podado."""
    return [
        _fila(BASELINE, "baseline"),
        _fila(PODADO, "pruning", BASELINE),
        _fila(OTRO_PODADO, "pruning", BASELINE),
        _fila(QAT_BASELINE, "qat", BASELINE),
        _fila(QAT_PODADO, "qat", PODADO),
        _fila("s_combined_prune-l1global-r50-ptqint8_9067dba2", "combined", PODADO),
    ]


def _informe(source_kind: str, parent: str | None = None) -> dict:
    """Informe PTQ mínimo."""
    informe: dict = {"source_kind": source_kind}
    if parent is not None:
        informe["parent_experiment_id"] = parent
    return informe


class TestPtqDelPodado:
    """La pareja (pruned, qat) solo sale si las dos ramas parten del mismo podado."""

    def test_mismo_podado_autoriza_la_pareja_de_la_poda(self, experimentos) -> None:
        pareja = resolve_d018_pair(_informe("pruned", PODADO), QAT_PODADO, experimentos)
        assert pareja == quantize.PRUNED_QAT_SOURCE_KINDS

    def test_sin_experimento_se_rechaza(self, experimentos) -> None:
        """Sin el experimento QAT no hay padre con el que cruzar."""
        with pytest.raises(PtqError, match="no se ha dado --experiment-id"):
            resolve_d018_pair(_informe("pruned", PODADO), "", experimentos)

    def test_otro_podado_se_rechaza(self, experimentos) -> None:
        with pytest.raises(PtqError, match="PTQ de ese modelo podado"):
            resolve_d018_pair(_informe("pruned", OTRO_PODADO), QAT_PODADO, experimentos)

    def test_el_qat_del_baseline_con_la_ptq_de_un_podado_se_rechaza(self, experimentos) -> None:
        """El simétrico del caso anterior."""
        with pytest.raises(PtqError, match="PTQ de ese modelo podado"):
            resolve_d018_pair(_informe("pruned", PODADO), QAT_BASELINE, experimentos)

    def test_un_informe_sin_padre_se_rechaza(self, experimentos) -> None:
        """Un informe que no dice de qué podado parte no autoriza nada."""
        with pytest.raises(PtqError, match="parent_experiment_id"):
            resolve_d018_pair(_informe("pruned"), QAT_PODADO, experimentos)

    def test_un_experimento_que_no_esta_en_el_registro_se_rechaza(self, experimentos) -> None:
        with pytest.raises(PtqError, match="no está en experiments.csv"):
            resolve_d018_pair(
                _informe("pruned", PODADO), "s_qat_int8-noexcl_00000000", experimentos
            )

    def test_un_experimento_que_no_es_qat_se_rechaza(self, experimentos) -> None:
        with pytest.raises(PtqError, match="no 'qat'"):
            resolve_d018_pair(_informe("pruned", BASELINE), PODADO, experimentos)

    def test_un_padre_comun_que_no_es_una_poda_se_rechaza(self, experimentos) -> None:
        """Un informe 'pruned' que dice partir del baseline no es la rama de la poda."""
        with pytest.raises(PtqError, match="no 'pruning'"):
            resolve_d018_pair(_informe("pruned", BASELINE), QAT_BASELINE, experimentos)


COMBINADO = "s_combined_prune-l1global-r50-ptqint8_9067dba2"


class TestQueElInformeCuadraConSuFila:
    """El informe declara su padre y su fila también."""

    def test_un_informe_que_cuadra_autoriza(self, experimentos) -> None:
        informe = _informe("pruned", PODADO)
        informe["experiment_id"] = COMBINADO

        assert resolve_d018_pair(informe, QAT_PODADO, experimentos) == (
            quantize.PRUNED_QAT_SOURCE_KINDS
        )

    def test_un_padre_distinto_del_inscrito_se_niega(self, experimentos) -> None:
        informe = _informe("pruned", OTRO_PODADO)
        informe["experiment_id"] = COMBINADO
        experimentos.append(_fila("s_qat_otro_11112222", "qat", OTRO_PODADO))

        with pytest.raises(PtqError, match="su fila del registro dice"):
            resolve_d018_pair(informe, "s_qat_otro_11112222", experimentos)

    def test_un_informe_de_un_experimento_no_inscrito_se_niega(self, experimentos) -> None:
        """Un informe cuyo experimento no está en el registro no autoriza nada."""
        informe = _informe("pruned", PODADO)
        informe["experiment_id"] = "s_combined_inventado_deadbeef"

        with pytest.raises(PtqError, match="no está en experiments.csv"):
            resolve_d018_pair(informe, QAT_PODADO, experimentos)

    def test_una_fila_de_otra_tecnica_se_niega(self, experimentos) -> None:
        """El informe y su fila describirían experimentos distintos."""
        informe = _informe("pruned", PODADO)
        informe["experiment_id"] = PODADO  # es la fila de la poda, no la combinada

        with pytest.raises(PtqError, match="technique="):
            resolve_d018_pair(informe, QAT_PODADO, experimentos)

    def test_un_informe_sin_identificador_avisa_y_no_para(
        self, experimentos, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            pareja = resolve_d018_pair(_informe("pruned", PODADO), QAT_PODADO, experimentos)

        assert pareja == quantize.PRUNED_QAT_SOURCE_KINDS
        assert "no declara 'experiment_id'" in caplog.text

    def test_tambien_se_comprueba_en_la_rama_del_baseline(self, experimentos) -> None:
        """La guarda no es solo de la rama de poda."""
        informe = _informe("baseline", BASELINE)
        informe["experiment_id"] = "s_ptq_int8-noexcl_5bcd71f0"
        experimentos.append(_fila("s_ptq_int8-noexcl_5bcd71f0", "ptq", OTRO_PODADO))

        with pytest.raises(PtqError, match="su fila del registro dice"):
            resolve_d018_pair(informe, QAT_BASELINE, experimentos)


class TestPtqDelBaseline:
    """La ruta anterior no cambia, y deja de servir para un QAT sobre podado."""

    def test_sin_experimento_se_mantiene_la_pareja_de_d018(self, experimentos) -> None:
        assert resolve_d018_pair(_informe("baseline"), "", experimentos) == (
            quantize.D018_SOURCE_KINDS
        )

    def test_con_el_qat_del_baseline_se_mantiene(self, experimentos) -> None:
        """El caso de siempre, ahora cruzado con el registro."""
        assert resolve_d018_pair(_informe("baseline"), QAT_BASELINE, experimentos) == (
            quantize.D018_SOURCE_KINDS
        )

    def test_con_un_qat_sobre_podado_se_rechaza(self, experimentos) -> None:
        with pytest.raises(PtqError, match="no de un baseline"):
            resolve_d018_pair(_informe("baseline"), QAT_PODADO, experimentos)


class TestOtrasProcedencias:
    """Solo una rama PTQ sirve de referencia."""

    @pytest.mark.parametrize("procedencia", ["qat", "", None, "otra"])
    def test_se_rechazan(self, experimentos, procedencia) -> None:
        with pytest.raises(PtqError, match="source_kind="):
            resolve_d018_pair({"source_kind": procedencia}, QAT_PODADO, experimentos)


# ===========================================================================
# Con el registro versionado y los informes que hay en disco
# ===========================================================================

RUTA_PTQ_PODADO = Path(
    "results/exports/combined/s_combined_prune-l1global-r50-ptqint8_9067dba2/"
    "s_combined_prune-l1global-r50-ptqint8_9067dba2_combined.json"
)
RUTA_PTQ_BASELINE = Path("results/exports/s_baseline_default_ptq/s_baseline_default_ptq.json")
RUTA_QAT_BASELINE = Path(
    "results/exports/s_qat_int8-noexcl_d4d403fa_qat/s_qat_int8-noexcl_d4d403fa_qat_int8.json"
)


def _leer(relativa: Path) -> dict:
    """Lee un informe del proyecto, u omite el test si no está."""
    ruta = get_project_paths().root / relativa
    if not ruta.is_file():
        pytest.skip(f"no está '{relativa}' (no versionado en este clon)")
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def registro_real() -> list[dict[str, str]]:
    """Las filas reales de ``experiments.csv``."""
    filas = ExperimentRegistry().read_rows("experiments")
    if not any(f["experiment_id"] == QAT_PODADO for f in filas):
        pytest.skip(f"'{QAT_PODADO}' no está inscrito en este registro")
    return filas


class TestConDatosReales:
    """La función de verdad, sobre lo que el paso 18 va a leer."""

    def test_el_qat_del_podado_con_la_ptq_de_su_podado_se_autoriza(self, registro_real) -> None:
        """Lo que va a hacer el comando propuesto, antes de materializar."""
        informe = _leer(RUTA_PTQ_PODADO)
        assert informe["source_kind"] == "pruned"
        assert informe["parent_experiment_id"] == PODADO
        assert resolve_d018_pair(informe, QAT_PODADO, registro_real) == (
            quantize.PRUNED_QAT_SOURCE_KINDS
        )

    def test_el_qat_del_podado_con_la_ptq_del_baseline_se_rechaza(self, registro_real) -> None:
        """Los hashes coinciden; la pareja no."""
        ptq_baseline = _leer(RUTA_PTQ_BASELINE)
        ptq_podado = _leer(RUTA_PTQ_PODADO)
        for runtime in ("onnxruntime", "openvino"):
            assert (
                ptq_baseline["int8_artifacts"][runtime]["config_hash"]
                == ptq_podado["int8_artifacts"][runtime]["config_hash"]
            )
            assert (
                ptq_baseline["int8_artifacts"][runtime]["calibration_hash"]
                == ptq_podado["int8_artifacts"][runtime]["calibration_hash"]
            )
        with pytest.raises(PtqError, match="no de un baseline"):
            resolve_d018_pair(ptq_baseline, QAT_PODADO, registro_real)

    def test_el_qat_del_baseline_sigue_con_su_ptq(self, registro_real) -> None:
        """La campaña de agosto se contrastaría hoy igual que entonces."""
        assert resolve_d018_pair(_leer(RUTA_PTQ_BASELINE), QAT_BASELINE, registro_real) == (
            quantize.D018_SOURCE_KINDS
        )

    @pytest.mark.parametrize("runtime", ["onnxruntime", "openvino"])
    def test_compare_with_qat_acepta_la_pareja_autorizada_y_solo_esa(self, runtime) -> None:
        ptq = _leer(RUTA_PTQ_PODADO)
        qat = _leer(RUTA_QAT_BASELINE)
        referencia = Int8BranchResult(
            source_kind=ptq["source_kind"], int8_artifacts=ptq["int8_artifacts"]
        )
        informe = compare_with_qat(
            referencia,
            qat["int8_artifacts"],
            runtime,
            admitted_pair=quantize.PRUNED_QAT_SOURCE_KINDS,
        )
        assert informe["same_path"]
        assert informe["config_hash"] == "6e6494c461cbdd63"
        assert informe["calibration_hash"] == "a27c6f11fb1865fd"
        with pytest.raises(
            MaterializationMismatchError, match="la comparación admitida es " "exactamente"
        ):
            compare_with_qat(referencia, qat["int8_artifacts"], runtime)
