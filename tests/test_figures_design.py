"""Tests de las figuras del capítulo 4 (diseño del estudio)."""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.reporting import figures_design as fd  # noqa: E402
from tfm_edge.reporting.style import apply_style  # noqa: E402

RAIZ = get_project_paths().root
EVIDENCIA = RAIZ / "docs" / "evidencia"


# ---------------------------------------------------------------------------
# Cadena QAT
# ---------------------------------------------------------------------------


class TestCadenaQat:
    """La forma que tiene que cumplir el diagrama conceptual."""

    def test_la_cadena_del_proyecto_es_coherente(self) -> None:
        """La tupla que se dibuja pasa su propia guarda."""
        fd.verify_cause_chain(fd.QAT_CAUSE_CHAIN)

    def test_cada_campana_una_entrada_y_la_reproduccion_hereda_la_suya(self) -> None:
        etapas = {e.key: e for e in fd.QAT_STAGES}
        entradas_de: dict[str, set[str]] = {}
        campanas_de: dict[str, set[int]] = {}
        for causa in fd.QAT_CAUSE_CHAIN:
            entradas_de.setdefault(causa.stage, set()).add(causa.decision)
            if etapas[causa.stage].campaign is not None:
                campanas_de.setdefault(causa.decision, set()).add(etapas[causa.stage].campaign)

        assert all(len(entradas) == 1 for entradas in entradas_de.values())
        assert all(len(campanas) == 1 for campanas in campanas_de.values())
        for previa, etapa in zip(fd.QAT_STAGES, fd.QAT_STAGES[1:]):
            if etapa.campaign is None:
                assert entradas_de[etapa.key] == entradas_de[previa.key], etapa.key
        assert {c.number: etapas[c.stage].label for c in fd.QAT_CAUSE_CHAIN} == {
            1: "Campaña 1",
            2: "Reproducción en local",
            3: "Reproducción en local",
            4: "Campaña 2",
            5: "Campaña 3",
            6: "Campaña 4",
        }

    def test_rechaza_una_etapa_que_retrocede(self) -> None:
        cadena = list(fd.QAT_CAUSE_CHAIN)
        cadena[4] = replace(cadena[4], stage="campana_1")
        with pytest.raises(fd.DesignFigureError, match="avanzan de una en una"):
            fd.verify_cause_chain(cadena)

    def test_rechaza_una_etapa_que_se_salta(self) -> None:
        """Pasar de la campaña 2 a la 4 dejaría la 3 sin eslabón."""
        cadena = list(fd.QAT_CAUSE_CHAIN)
        cadena[4] = replace(cadena[4], stage="campana_4")
        with pytest.raises(fd.DesignFigureError, match="avanzan de una en una"):
            fd.verify_cause_chain(cadena)

    def test_rechaza_una_causa_sin_etapa(self) -> None:
        cadena = list(fd.QAT_CAUSE_CHAIN)
        cadena[2] = replace(cadena[2], stage="reproduccion_2")
        with pytest.raises(fd.DesignFigureError, match="sin etapa declarada"):
            fd.verify_cause_chain(cadena)

    def test_rechaza_una_etapa_vacia(self) -> None:
        """Una etapa sin causas dibujaría una banda vacía."""
        cadena = list(fd.QAT_CAUSE_CHAIN)
        cadena[5] = replace(cadena[5], stage="campana_3")
        with pytest.raises(fd.DesignFigureError, match="sin causas"):
            fd.verify_cause_chain(cadena)

    def test_rechaza_la_reproduccion_local_antes_de_ninguna_campana(self) -> None:
        etapas = (fd.QAT_STAGES[1], fd.QAT_STAGES[0], *fd.QAT_STAGES[2:])
        cadena = list(fd.QAT_CAUSE_CHAIN)
        with pytest.raises(fd.DesignFigureError, match="bien formadas"):
            fd.verify_cause_chain(cadena, etapas)

    def test_rechaza_campanas_mal_numeradas(self) -> None:
        etapas = (*fd.QAT_STAGES[:-1], replace(fd.QAT_STAGES[-1], campaign=5))
        with pytest.raises(fd.DesignFigureError, match="bien formadas"):
            fd.verify_cause_chain(fd.QAT_CAUSE_CHAIN, etapas)

    def test_rechaza_una_etiqueta_larga(self) -> None:
        cadena = list(fd.QAT_CAUSE_CHAIN)
        cadena[5] = replace(cadena[5], label="Errata en el\nregistro")
        with pytest.raises(fd.DesignFigureError, match="tres palabras"):
            fd.verify_cause_chain(cadena)


# Tomas de la medida de latencia.


def _volcado(
    ruta: Path,
    repeticiones: list[list[float]],
    *,
    inicio: str = "2026-08-17T19:24:03+00:00",
    artefacto: str = "results/exports/x.onnx",
    en_reposo: bool = True,
    media: float | None = None,
    dispersion: float | None = None,
) -> Path:
    """Escribe un volcado sintético con la forma del protocolo de medida."""
    todos = [t * 1000 for r in repeticiones for t in r]
    medianas = [statistics.median(r) * 1000 for r in repeticiones]
    datos: dict[str, Any] = {
        "device_id": "workstation_cpu",
        "backend": "onnxruntime",
        "precision": "int8",
        "started_at": inicio,
        "latency": [{"timings_s": r} for r in repeticiones],
        "aggregated": {"latency_mean_ms": statistics.fmean(todos) if media is None else media},
        "between_run_std_ms": statistics.stdev(medianas) if dispersion is None else dispersion,
        "artifact": {"path": artefacto},
        "system_load": {"acceptable": en_reposo, "forced": False},
    }
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


REPS_A = [[0.060, 0.062, 0.064], [0.061, 0.066, 0.063], [0.065, 0.067, 0.069]]
REPS_B = [[0.070, 0.074, 0.072], [0.071, 0.075, 0.073], [0.066, 0.068, 0.067]]


class TestTomasDeLatencia:
    """Lectura de las tomas y la guarda que las declara comparables."""

    def test_lee_el_vector_y_lo_que_publica(self, tmp_path: Path) -> None:
        """El vector entero, las medianas por repetición y lo publicado."""
        toma = fd.load_latency_take(_volcado(tmp_path / "a.json", REPS_A), "a")
        assert len(toma.values_ms) == 9
        assert len(toma.repetition_medians_ms) == 3
        assert toma.idle

    def test_rechaza_una_media_que_no_sale_del_vector(self, tmp_path: Path) -> None:
        """La figura no puede marcar una media distinta de la del vector que dibuja."""
        ruta = _volcado(tmp_path / "a.json", REPS_A, media=99.0)
        with pytest.raises(fd.DesignFigureError, match="media publicada"):
            fd.load_latency_take(ruta, "a")

    def test_rechaza_otra_definicion_de_dispersion(self, tmp_path: Path) -> None:
        """Si la dispersión no es la de las medianas, el rótulo de la barra mentiría."""
        ruta = _volcado(tmp_path / "a.json", REPS_A, dispersion=0.5)
        with pytest.raises(fd.DesignFigureError, match="medianas por repetición"):
            fd.load_latency_take(ruta, "a")

    def test_rechaza_un_volcado_sin_vectores(self, tmp_path: Path) -> None:
        """Esta figura dibuja los valores; el resumen no basta."""
        ruta = tmp_path / "a.json"
        ruta.write_text(json.dumps({"latency": [], "aggregated": {}}), encoding="utf-8")
        with pytest.raises(fd.DesignFigureError, match="vectores"):
            fd.load_latency_take(ruta, "a")

    def test_acepta_dos_tomas_seguidas_del_mismo_artefacto(self, tmp_path: Path) -> None:
        a = fd.load_latency_take(_volcado(tmp_path / "a.json", REPS_A), "a")
        b = fd.load_latency_take(
            _volcado(tmp_path / "b.json", REPS_B, inicio="2026-08-17T19:29:32+00:00"), "b"
        )
        fd.verify_comparable_takes(a, b)

    def test_rechaza_la_remedida_de_d136(self, tmp_path: Path) -> None:
        a = fd.load_latency_take(_volcado(tmp_path / "a.json", REPS_A), "a")
        b = fd.load_latency_take(
            _volcado(tmp_path / "b.json", REPS_B, inicio="2026-09-02T22:12:04+00:00"), "b"
        )
        with pytest.raises(fd.DesignFigureError, match="separadas"):
            fd.verify_comparable_takes(a, b)

    def test_rechaza_otro_artefacto(self, tmp_path: Path) -> None:
        a = fd.load_latency_take(_volcado(tmp_path / "a.json", REPS_A), "a")
        b = fd.load_latency_take(
            _volcado(
                tmp_path / "b.json",
                REPS_B,
                inicio="2026-08-17T19:29:32+00:00",
                artefacto="results/exports/otro.onnx",
            ),
            "b",
        )
        with pytest.raises(fd.DesignFigureError, match="artifact_path"):
            fd.verify_comparable_takes(a, b)

    def test_rechaza_una_toma_sin_reposo(self, tmp_path: Path) -> None:
        """El título afirma que la máquina estaba en reposo."""
        a = fd.load_latency_take(_volcado(tmp_path / "a.json", REPS_A), "a")
        b = fd.load_latency_take(
            _volcado(
                tmp_path / "b.json", REPS_B, inicio="2026-08-17T19:29:32+00:00", en_reposo=False
            ),
            "b",
        )
        with pytest.raises(fd.DesignFigureError, match="reposo"):
            fd.verify_comparable_takes(a, b)


class TestLasTomasReales:
    """Las guardas, ejecutadas sobre la evidencia de verdad."""

    def test_las_dos_tomas_de_d095_son_comparables(self) -> None:
        """Si alguien sustituye uno de los dos ficheros, esto deja de pasar."""
        primera = fd.load_latency_take(
            EVIDENCIA / "d095_toma_descartada_qat_onnxruntime_int8.json", "1.ª"
        )
        segunda = fd.load_latency_take(
            EVIDENCIA / "d095_toma_registrada_qat_onnxruntime_int8.json", "2.ª"
        )
        fd.verify_comparable_takes(primera, segunda)
        assert segunda.started_at - primera.started_at < timedelta(minutes=10)
        assert len(primera.values_ms) == len(segunda.values_ms)

    def test_el_volcado_de_results_no_es_la_segunda_toma(self) -> None:
        primera = fd.load_latency_take(
            EVIDENCIA / "d095_toma_descartada_qat_onnxruntime_int8.json", "1.ª"
        )
        remedida = fd.load_latency_take(
            RAIZ
            / "results"
            / "raw"
            / "benchmark"
            / "workstation_cpu_s_qat_int8-noexcl_d4d403fa_onnxruntime_int8_640.json",
            "remedida",
        )
        with pytest.raises(fd.DesignFigureError, match="separadas"):
            fd.verify_comparable_takes(primera, remedida)


# ---------------------------------------------------------------------------
# Matriz de cobertura
# ---------------------------------------------------------------------------


def _perfiles(**invalidos: bool) -> dict[str, SimpleNamespace]:
    """Perfiles mínimos de la rejilla; los nombrados quedan no válidos para medir."""
    return {
        p: SimpleNamespace(valid_for_measurement=not invalidos.get(p, False))
        for p in fd.COVERAGE_PROFILES
    }


def _fila(dispositivo: str, motor: str, estado: str, nota: str = "") -> dict[str, str]:
    return {"device_id": dispositivo, "runtime": motor, "status": estado, "notes": nota}


class TestReglasDeCobertura:
    """El orden de las reglas, que es lo que decide cada celda."""

    def test_sin_filas_todo_es_no_ejecutado_salvo_lo_imposible(self) -> None:
        """Sin registro, lo único que no es «no ejecutado» es lo que no puede existir."""
        celdas = fd.coverage_matrix([], _perfiles())
        categorias = {par: c.category for par, c in celdas.items()}
        assert {par for par, cat in categorias.items() if cat == "no_aplicable"} == set(
            fd.NOT_APPLICABLE
        )
        assert all(
            cat == "no_ejecutado" for par, cat in categorias.items() if par not in fd.NOT_APPLICABLE
        )

    def test_medido_cuenta_sus_filas(self) -> None:
        """Una celda medida sabe cuántas filas la sostienen."""
        filas = [_fila("workstation_cpu", "onnxruntime", "ok")] * 3
        celdas = fd.coverage_matrix(filas, _perfiles())
        assert celdas[("workstation_cpu", "onnxruntime")].category == "medido"
        assert celdas[("workstation_cpu", "onnxruntime")].measured_rows == 3

    def test_excluido_gana_a_no_aplicable(self) -> None:
        """En cloud_gpu la exclusión es de la fila entera, incluido OpenVINO."""
        celdas = fd.coverage_matrix([], _perfiles(cloud_gpu=True))
        assert all(celdas[("cloud_gpu", m)].category == "excluido" for m in fd.COVERAGE_ENGINES)

    def test_no_aplicable_gana_a_no_disponible_de_motor(self) -> None:
        """La fila de TensorRT es de columna, pero en una CPU TensorRT no puede estar."""
        filas = [_fila("", "tensorrt", "no_disponible", "nunca instalado")]
        celdas = fd.coverage_matrix(filas, _perfiles())
        assert celdas[("workstation_cpu", "tensorrt")].category == "no_aplicable"
        assert celdas[("edge_gpu_jetson", "tensorrt")].category == "no_disponible"

    def test_no_disponible_de_perfil_cubre_sus_motores_aplicables(self) -> None:
        """Una fila de perfil vale para todos sus motores, menos los imposibles."""
        filas = [_fila("workstation_gpu", "", "no_disponible", "torch +cpu")]
        celdas = fd.coverage_matrix(filas, _perfiles())
        assert celdas[("workstation_gpu", "pytorch")].category == "no_disponible"
        assert celdas[("workstation_gpu", "pytorch")].reason == "torch +cpu"
        assert celdas[("workstation_gpu", "openvino")].category == "no_aplicable"
        assert celdas[("workstation_gpu", "tflite")].category == "no_aplicable"

    def test_una_medida_fuera_de_la_rejilla_es_un_error(self) -> None:
        """Una figura de cobertura no puede perder en silencio algo que se midió."""
        filas = [_fila("otro_dispositivo", "pytorch", "ok")]
        with pytest.raises(fd.DesignFigureError, match="fuera de la rejilla"):
            fd.coverage_matrix(filas, _perfiles())

    def test_una_ausencia_fuera_de_la_rejilla_es_un_error(self) -> None:
        """Lo mismo para una ausencia declarada de un motor que la figura no tiene."""
        filas = [_fila("", "coreml", "no_disponible")]
        with pytest.raises(fd.DesignFigureError, match="fuera de la rejilla"):
            fd.coverage_matrix(filas, _perfiles())

    def test_falta_un_perfil(self) -> None:
        """Sin el YAML no se sabe si el perfil está excluido."""
        perfiles = _perfiles()
        del perfiles["cloud_gpu"]
        with pytest.raises(fd.DesignFigureError, match="sin YAML"):
            fd.coverage_matrix([], perfiles)


class TestCoberturaSobreElRegistroReal:
    """La matriz construida con el registro y los perfiles de verdad."""

    def test_ninguna_medida_se_pierde(self) -> None:
        """Las filas «ok» que suma la matriz son todas las del registro."""
        from tfm_edge.benchmark.profiles import load_all_profiles

        filas = ExperimentRegistry().read_rows("deployment_metrics")
        celdas = fd.coverage_matrix(filas, load_all_profiles(RAIZ / "configs" / "benchmark"))
        assert len(celdas) == len(fd.COVERAGE_PROFILES) * len(fd.COVERAGE_ENGINES)
        assert sum(c.measured_rows for c in celdas.values()) == sum(
            1 for f in filas if f["status"] == "ok"
        )

    def test_la_celda_de_h1_esta_vacia(self) -> None:
        from tfm_edge.benchmark.profiles import load_all_profiles

        filas = ExperimentRegistry().read_rows("deployment_metrics")
        celdas = fd.coverage_matrix(filas, load_all_profiles(RAIZ / "configs" / "benchmark"))
        assert celdas[fd.H1_CELL].category == "no_disponible"


# ---------------------------------------------------------------------------
# Nombres de fichero
# ---------------------------------------------------------------------------


class TestPrefijos:
    """Capítulo 4 y el apartado de cada figura."""

    def test_cadena_va_al_apartado_4_2(self, tmp_path: Path) -> None:
        apply_style()
        rutas = fd.plot_qat_cause_chain(tmp_path)
        assert rutas["pdf"].name == "fig_4_2_cadena_causas_qat.pdf"
        assert rutas["png"].is_file()

    def test_tomas_van_al_apartado_4_3(self, tmp_path: Path) -> None:
        apply_style()
        a = fd.load_latency_take(_volcado(tmp_path / "a.json", REPS_A), "a")
        b = fd.load_latency_take(
            _volcado(tmp_path / "b.json", REPS_B, inicio="2026-08-17T19:29:32+00:00"), "b"
        )
        rutas = fd.plot_latency_error_floor(a, b, tmp_path)
        assert rutas["pdf"].name == "fig_4_3_suelo_error_latencia.pdf"

    def test_cobertura_va_al_apartado_4_5(self, tmp_path: Path) -> None:
        apply_style()
        celdas = fd.coverage_matrix([], _perfiles())
        rutas = fd.plot_coverage_matrix(celdas, tmp_path)
        assert rutas["pdf"].name == "fig_4_5_cobertura_perfiles_motores.pdf"

    def test_sesgo_va_al_apartado_4_1(self, tmp_path: Path) -> None:
        apply_style()
        brazos = fd.load_selection_bias(D171, EVIDENCIA)
        rutas = fd.plot_selection_bias(brazos, tmp_path)
        assert rutas["pdf"].name == "fig_4_1_sesgo_seleccion_best_pt.pdf"

    def test_flujo_va_al_apartado_4_3(self, tmp_path: Path) -> None:
        apply_style()
        rutas = fd.plot_experimental_flow(tmp_path, RAIZ / "scripts")
        assert rutas["pdf"].name == "fig_4_3_flujo_experimental.pdf"


# Sesgo de selección de best.pt.

D171 = EVIDENCIA / "d171_sesgo_de_seleccion.json"


def _copia_evidencia(destino: Path) -> Path:
    """Copia el JSON del sesgo de selección y sus dos curvas a un directorio desechable."""
    for origen in [D171, *sorted(EVIDENCIA.glob("d171_results_*.csv"))]:
        (destino / origen.name).write_bytes(origen.read_bytes())
    return destino / D171.name


def _reescribe_curva(ruta: Path, epoca: int, valor: str) -> None:
    """Cambia el mAP de Ultralytics de una época en un results.csv."""
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    cabecera = [c.strip() for c in lineas[0].split(",")]
    columna = cabecera.index(fd.ULTRALYTICS_MAP_COLUMN)
    for i, linea in enumerate(lineas[1:], start=1):
        campos = linea.split(",")
        if int(float(campos[0])) == epoca:
            campos[columna] = valor
            lineas[i] = ",".join(campos)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")


class TestLaMismaCurvaEnDosSitios:
    """La curva de cada brazo está versionada dos veces, y tienen que coincidir."""

    @pytest.mark.parametrize(
        "experimento",
        [
            "n_distillation_l2n_control_alpha0_f5c4a70e",
            "n_distillation_l2n_combined_t2_ac9ff009",
        ],
    )
    def test_las_dos_copias_son_el_mismo_fichero(self, experimento: str) -> None:
        en_evidencia = fd.selection_curve_path(EVIDENCIA, experimento)
        en_modelos = RAIZ / "models" / "distillation" / experimento / "results.csv"

        assert en_evidencia.is_file(), f"Falta la copia de la figura 4.1: {en_evidencia}"
        assert en_modelos.is_file(), (
            f"Falta la copia del inventario del capítulo 5: {en_modelos}. "
            "Se versionó con git add -f; ver la tercera excepción del .gitignore."
        )
        assert en_evidencia.read_bytes() == en_modelos.read_bytes(), (
            f"Las dos copias de la curva de {experimento} han divergido. "
            "Una de las dos figuras está dibujando otro entrenamiento."
        )


class TestSesgoDeSeleccionReal:
    """Las dos fuentes de la figura, contrastadas sobre la evidencia versionada."""

    def test_las_curvas_reales_cuadran_con_d171(self) -> None:
        brazos = fd.load_selection_bias(D171, EVIDENCIA)
        assert [(b.best_epoch, b.last_epoch) for b in brazos] == [(49, 50), (42, 50)]
        assert all(len(b.epochs) == 50 for b in brazos)

    def test_la_prima_es_la_de_d171(self) -> None:
        """Control +0,000146 y variante −0,000790, con pycocotools."""
        control, variante = fd.load_selection_bias(D171, EVIDENCIA)
        assert control.selection_premium == pytest.approx(0.000146, abs=1e-6)
        assert variante.selection_premium == pytest.approx(-0.000790, abs=1e-6)

    def test_rechaza_una_curva_con_otro_maximo(self, tmp_path: Path) -> None:
        """Si otra época supera a la del best.pt, la estrella caería mal."""
        json_ = _copia_evidencia(tmp_path)
        curva = fd.selection_curve_path(tmp_path, "n_distillation_l2n_combined_t2_ac9ff009")
        _reescribe_curva(curva, 30, "0.40000")
        with pytest.raises(fd.DesignFigureError, match="máximo"):
            fd.load_selection_bias(json_, tmp_path)

    def test_rechaza_un_empate_en_el_maximo(self, tmp_path: Path) -> None:
        json_ = _copia_evidencia(tmp_path)
        curva = fd.selection_curve_path(tmp_path, "n_distillation_l2n_control_alpha0_f5c4a70e")
        _reescribe_curva(curva, 47, "0.33278")
        with pytest.raises(fd.DesignFigureError, match="máximo"):
            fd.load_selection_bias(json_, tmp_path)

    def test_rechaza_un_valor_de_ultralytics_distinto(self, tmp_path: Path) -> None:
        json_ = _copia_evidencia(tmp_path)
        curva = fd.selection_curve_path(tmp_path, "n_distillation_l2n_control_alpha0_f5c4a70e")
        _reescribe_curva(curva, 50, "0.33240")
        with pytest.raises(fd.DesignFigureError, match="mismo validador"):
            fd.load_selection_bias(json_, tmp_path)

    def test_rechaza_una_curva_incompleta(self, tmp_path: Path) -> None:
        """Sin una época, el argmax contrastado podría no ser el del entrenamiento."""
        json_ = _copia_evidencia(tmp_path)
        curva = fd.selection_curve_path(tmp_path, "n_distillation_l2n_control_alpha0_f5c4a70e")
        lineas = curva.read_text(encoding="utf-8").splitlines()
        curva.write_text("\n".join(lineas[:10] + lineas[11:]) + "\n", encoding="utf-8")
        with pytest.raises(fd.DesignFigureError, match="consecutivas"):
            fd.load_selection_bias(json_, tmp_path)

    def test_rechaza_otro_evaluador(self, tmp_path: Path) -> None:
        """La figura rotula pycocotools; el JSON tiene que decir lo mismo."""
        json_ = _copia_evidencia(tmp_path)
        datos = json.loads(json_.read_text(encoding="utf-8"))
        datos["protocolo"]["evaluador"] = "ultralytics"
        json_.write_text(json.dumps(datos), encoding="utf-8")
        with pytest.raises(fd.DesignFigureError, match="pycocotools"):
            fd.load_selection_bias(json_, tmp_path)

    def test_sin_curvas_falla(self, tmp_path: Path) -> None:
        """Sin los results.csv extraídos no hay figura."""
        (tmp_path / D171.name).write_bytes(D171.read_bytes())
        with pytest.raises(fd.DesignFigureError, match="No existe la curva"):
            fd.load_selection_bias(tmp_path / D171.name, tmp_path)


# ---------------------------------------------------------------------------
# Flujo experimental
# ---------------------------------------------------------------------------


class TestFlujoContraElCodigo:
    """Lo que el diagrama afirma de quién escribe en el registro."""

    def test_el_diagrama_cuadra_con_scripts(self) -> None:
        ficheros = fd.verify_flow_against_scripts(RAIZ / "scripts")
        assert set(ficheros) == set(fd.FLOW_STEPS)

    def test_rechaza_un_paso_que_no_escribe_donde_dice(self, tmp_path: Path) -> None:
        """Si 11 dejara de escribir la calidad, el diagrama mentiría."""
        for fichero in (RAIZ / "scripts").glob("[0-9][0-9]_*.py"):
            (tmp_path / fichero.name).write_bytes(fichero.read_bytes())
        paso_11 = next(tmp_path.glob("11_*.py"))
        paso_11.write_text(
            paso_11.read_text(encoding="utf-8").replace("append_quality_metrics(", "x("),
            encoding="utf-8",
        )
        with pytest.raises(fd.DesignFigureError, match="no escribe en quality_metrics"):
            fd.verify_flow_against_scripts(tmp_path)

    def test_rechaza_un_paso_renumerado(self, tmp_path: Path) -> None:
        """Un paso citado que no existe para la figura."""
        for fichero in (RAIZ / "scripts").glob("[0-9][0-9]_*.py"):
            if not fichero.name.startswith("07_"):
                (tmp_path / fichero.name).write_bytes(fichero.read_bytes())
        with pytest.raises(fd.DesignFigureError, match="paso 07"):
            fd.verify_flow_against_scripts(tmp_path)


# Formato de la memoria.

FIGURAS_DEL_CAPITULO = (
    "sesgo_seleccion_best_pt",
    "cadena_causas_qat",
    "flujo_experimental",
    "suelo_error_latencia",
    "cobertura_perfiles_motores",
)

#: Las que tienen un solo panel: ahí el título de los ejes sería el título
#: general de la figura, y tampoco puede estar.
UN_SOLO_PANEL = ("cadena_causas_qat", "flujo_experimental", "cobertura_perfiles_motores")


@pytest.fixture(scope="class")
def figuras_reales(tmp_path_factory) -> Any:
    """Las cinco figuras, dibujadas sobre el registro y la evidencia reales."""
    import matplotlib.pyplot as plt

    from tfm_edge.benchmark.profiles import load_all_profiles

    destino = tmp_path_factory.mktemp("figuras_capitulo_4")
    original = fd.save_figure
    capturadas: dict[str, tuple[Any, dict[str, Path]]] = {}

    def guardar(figura, output_dir, capitulo, apartado, descripcion, close=True):
        rutas = original(figura, destino, capitulo, apartado, descripcion, close=False)
        capturadas[descripcion] = (figura, rutas)
        return rutas

    with pytest.MonkeyPatch.context() as parche:
        parche.setattr(fd, "save_figure", guardar)
        fd.generate_design_figures(
            destino,
            ExperimentRegistry().read_rows("deployment_metrics"),
            load_all_profiles(RAIZ / "configs" / "benchmark"),
            EVIDENCIA / "d095_toma_descartada_qat_onnxruntime_int8.json",
            EVIDENCIA / "d095_toma_registrada_qat_onnxruntime_int8.json",
            D171,
            RAIZ / "scripts",
        )
    yield capturadas
    plt.close("all")


def _textos_dibujados(figura: Any) -> list[Any]:
    """Los textos que de verdad salen en la figura."""
    from matplotlib.text import Text

    figura.canvas.draw()
    fuera = set()
    for eje in figura.axes:
        for eje_xy in (eje.xaxis, eje.yaxis):
            minimo, maximo = sorted(eje_xy.get_view_interval())
            margen = (maximo - minimo) * 1e-9
            for marca in eje_xy.get_major_ticks() + eje_xy.get_minor_ticks():
                if (
                    not eje.axison
                    or not eje_xy.get_visible()
                    or not minimo - margen <= marca.get_loc() <= maximo + margen
                ):
                    fuera.update((id(marca.label1), id(marca.label2)))
    return [
        t
        for t in figura.findobj(Text)
        if t.get_visible() and t.get_text().strip() and id(t) not in fuera
    ]


def _poligono(texto: Any, dibujo: Any) -> list[tuple[float, float]]:
    """Las cuatro esquinas del texto en pantalla, girado si lo está."""
    import math as m

    caja = texto.get_window_extent(dibujo)
    giro = texto.get_rotation()
    if giro % 90 == 0:
        return [(caja.x0, caja.y0), (caja.x1, caja.y0), (caja.x1, caja.y1), (caja.x0, caja.y1)]
    texto.set_rotation(0)
    recto = texto.get_window_extent(dibujo)
    texto.set_rotation(giro)
    cx, cy, c, s = recto.x0, recto.y0, m.cos(m.radians(giro)), m.sin(m.radians(giro))
    esquinas = [
        (cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c)
        for x, y in (
            (recto.x0, recto.y0),
            (recto.x1, recto.y0),
            (recto.x1, recto.y1),
            (recto.x0, recto.y1),
        )
    ]
    dx = caja.x0 - min(x for x, _ in esquinas)
    dy = caja.y0 - min(y for _, y in esquinas)
    return [(x + dx, y + dy) for x, y in esquinas]


def _se_solapan(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """Dos polígonos convexos se solapan si ningún eje de sus lados los separa."""
    for poligono in (a, b):
        for (x1, y1), (x2, y2) in zip(poligono, poligono[1:] + poligono[:1]):
            nx, ny = y2 - y1, x1 - x2
            pa = [nx * x + ny * y for x, y in a]
            pb = [nx * x + ny * y for x, y in b]
            if max(pa) <= min(pb) or max(pb) <= min(pa):
                return False
    return True


def _trazos(eje: Any) -> list[tuple[str, Any, Any, float]]:
    """Lo que dibujan unos ejes como trazo o contorno, en coordenadas de pantalla."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba_array
    from matplotlib.path import Path as Camino
    from matplotlib.text import Annotation

    escala = eje.figure.dpi / 72.0

    def recorte(artista: Any) -> Any:
        return artista.get_clip_box() if artista.get_clip_on() else None

    def grosor_borde(parche: Any) -> float:
        visible = to_rgba_array(parche.get_edgecolor())[:, 3].max() > 0
        return parche.get_linewidth() * escala if visible else 0.0

    caminos: list[tuple[str, Any, Any, float]] = []
    for linea in eje.get_lines():
        if linea.get_linestyle() not in ("None", "none", "") and len(linea.get_xydata()) > 1:
            camino = linea.get_transform().transform_path(linea.get_path())
            caminos.append(("línea", camino, recorte(linea), linea.get_linewidth() * escala))
    for coleccion in eje.collections:
        if isinstance(coleccion, LineCollection):
            grosor = max(coleccion.get_linewidths(), default=0.0) * escala
            caminos += [
                (
                    "segmento",
                    coleccion.get_transform().transform_path(p),
                    recorte(coleccion),
                    grosor,
                )
                for p in coleccion.get_paths()
            ]
    caminos += [
        (type(p).__name__, p.get_path().transformed(p.get_transform()), recorte(p), grosor_borde(p))
        for p in eje.patches
    ]
    caminos += [
        (
            "flecha",
            t.arrow_patch.get_path().transformed(t.arrow_patch.get_transform()),
            recorte(t.arrow_patch),
            t.arrow_patch.get_linewidth() * escala,
        )
        for t in eje.texts
        if isinstance(t, Annotation) and t.arrow_patch is not None
    ]
    return [
        (tipo, Camino(poligono), caja, grosor)
        for tipo, camino, caja, grosor in caminos
        for poligono in camino.to_polygons(closed_only=False)
        if len(poligono) > 1
    ]


def _cruza(camino: Any, recorte: Any, caja: Any, grosor: float = 0.0) -> bool:
    """Si la parte visible de un trazo, con su grosor, pasa por la caja de un rótulo."""
    from matplotlib.transforms import Bbox

    medio = grosor / 2
    caja = Bbox.from_extents(caja.x0 - medio, caja.y0 - medio, caja.x1 + medio, caja.y1 + medio)
    if recorte is not None:
        caja = Bbox.intersection(caja, recorte)
        if caja is None:
            return False
    return camino.intersects_bbox(caja, filled=False)


def _ancho_png_cm(ruta: Path) -> float:
    """Ancho real del PNG guardado, en centímetros."""
    from PIL import Image

    with Image.open(ruta) as imagen:
        return imagen.size[0] / imagen.info.get("dpi", (300, 300))[0] * 2.54


class TestElMedidor:
    """Control negativo: las medidas de abajo detectan lo que dicen detectar."""

    @staticmethod
    def _eje() -> tuple[Any, Any]:
        import matplotlib.pyplot as plt

        figura, eje = plt.subplots(figsize=(4, 3))
        eje.set_xlim(0, 10)
        eje.set_ylim(0, 10)
        return figura, eje

    def test_un_rotulo_sobre_una_linea_cruza_y_al_lado_no(self) -> None:
        import matplotlib.pyplot as plt

        figura, eje = self._eje()
        eje.plot([0, 10], [5, 5])
        encima = eje.text(5, 5, "encima", ha="center", va="center")
        lejos = eje.text(5, 8, "lejos", ha="center", va="center")
        figura.canvas.draw()
        dibujo = figura.canvas.get_renderer()
        trazos = _trazos(eje)
        assert any(_cruza(c, r, encima.get_window_extent(dibujo), g) for _, c, r, g in trazos)
        assert not any(_cruza(c, r, lejos.get_window_extent(dibujo), g) for _, c, r, g in trazos)
        plt.close(figura)

    def test_lo_recortado_fuera_de_los_ejes_no_cuenta(self) -> None:
        """La línea sigue hasta y = -20, fuera del eje; el rótulo de debajo no la ve."""
        import matplotlib.pyplot as plt

        figura, eje = self._eje()
        eje.plot([5, 5], [5, -20])
        debajo = figura.text(0.5, 0.01, "debajo", ha="center", va="bottom")
        figura.canvas.draw()
        caja = debajo.get_window_extent(figura.canvas.get_renderer())
        assert not any(_cruza(c, r, caja, g) for _, c, r, g in _trazos(eje))
        plt.close(figura)

    def test_un_rotulo_que_solo_toca_el_grosor_de_un_marco_cruza(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        figura, eje = self._eje()
        eje.add_patch(Rectangle((2, 2), 6, 6, fill=False, edgecolor="black", linewidth=8))
        rotulo = eje.text(2.1, 7.9, "H1", ha="left", va="top")
        figura.canvas.draw()
        caja = rotulo.get_window_extent(figura.canvas.get_renderer())
        trazos = _trazos(eje)
        assert not any(_cruza(c, r, caja) for _, c, r, _ in trazos), "sin grosor no se ve"
        assert any(_cruza(c, r, caja, g) for _, c, r, g in trazos)
        plt.close(figura)

    def test_dos_rotulos_girados_contiguos_no_se_pisan_y_superpuestos_si(self) -> None:
        import matplotlib.pyplot as plt

        figura, eje = self._eje()
        opciones = {"rotation": 30, "ha": "right", "rotation_mode": "anchor"}
        a = eje.text(3, 5, "onnxruntime", **opciones)
        b = eje.text(4, 5, "openvino", **opciones)
        c = eje.text(3.05, 5, "pytorch", **opciones)
        figura.canvas.draw()
        dibujo = figura.canvas.get_renderer()
        assert a.get_window_extent(dibujo).overlaps(b.get_window_extent(dibujo))
        assert not _se_solapan(_poligono(a, dibujo), _poligono(b, dibujo))
        assert _se_solapan(_poligono(a, dibujo), _poligono(c, dibujo))
        plt.close(figura)


class TestFormatoDeLaMemoria:
    """El estándar del capítulo 1, aplicado a las cinco figuras del 4."""

    def test_salen_las_cinco(self, figuras_reales) -> None:
        assert set(figuras_reales) == set(FIGURAS_DEL_CAPITULO)

    @pytest.mark.parametrize("nombre", FIGURAS_DEL_CAPITULO)
    def test_16_cm_y_7_pt(self, figuras_reales, nombre: str) -> None:
        """Ancho de la caja de texto, también en el PNG ya recortado, y nada fuera."""
        figura, rutas = figuras_reales[nombre]
        textos = _textos_dibujados(figura)
        dibujo = figura.canvas.get_renderer()

        assert figura.get_figwidth() == pytest.approx(16.0 / 2.54)
        assert _ancho_png_cm(rutas["png"]) <= 16.0 + 1e-6
        assert min(t.get_fontsize() for t in textos) >= 7.0
        for texto in textos:
            caja = texto.get_window_extent(dibujo)
            assert figura.bbox.x0 - 1 <= caja.x0 and caja.x1 <= figura.bbox.x1 + 1, texto.get_text()
            assert figura.bbox.y0 - 1 <= caja.y0 and caja.y1 <= figura.bbox.y1 + 1, texto.get_text()

    @pytest.mark.parametrize("nombre", FIGURAS_DEL_CAPITULO)
    def test_sin_titulo_general_notas_ni_tablas(self, figuras_reales, nombre: str) -> None:
        import re

        figura, _ = figuras_reales[nombre]

        assert figura.get_suptitle() == ""
        assert not figura.texts
        assert not any(eje.tables for eje in figura.axes)
        if nombre in UN_SOLO_PANEL:
            assert not any(eje.get_title() for eje in figura.axes)
        internos = [
            t.get_text() for t in _textos_dibujados(figura) if re.search(r"D-\d{3}", t.get_text())
        ]
        assert not internos

    @pytest.mark.parametrize("nombre", FIGURAS_DEL_CAPITULO)
    def test_ningun_texto_pisa_otro(self, figuras_reales, nombre: str) -> None:
        figura, _ = figuras_reales[nombre]
        textos = _textos_dibujados(figura)
        dibujo = figura.canvas.get_renderer()
        poligonos = [(t.get_text(), _poligono(t, dibujo)) for t in textos]

        choques = [
            (a, b)
            for i, (a, pa) in enumerate(poligonos)
            for b, pb in poligonos[i + 1 :]
            if _se_solapan(pa, pb)
        ]
        assert not choques

    @pytest.mark.parametrize("nombre", FIGURAS_DEL_CAPITULO)
    def test_ningun_rotulo_pisa_una_linea(self, figuras_reales, nombre: str) -> None:
        """Rótulos y leyendas frente a líneas, segmentos, flechas y contornos."""
        figura, _ = figuras_reales[nombre]
        _textos_dibujados(figura)
        dibujo = figura.canvas.get_renderer()
        trazos = [trazo for eje in figura.axes for trazo in _trazos(eje)]
        rotulos = [
            (t.get_text(), t.get_window_extent(dibujo))
            for eje in figura.axes
            for t in eje.texts
            if t.get_visible() and t.get_text().strip()
        ]
        leyendas = [eje.get_legend() for eje in figura.axes if eje.get_legend()] + figura.legends
        rotulos += [("leyenda", leyenda.get_window_extent(dibujo)) for leyenda in leyendas]

        choques = [
            (texto, tipo)
            for texto, caja in rotulos
            for tipo, camino, recorte, grosor in trazos
            if _cruza(camino, recorte, caja, grosor)
        ]
        assert not choques
