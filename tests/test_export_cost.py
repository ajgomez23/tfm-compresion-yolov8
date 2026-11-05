"""Tests del coste medido de producir cada artefacto de despliegue."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.common.paths import ProjectPaths, get_project_paths
from tfm_edge.export import cost as cost_mod


def _volcado(
    directorio: Path,
    nombre: str,
    fp32: dict | None = None,
    int8: dict | None = None,
) -> Path:
    """Escribe un volcado de exportación de mentira."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"{nombre}.json"
    ruta.write_text(
        json.dumps({"fp32_exports": fp32 or {}, "int8_artifacts": int8 or {}}),
        encoding="utf-8",
    )
    return ruta


def _paso(ruta: str, segundos: float | str, origen: str | None = None, status: str = "ok") -> dict:
    """Construye la entrada de un paso."""
    entrada: dict = {"path": ruta, "status": status}
    if segundos != "":
        entrada["export_seconds"] = segundos
        entrada["quantize_seconds"] = segundos
    if origen is not None:
        entrada["source_path"] = origen
    return entrada


class TestIndice:
    """El índice ata cada artefacto al paso que lo produjo."""

    def test_indexa_exportaciones_y_cuantizaciones(self, tmp_path: Path) -> None:
        """Los dos grupos del volcado cuestan reloj y los dos entran."""
        _volcado(
            tmp_path,
            "uno",
            fp32={"onnx": _paso("/x/m.onnx", 2.0)},
            int8={"onnxruntime": _paso("/x/m_int8.onnx", 30.0, origen="/x/m.onnx")},
        )

        indice = cost_mod.index_export_steps(tmp_path)

        assert len(indice) == 2
        clases = {paso.kind for paso in indice.values()}
        assert clases == {"export", "quantize"}

    def test_dos_volcados_que_se_contradicen_no_se_resuelven_eligiendo(
        self, tmp_path: Path
    ) -> None:
        _volcado(tmp_path, "uno", fp32={"onnx": _paso("/x/m.onnx", 2.0)})
        _volcado(tmp_path, "dos", fp32={"onnx": _paso("/x/m.onnx", 9.0)})

        with pytest.raises(cost_mod.ExportCostError, match="tiempos"):
            cost_mod.index_export_steps(tmp_path)

    def test_dos_volcados_que_coinciden_no_son_un_problema(self, tmp_path: Path) -> None:
        """Repetir la misma medida no es contradecirse."""
        _volcado(tmp_path, "uno", fp32={"onnx": _paso("/x/m.onnx", 2.0)})
        _volcado(tmp_path, "dos", fp32={"onnx": _paso("/x/m.onnx", 2.0)})

        assert len(cost_mod.index_export_steps(tmp_path)) == 1

    def test_un_volcado_ilegible_no_tumba_el_indice(self, tmp_path: Path) -> None:
        """Un JSON roto se salta con un aviso; los demás siguen contando."""
        _volcado(tmp_path, "bueno", fp32={"onnx": _paso("/x/m.onnx", 2.0)})
        (tmp_path / "roto.json").write_text("{no es json", encoding="utf-8")

        assert len(cost_mod.index_export_steps(tmp_path)) == 1

    def test_un_tiempo_que_no_es_numero_es_un_error(self, tmp_path: Path) -> None:
        """Darlo por ausente borraría la diferencia con «se apuntó mal»."""
        _volcado(tmp_path, "uno", fp32={"onnx": _paso("/x/m.onnx", "dos coma cero")})

        with pytest.raises(cost_mod.ExportCostError, match="no es un número"):
            cost_mod.index_export_steps(tmp_path)

    def test_la_procedencia_tambien_sale_de_los_metadatos(self, tmp_path: Path) -> None:
        """La conversión a OpenVINO la escribe en ``metadata.from_onnx``."""
        entrada = _paso("/x/m.xml", 0.7)
        entrada.pop("source_path", None)
        entrada["metadata"] = {"from_onnx": "/x/m.onnx"}
        _volcado(tmp_path, "uno", fp32={"openvino": entrada})

        indice = cost_mod.index_export_steps(tmp_path)
        rutas = list(indice.values())

        assert rutas[0].source is not None
        assert rutas[0].source.endswith("m.onnx")


class TestCosteDeUnArtefacto:
    """El coste de un artefacto es el de toda su cadena."""

    def test_suma_la_cadena_entera(self, tmp_path: Path) -> None:
        """Un IR en INT8 costó el ONNX, la conversión y la cuantización."""
        _volcado(
            tmp_path,
            "uno",
            fp32={
                "onnx": _paso("/x/m.onnx", 2.0),
                "openvino": _paso("/x/m.xml", 0.7, origen="/x/m.onnx"),
            },
            int8={"openvino": _paso("/x/m_int8.xml", 20.0, origen="/x/m.xml")},
        )
        indice = cost_mod.index_export_steps(tmp_path)
        artefacto = next(k for k in indice if k.endswith("m_int8.xml"))

        coste = cost_mod.artifact_cost(artefacto, indice)

        assert coste.seconds == pytest.approx(22.7)
        assert len(coste.steps) == 3

    def test_un_eslabon_sin_tiempo_deja_la_cadena_sin_total(self, tmp_path: Path) -> None:
        """Sumar los que hay daría una cifra menor que la real."""
        _volcado(
            tmp_path,
            "uno",
            fp32={"onnx": _paso("/x/m.onnx", "")},
            int8={"onnxruntime": _paso("/x/m_int8.onnx", 30.0, origen="/x/m.onnx")},
        )
        indice = cost_mod.index_export_steps(tmp_path)
        artefacto = next(k for k in indice if k.endswith("m_int8.onnx"))

        coste = cost_mod.artifact_cost(artefacto, indice)

        assert coste.seconds is None
        assert "menor que la real" in coste.reason
        assert len(coste.steps) == 2

    def test_un_paso_fallido_no_es_el_coste_de_nada(self, tmp_path: Path) -> None:
        """Un paso que no terminó bien no produjo lo que se midió."""
        _volcado(tmp_path, "uno", fp32={"onnx": _paso("/x/m.onnx", 2.0, status="fallido")})
        indice = cost_mod.index_export_steps(tmp_path)

        coste = cost_mod.artifact_cost(next(iter(indice)), indice)

        assert coste.seconds is None
        assert "estado distinto de" in coste.reason
        assert "m.onnx" in coste.reason

    def test_una_cadena_rota_se_distingue_de_un_artefacto_desconocido(self, tmp_path: Path) -> None:
        _volcado(
            tmp_path,
            "uno",
            int8={"onnxruntime": _paso("/x/m_int8.onnx", 30.0, origen="/x/perdido.onnx")},
        )
        indice = cost_mod.index_export_steps(tmp_path)

        coste = cost_mod.artifact_cost(next(iter(indice)), indice)

        assert coste.seconds is None
        assert "Falta un eslabón" in coste.reason
        assert coste.steps

    def test_un_artefacto_que_nadie_produjo_no_tiene_cadena(self, tmp_path: Path) -> None:
        coste = cost_mod.artifact_cost("models/x/best.pt", {})

        assert coste.seconds is None
        assert coste.steps == ()
        assert "no se exportó" in coste.reason

    def test_una_procedencia_circular_es_un_error(self, tmp_path: Path) -> None:
        """No se puede sumar, y devolver un total parcial sería peor."""
        _volcado(
            tmp_path,
            "uno",
            fp32={
                "onnx": _paso("/x/a.onnx", 1.0, origen="/x/b.onnx"),
                "openvino": _paso("/x/b.onnx", 1.0, origen="/x/a.onnx"),
            },
        )
        indice = cost_mod.index_export_steps(tmp_path)

        with pytest.raises(cost_mod.ExportCostError, match="dos veces"):
            cost_mod.artifact_cost(next(iter(indice)), indice)


def _paths_de_mentira(raiz: Path) -> ProjectPaths:
    """Construye unas rutas de proyecto ancladas en otro sitio."""
    return ProjectPaths(
        root=raiz,
        configs=raiz / "configs",
        data=raiz / "data",
        models=raiz / "models",
        results=raiz / "results",
    )


class TestRutasDeOtraMaquina:
    """Un volcado no deja de ser legible por haberse escrito en otra máquina."""

    def test_una_ruta_de_otra_maquina_se_indexa_como_si_fuera_de_esta(self, tmp_path: Path) -> None:
        """El tramo desde ``results/`` es lo único que se conserva."""
        _volcado(
            tmp_path / "results" / "exports" / "x",
            "uno",
            fp32={"onnx": _paso("/otra/maquina/results/exports/x/m.onnx", 2.0)},
        )

        indice = cost_mod.index_export_steps(
            tmp_path / "results" / "exports", paths=_paths_de_mentira(tmp_path)
        )

        assert list(indice) == ["results/exports/x/m.onnx"]

    def test_la_cadena_entera_se_reubica_incluida_la_procedencia(self, tmp_path: Path) -> None:
        """Reubicar solo el artefacto dejaría la cadena rota por el eslabón de origen."""
        _volcado(
            tmp_path / "results" / "exports" / "x",
            "uno",
            fp32={"onnx": _paso("/otra/maquina/results/exports/x/m.onnx", 2.0)},
            int8={
                "onnxruntime": _paso(
                    "/otra/maquina/results/exports/x/m_int8.onnx",
                    30.0,
                    origen="/otra/maquina/results/exports/x/m.onnx",
                )
            },
        )
        indice = cost_mod.index_export_steps(
            tmp_path / "results" / "exports", paths=_paths_de_mentira(tmp_path)
        )

        coste = cost_mod.artifact_cost("results/exports/x/m_int8.onnx", indice)

        assert coste.seconds == pytest.approx(32.0)
        assert [paso.artifact for paso in coste.steps] == [
            "results/exports/x/m_int8.onnx",
            "results/exports/x/m.onnx",
        ]

    def test_una_ruta_de_fuera_de_results_no_se_reubica_a_ciegas(self, tmp_path: Path) -> None:
        """Adivinar dónde cuelga sería inventarse la procedencia."""
        _volcado(
            tmp_path / "results" / "exports" / "x",
            "uno",
            fp32={"onnx": _paso("/algun/sitio/m.onnx", 2.0)},
        )

        indice = cost_mod.index_export_steps(
            tmp_path / "results" / "exports", paths=_paths_de_mentira(tmp_path)
        )

        assert list(indice) == ["/algun/sitio/m.onnx"]


def _cargar_script() -> ModuleType:
    """Carga ``scripts/31_register_export_cost.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "31_register_export_cost.py"
    especificacion = importlib.util.spec_from_file_location("register_export_cost", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


class TestLaEtiquetaDeUnaFilaConLaClaveCoja:
    """``/tensorrt///`` era una etiqueta real del volcado del paso 31."""

    def test_una_clave_completa_se_sigue_nombrando_con_barras(self) -> None:
        script = _cargar_script()

        clave = {"experiment_id": "s_baseline_default_745e3d08", "runtime": "onnxruntime"}

        assert script.etiqueta_de_clave(clave) == "s_baseline_default_745e3d08/onnxruntime"

    @pytest.mark.parametrize(
        ("puesto", "valor", "vieja"),
        [
            ("runtime", "tensorrt", "/tensorrt///"),
            ("device_id", "edge_cpu_arm", "///edge_cpu_arm/"),
        ],
    )
    def test_una_clave_coja_dice_que_campos_le_faltan(
        self, puesto: str, valor: str, vieja: str
    ) -> None:
        script = _cargar_script()

        clave = dict.fromkeys(
            ("experiment_id", "runtime", "precision", "device_id", "batch_size"), ""
        )
        clave[puesto] = valor

        etiqueta = script.etiqueta_de_clave(clave)

        assert etiqueta != vieja
        assert f"{puesto}={valor}" in etiqueta
        assert "experiment_id" in etiqueta and "batch_size" in etiqueta

    def test_ninguna_fila_real_se_nombra_con_un_campo_en_blanco(self) -> None:
        from tfm_edge.common.registry import ExperimentRegistry

        rutas = get_project_paths()
        script = _cargar_script()
        pasos = cost_mod.index_export_steps(rutas.results / "exports", paths=rutas)

        rellenables, sin_coste = script.planificar(
            ExperimentRegistry().read_rows("deployment_metrics"), pasos
        )

        assert len(rellenables) + len(sin_coste) > 50, "no se ha planificado casi nada"
        for entrada in (*rellenables, *sin_coste):
            etiqueta = entrada["label"]
            assert "//" not in etiqueta, f"etiqueta con un campo en blanco: {etiqueta!r}"
            assert not etiqueta.startswith("/") and not etiqueta.endswith(
                "/"
            ), f"etiqueta con un campo en blanco: {etiqueta!r}"


class TestElPasoSobreLaCampanaReal:
    """Comprobaciones contra el registro y los volcados que hay en el repositorio."""

    def test_toda_fila_exportada_tiene_su_coste_reconstruible(self) -> None:
        """Si una deja de tenerlo, es que su volcado se ha perdido."""
        from tfm_edge.common.registry import ExperimentRegistry

        rutas = get_project_paths()
        script = _cargar_script()
        pasos = cost_mod.index_export_steps(rutas.results / "exports", paths=rutas)
        # Solo las filas medidas; las de cobertura con status='no_disponible' no tienen coste de
        # exportación.
        filas = [
            fila
            for fila in ExperimentRegistry().read_rows("deployment_metrics")
            if fila.get("status", "") == "ok"
        ]

        rellenables, sin_coste = script.planificar(filas, pasos)

        assert len(rellenables) + len(sin_coste) == len(filas)
        assert all(entrada["seconds"] > 0 for entrada in rellenables)

        # Solo las filas de PyTorch, que no se exportan, quedan sin coste.
        motores = [entrada["key"]["runtime"] for entrada in sin_coste]
        assert set(motores) == {"pytorch"}
        assert len(motores) == 4

    def test_desde_un_clon_en_otra_ruta_sale_exactamente_lo_mismo(self, tmp_path: Path) -> None:
        from tfm_edge.common.registry import ExperimentRegistry

        rutas = get_project_paths()
        script = _cargar_script()
        exportaciones = rutas.results / "exports"
        filas = ExperimentRegistry().read_rows("deployment_metrics")

        aqui, sin_aqui = script.planificar(
            filas, cost_mod.index_export_steps(exportaciones, paths=rutas)
        )

        destino = tmp_path / "results" / "exports"
        for volcado in exportaciones.glob("**/*.json"):
            copia = destino / volcado.relative_to(exportaciones)
            copia.parent.mkdir(parents=True, exist_ok=True)
            copia.write_bytes(volcado.read_bytes())
        alli, sin_alli = script.planificar(
            filas, cost_mod.index_export_steps(destino, paths=_paths_de_mentira(tmp_path))
        )

        assert aqui, "sin filas reconstruibles aquí, el test no compara nada"
        assert {e["label"]: e["seconds"] for e in alli} == {e["label"]: e["seconds"] for e in aqui}
        assert [e["label"] for e in sin_alli] == [e["label"] for e in sin_aqui]

    def test_el_registro_no_tiene_ceros_donde_no_hubo_exportacion(self) -> None:
        """Un cero diría «exportar fue gratis»; la ausencia dice «no se exportó»."""
        from tfm_edge.common.registry import ExperimentRegistry

        filas = ExperimentRegistry().read_rows("deployment_metrics")
        nativas = [fila for fila in filas if fila["runtime"] == "pytorch"]

        assert nativas, "el registro debería tener alguna fila de PyTorch"
        assert all((fila["export_time_s"] or "").strip() == "" for fila in nativas)

    def test_engine_build_time_sigue_vacia_porque_no_se_compilo_ningun_motor(self) -> None:
        """Rellenarla con el tiempo de exportar mezclaría dos cosas distintas."""
        from tfm_edge.common.registry import ExperimentRegistry

        filas = ExperimentRegistry().read_rows("deployment_metrics")

        assert all((fila["engine_build_time_s"] or "").strip() == "" for fila in filas)
