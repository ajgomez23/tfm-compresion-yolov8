"""Tests del relleno de celdas del registro."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tfm_edge.common import registry as registry_mod
from tfm_edge.common.exceptions import (
    RegistryConflictError,
    RegistryIntegrityError,
    RegistrySchemaError,
)
from tfm_edge.common.paths import get_project_paths
from tfm_edge.common.registry import DEPLOYMENT_COLUMNS, ExperimentRegistry


def _fila(experiment_id: str, runtime: str = "onnxruntime", **extra: str) -> dict[str, str]:
    """Construye una fila mínima de despliegue."""
    fila = {columna: "" for columna in DEPLOYMENT_COLUMNS}
    fila.update(
        {
            "experiment_id": experiment_id,
            "runtime": runtime,
            "precision": "fp32",
            "device_id": "workstation_cpu",
            "batch_size": "1",
            "status": "ok",
        }
    )
    fila.update(extra)
    return fila


@pytest.fixture()
def registro(tmp_path: Path) -> ExperimentRegistry:
    """Registro con tres filas de despliegue en un orden concreto."""
    reg = ExperimentRegistry(registry_dir=tmp_path)
    for identificador in ("a_baseline_default_00000000", "b_ptq_x_11111111", "c_qat_y_22222222"):
        reg.append_deployment_metrics(_fila(identificador))
    return reg


class TestRellenoBasico:
    """Rellenar una celda vacía es lo que este método hace."""

    def test_rellena_la_celda_y_no_toca_las_demas(self, registro: ExperimentRegistry) -> None:
        """Todo lo que no se nombra se queda exactamente como estaba."""
        antes = registro.read_rows("deployment_metrics")

        registro.update_cells(
            "deployment_metrics",
            [
                {
                    "experiment_id": "b_ptq_x_11111111",
                    "runtime": "onnxruntime",
                    "precision": "fp32",
                    "device_id": "workstation_cpu",
                    "batch_size": "1",
                    "export_time_s": 36.07,
                }
            ],
        )
        despues = registro.read_rows("deployment_metrics")

        assert despues[1]["export_time_s"] == "36.07"
        for fila_antes, fila_despues in zip(antes, despues):
            for columna in DEPLOYMENT_COLUMNS:
                if columna == "export_time_s":
                    continue
                assert fila_antes[columna] == fila_despues[columna]

    def test_el_orden_de_las_filas_se_conserva(self, registro: ExperimentRegistry) -> None:
        """Es la razón de existir del método."""
        registro.update_cells(
            "deployment_metrics",
            [
                {
                    "experiment_id": "a_baseline_default_00000000",
                    "runtime": "onnxruntime",
                    "precision": "fp32",
                    "device_id": "workstation_cpu",
                    "batch_size": "1",
                    "export_time_s": 2.04,
                }
            ],
        )

        identificadores = [f["experiment_id"] for f in registro.read_rows("deployment_metrics")]
        assert identificadores == [
            "a_baseline_default_00000000",
            "b_ptq_x_11111111",
            "c_qat_y_22222222",
        ]

    def test_repetirlo_no_reescribe_el_fichero(self, registro: ExperimentRegistry) -> None:
        actualizacion = [
            {
                "experiment_id": "c_qat_y_22222222",
                "runtime": "onnxruntime",
                "precision": "fp32",
                "device_id": "workstation_cpu",
                "batch_size": "1",
                "export_time_s": 37.85,
            }
        ]
        registro.update_cells("deployment_metrics", actualizacion)
        fichero = registro.path_for("deployment_metrics")
        marca = fichero.stat().st_mtime_ns

        informe = registro.update_cells("deployment_metrics", actualizacion)

        assert informe["written"] == []
        assert len(informe["unchanged"]) == 1
        assert fichero.stat().st_mtime_ns == marca


class TestGuardas:
    """Lo que se niega a hacer."""

    def test_no_pisa_una_celda_que_ya_tiene_otro_valor(self, registro: ExperimentRegistry) -> None:
        """Una celda con contenido es evidencia de una medición real."""
        clave = {
            "experiment_id": "b_ptq_x_11111111",
            "runtime": "onnxruntime",
            "precision": "fp32",
            "device_id": "workstation_cpu",
            "batch_size": "1",
        }
        registro.update_cells("deployment_metrics", [{**clave, "export_time_s": 36.07}])

        with pytest.raises(RegistryConflictError, match="36.07"):
            registro.update_cells("deployment_metrics", [{**clave, "export_time_s": 99.0}])

        filas = registro.read_rows("deployment_metrics")
        assert filas[1]["export_time_s"] == "36.07"

    def test_skip_deja_el_valor_anterior_y_lo_declara(self, registro: ExperimentRegistry) -> None:
        clave = {
            "experiment_id": "b_ptq_x_11111111",
            "runtime": "onnxruntime",
            "precision": "fp32",
            "device_id": "workstation_cpu",
            "batch_size": "1",
        }
        registro.update_cells("deployment_metrics", [{**clave, "export_time_s": 36.07}])

        informe = registro.update_cells(
            "deployment_metrics", [{**clave, "export_time_s": 99.0}], on_conflict="skip"
        )

        assert informe["conflicts"]
        assert informe["skipped"]
        assert registro.read_rows("deployment_metrics")[1]["export_time_s"] == "36.07"

    def test_una_fila_que_no_existe_es_un_error(self, registro: ExperimentRegistry) -> None:
        """Rellenar filas es una cosa y crearlas es otra."""
        with pytest.raises(RegistryIntegrityError, match="ninguna fila"):
            registro.update_cells(
                "deployment_metrics",
                [
                    {
                        "experiment_id": "z_no_existe_99999999",
                        "runtime": "onnxruntime",
                        "precision": "fp32",
                        "device_id": "workstation_cpu",
                        "batch_size": "1",
                        "export_time_s": 1.0,
                    }
                ],
            )

    def test_una_columna_fuera_del_esquema_es_un_error(self, registro: ExperimentRegistry) -> None:
        """El esquema está fijado en la sección 16 y no se amplía por descuido."""
        with pytest.raises(RegistrySchemaError, match="no están en el esquema"):
            registro.update_cells(
                "deployment_metrics",
                [
                    {
                        "experiment_id": "b_ptq_x_11111111",
                        "runtime": "onnxruntime",
                        "precision": "fp32",
                        "device_id": "workstation_cpu",
                        "batch_size": "1",
                        "coste_de_exportar": 1.0,
                    }
                ],
            )

    def test_sin_la_clave_primaria_no_se_sabe_a_que_fila_va(
        self, registro: ExperimentRegistry
    ) -> None:
        """Con una clave parcial, la elección sería por posición."""
        with pytest.raises(RegistrySchemaError, match="clave primaria"):
            registro.update_cells(
                "deployment_metrics",
                [{"experiment_id": "b_ptq_x_11111111", "export_time_s": 1.0}],
            )

    def test_una_politica_desconocida_no_se_interpreta(self, registro: ExperimentRegistry) -> None:
        """Un nombre mal escrito no puede acabar comportándose como el defecto."""
        with pytest.raises(ValueError, match="desconocida"):
            registro.update_cells("deployment_metrics", [], on_conflict="pisar")


class TestPasaPorLasMismasGuardasQueAppend:
    """``update_cells`` valida y normaliza igual que ``append``."""

    def test_un_valor_que_codifica_ausencia_como_texto_se_rechaza(
        self, registro: ExperimentRegistry
    ) -> None:
        registro.append_deployment_metrics(_fila("s_baseline_default_aaaa1111"))

        with pytest.raises(RegistrySchemaError, match="NaN"):
            registro.update_cells(
                "deployment_metrics",
                [
                    {
                        "experiment_id": "s_baseline_default_aaaa1111",
                        "runtime": "onnxruntime",
                        "precision": "fp32",
                        "device_id": "workstation_cpu",
                        "batch_size": "1",
                        "notes": "NaN",
                    }
                ],
            )

    def test_un_status_inventado_se_rechaza(self, registro: ExperimentRegistry) -> None:
        """Por ``append`` no entraba y por aquí sí."""
        registro.append_deployment_metrics(_fila("s_baseline_default_aaaa2222"))

        with pytest.raises(RegistrySchemaError, match="status"):
            registro.update_cells(
                "deployment_metrics",
                [
                    {
                        "experiment_id": "s_baseline_default_aaaa2222",
                        "runtime": "onnxruntime",
                        "precision": "fp32",
                        "device_id": "workstation_cpu",
                        "batch_size": "1",
                        "status": "casi_bien",
                    }
                ],
            )

    def test_una_ruta_del_arbol_se_guarda_relativa(self, registro: ExperimentRegistry) -> None:
        """Lo que hacía ``append`` y aquí no se hacía."""
        registro.append_deployment_metrics(_fila("s_baseline_default_aaaa3333"))
        raiz = get_project_paths().root
        absoluta = raiz / "results" / "exports" / "x.onnx"

        registro.update_cells(
            "deployment_metrics",
            [
                {
                    "experiment_id": "s_baseline_default_aaaa3333",
                    "runtime": "onnxruntime",
                    "precision": "fp32",
                    "device_id": "workstation_cpu",
                    "batch_size": "1",
                    "artifact_path": str(absoluta),
                }
            ],
        )

        fila = next(
            f
            for f in registro.read_rows("deployment_metrics")
            if f["experiment_id"] == "s_baseline_default_aaaa3333"
        )
        assert fila["artifact_path"] == "results/exports/x.onnx"

    def test_toda_escritura_publica_pasa_por_la_guarda(self) -> None:
        """La pregunta, hecha al árbol sintáctico y no a la memoria de nadie."""
        arbol = ast.parse(
            Path(registry_mod.__file__).read_text(encoding="utf-8"), filename=registry_mod.__file__
        )
        clase = next(
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.ClassDef) and nodo.name == "ExperimentRegistry"
        )
        metodos = {nodo.name: nodo for nodo in clase.body if isinstance(nodo, ast.FunctionDef)}

        def alcanza(nombre: str, objetivo: str, vistos: frozenset[str] = frozenset()) -> bool:
            """Si un método llega al objetivo por sí mismo o por los que llama."""
            if nombre in vistos or nombre not in metodos:
                return False
            llamados = {
                nodo.func.attr
                for nodo in ast.walk(metodos[nombre])
                if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Attribute)
            }
            if objetivo in llamados:
                return True
            return any(alcanza(otro, objetivo, vistos | {nombre}) for otro in llamados)

        escritoras = sorted(
            nombre
            for nombre in metodos
            if not nombre.startswith("_") and alcanza(nombre, "_write_rows_atomic")
        )

        assert escritoras, "el análisis no ha encontrado ninguna escritura pública"
        sin_guarda = [n for n in escritoras if not alcanza(n, "_validar_celdas")]
        assert not sin_guarda, (
            f"escriben sin pasar por _validar_celdas: {sin_guarda}. "
            "Si es a propósito, tiene que llevar su propia verificación encima y "
            "quedar escrito, como en 19_normalize_registry_paths.py."
        )
