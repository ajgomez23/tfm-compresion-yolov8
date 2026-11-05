"""Registro único de resultados: esquema, escritura atómica y validación.

Todas las tablas y figuras de la memoria salen de sus cuatro CSV. Las columnas se declaran solo aquí
y una fila que no encaja es un error. Se escribe en un fichero temporal y se renombra, para no dejar
un CSV a medias. Los valores ausentes son celdas vacías, nunca 0, -1 ni NaN.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, get_args

from tfm_edge.common.environment import ENVIRONMENT_COLUMNS
from tfm_edge.common.exceptions import (
    RegistryConflictError,
    RegistryDuplicateError,
    RegistryIntegrityError,
    RegistrySchemaError,
)
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, get_project_paths, relative_to_root, slugify

logger = get_logger(__name__)

# Esquema de columnas

#: Columnas de experiments.csv.
EXPERIMENTS_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "parent_experiment_id",
    "timestamp",
    "model_family",
    "model_size",
    "technique",
    "technique_variant",
    "config_hash",
    "config_path",
    "compression_ratio",
    "effective_params",
    "effective_params_ratio",
    "precision",
    "training_epochs",
    "training_time_s",
    "training_device",
    "weights_path",
    "status",
    "notes",
)

#: Columnas de quality_metrics.csv anteriores al AP por clase.
QUALITY_COLUMNS_BEFORE_CLASSES: tuple[str, ...] = (
    "experiment_id",
    "runtime",
    "precision",
    "device_id",
    "environment_id",
    # Evaluador con el que se obtuvo la cifra. Va en su propia columna porque decide qué filas son
    # comparables: sobre el mismo checkpoint, pycocotools da 0.325636 y Ultralytics 0.330215.
    "evaluator",
    "split",
    "input_resolution",
    "conf_threshold",
    "iou_threshold",
    "map_50",
    "map_50_95",
    "ap_75",
    # Desglose por tamaño de objeto con los dos criterios, en columnas cuyo nombre indica cuál
    # aplica. size_criterion_primary indica el primario y size_criterion_inference_resolution la
    # resolución usada.
    "size_criterion_primary",
    "size_criterion_inference_resolution",
    "ap_small_inference_res",
    "ap_medium_inference_res",
    "ap_large_inference_res",
    "ap_small_original_res",
    "ap_medium_original_res",
    "ap_large_original_res",
    "ar_1",
    "ar_10",
    "ar_100",
    "ar_small_inference_res",
    "ar_medium_inference_res",
    "ar_large_inference_res",
    "ar_small_original_res",
    "ar_medium_original_res",
    "ar_large_original_res",
    "precision_at_best_f1",
    "recall_at_best_f1",
    "f1_best",
)

#: Columnas de quality_metrics añadidas al esquema base, para que los tests distingan una extensión
#: deliberada de un cambio accidental.
QUALITY_SCHEMA_EXTENSIONS: tuple[str, ...] = (
    "evaluator",
    "size_criterion_primary",
    "size_criterion_inference_resolution",
    "ap_small_inference_res",
    "ap_medium_inference_res",
    "ap_large_inference_res",
    "ap_small_original_res",
    "ap_medium_original_res",
    "ap_large_original_res",
    "ar_small_inference_res",
    "ar_medium_inference_res",
    "ar_large_inference_res",
    "ar_small_original_res",
    "ar_medium_original_res",
    "ar_large_original_res",
)

#: Qué columna del esquema base sustituye cada una.
QUALITY_SPEC_COLUMN_MAPPING: dict[str, tuple[str, str]] = {
    "ap_small": ("ap_small_inference_res", "ap_small_original_res"),
    "ap_medium": ("ap_medium_inference_res", "ap_medium_original_res"),
    "ap_large": ("ap_large_inference_res", "ap_large_original_res"),
    "ar_small": ("ar_small_inference_res", "ar_small_original_res"),
    "ar_medium": ("ar_medium_inference_res", "ar_medium_original_res"),
    "ar_large": ("ar_large_inference_res", "ar_large_original_res"),
}

#: Columnas de quality_metrics.csv posteriores al AP por clase.
QUALITY_COLUMNS_AFTER_CLASSES: tuple[str, ...] = (
    "error_classification",
    "error_localization",
    "error_both",
    "error_duplicate",
    "error_background",
    "error_missed",
    "num_images",
    "num_detections",
    "eval_time_s",
    "predictions_path",
    "status",
    "notes",
)

#: Columnas de deployment_metrics.csv, con dos añadidas al esquema base. ``cpu_affinity`` distingue
#: workstation_cpu de constrained_cpu, que usan la misma CPU. ``latency_representative`` es false
#: cuando la latencia no representa al dispositivo (TFLite medido en x86), y el análisis de Pareto
#: excluye esas filas.
DEPLOYMENT_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "runtime",
    "precision",
    "device_id",
    "environment_id",
    "input_resolution",
    "batch_size",
    "num_threads",
    "cpu_affinity",
    "execution_provider",
    "warmup_iters",
    "measure_iters",
    "num_repeats",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_min_ms",
    "latency_max_ms",
    "latency_p50_ms",
    "latency_p90_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "preprocess_mean_ms",
    "inference_mean_ms",
    "postprocess_mean_ms",
    "throughput_img_s",
    "peak_gpu_memory_mb",
    "peak_host_memory_mb",
    "model_load_memory_mb",
    "artifact_size_bytes",
    "artifact_size_gzip_bytes",
    "artifact_path",
    "export_time_s",
    "engine_build_time_s",
    "temperature_start_c",
    "temperature_end_c",
    "raw_timings_path",
    "latency_representative",
    "timestamp",
    "status",
    "notes",
)

#: Columnas añadidas al esquema base de deployment_metrics; los tests las contrastan.
DEPLOYMENT_SCHEMA_EXTENSIONS: tuple[str, ...] = ("cpu_affinity", "latency_representative")

TABLE_FILENAMES: dict[str, str] = {
    "experiments": "experiments.csv",
    "quality_metrics": "quality_metrics.csv",
    "deployment_metrics": "deployment_metrics.csv",
    "environments": "environments.csv",
}

#: Claves primarias de cada tabla.
TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "experiments": ("experiment_id",),
    "quality_metrics": (
        "experiment_id",
        "runtime",
        "precision",
        "device_id",
        "evaluator",
        "split",
    ),
    "deployment_metrics": ("experiment_id", "runtime", "precision", "device_id", "batch_size"),
    "environments": ("environment_id",),
}

#: Columnas que contienen rutas. append las convierte en relativas a la raíz al escribir.
TABLE_PATH_COLUMNS: dict[str, tuple[str, ...]] = {
    "experiments": ("config_path", "weights_path"),
    "quality_metrics": ("predictions_path",),
    "deployment_metrics": ("artifact_path", "raw_timings_path"),
    "environments": (),
}

#: Política al rellenar una celda que ya tiene otro valor; por defecto se falla.
CellConflictPolicy = Literal["fail", "skip", "overwrite"]

VALID_CELL_CONFLICT_POLICIES: tuple[str, ...] = get_args(CellConflictPolicy)

#: Valores admitidos en ``status``.
VALID_STATUS: tuple[str, ...] = ("ok", "fallido", "no_disponible")

#: Evaluadores admitidos en ``evaluator``. La lista es cerrada: un evaluador nuevo da cifras en otra
#: escala.
VALID_EVALUATORS: tuple[str, ...] = ("pycocotools", "ultralytics")

#: Textos que codifican mal una ausencia y nunca deben aparecer como valor.
FORBIDDEN_NULL_TEXTS: frozenset[str] = frozenset({"nan", "none", "null", "n/a", "na", "-"})

DuplicatePolicy = Literal["skip", "overwrite", "version", "fail"]

#: Las mismas políticas en tiempo de ejecución, para los ``choices`` de ``--on-duplicate``.
VALID_DUPLICATE_POLICIES: tuple[str, ...] = get_args(DuplicatePolicy)


def is_absolute_path_text(value: str) -> bool:
    """True si el texto es una ruta absoluta POSIX, UNC o de unidad de Windows.

    No usa pathlib porque debe reconocer tanto rutas de Colab como de Windows desde cualquier
    sistema.
    """
    texto = value.strip()
    if not texto:
        return False
    if texto.startswith(("/", "\\")):
        return True
    return len(texto) > 2 and texto[1] == ":" and texto[2] in "\\/"


def build_ap_class_columns(class_names: Sequence[str]) -> tuple[str, ...]:
    """Nombres de las columnas de AP por clase: ``ap_class_{idx:02d}_{nombre_normalizado}``.

    Falla si la lista está vacía o si dos clases dan el mismo nombre normalizado.
    """
    if not class_names:
        raise RegistrySchemaError(
            "quality_metrics",
            "la lista de nombres de clase está vacía; no se pueden construir las "
            "columnas de AP por clase.",
        )
    columnas = []
    vistos: dict[str, int] = {}
    for indice, nombre in enumerate(class_names):
        normalizado = slugify(nombre)
        if not normalizado:
            raise RegistrySchemaError(
                "quality_metrics",
                f"el nombre de la clase {indice} ('{nombre}') se normaliza a una " "cadena vacía.",
            )
        if normalizado in vistos:
            raise RegistrySchemaError(
                "quality_metrics",
                f"las clases {vistos[normalizado]} y {indice} se normalizan ambas a "
                f"'{normalizado}', lo que haría ambiguas las columnas.",
            )
        vistos[normalizado] = indice
        columnas.append(f"ap_class_{indice:02d}_{normalizado}")
    return tuple(columnas)


def build_quality_columns(class_names: Sequence[str]) -> tuple[str, ...]:
    """Esquema completo de quality_metrics.csv."""
    return (
        *QUALITY_COLUMNS_BEFORE_CLASSES,
        *build_ap_class_columns(class_names),
        *QUALITY_COLUMNS_AFTER_CLASSES,
    )


# Normalización de valores


def normalize_value(value: Any) -> str:
    """Valor tal como se escribe en el CSV. None y NaN se escriben como celda vacía."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def utc_timestamp() -> str:
    """Instante actual en UTC con formato ISO 8601."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Informe de validación


@dataclass
class RegistryValidationReport:
    """Resultado de validar el registro completo."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    status_counts: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """True si no hay errores de integridad."""
        return not self.errors

    def raise_if_invalid(self) -> None:
        """Lanza RegistryIntegrityError si hay algún error."""
        if self.errors:
            raise RegistryIntegrityError("\n".join(f"  - {e}" for e in self.errors))

    def summary(self) -> str:
        """Genera un resumen legible del informe."""
        lineas = ["Validación del registro de resultados:"]
        for tabla, total in sorted(self.row_counts.items()):
            estados = self.status_counts.get(tabla, {})
            detalle = ", ".join(f"{k}={v}" for k, v in sorted(estados.items())) or "sin estados"
            lineas.append(f"  {tabla}: {total} filas ({detalle})")
        if self.errors:
            lineas.append(f"  ERRORES ({len(self.errors)}):")
            lineas.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            lineas.append(f"  Avisos ({len(self.warnings)}):")
            lineas.extend(f"    - {w}" for w in self.warnings)
        if not self.errors and not self.warnings:
            lineas.append("  Sin errores ni avisos.")
        return "\n".join(lineas)


# Registro


class ExperimentRegistry:
    """Lectura y escritura de los cuatro CSV del registro.

    El esquema de quality_metrics depende de los nombres de clase, que se pasan al construir el
    registro. Sin ellos, las operaciones sobre esa tabla fallan.
    """

    def __init__(
        self,
        registry_dir: Path | str | None = None,
        class_names: Sequence[str] | None = None,
    ) -> None:
        """Inicializa el registro. Sin ``registry_dir`` se usa results/registry."""
        self.registry_dir = (
            Path(registry_dir) if registry_dir is not None else get_project_paths().results_registry
        )
        ensure_dir(self.registry_dir)
        self._class_names = list(class_names) if class_names is not None else None

    @property
    def class_names(self) -> list[str]:
        """Nombres de las clases. Falla si no se pasaron al construir el registro."""
        if self._class_names is None:
            raise RegistrySchemaError(
                "quality_metrics",
                "no se han proporcionado los nombres de las clases. Construye el registro con "
                "ExperimentRegistry(class_names=load_class_names(...)).",
            )
        return self._class_names

    def columns_for(self, table: str) -> tuple[str, ...]:
        """Columnas de una tabla en su orden."""
        if table == "experiments":
            return EXPERIMENTS_COLUMNS
        if table == "deployment_metrics":
            return DEPLOYMENT_COLUMNS
        if table == "environments":
            return ENVIRONMENT_COLUMNS
        if table == "quality_metrics":
            return build_quality_columns(self.class_names)
        raise RegistrySchemaError(
            table, f"tabla desconocida (esperadas: {', '.join(sorted(TABLE_FILENAMES))})"
        )

    def path_for(self, table: str) -> Path:
        """Ruta del CSV de una tabla."""
        if table not in TABLE_FILENAMES:
            raise RegistrySchemaError(
                table, f"tabla desconocida (esperadas: {', '.join(sorted(TABLE_FILENAMES))})"
            )
        return self.registry_dir / TABLE_FILENAMES[table]

    def read_rows(self, table: str) -> list[dict[str, str]]:
        """Lee las filas de una tabla como diccionarios de cadenas, sin convertir tipos.

        Lista vacía si el fichero aún no existe.
        """
        fichero = self.path_for(table)
        if not fichero.is_file():
            return []
        with fichero.open("r", encoding="utf-8", newline="") as manejador:
            return [dict(fila) for fila in csv.DictReader(manejador)]

    def read_dataframe(self, table: str) -> Any:
        """Lee una tabla como DataFrame de texto, conservando las celdas vacías."""
        import pandas as pd

        fichero = self.path_for(table)
        if not fichero.is_file():
            return pd.DataFrame(columns=list(self.columns_for(table)))
        return pd.read_csv(fichero, dtype=str, keep_default_na=False, na_filter=False)

    def valid_rows(self, table: str) -> tuple[list[dict[str, str]], int]:
        """Filas con ``status == "ok"`` y número de filas descartadas."""
        todas = self.read_rows(table)
        validas = [fila for fila in todas if fila.get("status", "") == "ok"]
        descartadas = len(todas) - len(validas)
        if descartadas:
            logger.info(
                "Tabla '%s': %d filas descartadas del análisis por tener status != 'ok'.",
                table,
                descartadas,
            )
        return validas, descartadas

    def latency_comparable_rows(self) -> tuple[list[dict[str, str]], dict[str, int]]:
        """Filas de despliegue válidas para comparar latencias y recuento de descartes por motivo.

        Además de ``status == "ok"``, descarta las filas con ``latency_representative == "false"``.
        """
        todas = self.read_rows("deployment_metrics")
        descartes = {"status_no_ok": 0, "latencia_no_representativa": 0}
        utilizables: list[dict[str, str]] = []
        for fila in todas:
            if fila.get("status", "") != "ok":
                descartes["status_no_ok"] += 1
                continue
            if fila.get("latency_representative", "").strip().lower() == "false":
                descartes["latencia_no_representativa"] += 1
                continue
            utilizables.append(fila)

        if any(descartes.values()):
            logger.info(
                "Comparación de latencias: %d filas utilizables de %d. Descartadas %d por "
                "status != 'ok' y %d por latencia no representativa.",
                len(utilizables),
                len(todas),
                descartes["status_no_ok"],
                descartes["latencia_no_representativa"],
            )
        return utilizables, descartes

    def _write_rows_atomic(self, table: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        """Reescribe una tabla en un fichero temporal y lo renombra con os.replace."""
        columnas = self.columns_for(table)
        destino = self.path_for(table)
        ensure_dir(destino.parent)

        descriptor, ruta_temporal = tempfile.mkstemp(
            dir=str(destino.parent), prefix=f".{destino.stem}_", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as manejador:
                escritor = csv.DictWriter(
                    manejador, fieldnames=list(columnas), extrasaction="raise"
                )
                escritor.writeheader()
                for fila in rows:
                    escritor.writerow({c: normalize_value(fila.get(c)) for c in columnas})
                manejador.flush()
                os.fsync(manejador.fileno())
            os.replace(ruta_temporal, destino)
        except BaseException:
            Path(ruta_temporal).unlink(missing_ok=True)
            raise
        return destino

    def _validate_row_schema(self, table: str, row: Mapping[str, Any]) -> dict[str, Any]:
        """Comprueba que una fila encaja en el esquema y la devuelve en el orden de columnas."""
        columnas = set(self.columns_for(table))
        recibidas = set(row)

        sobrantes = recibidas - columnas
        if sobrantes:
            raise RegistrySchemaError(
                table,
                f"columnas no reconocidas: {', '.join(sorted(sobrantes))}. "
                "El esquema está fijado en la sección 16 de la especificación.",
            )
        faltantes = columnas - recibidas
        if faltantes:
            raise RegistrySchemaError(
                table,
                f"faltan columnas obligatorias: {', '.join(sorted(faltantes))}. Si no hay valor, "
                "pásalas como None.",
            )

        if table == "quality_metrics":
            evaluador = row.get("evaluator")
            if evaluador is None or not str(evaluador).strip():
                raise RegistrySchemaError(
                    table,
                    f"falta 'evaluator' (esperado: {', '.join(VALID_EVALUATORS)}).",
                )

        self._validar_celdas(table, row)
        return {columna: row.get(columna) for columna in self.columns_for(table)}

    def _validar_celdas(self, table: str, celdas: Mapping[str, Any]) -> None:
        """Comprueba los valores de unas celdas, de una fila completa o parcial.

        La usan append y update_cells. Solo mira las columnas que recibe.
        """
        estado = celdas.get("status")
        if estado is not None and str(estado) not in VALID_STATUS:
            raise RegistrySchemaError(
                table,
                f"status='{estado}' no es válido (esperado: {', '.join(VALID_STATUS)}).",
            )

        if table == "quality_metrics" and "evaluator" in celdas:
            evaluador = celdas.get("evaluator")
            if evaluador is not None and str(evaluador) not in VALID_EVALUATORS:
                raise RegistrySchemaError(
                    table,
                    f"evaluator='{evaluador}' no es válido "
                    f"(esperado: {', '.join(VALID_EVALUATORS)}).",
                )

        for clave, valor in celdas.items():
            if isinstance(valor, str) and valor.strip().lower() in FORBIDDEN_NULL_TEXTS:
                raise RegistrySchemaError(
                    table,
                    f"la columna '{clave}' tiene el valor '{valor}', que codifica una ausencia. "
                    "Usa None.",
                )

    def _row_key(self, table: str, row: Mapping[str, Any]) -> tuple[str, ...]:
        """Clave primaria de una fila."""
        return tuple(normalize_value(row.get(c)) for c in TABLE_PRIMARY_KEYS[table])

    def _normalize_path_columns(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        """Convierte las columnas de ruta en relativas a la raíz.

        Las celdas vacías se dejan igual y las rutas de fuera del proyecto quedan absolutas;
        scripts/14_import_colab_registry.py se encarga de reapuntarlas.
        """
        for columna in TABLE_PATH_COLUMNS.get(table, ()):
            valor = row.get(columna)
            if valor is None:
                continue
            texto = str(valor).strip()
            if not texto:
                continue
            row[columna] = relative_to_root(texto)
        return row

    def append(
        self,
        table: str,
        row: Mapping[str, Any],
        on_duplicate: DuplicatePolicy = "skip",
    ) -> bool:
        """Añade una fila a una tabla.

        ``on_duplicate``: ``skip`` no escribe, ``overwrite`` sustituye, ``version`` conserva las dos
        y ``fail`` lanza excepción. Devuelve True si se escribió algo.
        """
        if on_duplicate not in VALID_DUPLICATE_POLICIES:
            raise ValueError(
                f"Política de duplicados desconocida: '{on_duplicate}'. "
                f"Válidas: {', '.join(VALID_DUPLICATE_POLICIES)}."
            )

        fila = self._normalize_path_columns(table, self._validate_row_schema(table, row))
        existentes = self.read_rows(table)
        clave = self._row_key(table, fila)
        indices_duplicados = [
            i for i, existente in enumerate(existentes) if self._row_key(table, existente) == clave
        ]

        if indices_duplicados:
            clave_texto = "/".join(clave)
            if on_duplicate == "fail":
                raise RegistryDuplicateError(table, clave_texto)
            if on_duplicate == "skip":
                logger.info(
                    "Tabla '%s': ya existe la fila '%s'; se omite (política 'skip').",
                    table,
                    clave_texto,
                )
                return False
            if on_duplicate == "overwrite":
                logger.info(
                    "Tabla '%s': se sobrescribe la fila '%s' (política 'overwrite').",
                    table,
                    clave_texto,
                )
                for indice in reversed(indices_duplicados):
                    existentes.pop(indice)
            else:
                logger.warning(
                    "Tabla '%s': se conserva la fila anterior y se añade otra con la misma "
                    "clave '%s' (política 'version').",
                    table,
                    clave_texto,
                )

        existentes.append(fila)
        self._write_rows_atomic(table, existentes)
        logger.debug("Tabla '%s': fila '%s' escrita.", table, "/".join(clave))
        return True

    # Atajos por tabla

    def append_experiment(
        self, row: Mapping[str, Any], on_duplicate: DuplicatePolicy = "skip"
    ) -> bool:
        """Añade una fila a experiments.csv."""
        return self.append("experiments", row, on_duplicate)

    def append_quality_metrics(
        self, row: Mapping[str, Any], on_duplicate: DuplicatePolicy = "skip"
    ) -> bool:
        """Añade una fila a quality_metrics.csv."""
        return self.append("quality_metrics", row, on_duplicate)

    def append_deployment_metrics(
        self, row: Mapping[str, Any], on_duplicate: DuplicatePolicy = "skip"
    ) -> bool:
        """Añade una fila a deployment_metrics.csv."""
        return self.append("deployment_metrics", row, on_duplicate)

    def append_environment(
        self, row: Mapping[str, Any], on_duplicate: DuplicatePolicy = "skip"
    ) -> bool:
        """Añade una huella de entorno a environments.csv; por defecto omite las que ya existen."""
        return self.append("environments", row, on_duplicate)

    # Relleno de celdas

    def update_cells(
        self,
        table: str,
        updates: Sequence[Mapping[str, Any]],
        on_conflict: CellConflictPolicy = "fail",
    ) -> dict[str, list[str]]:
        """Rellena columnas de filas existentes sin cambiar el orden del fichero.

        Cada actualización lleva la clave primaria y las columnas a rellenar. Una celda con otro
        valor no se sobrescribe por defecto (``on_conflict="fail"``). Si el valor ya es el mismo, la
        fila cuenta como ``unchanged``. Devuelve las claves en ``written``, ``unchanged``,
        ``skipped`` y ``conflicts``.
        """
        if on_conflict not in VALID_CELL_CONFLICT_POLICIES:
            raise ValueError(
                f"Política de conflictos desconocida: '{on_conflict}'. "
                f"Válidas: {', '.join(VALID_CELL_CONFLICT_POLICIES)}."
            )

        columnas = set(self.columns_for(table))
        claves = TABLE_PRIMARY_KEYS[table]
        filas = self.read_rows(table)
        informe: dict[str, list[str]] = {
            "written": [],
            "unchanged": [],
            "skipped": [],
            "conflicts": [],
        }
        cambiado = False

        for actualizacion in updates:
            faltan = [c for c in claves if c not in actualizacion]
            if faltan:
                raise RegistrySchemaError(
                    table,
                    f"a una actualización le faltan columnas de la clave primaria: {faltan}.",
                )
            desconocidas = sorted(set(actualizacion) - columnas)
            if desconocidas:
                raise RegistrySchemaError(
                    table, f"columnas que no están en el esquema: {desconocidas}"
                )

            # Las mismas comprobaciones que en append: valores y rutas.
            self._validar_celdas(table, actualizacion)
            actualizacion = self._normalize_path_columns(table, dict(actualizacion))

            clave = tuple(normalize_value(actualizacion.get(c)) for c in claves)
            clave_texto = "/".join(clave)
            indices = [i for i, fila in enumerate(filas) if self._row_key(table, fila) == clave]
            if not indices:
                raise RegistryIntegrityError(
                    f"La tabla '{table}' no tiene ninguna fila con clave '{clave_texto}'. Para "
                    "crear una fila nueva usa append."
                )
            if len(indices) > 1:
                raise RegistryIntegrityError(
                    f"La tabla '{table}' tiene {len(indices)} filas con clave "
                    f"'{clave_texto}'. Rellenar una de ellas elegiría por posición."
                )

            fila = filas[indices[0]]
            for columna, valor in actualizacion.items():
                if columna in claves:
                    continue
                nuevo = normalize_value(valor)
                actual = normalize_value(fila.get(columna))
                if actual == nuevo:
                    informe["unchanged"].append(f"{clave_texto}:{columna}")
                    continue
                if actual:
                    detalle = f"{clave_texto}:{columna} ({actual!r} -> {nuevo!r})"
                    informe["conflicts"].append(detalle)
                    if on_conflict == "fail":
                        raise RegistryConflictError(table, clave_texto, columna, actual, nuevo)
                    if on_conflict == "skip":
                        informe["skipped"].append(detalle)
                        continue
                fila[columna] = nuevo
                informe["written"].append(f"{clave_texto}:{columna}")
                cambiado = True

        if cambiado:
            self._write_rows_atomic(table, filas)
            logger.info(
                "Tabla '%s': %d celdas rellenadas, %d ya estaban igual.",
                table,
                len(informe["written"]),
                len(informe["unchanged"]),
            )
        else:
            logger.info(
                "Tabla '%s': nada que cambiar; el fichero no se reescribe.",
                table,
            )
        return informe

    def validate(self) -> RegistryValidationReport:
        """Valida el registro: esquema, status, ausencias mal codificadas, duplicados e
        identificadores huérfanos.
        """
        informe = RegistryValidationReport()

        tablas: dict[str, list[dict[str, str]]] = {}
        for tabla in TABLE_FILENAMES:
            fichero = self.path_for(tabla)
            if not fichero.is_file():
                informe.row_counts[tabla] = 0
                continue

            if tabla == "quality_metrics" and self._class_names is None:
                informe.warnings.append(
                    "No se han proporcionado los nombres de clase: no se ha podido "
                    "validar el esquema de columnas de quality_metrics.csv."
                )
                esperadas: tuple[str, ...] | None = None
            else:
                esperadas = self.columns_for(tabla)

            filas = self.read_rows(tabla)
            tablas[tabla] = filas
            informe.row_counts[tabla] = len(filas)

            with fichero.open("r", encoding="utf-8", newline="") as manejador:
                cabecera = next(csv.reader(manejador), [])
            if esperadas is not None and tuple(cabecera) != esperadas:
                faltan = set(esperadas) - set(cabecera)
                sobran = set(cabecera) - set(esperadas)
                detalle = []
                if faltan:
                    detalle.append(f"faltan {sorted(faltan)}")
                if sobran:
                    detalle.append(f"sobran {sorted(sobran)}")
                if not detalle:
                    detalle.append("el orden de las columnas no es el canónico")
                informe.errors.append(f"{tabla}: esquema incorrecto ({'; '.join(detalle)}).")

            self._validar_filas(tabla, filas, informe)

        self._validar_referencias(tablas, informe)
        return informe

    def _validar_filas(
        self, table: str, rows: Sequence[Mapping[str, str]], report: RegistryValidationReport
    ) -> None:
        """Valida estados, ausencias mal codificadas y duplicados de una tabla."""
        recuento_estados: dict[str, int] = {}
        claves_vistas: dict[tuple[str, ...], int] = {}
        # Los avisos se agrupan en lugar de dar uno por celda.
        rutas_absolutas: list[str] = []

        for numero, fila in enumerate(rows, start=2):  # 2 es la primera fila tras la cabecera
            estado = fila.get("status", "")
            if "status" in fila:
                recuento_estados[estado or "(vacío)"] = (
                    recuento_estados.get(estado or "(vacío)", 0) + 1
                )
                if estado not in VALID_STATUS:
                    report.errors.append(
                        f"{table} línea {numero}: status='{estado}' no es válido "
                        f"(esperado: {', '.join(VALID_STATUS)})."
                    )

            for columna, valor in fila.items():
                if valor.strip().lower() in FORBIDDEN_NULL_TEXTS:
                    report.errors.append(
                        f"{table} línea {numero}, columna '{columna}': el valor "
                        f"'{valor}' codifica una ausencia como texto; debe ser celda vacía."
                    )

            # Las rutas absolutas atan el registro a una máquina. append ya las normaliza, así que
            # vienen de filas antiguas, importaciones o ediciones a mano.
            for columna in TABLE_PATH_COLUMNS.get(table, ()):
                if is_absolute_path_text(fila.get(columna, "")):
                    rutas_absolutas.append(f"línea {numero}/{columna}")

            clave = tuple(fila.get(c, "") for c in TABLE_PRIMARY_KEYS[table])
            if clave in claves_vistas:
                report.warnings.append(
                    f"{table}: clave duplicada {clave} en las líneas "
                    f"{claves_vistas[clave]} y {numero}."
                )
            else:
                claves_vistas[clave] = numero

        if rutas_absolutas:
            report.warnings.append(
                f"{table}: {len(rutas_absolutas)} celdas con ruta absoluta "
                f"({', '.join(rutas_absolutas)}). El registro es un índice sobre ESTA "
                "máquina: una ruta absoluta lo ata a ella y no sobrevive a moverlo. "
                "Las que apunten fuera del árbol hay que reapuntarlas al "
                "ejemplar local (scripts/14_import_colab_registry.py)."
            )

        if recuento_estados:
            report.status_counts[table] = recuento_estados

    def _validar_referencias(
        self, tables: Mapping[str, Sequence[Mapping[str, str]]], report: RegistryValidationReport
    ) -> None:
        """Comprueba que no hay identificadores huérfanos entre tablas."""
        experimentos = {fila.get("experiment_id", "") for fila in tables.get("experiments", [])}
        entornos = {fila.get("environment_id", "") for fila in tables.get("environments", [])}

        for tabla in ("quality_metrics", "deployment_metrics"):
            for numero, fila in enumerate(tables.get(tabla, []), start=2):
                experiment_id = fila.get("experiment_id", "")
                if experiment_id and experiment_id not in experimentos:
                    report.errors.append(
                        f"{tabla} línea {numero}: experiment_id '{experiment_id}' no "
                        "existe en experiments.csv (identificador huérfano)."
                    )
                environment_id = fila.get("environment_id", "")
                if environment_id and environment_id not in entornos:
                    report.errors.append(
                        f"{tabla} línea {numero}: environment_id '{environment_id}' no "
                        "existe en environments.csv (identificador huérfano)."
                    )

    def initialize_empty(self, tables: Iterable[str] | None = None) -> list[Path]:
        """Crea los CSV que falten, solo con la cabecera."""
        creados: list[Path] = []
        for tabla in tables if tables is not None else TABLE_FILENAMES:
            if tabla == "quality_metrics" and self._class_names is None:
                logger.info(
                    "No se inicializa quality_metrics.csv: hacen falta los nombres de "
                    "clase, que se generan al preparar el dataset."
                )
                continue
            if self.path_for(tabla).is_file():
                continue
            creados.append(self._write_rows_atomic(tabla, []))
            logger.info("Tabla '%s' inicializada vacía.", tabla)
        return creados
