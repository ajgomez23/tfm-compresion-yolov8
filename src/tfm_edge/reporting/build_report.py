"""Genera de una vez las tablas y figuras de la memoria.

Lleva un catálogo de lo que necesita la memoria, con el paso que produce cada pieza, y el manifiesto
distingue lo generado de lo que falta. Las tablas usan los mismos filtros que el análisis de Pareto
(valid_rows y latency_comparable_rows) y el recuento se contrasta con el volcado del paso 26. Los
suelos de medida se leen de ese volcado.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tfm_edge.analysis import pareto
from tfm_edge.analysis.coherence import assert_analysis_matches_registry
from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, parse_experiment_id
from tfm_edge.common.registry import ExperimentRegistry
from tfm_edge.reporting import style, tables

logger = get_logger(__name__)

ANALYSIS_NAME = "pareto_analysis.json"

MANIFEST_NAME = "report_manifest.json"

#: Capítulo y apartado de las tablas de medidas y de frontera; la de frontera comparte apartado con
#: sus figuras.
MEASUREMENTS_CHAPTER, MEASUREMENTS_SECTION = 6, 1
FRONTIER_CHAPTER, FRONTIER_SECTION = 6, 4

#: Bytes por MB, decimal, como se citan los tamaños de fichero.
BYTES_PER_MB = 1_000_000.0

#: Clave con la huella de entorno de la fila de despliegue. No se imprime; la usa tables.best_marks
#: para elegir el suelo.
ENVIRONMENT_KEY = "environment_id"


@dataclass(frozen=True)
class ReportItem:
    """Pieza del informe y el paso que la produce, si existe."""

    key: str
    kind: str
    chapter: int
    section: int
    description: str
    producer: str
    available: bool
    note: str = ""


#: Lo que necesita la memoria. Lo que no está disponible se indica con el módulo que lo producirá.
CATALOGUE: tuple[ReportItem, ...] = (
    ReportItem(
        key="introduccion_esquemas",
        kind="figura",
        chapter=1,
        section=1,
        description=(
            "Esquemas de la introducción: el problema y las tres técnicas (1.1), el "
            "espacio de decisión (1.2) y el flujo del estudio (1.3)"
        ),
        producer="scripts/35_build_intro_figures.py",
        available=True,
        note=(
            "El espacio de decisión es esquemático: sus puntos son sintéticos y sus "
            "ejes no llevan marcas. El flujo y las técnicas se contrastan con el "
            "registro en cada ejecución (D-191)."
        ),
    ),
    ReportItem(
        key="introduccion_muestra",
        kind="figura",
        chapter=1,
        section=1,
        description="Muestra de SH17 anotada, con cabezas y caras difuminadas",
        producer="scripts/35_build_intro_figures.py",
        available=True,
        note=(
            "Necesita data/, que no se versiona. La atribución y la licencia de cada "
            "imagen están en results/raw/intro/intro_figures_manifest.json y van en el "
            "pie, en el texto (D-191)."
        ),
    ),
    ReportItem(
        key="introduccion_resultado",
        kind="figura",
        chapter=1,
        section=2,
        description=(
            "El resultado principal: la frontera de mAP@50-95 frente a la latencia p50 "
            "por motor, simplificada a partir de la del 6.4"
        ),
        producer="scripts/27_build_pareto_figures.py",
        available=True,
        note=(
            "La dibuja el paso 27 porque sale de sus fronteras ya contrastadas con el "
            "volcado del paso 26; su nombre empieza por fig_1_2 (D-191)."
        ),
    ),
    ReportItem(
        key="dataset",
        kind="figura",
        chapter=3,
        section=4,
        description=(
            "Figuras del dataset: ejemplos anotados (3.2), particionado y "
            "redimensionado (3.3), distribuciones y coocurrencia (3.4)"
        ),
        producer="scripts/03_dataset_eda.py",
        available=True,
        note=(
            "Se generan en su propio paso; aquí solo se comprueba que están. "
            "El apartado de la ficha es el del grueso de las figuras: el paso "
            "produce las de 3.2, 3.3 y 3.4 (D-146)."
        ),
    ),
    ReportItem(
        key="diseno",
        kind="figura",
        chapter=4,
        section=5,
        description=(
            "Figuras del diseño del estudio: sesgo de selección de best.pt (4.1), "
            "cadena de causas QAT (4.2), flujo experimental y suelo de error de "
            "latencia (4.3) y cobertura perfil × motor (4.5)"
        ),
        producer="scripts/33_build_design_figures.py",
        available=True,
        note="Se generan en su propio paso; aquí solo se comprueba que están.",
    ),
    ReportItem(
        key="frontera_figuras",
        kind="figura",
        chapter=6,
        section=4,
        description=(
            "Frontera de Pareto por dispositivo y versión con la memoria en el marcador; "
            "y por motor: frontera rotulada, tres objetivos, nano frente a small, "
            "dominaciones perdidas por el suelo y pertenencia por técnica"
        ),
        producer="scripts/27_build_pareto_figures.py",
        available=True,
    ),
    ReportItem(
        key="medidas",
        kind="tabla",
        chapter=MEASUREMENTS_CHAPTER,
        section=MEASUREMENTS_SECTION,
        description="Calidad y despliegue de cada configuración medida, por motor",
        producer="tfm_edge.reporting.build_report",
        available=True,
    ),
    ReportItem(
        key="frontera_tabla",
        kind="tabla",
        chapter=FRONTIER_CHAPTER,
        section=FRONTIER_SECTION,
        description="Situación de cada punto respecto a la frontera, por vista",
        producer="tfm_edge.reporting.build_report",
        available=True,
    ),
    ReportItem(
        key="entrenamiento",
        kind="figura",
        chapter=5,
        section=1,
        description="Curvas de pérdida y de mAP por época de cada modelo",
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "Cubre las nueve ejecuciones que dejaron results.csv en models/. Faltan "
            "siete, entre ellas la del baseline principal: de esas ejecuciones "
            "volvieron los pesos pero no la curva, y la figura lo declara en su pie."
        ),
    ),
    ReportItem(
        key="familias",
        kind="figura",
        chapter=5,
        section=3,
        description="mAP frente a nivel de compresión por familia",
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "Sin las barras de error que pide la especificación: no hay dispersión "
            "medida (D-110). En su lugar se dibuja el suelo del eje, que sí está "
            "medido, con su nombre y sin fingir que es lo mismo."
        ),
    ),
    ReportItem(
        key="comparativas",
        kind="figura",
        chapter=5,
        section=4,
        description="Barras agrupadas de mAP, latencia y memoria por técnica",
        producer="scripts/30_build_results_figures.py",
        available=True,
    ),
    ReportItem(
        key="degradacion",
        kind="figura",
        chapter=5,
        section=5,
        description=(
            "Retención de AP por clase, una figura por motor (ONNX Runtime y OpenVINO "
            "para el cuerpo, PyTorch para el anexo), y retención por tamaño de objeto"
        ),
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "En el cuerpo, las compresiones de la figura del 6.2, en su orden y con sus "
            "etiquetas; lo de PyTorch, en el anexo. Sin pie dentro de la imagen: el pie "
            "va en el texto, y el que se calcula de los datos está en el manifiesto del "
            "paso 30, en 'caption' (D-190)."
        ),
    ),
    ReportItem(
        key="tipos_de_error",
        kind="figura",
        chapter=5,
        section=5,
        description="Descomposición de los tipos de error de detección",
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "No cubre toda la campaña: solo las filas de calidad cuyas detecciones "
            "por instancia quedaron en disco y se evaluaron con pycocotools. El "
            "manifiesto del paso 30 da la cobertura, cruzando el registro con el "
            "volcado del paso 34, y la figura marca las cifras que no tienen el "
            "contraste contra la curva de operación (D-183). Sin pie dentro de la "
            "imagen: el que se calcula de los datos está en 'caption' (D-190)."
        ),
    ),
    ReportItem(
        key="coste",
        kind="figura",
        chapter=5,
        section=6,
        description="Coste de entrenamiento o compresión frente a la ganancia obtenida",
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "Solo con las configuraciones cuya cadena entera tiene coste registrado. "
            "Desde D-132 eso incluye las doce combinaciones, cuyo coste de cuantizar "
            "estaba medido en los volcados y ahora está en el registro. Siguen fuera "
            "las seis podas sin reajustar: no se exportaron y podar no se cronometró."
        ),
    ),
    ReportItem(
        key="perdida_por_compresion",
        kind="figura",
        chapter=6,
        section=2,
        description=(
            "Pérdida de mAP@50-95 de cada compresión frente al FP32 de su tamaño, en "
            "ONNX Runtime y OpenVINO con el mismo orden de filas"
        ),
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "Sin pie dentro de la imagen: el pie y la nota van en el texto. Los pares "
            "vecinos que no superan el suelo de lectura y las inversiones de orden "
            "entre motores están en el manifiesto del paso 30 (D-188)."
        ),
    ),
    ReportItem(
        key="validacion_frente_a_prueba",
        kind="figura",
        chapter=6,
        section=7,
        description=(
            "mAP@50-95 en validación frente a prueba de las 23 configuraciones de ONNX "
            "Runtime y OpenVINO, con la diagonal y la banda del suelo de lectura"
        ),
        producer="scripts/30_build_results_figures.py",
        available=True,
        note=(
            "Sin pie dentro de la imagen. Sale del volcado de D-193, contrastado cifra a "
            "cifra con el registro; test no decide nada. Los puntos dentro de la banda y "
            "la colocación de cada rótulo están en el manifiesto del paso 30 (D-194)."
        ),
    ),
)


#: Comprobaciones que este paso no hace; se imprimen en cada ejecución.
PENDING_VERIFICATIONS: tuple[str, ...] = (
    "Este paso no compila los .tex que escribe. Los tests compilan tablas de prueba "
    "con pdflatex cuando está instalado, y las de la campaña se compilaron y se "
    "miraron a mano el 18/09/2026; si cambian las filas, hay que volver a "
    "compilarlas y mirarlas (D-130).",
    "Las tablas van apaisadas y reducidas hasta caber en una página: con cincuenta "
    "filas quedan entre 6,4 y 6,6 pt de letra. Ese PDF sirve para verificar; la memoria "
    "se monta en Word a partir del XLSX, que no se reduce (D-130).",
    "Las figuras se han revisado abriéndolas, pero no hay comprobación automática de "
    "solapamientos de texto ni de legibilidad en escala de grises (D-129, D-131).",
)


def pending_items() -> tuple[ReportItem, ...]:
    """Piezas que todavía no se pueden generar."""
    return tuple(item for item in CATALOGUE if not item.available)


def load_analysis(analysis_dir: Path | str) -> dict[str, Any]:
    """Carga el volcado del paso 26. Si no está, falla."""
    ruta = Path(analysis_dir) / ANALYSIS_NAME
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No está el volcado del análisis de Pareto en {ruta}. Ejecuta antes "
                f"scripts/26_analyze_pareto.py."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def floor_of(analysis: Mapping[str, Any], view: str, column: str) -> dict[str, Any] | None:
    """Suelo medido de una columna según el volcado del paso 26, o None."""
    suelos = analysis.get("views", {}).get(view, {}).get("floors", {}).get("floors", {})
    suelo = suelos.get(column)
    if suelo is None or suelo.get("cross_session") is None:
        return None
    return suelo


def _floor_kwargs(analysis: Mapping[str, Any], view: str, column: str) -> dict[str, Any]:
    """Argumentos de suelo para una columna de tabla.

    Se pasan los dos suelos: el de entre sesiones como base y el de la misma sesión como
    ``floor_same_session``; tables.best_marks elige en cada pareja según las huellas de entorno.
    """
    suelo = floor_of(analysis, view, column)
    if suelo is None:
        return {}
    valor = float(suelo["cross_session"])
    clave = "floor_relative" if suelo.get("kind") == "relativo" else "floor"
    if suelo.get("same_session") is None:
        return {clave: valor, "floor_note": f"entre sesiones, {suelo['provenance']}"}
    return {
        clave: valor,
        "floor_same_session": float(suelo["same_session"]),
        "floor_note": str(suelo["provenance"]),
    }


def _technique(experiment_id: str) -> str:
    """Etiqueta de la familia de técnica, la misma que usan las figuras."""
    return style.style_for(parse_experiment_id(experiment_id)["technique"]).label


def _number(row: Mapping[str, str], column: str) -> str:
    """Valor de una celda sin convertir, o cadena vacía."""
    return (row.get(column, "") or "").strip()


def build_measurements_tables(
    quality_rows: Sequence[Mapping[str, str]],
    deployment_rows: Sequence[Mapping[str, str]],
    analysis: Mapping[str, Any],
    device_id: str,
) -> tuple[tables.Table, tables.Table]:
    """Tablas de medidas de un dispositivo: calidad y despliegue.

    El cruce usa las mismas claves que pareto.build_points. Son dos tablas con las mismas filas
    porque una sola no cabía en la página; las columnas que identifican la fila van en las dos.
    """
    indice: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for fila in quality_rows:
        indice.setdefault(tuple(fila[c] for c in pareto.JOIN_KEYS), []).append(fila)

    filas: list[dict[str, Any]] = []
    sin_calidad = 0
    for despliegue in deployment_rows:
        if despliegue["device_id"] != device_id:
            continue
        clave = tuple(despliegue[c] for c in pareto.JOIN_KEYS)
        candidatas = indice.get(clave, [])
        if not candidatas:
            sin_calidad += 1
            continue
        if len(candidatas) > 1:
            raise ValueError(
                f"La configuración {clave} tiene {len(candidatas)} filas de calidad. "
                f"Elegir una sería inventarse el resultado."
            )
        calidad = candidatas[0]
        bytes_artefacto = _number(despliegue, "artifact_size_bytes")
        filas.append(
            {
                "experiment_id": despliegue["experiment_id"],
                "technique": _technique(despliegue["experiment_id"]),
                "runtime": despliegue["runtime"],
                "precision": despliegue["precision"],
                "map_50": _number(calidad, "map_50"),
                "map_50_95": _number(calidad, "map_50_95"),
                "latency_p50_ms": _number(despliegue, "latency_p50_ms"),
                "latency_p95_ms": _number(despliegue, "latency_p95_ms"),
                "throughput_img_s": _number(despliegue, "throughput_img_s"),
                "peak_host_memory_mb": _number(despliegue, "peak_host_memory_mb"),
                "artifact_size_mb": (
                    float(bytes_artefacto) / BYTES_PER_MB if bytes_artefacto else ""
                ),
                # No se imprime: decide el suelo de cada pareja.
                ENVIRONMENT_KEY: _number(despliegue, "environment_id"),
            }
        )
    if sin_calidad:
        logger.info(
            "Filas de despliegue sin calidad, fuera de la tabla de medidas: %d", sin_calidad
        )

    filas.sort(key=lambda f: (f["experiment_id"], f["runtime"]))
    identificacion = (
        tables.text_column("experiment_id", "Experimento"),
        tables.text_column("technique", "Técnica"),
        tables.text_column("runtime", "Motor"),
        tables.text_column("precision", "Precisión"),
    )
    calidad_columnas = (
        *identificacion,
        tables.quality_column(
            "map_50", "mAP@50", **_floor_kwargs(analysis, "plano_map50_p50", "map_50")
        ),
        tables.quality_column(
            "map_50_95",
            "mAP@50-95",
            **_floor_kwargs(analysis, "plano_map5095_p50", "map_50_95"),
        ),
    )
    despliegue_columnas = (
        *identificacion,
        tables.latency_column(
            "latency_p50_ms",
            "latencia p50",
            **_floor_kwargs(analysis, "plano_map5095_p50", "latency_p50_ms"),
        ),
        tables.latency_column(
            "latency_p95_ms",
            "latencia p95",
            **_floor_kwargs(analysis, "spec_3d_map5095_p95_memoria", "latency_p95_ms"),
        ),
        tables.Column(
            key="throughput_img_s",
            header="rendimiento",
            unit="img/s",
            decimals=tables.LATENCY_DECIMALS,
            better="max",
        ),
        tables.memory_column(
            "peak_host_memory_mb",
            "memoria pico",
            unit="MiB",
            **_floor_kwargs(analysis, "spec_3d_map5095_p95_memoria", "peak_host_memory_mb"),
        ),
        tables.size_column("artifact_size_mb", "tamaño del artefacto"),
    )

    notas_calidad = []
    if floor_of(analysis, "plano_map50_p50", "map_50") is None:
        notas_calidad.append(
            "mAP@50 no tiene suelo de medida medido, así que su mejor valor se marca "
            "por comparación exacta y no certifica ninguna ventaja (D-127)."
        )
    notas_despliegue = [
        "Latencia y memoria medidas con el protocolo homogéneo del capítulo 4; "
        "el rendimiento es a lote 1.",
    ]

    # Cincuenta filas no caben en una página apaisada; se reducen a 6,6 pt.
    calidad = tables.Table(
        name=f"medidas_calidad_{device_id}",
        caption=f"Calidad de cada configuración medida en {device_id}, por motor de inferencia",
        chapter=MEASUREMENTS_CHAPTER,
        section=MEASUREMENTS_SECTION,
        columns=calidad_columnas,
        rows=tuple(filas),
        notes=tuple(notas_calidad),
        latex_layout="landscape",
        environment_key=ENVIRONMENT_KEY,
    )
    despliegue = tables.Table(
        name=f"medidas_despliegue_{device_id}",
        caption=(
            f"Despliegue de cada configuración medida en {device_id}, por motor de inferencia"
        ),
        chapter=MEASUREMENTS_CHAPTER,
        section=MEASUREMENTS_SECTION,
        columns=despliegue_columnas,
        rows=tuple(filas),
        notes=tuple(notas_despliegue),
        latex_layout="landscape",
        environment_key=ENVIRONMENT_KEY,
    )
    return calidad, despliegue


def build_frontier_table(
    points: Sequence[pareto.Point],
    analysis: Mapping[str, Any],
    view: str,
    device_id: str,
) -> tables.Table:
    """Tabla de la frontera de una vista, leída del volcado del paso 26.

    Tiene dos columnas de frontera: frente a todos los motores del dispositivo y frente al mismo
    motor. Falla si los puntos no coinciden con los del volcado.
    """
    agrupacion = analysis.get("views", {}).get(view, {}).get("groupings", {}).get("device_id", {})
    if device_id not in agrupacion:
        raise ValueError(
            f"El volcado del paso 26 no tiene la vista '{view}' para '{device_id}'. Vuelve a "
            "ejecutar scripts/26_analyze_pareto.py."
        )
    grupo = agrupacion[device_id]
    estricta = set(grupo["frontier_strict"])
    robusta = set(grupo["frontier_robust"])
    distancias = grupo["distance_to_ideal"]
    por_motor = (
        analysis.get("views", {}).get(view, {}).get("groupings", {}).get("device_id+runtime", {})
    )

    etiquetas = {punto.label for punto in points}
    if etiquetas != set(distancias):
        faltan = sorted(etiquetas - set(distancias))
        sobran = sorted(set(distancias) - etiquetas)
        raise ValueError(
            f"Los puntos de la tabla y del volcado del paso 26 no coinciden en '{view}': "
            f"{len(faltan)} no están en el volcado ({faltan[:3]}) y {len(sobran)} solo están "
            f"allí ({sobran[:3]})."
        )

    def situacion(etiqueta: str, en_estricta: set[str], en_robusta: set[str]) -> str:
        if etiqueta in en_estricta:
            return "estricta"
        return "solo con suelo" if etiqueta in en_robusta else "dominado"

    objetivos = pareto.named_views()[view]
    filas = []
    for punto in points:
        clave_motor = f"{device_id}/{punto.runtime}"
        if clave_motor not in por_motor:
            raise ValueError(
                f"El volcado del paso 26 no tiene el grupo '{clave_motor}' de la vista '{view}'. "
                "Vuelve a ejecutar scripts/26_analyze_pareto.py."
            )
        del_motor = por_motor[clave_motor]
        if punto.label not in del_motor["distance_to_ideal"]:
            raise ValueError(
                f"'{punto.label}' no está en el grupo '{clave_motor}' del volcado del paso 26."
            )
        fila: dict[str, Any] = {
            "experiment_id": punto.experiment_id,
            "technique": _technique(punto.experiment_id),
            "runtime": punto.runtime,
            "precision": punto.precision,
            "frontier": situacion(punto.label, estricta, robusta),
            "frontier_runtime": situacion(
                punto.label,
                set(del_motor["frontier_strict"]),
                set(del_motor["frontier_robust"]),
            ),
            "distance_to_ideal": distancias[punto.label],
            ENVIRONMENT_KEY: punto.environment_id,
        }
        for objetivo in objetivos:
            fila[objetivo.column] = punto.values[objetivo.column]
        filas.append(fila)
    filas.sort(key=lambda f: (f["distance_to_ideal"], f["experiment_id"]))

    columnas: list[tables.Column] = [
        tables.text_column("experiment_id", "Experimento"),
        tables.text_column("technique", "Técnica"),
        tables.text_column("runtime", "Motor"),
        tables.text_column("precision", "Precisión"),
    ]
    for objetivo in objetivos:
        argumentos = _floor_kwargs(analysis, view, objetivo.column)
        if objetivo.source == "quality":
            columnas.append(
                tables.quality_column(
                    objetivo.column, objetivo.name, better=objetivo.direction, **argumentos
                )
            )
        elif objetivo.column.startswith("latency"):
            columnas.append(tables.latency_column(objetivo.column, objetivo.name, **argumentos))
        else:
            columnas.append(
                tables.memory_column(objetivo.column, objetivo.name, unit="MiB", **argumentos)
            )
    columnas.append(tables.text_column("frontier", "Frontera común"))
    columnas.append(tables.text_column("frontier_runtime", "En su motor"))
    columnas.append(
        tables.Column(
            key="distance_to_ideal",
            header="distancia al ideal",
            decimals=tables.RATIO_DECIMALS,
            better="min",
        )
    )

    return tables.Table(
        name=f"frontera_{view}_{device_id}",
        caption=(
            f"Situación de cada configuración respecto a la frontera de Pareto en "
            f"{device_id}, vista «{view}»"
        ),
        chapter=FRONTIER_CHAPTER,
        section=FRONTIER_SECTION,
        columns=tuple(columnas),
        rows=tuple(filas),
        notes=(
            "«estricta»: no dominada por nadie. «solo con suelo»: entra en la frontera "
            "únicamente cuando se exige que la ventaja del rival supere la resolución de "
            "la medida (D-128). «dominado»: hay otra configuración que no es peor en "
            "ningún eje y es mejor por encima del suelo de la medida en al menos uno.",
            "«Frontera común» compara cada configuración con las de todos los motores del "
            "dispositivo; «En su motor», solo con las del mismo motor.",
            "La distancia al ideal está normalizada sobre el rango observado: sirve para "
            "ordenar, no como magnitud física.",
        ),
        # De 742 a 849 pt de ancho frente a 455 de una página vertical: apaisadas y reducidas a 6,4
        # pt.
        latex_layout="landscape",
        environment_key=ENVIRONMENT_KEY,
    )


def build_tables(
    registry_dir: Path | str,
    analysis_dir: Path | str,
    device_id: str,
) -> list[tables.Table]:
    """Construye todas las tablas que se pueden generar, sin escribirlas."""
    analisis = load_analysis(analysis_dir)
    registro = ExperimentRegistry(Path(registry_dir))
    calidad, _ = analysis_quality_rows(registro)
    despliegue, _ = registro.latency_comparable_rows()

    # Se comprueban los cuatro contadores, no solo el de despliegue.
    assert_analysis_matches_registry(analisis, registro, step="scripts/26_analyze_pareto.py")

    construidas = list(build_measurements_tables(calidad, despliegue, analisis, device_id))
    for vista, objetivos in pareto.named_views().items():
        puntos = [
            punto
            for punto in pareto.build_points(calidad, despliegue, objetivos)
            if punto.device_id == device_id
        ]
        if not puntos:
            logger.warning(
                "La vista '%s' no tiene ningún punto en '%s': no se hace su tabla.",
                vista,
                device_id,
            )
            continue
        construidas.append(build_frontier_table(puntos, analisis, vista, device_id))
    return construidas


@dataclass(frozen=True)
class FigureStep:
    """Paso que dibuja figuras y el apartado del que es responsable.

    ``analysis_source`` indica qué volcado espera en ``--analysis-dir``: el 27 usa el del paso 26
    (results/raw/pareto) y el 30 el del paso 29 (results/raw/analysis).
    """

    script: str
    sections: str
    analysis_source: str


FIGURE_STEPS: tuple[FigureStep, ...] = (
    FigureStep("27_build_pareto_figures.py", "6.4", "pareto"),
    FigureStep("30_build_results_figures.py", "5.1 a 5.6", "results"),
)


def run_figures(
    project_root: Path,
    on_existing: str,
    *,
    registry_dir: Path | str,
    pareto_analysis_dir: Path | str,
    results_analysis_dir: Path | str,
    figures_dir: Path | str,
) -> dict[str, Any]:
    """Ejecuta los pasos que generan figuras como procesos aparte.

    Los cuatro directorios se pasan siempre de forma explícita, para que un directorio de salida
    temporal se aplique también a los procesos hijos.
    """
    volcados = {"pareto": pareto_analysis_dir, "results": results_analysis_dir}
    resultados: dict[str, Any] = {}
    for paso in FIGURE_STEPS:
        guion, apartados = paso.script, paso.sections
        orden = [
            sys.executable,
            str(project_root / "scripts" / guion),
            "--on-existing",
            on_existing,
            "--registry-dir",
            str(registry_dir),
            "--analysis-dir",
            str(volcados[paso.analysis_source]),
            "--output-dir",
            str(figures_dir),
        ]
        logger.info("Ejecutando el paso de figuras: %s", " ".join(orden))
        completado = subprocess.run(orden, cwd=project_root, check=False)
        if completado.returncode != 0:
            logger.error(
                "El paso %s ha terminado con código %d. Las figuras del apartado %s no "
                "se han regenerado.",
                guion,
                completado.returncode,
                apartados,
            )
        resultados[guion] = {
            "command": orden,
            "returncode": completado.returncode,
            "sections": apartados,
        }
    return resultados


def build(
    project_root: Path | str,
    registry_dir: Path | str,
    analysis_dir: Path | str,
    tables_dir: Path | str,
    figures_dir: Path | str,
    device_id: str,
    *,
    results_analysis_dir: Path | str | None = None,
    skip_figures: bool = False,
    on_existing: str = "overwrite",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Genera el informe: las figuras disponibles y las tablas de este paso.

    ``results_analysis_dir`` es obligatorio si se lanzan las figuras, porque su valor por defecto
    escribe en results/.
    """
    raiz = Path(project_root)
    if not skip_figures and not dry_run and results_analysis_dir is None:
        raise ValueError(
            "Para lanzar el paso 30 hace falta 'results_analysis_dir' (el volcado del paso 29)."
        )
    construidas = build_tables(registry_dir, analysis_dir, device_id)

    manifiesto: dict[str, Any] = {
        "device_id": device_id,
        "dry_run": dry_run,
        "tables": [],
        "figures": {},
        "pending_verifications": list(PENDING_VERIFICATIONS),
        "pending": [
            {
                "key": item.key,
                "kind": item.kind,
                "chapter": item.chapter,
                "section": item.section,
                "description": item.description,
                "producer": item.producer,
                "note": item.note,
            }
            for item in pending_items()
        ],
    }

    for tabla in construidas:
        resumen = tables.table_summary(tabla)
        if not dry_run:
            rutas = tables.save_table(tabla, tables_dir)
            resumen["files"] = style.manifest_files(rutas)
        manifiesto["tables"].append(resumen)

    if skip_figures:
        manifiesto["figures"] = {"skipped": True}
    elif dry_run:
        manifiesto["figures"] = {"skipped": True, "reason": "dry-run"}
    else:
        # Por la comprobación anterior, aquí results_analysis_dir no es None.
        manifiesto["figures"] = run_figures(
            raiz,
            on_existing,
            registry_dir=registry_dir,
            pareto_analysis_dir=analysis_dir,
            results_analysis_dir=results_analysis_dir,
            figures_dir=figures_dir,
        )

    manifiesto["figures_on_disk"] = sorted(
        ruta.name for ruta in Path(figures_dir).glob("fig_*.pdf")
    )
    # Los capítulos se deducen de los nombres de fichero, no de una lista fija.
    capitulos = sorted(
        {nombre.split("_")[1] for nombre in manifiesto["figures_on_disk"]}, key=int
    )
    manifiesto["figures_by_chapter"] = {
        capitulo: sorted(
            nombre
            for nombre in manifiesto["figures_on_disk"]
            if nombre.startswith(f"fig_{capitulo}_")
        )
        for capitulo in capitulos
    }
    return manifiesto


def write_manifest(manifest: Mapping[str, Any], output_dir: Path | str) -> Path:
    """Escribe el manifiesto del informe."""
    destino = ensure_dir(output_dir) / MANIFEST_NAME
    temporal = destino.with_name(destino.name + ".tmp")
    temporal.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporal, destino)
    return destino


def summarize(manifest: Mapping[str, Any]) -> list[str]:
    """Líneas del resumen para la consola."""
    lineas = []
    for tabla in manifest["tables"]:
        lineas.append(f"{tabla['basename']}  ({tabla['num_rows']} filas)")
        sin_distincion = tabla["columns_without_distinction"]
        if sin_distincion:
            lineas.append(
                f"    sin mejor valor que marcar: {', '.join(sin_distincion)} "
                f"(no hay diferencias por encima del suelo)"
            )
    figuras = manifest.get("figures", {})
    if figuras.get("skipped"):
        lineas.append("Figuras: no se han regenerado en esta pasada.")
    else:
        for guion, detalle in sorted(figuras.items()):
            estado = (
                "correcto" if detalle["returncode"] == 0 else f"FALLÓ ({detalle['returncode']})"
            )
            lineas.append(
                f"Figuras del apartado {detalle['sections']}: {guion} ejecutado, {estado}."
            )
    lineas.append("")
    lineas.append(f"Sin comprobar ({len(PENDING_VERIFICATIONS)}):")
    for aviso in PENDING_VERIFICATIONS:
        lineas.append(f"    {aviso}")
    lineas.append("")
    if not manifest["pending"]:
        # Se imprime aunque la lista esté vacía.
        lineas.append(
            "Pendiente de la memoria: ninguna pieza del catálogo. Las que no cubren "
            "toda la campaña lo dicen en su nota del catálogo y en el manifiesto de "
            "su paso."
        )
        return lineas
    lineas.append(f"Pendiente de la memoria ({len(manifest['pending'])} piezas):")
    for pieza in manifest["pending"]:
        lineas.append(
            f"    [{pieza['kind']} {pieza['chapter']}.{pieza['section']}] "
            f"{pieza['description']}: lo producirá {pieza['producer']}"
        )
        if pieza["note"]:
            lineas.append(f"        {pieza['note']}")
    return lineas
