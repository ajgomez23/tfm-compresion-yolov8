#!/usr/bin/env python
"""Dibuja las figuras de resultados a partir del volcado del paso 29.

Genera seis figuras del capítulo 5 (curvas de entrenamiento, mAP frente al nivel de compresión,
comparativa por técnica, degradación por clase y por tamaño, tipos de error y coste frente a
ganancia), la de pérdida de mAP frente al FP32 de su tamaño (6.2) y la de validación frente a prueba
(6.7). Cada una se nombra por su capítulo y apartado.

La del 6.7 es la única que lee filas de test: cada cifra del volcado se busca en el registro por
clave primaria completa y tiene que coincidir. Las del 6.2, por clase y de tipos de error también se
contrastan con el registro, y el paso aborta si no coinciden. La de tipos de error solo cubre las
filas con detecciones por instancia en disco evaluadas con pycocotools; su cobertura va al
manifiesto.

Cada figura indica su alcance: la de curvas enumera en el pie los entrenamientos sin ``results.csv``
y la de coste las configuraciones sin coste registrado, en vez de dibujarlas en el cero. Las del 5.5
y la del 6.2 no llevan pie en la imagen; su texto queda en ``caption`` del manifiesto. Nada se
recalcula aquí.

Códigos de salida: 0 si se generan las figuras posibles, 1 si hay un error de configuración o las
figuras ya existen sin indicar qué hacer, 2 si los datos de entrada no permiten generarlas.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Matplotlib lee SOURCE_DATE_EPOCH al importarse; se fija antes para que dos ejecuciones iguales den
# el mismo PDF.
from tfm_edge.reporting import style as _style  # noqa: E402

_style.freeze_creation_date()

from tfm_edge.analysis import degradation  # noqa: E402
from tfm_edge.analysis import pareto  # noqa: E402
from tfm_edge.analysis.coherence import assert_analysis_matches_registry  # noqa: E402
from tfm_edge.analysis.splits import analysis_quality_rows  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.evaluation.error_decomposition import DUMP_SUBDIR, dump_name  # noqa: E402
from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE  # noqa: E402
from tfm_edge.reporting import figures_results  # noqa: E402

logger = get_logger(__name__)

#: Nombre del volcado que produce el paso 29.
ANALYSIS_NAME = "results_analysis.json"

#: Nombre del volcado de validación frente a test, en docs/evidencia.
SPLIT_CHECK_NAME = "d193_evaluacion_test.json"

#: Nombre del manifiesto de este paso.
MANIFEST_NAME = "results_figures_manifest.json"

#: Ejes que necesita la figura comparativa.
COMPARISON_OBJECTIVES: tuple[pareto.Objective, ...] = (
    pareto.Objective("mAP@50-95", "map_50_95", "quality", "max"),
    pareto.Objective("latencia p95", "latency_p95_ms", "deployment", "min"),
    pareto.Objective("memoria pico", "peak_host_memory_mb", "deployment", "min"),
)

#: Segundos en una hora.
SECONDS_PER_HOUR = 3600.0


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="30_build_results_figures.py",
        description=(
            "Dibuja las figuras del capítulo 5 y la de pérdida por compresión del 6.2 a partir "
            "del volcado del paso 29, y la de validación frente a prueba del 6.7 a partir de su "
            "volcado. El pie de cada una indica qué parte de la campaña cubre."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Todas las figuras que hoy se pueden generar\n"
            "  python scripts/30_build_results_figures.py\n\n"
            "  # Regenerar y comprobar si alguna ha cambiado\n"
            "  python scripts/30_build_results_figures.py --on-existing overwrite\n"
            "  git status --short results/figures/\n"
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directorio del volcado del paso 29 (por defecto: results/raw/analysis). El "
            "manifiesto de este paso se escribe aquí; para una ejecución aislada hay que "
            "copiar el volcado a un temporal."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Dónde buscar las curvas de entrenamiento (por defecto: models).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dónde dejar las figuras (por defecto: results/figures).",
    )
    parser.add_argument(
        "--error-decomposition-dir",
        type=Path,
        default=None,
        help=(
            "Directorio del volcado del paso 34 (por defecto: "
                f"results/raw/{DUMP_SUBDIR}). Sin él no sale la figura de tipos de error."
        ),
    )
    parser.add_argument(
        "--coco-eval-dir",
        type=Path,
        default=None,
        help=(
            "Directorio de los volcados de evaluación de los baselines (por defecto: "
            "results/raw/coco_eval). De ahí sale el número de anotaciones por clase."
        ),
    )
    parser.add_argument(
        "--split-check",
        type=Path,
        default=None,
        help=(
            "Volcado de validación frente a test (por defecto: "
                f"docs/evidencia/{SPLIT_CHECK_NAME}). Sin él no sale la figura del 6.7."
        ),
    )
    parser.add_argument(
        "--on-existing",
        default="overwrite",
        choices=("fail", "skip", "overwrite"),
        help=(
            "Qué hacer si las figuras ya existen (por defecto: overwrite). Las figuras se "
            "pueden regenerar a partir del volcado."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Comprueba las entradas y resume, pero no escribe ninguna figura.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: INFO).",
    )
    return parser


def cargar_analisis(analysis_dir: Path) -> dict[str, Any]:
    """Carga el volcado del paso 29. Lanza FileNotFoundError si no está."""
    ruta = analysis_dir / ANALYSIS_NAME
    if not ruta.is_file():
        raise FileNotFoundError(
            f"No está el volcado del análisis en {ruta}. Ejecuta antes "
            "scripts/29_analyze_results.py."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_reparto(decomposition_dir: Path) -> dict[str, Any] | None:
    """Carga el volcado del paso 34, o devuelve ``None`` si no está (solo lo necesita una figura).
    """
    ruta = decomposition_dir / dump_name(REFERENCE_CONFIDENCE)
    if not ruta.is_file():
        logger.warning("No está el volcado del paso 34 en %s.", ruta)
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def cargar_contraste(ruta: Path) -> dict[str, Any] | None:
    """Carga el volcado de validación frente a test, o devuelve ``None`` si no está."""
    if not ruta.is_file():
        logger.warning("No está el volcado de val frente a test en %s.", ruta)
        return None
    return json.loads(ruta.read_text(encoding="utf-8"))


def filas_comparativa(
    quality_rows: list[dict[str, str]],
    deployment_rows: list[dict[str, str]],
    experiments: dict[str, dict[str, str]],
    device_id: str,
) -> list[dict[str, Any]]:
    """Construye las filas de la figura comparativa por técnica para un dispositivo."""
    puntos = pareto.build_points(quality_rows, deployment_rows, COMPARISON_OBJECTIVES)
    filas: list[dict[str, Any]] = []
    for punto in puntos:
        if punto.device_id != device_id:
            continue
        experimento = experiments.get(punto.experiment_id, {})
        filas.append(
            {
                "experiment_id": punto.experiment_id,
                "technique": (experimento.get("technique", "") or "").strip(),
                "runtime": punto.runtime,
                "precision": punto.precision,
                **dict(punto.values),
            }
        )
    return filas


def costes_de_exportacion(
    deployment_rows: list[dict[str, str]]
) -> dict[tuple[str, str, str], float]:
    """Indexa el coste de exportación por ``(experimento, motor, precisión)``.

    Es una clave más corta que la de la tabla. Dos filas pueden compartirla (el mismo artefacto
    medido en dos dispositivos), pero si declaran costes distintos se lanza ValueError.
    """
    indice: dict[tuple[str, str, str], float] = {}
    for fila in deployment_rows:
        bruto = (fila.get("export_time_s") or "").strip()
        if not bruto:
            continue
        clave = (fila["experiment_id"], fila["runtime"], fila["precision"])
        valor = float(bruto)
        anterior = indice.get(clave)
        if anterior is not None and abs(anterior - valor) > 1e-9:
            raise ValueError(
                f"Dos filas de despliegue con clave {clave} declaran costes de exportación "
                f"distintos ({anterior} s y {valor} s)."
            )
        indice[clave] = valor
    return indice


def filas_coste(
    analysis: dict[str, Any],
    experiments: dict[str, dict[str, str]],
    export_costs: dict[tuple[str, str, str], float],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Construye las filas de la figura de coste frente a ganancia.

    Solo entran las configuraciones con coste registrado en toda su cadena; las demás se devuelven
    aparte con el motivo. El coste de PTQ y combinación depende del motor, por eso se busca con la
    terna completa.
    """
    dibujables: list[dict[str, Any]] = []
    excluidas: dict[str, str] = {}

    for fila in analysis["degradation"]["rows"]:
        experiment_id = fila["experiment_id"]
        exportacion = export_costs.get((experiment_id, fila["runtime"], fila["precision"]))
        coste = degradation.chain_cost(
            experiment_id,
            experiments,
            extra_costs={experiment_id: exportacion} if exportacion is not None else None,
        )
        # La etiqueta lleva la etapa y la precisión para distinguir las podas sin reajustar y el
        # FP32 del INT8 de un mismo QAT.
        etiqueta = (
            f"{experiment_id}{figures_results.stage_suffix(fila)} "
            f"[{fila['runtime']}/{fila['precision']}]"
        )
        if coste.seconds is None:
            excluidas.setdefault(etiqueta, coste.reason)
            continue

        # Sin eficiencia en p95 falta latencia en un lado; con ella pero sin certificar, el factor
        # no supera la resolución del eje. El pie cuenta los dos casos por separado.
        eficiencia = fila["efficiency"].get("latency_p95_ms")
        if eficiencia is None:
            estado = figures_results.SPEEDUP_NO_LATENCY
        elif eficiencia["certified"]:
            estado = figures_results.SPEEDUP_CERTIFIED
        else:
            estado = figures_results.SPEEDUP_NOT_CERTIFIED
        aceleracion = eficiencia["factor"] if estado == figures_results.SPEEDUP_CERTIFIED else None
        calidad = fila["overall"].get("map_50_95")
        dibujables.append(
            {
                "label": etiqueta,
                "technique": fila["technique"],
                "cost_hours": coste.seconds / SECONDS_PER_HOUR,
                "speedup": aceleracion,
                "speedup_status": estado,
                "quality_delta": calidad["absolute"] if calidad is not None else None,
            }
        )

    return dibujables, sorted(excluidas.items())


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("30_build_results_figures", level=args.log_level)

    try:
        rutas = get_project_paths()
        registry_dir = args.registry_dir or rutas.results_registry
        analysis_dir = args.analysis_dir or (rutas.results_raw / "analysis")
        models_dir = args.models_dir or (rutas.root / "models")
        output_dir = args.output_dir or rutas.results_figures
        reparto_dir = args.error_decomposition_dir or (rutas.results_raw / DUMP_SUBDIR)
        coco_eval_dir = args.coco_eval_dir or (rutas.results_raw / "coco_eval")
        contraste_path = args.split_check or (rutas.root / "docs" / "evidencia" / SPLIT_CHECK_NAME)

        analisis = cargar_analisis(analysis_dir)
        volcado_reparto = cargar_reparto(reparto_dir)
        volcado_contraste = cargar_contraste(contraste_path)

        registro = ExperimentRegistry(registry_dir)
        experimentos, _ = registro.valid_rows("experiments")
        calidad, _ = analysis_quality_rows(registro)
        despliegue, _ = registro.latency_comparable_rows()

        # La misma guarda que el paso 28.
        assert_analysis_matches_registry(analisis, registro, step="scripts/29_analyze_results.py")

        indice = {fila["experiment_id"]: fila for fila in experimentos}
        dispositivo = analisis.get("selection", {}).get("device_id", "")
        comparativa = (
            filas_comparativa(calidad, despliegue, indice, dispositivo) if dispositivo else []
        )
        coste, coste_excluidas = filas_coste(analisis, indice, costes_de_exportacion(despliegue))
        inventario = figures_results.training_curve_inventory(models_dir, experimentos)
        # Un registro y un volcado que no cuentan lo mismo abortan el paso entero,
        # no solo esta figura: es una contradicción entre la cifra y su evidencia.
        reparto = (
            figures_results.error_decomposition_rows(calidad, indice, volcado_reparto)
            if volcado_reparto is not None
            else None
        )
        # Lo mismo para la figura por clase: si los volcados de evaluación
        # discrepan, o si la referencia reconstruida no es la del paso 29, aborta
        # el paso entero.
        instancias = figures_results.class_instances(coco_eval_dir)
        dispositivos = figures_results.comparison_devices(analisis["degradation"]["rows"], calidad)
        # Y para la de pérdida por compresión, por la misma razón.
        compresiones = figures_results.compression_losses(
            analisis["degradation"]["rows"], indice, calidad
        )
        # Y para la de validación frente a test, buscando cada fila con la partición en la clave.
        contraste = None
        if volcado_contraste is not None:
            calidad_ambas, _ = registro.valid_rows("quality_metrics")
            contraste = figures_results.split_check_entries(
                volcado_contraste, indice, calidad_ambas
            )

        # Las figuras se buscan por nombre porque son de dos capítulos.
        candidatas = [
            output_dir / f"{_style.figure_filename(capitulo, apartado, clave)}.pdf"
            for clave, (capitulo, apartado) in figures_results.PLACEMENT.items()
        ]
        existentes = sorted(ruta for ruta in candidatas if ruta.is_file())
        if existentes and not args.dry_run:
            if args.on_existing == "skip":
                logger.info("Ya hay figuras de este paso y --on-existing=skip.")
                print(f"Sin cambios: ya existen {len(existentes)} figuras de este paso.")
                return 0
            if args.on_existing == "fail":
                logger.error(
                    "Ya hay %d figuras de este paso en %s. Con --on-existing=fail no "
                    "se tocan.",
                    len(existentes),
                    output_dir,
                )
                return 1

    except TFMEdgeError as exc:
        logger.error("%s", exc)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        logger.error("Error de configuración o de entrada/salida: %s", exc)
        return 1

    if args.dry_run:
        print(f"Configuraciones para la comparativa: {len(comparativa)}")
        print(f"Configuraciones con coste registrado: {len(coste)}")
        print(f"Compresiones de la figura de pérdida: {len(compresiones)}")
        if contraste is None:
            print("Val frente a test: sin volcado, la figura del 6.7 no saldría.")
        else:
            print(f"Val frente a test: {len(contraste['entries'])} puntos.")
        print(f"Configuraciones sin coste registrado: {len(coste_excluidas)}")
        print(f"Curvas de entrenamiento encontradas: {len(inventario.curves)}")
        for identificador, motivo in inventario.missing:
            print(f"    sin curva: {identificador}: {motivo}")
        if reparto is None:
            print("Reparto de errores: sin volcado del paso 34, la figura no saldría.")
        else:
            print(f"Reparto de errores: {reparto.coverage()}")
        print("\n--dry-run: no se ha escrito ninguna figura.")
        return 0

    try:
        manifiesto = figures_results.save_results_figures(
            analisis,
            comparativa,
            coste,
            coste_excluidas,
            inventario,
            output_dir,
            error_decomposition=reparto,
            instances=instancias,
            devices=dispositivos,
            compression=compresiones,
            experiments=indice,
            split_check=contraste,
        )
    except TFMEdgeError as exc:
        logger.error("%s", exc)
        return 2

    # El manifiesto se escribe junto al volcado leído, para no escribir en el árbol real al trabajar
    # con directorios temporales.
    destino = ensure_parent_dir(analysis_dir / MANIFEST_NAME)
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    temporal.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporal, destino)

    print(f"Figuras generadas: {len(manifiesto['figures'])}")
    for clave in sorted(manifiesto["figures"]):
        print(f"    {_style.figure_filename(*figures_results.PLACEMENT[clave], clave)}")
    if manifiesto["skipped"]:
        print(f"\nNo generadas ({len(manifiesto['skipped'])}):")
        for clave, motivo in sorted(manifiesto["skipped"].items()):
            print(f"    {clave}: {motivo}")
    if inventario.missing:
        print(f"\nEntrenamientos sin curva ({len(inventario.missing)}):")
        for identificador, motivo in inventario.missing:
            print(f"    {identificador}: {motivo}")
    print(f"\nManifiesto: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
