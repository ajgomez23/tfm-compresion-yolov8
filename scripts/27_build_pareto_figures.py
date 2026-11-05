#!/usr/bin/env python
"""Dibuja las figuras de la frontera de Pareto del apartado 6.4.

Por dispositivo dibuja la dispersión de mAP frente a latencia con la frontera resaltada y la versión
con la memoria en el tamaño del marcador. Además dibuja cinco figuras por motor, con los miembros
rotulados: la frontera mAP@50-95/p50, la de tres objetivos, n FP32 y n PTQ frente a las compresiones
de small, las dominaciones que el suelo no sostiene y qué configuración entra en cada frontera.

Los puntos, suelos y fronteras salen de :mod:`tfm_edge.analysis.pareto` y
:mod:`tfm_edge.analysis.floors`, igual que en el paso 26, y se contrastan con su volcado si existe;
si discrepan, se aborta. Se marcan los puntos medidos en otra sesión (la mayoritaria se toma como
referencia) y se declara el suelo del eje de latencia en la figura. Los puntos marcados quedan en el
manifiesto.

Las figuras van a ``--output-dir`` y el manifiesto a ``--analysis-dir``, junto al volcado del paso
26, así que para una ejecución aislada hay que copiar también ese volcado a un temporal. Se fija
``SOURCE_DATE_EPOCH`` para que los PDF sean reproducibles.

Códigos de salida: 0 si se generan las figuras, 1 si hay un error de configuración o las figuras ya
existen sin indicar qué hacer, 2 si los datos de entrada no permiten dibujarlas.
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

# Matplotlib lee SOURCE_DATE_EPOCH al importarse, así que se fija antes.
from tfm_edge.reporting import style as _style  # noqa: E402

_style.freeze_creation_date()

from tfm_edge.analysis import floors as floors_mod  # noqa: E402
from tfm_edge.analysis import pareto  # noqa: E402
from tfm_edge.analysis.splits import analysis_quality_rows  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.reporting import figures_pareto, style  # noqa: E402

logger = get_logger(__name__)

#: Nombre del volcado del paso 26 contra el que se contrasta.
ANALYSIS_DUMP = "pareto_analysis.json"

#: Manifiesto de este paso: qué se ha dibujado y qué se ha marcado en cada panel.
MANIFEST_NAME = "figures_manifest.json"

#: Tipo de figura y fragmento del nombre de fichero de cada vista; los objetivos están en
#: pareto.named_views. La vista de tres objetivos solo genera la figura de memoria, porque su
#: frontera proyectada en dos ejes no es la frontera de esos dos ejes. El fragmento no se deduce del
#: nombre de la vista para no renombrar figuras existentes.
FIGURE_KINDS: dict[str, tuple[str, str]] = {
    "spec_3d_map5095_p95_memoria": ("memoria", "map5095_p95_memoria"),
    "plano_map50_p50": ("plano", "map50_p50"),
    "plano_map5095_p50": ("plano", "map5095_p50"),
}

FIGURES: dict[str, dict[str, Any]] = {
    nombre: {
        "objectives": objetivos,
        "kind": FIGURE_KINDS[nombre][0],
        "slug": FIGURE_KINDS[nombre][1],
    }
    for nombre, objetivos in pareto.named_views().items()
}


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="27_build_pareto_figures.py",
        description=(
            "Dibuja la frontera de Pareto del apartado 6.4 con el suelo de la medida en la "
            "figura y los puntos de otra sesión marcados."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Todas las figuras\n"
            "  python scripts/27_build_pareto_figures.py\n\n"
            "  # Solo el plano en el que razona el capítulo 5\n"
            "  python scripts/27_build_pareto_figures.py --view plano_map50_p50\n\n"
            "  # Ver qué se dibujaría sin tocar results/figures\n"
            "  python scripts/27_build_pareto_figures.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio de las figuras (por defecto: results/figures).",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help=(
            "Directorio del volcado del paso 26 con el que se contrasta lo dibujado (por "
            "defecto: results/raw/pareto). Si no está, se dibuja igual y se avisa. El "
            "manifiesto de este paso se escribe aquí; para una ejecución aislada hay que "
            "copiar el volcado a un temporal."
        ),
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=sorted(FIGURES),
        default=None,
        help="Vista a dibujar; repetible. Por defecto, todas.",
    )
    parser.add_argument(
        "--on-existing",
        choices=("fail", "skip", "overwrite"),
        default="fail",
        help=(
            "Qué hacer si la figura ya existe (por defecto: fail)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Calcula e informa sin escribir en results/figures ni el manifiesto. El log se "
            "escribe igualmente."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: INFO).",
    )
    return parser


def _cargar_puntos(
    registry_dir: Path, project_root: Path, objetivos: tuple[pareto.Objective, ...]
) -> tuple[tuple[pareto.Point, ...], floors_mod.FloorSet]:
    """Carga los puntos y mide los suelos de una vista. Lanza ParetoError si se queda sin puntos."""
    registro = ExperimentRegistry(registry_dir)
    calidad, _ = analysis_quality_rows(registro)
    despliegue, _ = registro.latency_comparable_rows()
    columnas = [o.column for o in objetivos]
    suelos = floors_mod.measure_floors(despliegue, columnas, project_root)
    puntos = pareto.build_points(calidad, despliegue, objetivos)
    if not puntos:
        raise pareto.ParetoError(
            f"No hay ningún punto con {columnas} medidos a la vez: no hay nada que dibujar."
        )
    return puntos, suelos


def _fronteras(
    puntos: list[pareto.Point],
    objetivos: tuple[pareto.Objective, ...],
    suelos: floors_mod.FloorSet,
) -> tuple[pareto.FrontierResult, pareto.FrontierResult]:
    """Calcula la frontera estricta y la robusta de un grupo."""
    estricta = pareto.frontier(puntos, objetivos, suelos, pareto.MODE_STRICT)
    robusta = pareto.frontier(puntos, objetivos, suelos, pareto.MODE_ROBUST)
    return estricta, robusta


def _discrepancias(
    esperado: dict[str, Any],
    estricta: pareto.FrontierResult,
    robusta: pareto.FrontierResult,
) -> list[str]:
    """Diferencias entre un grupo del volcado del paso 26 y lo recién calculado.

    Se comparan las dos fronteras y los tres recuentos de dominaciones, que también se dibujan.
    """
    obtenido = {
        "frontier_strict": sorted(p.label for p in estricta.nondominated),
        "frontier_robust": sorted(p.label for p in robusta.nondominated),
        "num_dominations_strict": len(estricta.dominations),
        "num_dominations_robust": len(robusta.dominations),
        "dominations_lost_to_floor": len(pareto.dominations_lost_to_floor(estricta, robusta)),
    }
    volcado = {
        "frontier_strict": sorted(esperado["frontier_strict"]),
        "frontier_robust": sorted(esperado["frontier_robust"]),
        "num_dominations_strict": esperado["num_dominations_strict"],
        "num_dominations_robust": esperado["num_dominations_robust"],
        "dominations_lost_to_floor": len(esperado["dominations_lost_to_floor"]),
    }
    return [
        f"{clave}: esperado {volcado[clave]}, obtenido {obtenido[clave]}"
        for clave in obtenido
        if obtenido[clave] != volcado[clave]
    ]


def _contrastar(
    vista: str,
    dispositivo: str,
    estricta: pareto.FrontierResult,
    robusta: pareto.FrontierResult,
    analisis: dict[str, Any] | None,
) -> None:
    """Comprueba que lo que se va a dibujar coincide con el volcado del paso 26."""
    if analisis is None:
        return
    grupo = analisis.get("views", {}).get(vista, {}).get("groupings", {}).get("device_id", {})
    if dispositivo not in grupo:
        logger.warning(
            "El volcado del paso 26 no tiene el grupo '%s' de la vista '%s': no "
            "se puede contrastar esta figura contra él.",
            dispositivo,
            vista,
        )
        return
    diferencias = _discrepancias(grupo[dispositivo], estricta, robusta)
    if diferencias:
        raise pareto.ParetoError(
            f"La figura de '{vista}' y el volcado del paso 26 no dicen lo mismo en "
            f"'{dispositivo}': {'; '.join(diferencias)}. Vuelve a ejecutar el paso 26."
        )
    logger.info(
        "Vista '%s', grupo '%s': la figura coincide con el volcado del paso 26.",
        vista,
        dispositivo,
    )


def _contrastar_motores(
    vista: str,
    dispositivo: str,
    fronteras_motor: dict[str, tuple[pareto.FrontierResult, pareto.FrontierResult]],
    analisis: dict[str, Any] | None,
) -> None:
    """Igual que :func:`_contrastar`, para cada panel por motor. Si el volcado existe y le falta el
    grupo, también se considera discrepancia.
    """
    if analisis is None:
        return
    grupos = analisis.get("views", {}).get(vista, {}).get("groupings", {}).get(
        "device_id+runtime", {}
    )
    for motor, (estricta, robusta) in sorted(fronteras_motor.items()):
        clave = f"{dispositivo}/{motor}"
        if clave not in grupos:
            raise pareto.ParetoError(
                f"El volcado del paso 26 no tiene el grupo '{clave}' de la vista '{vista}', "
                f"y el panel de ese motor no se puede contrastar. Vuelve a ejecutar el paso 26."
            )
        diferencias = _discrepancias(grupos[clave], estricta, robusta)
        if diferencias:
            raise pareto.ParetoError(
                f"El panel de '{motor}' en la vista '{vista}' y el volcado del paso 26 no dicen "
                f"lo mismo: {'; '.join(diferencias)}."
            )
    logger.info(
        "Vista '%s', dispositivo '%s': los paneles por motor coinciden con el volcado.",
        vista,
        dispositivo,
    )


def _cargar_analisis(analysis_dir: Path) -> dict[str, Any] | None:
    """Carga el volcado del paso 26, o devuelve ``None`` si no existe."""
    destino = analysis_dir / ANALYSIS_DUMP
    if not destino.is_file():
        logger.warning(
            "No hay volcado del paso 26 en %s. Se dibuja sin comprobar contra el análisis "
            "registrado. Ejecuta 'python scripts/26_analyze_pareto.py' para tener la "
            "comprobación.",
            destino,
        )
        return None
    return json.loads(destino.read_text(encoding="utf-8"))


def generar(
    registry_dir: Path,
    project_root: Path,
    analysis_dir: Path,
    vistas: list[str],
) -> tuple[list[tuple[str, Any]], dict[str, Any]]:
    """Construye todas las figuras pedidas y el manifiesto de lo dibujado."""
    style.apply_style()
    analisis = _cargar_analisis(analysis_dir)

    figuras: list[tuple[str, Any]] = []
    manifiesto: dict[str, Any] = {"views": {}}
    # Puntos, suelos y fronteras ya contrastadas, por dispositivo y vista.
    calculado: dict[str, dict[str, dict[str, Any]]] = {}

    for vista in vistas:
        receta = FIGURES[vista]
        objetivos = receta["objectives"]
        puntos, suelos = _cargar_puntos(registry_dir, project_root, objetivos)
        logger.info("Vista '%s': %d puntos.", vista, len(puntos))

        por_dispositivo = pareto.group_points(puntos, ("device_id",))
        manifiesto["views"][vista] = {}

        for clave, del_dispositivo in sorted(por_dispositivo.items()):
            dispositivo = clave[0]
            estricta, robusta = _fronteras(del_dispositivo, objetivos, suelos)
            _contrastar(vista, dispositivo, estricta, robusta, analisis)
            por_motor = pareto.group_points(del_dispositivo, ("device_id", "runtime"))
            puntos_motor = {k[1]: v for k, v in por_motor.items()}
            fronteras_motor = {
                motor: _fronteras(lista, objetivos, suelos) for motor, lista in puntos_motor.items()
            }
            _contrastar_motores(vista, dispositivo, fronteras_motor, analisis)
            calculado.setdefault(dispositivo, {})[vista] = {
                "objectives": objetivos,
                "floors": suelos,
                "points": del_dispositivo,
                "device": (estricta, robusta),
                "points_by_runtime": puntos_motor,
                "frontiers_by_runtime": fronteras_motor,
            }

            if receta["kind"] == "memoria":
                figura, resumen = figures_pareto.figure_frontier_with_memory(
                    del_dispositivo, estricta, robusta, objetivos, suelos, dispositivo
                )
            else:
                figura, resumen = figures_pareto.figure_frontier_by_device(
                    puntos_motor,
                    fronteras_motor,
                    del_dispositivo,
                    (estricta, robusta),
                    objetivos,
                    suelos,
                    dispositivo,
                )

            descripcion = f"frontera_{receta['slug']}_{dispositivo}"
            figuras.append((descripcion, figura))
            manifiesto["views"][vista][dispositivo] = {
                "figure": descripcion,
                "panels": resumen,
                "num_points": len(del_dispositivo),
                "frontier_strict": sorted(p.label for p in estricta.nondominated),
                "frontier_robust": sorted(p.label for p in robusta.nondominated),
                "dominations_lost_to_floor": len(
                    pareto.dominations_lost_to_floor(estricta, robusta)
                ),
            }

    manifiesto["checked_against_step_26"] = analisis is not None
    if calculado:
        experimentos, _ = ExperimentRegistry(registry_dir).valid_rows("experiments")
        indice_experimentos = {fila["experiment_id"]: fila for fila in experimentos}
        manifiesto["engine_figures"] = {}
        for dispositivo, por_vista in sorted(calculado.items()):
            for descripcion, figura, resumen in _figuras_por_motor(
                dispositivo, por_vista, indice_experimentos
            ):
                figuras.append((descripcion, figura))
                manifiesto["engine_figures"][descripcion] = resumen
        # La figura del resultado principal (capítulo 1) sale de las mismas fronteras por motor.
        manifiesto["intro_figures"] = {}
        for dispositivo, por_vista in sorted(calculado.items()):
            plano = por_vista.get("plano_map5095_p50")
            if plano is None:
                continue
            todos = list({p.label: p for p in plano["points"]}.values())
            identidades = figures_pareto.point_identities(todos, indice_experimentos)
            figura, resumen = figures_pareto.figure_headline_frontier(
                {m: plano["points_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES},
                {m: plano["frontiers_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES},
                plano["objectives"],
                plano["floors"],
                identidades,
                dispositivo,
            )
            descripcion = f"{figures_pareto.HEADLINE_DESCRIPTION}_{dispositivo}"
            figuras.append((descripcion, figura))
            manifiesto["intro_figures"][descripcion] = {
                "figure": descripcion,
                "views": ["plano_map5095_p50"],
                "summary": resumen,
            }
    return figuras, manifiesto


def _figuras_por_motor(
    dispositivo: str,
    por_vista: dict[str, dict[str, Any]],
    experimentos: dict[str, dict[str, str]],
) -> list[tuple[str, Any, dict[str, Any]]]:
    """Figuras por motor de un dispositivo.

    Cada figura sale solo si están todas las vistas que necesita.
    """
    todos = [p for datos in por_vista.values() for p in datos["points"]]
    identidades = figures_pareto.point_identities(
        list({p.label: p for p in todos}.values()), experimentos
    )
    salida: list[tuple[str, Any, dict[str, Any]]] = []

    def anadir(descripcion: str, figura: Any, resumen: dict[str, Any], vistas: list[str]) -> None:
        nombre = f"{descripcion}_{dispositivo}"
        salida.append((nombre, figura, {"figure": nombre, "views": vistas, "summary": resumen}))

    plano = por_vista.get("plano_map5095_p50")
    if plano is not None:
        argumentos = (
            {m: plano["points_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES},
            {m: plano["frontiers_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES},
            plano["objectives"],
            plano["floors"],
            identidades,
            dispositivo,
        )
        anadir(
            "frontera_map5095_p50_por_motor",
            *figures_pareto.figure_engine_frontier(*argumentos),
            ["plano_map5095_p50"],
        )
        anadir(
            "n_frente_a_compresiones_s",
            *figures_pareto.figure_nano_against_small(*argumentos),
            ["plano_map5095_p50"],
        )
    espec = por_vista.get("spec_3d_map5095_p95_memoria")
    if espec is not None:
        anadir(
            "frontera_3obj_por_motor",
            *figures_pareto.figure_engine_three_objectives(
                {m: espec["points_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES},
                {m: espec["frontiers_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES},
                espec["objectives"],
                espec["floors"],
                identidades,
                dispositivo,
            ),
            ["spec_3d_map5095_p95_memoria"],
        )
    if set(por_vista) == set(FIGURES):
        recuentos = []
        for vista, datos in por_vista.items():
            grupos = {"dispositivo": datos["device"]}
            grupos.update(
                {m: datos["frontiers_by_runtime"][m] for m in figures_pareto.ENGINE_RUNTIMES}
            )
            for grupo, (estricta, robusta) in grupos.items():
                recuentos.append(
                    {
                        "view": vista,
                        "group": grupo,
                        "strict": len(estricta.dominations),
                        "robust": len(robusta.dominations),
                        "lost": len(pareto.dominations_lost_to_floor(estricta, robusta)),
                    }
                )
        anadir(
            "dominaciones_perdidas_por_suelo",
            *figures_pareto.figure_lost_dominations(recuentos),
            sorted(por_vista),
        )
        anadir(
            "miembros_por_tecnica",
            *figures_pareto.figure_membership(
                {
                    vista: {
                        m: (
                            datos["points_by_runtime"][m],
                            *datos["frontiers_by_runtime"][m],
                        )
                        for m in figures_pareto.ENGINE_RUNTIMES
                    }
                    for vista, datos in por_vista.items()
                },
                identidades,
            ),
            sorted(por_vista),
        )
    return salida


def resumir(manifiesto: dict[str, Any]) -> list[str]:
    """Construye el resumen que se imprime al usuario."""
    lineas: list[str] = []
    if not manifiesto["checked_against_step_26"]:
        lineas.append("AVISO: no se ha contrastado con el volcado del paso 26 porque no estaba.")
    for vista, dispositivos in manifiesto["views"].items():
        for dispositivo, datos in dispositivos.items():
            lineas.append(f"{datos['figure']}  ({datos['num_points']} puntos)")
            perdidas = datos["dominations_lost_to_floor"]
            lineas.append(
                "    frontera %d (estricta) / %d (robusta); %d %s no %s al suelo"
                % (
                    len(datos["frontier_strict"]),
                    len(datos["frontier_robust"]),
                    perdidas,
                    "dominación" if perdidas == 1 else "dominaciones",
                    "sobrevive" if perdidas == 1 else "sobreviven",
                )
            )
            for panel, resumen in sorted(datos["panels"].items()):
                marcados = resumen["cross_session"]
                halo = resumen["robust_only"]
                lineas.append(
                    "    panel %-22s sesión de referencia %s, %d punto(s) de "
                    "otra sesión, %d con halo"
                    % (panel, resumen["reference_environment"], len(marcados), len(halo))
                )
    for nombre, datos in manifiesto.get("engine_figures", {}).items():
        lineas.append(f"{nombre}  (por motor; vistas: {', '.join(datos['views'])})")
    return lineas


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("27_build_pareto_figures", level=args.log_level)

    try:
        rutas = get_project_paths()
        registry_dir = args.registry_dir or rutas.results_registry
        output_dir = args.output_dir or rutas.results_figures
        analysis_dir = args.analysis_dir or (rutas.results_raw / "pareto")
        vistas = sorted(set(args.view)) if args.view else sorted(FIGURES)

        figuras, manifiesto = generar(registry_dir, rutas.root, analysis_dir, vistas)

        existentes = [
            descripcion
            for descripcion, _ in figuras
            if (
                output_dir
                / f"{style.figure_filename(*figures_pareto.figure_placement(descripcion), descripcion)}.pdf"
            ).exists()
        ]
        if existentes and not args.dry_run:
            if args.on_existing == "skip":
                logger.info("Ya existen %d figura(s) y --on-existing=skip.", len(existentes))
                print(f"Sin cambios: ya existen {len(existentes)} figura(s).")
                return 0
            if args.on_existing == "fail":
                logger.error(
                    "Ya existen estas figuras: %s. Con --on-existing overwrite, 'git status' "
                    "indica después si han cambiado.",
                    ", ".join(existentes),
                )
                return 1

    except TFMEdgeError as exc:
        logger.error("%s", exc)
        return 2
    except (OSError, ValueError) as exc:
        logger.error("Error de configuración o de entrada/salida: %s", exc)
        return 1

    for linea in resumir(manifiesto):
        print(linea)

    if args.dry_run:
        logger.info("--dry-run: no se ha tocado results/figures.")
        print(
            "\n--dry-run: no se ha escrito ninguna figura ni el manifiesto. "
            "El log sí está en results/logs/."
        )
        return 0

    escritas = figures_pareto.save_pareto_figures(figuras, output_dir)
    destino = analysis_dir / MANIFEST_NAME
    ensure_parent_dir(destino)
    temporal = destino.with_suffix(destino.suffix + ".tmp")
    temporal.write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporal, destino)

    print(f"\n{len(escritas)} figura(s) en {output_dir} (PDF y PNG).")
    print(f"Manifiesto: {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
