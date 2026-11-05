#!/usr/bin/env python
"""Pasa al registro el coste de exportar y cuantizar, que ya estaba cronometrado.

exporter guarda export_seconds y quantize guarda quantize_seconds en los volcados de
results/exports/; este script rellena export_time_s en deployment_metrics sin medir nada. Cada fila
se ata a su volcado por la ruta del artefacto y se sigue la cadena de procedencia (un IR INT8 de
OpenVINO incluye el ONNX del que sale). Las filas de PyTorch quedan vacías porque no se exportan, y
engine_build_time_s también, porque no se compiló ningún motor. Por defecto solo muestra el plan
(``--apply`` para escribir), no sobrescribe valores distintos, no rellena cadenas incompletas y es
idempotente. El desglose por pasos se escribe en results/raw/exports/.

Códigos de salida: 0 si no hay nada que hacer, se muestra el plan o se aplica y verifica; 1 si hay
un error de uso o de entorno; 2 si hay filas sin coste reconstruible o falla la verificación.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_parent_dir, get_project_paths  # noqa: E402
from tfm_edge.common.registry import TABLE_PRIMARY_KEYS, ExperimentRegistry  # noqa: E402
from tfm_edge.export import cost as cost_mod  # noqa: E402

logger = get_logger(__name__)

TABLE = "deployment_metrics"

#: engine_build_time_s queda vacía: no se compiló ningún motor.
COLUMN = "export_time_s"

#: Runtime sin artefacto exportado; su celda queda vacía.
NATIVE_RUNTIME = "pytorch"

DUMP_NAME = "export_cost.json"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="31_register_export_cost.py",
        description=(
            "Lleva a deployment_metrics.export_time_s el coste de exportar y "
            "cuantizar que ya está cronometrado en los volcados de "
            "results/exports/. No mide nada nuevo."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué se rellenaría, sin escribir nada\n"
            "  python scripts/31_register_export_cost.py\n"
            "\n"
            "  # Aplicarlo\n"
            "  python scripts/31_register_export_cost.py --apply\n"
            "\n"
            "  # Sobre copias desechables, para diagnosticar sin tocar el registro\n"
            "  python scripts/31_register_export_cost.py --apply \\\n"
            "      --registry-dir /tmp/reg --output-dir /tmp/raw\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de los volcados de exportación (por defecto: results/exports).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio del volcado del desglose (por defecto: results/raw/exports).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Escribe en el registro. Sin esto solo se muestra el plan.",
    )
    parser.add_argument(
        "--on-conflict",
        choices=("fail", "skip", "overwrite"),
        default="fail",
        help=(
            "Qué hacer con una celda que ya tiene otro valor (por defecto: "
            "%(default)s). 'overwrite' descarta una medida registrada."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def etiqueta_de_clave(clave: dict[str, str]) -> str:
    """Nombre legible de una fila de despliegue, aunque falten campos de la clave.

    Si falta alguno, indica qué campos están y cuáles no en lugar de dejar barras vacías.
    """
    vacias = [campo for campo, valor in clave.items() if not str(valor).strip()]
    if not vacias:
        return "/".join(str(valor) for valor in clave.values())
    puestos = [f"{campo}={valor}" for campo, valor in clave.items() if str(valor).strip()]
    return f"<fila incompleta: {', '.join(puestos) or 'ningún campo'}; " f"sin {', '.join(vacias)}>"


def planificar(
    rows: list[dict[str, str]],
    steps: dict[str, cost_mod.ExportStep],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Coste de cada fila de despliegue: las que se pueden rellenar y las que no, con su motivo."""
    rellenables: list[dict[str, Any]] = []
    sin_coste: list[dict[str, Any]] = []

    for fila in rows:
        clave = {c: fila.get(c, "") for c in TABLE_PRIMARY_KEYS[TABLE]}
        artefacto = (fila.get("artifact_path") or "").strip()
        etiqueta = etiqueta_de_clave(clave)

        if not artefacto:
            sin_coste.append(
                {
                    "key": clave,
                    "label": etiqueta,
                    "reason": "La fila no dice qué artefacto se midió (artifact_path vacío).",
                    "steps": [],
                }
            )
            continue

        coste = cost_mod.artifact_cost(artefacto, steps)
        if coste.seconds is None:
            motivo = coste.reason
            if fila.get("runtime") == NATIVE_RUNTIME:
                motivo = (
                    "Es el modelo de PyTorch, que no se exporta: su coste de "
                    "exportación no es cero, no existe, y la celda se queda vacía."
                )
            sin_coste.append(
                {
                    "key": clave,
                    "label": etiqueta,
                    "reason": motivo,
                    "steps": [paso.to_dict() for paso in coste.steps],
                }
            )
            continue

        rellenables.append(
            {
                "key": clave,
                "label": etiqueta,
                "artifact": coste.artifact,
                "seconds": coste.seconds,
                "steps": [paso.to_dict() for paso in coste.steps],
            }
        )

    return rellenables, sin_coste


def verificar(
    registry: ExperimentRegistry,
    rellenables: list[dict[str, Any]],
) -> list[str]:
    """Relee el registro y compara celda a celda lo escrito; devuelve las discrepancias."""
    filas = registry.read_rows(TABLE)
    indice = {tuple(fila.get(c, "") for c in TABLE_PRIMARY_KEYS[TABLE]): fila for fila in filas}
    problemas: list[str] = []
    for entrada in rellenables:
        clave = tuple(entrada["key"][c] for c in TABLE_PRIMARY_KEYS[TABLE])
        fila = indice.get(clave)
        if fila is None:
            problemas.append(f"{entrada['label']}: la fila ha desaparecido del registro.")
            continue
        escrito = (fila.get(COLUMN) or "").strip()
        try:
            if abs(float(escrito) - entrada["seconds"]) > 1e-9:
                problemas.append(
                    f"{entrada['label']}: el registro dice {escrito}, se planificó "
                    f"{entrada['seconds']}."
                )
        except ValueError:
            problemas.append(
                f"{entrada['label']}: el registro dice {escrito!r}, que no es un número."
            )
    return problemas


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("31_register_export_cost", level=args.log_level)

    try:
        paths = get_project_paths()
        registry_dir = args.registry_dir or paths.results_registry
        exports_dir = args.exports_dir or (paths.results / "exports")
        output_dir = args.output_dir or (paths.results_raw / "exports")

        if not exports_dir.is_dir():
            logger.error("No existe el directorio de exportaciones: %s", exports_dir)
            return 1

        registry = ExperimentRegistry(registry_dir=registry_dir)
        filas = registry.read_rows(TABLE)
        if not filas:
            logger.error("La tabla '%s' de %s está vacía.", TABLE, registry_dir)
            return 1

        pasos = cost_mod.index_export_steps(exports_dir, paths=paths)
        rellenables, sin_coste = planificar(filas, pasos)
    except TFMEdgeError as error:
        logger.error("%s", error)
        return 1

    print()
    print(f"Filas de despliegue: {len(filas)}")
    print(f"Pasos de exportación medidos: {len(pasos)}")
    print(f"Con coste reconstruible: {len(rellenables)}")
    print(f"Sin coste reconstruible: {len(sin_coste)}")
    print()
    for entrada in rellenables:
        cadena = " <- ".join(f"{paso['target']}:{paso['seconds']}s" for paso in entrada["steps"])
        print(f"  {entrada['label'][:62]:62s} {entrada['seconds']:>8.2f} s   {cadena}")
    if sin_coste:
        print()
        print("Sin coste, con su motivo:")
        for entrada in sin_coste:
            print(f"  {entrada['label']}")
            print(f"      {entrada['reason']}")

    if not args.apply:
        print()
        print("Plan mostrado. Nada escrito: vuelve a lanzarlo con --apply.")
        return 0

    try:
        informe = registry.update_cells(
            TABLE,
            [{**entrada["key"], COLUMN: entrada["seconds"]} for entrada in rellenables],
            on_conflict=args.on_conflict,
        )
    except TFMEdgeError as error:
        logger.error("%s", error)
        return 2

    problemas = verificar(registry, rellenables)

    volcado = Path(output_dir) / DUMP_NAME
    ensure_parent_dir(volcado)
    volcado.write_text(
        json.dumps(
            {
                "table": TABLE,
                "column": COLUMN,
                "rows_total": len(filas),
                "filled": rellenables,
                "without_cost": sin_coste,
                "verification_problems": problemas,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Celdas escritas: {len(informe['written'])}")
    print(f"Celdas que ya estaban igual: {len(informe['unchanged'])}")
    print(f"Desglose por pasos: {volcado}")

    if problemas:
        print()
        print("La verificación posterior ha encontrado discrepancias:")
        for problema in problemas:
            print(f"  {problema}")
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
