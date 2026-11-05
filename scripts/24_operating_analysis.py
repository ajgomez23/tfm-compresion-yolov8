#!/usr/bin/env python
"""Deriva tres análisis de las curvas a umbral fijo del paso 23.

Qué eje cae al aplicar cada técnica, cuántas detecciones cruzan el umbral y de qué tipo, y cuánto se
recupera reajustando el umbral. Solo lee results/raw/operating_curve/ y experiments.csv y escribe un
volcado. El volcado no lleva fecha, así que con las mismas curvas sale idéntico y ``--on-existing
overwrite`` seguido de ``git status`` indica si ha cambiado algo.

Códigos de salida: 0 si se escribe el volcado, 1 si hay un error de configuración o el volcado ya
existe sin indicar qué hacer, 2 si los datos no permiten derivar (curva sin anclar, par imposible).
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
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.evaluation import operating_analysis  # noqa: E402
from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="24_operating_analysis.py",
        description=(
            "Deriva de las curvas a umbral fijo qué eje cae con cada técnica, cuántas "
            "detecciones cruzan el umbral y cuánto se recupera reajustándolo."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # La derivación completa, al umbral de referencia\n"
            "  python scripts/24_operating_analysis.py\n\n"
            "  # Comprobar que nada ha cambiado tras añadir artefactos\n"
            "  python scripts/24_operating_analysis.py --on-existing overwrite\n"
            "  git status --short results/raw/operating_analysis/\n\n"
            "  # A otro umbral, sin tocar el volcado de referencia\n"
            "  python scripts/24_operating_analysis.py --confidence 0.50\n"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=REFERENCE_CONFIDENCE,
        metavar="C",
        help=(
            "Umbral común de comparación (por defecto: %(default)s). Debe estar en la rejilla "
            "del paso 23."
        ),
    )
    parser.add_argument(
        "--shift-confidences",
        type=float,
        nargs="+",
        default=list(operating_analysis.SHIFT_CONFIDENCES),
        metavar="C",
        help=(
            "Umbrales a los que se comprueba que el signo del desplazamiento se "
            "mantiene (por defecto: %(default)s)."
        ),
    )
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=None,
        help="Directorio de las curvas (por defecto: results/raw/operating_curve).",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro (por defecto: results/registry). Solo se lee.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dónde volcar la derivación (por defecto: results/raw/operating_analysis).",
    )
    parser.add_argument(
        "--on-existing",
        type=str,
        default="fail",
        choices=["fail", "skip", "overwrite"],
        help=(
            "Qué hacer si el volcado ya existe (por defecto: %(default)s). Como es "
            "determinista, 'overwrite' sirve para comprobar que nada ha cambiado."
        ),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def nombre_volcado(confidence: float) -> str:
    """Nombre del volcado para un umbral; cada umbral tiene su propio fichero."""
    return f"operating_analysis_conf{int(round(confidence * 100)):03d}.json"


def _relativa(ruta: Path, raiz: Path) -> str:
    """Ruta relativa a la raíz si cuelga de ella; si no, absoluta (por ejemplo, en pruebas con
    directorios temporales).
    """
    try:
        return str(ruta.relative_to(raiz))
    except ValueError:
        return str(ruta)


def _formatear(valor: Any, patron: str) -> str:
    """Formatea un valor que puede ser None; sin valor devuelve guiones."""
    return format(valor, patron) if valor is not None else "—".rjust(len(format(0, patron)))


def imprimir_tablas(analisis: dict[str, Any]) -> None:
    """Imprime las tres derivaciones como tablas."""
    conf = analisis["confidence"]
    cuant = [p for p in analisis["pairs"] if p["kind"] == operating_analysis.KIND_QUANTIZATION]
    poda = [p for p in analisis["pairs"] if p["kind"] == operating_analysis.KIND_PRUNING]

    print("=" * 100)
    print(f"1. QUÉ EJE CAE, a conf={conf:.2f} fijo y común")
    print("=" * 100)
    for titulo, filas in (("CUANTIZAR", cuant), ("PODAR", poda)):
        if not filas:
            continue
        print(f"\n--- {titulo} " + "-" * (94 - len(titulo)))
        print(f"{'configuracion':<30}{'via':<12}{'motor':<13}{'dP':>10}{'dR':>10}   cae mas")
        for p in filas:
            eje = p["axis"]
            print(
                f"{p['label']:<30}{p['via']:<12}{p['runtime']:<13}"
                f"{eje['delta_precision']:>+10.4f}{eje['delta_recall']:>+10.4f}   "
                f"{eje['dominant_axis']}"
            )

    print("\n" + "=" * 100)
    print(f"2. DETECCIONES QUE CRUZAN conf={conf:.2f}, y de qué están hechas (solo cuantizar)")
    print("=" * 100)
    print(
        f"{'configuracion':<30}{'via':<12}{'motor':<13}{'antes':>9}{'despues':>9}"
        f"{'cambio':>9}{'dTP':>9}{'dFP':>9}"
    )
    for p in cuant:
        s = p["shift"]
        cambio = _formatear(
            100 * s["relative_change"] if s["relative_change"] is not None else None, "+8.1f"
        )
        print(
            f"{p['label']:<30}{p['via']:<12}{p['runtime']:<13}"
            f"{s['detections_before']:>9,}{s['detections_after']:>9,}{cambio}%"
            f"{s['delta_true_positives']:>+9,}{s['delta_false_positives']:>+9,}"
        )

    print("\n--- El mismo signo a otros umbrales (cambio relativo del total) " + "-" * 36)
    print(
        f"{'configuracion':<30}{'via':<12}{'motor':<13}"
        + "".join(f"{c:>9.2f}" for c in analisis["shift_confidences"])
    )
    for p in cuant:
        celdas = "".join(
            _formatear(
                100 * s["relative_change"] if s["relative_change"] is not None else None, "+8.1f"
            )
            + "%"
            for s in p["shift_across_confidences"]
        )
        print(f"{p['label']:<30}{p['via']:<12}{p['runtime']:<13}{celdas}")

    print("\n" + "=" * 100)
    print("3. CUÁNTO SE RECUPERA REAJUSTANDO EL UMBRAL (F1 máximo sobre la rejilla medida)")
    print("=" * 100)
    print(
        f"{'configuracion':<30}{'via':<12}{'motor':<13}{'dF1 fijo':>10}{'dF1 reaj.':>11}"
        f"{'conf antes':>11}{'conf desp.':>11}{'recuperado':>12}"
    )
    for p in cuant:
        r = p["retune"]
        recuperado = _formatear(
            100 * r["recovered_fraction"] if r["recovered_fraction"] is not None else None, "10.0f"
        )
        print(
            f"{p['label']:<30}{p['via']:<12}{p['runtime']:<13}"
            f"{r['f1_gap_fixed']:>+10.4f}{r['f1_gap_retuned']:>+11.4f}"
            f"{r['best_confidence_before']:>11.2f}{r['best_confidence_after']:>11.2f}"
            f"{recuperado}%  {r['optimum_moves']}"
        )


def imprimir_resumen(analisis: dict[str, Any]) -> None:
    """Imprime los agregados de las derivaciones."""
    conf = analisis["confidence"]
    q = analisis["summary"][operating_analysis.KIND_QUANTIZATION]
    p = analisis["summary"][operating_analysis.KIND_PRUNING]

    print("\n" + "=" * 100)
    print("RESUMEN")
    print("=" * 100)
    if q.get("pairs"):
        print(
            f"Cuantizar, a conf={conf:.2f}: la precisión cae más que el recall en "
            f"{q['precision_dominated']} de {q['pairs']} comparaciones."
        )
        print(
            f"  Cruzan el umbral más detecciones en {q['detections_up']} de "
            f"{q['detections_compared']} (entre {100 * q['relative_change_min']:+.1f}% y "
            f"{100 * q['relative_change_max']:+.1f}%), con los aciertos entre "
            f"{q['delta_true_positives_min']:+,} y {q['delta_true_positives_max']:+,} "
            f"y los falsos positivos entre {q['delta_false_positives_min']:+,} y "
            f"{q['delta_false_positives_max']:+,}."
        )
        if q["recovered_median"] is not None:
            print(
                f"  Reajustando el umbral se recupera una mediana del "
                f"{100 * q['recovered_median']:.0f}% (de {100 * q['recovered_min']:.0f}% a "
                f"{100 * q['recovered_max']:.0f}%, sobre {q['recovered_compared']} pares). "
                f"El óptimo sube en {q['optimum_up']}, se queda en {q['optimum_equal']} "
                f"y baja en {q['optimum_down']}."
            )
    if p.get("pairs"):
        print(
            f"Podar, a conf={conf:.2f}: la precisión cae más que el recall en "
            f"{p['precision_dominated']} de {p['pairs']} configuraciones."
        )


def main() -> int:
    """Punto de entrada."""
    args = construir_parser().parse_args()
    setup_logging("24_operating_analysis", level=args.log_level)
    rutas = get_project_paths()

    curve_dir = args.curve_dir or (rutas.results_raw / "operating_curve")
    output_dir = args.output_dir or (rutas.results_raw / "operating_analysis")
    destino = output_dir / nombre_volcado(args.confidence)

    if destino.exists():
        if args.on_existing == "skip":
            print(f"'{destino}' ya existe; nada que hacer.")
            return 0
        if args.on_existing == "fail":
            print(
                f"ERROR: '{destino}' ya existe. Usa --on-existing overwrite y después 'git "
                "status' para ver si ha cambiado algo.",
                file=sys.stderr,
            )
            return 1

    try:
        curvas = operating_analysis.load_curve_dumps(curve_dir)
        registro = ExperimentRegistry(registry_dir=args.registry_dir)
        filas = registro.read_rows("experiments")
        pares = operating_analysis.derive_pairs(filas, curvas)
        if not pares:
            print(
                "ERROR: el registro y las curvas no producen ningún par "
                "comparable. Sin pares no hay nada que derivar.",
                file=sys.stderr,
            )
            return 2
        analisis = operating_analysis.analyze(
            curvas, pares, args.confidence, tuple(args.shift_confidences)
        )
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    analisis["source"] = {
        "curve_dir": _relativa(curve_dir, rutas.root),
        "registry": _relativa(registro.path_for("experiments"), rutas.root),
        "note": (
            "Derivación sobre curvas ya medidas y ancladas. No mide nada: si un "
            "número de aquí no cuadra, el que mide es scripts/23_operating_curve.py."
        ),
    }

    imprimir_tablas(analisis)
    imprimir_resumen(analisis)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(analisis, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(destino)
    print(f"\nVolcado en '{_relativa(destino, rutas.root)}'.")
    logger.info("Derivación volcada en '%s'", destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
