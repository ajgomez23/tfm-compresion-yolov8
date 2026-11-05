#!/usr/bin/env python
"""Reparte la pérdida entre emparejamiento perdido y orden roto, sin volver a inferir.

La pérdida tiene dos causas posibles: el modelo deja de encontrar el objeto (emparejamiento) o lo
sigue encontrando con otra confianza (orden). Se reconstruyen las detecciones ya guardadas con
``evaluation/reconstruction.py``, igual que el paso 23. Los tres sumandos tienen que sumar el cambio
de aciertos en cada umbral, y los recuentos se contrastan con la curva ya volcada de la misma
evaluación. Los pares salen del registro por ``parent_experiment_id``, como en el paso 24. No toca
el registro ni mide latencia.

Códigos de salida: 0 si el reparto se mide y se vuelca, 1 si hay un error de configuración, 2 si
alguna reconstrucción no se ancla o alguna identidad no se cumple.
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

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.evaluation import (  # noqa: E402
    coco_eval,
    matching_analysis,
    operating_analysis,
    reconstruction,
)
from tfm_edge.evaluation.runner import load_split_file_names  # noqa: E402

logger = get_logger(__name__)


def nombre_volcado(confidence: float) -> str:
    """Nombre del volcado para una confianza de referencia, con la confianza en centésimas."""
    return f"matching_ranking_conf{int(round(confidence * 100)):03d}.json"


def _relativa(ruta: Path, raiz: Path) -> str:
    """Ruta relativa a la raíz del proyecto, o la absoluta si no cuelga de ella."""
    try:
        return str(ruta.relative_to(raiz))
    except ValueError:
        return str(ruta)


def resolver_procedencia(
    curva: dict[str, Any], raiz: Path, eval_dir: Path
) -> tuple[Path, Path, str]:
    """Localiza las detecciones y el volcado registrado de una curva y devuelve ``(detecciones,
    volcado_registrado, estrategia)``.

    La procedencia se lee del volcado de la curva y no del nombre, porque algunos directorios de
    validador no llevan el ``experiment_id``. Los volcados antiguos guardan rutas absolutas; si no
    existen, se busca el mismo nombre bajo el directorio de evaluaciones y se registra qué
    estrategia funcionó.
    """
    fuente = curva.get("source", {})
    estrategias: list[str] = []

    def resolver(clave: str, alternativa: Path) -> Path:
        """Resuelve una ruta grabada, con el directorio de evaluaciones de reserva."""
        grabada = Path(str(fuente.get(clave, "")))
        candidata = grabada if grabada.is_absolute() else raiz / grabada
        if str(grabada) and candidata.is_file():
            estrategias.append(f"{clave}=grabada")
            return candidata
        if alternativa.is_file():
            estrategias.append(f"{clave}=por-nombre")
            return alternativa
        raise TFMEdgeError(
            f"No se encuentran las {clave} de "
            f"{curva['experiment_id']}/{curva['runtime']}/{curva['precision']}: "
            f"ni en '{candidata}' ni en '{alternativa}'.",
            "Si la ruta grabada se ha movido, deja el fichero bajo el directorio de "
            "evaluaciones con su nombre original.",
        )

    detecciones = resolver(
        "predictions",
        eval_dir / Path(str(fuente.get("predictions", "x/x"))).parent.name / "predictions.json",
    )
    registrado = resolver(
        "recorded_dump", eval_dir / Path(str(fuente.get("recorded_dump", "x"))).name
    )
    return detecciones, registrado, ", ".join(estrategias)


def leer_evaluacion(
    curva: dict[str, Any],
    raiz: Path,
    eval_dir: Path,
    referencia: dict[str, Any],
    images_dir: Path,
    resolucion: int,
    num_clases: int,
) -> tuple[matching_analysis.EvaluationFacts, dict[str, Any]]:
    """Reconstruye una evaluación y devuelve la cobertura y el estado de cada detección, junto con
    la procedencia. Falla si la reconstrucción no se ancla o si algún recuento no coincide con la
    curva.
    """
    detecciones, registrado, estrategia = resolver_procedencia(curva, raiz, eval_dir)
    reconstruida = reconstruction.rebuild_evaluation(
        json.loads(detecciones.read_text(encoding="utf-8")),
        json.loads(registrado.read_text(encoding="utf-8")),
        referencia,
        images_dir,
        resolucion,
        num_clases,
    )
    hechos = matching_analysis.evaluation_facts(reconstruida.evaluator)
    contraste = matching_analysis.check_against_curve(hechos, curva)
    procedencia = {
        "predictions": _relativa(detecciones, raiz),
        "recorded_dump": _relativa(registrado, raiz),
        "path_resolution": estrategia,
        "anchor_max_difference": reconstruida.anchor["max_difference"],
        "num_detections": reconstruida.num_detections,
        "matched_iou_min": hechos.matched_iou_min,
        "curve_cross_check": contraste,
    }
    logger.info(
        "%s/%s/%s: %d anotaciones, %d cubiertas, %d detecciones evaluadas "
        "(ancla %.1e, IoU mínima emparejada %.4f)",
        curva["experiment_id"],
        curva["runtime"],
        curva["precision"],
        hechos.num_gt,
        len(hechos.covered),
        int(hechos.scores.size),
        reconstruida.anchor["max_difference"],
        hechos.matched_iou_min,
    )
    return hechos, procedencia


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="25_matching_ranking.py",
        description=(
            "Reparte la pérdida de cada par entre emparejamiento perdido y orden roto a partir "
            "de las detecciones guardadas. Los pares salen del registro."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # El reparto completo, en la rejilla por defecto\n"
            "  python scripts/25_matching_ranking.py\n\n"
            "  # Con otra confianza de referencia para las tablas\n"
            "  python scripts/25_matching_ranking.py --confidence 0.50\n"
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=operating_analysis.REFERENCE_CONFIDENCE,
        metavar="C",
        help=(
            "Confianza de referencia de las tablas y del nombre del volcado "
            "(por defecto: %(default)s). Tiene que estar en la rejilla."
        ),
    )
    parser.add_argument(
        "--confidences",
        type=float,
        nargs="+",
        default=list(matching_analysis.DECOMPOSITION_CONFIDENCES),
        metavar="C",
        help="Rejilla de umbrales en la que se reparte (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=None,
        help="Directorio de las curvas (por defecto: results/raw/operating_curve).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Directorio de las evaluaciones (por defecto: results/raw/artifact_eval).",
    )
    parser.add_argument("--registry-dir", type=Path, default=None, help="Directorio del registro.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dónde volcar el reparto (por defecto: results/raw/matching_ranking).",
    )
    parser.add_argument(
        "--on-existing",
        type=str,
        default="fail",
        choices=["fail", "skip", "overwrite"],
        help=("Qué hacer si el reparto ya está volcado (por defecto: %(default)s)."),
    )
    parser.add_argument("--coco", type=Path, default=None, help="COCO de referencia.")
    parser.add_argument("--data", type=Path, default=None, help="YAML de dataset.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--resolution", type=int, default=640, metavar="N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def _formatear(valor: Any, patron: str) -> str:
    """Formatea un valor que puede faltar; un valor ausente se escribe como raya, no como cero."""
    return "—" if valor is None else format(valor, patron)


def _etiqueta(par: dict[str, Any]) -> str:
    """Etiqueta y técnica de un par, recortadas al ancho de la columna.

    La técnica hace falta porque PTQ y QAT parten del mismo baseline y una variante de poda aparece
    también como combinación.
    """
    return f"{par['label']} [{par['via']}]"[:44]


def imprimir_cobertura(entradas: list[dict[str, Any]], confianza: float) -> None:
    """Imprime el reparto del cambio de aciertos."""
    print(f"\nREPARTO DEL CAMBIO DE ACIERTOS EN conf={confianza:.2f}")
    print("-" * 110)
    print(
        f"{'par [técnica]':<44} {'motor':<11} {'cubre-A':>8} {'cubre-D':>8} "
        f"{'ΔTP':>7} {'empar.':>8} {'orden':>7}"
    )
    print("-" * 110)
    for entrada in entradas:
        cobertura = entrada["analysis"]["coverage"]
        reparto = next(
            r
            for r in entrada["analysis"]["recall_decomposition"]
            if abs(r["confidence"] - confianza) < 1e-9
        )
        emparejamiento = reparto["from_matching_lost"] + reparto["from_matching_gained"]
        print(
            f"{_etiqueta(entrada['pair']):<44} {entrada['pair']['runtime']:<11} "
            f"{cobertura['covered_before']:>8,} {cobertura['covered_after']:>8,} "
            f"{reparto['delta_tp']:>+7,} {emparejamiento:>+8,} "
            f"{reparto['from_score_shift']:>+7,}"
        )


def imprimir_falsos_positivos(entradas: list[dict[str, Any]], confianza: float) -> None:
    """Imprime el desglose del aumento de falsos positivos."""
    print(f"\nDE QUÉ TIPO SON LOS FALSOS POSITIVOS QUE APARECEN (conf={confianza:.2f})")
    print("-" * 110)
    print(
        f"{'par [técnica]':<44} {'motor':<11} {'ΔFP':>9} {'Δdupl.':>9} "
        f"{'Δlocal.':>9} {'Δespur.':>9} {'%espur.':>8}"
    )
    print("-" * 110)
    for entrada in entradas:
        bloque = next(
            b
            for b in entrada["analysis"]["false_positives"]
            if abs(b["confidence"] - confianza) < 1e-9
        )
        delta = bloque["delta"]
        cuota = (delta["spurious"] / delta["total"]) if delta["total"] else None
        print(
            f"{_etiqueta(entrada['pair']):<44} {entrada['pair']['runtime']:<11} "
            f"{delta['total']:>+9,} {delta['duplicate']:>+9,} "
            f"{delta['poor_localization']:>+9,} {delta['spurious']:>+9,} "
            f"{_formatear(cuota, '>8.1%')}"
        )


def imprimir_orden(entradas: list[dict[str, Any]]) -> None:
    """Imprime la separación del orden, que no depende de ningún umbral."""
    print("\nSEPARACIÓN DEL ORDEN: P(acierto puntúa por encima de falso positivo)")
    print("-" * 110)
    print(
        f"{'par [técnica]':<44} {'motor':<11} {'antes':>9} {'después':>9} {'Δ':>9} "
        f"{'conf.med.A':>11} {'conf.med.D':>11}"
    )
    print("-" * 110)
    for entrada in entradas:
        orden = entrada["analysis"]["ranking"]
        desplazamiento = entrada["analysis"]["covered_score_shift"]
        antes, despues = orden["before"]["macro"], orden["after"]["macro"]
        diferencia = None if (antes is None or despues is None) else despues - antes
        print(
            f"{_etiqueta(entrada['pair']):<44} {entrada['pair']['runtime']:<11} "
            f"{_formatear(antes, '>9.4f')} {_formatear(despues, '>9.4f')} "
            f"{_formatear(diferencia, '>+9.4f')} "
            f"{_formatear(desplazamiento['median_before'], '>11.4f')} "
            f"{_formatear(desplazamiento['median_after'], '>11.4f')}"
        )


def resumir(entradas: list[dict[str, Any]], confianza: float) -> dict[str, Any]:
    """Resume el reparto por clase de par."""
    resumen: dict[str, Any] = {}
    for clase in sorted({e["pair"]["kind"] for e in entradas}):
        de_la_clase = [e for e in entradas if e["pair"]["kind"] == clase]
        repartos = [
            next(
                r
                for r in e["analysis"]["recall_decomposition"]
                if abs(r["confidence"] - confianza) < 1e-9
            )
            for e in de_la_clase
        ]
        falsos = [
            next(
                b
                for b in e["analysis"]["false_positives"]
                if abs(b["confidence"] - confianza) < 1e-9
            )["delta"]
            for e in de_la_clase
        ]
        ordenes = [
            (
                e["analysis"]["ranking"]["before"]["macro"],
                e["analysis"]["ranking"]["after"]["macro"],
            )
            for e in de_la_clase
        ]
        medibles = [(a, d) for a, d in ordenes if a is not None and d is not None]
        resumen[clase] = {
            "num_pairs": len(de_la_clase),
            "coverage_lost_total": sum(
                e["analysis"]["coverage"]["matching_lost"] for e in de_la_clase
            ),
            "coverage_gained_total": sum(
                e["analysis"]["coverage"]["matching_gained"] for e in de_la_clase
            ),
            "pairs_dominated_by_matching": sum(
                1
                for r in repartos
                if abs(r["from_matching_lost"] + r["from_matching_gained"])
                > abs(r["from_score_shift"])
            ),
            "pairs_dominated_by_order": sum(
                1
                for r in repartos
                if abs(r["from_score_shift"])
                > abs(r["from_matching_lost"] + r["from_matching_gained"])
            ),
            "fp_delta_total": sum(f["total"] for f in falsos),
            "fp_delta_by_kind": {
                tipo: sum(f[tipo] for f in falsos) for tipo in matching_analysis.FP_KINDS
            },
            "pairs_with_order_degraded": sum(1 for a, d in medibles if d < a),
            "pairs_with_order_measured": len(medibles),
        }
    return resumen


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("25_matching_ranking", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    curve_dir = args.curve_dir or (rutas.results_raw / "operating_curve")
    eval_dir = args.eval_dir or (rutas.results_raw / "artifact_eval")
    salida = args.output_dir or (rutas.results_raw / "matching_ranking")
    coco = args.coco or (rutas.data_processed / "sh17_coco.json")
    umbrales = sorted(float(c) for c in args.confidences)

    print("=" * 110)
    print("PASO 25: EMPAREJAMIENTO PERDIDO FRENTE A ORDEN ROTO")
    print("=" * 110)

    if not any(abs(c - args.confidence) < 1e-9 for c in umbrales):
        print(
            f"ERROR: la confianza de referencia {args.confidence} no está en la rejilla "
            f"{umbrales}.",
            file=sys.stderr,
        )
        return 1
    if not coco.is_file():
        print(f"ERROR: no existe el COCO de referencia '{coco}'.", file=sys.stderr)
        return 1

    destino = salida / nombre_volcado(args.confidence)
    if destino.exists():
        if args.on_existing == "skip":
            print(f"\n'{destino}' ya existe; no se rehace.")
            return 0
        if args.on_existing == "fail":
            print(
                f"ERROR: '{destino}' ya existe. Usa --on-existing skip para continuar u "
                "overwrite para rehacerlo.",
                file=sys.stderr,
            )
            return 1

    try:
        data_yaml = args.data or resize.find_dataset_for_resolution(rutas.data, args.resolution)
        curvas = operating_analysis.load_curve_dumps(curve_dir)
        registro = ExperimentRegistry(registry_dir=args.registry_dir)
        filas = registro.read_rows("experiments")
        pares = operating_analysis.derive_pairs(filas, curvas)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not pares:
        print("ERROR: el registro no declara ningún par medible.", file=sys.stderr)
        return 1

    clases = dict(
        enumerate(load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json"))
    )
    referencia_completa = json.loads(coco.read_text(encoding="utf-8"))
    nombres = load_split_file_names(rutas.data_splits, args.split, referencia_completa)
    referencia = coco_eval.subset_ground_truth(referencia_completa, nombres)
    images_dir = Path(data_yaml).parent / "images"

    print(f"\nPares del registro: {len(pares)}")
    print(f"Umbrales:           {', '.join(f'{c:.2f}' for c in umbrales)}")
    print(f"Referencia:         conf={args.confidence:.2f}")
    print(f"Salida:             {destino}\n")

    # Cada evaluación se lee una sola vez.
    necesarias = sorted(
        {(p["before"], p["runtime"], p["before_precision"]) for p in pares}
        | {(p["after"], p["runtime"], p["after_precision"]) for p in pares}
    )
    hechos: dict[tuple[str, str, str], matching_analysis.EvaluationFacts] = {}
    procedencias: dict[str, dict[str, Any]] = {}
    for clave in necesarias:
        try:
            hechos[clave], procedencias["/".join(clave)] = leer_evaluacion(
                curvas[clave],
                rutas.root,
                eval_dir,
                referencia,
                images_dir,
                args.resolution,
                len(clases),
            )
        except TFMEdgeError as error:
            print(f"ERROR al leer {clave}: {error}", file=sys.stderr)
            return 2

    # El reparto.
    entradas: list[dict[str, Any]] = []
    for par in pares:
        antes = hechos[(par["before"], par["runtime"], par["before_precision"])]
        despues = hechos[(par["after"], par["runtime"], par["after_precision"])]
        try:
            analisis = matching_analysis.analyze_pair(antes, despues, umbrales, clases)
        except TFMEdgeError as error:
            print(f"ERROR al repartir {par['label']}/{par['runtime']}: {error}", file=sys.stderr)
            return 2
        entradas.append({"pair": par, "analysis": analisis})

    imprimir_cobertura(entradas, args.confidence)
    imprimir_falsos_positivos(entradas, args.confidence)
    imprimir_orden(entradas)

    resumen = resumir(entradas, args.confidence)
    print("\nRESUMEN")
    print("-" * 110)
    for clase, datos in resumen.items():
        print(
            f"{clase}: {datos['num_pairs']} pares; "
            f"cobertura perdida {datos['coverage_lost_total']:,} / "
            f"ganada {datos['coverage_gained_total']:,}; "
            f"ΔTP dominado por emparejamiento en {datos['pairs_dominated_by_matching']}, "
            f"por orden en {datos['pairs_dominated_by_order']}"
        )
        print(
            f"    ΔFP {datos['fp_delta_total']:+,} = "
            f"duplicados {datos['fp_delta_by_kind']['duplicate']:+,}, "
            f"localización {datos['fp_delta_by_kind']['poor_localization']:+,}, "
            f"espurios {datos['fp_delta_by_kind']['spurious']:+,}; "
            f"orden peor en {datos['pairs_with_order_degraded']} de "
            f"{datos['pairs_with_order_measured']}"
        )

    volcado = {
        "reference_confidence": args.confidence,
        "confidences": umbrales,
        "split": args.split,
        "input_resolution": args.resolution,
        "size_criterion": reconstruction.SIZE_CRITERION,
        "thresholds": {
            "matching_iou": matching_analysis.MATCHING_IOU,
            "duplicate_iou": matching_analysis.DUPLICATE_IOU,
            "spurious_iou": matching_analysis.SPURIOUS_IOU,
        },
        "source": {
            "curve_dir": _relativa(curve_dir, rutas.root),
            "registry": _relativa(registro.path_for("experiments"), rutas.root),
            "evaluations": procedencias,
            "note": (
                "Las detecciones NO se han vuelto a generar: son las de las "
                "evaluaciones ya registradas, releídas para repartir la pérdida."
            ),
        },
        "pairs": entradas,
        "summary": resumen,
    }

    salida.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(volcado, indent=2, ensure_ascii=False), encoding="utf-8")
    temporal.replace(destino)

    print("-" * 110)
    print(f"Reparto volcado en {destino}")
    print("-" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
