#!/usr/bin/env python
"""Rellena las seis columnas ``error_*`` de quality_metrics y vuelca los recuentos.

El reparto (``evaluation/error_decomposition``) necesita las detecciones por instancia, que llegan
por dos rutas: ``curva``, desde el volcado de curva del paso 23, con ancla contra el mAP registrado
y contraste de recuentos con la curva; y ``volcado``, desde el ``validator_dir`` del volcado de
evaluación, solo con el ancla. Cada fila indica en ``notes`` su ruta. Las filas sin detecciones por
instancia o del evaluador ``ultralytics`` quedan vacías con el motivo en ``notes``.

Un volcado solo identifica ``(experiment_id, runtime, precision)`` y la clave del registro tiene
seis campos, así que si una procedencia apunta a más de una fila se aborta. Los volcados de
``--smoke-test`` se descartan por su campo ``smoke_test``.

Por defecto solo muestra el plan; ``--apply`` escribe. Exige el registro limpio en git (salvo
``--allow-dirty``) y la cabecera igual al esquema. No sobrescribe celdas con valor, añade a
``notes`` sin sustituir, relee y compara celda a celda después de escribir y restaura si hay
diferencias. Es idempotente. Como las detecciones por instancia no se versionan, escribe un volcado
versionado con los recuentos, los anclajes y el cuadre de detecciones.

Códigos de salida: 0 si no hay nada que hacer, se muestra el plan o se aplica y verifica, 1 si hay
un error de uso o de entorno, 2 si hay incidencias o falla la verificación posterior.
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
from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths, relative_to_root  # noqa: E402
from tfm_edge.common.registry import TABLE_PRIMARY_KEYS, ExperimentRegistry  # noqa: E402
from tfm_edge.common.registry_guards import check_git_clean, read_header  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.evaluation import coco_eval, error_decomposition, reconstruction  # noqa: E402
from tfm_edge.evaluation.matching_analysis import DECOMPOSITION_CONFIDENCES  # noqa: E402
from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE  # noqa: E402
from tfm_edge.evaluation.runner import load_split_file_names  # noqa: E402

logger = get_logger(__name__)

#: Tabla que se rellena.
TABLE = "quality_metrics"

#: Solo se usan las filas de pycocotools; las de ultralytics no se mezclan.
EVALUATOR = "pycocotools"

#: Las dos rutas de procedencia. El texto va tal cual a notes.
ROUTES: dict[str, str] = {
    "curva": (
        "procedencia del volcado de curva del paso 23: anclada contra el mAP "
        "registrado y contrastada contra la curva ya volcada"
    ),
    "volcado": (
        "procedencia del validator_dir del volcado de evaluacion: anclada contra "
        "el mAP registrado, SIN contraste contra curva porque no se midio"
    ),
}

#: Motivos por los que una fila se queda vacía; también van a notes.
REASONS: dict[str, str] = {
    "sin_detecciones": (
        "sin descomposicion de errores: esta evaluacion no dejo detecciones por "
        "instancia en disco (la ruta de PyTorch vuelca un resumen y no guarda ni "
        "las predicciones ni el directorio del validador), asi que recuperarla "
        "exigiria volver a inferir"
    ),
    "otro_evaluador": (
        "sin descomposicion de errores: la cifra es del evaluador ultralytics y no "
        "se mezcla con las de pycocotools en una comparacion"
    ),
    "otro_split": (
        "sin descomposicion de errores: la particion de esta fila no es la que se "
        "ha reconstruido"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser de la CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Reparte los errores de detección en las seis clases del registro y "
            "rellena las columnas error_* de quality_metrics."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python scripts/34_register_error_decomposition.py\n"
            "  python scripts/34_register_error_decomposition.py --apply\n"
            "  python scripts/34_register_error_decomposition.py --only "
            "s_baseline_default_745e3d08 --apply\n"
        ),
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directorio del registro (por defecto: results/registry).",
    )
    parser.add_argument(
        "--curve-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Volcados de curva del paso 23 (por defecto: results/raw/operating_curve).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Volcados de evaluación y directorios de validador (por defecto: "
        "results/raw/artifact_eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directorio del volcado versionado de este paso (por defecto: "
            "results/raw/error_decomposition). Para una ejecución aislada usa también "
            "--registry-dir con una copia."
        ),
    )
    parser.add_argument(
        "--coco",
        type=Path,
        default=None,
        metavar="FICHERO",
        help="COCO de referencia (por defecto: data/processed/sh17_coco.json).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="FICHERO",
        help="YAML del dataset a la resolución de inferencia (se busca si no se da).",
    )
    parser.add_argument("--split", default="val", help="Partición (por defecto: %(default)s).")
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        metavar="PX",
        help="Resolución de inferencia (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=REFERENCE_CONFIDENCE,
        metavar="C",
        help=(
            "Confianza a la que se toman los seis recuentos del registro (por defecto: "
            "%(default)s). Tiene que estar en la rejilla y se escribe en notes."
        ),
    )
    parser.add_argument(
        "--confidences",
        type=float,
        nargs="+",
        default=list(DECOMPOSITION_CONFIDENCES),
        metavar="C",
        help="Rejilla completa que va al volcado (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="EXPERIMENT_ID",
        help="Limita el trabajo a estos experimentos. Sin esto, se hacen todos.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Escribe de verdad. Sin esto solo se enseña el plan.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permite escribir con el registro sucio en git, a sabiendas.",
    )
    parser.add_argument(
        "--on-conflict",
        choices=("fail", "skip", "overwrite"),
        default="fail",
        help=(
            "Qué hacer con una celda error_* que ya tenga otro valor (por defecto: %(default)s)."
        ),
    )
    add_overwrite_dump_argument(parser, "el volcado de la descomposición de errores")
    parser.add_argument("--log-level", default="INFO", help="Nivel de log.")
    return parser


def load_curve_sources(curve_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Indexa los volcados de curva por ``(experiment_id, runtime, precision)``. Falla si dos
    comparten clave.
    """
    indice: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fichero in sorted(curve_dir.glob("*.json")):
        volcado = json.loads(fichero.read_text(encoding="utf-8"))
        clave = (
            str(volcado.get("experiment_id", "")),
            str(volcado.get("runtime", "")),
            str(volcado.get("precision", "")),
        )
        if not all(clave) or not volcado.get("source", {}).get("predictions"):
            continue
        if clave in indice:
            raise TFMEdgeError(
                f"Dos volcados de curva para {'/'.join(clave)}: "
                f"'{fichero.name}' y el que ya estaba.",
                "Dos curvas de la misma evaluación se contradicen o sobran. No se "
                "elige una: retira la que no toque.",
            )
        indice[clave] = volcado
    return indice


def load_eval_sources(eval_dir: Path, split: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Indexa los volcados de evaluación de una partición, sin los de ``--smoke-test``.

    La partición se lee de ``split`` o, en los volcados antiguos, de ``coco.split``. Falla si un
    volcado no declara partición o si queda más de uno por clave.
    """
    indice: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fichero in sorted(eval_dir.glob("*.json")):
        try:
            volcado = json.loads(fichero.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(volcado, dict) or volcado.get("smoke_test"):
            continue
        clave = (
            str(volcado.get("experiment_id", "")),
            str(volcado.get("runtime", "")),
            str(volcado.get("precision", "")),
        )
        if not all(clave):
            continue
        coco = volcado.get("coco")
        particion = volcado.get("split") or (coco.get("split") if isinstance(coco, dict) else "")
        if not particion:
            raise TFMEdgeError(
                f"El volcado de evaluación '{fichero.name}' no declara su partición.",
                "Sin ella no se sabe si sus detecciones son de la partición que se "
                "reconstruye. Mira de qué evaluación sale antes de seguir.",
            )
        if particion != split:
            continue
        if clave in indice:
            raise TFMEdgeError(
                f"Dos volcados de evaluación no marcados como humo para "
                f"{'/'.join(clave)}: '{fichero.name}' y el que ya estaba.",
                "No se elige uno por orden de fichero: eso es exactamente cómo se "
                "lee la medida equivocada. Comprueba cuál sobra.",
            )
        indice[clave] = volcado
    return indice


def resolve_paths(
    recorded: dict[str, Any], root: Path, eval_dir: Path, route: str
) -> tuple[Path, dict[str, Any]] | None:
    """Localiza las detecciones y devuelve ``(detecciones, volcado_que_ancla)``, o ``None`` si falta
    algo.

    Las rutas antiguas son absolutas; si no existen, se busca el mismo nombre bajo el directorio de
    evaluaciones. En la ruta ``volcado`` el ancla es el propio volcado ya leído.
    """

    def existente(*candidatas: Path) -> Path | None:
        """Primera candidata que sea un fichero."""
        for candidata in candidatas:
            if candidata.is_file():
                return candidata
        return None

    if route == "volcado":
        directorio = Path(str(recorded.get("validator_dir") or ""))
        if not str(directorio):
            return None
        detecciones = existente(
            (directorio if directorio.is_absolute() else root / directorio) / "predictions.json",
            eval_dir / directorio.name / "predictions.json",
        )
        return (detecciones, recorded) if detecciones else None

    fuente = recorded.get("source", {})
    grabada_pred = Path(str(fuente.get("predictions", "")))
    grabada_dump = Path(str(fuente.get("recorded_dump", "")))
    detecciones = existente(
        grabada_pred if grabada_pred.is_absolute() else root / grabada_pred,
        eval_dir / grabada_pred.parent.name / "predictions.json",
    )
    registrado = existente(
        grabada_dump if grabada_dump.is_absolute() else root / grabada_dump,
        eval_dir / grabada_dump.name,
    )
    if detecciones is None or registrado is None:
        return None
    return detecciones, json.loads(registrado.read_text(encoding="utf-8"))


def plan_rows(
    rows: list[dict[str, str]],
    curves: dict[tuple[str, str, str], dict[str, Any]],
    evals: dict[tuple[str, str, str], dict[str, Any]],
    split: str,
    only: list[str] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Decide por qué ruta entra cada fila, o por qué no, y devuelve ``(plan, incidencias)``.

    Falla si una procedencia apunta a más de una fila rellenable.
    """
    plan: list[dict[str, Any]] = []
    incidencias: list[str] = []
    rellenables_por_clave: dict[tuple[str, str, str], list[int]] = {}

    for indice, fila in enumerate(rows):
        clave_corta = (fila["experiment_id"], fila["runtime"], fila["precision"])
        clave = tuple(fila[columna] for columna in TABLE_PRIMARY_KEYS[TABLE])
        entrada: dict[str, Any] = {
            "index": indice,
            "line": indice + 2,
            "key": clave,
            "short_key": clave_corta,
            "route": None,
            "reason": None,
            "recorded": None,
        }

        if only is not None and fila["experiment_id"] not in only:
            entrada["reason"] = "fuera de --only"
            entrada["skip_note"] = False
            plan.append(entrada)
            continue

        if fila["evaluator"] != EVALUATOR:
            entrada["reason"] = "otro_evaluador"
        elif fila["split"] != split:
            entrada["reason"] = "otro_split"
        elif clave_corta in curves:
            entrada["route"] = "curva"
            entrada["recorded"] = curves[clave_corta]
        elif clave_corta in evals and evals[clave_corta].get("validator_dir"):
            entrada["route"] = "volcado"
            entrada["recorded"] = evals[clave_corta]
        else:
            entrada["reason"] = "sin_detecciones"

        if entrada["route"]:
            rellenables_por_clave.setdefault(clave_corta, []).append(indice)
        plan.append(entrada)

    for clave_corta, indices in rellenables_por_clave.items():
        if len(indices) > 1:
            lineas = ", ".join(f"L{i + 2}" for i in indices)
            raise TFMEdgeError(
                f"La procedencia de {'/'.join(clave_corta)} apunta a {len(indices)} "
                f"filas rellenables ({lineas}).",
                "Un volcado identifica tres campos y la clave del registro tiene seis; no se "
                "rellena ninguna fila hasta distinguirlas.",
            )

    for entrada in plan:
        if entrada["reason"] in REASONS:
            incidencias.append(
                f"L{entrada['line']}: {entrada['key'][0]}/{entrada['key'][1]}/"
                f"{entrada['key'][2]} -> {entrada['reason']}"
            )
    return plan, incidencias


def note_suffix(route: str | None, reason: str | None, confidence: float) -> str:
    """Texto que se añade a notes, con su umbral."""
    if route:
        return (
            f"descomposicion de errores (paso 34) a confianza {confidence:.2f}; " f"{ROUTES[route]}"
        )
    return REASONS[reason or "sin_detecciones"]


def build_dump(
    args: argparse.Namespace,
    confidences: list[float],
    rows: list[dict[str, str]],
    results: list[dict[str, Any]],
    curve_dir: Path,
    eval_dir: Path,
    registry_csv: Path,
    coco_path: Path,
) -> str:
    """Serializa el volcado de evidencia como texto listo para escribir.

    Se construye antes de tocar el registro para que un fallo al serializar no deje celdas sin
    evidencia. Las rutas usan relative_to_root, que deja absolutas las que quedan fuera del
    proyecto.
    """
    limpios = [{k: v for k, v in resultado.items() if k != "entry"} for resultado in results]
    contenido = {
        "registry_confidence": args.confidence,
        "confidences": confidences,
        "split": args.split,
        "input_resolution": args.resolution,
        "kinds": list(error_decomposition.ERROR_KINDS),
        "registry_columns": dict(error_decomposition.REGISTRY_COLUMNS),
        "routes": dict(ROUTES),
        "reasons": dict(REASONS),
        "coverage": {
            "quality_rows": len(rows),
            "with_decomposition": len(limpios),
            "by_route": {ruta: sum(1 for r in limpios if r["route"] == ruta) for ruta in ROUTES},
            "without_decomposition": len(rows) - len(limpios),
        },
        "source": {
            "curve_dir": relative_to_root(curve_dir),
            "eval_dir": relative_to_root(eval_dir),
            "registry": relative_to_root(registry_csv),
            "coco": relative_to_root(coco_path),
        },
        "evaluations": limpios,
    }
    return json.dumps(contenido, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    """Punto de entrada del script."""
    args = build_parser().parse_args()
    setup_logging("34_register_error_decomposition", level=args.log_level)
    set_global_seed(0)

    rutas = get_project_paths()
    curve_dir = args.curve_dir or (rutas.results_raw / "operating_curve")
    eval_dir = args.eval_dir or (rutas.results_raw / "artifact_eval")
    salida = args.output_dir or (rutas.results_raw / error_decomposition.DUMP_SUBDIR)
    coco_path = args.coco or (rutas.data_processed / "sh17_coco.json")
    umbrales = sorted(float(c) for c in args.confidences)

    print("=" * 110)
    print("PASO 34: DESCOMPOSICIÓN DE LOS TIPOS DE ERROR")
    print("=" * 110)

    if not any(abs(c - args.confidence) < 1e-9 for c in umbrales):
        print(
            f"ERROR: la confianza de referencia {args.confidence} no está en la rejilla "
            f"{umbrales}.",
            file=sys.stderr,
        )
        return 1
    if not coco_path.is_file():
        print(f"ERROR: no existe el COCO de referencia '{coco_path}'.", file=sys.stderr)
        return 1

    # Las entradas de este volcado no se versionan y no se puede recalcular en un clon limpio, así
    # que se protege con la opción común antes de reconstruir nada.
    destino_volcado = salida / error_decomposition.dump_name(args.confidence)
    try:
        refuse_to_overwrite_dump(
            destino_volcado,
            overwrite=args.overwrite_dump,
            what="el volcado de la descomposición de errores",
        )
    except FileExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    try:
        # El esquema lleva una columna de AP por clase, así que hacen falta los nombres de las
        # clases.
        clases = dict(
            enumerate(load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json"))
        )
        registro = ExperimentRegistry(
            registry_dir=args.registry_dir, class_names=[clases[i] for i in sorted(clases)]
        )
        filas = registro.read_rows(TABLE)
        curvas = load_curve_sources(curve_dir)
        volcados = load_eval_sources(eval_dir, args.split)
        plan, incidencias = plan_rows(filas, curvas, volcados, args.split, args.only)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    destino_csv = registro.path_for(TABLE)
    esquema = registro.columns_for(TABLE)
    cabecera = read_header(destino_csv)
    if cabecera != tuple(esquema):
        print(
            f"ERROR: la cabecera de '{destino_csv}' no coincide con el esquema.",
            file=sys.stderr,
        )
        return 1

    # Se comprueba git antes de reconstruir, que lleva minutos, y otra vez antes de escribir.
    if args.apply and not args.allow_dirty:
        motivo_temprano = check_git_clean(registro.registry_dir)
        if motivo_temprano:
            print(
                f"ERROR: el registro no está limpio en git ({motivo_temprano}). Confirma los "
                "cambios o descártalos, o usa --allow-dirty.",
                file=sys.stderr,
            )
            return 2

    rellenables = [e for e in plan if e["route"]]
    print(f"\nFilas de calidad:   {len(filas)}")
    print(f"Con reparto:        {len(rellenables)}")
    for ruta in ROUTES:
        print(f"  por {ruta:<8}      {sum(1 for e in rellenables if e['route'] == ruta)}")
    print(f"Sin reparto:        {len(filas) - len(rellenables)}")
    print(f"Confianza al CSV:   {args.confidence:.2f}")
    print(f"Rejilla al volcado: {', '.join(f'{c:.2f}' for c in umbrales)}")
    print(f"Volcado:            {destino_volcado}")

    if not rellenables:
        print("\nNada que repartir.")
        return 0

    # Preparación de la referencia, igual que en el paso 25.
    try:
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        nombres = load_split_file_names(rutas.data_splits, args.split, coco)
        referencia = coco_eval.subset_ground_truth(coco, nombres)
        data_yaml = args.data or resize.find_dataset_for_resolution(rutas.data, args.resolution)
    except TFMEdgeError as error:
        print(f"ERROR al preparar la referencia: {error}", file=sys.stderr)
        return 1
    images_dir = Path(data_yaml).parent / "images"
    print(
        f"Referencia:         {len(referencia['images'])} imágenes, "
        f"{len(referencia['annotations'])} anotaciones, {len(clases)} clases"
    )

    # Reparto por evaluación.
    resultados: list[dict[str, Any]] = []
    fallos_de_lectura: list[str] = []
    for entrada in rellenables:
        etiqueta = "/".join(entrada["short_key"])
        piezas = resolve_paths(entrada["recorded"], rutas.root, eval_dir, entrada["route"])
        if piezas is None:
            fallos_de_lectura.append(f"{etiqueta}: no se localizan las detecciones")
            continue
        detecciones_path, ancla = piezas

        try:
            reconstruida = reconstruction.rebuild_evaluation(
                json.loads(detecciones_path.read_text(encoding="utf-8")),
                ancla,
                referencia,
                images_dir,
                args.resolution,
                len(clases),
            )
            reparto = error_decomposition.decompose(reconstruida.evaluator, umbrales)
            cuadre = error_decomposition.detection_accounting(
                reconstruida.evaluator,
                reconstruida.detections,
                ancla.get("num_raw_detections"),
            )
            # La ruta de curva añade el contraste con los recuentos crudos de la curva, que valida
            # ``missed``.
            if entrada["route"] == "curva":
                contraste = error_decomposition.cross_check_curve(
                    reparto, entrada["recorded"]["curve"]["points"]
                )
            else:
                contraste = {
                    "confidences_checked": [],
                    "matches": None,
                    "why_not": "esta evaluacion no tiene volcado de curva",
                }
        except TFMEdgeError as error:
            fallos_de_lectura.append(f"{etiqueta}: {error}")
            continue

        # Si lo reconstruido no coincide con lo que registró el validador, no es la misma
        # evaluación.
        if not (cuadre["closes_without_remainder"] and cuadre["matches_recorded_raw_count"]):
            fallos_de_lectura.append(
                f"{etiqueta}: el cuadre de detecciones no cierra "
                f"({json.dumps(cuadre, ensure_ascii=False)})"
            )
            continue

        resultados.append(
            {
                "primary_key": dict(zip(TABLE_PRIMARY_KEYS[TABLE], entrada["key"])),
                "route": entrada["route"],
                "route_means": ROUTES[entrada["route"]],
                "predictions": str(detecciones_path.name),
                "anchor_max_difference": reconstruida.anchor["max_difference"],
                "detection_accounting": cuadre,
                "curve_cross_check": contraste,
                "decomposition": reparto,
                "entry": entrada,
            }
        )
        logger.info(
            "%s por la ruta '%s': ancla %.1e, alineamiento %.1e, %d evaluadas",
            etiqueta,
            entrada["route"],
            reconstruida.anchor["max_difference"],
            reparto["own_iou_anchor_max_gap"],
            reparto["num_evaluated_detections"],
        )

    if fallos_de_lectura:
        print(f"\nERROR: {len(fallos_de_lectura)} evaluaciones no se han podido repartir:")
        for fallo in fallos_de_lectura[:10]:
            print(f"  {fallo}", file=sys.stderr)
        print(
            "\nABORTADO. No se escribe un registro rellenado a medias.",
            file=sys.stderr,
        )
        return 2

    # Plan de celdas.
    columnas = error_decomposition.REGISTRY_COLUMNS
    cambios: dict[int, dict[str, str]] = {}
    for resultado in resultados:
        entrada = resultado["entry"]
        fila = filas[entrada["index"]]
        del_umbral = next(
            f
            for f in resultado["decomposition"]["by_confidence"]
            if abs(f["confidence"] - args.confidence) < 1e-9
        )
        celdas = {columnas[clase]: str(del_umbral[clase]) for clase in columnas}
        ya_puestas = {c: fila[c] for c in celdas if (fila[c] or "").strip()}
        if ya_puestas and args.on_conflict == "skip":
            continue
        cambios[entrada["index"]] = celdas

    notas: dict[int, str] = {}
    for entrada in plan:
        if entrada.get("skip_note") is False:
            continue
        sufijo = note_suffix(entrada["route"], entrada["reason"], args.confidence)
        actual = (filas[entrada["index"]]["notes"] or "").strip()
        if sufijo in actual:
            continue
        notas[entrada["index"]] = f"{actual}; {sufijo}" if actual else sufijo

    print(f"\nCeldas a rellenar:  {len(cambios) * len(columnas)} en {len(cambios)} filas")
    print(f"Notas a ampliar:    {len(notas)}")
    if incidencias:
        print(f"\nFilas que se quedan vacías, con su motivo en notes: {len(incidencias)}")
        for linea in incidencias[:6]:
            print(f"  {linea}")
        if len(incidencias) > 6:
            print(f"  ... y {len(incidencias) - 6} más")

    if not args.apply:
        print("\n--- plan, sin escribir nada ---")
        for resultado in resultados[:8]:
            clave = resultado["primary_key"]
            del_umbral = next(
                f
                for f in resultado["decomposition"]["by_confidence"]
                if abs(f["confidence"] - args.confidence) < 1e-9
            )
            print(
                f"  L{resultado['entry']['line']:>3} {clave['experiment_id'][:34]:<36}"
                f"{clave['runtime']:<12}{clave['precision']:<5}"
                f"[{resultado['route']:<8}] "
                + " ".join(f"{c}={del_umbral[c]}" for c in error_decomposition.ERROR_KINDS)
            )
        if len(resultados) > 8:
            print(f"  ... y {len(resultados) - 8} filas más")
        print("\nPlan mostrado. Nada escrito. Pasa --apply para aplicarlo.")
        return 0

    motivo = None if args.allow_dirty else check_git_clean(registro.registry_dir)
    if motivo:
        logger.error("El registro no está limpio en git: %s", motivo)
        print(
            "\nABORTADO. Confirma o descarta esos cambios, o pasa --allow-dirty a " "sabiendas.",
            file=sys.stderr,
        )
        return 2

    # El volcado se serializa antes de escribir en el registro; si después no se puede escribir, se
    # restaura el registro.
    try:
        texto_volcado = build_dump(
            args, umbrales, filas, resultados, curve_dir, eval_dir, destino_csv, coco_path
        )
    except (TypeError, ValueError) as error:
        logger.error("El volcado de evidencia no se puede construir: %s", error)
        print(
            "\n"
            "ABORTADO antes de escribir: no se puede construir el volcado de evidencia "
            f"({error}).",
            file=sys.stderr,
        )
        return 2

    # Escritura.
    previas = [dict(f) for f in filas]
    clave_columnas = TABLE_PRIMARY_KEYS[TABLE]

    actualizaciones = [
        {**dict(zip(clave_columnas, tuple(previas[i][c] for c in clave_columnas))), **celdas}
        for i, celdas in sorted(cambios.items())
    ]
    actualizaciones_notas = [
        {**dict(zip(clave_columnas, tuple(previas[i][c] for c in clave_columnas))), "notes": texto}
        for i, texto in sorted(notas.items())
    ]

    escritas_err: list[Any] = []
    sin_cambio_err: list[Any] = []
    escritas_notas: list[Any] = []
    try:
        if actualizaciones:
            resumen = registro.update_cells(TABLE, actualizaciones, on_conflict=args.on_conflict)
            logger.info(
                "error_*: %d filas escritas, %d sin cambio, %d saltadas.",
                len(resumen["written"]),
                len(resumen["unchanged"]),
                len(resumen["skipped"]),
            )
            escritas_err = resumen["written"]
            sin_cambio_err = resumen["unchanged"]
        # notes se añade, así que se usa overwrite: el valor nuevo ya incluye el anterior.
        if actualizaciones_notas:
            resumen_notas = registro.update_cells(
                TABLE, actualizaciones_notas, on_conflict="overwrite"
            )
            logger.info(
                "notes: %d filas ampliadas, %d sin cambio.",
                len(resumen_notas["written"]),
                len(resumen_notas["unchanged"]),
            )
            escritas_notas = resumen_notas["written"]
    except TFMEdgeError as error:
        logger.error("La escritura ha fallado: %s", error)
        print(f"\nERROR al escribir: {error}", file=sys.stderr)
        return 2

    # Verificación celda a celda sobre la tabla entera.
    releidas = registro.read_rows(TABLE)
    fallos: list[str] = []
    if len(releidas) != len(previas):
        fallos.append(f"el fichero tiene {len(releidas)} filas y tenía {len(previas)}")
    else:
        for indice, (antes, ahora) in enumerate(zip(previas, releidas)):
            linea = indice + 2
            for columna in esquema:
                esperado = (antes.get(columna) or "").strip()
                if indice in cambios and columna in cambios[indice]:
                    esperado = cambios[indice][columna]
                if indice in notas and columna == "notes":
                    esperado = notas[indice]
                obtenido = (ahora.get(columna) or "").strip()
                if obtenido != esperado:
                    fallos.append(
                        f"L{linea}/{columna}: '{obtenido}' donde se esperaba '{esperado}'"
                    )

    if fallos:
        logger.error("La verificación posterior ha fallado; se restaura la imagen previa.")
        for fallo in fallos[:20]:
            logger.error("  %s", fallo)
        registro._write_rows_atomic(TABLE, previas)  # noqa: SLF001
        print(
            f"\nRESTAURADO el estado previo ({len(fallos)} diferencias). Revisa el log.",
            file=sys.stderr,
        )
        return 2

    # Volcado versionado.
    try:
        salida.mkdir(parents=True, exist_ok=True)
        destino_volcado.write_text(texto_volcado, encoding="utf-8")
    except OSError as error:
        logger.error(
            "El registro quedó escrito y el volcado de evidencia no: se restaura. %s", error
        )
        registro._write_rows_atomic(TABLE, previas)  # noqa: SLF001
        print(
            "\n"
            "RESTAURADO el estado previo: no se ha podido escribir el volcado de evidencia "
            f"({error}).",
            file=sys.stderr,
        )
        return 2

    # Se cuenta lo que update_cells dice haber escrito, que devuelve una entrada por celda
    # («clave:columna»); las filas se cuentan como claves distintas.
    def _filas(entradas: list[Any]) -> int:
        """Número de filas distintas en una lista de entradas «clave:columna»."""
        return len({str(e).rsplit(":", 1)[0] for e in entradas})

    print(
        f"\nAplicado y verificado: {len(escritas_err)} celdas error_* escritas en "
        f"{_filas(escritas_err)} filas, {len(sin_cambio_err)} celdas que ya tenían "
        f"ese mismo valor, y {_filas(escritas_notas)} notas ampliadas."
    )
    print(f"Volcado versionado en '{destino_volcado}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
