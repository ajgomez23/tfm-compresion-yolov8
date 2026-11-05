#!/usr/bin/env python
"""Combinaciones de poda y PTQ INT8 para la rejilla del apartado 5.5.

Toma checkpoints podados y reajustados del registro, materializa INT8 desde cada uno con
``materialize_int8_branch`` (la misma función que PTQ y QAT) e inscribe una fila
``technique=combined`` por partida, con ``source_kind="pruned"``. Con ``--evaluate`` llama a
``11_evaluate_artifact.py`` por cada artefacto, en un proceso aparte, para que haya un solo
productor de filas de calidad.

La nota se deriva de ``training_epochs`` del padre, porque el fichero de pesos no dice si el podado
se reajustó. No cuantiza podados sin reajustar ni sobrescribe artefactos existentes salvo con
``--overwrite-artifacts`` (o ``--skip-existing`` para seguir con los que faltan).

Códigos de salida: 0 si todas las partidas producen artefacto (y evaluación, si se pide), 1 si hay
un error de configuración, 2 si alguna partida queda sin artefacto en algún runtime, 3 si los
artefactos están pero falla alguna evaluación.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import compute_config_hash  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import (  # noqa: E402
    build_experiment_id,
    ensure_dir,
    get_project_paths,
    relativize_dump_paths,
)
from tfm_edge.common.registry import (  # noqa: E402
    VALID_DUPLICATE_POLICIES,
    ExperimentRegistry,
    utc_timestamp,
)
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.compression.quantization.ptq import materialize_int8_branch  # noqa: E402
from tfm_edge.data.calibration_set import (  # noqa: E402
    load_calibration_set,
    resolve_calibration_images,
)
from tfm_edge.export.quantize import Int8Config, calibration_hash  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402

logger = get_logger(__name__)

#: Sufijo de la variante combinada; ``prune-...-ptqint8`` indica podar y después cuantizar.
COMBINED_SUFFIX = "ptqint8"

#: Técnica de la que tienen que ser los padres admisibles.
PARENT_TECHNIQUE = "pruning"

EPILOGO = """Ejemplos:
  # La rejilla entera: los podados reajustados del registro, los dos runtimes
  python scripts/22_combine.py --all-pruned --evaluate

  # Una sola partida, con directorios desechables
  python scripts/22_combine.py --parents s_pruning_l1global_r50_9419b26f
      --output-dir /tmp/x --registry-dir /tmp/x/registry
"""


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="22_combine.py",
        description=(
            "Materializa INT8 desde checkpoints podados y reajustados, e inscribe "
            "cada combinación como una variante combined. Con --evaluate encadena "
            "la medida de calidad de cada artefacto."
        ),
        epilog=EPILOGO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    seleccion = parser.add_mutually_exclusive_group(required=True)
    seleccion.add_argument(
        "--parents",
        nargs="+",
        default=[],
        metavar="ID",
        help="Experimentos de poda de los que partir, por su experiment_id.",
    )
    seleccion.add_argument(
        "--all-pruned",
        action="store_true",
        help=("Todos los podados reajustados que haya en el registro."),
    )
    parser.add_argument(
        "--runtimes",
        nargs="+",
        default=["onnxruntime", "openvino"],
        choices=["onnxruntime", "openvino"],
        metavar="NOMBRE",
        help="Runtimes en los que materializar INT8 (por defecto: los dos).",
    )
    parser.add_argument(
        "--calibration-set",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Lista de calibración (por defecto: data/calibration/calibration_128.txt).",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=None,
        metavar="N",
        help="Imágenes de calibración a usar (por defecto: las de la lista).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de imágenes (por defecto: data/resized_640/images).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="NODO",
        help=(
            "Nodos a dejar en coma flotante. Vacío por defecto, igual que en la rama PTQ sin "
            "combinar."
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de entrada."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Raíz de los artefactos (por defecto: results/exports/combined).",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Directorio del registro (por defecto: results/registry). Apúntalo a "
            "otro sitio al probar: el registro del estudio no recoge pruebas."
        ),
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="overwrite",
        choices=list(VALID_DUPLICATE_POLICIES),
        help="Qué hacer si la fila ya existe (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--overwrite-artifacts",
        action="store_true",
        help="Rehace los artefactos de una partida aunque ya existan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Salta las partidas que ya tienen artefactos y sigue con las demás. Es "
            "lo que hace reanudable una rejilla que se cortó a la mitad."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help=(
            "Encadena 11_evaluate_artifact.py sobre cada artefacto producido. Sin esta opción "
            "la calidad se mide después."
        ),
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Volcados de evaluación (por defecto: el del paso 11).",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de preparación del paso 11 (por defecto: el suyo).",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="workstation_cpu",
        metavar="ID",
        help="Dispositivo donde se evalúa (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        metavar="TEXTO",
        help=(
            "Texto adicional para las filas. La parte que dice si el padre estaba "
            "reajustado NO se teclea: se deriva del registro."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Materializa pero no toca el registro."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Calibración mínima (8 imágenes) y una sola partida. NO es un resultado.",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    return parser


def seleccionar_padres(
    filas: list[dict[str, str]], pedidos: list[str], todos: bool
) -> tuple[list[dict[str, str]], list[str]]:
    """Elige y valida las filas de poda de las que puede partir la rejilla y devuelve ``(padres,
    motivos)``.

    Un padre válido es una fila de pruning con status ok, pesos en disco y reajustada.
    """
    por_id = {f["experiment_id"]: f for f in filas}
    motivos: list[str] = []

    if todos:
        # Se toman todas las filas de poda para que el log indique por qué se descarta cada una.
        candidatos = [f["experiment_id"] for f in filas if f["technique"] == PARENT_TECHNIQUE]
    else:
        candidatos = list(pedidos)

    admitidos: list[dict[str, str]] = []
    for identificador in candidatos:
        fila = por_id.get(identificador)
        if fila is None:
            motivos.append(f"{identificador}: no está en el registro")
            continue
        if fila["technique"] != PARENT_TECHNIQUE:
            motivos.append(
                f"{identificador}: es de técnica {fila['technique']}, y esta rejilla "
                f"parte de {PARENT_TECHNIQUE}"
            )
            continue
        if fila["status"] != "ok":
            motivos.append(f"{identificador}: la fila está como {fila['status']}")
            continue
        if not fila["training_epochs"].strip():
            motivos.append(
                f"{identificador}: SIN REAJUSTAR (training_epochs vacío). Cuantizar "
                f"un podado que no se ha reajustado no mide la combinación de dos "
                f"técnicas: mide una poda rota, dos veces"
            )
            continue
        if not fila["weights_path"].strip() or not Path(fila["weights_path"]).is_file():
            motivos.append(f"{identificador}: no hay pesos en {fila['weights_path'] or '(vacío)'}")
            continue
        admitidos.append(fila)

    return admitidos, motivos


def construir_variante(variante_padre: str) -> str:
    """Deriva la etiqueta combinada de la del padre, por ejemplo ``l1global_r50_iter4`` da
    ``prune-l1global-r50-iter4-ptqint8``.
    """
    cuerpo = variante_padre.strip().replace("_", "-")
    return f"prune-{cuerpo}-{COMBINED_SUFFIX}"


def construir_identidad(
    *,
    model_size: str,
    variante: str,
    padre: Mapping[str, str],
    input_resolution: int,
    int8_config: Mapping[str, Any],
    calibration_set: str,
    calibration_hash: str,
    runtimes: Sequence[str],
    seed: int,
) -> tuple[dict[str, Any], str, str]:
    """Construye el payload que identifica una rama combinada y devuelve ``(payload, config_hash,
    experiment_id)``.

    ``calibration_set`` es el nombre de la lista de calibración, no su ruta, para que no dependa de
    la máquina.
    """
    payload = {
        "model_family": "yolov8",
        "model_size": model_size,
        "technique": "combined",
        "variant": variante,
        # El padre entra en el hash: dos podas distintas con la misma
        # etiqueta de variante no pueden compartir identificador.
        "parent_experiment_id": padre["experiment_id"],
        "parent_config_hash": padre["config_hash"],
        "input_resolution": input_resolution,
        "int8_config": dict(int8_config),
        "calibration_set": calibration_set,
        "calibration_hash": calibration_hash,
        "runtimes": list(runtimes),
        "seed": seed,
    }
    config_hash = compute_config_hash(payload)
    experiment_id = build_experiment_id(
        model_size=model_size,
        technique="combined",
        variant=variante,
        config_hash=config_hash,
    )
    return payload, config_hash, experiment_id


def volcar_informe(
    output_dir: Path,
    *,
    experiment_id: str,
    resultado: Any,
    checkpoint: str,
    parent_experiment_id: str,
    calibration_set: Path,
    calibration_images: Sequence[Path],
    identity_payload: Mapping[str, Any],
) -> Path:
    """Escribe el informe de una rama combinada."""
    destino = output_dir / f"{experiment_id}_combined.json"
    datos = resultado.to_dict()
    datos["checkpoint"] = checkpoint
    datos["parent_experiment_id"] = parent_experiment_id
    datos["calibration_set"] = str(calibration_set)
    datos["calibration_image_names"] = [i.name for i in calibration_images]
    # Experimento al que pertenece el informe.
    datos["experiment_id"] = experiment_id
    # El payload exacto que se hashea; un test comprueba que su hash da el sufijo del experiment_id.
    datos["experiment_identity_payload"] = dict(identity_payload)
    destino.parent.mkdir(parents=True, exist_ok=True)
    # Rutas relativas al serializar, porque el informe se versiona.
    destino.write_text(
        json.dumps(relativize_dump_paths(datos), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destino


def construir_nota(padre: dict[str, str], extra: str = "") -> str:
    """Compone la nota de una combinación a partir de la fila del padre.

    Indica si el podado se reajustó, cuánto y con qué fila se compara.
    """
    reduccion = padre["effective_params_ratio"].strip()
    porcentaje = f"{(1 - float(reduccion)) * 100:.1f}%" if reduccion else "no consta"
    partes = [
        (
            f"COMBINACIÓN poda->PTQ. Parte del podado {padre['experiment_id']} "
            f"(variante {padre['technique_variant']}, reducción medida de parámetros "
            f"{porcentaje}), REAJUSTADO {padre['training_epochs']} épocas antes de "
            f"cuantizar."
        ),
        (
            "La comparación que aísla el coste de cuantizar es contra la fila FP32 "
            "de ese mismo podado; contra el baseline mezcla las dos técnicas."
        ),
        extra.strip(),
    ]
    return " ".join(p for p in partes if p)


def hay_artefactos(directorio: Path) -> bool:
    """Indica si una partida ya tiene artefactos en disco."""
    return directorio.is_dir() and any(directorio.iterdir())


def evaluar_artefacto(
    args: argparse.Namespace,
    experiment_id: str,
    runtime: str,
    artefacto: str,
    nota: str,
) -> int:
    """Ejecuta ``11_evaluate_artifact.py`` sobre un artefacto en un proceso aparte y devuelve su
    código de salida.
    """
    script = Path(__file__).resolve().parent / "11_evaluate_artifact.py"
    orden = [
        sys.executable,
        str(script),
        "--artifact",
        artefacto,
        "--runtime",
        runtime,
        "--precision",
        "int8",
        "--experiment-id",
        experiment_id,
        "--resolution",
        str(args.resolution),
        "--device-id",
        args.device_id,
        "--on-duplicate",
        args.on_duplicate,
        "--seed",
        str(args.seed),
        "--notes",
        nota,
    ]
    for bandera, valor in (
        ("--registry-dir", args.registry_dir),
        ("--output-dir", args.eval_output_dir),
        ("--staging-dir", args.staging_dir),
    ):
        if valor is not None:
            orden += [bandera, str(valor)]
    # La prueba de humo se propaga como --smoke-test, que evalúa 40 imágenes en vez de 1213.
    if args.smoke_test:
        orden.append("--smoke-test")
    elif args.dry_run:
        orden.append("--dry-run")

    logger.info("Evaluando %s en %s", experiment_id, runtime)
    # Se vacía el búfer antes de que el hijo escriba para no desordenar el log.
    sys.stdout.flush()
    completado = subprocess.run(orden, check=False)
    return completado.returncode


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("22_combine", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    lista = args.calibration_set or (rutas.data_calibration / "calibration_128.txt")
    directorio_imagenes = args.images_dir or (rutas.data / "resized_640" / "images")
    raiz_salida = ensure_dir(args.output_dir or (rutas.results / "exports" / "combined"))
    directorio_registro = args.registry_dir or (rutas.results / "registry")
    registro = ExperimentRegistry(directorio_registro)

    print("=" * 78)
    print("PASO 22: COMBINACIONES DE PODA Y PTQ INT8")
    print("=" * 78)

    try:
        filas = registro.read_rows("experiments")
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    padres, motivos = seleccionar_padres(filas, args.parents, args.all_pruned)
    for motivo in motivos:
        print(f"DESCARTADO  {motivo}")
    if not padres:
        print()
        print("ERROR: no queda ninguna partida admisible.", file=sys.stderr)
        return 1
    if args.smoke_test:
        padres = padres[:1]
        print()
        print("MODO --smoke-test: una sola partida y calibración de 8 imágenes.")
        print("NO es un resultado del estudio.")

    print()
    print(f"Partidas    : {len(padres)}   runtimes: {args.runtimes}")
    print(f"Calibración : {lista}")
    print(f"Exclusiones : {list(args.exclude) or 'ninguna'}")
    print(f"Destino     : {raiz_salida}")
    print(f"Registro    : {directorio_registro}")
    print()

    pedidas = 8 if args.smoke_test else args.calibration_size
    try:
        stems = load_calibration_set(lista)
        imagenes = resolve_calibration_images(stems, directorio_imagenes, pedidas)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    # Igual que en el paso 10, el tamaño declarado es el de las imágenes resueltas.
    configuracion = Int8Config(
        calibration_size=len(imagenes),
        excluded_nodes=tuple(args.exclude),
        seed=args.seed,
    )
    huella_calibracion = calibration_hash(imagenes)
    print(f"Imágenes de calibración resueltas: {len(imagenes)}")
    print(f"Hash del conjunto de calibración : {huella_calibracion}")
    print()

    sin_artefacto: list[str] = []
    evaluaciones_fallidas: list[str] = []
    producidos: list[dict[str, Any]] = []

    for indice, padre in enumerate(padres, start=1):
        variante = construir_variante(padre["technique_variant"])
        print("-" * 78)
        print(f"[{indice}/{len(padres)}] {padre['experiment_id']}  =>  {variante}")

        try:
            detector = Detector.load(Path(padre["weights_path"]), input_resolution=args.resolution)
            detector.assert_float32()
        except TFMEdgeError as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            sin_artefacto.append(f"{padre['experiment_id']}: no se pudo cargar ({error})")
            continue

        # El identificador se calcula antes de materializar para nombrar con él los artefactos.
        try:
            payload_identidad, config_hash, experiment_id = construir_identidad(
                model_size=detector.size,
                variante=variante,
                padre=padre,
                input_resolution=args.resolution,
                int8_config=configuracion.to_dict(),
                calibration_set=lista.name,
                calibration_hash=huella_calibracion,
                runtimes=sorted(args.runtimes),
                seed=args.seed,
            )
        except TFMEdgeError as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            sin_artefacto.append(f"{padre['experiment_id']}: identificador inválido")
            continue

        salida = raiz_salida / experiment_id
        print(f"  experiment_id: {experiment_id}")

        # Lo que ya está en disco no se sobrescribe sin una opción explícita.
        if hay_artefactos(salida) and not args.overwrite_artifacts:
            if args.skip_existing:
                print(f"  SALTADA: ya hay artefactos en {salida}")
                continue
            print(
                f"  ERROR: ya hay artefactos en {salida}. Usa --overwrite-artifacts "
                f"para rehacerlos o --skip-existing para continuar con los demás.",
                file=sys.stderr,
            )
            return 1

        try:
            resultado = materialize_int8_branch(
                detector,
                ensure_dir(salida),
                experiment_id,
                calibration_images=imagenes,
                runtimes=tuple(args.runtimes),
                config=configuracion,
                variant=variante,
                # Esto es lo que distingue esta rama de la PTQ sin combinar.
                source_kind="pruned",
            )
        except TFMEdgeError as error:
            print(f"  ERROR: {error}", file=sys.stderr)
            sin_artefacto.append(f"{experiment_id}: {error}")
            continue

        print()
        print(resultado.summary())

        informe = volcar_informe(
            salida,
            experiment_id=experiment_id,
            resultado=resultado,
            checkpoint=padre["weights_path"],
            parent_experiment_id=padre["experiment_id"],
            calibration_set=lista,
            calibration_images=imagenes,
            identity_payload=payload_identidad,
        )
        print(f"  Informe: {informe}")

        faltan = [r for r in args.runtimes if r not in resultado.runtimes_ok]
        for runtime in faltan:
            sin_artefacto.append(
                f"{experiment_id}/{runtime}: "
                f"{resultado.int8_artifacts.get(runtime, {}).get('notes', 'sin motivo')}"
            )

        nota = construir_nota(padre, args.notes)
        if not args.dry_run:
            registro.append_experiment(
                {
                    "experiment_id": experiment_id,
                    "parent_experiment_id": padre["experiment_id"],
                    "timestamp": utc_timestamp(),
                    "model_family": "yolov8",
                    "model_size": detector.size,
                    "technique": "combined",
                    "technique_variant": variante,
                    "config_hash": config_hash,
                    "config_path": str(informe),
                    # La cuantización no cambia el número de parámetros, así que la compresión
                    # estructural es la del padre. El tamaño en disco está en deployment_metrics.
                    "compression_ratio": padre["compression_ratio"],
                    "effective_params": padre["effective_params"],
                    "effective_params_ratio": padre["effective_params_ratio"],
                    "precision": "int8",
                    # Esta variante no entrena; las épocas de reajuste están en la fila del padre.
                    "training_epochs": "",
                    "training_time_s": "",
                    "training_device": "",
                    "weights_path": padre["weights_path"],
                    "status": "ok" if resultado.runtimes_ok else "fallido",
                    "notes": (
                        nota + (" PRUEBA DE HUMO, no es un resultado." if args.smoke_test else "")
                    ),
                },
                on_duplicate=args.on_duplicate,
            )
            print(f"  Inscrito en el registro: {experiment_id}")

        producidos.append(
            {
                "experiment_id": experiment_id,
                "parent": padre["experiment_id"],
                "nota": nota,
                "artefactos": {
                    r: resultado.int8_artifacts[r]["path"] for r in resultado.runtimes_ok
                },
            }
        )

    # La evaluación va en una segunda pasada para que un fallo no deje la rejilla a medias.
    if args.evaluate:
        print()
        print("=" * 78)
        print("CALIDAD DE CADA ARTEFACTO (paso 11)")
        print("=" * 78)
        for producido in producidos:
            for runtime, artefacto in producido["artefactos"].items():
                codigo = evaluar_artefacto(
                    args, producido["experiment_id"], runtime, artefacto, producido["nota"]
                )
                if codigo != 0:
                    evaluaciones_fallidas.append(
                        f"{producido['experiment_id']}/{runtime}: código {codigo}"
                    )

    print()
    print("=" * 78)
    print(f"Partidas con artefacto: {len(producidos)} de {len(padres)}")
    if sin_artefacto:
        print("SIN artefacto:")
        for fallo in sin_artefacto:
            print(f"  {fallo}")
    if evaluaciones_fallidas:
        print("Evaluaciones fallidas:")
        for fallo in evaluaciones_fallidas:
            print(f"  {fallo}")
    if not args.evaluate:
        print()
        print("Los artefactos están; la calidad NO se ha medido. Vuelve a lanzar con")
        print("--evaluate, o mide cada artefacto con 11_evaluate_artifact.py.")
    print("=" * 78)

    if sin_artefacto:
        return 2
    if evaluaciones_fallidas:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
