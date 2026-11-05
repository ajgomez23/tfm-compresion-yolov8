#!/usr/bin/env python
"""Cuantización posterior al entrenamiento (PTQ) a INT8 en los runtimes disponibles.

Exporta el modelo a FP32 una sola vez y materializa INT8 desde ahí en cada runtime. Por defecto no
excluye ninguna capa: el orden de sensibilidad de 09_sensitivity.py se mide sin fusionar la
BatchNorm y el corte cambia al fusionarla, y además hace falta la referencia sin exclusiones. Usa el
mismo camino de export/quantize.py que la rama QAT. No mide latencia; eso lo hace 07_benchmark.py.

Códigos de salida: 0 si se materializa INT8 en todos los runtimes, 1 si hay un error de
configuración, 2 si algún runtime no produce artefacto (queda registrado con su motivo).
"""

from __future__ import annotations

import argparse
import json
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
from tfm_edge.data.download import compute_sha256  # noqa: E402
from tfm_edge.export.quantize import Int8Config, calibration_hash  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402

logger = get_logger(__name__)

#: Variante por defecto; el nombre indica que no se excluye ninguna capa.
DEFAULT_VARIANT = "int8-noexcl"


def construir_identidad(
    *,
    model_size: str,
    variante: str,
    resolucion: int,
    configuracion: Int8Config,
    nombre_calibracion: str,
    huella_calibracion: str,
    runtimes: list[str],
    seed: int,
    huella_checkpoint: str,
    parent_experiment_id: str,
) -> tuple[dict[str, Any], str, str]:
    """Construye el payload que identifica una PTQ y devuelve ``(payload, config_hash,
    experiment_id)``.

    Incluye la huella del contenido del checkpoint de partida y el parent_experiment_id, para que
    dos PTQ sobre checkpoints distintos no compartan identificador ni fichero.
    """
    payload = {
        "model_family": "yolov8",
        "model_size": model_size,
        "technique": "ptq",
        "variant": variante,
        "input_resolution": resolucion,
        "int8_config": configuracion.to_dict(),
        "calibration_set": nombre_calibracion,
        "calibration_hash": huella_calibracion,
        "runtimes": sorted(runtimes),
        "seed": seed,
        # Punto de partida.
        "starting_checkpoint_sha256": huella_checkpoint,
        "parent_experiment_id": parent_experiment_id,
    }
    config_hash = compute_config_hash(payload)
    experiment_id = build_experiment_id(
        model_size=model_size,
        technique="ptq",
        variant=variante,
        config_hash=config_hash,
    )
    return payload, config_hash, experiment_id


def volcar_informe(
    output_dir: Path,
    *,
    experiment_id: str,
    resultado: Any,
    checkpoint: Path | str,
    calibration_set: Path | str,
    calibration_images: Sequence[Path],
    identity_payload: Mapping[str, Any],
) -> Path:
    """Escribe el informe de la materialización INT8.

    Aquí ``--dry-run`` significa materializar sin escribir en el registro, así que el informe se
    escribe igualmente.
    """
    destino = output_dir / f"{experiment_id}_ptq.json"
    datos = resultado.to_dict()
    # La procedencia va dentro del informe, no solo en el nombre.
    datos["experiment_id"] = experiment_id
    datos["checkpoint"] = str(checkpoint)
    datos["calibration_set"] = str(calibration_set)
    datos["calibration_image_names"] = [imagen.name for imagen in calibration_images]
    # El payload exacto que se hashea; un test comprueba que su hash da el sufijo del experiment_id.
    datos["experiment_identity_payload"] = dict(identity_payload)
    destino.parent.mkdir(parents=True, exist_ok=True)
    # Las rutas del informe se escriben relativas a la raíz porque se versiona; en memoria siguen
    # absolutas.
    destino.write_text(
        json.dumps(relativize_dump_paths(datos), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destino


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="10_quantize_ptq.py",
        description=(
            "Aplica PTQ a INT8 sobre un checkpoint entrenado en los runtimes indicados y lo "
            "inscribe en el registro. Por defecto no excluye ninguna capa."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # PTQ del baseline principal en los dos runtimes\n"
            "  python scripts/10_quantize_ptq.py \\\n"
            "      --checkpoint models/baselines/s_baseline_default/best.pt\n"
            "\n"
            "  # Solo ONNX Runtime, con una calibración más pequeña\n"
            "  python scripts/10_quantize_ptq.py --checkpoint ... \\\n"
            "      --runtimes onnxruntime --calibration-size 64\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Pesos entrenados de partida (rama 'baseline').",
    )
    parser.add_argument(
        "--parent-experiment-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Experimento del que desciende esta variante. Sin él, la fila queda "
            "sin enlazar con el baseline del que salió."
        ),
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
        help="Imágenes de calibración a usar (por defecto: las del Int8Config).",
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
            "Nodos que se dejan en coma flotante. Vacío por defecto para tener la referencia "
            "sin exclusiones."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        metavar="ETIQUETA",
        help=f"Etiqueta de la variante (por defecto: {DEFAULT_VARIANT}).",
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de entrada."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de artefactos (por defecto: results/exports/<experiment_id>).",
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
        "--dry-run", action="store_true", help="Materializa pero no toca el registro."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Calibración mínima (8 imágenes). NO es un resultado del estudio.",
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


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("10_quantize_ptq", level=args.log_level)
    set_global_seed(args.seed)

    if not args.checkpoint.is_file():
        print(f"ERROR: no existe el checkpoint '{args.checkpoint}'.", file=sys.stderr)
        return 1

    rutas = get_project_paths()
    lista = args.calibration_set or (rutas.data_calibration / "calibration_128.txt")
    directorio_imagenes = args.images_dir or (rutas.data / "resized_640" / "images")
    variante = args.variant or DEFAULT_VARIANT

    print("=" * 78)
    print("PASO 10: CUANTIZACIÓN POST-ENTRENAMIENTO A INT8 (PTQ)")
    print("=" * 78)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Variante    : {variante}   runtimes: {args.runtimes}")
    print(f"Calibración : {lista}")
    print(f"Exclusiones : {list(args.exclude) or 'ninguna'}")
    if args.smoke_test:
        print()
        print("MODO --smoke-test: calibración de 8 imágenes. NO es un resultado.")
    print()

    pedidas = 8 if args.smoke_test else args.calibration_size

    try:
        stems = load_calibration_set(lista)
        imagenes = resolve_calibration_images(stems, directorio_imagenes, pedidas)

        # calibration_size se fija a las imágenes resueltas, no a las pedidas, para que el registro
        # y el config_hash describan lo que se ejecutó.
        configuracion = Int8Config(
            calibration_size=len(imagenes),
            excluded_nodes=tuple(args.exclude),
            seed=args.seed,
        )

        # El hash del conjunto sale de las rutas, así queda registrado aunque falle algún runtime.
        huella_calibracion = calibration_hash(imagenes)
        print(f"Imágenes de calibración resueltas: {len(imagenes)}")
        print(f"Hash del conjunto de calibración : {huella_calibracion}")

        detector = Detector.load(args.checkpoint, input_resolution=args.resolution)
        detector.assert_float32()
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    # La identidad se calcula antes de materializar para nombrar con ella directorio, artefactos e
    # informe.
    huella_checkpoint = compute_sha256(args.checkpoint)
    payload_identidad, config_hash, experiment_id = construir_identidad(
        model_size=detector.size,
        variante=variante,
        resolucion=args.resolution,
        configuracion=configuracion,
        nombre_calibracion=lista.name,
        huella_calibracion=huella_calibracion,
        runtimes=list(args.runtimes),
        seed=args.seed,
        huella_checkpoint=huella_checkpoint,
        parent_experiment_id=args.parent_experiment_id,
    )
    salida = ensure_dir(args.output_dir or (rutas.results / "exports" / experiment_id))
    print(f"Huella del checkpoint            : sha256 {huella_checkpoint}")
    print(f"experiment_id: {experiment_id}")
    print(f"Destino     : {salida}")

    try:
        resultado = materialize_int8_branch(
            detector,
            salida,
            experiment_id,
            calibration_images=imagenes,
            runtimes=tuple(args.runtimes),
            config=configuracion,
            variant=variante,
            # La rama PTQ parte del baseline; QAT llama a la misma función.
            source_kind="baseline",
        )
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print()
    print(resultado.summary())

    informe = volcar_informe(
        salida,
        experiment_id=experiment_id,
        resultado=resultado,
        checkpoint=args.checkpoint,
        calibration_set=lista,
        calibration_images=imagenes,
        identity_payload=payload_identidad,
    )
    print()
    print(f"Informe detallado en: {informe}")

    if not args.dry_run:
        registro = ExperimentRegistry(args.registry_dir or (rutas.results / "registry"))
        registro.append_experiment(
            {
                "experiment_id": experiment_id,
                "parent_experiment_id": args.parent_experiment_id,
                "timestamp": utc_timestamp(),
                "model_family": "yolov8",
                "model_size": detector.size,
                "technique": "ptq",
                "technique_variant": variante,
                "config_hash": config_hash,
                "config_path": str(informe),
                # PTQ no cambia el número de parámetros sino los bits de cada uno; el tamaño real
                # está en deployment_metrics.
                "compression_ratio": 1.0,
                "effective_params": "",
                "effective_params_ratio": 1.0,
                "precision": "int8",
                "training_epochs": "",
                "training_time_s": "",
                "training_device": "",
                "weights_path": str(args.checkpoint),
                "status": "ok" if resultado.runtimes_ok else "fallido",
                "notes": (
                    f"PTQ sin exclusiones (D-069); calibración "
                    f"{resultado.calibration_images} imágenes, hash "
                    f"{huella_calibracion}."
                    + (" PRUEBA DE HUMO, no es un resultado." if args.smoke_test else "")
                ),
            },
            on_duplicate=args.on_duplicate,
        )
        print(f"Inscrito en el registro como {experiment_id}")
        print()
        print("La fila de deployment_metrics la escribe 07_benchmark.py sobre cada")
        print("artefacto, con el protocolo homogéneo y sobre un perfil válido.")

    print()
    print("=" * 78)
    faltan = [r for r in args.runtimes if r not in resultado.runtimes_ok]
    print(f"INT8 materializado en: {resultado.runtimes_ok or 'ninguno'}")
    if faltan:
        print(f"SIN artefacto: {faltan}")
        for runtime in faltan:
            print(f"  {runtime}: {resultado.int8_artifacts.get(runtime, {}).get('notes', '?')}")
    print("=" * 78)
    return 2 if faltan else 0


if __name__ == "__main__":
    sys.exit(main())
