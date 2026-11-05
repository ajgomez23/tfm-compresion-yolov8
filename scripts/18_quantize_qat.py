#!/usr/bin/env python
"""Materializa el INT8 de la rama QAT por el mismo camino que la rama PTQ.

Quita la simulación de los pesos afinados (el checkpoint no se modifica), exporta a FP32 y
materializa INT8 con la misma Int8Config y la misma calibración que la PTQ, y después compara las
dos ramas con compare_with_qat. No crea un experimento nuevo: los artefactos son de la variante QAT
que ya escribió 16_qat.py. Antes de materializar comprueba que config_hash y calibration_hash
coinciden con el informe PTQ. Para un QAT sobre un modelo podado, la referencia es la PTQ de ese
mismo modelo; la pareja la autoriza resolve_d018_pair contra experiments.csv y exige
``--experiment-id``.

Códigos de salida: 0 si se materializa INT8 en todos los runtimes y coinciden los caminos, 1 si hay
un error de configuración o de uso, 2 si falta algún artefacto o los caminos no coinciden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths, relativize_dump_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.compression.quantization import qat  # noqa: E402
from tfm_edge.compression.quantization.finetune import (  # noqa: E402
    count_fake_quant_modules,
    wrap_prepared_model,
)
from tfm_edge.compression.quantization.ptq import (  # noqa: E402
    Int8BranchResult,
    compare_with_qat,
    materialize_int8_branch,
    resolve_d018_pair,
)
from tfm_edge.data.calibration_set import (  # noqa: E402
    load_calibration_set,
    resolve_calibration_images,
)
from tfm_edge.export.quantize import Int8Config, calibration_hash  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402

logger = get_logger(__name__)

#: Misma etiqueta de variante que la rama PTQ, para compararlas en la tabla del apartado 5.2.
DEFAULT_VARIANT = "int8-noexcl"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="18_quantize_qat.py",
        description=(
            "Materializa el INT8 de la rama QAT por el mismo camino que la rama PTQ y comprueba "
            "que coinciden."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Materializar y contrastar con el informe de la rama PTQ\n"
            "  python scripts/18_quantize_qat.py \\\n"
            "      --checkpoint models/qat/s_qat_int8-noexcl_d4d403fa/"
            "qat_stripped_fp32.pt \\\n"
            "      --ptq-report results/exports/s_baseline_default_ptq/"
            "s_baseline_default_ptq.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        metavar="RUTA",
        help="Pesos afinados con QAT. Si llevan nodos de simulación, se quitan al cargarlos.",
    )
    parser.add_argument(
        "--ptq-report",
        type=Path,
        required=True,
        metavar="RUTA",
        help=(
            "Informe JSON de la rama PTQ, del que salen la configuración y la calibración. Para "
            "un QAT sobre un modelo podado, el informe de la PTQ de ese mismo modelo."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Experimento QAT al que pertenecen estos artefactos. Sirve para autorizar la pareja "
            "contra el registro y contrastar los parámetros del checkpoint. Obligatorio si "
            "--ptq-report es la PTQ de un modelo podado."
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
        "--images-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de imágenes (por defecto: data/resized_640/images).",
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
        help="Directorio de artefactos (por defecto: results/exports/<nombre>_qat).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Calibra con 8 imágenes para probar el camino. No es un resultado y la comparación "
            "con la rama PTQ fallará, porque el hash de calibración no puede coincidir."
        ),
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


def cargar_informe_ptq(ruta: Path) -> dict:
    """Lee el informe de la rama PTQ y comprueba que declara su configuración."""
    if not ruta.is_file():
        raise TFMEdgeError(
            f"No existe el informe de la rama PTQ '{ruta}'.",
            "Ejecuta antes scripts/10_quantize_ptq.py: hacen falta su configuración y su "
            "conjunto de calibración.",
        )
    informe = json.loads(ruta.read_text(encoding="utf-8"))
    if not informe.get("config") or not informe.get("calibration_set"):
        raise TFMEdgeError(
            f"El informe '{ruta}' no declara 'config' o 'calibration_set'.",
            "Sin ellos no se puede repetir la materialización, solo aproximarla.",
        )
    return informe


def despojar_si_hace_falta(detector: Detector) -> tuple[Detector, int]:
    """Quita la cuantización simulada del detector. Devuelve ``(detector, nodos retirados)``."""
    nodos = count_fake_quant_modules(detector.model)
    if not nodos:
        qat.assert_no_simulation_residue(detector.model)
        logger.info("El checkpoint ya venía despojado: no hay simulación que retirar.")
        return detector, 0

    logger.info(
        "El checkpoint trae %d nodos de cuantización simulada. Se quitan en memoria; el fichero "
        "no se modifica.",
        nodos,
    )
    limpio = qat.strip_fake_quantization(detector.model)
    qat.assert_no_simulation_residue(limpio)
    return wrap_prepared_model(limpio, detector), nodos


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("18_quantize_qat", level=args.log_level)
    set_global_seed(args.seed)

    if not args.checkpoint.is_file():
        print(f"ERROR: no existe el checkpoint '{args.checkpoint}'.", file=sys.stderr)
        return 1

    rutas = get_project_paths()
    directorio_imagenes = args.images_dir or (rutas.data / "resized_640" / "images")
    variante = args.variant or DEFAULT_VARIANT
    nombre = args.checkpoint.parent.name
    salida = ensure_dir(args.output_dir or (rutas.results / "exports" / f"{nombre}_qat"))

    print("=" * 78)
    print("PASO 18: MATERIALIZACIÓN INT8 DE LA RAMA QAT")
    print("=" * 78)
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"Referencia  : {args.ptq_report}")
    print(f"Variante    : {variante}   runtimes: {args.runtimes}")
    print(f"Destino     : {salida}")
    if args.smoke_test:
        print()
        print("MODO --smoke-test: calibración de 8 imágenes. NO es un resultado, y")
        print("la comparación con la rama PTQ fallará a propósito.")
    print()

    try:
        informe_ptq = cargar_informe_ptq(args.ptq_report)

        # La pareja la decide el registro: la PTQ del baseline y la de un podado comparten hashes.
        experimentos = ExperimentRegistry().read_rows("experiments")
        pareja = resolve_d018_pair(informe_ptq, args.experiment_id, experimentos)
        print(
            f"Referencia  : source_kind='{informe_ptq.get('source_kind')}', "
            f"pareja autorizada {sorted(pareja)}"
        )

        # La configuración se toma del informe de la rama PTQ, no se reconstruye a mano.
        config_ptq = dict(informe_ptq["config"])
        lista = Path(informe_ptq["calibration_set"])
        if not lista.is_file():
            lista = rutas.data_calibration / lista.name

        stems = load_calibration_set(lista)
        pedidas = 8 if args.smoke_test else config_ptq.get("calibration_size")
        imagenes = resolve_calibration_images(stems, directorio_imagenes, pedidas)

        configuracion = Int8Config(
            **{**config_ptq, "calibration_size": len(imagenes)},
        )
        huella_calibracion = calibration_hash(imagenes)

        print(f"Conjunto de calibración : {lista}")
        print(f"Imágenes resueltas      : {len(imagenes)}")
        huella_ptq = _hash_ptq(informe_ptq)
        print(f"Hash de calibración     : {huella_calibracion}")
        print(f"  el de la rama PTQ     : {huella_ptq}")
        print()

        # Se comprueba antes de materializar, que tarda minutos. Con --smoke-test no se para, para
        # poder recorrer el camino entero; la comprobación final indicará que no son comparables.
        if huella_calibracion != huella_ptq and not args.smoke_test:
            raise TFMEdgeError(
                f"El conjunto de calibración no coincide con el de la rama PTQ: "
                f"{huella_calibracion} frente a {huella_ptq}.",
                "Las dos ramas tienen que calibrar con las mismas imágenes en el mismo orden. "
                "Comprueba que --ptq-report es el informe correcto y que el conjunto de "
                "calibración no se ha regenerado.",
            )
        if huella_calibracion != huella_ptq:
            print("AVISO: la calibración no coincide con la de PTQ, como corresponde")
            print("a --smoke-test. Lo que salga de aquí NO es un resultado.")
            print()

        detector = Detector.load(args.checkpoint, input_resolution=args.resolution)
        detector.assert_float32()
        detector, nodos_retirados = despojar_si_hace_falta(detector)
        print(f"Nodos de simulación retirados al cargar: {nodos_retirados}")
        if args.experiment_id:
            # Comprueba que los pesos son los de ese experimento.
            fila = next(f for f in experimentos if f["experiment_id"] == args.experiment_id)
            esperados = fila.get("effective_params", "").strip()
            contados = detector.count_parameters()
            if not esperados or int(esperados) != contados:
                raise TFMEdgeError(
                    f"El checkpoint tiene {contados:,} parámetros y la fila de "
                    f"'{args.experiment_id}' declara '{esperados}'.",
                    "Los pesos no son los de ese experimento: corrige --checkpoint o "
                    "--experiment-id antes de materializar nada.",
                )
            print(f"Parámetros  : {contados:,}, los de '{args.experiment_id}'")
        print()

        resultado = materialize_int8_branch(
            detector,
            salida,
            nombre,
            calibration_images=imagenes,
            runtimes=tuple(args.runtimes),
            config=configuracion,
            variant=variante,
            # La única diferencia admisible con la rama PTQ.
            source_kind="qat",
        )
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print()
    print(resultado.summary())

    # Comparación de los dos caminos, runtime por runtime
    print()
    print("=" * 78)
    print("COMPARACIÓN: ¿siguen las dos ramas el mismo camino?")
    print("=" * 78)
    referencia = Int8BranchResult(
        source_kind=str(informe_ptq.get("source_kind", "")),
        variant=str(informe_ptq.get("variant", "")),
        int8_artifacts=dict(informe_ptq.get("int8_artifacts") or {}),
    )
    comparaciones: dict[str, dict] = {}
    fallos: list[str] = []
    for runtime in args.runtimes:
        try:
            comparaciones[runtime] = compare_with_qat(
                referencia, resultado.int8_artifacts, runtime, admitted_pair=pareja
            )
            print(
                f"  {runtime:14s} MISMO CAMINO (config {comparaciones[runtime]['config_hash']}, "
                f"calibración {comparaciones[runtime]['calibration_hash']})"
            )
        except TFMEdgeError as error:
            fallos.append(f"{runtime}: {error}")
            print(f"  {runtime:14s} NO COMPARABLE")
            print(f"                 {error}")

    informe = salida / f"{nombre}_qat_int8.json"
    datos = resultado.to_dict()
    datos["checkpoint"] = str(args.checkpoint)
    datos["experiment_id"] = args.experiment_id
    datos["fake_quant_nodes_stripped_on_load"] = nodos_retirados
    datos["calibration_set"] = str(lista)
    datos["calibration_hash"] = huella_calibracion
    datos["calibration_image_names"] = [i.name for i in imagenes]
    datos["ptq_report"] = str(args.ptq_report)
    datos["d018_pair"] = sorted(pareja)
    datos["d018_comparison"] = comparaciones
    datos["d018_failures"] = fallos
    datos["smoke_test"] = args.smoke_test
    # Rutas relativas al serializar, porque el informe se versiona.
    informe.write_text(
        json.dumps(relativize_dump_paths(datos), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print()
    print(f"Informe detallado en: {informe}")

    print()
    print("=" * 78)
    faltan = [r for r in args.runtimes if r not in resultado.runtimes_ok]
    print(f"INT8 materializado en: {resultado.runtimes_ok or 'ninguno'}")
    if faltan:
        print(f"SIN artefacto: {faltan}")
        for runtime in faltan:
            print(f"  {runtime}: {resultado.int8_artifacts.get(runtime, {}).get('notes', '?')}")
    if fallos:
        print()
        print("Los caminos no coinciden en todos los runtimes. Las cifras de estos")
        print("artefactos NO son comparables con las de la rama PTQ hasta arreglarlo.")
    else:
        print()
        print("Caminos verificados: las dos ramas comparten configuración y calibración,")
        print("y difieren solo en los pesos de partida.")
        print()
        print("SIGUIENTE PASO: la calidad de cada artefacto la mide")
        print("11_evaluate_artifact.py y la latencia 07_benchmark.py, los dos contra")
        print(f"el experiment_id que ya existe ({args.experiment_id or 'el de la campaña QAT'}).")
    print("=" * 78)
    return 2 if (faltan or fallos) else 0


def _hash_ptq(informe: dict) -> str:
    """Hash de calibración que declara el informe de la rama PTQ, o un texto si no está."""
    for artefacto in (informe.get("int8_artifacts") or {}).values():
        huella = artefacto.get("calibration_hash")
        if huella:
            return str(huella)
    return "(no declarado)"


if __name__ == "__main__":
    sys.exit(main())
