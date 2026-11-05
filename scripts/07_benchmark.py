#!/usr/bin/env python
"""Mide latencia, memoria, rendimiento y tamaño de artefacto sobre un perfil con el protocolo común.

Todas las variantes se miden igual: mismo calentamiento, mismas iteraciones y mismas imágenes en el
mismo orden. Se rechaza medir sobre un perfil no válido (Colab, SageMaker), con la máquina cargada
salvo con ``--force`` (que queda registrado) y sobre una sola imagen, porque el postprocesado
depende del número de detecciones.

Se puede medir un checkpoint de PyTorch (``--checkpoint``) o un artefacto exportado
(``--artifact``), con el mismo preprocesado y postprocesado de ``InferenceBackend`` en los dos
casos. La fila de deployment_metrics solo se escribe si se indica ``--experiment-id``.

Códigos de salida: 0 si la medida se completa, 1 si hay un error de configuración o el perfil no
permite medir, 2 si la medida termina con avisos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.benchmark import profiles, protocol  # noqa: E402
from tfm_edge.benchmark.backends import build_backend  # noqa: E402
from tfm_edge.common.environment import capture_environment, persist_environment  # noqa: E402
from tfm_edge.common.evidence import (  # noqa: E402
    add_overwrite_dump_argument,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import VALID_DUPLICATE_POLICIES, ExperimentRegistry  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.models.detector import Detector  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="07_benchmark.py",
        description=(
            "Mide latencia, memoria, rendimiento y tamaño de artefacto con el protocolo común. "
            "Rechaza los perfiles que no dan latencias comparables."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué perfiles hay y cuáles pueden medir\n"
            "  python scripts/07_benchmark.py --list-profiles\n"
            "\n"
            "  # Medir el baseline sobre CPU\n"
            "  python scripts/07_benchmark.py --profile workstation_cpu \\\n"
            "      --checkpoint results/runs/s_baseline_default/weights/best.pt\n"
            "\n"
            "  # Comprobación rápida de que el protocolo funciona\n"
            "  python scripts/07_benchmark.py --profile workstation_cpu \\\n"
            "      --checkpoint ... --smoke-test\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        metavar="ID",
        help="Perfil de medida, por su device_id.",
    )
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de perfiles (por defecto: configs/benchmark).",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Muestra la tabla de perfiles con su estado, y termina.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, metavar="RUTA", help="Pesos a medir."
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Artefacto exportado a medir (ONNX, XML de OpenVINO...) en lugar de un checkpoint "
            "de PyTorch."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help=(
            "Precisión del artefacto medido (por defecto: %(default)s). Se escribe tal cual en "
            "la columna 'precision' del registro."
        ),
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Experimento al que pertenece la medida. Sin él no se escribe la fila de "
            "deployment_metrics."
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de imágenes reales (por defecto: las de la copia del dataset).",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=16,
        metavar="N",
        help=(
            "Imágenes reales sobre las que rotar (por defecto: %(default)s). "
            "Nunca una sola: el postprocesado cuesta según las detecciones."
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de entrada."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        metavar="NOMBRE",
        help=(
            "Motor de inferencia (por defecto: %(default)s). Con 'onnxruntime' u 'openvino' "
            "hace falta --artifact; para OpenVINO se pasa el .xml y el tamaño se mide sobre su "
            "carpeta, que incluye el .bin."
        ),
    )
    parser.add_argument(
        "--warmup", type=int, default=None, metavar="N", help="Iteraciones de calentamiento."
    )
    parser.add_argument(
        "--iterations", type=int, default=None, metavar="N", help="Iteraciones de medida."
    )
    parser.add_argument(
        "--repetitions", type=int, default=None, metavar="N", help="Repeticiones del bloque."
    )
    parser.add_argument(
        "--no-throughput", action="store_true", help="Omite el barrido de rendimiento."
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="",
        metavar="TEXTO",
        help=(
            "Condición de medida declarada de la campaña (por ejemplo 'encadenada con "
            "presaturacion'). Se guarda en el volcado y en la columna 'notes'. No cuenta como "
            "aviso."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Mide aunque la máquina esté cargada. Queda registrado: los "
            "percentiles no serán comparables con los de otras medidas."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="RUTA",
        help="JSON de salida (por defecto: results/raw/benchmark/<perfil>_<modelo>.json).",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Directorio del registro donde inscribir el entorno (por defecto: "
            "results/registry). Usa otro al hacer pruebas."
        ),
    )
    parser.add_argument(
        "--on-duplicate",
        type=str,
        default="overwrite",
        choices=list(VALID_DUPLICATE_POLICIES),
        help="Qué hacer si la fila ya existe (por defecto: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Campaña mínima (5 calentamiento, 20 medidas, 1 repetición). No es un resultado.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log.",
    )
    add_overwrite_dump_argument(parser, "el volcado de latencias")
    return parser


def componer_notas(regime: str, avisos: list[str]) -> str:
    """Compone la columna ``notes`` con el régimen declarado delante de los avisos."""
    partes = ([f"régimen declarado: {regime}"] if regime else []) + list(avisos)
    return "; ".join(partes)


def cargar_imagenes(directorio: Path, cuantas: int) -> list:
    """Carga imágenes reales sobre las que medir. Lanza TFMEdgeError si no hay suficientes."""
    import cv2

    rutas = sorted(
        p for p in directorio.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:cuantas]
    if len(rutas) < 2:
        raise TFMEdgeError(
            f"Solo se han encontrado {len(rutas)} imágenes en '{directorio}'.",
            "Hacen falta al menos 2, porque la latencia del postprocesado depende del número de "
            "detecciones.",
        )
    imagenes = [cv2.imread(str(r)) for r in rutas]
    if any(i is None for i in imagenes):
        raise TFMEdgeError(
            "Alguna imagen no se ha podido leer.", "Comprueba que no están corruptas."
        )
    return imagenes


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("07_benchmark", level=args.log_level)
    set_global_seed(args.seed)

    rutas = get_project_paths()
    dir_perfiles = args.profiles_dir or (rutas.root / "configs" / "benchmark")

    try:
        todos = profiles.load_all_profiles(dir_perfiles)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.list_profiles:
        print("PERFILES DE MEDIDA")
        print()
        print(profiles.describe_profiles(todos))
        print()
        print("'mide' = valid_for_measurement. Un perfil con 'NO' se RECHAZA con")
        print("error si se intenta medir sobre él: Colab y SageMaker entrenan, el")
        print("hardware físico mide.")
        print()
        print("'repr.' = latency_representative. Un 'NO' significa que la latencia")
        print("se mide de verdad, pero NO representa al dispositivo objetivo, y se")
        print("excluye de las fronteras de Pareto de latencia.")
        return 0

    if not args.profile:
        print("ERROR: hace falta --profile. Usa --list-profiles para verlos.", file=sys.stderr)
        return 1
    if args.profile not in todos:
        print(
            f"ERROR: no existe el perfil '{args.profile}'. " f"Disponibles: {sorted(todos)}",
            file=sys.stderr,
        )
        return 1
    if bool(args.checkpoint) == bool(args.artifact):
        print(
            "ERROR: hace falta --checkpoint O --artifact, exactamente uno.",
            file=sys.stderr,
        )
        return 1
    medido = args.checkpoint or args.artifact
    if not medido.exists():
        print(f"ERROR: no existe '{medido}'.", file=sys.stderr)
        return 1
    if args.checkpoint and args.backend != "pytorch":
        print(
            f"ERROR: --checkpoint carga un modelo de PyTorch y el motor pedido es "
            f"'{args.backend}'. Para medir otro motor usa --artifact con el "
            f"fichero exportado.",
            file=sys.stderr,
        )
        return 1

    perfil = todos[args.profile]
    # El destino se comprueba antes de medir, que lleva minutos.
    etiqueta = args.experiment_id or medido.stem
    destino = args.output or (
        rutas.results_raw
        / "benchmark"
        / f"{perfil.device_id}_{etiqueta}_{args.backend}_{args.precision}_"
        f"{args.resolution}.json"
    )
    try:
        refuse_to_overwrite_dump(
            destino, overwrite=args.overwrite_dump, what="el volcado de latencias"
        )
    except FileExistsError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("=" * 78)
    print(f"PASO 07: MEDIDA SOBRE '{perfil.device_id}'")
    print("=" * 78)
    print(f"{perfil.description.strip()}")
    print()

    # Se rechaza antes de cargar nada.
    try:
        perfil.assert_can_measure()
    except TFMEdgeError as error:
        print(f"{error}", file=sys.stderr)
        # Se registra status=no_disponible. Con skip y no overwrite, para no sustituir una fila
        # existente cuyo motivo trae más información que el del YAML del perfil.
        try:
            registro = ExperimentRegistry(args.registry_dir or (rutas.results / "registry"))
            escrita = registro.append_deployment_metrics(
                perfil.unavailable_coverage_row(), on_duplicate="skip"
            )
        except (TFMEdgeError, ValueError) as fallo:
            print(f"AVISO: no se ha podido inscribir la cobertura: {fallo}", file=sys.stderr)
        else:
            if escrita:
                print(
                    f"Inscrita la cobertura de '{perfil.device_id}' con " f"status=no_disponible.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"La cobertura de '{perfil.device_id}' ya estaba inscrita.",
                    file=sys.stderr,
                )
        # Devuelve 1 aunque quede registrado, porque no se ha medido nada.
        return 1

    calentamiento = args.warmup if args.warmup is not None else perfil.warmup_iterations
    iteraciones = args.iterations if args.iterations is not None else perfil.measure_iterations
    repeticiones = args.repetitions if args.repetitions is not None else perfil.repetitions
    if args.smoke_test:
        calentamiento, iteraciones, repeticiones = 5, 20, 1
        logger.info("Modo --smoke-test: 5/20/1. NO es un resultado del estudio.")

    try:
        carga = protocol.check_system_load(force=args.force)
        print(
            f"Carga del sistema: {carga.get('load_per_core', 'n/d')} por núcleo "
            f"({'aceptable' if carga.get('acceptable', True) else 'ALTA, forzada'})"
        )

        cpu = protocol.configure_cpu(
            num_threads=perfil.num_threads, affinity=perfil.cpu_affinity or None
        )
        dispositivo = "cuda" if perfil.device_type == "cuda" else "cpu"

        imagenes_dir = args.images_dir or (rutas.data / f"resized_{args.resolution}" / "images")
        if not imagenes_dir.is_dir():
            imagenes_dir = rutas.data_raw / "images"
        imagenes = cargar_imagenes(imagenes_dir, args.num_images)
        print(f"Imágenes reales  : {len(imagenes)} de {imagenes_dir}")

        # Un checkpoint se carga como modelo y un artefacto lo carga el motor desde su ruta. El pre
        # y el postprocesado son los mismos.
        if args.checkpoint:
            cargable = Detector.load(args.checkpoint, input_resolution=args.resolution)
        else:
            cargable = args.artifact
        backend = build_backend(args.backend, args.resolution, dispositivo)
        backend.load(cargable)
        print(f"Motor            : {args.backend} sobre {dispositivo}")
        print(f"Precisión        : {args.precision}")
        print(
            f"Protocolo        : {calentamiento} calentamiento + {iteraciones} x "
            f"{repeticiones} repeticiones"
        )
        print()

        medidas = protocol.measure_latency(
            backend,
            imagenes,
            warmup_iterations=calentamiento,
            measure_iterations=iteraciones,
            repetitions=repeticiones,
            with_temperature=perfil.requires_temperature,
        )
        agregado, entre = protocol.aggregate_latency(medidas)

        entorno = capture_environment(device_id=perfil.device_id, repo_dir=rutas.root)
        resultado = protocol.BenchmarkResult(
            device_id=perfil.device_id,
            backend=args.backend,
            precision=args.precision,
            input_resolution=args.resolution,
            num_threads=cpu["num_threads"],
            cpu_affinity=cpu["cpu_affinity"],
            warmup_iterations=calentamiento,
            measure_iterations=iteraciones,
            repetitions=repeticiones,
            latency=medidas,
            aggregated=agregado,
            between_run_std_ms=entre,
            system_load=carga,
            environment_id=persist_environment(entorno, registry_dir=args.registry_dir),
            started_at=medidas[0].started_at,
            ended_at=medidas[-1].ended_at,
        )
        resultado.memory = protocol.measure_memory(backend, imagenes)
        if not args.no_throughput:
            resultado.throughput = protocol.measure_throughput(
                backend,
                imagenes,
                batch_sizes=perfil.throughput_batch_sizes if not args.smoke_test else [1, 2],
                window_s=protocol.DEFAULT_THROUGHPUT_WINDOW_S if not args.smoke_test else 2.0,
            )
        # El IR de OpenVINO son dos ficheros (.xml con el grafo y .bin con los pesos), así que se
        # mide el directorio.
        a_medir = medido.parent if medido.suffix == ".xml" else medido
        resultado.artifact = protocol.measure_artifact_size(a_medir)

        avisos = []
        if args.smoke_test:
            avisos.append("PRUEBA DE HUMO: no es un resultado del estudio")
        if not carga.get("acceptable", True):
            avisos.append("medida con carga alta y --force: percentiles no comparables")
        if not perfil.latency_representative:
            avisos.append(
                "latency_representative=false: la latencia NO representa al "
                "dispositivo objetivo y se excluye de las fronteras de Pareto"
            )
        if any(m.statistics.get("postprocess_representative") == 0 for m in medidas):
            avisos.append("el postprocesado no produjo detecciones: su coste no es representativo")
        # El régimen no es un aviso y no afecta al código de salida; va al principio de notes.
        resultado.measurement_regime = args.regime
        resultado.notes = componer_notas(args.regime, avisos)

        print()
        print(resultado.summary())
        if avisos:
            print()
            print("AVISOS:")
            for aviso in avisos:
                print(f"  - {aviso}")

        protocol.save_benchmark_result(resultado, destino)
        print()
        print(f"Resultado completo (con el vector de tiempos) en: {destino}")

        # La fila solo se escribe si se conoce el experimento.
        if not args.experiment_id:
            print()
            print("Sin --experiment-id: no se escribe fila en deployment_metrics.")
            print("La medida queda en el JSON, que es de donde se puede reinscribir.")
        else:
            fila = protocol.to_deployment_row(
                resultado,
                experiment_id=args.experiment_id,
                raw_timings_path=destino,
                artifact_path=medido,
            )
            # El perfil declara si su latencia representa al dispositivo objetivo; esa columna
            # decide qué entra en las fronteras de Pareto de latencia.
            fila["latency_representative"] = str(perfil.latency_representative).lower()
            registro = ExperimentRegistry(args.registry_dir or (rutas.results / "registry"))
            registro.append_deployment_metrics(fila, on_duplicate=args.on_duplicate)
            print(f"Inscrito en deployment_metrics como {args.experiment_id}")
            print()
            print(registro.validate().summary())

    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    print("=" * 78)
    return 2 if avisos else 0


if __name__ == "__main__":
    sys.exit(main())
