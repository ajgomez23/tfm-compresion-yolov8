#!/usr/bin/env python
"""Entrena los modelos de referencia sobre SH17.

Cada variante comprimida se compara con el baseline de su tamaño, así que la resolución, el
calendario, la partición y la semilla se fijan aquí y se guardan con cada resultado. Por defecto
entrena n, s, m y l (s es la base de las compresiones y l el maestro de la destilación), sobre
data/resized_640, que entrena unas 14 veces más rápido que data/raw. Antes de empezar calcula con
tiempos medidos si la campaña cabe en las sesiones de Colab.

Códigos de salida: 0 si se entrenan todos los modelos, 1 si hay un error de configuración o falta el
dataset, 2 si algún modelo falla (los demás quedan registrados).
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
from tfm_edge.common.paths import ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import resize  # noqa: E402
from tfm_edge.models.complexity import measure_complexity  # noqa: E402
from tfm_edge.models.detector import VALID_SIZES  # noqa: E402
from tfm_edge.models.factory import build_detector  # noqa: E402
from tfm_edge.training import schedule as sched  # noqa: E402
from tfm_edge.training.trainer import OPTIMIZADORES_ADMITIDOS, Trainer  # noqa: E402

logger = get_logger(__name__)

#: Minutos por época medidos en T4 sobre data/resized_640 (lote 16, 640 px, AMP, workers=2), solo
#: para planificar sesiones. Salen del reloj de pared de las cuatro campañas de 100 épocas, que
#: incluye la validación, el guardado en Drive y la evaluación final; el medidor de rendimiento da
#: valores entre un 31 % y un 35 % más bajos.
MEASURED_EPOCH_MINUTES_T4 = {"n": 1.169, "s": 1.942, "m": 3.958, "l": 6.128}


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="04_train_baselines.py",
        description=(
            "Entrena los modelos de referencia de SH17 (yolov8n/s/m/l) con una receta común. "
            "Guarda checkpoint en cada época y se reanuda si la sesión se corta."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # El baseline principal del estudio\n"
            "  python scripts/04_train_baselines.py --sizes s\n"
            "\n"
            "  # La familia completa de referencia\n"
            "  python scripts/04_train_baselines.py --sizes n s m l\n"
            "\n"
            "  # Comprobación rápida de que el pipeline entero funciona\n"
            "  python scripts/04_train_baselines.py --smoke-test\n"
            "\n"
            "  # Ver el plan de sesiones sin entrenar nada\n"
            "  python scripts/04_train_baselines.py --sizes n s m l --dry-run\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["s"],
        choices=list(VALID_SIZES),
        metavar="TAM",
        help="Tamaños a entrenar (por defecto: %(default)s). Válidos: n, s, m, l.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        metavar="RUTA",
        help=("YAML de dataset de Ultralytics. Por defecto, el de data/resized_640."),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de las ejecuciones (por defecto: results/runs).",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="Épocas (por defecto: %(default)s)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="Tamaño de lote (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=640,
        metavar="N",
        help=(
            "Resolución de entrada (por defecto: %(default)s). No puede superar "
            "la del dataset pre-redimensionado."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Procesos de carga de datos (por defecto: %(default)s). Con la copia "
            "redimensionada, más no ayuda."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        metavar="LR",
        help="Tasa de aprendizaje inicial (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--optimizer",
        choices=sorted(OPTIMIZADORES_ADMITIDOS),
        default="SGD",
        help=(
            "Optimizador (por defecto: %(default)s). Con 'auto', Ultralytics ignora "
            "--learning-rate y elige optimizador y tasa por su cuenta."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Pasos de acumulación de gradientes (por defecto: %(default)s). El "
            "lote efectivo es batch-size x este valor."
        ),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        metavar="N",
        help="Épocas sin mejora antes de la parada temprana (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--no-amp", action="store_true", help="Desactiva la precisión mixta automática."
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="No reanuda desde el último checkpoint aunque exista.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        metavar="DEV",
        help="Dispositivo. Vacío deja que Ultralytics elija.",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Semilla global.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Muestra el plan de sesiones y la configuración, y no entrena nada.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Campaña mínima para comprobar que el pipeline entero funciona: "
            "yolov8n, 2 épocas, lote 4, 320 px. No produce resultados del estudio."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def _resolver_dataset(args: argparse.Namespace, rutas: object) -> Path:
    """YAML de dataset que se usa; avisa si no es la copia redimensionada."""
    if args.data is not None:
        ruta = Path(args.data)
        if not ruta.is_file():
            raise TFMEdgeError(
                f"No existe el YAML de dataset '{ruta}'.",
                "Comprueba la ruta que has pasado en --data.",
            )
        return ruta

    ruta = resize.find_dataset_for_resolution(rutas.data, args.resolution)
    if "resized" not in str(ruta):
        logger.warning(
            "Vas a entrenar sobre '%s', que no es la copia redimensionada; será unas 14 veces "
            "más lento.",
            ruta,
        )
    return ruta


def _plan_de_sesiones(args: argparse.Namespace) -> None:
    """Imprime el plan de sesiones con los tiempos medidos."""
    conocidos = {
        s: MEASURED_EPOCH_MINUTES_T4[s] for s in args.sizes if s in MEASURED_EPOCH_MINUTES_T4
    }
    sin_medir = [s for s in args.sizes if s not in MEASURED_EPOCH_MINUTES_T4]
    if not conocidos:
        print("No hay tiempos medidos para los tamaños pedidos: no se puede planificar.")
        print("Mide con scripts/measure_training_throughput.py.\n")
        return

    plan = sched.plan_campaign(
        conocidos,
        epochs=args.epochs,
        source=(
            "reloj de pared de las campañas de 100 épocas en Tesla T4 sobre "
            "data/resized_640, lote 16, 640 px, AMP, workers=2"
        ),
    )
    print(sched.summarize_campaign(plan))
    if sin_medir:
        print(
            f"\nSin medir, y por tanto sin planificar: {sin_medir}. "
            "Se entrenarán igual, pero no se sabe cuánto tardarán."
        )
    print()


def entrenar_uno(size: str, args: argparse.Namespace, data_yaml: Path, salida: Path) -> dict:
    """Entrena un modelo de referencia y devuelve su resultado."""
    print("=" * 78)
    print(f"ENTRENANDO yolov8{size}")
    print("=" * 78)

    detector = build_detector(
        size,
        num_classes=17,
        input_resolution=args.resolution,
        pretrained=True,
        device="cpu",
    )
    complejidad = measure_complexity(detector, resolution=args.resolution)
    print(f"  {complejidad.params_millions} M parámetros, {complejidad.gflops} GFLOPs")

    # En campañas muy cortas (prueba de humo) el cierre del mosaico y el calentamiento se recortan
    # para que quepan, y se avisa.
    cierre = min(sched.DEFAULT_CLOSE_MOSAIC_EPOCHS, max(0, args.epochs - 1))
    calentamiento = min(sched.DEFAULT_WARMUP_EPOCHS, max(0.0, args.epochs - 1.0))
    if cierre != sched.DEFAULT_CLOSE_MOSAIC_EPOCHS or calentamiento != sched.DEFAULT_WARMUP_EPOCHS:
        logger.warning(
            "Campaña de %d épocas: se recortan el cierre de mosaico (%d) y el calentamiento "
            "(%.1f). Esta receta no es comparable con la del estudio.",
            args.epochs,
            cierre,
            calentamiento,
        )
    calendario = sched.TrainingSchedule(
        epochs=args.epochs,
        warmup_epochs=calentamiento,
        close_mosaic_epochs=cierre,
        initial_lr=args.learning_rate,
        patience=args.patience,
        checkpoint_every_epochs=1,
    )

    entrenador = Trainer(
        detector=detector,
        data_config=data_yaml,
        output_dir=salida,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        amp=not args.no_amp,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        checkpoint_every_epochs=1,
        resume_if_possible=not args.no_resume,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        seed=args.seed,
        # close_mosaic y lrf van como parámetros: el Trainer los pasa siempre y no los admite en
        # extra_args.
        close_mosaic_epochs=calendario.close_mosaic_epochs,
        final_learning_rate_factor=calendario.final_lr_fraction,
        # warmup_epochs, por el mismo motivo.
        warmup_epochs=calendario.warmup_epochs,
        extra_args={
            **({"fraction": args.fraction} if getattr(args, "fraction", None) else {}),
        },
    )

    nombre = f"{size}_baseline_default"
    resultado = entrenador.train(run_name=nombre)

    # Complejidad del modelo entrenado, junto a la configuración efectiva.
    directorio = salida / nombre
    (directorio / "complexity.json").write_text(
        json.dumps(complejidad.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directorio / "schedule.json").write_text(
        json.dumps(calendario.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "size": size,
        "run_name": nombre,
        "run_dir": str(directorio),
        "complexity": complejidad.to_dict(),
        "result": resultado.to_dict() if hasattr(resultado, "to_dict") else str(resultado),
    }


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("04_train_baselines", level=args.log_level)

    if args.smoke_test:
        args.sizes = ["n"]
        args.epochs = 2
        args.batch_size = 4
        args.resolution = 320
        args.patience = 2
        # La prueba de humo usa una fracción pequeña de train; solo comprueba que el pipeline
        # funciona.
        args.fraction = 0.02
        logger.info(
            "Modo --smoke-test: yolov8n, 2 épocas, lote 4, 320 px, 2 %% del "
            "train. NO produce resultados del estudio."
        )

    set_global_seed(args.seed)
    rutas = get_project_paths()
    salida = ensure_dir(args.output_dir or (rutas.results / "runs"))

    print("=" * 78)
    print("PASO 04: MODELOS DE REFERENCIA")
    print("=" * 78)
    print(f"Tamaños    : {args.sizes}")
    print(
        f"Épocas     : {args.epochs}   lote: {args.batch_size}   "
        f"resolución: {args.resolution} px"
    )
    print(f"workers    : {args.workers}   AMP: {not args.no_amp}   semilla: {args.seed}")
    print(f"Salida     : {salida}")
    print()

    try:
        data_yaml = _resolver_dataset(args, rutas)
    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Dataset    : {data_yaml}\n")

    _plan_de_sesiones(args)

    if args.dry_run:
        print("--dry-run: no se entrena nada.")
        return 0

    resultados: list[dict] = []
    fallidos: list[str] = []
    for size in args.sizes:
        try:
            resultados.append(entrenar_uno(size, args, data_yaml, salida))
        except TFMEdgeError as error:
            logger.error("yolov8%s ha fallado: %s", size, error)
            print(f"\nERROR entrenando yolov8{size}: {error}", file=sys.stderr)
            fallidos.append(size)
        except Exception as error:  # pragma: no cover - depende del entorno
            logger.exception("yolov8%s ha fallado de forma inesperada", size)
            print(f"\nERROR inesperado en yolov8{size}: {error}", file=sys.stderr)
            fallidos.append(size)

    resumen = salida / "baselines_summary.json"
    resumen.write_text(
        json.dumps(
            {
                "trained": resultados,
                "failed": fallidos,
                "config": vars(args) | {"data": str(data_yaml)},
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(f"Entrenados: {len(resultados)}   fallidos: {len(fallidos)}")
    if fallidos:
        print(f"Han fallado: {fallidos}")
    print(f"Resumen en: {resumen}")
    print("\nSiguiente paso:")
    print("  python scripts/05_evaluate_baselines.py")
    print("=" * 78)
    return 2 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
