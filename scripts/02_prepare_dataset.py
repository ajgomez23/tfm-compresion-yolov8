#!/usr/bin/env python
"""Valida la procedencia del dataset SH17, lo convierte y lo particiona.

Pasos: valida la procedencia con los ocho criterios de prepare.validate_provenance (aborta si
falla); comprueba sobre una muestra las dimensiones de meta-data/; convierte las anotaciones YOLO a
COCO con áreas en la resolución original; extrae la atribución de autoría; particiona en
train/val/test con estratificación multietiqueta agrupando casi duplicados; calcula estadísticas;
genera los conjuntos de calibración de 128, 256 y 512 imágenes; y guarda el subconjunto de humo como
otra partición.

Códigos de salida: 0 si el dataset queda preparado, 1 si falla la procedencia o falta el dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import ensure_dir, get_project_paths  # noqa: E402
from tfm_edge.common.seed import set_global_seed  # noqa: E402
from tfm_edge.data import calibration_set as calib  # noqa: E402
from tfm_edge.data import prepare, splits, statistics  # noqa: E402

logger = get_logger(__name__)


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="02_prepare_dataset.py",
        description=(
            "Valida la PROCEDENCIA del dataset SH17, convierte sus anotaciones a "
            "COCO, lo particiona con estratificación multietiqueta agrupando casi "
            "duplicados, calcula las estadísticas y genera los conjuntos de "
            "calibración. Aborta si el dataset de entrada no es SH17 original."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Caso por defecto: preparar el dataset completo\n"
            "  python scripts/02_prepare_dataset.py\n"
            "\n"
            "  # Prueba rápida de extremo a extremo sobre un subconjunto\n"
            "  python scripts/02_prepare_dataset.py --smoke-test\n"
            "\n"
            "  # Solo validar la procedencia, sin escribir nada\n"
            "  python scripts/02_prepare_dataset.py --validate-only\n"
            "\n"
            "  # Umbral de casi duplicados más estricto y otra semilla\n"
            "  python scripts/02_prepare_dataset.py --phash-threshold 3 --seed 7\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dataset extraído (por defecto: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Destino de los datos preparados (por defecto: data/processed).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Destino de las listas de partición (por defecto: data/splits).",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Destino de los conjuntos de calibración (por defecto: data/calibration).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla del particionado y del muestreo (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proporción de entrenamiento (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proporción de validación (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proporción de prueba (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=splits.DEFAULT_PHASH_THRESHOLD,
        metavar="N",
        help=(
            "Distancia de Hamming máxima para considerar dos imágenes casi "
            "duplicadas (por defecto: %(default)s). 0 desactiva la detección."
        ),
    )
    parser.add_argument(
        "--calibration-sizes",
        type=int,
        nargs="+",
        default=list(calib.DEFAULT_CALIBRATION_SIZES),
        metavar="N",
        help="Tamaños de los conjuntos de calibración (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--smoke-images",
        type=int,
        default=64,
        metavar="N",
        help="Tamaño del subconjunto de smoke test (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--metadata-sample",
        type=int,
        default=200,
        metavar="N",
        help=(
            "Imágenes a abrir para comprobar que las dimensiones de "
            "meta-data/ son correctas (por defecto: %(default)s)."
        ),
    )
    parser.add_argument(
        "--inference-resolution",
        type=int,
        default=640,
        metavar="N",
        help=(
            "Resolución de entrada del modelo, que determina el criterio de tamaño primario "
            "(por defecto: %(default)s). Queda registrada con cada valor calculado."
        ),
    )
    parser.add_argument(
        "--primary-criterion",
        type=str,
        default=statistics.DEFAULT_PRIMARY_CRITERION,
        choices=list(statistics.SIZE_CRITERIA),
        help=(
            "Cuál de los dos criterios de tamaño es el primario (por defecto: "
            "%(default)s). Ambos se calculan y se registran siempre."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Solo valida la procedencia y sale, sin escribir nada.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Procesa solo un subconjunto reducido, para validar el pipeline.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def main() -> int:
    """Punto de entrada del script."""
    args = construir_parser().parse_args()
    setup_logging("02_prepare_dataset", level=args.log_level)
    set_global_seed(args.seed, deterministic=True)

    rutas = get_project_paths()
    crudo = args.raw_dir or rutas.data_raw
    procesado = args.processed_dir or rutas.data_processed
    dir_splits = args.splits_dir or rutas.data_splits
    dir_calib = args.calibration_dir or rutas.data_calibration

    # La prueba de humo escribe en subdirectorios propios para no sobrescribir las particiones y la
    # calibración reales.
    if args.smoke_test:
        if args.splits_dir is None:
            dir_splits = rutas.data_splits / "smoke"
        if args.calibration_dir is None:
            dir_calib = rutas.data_calibration / "smoke"

    print("=" * 78)
    print("PASO 02: VALIDACIÓN Y PREPARACIÓN DEL DATASET")
    print("=" * 78)
    if args.smoke_test:
        print("MODO --smoke-test: subconjunto reducido.")
        print(f"  particiones y calibración van a subdirectorios propios,")
        print(f"  para no sobrescribir las del estudio: {dir_splits}")

    try:
        # 1. Procedencia
        print("\n[1/8] Validando la procedencia del dataset...")
        informe = prepare.validate_provenance(crudo)
        print(informe.summary())
        if not informe.passed:
            raise prepare.ProvenanceError(informe.failures)

        if args.validate_only:
            print("\n--validate-only: la procedencia es correcta. No se ha escrito nada.")
            return 0

        # 2 a 4. Conversión, anomalías y atribución
        print("\n[2/8] Convirtiendo anotaciones y verificando metadatos...")
        resumen = prepare.prepare_dataset(
            raw_dir=crudo,
            processed_dir=procesado,
            dataset_sha256="",
            metadata_sample=args.metadata_sample,
            smoke_test=args.smoke_test,
        )
        print(
            f"      COCO         : {resumen['num_images']:,} imágenes, "
            f"{resumen['num_annotations']:,} anotaciones"
        )
        print(f"      Anomalías    : {resumen['num_anomalies']} en {resumen['anomalies_path']}")
        print(f"      Atribuciones : {resumen['attributions_path']}")
        print(
            f"      Dimensiones de meta-data/ fiables: "
            f"{'sí' if resumen['metadata_dimensions_usable'] else 'NO (se leen las imágenes)'}"
        )

        coco = json.loads(Path(resumen["coco_path"]).read_text(encoding="utf-8"))
        clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
        etiquetas = statistics.labels_by_image(coco)
        tamanos = statistics.sizes_by_image(
            coco,
            inference_resolution=args.inference_resolution,
            criterion=args.primary_criterion,
        )
        extensiones = {
            Path(i["file_name"]).stem: Path(i["file_name"]).suffix for i in coco["images"]
        }

        # 5. Casi duplicados y particionado
        print("\n[3/8] Buscando casi duplicados con hash perceptual...")
        grupos: list[splits.DuplicateGroup] = []
        if args.phash_threshold > 0:
            hashes: dict[str, int] = {}
            for nombre, extension in extensiones.items():
                try:
                    hashes[nombre] = splits.compute_dhash(crudo / "images" / f"{nombre}{extension}")
                except OSError as error:
                    logger.warning("No se pudo hashear %s: %s", nombre, error)
            grupos = splits.find_near_duplicates(hashes, threshold=args.phash_threshold)
            agrupadas = sum(g.size for g in grupos)
            print(f"      {len(grupos)} grupos de casi duplicados, {agrupadas} imágenes implicadas")
            if grupos:
                print(
                    f"      grupo mayor: {grupos[0].size} imágenes "
                    f"(distancia máxima {grupos[0].max_distance})"
                )
                print("      NO se elimina ninguna: se agrupan en la misma partición.")
        else:
            print("      detección desactivada (--phash-threshold 0)")

        print("\n[4/8] Particionando con estratificación multietiqueta...")
        proporciones = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
        particiones = splits.stratified_multilabel_split(
            etiquetas, proporciones, groups=grupos, seed=args.seed
        )
        solapamiento = splits.verify_no_overlap(particiones)
        if solapamiento:
            raise TFMEdgeError(
                f"Hay solapamiento entre particiones: {solapamiento}",
                "Es un fallo del particionado; no continúes con estos datos.",
            )
        for nombre, elementos in particiones.items():
            print(
                f"      {nombre:6s}: {len(elementos):>6,} imágenes "
                f"({len(elementos)/len(etiquetas):.1%})"
            )
        print("      Sin solapamiento entre particiones.")

        # 6. Subconjunto de humo
        print("\n[5/8] Construyendo el subconjunto de smoke test...")
        smoke = splits.build_smoke_split(
            etiquetas, tamanos, num_images=args.smoke_images, seed=args.seed
        )
        particiones[splits.SMOKE_SPLIT_NAME] = smoke
        print(f"      {len(smoke)} imágenes, cubriendo clases y rangos de tamaño")

        rutas_split = splits.write_split_files(particiones, dir_splits, extension_map=extensiones)

        # YAML de Ultralytics que apunta a estas particiones. Va en la raíz del dataset, junto a
        # images/, para que el prefijo './' de las listas se resuelva bien (ver
        # write_ultralytics_dataset_yaml).
        ruta_yaml = prepare.write_ultralytics_dataset_yaml(
            dataset_dir=crudo,
            splits_dir=dir_splits,
            class_names=clases,
            image_extensions=extensiones,
        )
        print(f"      YAML para Ultralytics: {ruta_yaml}")

        # 7. Comparación con la partición de los autores
        print("\n[6/8] Comparando con la partición oficial de los autores...")
        comparacion: dict = {}
        fichero_train = crudo / "train_files.txt"
        fichero_val = crudo / "val_files.txt"
        if fichero_train.is_file() and fichero_val.is_file():
            comparacion = splits.compare_with_reference_split(
                {k: v for k, v in particiones.items() if k in splits.SPLIT_NAMES},
                fichero_train.read_text(encoding="utf-8").splitlines(),
                fichero_val.read_text(encoding="utf-8").splitlines(),
            )
            print(f"      oficial: {comparacion['reference_sizes']}")
            for clave, valor in comparacion["agreement"].items():
                print(
                    f"      {clave}: Jaccard {valor['jaccard']:.3f}, "
                    f"{valor['fraction_of_official']:.1%} de la oficial"
                )
        else:
            print("      no se han encontrado train_files.txt / val_files.txt")

        informe_split = splits.SplitReport(
            counts={k: len(v) for k, v in particiones.items()},
            duplicate_groups=grupos,
            overlap=solapamiento,
            seed=args.seed,
            phash_threshold=args.phash_threshold,
        )
        for nombre, elementos in particiones.items():
            conteo: dict[str, int] = {}
            for imagen in elementos:
                for indice in etiquetas.get(imagen, set()):
                    conteo[clases[indice]] = conteo.get(clases[indice], 0) + 1
            informe_split.class_distribution[nombre] = conteo
        ruta_informe = splits.write_split_report(
            informe_split, Path(dir_splits) / "split_report.json"
        )
        if comparacion:
            (Path(dir_splits) / "reference_split_comparison.json").write_text(
                json.dumps(comparacion, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # 8. Estadísticas
        print("\n[7/8] Calculando estadísticas...")
        estadisticas = statistics.compute_statistics(
            coco,
            clases,
            inference_resolution=args.inference_resolution,
            primary_criterion=args.primary_criterion,
        )
        estadisticas.to_json(Path(procesado) / "dataset_statistics.json")
        estadisticas.to_csv(ensure_dir(procesado))
        for criterio in statistics.SIZE_CRITERIA:
            reparto = estadisticas.payload["size_distribution"][criterio]["fractions"]
            marca = "PRIMARIO  " if criterio == args.primary_criterion else "secundario"
            print(
                f"      [{marca}] {criterio:14s} "
                + "  ".join(f"{r}={reparto[r]:.1%}" for r in statistics.SIZE_RANGES)
            )
        print(
            f"      resoluciones distintas: "
            f"{estadisticas.payload['summary']['num_distinct_resolutions']}"
        )

        # 9. Calibración
        print("\n[8/8] Generando conjuntos de calibración...")
        disponibles = len(particiones["train"])
        tamanos_calib = [t for t in args.calibration_sizes if t <= disponibles]
        if len(tamanos_calib) < len(args.calibration_sizes):
            print(
                f"      aviso: se omiten los tamaños mayores que la partición de "
                f"entrenamiento ({disponibles} imágenes)"
            )
        if not tamanos_calib:
            # Con --smoke-test la partición es menor que el tamaño de calibración más pequeño; se
            # genera un solo conjunto del tamaño que quepa.
            tamanos_calib = [max(1, min(disponibles, min(args.calibration_sizes)))]
            print(
                f"      ningún tamaño pedido cabe en la partición; se genera uno "
                f"de {tamanos_calib[0]} imágenes para poder recorrer el pipeline"
            )
        conjuntos = calib.build_calibration_sets(
            particiones["train"],
            etiquetas,
            tamanos,
            clases,
            sizes=tamanos_calib,
            seed=args.seed,
            source_split="train",
        )
        calib.write_calibration_sets(conjuntos, dir_calib)

    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interacción del usuario
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("Dataset preparado.")
    print(f"  anotaciones COCO : {resumen['coco_path']}")
    print(f"  manifiesto       : {resumen['manifest_path']}")
    print(f"  particiones      : {dir_splits}")
    print(f"  informe de split : {ruta_informe}")
    print(f"  calibración      : {dir_calib}")
    print("\nSiguiente paso: python scripts/03_dataset_eda.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
