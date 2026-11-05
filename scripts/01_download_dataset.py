#!/usr/bin/env python
"""Verifica la integridad del dataset SH17 y lo descomprime.

Kaggle exige cuenta y credenciales, así que lo normal es verificar un archivo descargado a mano. La
descarga por la API de Kaggle es opcional. Si el archivo ya está verificado y descomprimido no se
repite el trabajo salvo con ``--force``.

Códigos de salida: 0 si el dataset está verificado y disponible, 1 si falta el archivo o falla la
integridad o la descompresión.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.data.download import (  # noqa: E402
    INTEGRITY_MANIFEST,
    SH17_KAGGLE_DATASET,
    SH17_LICENSE,
    DatasetNotAvailableError,
    download_via_kaggle_api,
    extract_archive,
    find_archive,
    verify_archive,
)

logger = get_logger(__name__)

#: SHA-256 del paquete oficial, para comprobar que se trabaja con los mismos datos.
SH17_CANONICAL_SHA256 = "4747f51cac891a59a55c354a7b0f3c3addb4478ab214e74c16f26a6a205abf73"


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="01_download_dataset.py",
        description=(
            "Verifica la integridad del archivo del dataset SH17 y lo descomprime "
            "en data/raw. SH17 se descarga manualmente desde Kaggle: este script "
            "NO lo descarga por ti salvo que uses --use-kaggle-api con credenciales "
            f"ya configuradas. Licencia del dataset: {SH17_LICENSE} (no comercial)."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Caso por defecto: verificar y descomprimir lo que hay en data/raw\n"
            "  python scripts/01_download_dataset.py\n"
            "\n"
            "  # Archivo en otra ubicación, sin descomprimir todavía\n"
            "  python scripts/01_download_dataset.py --archive /mnt/d/sh17.zip --no-extract\n"
            "\n"
            "  # Registrar el checksum de una versión nueva del dataset\n"
            "  python scripts/01_download_dataset.py --record-checksum\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Archivo comprimido del dataset. Por defecto se busca el mayor .zip o "
            ".tar* dentro de data/raw."
        ),
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio donde descomprimir (por defecto: data/raw).",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        metavar="HASH",
        help=(
            "Checksum esperado. Por defecto se usa el canónico registrado en el "
            "propio script, verificado contra el paquete oficial de Kaggle."
        ),
    )
    parser.add_argument(
        "--record-checksum",
        action="store_true",
        help=(
            "Registra el checksum del archivo presente como nueva referencia en lugar de "
            "comparar con el canónico. Úsalo solo si el dataset ha cambiado en origen."
        ),
    )
    parser.add_argument(
        "--use-kaggle-api",
        action="store_true",
        help=(
            "Descarga el dataset con la API de Kaggle. Requiere el paquete 'kaggle' "
            "y credenciales ya configuradas por ti; el proyecto no las gestiona."
        ),
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Solo verifica la integridad, sin descomprimir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Descomprime aunque el destino ya tenga contenido.",
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
    setup_logging("01_download_dataset", level=args.log_level)
    rutas = get_project_paths()
    destino = args.dest or rutas.data_raw

    print("=" * 78)
    print("PASO 01: VERIFICACIÓN E INSTALACIÓN DEL DATASET SH17")
    print("=" * 78)
    print(f"Licencia del dataset: {SH17_LICENSE}  (cláusula NO COMERCIAL)")
    print(f"Fuente oficial      : kaggle:{SH17_KAGGLE_DATASET}")
    print()

    try:
        if args.use_kaggle_api:
            download_via_kaggle_api(destino)

        archivo = args.archive or find_archive(destino)
        if archivo is None:
            raise DatasetNotAvailableError(destino / "sh17.zip")

        esperado = None if args.record_checksum else (args.expected_sha256 or SH17_CANONICAL_SHA256)
        print(f"Archivo             : {archivo}")
        print(f"Tamaño              : {archivo.stat().st_size / 2**30:.2f} GiB")
        print("Calculando SHA-256 (puede tardar varios minutos)...")

        registro = verify_archive(
            archivo,
            expected_sha256=esperado,
            manifest_path=destino / INTEGRITY_MANIFEST,
        )
        print(f"SHA-256             : {registro.sha256}")
        if esperado:
            print("Integridad          : COINCIDE con el checksum canónico")
        else:
            print("Integridad          : registrada como NUEVA referencia (no verificada)")

        if not args.no_extract:
            print(f"\nDescomprimiendo en {destino} ...")
            extract_archive(archivo, destino, force=args.force)
            print("Descompresión terminada.")

    except TFMEdgeError as error:
        logger.error("%s", error)
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interacción del usuario
        print("\nInterrumpido por el usuario.", file=sys.stderr)
        return 1

    print("\n" + "=" * 78)
    print("Dataset verificado y disponible.")
    print("Siguiente paso: python scripts/02_prepare_dataset.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
