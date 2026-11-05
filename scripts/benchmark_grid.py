#!/usr/bin/env python
"""Mide la latencia de la rejilla del apartado 5.5 llamando a 07_benchmark.py en cada entrada.

La rejilla se deduce de los manifiestos ``*_combined.json`` (que indican el podado de origen y qué
exportaciones están ok). Se salta lo que ya tiene fila y volcado, y cada llamada lleva
``--on-duplicate fail``. El código 2 del paso 07 significa medida con avisos, no fallo. Son
veinticuatro entradas: los doce INT8 combinados y los doce FP32 del podado de origen, en los dos
motores; los FP32 se inscriben bajo el experiment_id del podado.

Códigos de salida: 0 si se miden todas las entradas pendientes (o no hay ninguna), 1 si hay un error
de configuración o no se puede construir la rejilla, 2 si falla alguna entrada.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths, relocate_under_results  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402

logger = get_logger(__name__)

#: Código del paso 07 para medida con avisos.
EXIT_MEDIDO_CON_AVISOS = 2

EXITOS = (0, EXIT_MEDIDO_CON_AVISOS)

RUNTIMES = ("onnxruntime", "openvino")

#: Clave de cada motor en fp32_exports del manifiesto. No coincide con el nombre del motor: el
#: bloque FP32 se indexa por formato (onnx) y el INT8 por runtime (onnxruntime).
FP32_EXPORT_KEYS = {"onnxruntime": "onnx", "openvino": "openvino"}


@dataclass(frozen=True)
class GridEntry:
    """Medida pendiente de la rejilla. Los FP32 llevan el experiment_id del podado y los INT8 el del
    combinado.
    """

    experiment_id: str
    runtime: str
    precision: str
    artifact: Path
    origin: str

    @property
    def label(self) -> str:
        """Etiqueta corta para el log."""
        return f"{self.experiment_id} [{self.runtime}/{self.precision}]"


def resolve_under_root(recorded: str, root: Path) -> Path:
    """Reubica bajo la raíz actual una ruta absoluta grabada en un manifiesto, a partir de results/.

    Usa relocate_under_results; si la ruta no cuelga de results/, lanza TFMEdgeError.
    """
    reubicada = relocate_under_results(recorded, root / "results")
    if reubicada is None:
        raise TFMEdgeError(
            f"La ruta '{recorded}' del manifiesto no cuelga de 'results/'.",
            "Este script solo sabe reubicar artefactos dentro de 'results/'. "
            "Comprueba el manifiesto o pasa el artefacto a mano al paso 07.",
        )
    return reubicada


def _bloque(datos: dict, seccion: str, clave: str, manifiesto: Path) -> dict:
    """Bloque de una sección del manifiesto. Si la clave no existe lanza error, para no confundirlo
    con una exportación fallida.
    """
    if seccion not in datos:
        raise TFMEdgeError(
            f"El manifiesto '{manifiesto}' no tiene la sección '{seccion}'.",
            "Regenera el combinado con el paso 22, o revisa si el formato del "
            "manifiesto ha cambiado.",
        )
    if clave not in datos[seccion]:
        raise TFMEdgeError(
            f"La sección '{seccion}' de '{manifiesto}' no tiene la clave '{clave}'. Tiene: "
            f"{sorted(datos[seccion])}.",
            "Las claves de 'fp32_exports' son formatos y las de 'int8_artifacts' son runtimes. "
            "Revisa FP32_EXPORT_KEYS.",
        )
    return datos[seccion][clave]


def build_grid(exports_dir: Path, root: Path) -> list[GridEntry]:
    """Construye la rejilla a partir de los manifiestos de los combinados.

    Cada manifiesto aporta dos FP32 (bajo el podado padre) y dos INT8 (bajo el combinado), solo los
    que están en status=ok.
    """
    if not exports_dir.is_dir():
        raise TFMEdgeError(
            f"No existe el directorio de combinados '{exports_dir}'.",
            "Ejecuta antes 'scripts/22_combine.py', o apunta --exports-dir "
            "a donde estén los artefactos.",
        )

    entradas: list[GridEntry] = []
    manifiestos = sorted(exports_dir.glob("*/*_combined.json"))
    if not manifiestos:
        raise TFMEdgeError(
            f"No hay ningún manifiesto '*_combined.json' bajo '{exports_dir}'.",
            "Sin manifiesto no se sabe de qué podado viene cada artefacto.",
        )

    for manifiesto in manifiestos:
        datos = json.loads(manifiesto.read_text(encoding="utf-8"))
        combinado = manifiesto.parent.name
        padre = datos.get("parent_experiment_id", "")
        if not padre:
            raise TFMEdgeError(
                f"El manifiesto '{manifiesto}' no declara 'parent_experiment_id'.",
                "Sin el padre no se sabe bajo qué experimento inscribir la "
                "latencia del FP32. Regenera el combinado con el paso 22.",
            )

        for runtime in RUNTIMES:
            # El FP32 del directorio del combinado es el podado exportado (parent_experiment_id).
            fp32 = _bloque(datos, "fp32_exports", FP32_EXPORT_KEYS[runtime], manifiesto)
            if fp32.get("status") == "ok" and fp32.get("path"):
                entradas.append(
                    GridEntry(
                        experiment_id=padre,
                        runtime=runtime,
                        precision="fp32",
                        artifact=resolve_under_root(fp32["path"], root),
                        origin=combinado,
                    )
                )
            else:
                logger.warning(
                    "Sin FP32 de %s para %s: status=%r",
                    runtime,
                    combinado,
                    fp32.get("status", ""),
                )

            int8 = _bloque(datos, "int8_artifacts", runtime, manifiesto)
            if int8.get("status") == "ok" and int8.get("path"):
                entradas.append(
                    GridEntry(
                        experiment_id=combinado,
                        runtime=runtime,
                        precision="int8",
                        artifact=resolve_under_root(int8["path"], root),
                        origin=combinado,
                    )
                )
            else:
                logger.warning(
                    "Sin INT8 de %s para %s: status=%r",
                    runtime,
                    combinado,
                    int8.get("status", ""),
                )

    return entradas


def already_measured(
    entry: GridEntry, rows: list[dict[str, str]], device_id: str, resolution: int
) -> bool:
    """True si el registro ya tiene la fila de esta entrada (experimento, motor, precisión,
    dispositivo, lote y resolución).
    """
    for fila in rows:
        if (
            fila.get("experiment_id") == entry.experiment_id
            and fila.get("runtime") == entry.runtime
            and fila.get("precision") == entry.precision
            and fila.get("device_id") == device_id
            and str(fila.get("input_resolution", "")) == str(resolution)
        ):
            return True
    return False


def build_command(
    entry: GridEntry, args: argparse.Namespace, script: Path, output: Path
) -> list[str]:
    """Orden para llamar al paso 07 con una entrada."""
    orden = [
        sys.executable,
        str(script),
        "--profile",
        args.profile,
        "--artifact",
        str(entry.artifact),
        "--backend",
        entry.runtime,
        "--precision",
        entry.precision,
        "--experiment-id",
        entry.experiment_id,
        "--resolution",
        str(args.resolution),
        "--seed",
        str(args.seed),
        # Si el filtro anterior fallara, el paso 07 se negará a sobrescribir.
        "--on-duplicate",
        "fail",
        "--output",
        str(output),
    ]
    if args.registry_dir is not None:
        orden += ["--registry-dir", str(args.registry_dir)]
    if args.force:
        orden.append("--force")
    return orden


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos."""
    parser = argparse.ArgumentParser(
        prog="benchmark_grid.py",
        description=(
            "Recorre la rejilla de latencia del apartado 5.5 llamando a "
            "07_benchmark.py una vez por entrada. Salta lo que ya está medido "
            "y no sobrescribe nada."
        ),
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué mediría y qué saltaría, sin medir nada\n"
            "  python scripts/benchmark_grid.py --dry-run\n"
            "\n"
            "  # Campaña completa sobre el perfil de la estación de trabajo\n"
            "  python scripts/benchmark_grid.py --profile workstation_cpu\n"
            "\n"
            "  # Solo los INT8 de un combinado concreto\n"
            "  python scripts/benchmark_grid.py --only bnscale --precision int8\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="workstation_cpu",
        metavar="ID",
        help="Perfil de medida (por defecto: %(default)s).",
    )
    parser.add_argument(
        "--exports-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Directorio de combinados (por defecto: results/exports/combined).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Dónde dejar los volcados crudos (por defecto: results/raw/benchmark).",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        metavar="RUTA",
        help=(
            "Directorio del registro (por defecto: results/registry). Usa otro al hacer pruebas."
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=640, metavar="N", help="Resolución de entrada."
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        metavar="TEXTO",
        help="Mide solo las entradas cuyo experimento contenga este texto.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="",
        choices=["", "fp32", "int8"],
        help="Mide solo las entradas de esta precisión.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumera la rejilla y qué haría con cada entrada, sin medir nada.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Se lo pasa al paso 07: mide aunque la máquina esté cargada.",
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
    setup_logging("benchmark_grid", level=args.log_level)

    rutas = get_project_paths()
    exports_dir = args.exports_dir or (rutas.results / "exports" / "combined")
    output_dir = args.output_dir or (rutas.results_raw / "benchmark")
    registry_dir = args.registry_dir or (rutas.results / "registry")
    script = Path(__file__).resolve().parent / "07_benchmark.py"

    try:
        entradas = build_grid(exports_dir, rutas.root)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if args.only:
        entradas = [e for e in entradas if args.only in e.experiment_id]
    if args.precision:
        entradas = [e for e in entradas if e.precision == args.precision]

    filas = ExperimentRegistry(registry_dir).read_rows("deployment_metrics")

    print("=" * 78)
    print(f"REJILLA DE LATENCIA: perfil '{args.profile}' a {args.resolution} px")
    print("=" * 78)
    print(f"Entradas          : {len(entradas)}")
    print(f"Registro          : {registry_dir}")
    print(f"Volcados          : {output_dir}")
    print()

    pendientes: list[tuple[GridEntry, Path]] = []
    saltadas = 0
    for entrada in entradas:
        destino = output_dir / (
            f"{args.profile}_{entrada.experiment_id}_{entrada.runtime}_"
            f"{entrada.precision}_{args.resolution}.json"
        )
        tiene_fila = already_measured(entrada, filas, args.profile, args.resolution)
        tiene_volcado = destino.is_file()
        if tiene_fila and tiene_volcado:
            print(f"  SALTA   {entrada.label}  (fila y volcado ya existen)")
            saltadas += 1
            continue
        if tiene_fila or tiene_volcado:
            # Si solo está la fila o solo el volcado, se indica cuál falta.
            print(
                f"  PARCIAL {entrada.label}  "
                f"(fila={'sí' if tiene_fila else 'no'}, "
                f"volcado={'sí' if tiene_volcado else 'no'}); se mide"
            )
        else:
            print(f"  MIDE    {entrada.label}")
        if not entrada.artifact.is_file():
            print(f"          FALTA el artefacto '{entrada.artifact}'", file=sys.stderr)
            continue
        pendientes.append((entrada, destino))

    print()
    print(f"Pendientes: {len(pendientes)}   Saltadas: {saltadas}")

    if args.dry_run:
        print()
        print("--dry-run: no se ha medido ni escrito nada.")
        print("=" * 78)
        return 0

    medidas, fallidas = 0, []
    for indice, (entrada, destino) in enumerate(pendientes, start=1):
        print()
        print("-" * 78)
        print(f"[{indice}/{len(pendientes)}] {entrada.label}")
        print(f"          artefacto: {entrada.artifact}")
        orden = build_command(entrada, args, script, destino)
        logger.info("Ejecutando: %s", " ".join(orden))
        completado = subprocess.run(orden, check=False)
        # El 2 del paso 07 es medida con avisos; cualquier otro distinto de 0 es fallo.
        if completado.returncode in EXITOS:
            medidas += 1
            if completado.returncode == EXIT_MEDIDO_CON_AVISOS:
                logger.info("%s: medido con avisos (los lleva la fila).", entrada.label)
        else:
            fallidas.append((entrada.label, completado.returncode))
            logger.error("%s: el paso 07 devolvió %d.", entrada.label, completado.returncode)

    print()
    print("=" * 78)
    print(f"Medidas completadas: {medidas} de {len(pendientes)}")
    if fallidas:
        print("Entradas fallidas:")
        for etiqueta, codigo in fallidas:
            print(f"  - {etiqueta}: código {codigo}")
    print("=" * 78)
    # Si no queda nada pendiente, no es un fallo.
    return EXIT_MEDIDO_CON_AVISOS if fallidas else 0


if __name__ == "__main__":
    sys.exit(main())
