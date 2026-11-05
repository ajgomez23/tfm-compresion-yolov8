#!/usr/bin/env python
"""Coloca los checkpoints podados donde indica el volcado que se escribieron.

Los checkpoints de reajuste se generan en Colab y se descargan a mano, a veces con otra estructura
de directorios. La estructura correcta se lee de finetune.checkpoint_before y checkpoint_after del
volcado de cada experimento (la parte posterior al directorio del experimento). Hoy es:

models/pruned/<experiment_id>/finetune/pruned_before_finetune.pt
models/pruned/<experiment_id>/finetune/train/weights/best.pt

No escribe en results/. Por defecto solo muestra el plan (``--apply`` para mover). Comprueba el
SHA-256 antes y después, nunca sobrescribe ni borra, y no mueve candidatos ambiguos.

Códigos de salida: 0 si todo está en su sitio o se muestra el plan, 1 si hay un error de uso, 2 si
hay incidencias.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402

logger = get_logger(__name__)

#: Claves del volcado de las que se deduce la estructura.
CHECKPOINT_KEYS: tuple[tuple[str, str], ...] = (
    ("checkpoint_before", "podado sin reajustar"),
    ("checkpoint_after", "reajustado"),
)


@dataclass
class Movimiento:
    """Fichero que hay que mover a su sitio."""

    experiment_id: str
    etiqueta: str
    origen: Path
    destino: Path
    digest: str
    tamano: int


def sha256(ruta: Path, bloque: int = 1 << 20) -> str:
    """SHA-256 de un fichero."""
    resumen = hashlib.sha256()
    with ruta.open("rb") as manejador:
        for trozo in iter(lambda: manejador.read(bloque), b""):
            resumen.update(trozo)
    return resumen.hexdigest()


def relative_to_experiment(ruta_declarada: str, experiment_id: str) -> Path | None:
    """Parte de una ruta declarada posterior al directorio del experimento, o None."""
    if not ruta_declarada:
        return None
    partes = Path(ruta_declarada).parts
    if experiment_id not in partes:
        return None
    corte = len(partes) - 1 - partes[::-1].index(experiment_id)
    resto = partes[corte + 1 :]
    return Path(*resto) if resto else None


def buscar_candidato(directorio: Path, nombre: str) -> tuple[Path | None, list[Path]]:
    """Busca un fichero por nombre bajo el directorio de un experimento.

    Devuelve el único candidato (o None) y todas las coincidencias, para distinguir "no está" de
    "está duplicado".
    """
    coincidencias = sorted(p for p in directorio.rglob(nombre) if p.is_file())
    return (coincidencias[0] if len(coincidencias) == 1 else None), coincidencias


def construir_plan(args: argparse.Namespace, rutas) -> tuple[list[Movimiento], list[str]]:
    """Recorre los experimentos y devuelve los movimientos necesarios y las incidencias."""
    if not args.pruned_root.is_dir():
        raise TFMEdgeError(
            f"No existe '{args.pruned_root}'. Es el directorio bajo el que cuelgan "
            "los modelos podados, uno por experiment_id."
        )

    movimientos: list[Movimiento] = []
    incidencias: list[str] = []

    experimentos = sorted(p for p in args.pruned_root.iterdir() if p.is_dir())
    if args.experiment_id:
        pedidos = set(args.experiment_id)
        desconocidos = pedidos - {p.name for p in experimentos}
        if desconocidos:
            raise TFMEdgeError(
                f"No hay directorio para: {', '.join(sorted(desconocidos))} bajo "
                f"'{args.pruned_root}'."
            )
        experimentos = [p for p in experimentos if p.name in pedidos]

    for directorio in experimentos:
        eid = directorio.name
        volcado = args.raw_dir / f"{eid}.json"
        if not volcado.is_file():
            incidencias.append(
                f"[{eid}] no hay volcado en '{volcado}'. Sin él no se sabe dónde "
                "escribió el pipeline; no se toca nada."
            )
            continue

        with volcado.open("r", encoding="utf-8") as manejador:
            datos = json.load(manejador)
        reajuste = datos.get("finetune", {})

        for clave, etiqueta in CHECKPOINT_KEYS:
            relativo = relative_to_experiment(reajuste.get(clave, ""), eid)
            if relativo is None:
                # checkpoint_after vacío significa que la campaña no tuvo reajuste.
                continue

            destino = directorio / relativo
            if destino.is_file():
                logger.debug("[%s] %s ya está en su sitio.", eid, etiqueta)
                continue

            candidato, coincidencias = buscar_candidato(directorio, relativo.name)
            if not coincidencias:
                incidencias.append(
                    f"[{eid}] falta el checkpoint '{etiqueta}' ({relativo.name}). "
                    f"El volcado dice que se escribió; tráelo de Drive a "
                    f"'{destino.relative_to(args.pruned_root.parent.parent)}'."
                )
                continue
            if candidato is None:
                lista = ", ".join(str(p.relative_to(directorio)) for p in coincidencias)
                incidencias.append(
                    f"[{eid}] hay {len(coincidencias)} ficheros llamados "
                    f"'{relativo.name}' ({lista}). Ambiguo: no se mueve ninguno."
                )
                continue

            movimientos.append(
                Movimiento(
                    experiment_id=eid,
                    etiqueta=etiqueta,
                    origen=candidato,
                    destino=destino,
                    digest=sha256(candidato),
                    tamano=candidato.stat().st_size,
                )
            )

    return movimientos, incidencias


def aplicar(movimientos: list[Movimiento]) -> list[str]:
    """Mueve los ficheros comprobando el SHA-256 antes y después; devuelve los fallos."""
    fallos: list[str] = []
    for m in movimientos:
        if m.destino.exists():
            fallos.append(f"[{m.experiment_id}] '{m.destino}' apareció mientras tanto; no se pisa.")
            continue
        m.destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(m.origen), str(m.destino))

        digest_final = sha256(m.destino)
        if digest_final != m.digest:
            fallos.append(
                f"[{m.experiment_id}] '{m.destino.name}' NO llegó idéntico: "
                f"{digest_final[:16]}… frente a {m.digest[:16]}…"
            )
            continue
        logger.info("[%s] %s colocado en %s", m.experiment_id, m.etiqueta, m.destino)
    return fallos


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y analiza los argumentos de la CLI."""
    rutas = get_project_paths()
    analizador = argparse.ArgumentParser(
        description=(
            "Coloca los checkpoints podados en la estructura que declara su propio "
            "volcado crudo. Por defecto solo enseña el plan."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué haría (no mueve nada)\n"
            "  python scripts/15_normalize_pruned_layout.py\n\n"
            "  # Aplicarlo\n"
            "  python scripts/15_normalize_pruned_layout.py --apply\n\n"
            "  # Solo un experimento\n"
            "  python scripts/15_normalize_pruned_layout.py --experiment-id "
            "s_pruning_l1global_r25_4e91ddbc --apply\n"
        ),
    )
    analizador.add_argument(
        "--pruned-root",
        type=Path,
        default=rutas.root / "models" / "pruned",
        help="Directorio de los modelos podados (por defecto: %(default)s).",
    )
    analizador.add_argument(
        "--raw-dir",
        type=Path,
        default=rutas.results_raw / "pruning",
        help="Directorio de los volcados crudos (por defecto: %(default)s).",
    )
    analizador.add_argument(
        "--experiment-id",
        nargs="+",
        default=None,
        help="Limitar a estos experimentos. Por defecto, todos los que haya en disco.",
    )
    analizador.add_argument(
        "--apply",
        action="store_true",
        help="Ejecuta los movimientos. Sin esta bandera no se mueve ningún fichero.",
    )
    analizador.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return analizador.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada."""
    args = parse_args(argv)
    setup_logging("15_normalize_pruned_layout", level=args.log_level)
    rutas = get_project_paths()

    try:
        movimientos, incidencias = construir_plan(args, rutas)
    except TFMEdgeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("\n=== Plan de colocación ===")
    print(f"  modelos  : {args.pruned_root}")
    print(f"  volcados : {args.raw_dir}")
    if not movimientos:
        print("\n  Nada que mover: todo está donde su volcado dice.")
    for m in movimientos:
        print(f"\n  [{m.experiment_id}] {m.etiqueta}  ({m.tamano / 1e6:.2f} MB)")
        print(f"      de : {m.origen.relative_to(args.pruned_root)}")
        print(f"      a  : {m.destino.relative_to(args.pruned_root)}")
        print(f"      sha: {m.digest[:32]}…")

    if incidencias:
        print(f"\n=== Incidencias ({len(incidencias)}) ===")
        for texto in incidencias:
            print(f"  - {texto}")

    if not args.apply:
        if movimientos:
            print(
                f"\nNo se ha movido nada. Son {len(movimientos)} fichero(s); "
                "vuelve a lanzarlo con --apply para aplicarlo."
            )
        return 2 if incidencias else 0

    fallos = aplicar(movimientos)
    if fallos:
        print(f"\n=== Fallos al mover ({len(fallos)}) ===", file=sys.stderr)
        for texto in fallos:
            print(f"  - {texto}", file=sys.stderr)
        return 2

    if movimientos:
        print(f"\nColocados {len(movimientos)} fichero(s), con el SHA-256 verificado tras mover.")
        logger.info("Colocados %d ficheros.", len(movimientos))
    else:
        print("\nNo ha hecho falta mover nada.")
    return 2 if incidencias else 0


if __name__ == "__main__":
    sys.exit(main())
