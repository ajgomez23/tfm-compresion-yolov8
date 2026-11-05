#!/usr/bin/env python
"""Incorpora al registro local los resultados medidos en Colab.

Hay tres disposiciones de origen. ``--layout pruning`` (por defecto) trae un volcado crudo
``{experiment_id}.json`` por experimento y lo copia a ``results/raw/pruning/``. ``--layout
baselines`` no trae volcados: la evidencia (``complexity.json``, ``schedule.json`` y
``training_cost.json``) llega de Drive junto al checkpoint, en
``models/baselines/{size}_baseline_default/``, y solo se inscriben las filas que falten. ``--layout
distillation-50ep`` lee ``registry/`` y ``raw_epochs50/`` de la carpeta
``resultados_distillation_50ep/``, copia los volcados a ``results/raw/distillation/epochs50/`` y
comprueba además que volcado, fila y checkpoint son del mismo experimento y de 50 épocas
(:func:`comprobar_identidad`), porque todos los alumnos ``n`` tienen los mismos parámetros.

Una fila ya inscrita no se reescribe: se compara columna a columna con la del origen. Si coincide se
anota como confirmada; si solo difiere en rutas, ``timestamp`` o ``notes`` se muestra en el plan; si
difiere en otra columna no se importa nada. Lo mismo vale para filas de calidad, de entorno y
volcados crudos (estos se comparan byte a byte).

Los volcados crudos se copian tal cual, con sus rutas ``/content/...``. En las filas del registro se
reescriben las columnas de ruta, y ``weights_path`` apunta al ejemplar local solo si existe y
coincide en número de parámetros.

Antes de escribir se comprueba que el volcado de destino no existe, que la fila no está ya inscrita,
que coinciden los parámetros del volcado, de ``effective_params`` y del checkpoint local, que el
``environment_id`` de las filas de calidad existe o viene en el origen y, con
``identidad_estricta``, la identidad del experimento y las épocas. Si algo falla no se escribe nada.
Sin ``--apply`` solo se escribe el log en ``results/logs/``; ``--dry-run`` se acepta pero sobra.

Códigos de salida: 0 si la importación se completa con --apply o se muestra el plan sin él, 2 si
falla alguna comprobación de integridad o hay un error de uso; en ambos casos no se escribe nada.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from tfm_edge.common.config import load_class_names  # noqa: E402
from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths, parse_experiment_id  # noqa: E402
from tfm_edge.common.registry import (  # noqa: E402
    TABLE_PRIMARY_KEYS,
    ExperimentRegistry,
    normalize_value,
    utc_timestamp,
)

logger = get_logger(__name__)

#: Columnas de cada tabla que contienen rutas del sistema de ficheros de origen.
PATH_COLUMNS: dict[str, tuple[str, ...]] = {
    "experiments": ("config_path", "weights_path"),
    "quality_metrics": ("predictions_path",),
}


#: Dónde declara cada técnica el número de parámetros dentro de su volcado crudo, la primera de las
#: tres fuentes que se cruzan. En destilación sale de la comprobación previa de memoria; con
#: ``--reuse-run`` esa comprobación no corre, así que el volcado no declara parámetros y no se
#: importa.
PARAMS_EN_EL_VOLCADO: dict[str, tuple[str, str]] = {
    "pruning": ("pruning", "parameters_after"),
    "qat": ("qat", "parameters"),
    "distillation": ("preflight", "student_params"),
}


def parametros_del_volcado(volcado: dict[str, Any]) -> tuple[int | None, str]:
    """Extrae el número de parámetros que declara el volcado crudo y devuelve ``(parametros,
    procedencia)``; ``parametros`` es ``None`` si no aparece bajo ninguna clave conocida.
    """
    for tecnica, (seccion, clave) in PARAMS_EN_EL_VOLCADO.items():
        # 'or {}' porque el volcado de un --reuse-run trae la sección a null.
        valor = (volcado.get(seccion) or {}).get(clave)
        if valor is not None:
            return int(valor), f"volcado ({seccion}.{clave})"
    caminos = ", ".join(f"{s}.{c}" for s, c in PARAMS_EN_EL_VOLCADO.values())
    return None, f"volcado (no declara parámetros en ninguna de: {caminos})"


@dataclass(frozen=True)
class SourceLayout:
    """Dónde deja cada campaña su evidencia y sus pesos.

    La de poda escribe un volcado crudo por experimento; la de baselines no, y su evidencia llega de
    Drive junto al checkpoint. ``dirname_sin_hash`` indica que el directorio se llama como el
    ``experiment_id`` sin el hash, como en ``models/baselines/l_baseline_default/``.
    ``identidad_estricta`` hace falta cuando el cruce de parámetros no distingue experimentos, como
    en destilación.
    """

    name: str
    raw_subdir: str | None
    weights_subdir: str
    weights_relpath: Path
    dirname_sin_hash: bool
    evidence_files: tuple[str, ...] = ()
    params_en_la_evidencia: tuple[str, str] | None = None
    csv_subdir: str = ""
    raw_source_subdir: str = ""
    epochs_esperadas: int | None = None
    identidad_estricta: bool = False
    aviso: str = ""


#: Aviso de la disposición de destilación a 50 épocas, que se imprime con el plan.
AVISO_CAMPANA_50EP = (
    "La campaña de 50 épocas encadena el registro entre sesiones: cada sesión adopta la "
    "copia de Drive solo si el registro del clon es un prefijo exacto de ella. "
    "Una importación que llega al repositorio remoto antes de la última sesión rompe esa "
    "cadena, porque reescribe weights_path y notes. Entre sesiones, solo sin --apply o "
    "con un --registry-dir desechable."
)

#: Disposiciones conocidas. Las importaciones de QAT usan la de poda con ``--raw-dir``,
#: ``--weights-root`` y ``--weights-relpath``. La de destilación solo cubre la campaña de 50 épocas.
LAYOUTS: dict[str, SourceLayout] = {
    "pruning": SourceLayout(
        name="pruning",
        raw_subdir="pruning",
        weights_subdir="pruned",
        weights_relpath=Path("finetune") / "train" / "weights" / "best.pt",
        dirname_sin_hash=False,
    ),
    "baselines": SourceLayout(
        name="baselines",
        raw_subdir=None,
        weights_subdir="baselines",
        weights_relpath=Path("best.pt"),
        dirname_sin_hash=True,
        evidence_files=("complexity.json", "schedule.json", "training_cost.json"),
        params_en_la_evidencia=("complexity.json", "params_total"),
    ),
    # Lo que deja en Drive el cuaderno 03 (sección 11): 'registry/' y 'raw_epochs50/' bajo
    # resultados_distillation_50ep/, y los pesos en distillation/<id>/distill/. Los volcados van a
    # su propio subdirectorio porque el paso 32 lee results/raw/distillation sin descender.
    "distillation-50ep": SourceLayout(
        name="distillation-50ep",
        raw_subdir="distillation/epochs50",
        weights_subdir="distillation",
        weights_relpath=Path("distill") / "weights" / "best.pt",
        dirname_sin_hash=False,
        csv_subdir="registry",
        raw_source_subdir="raw_epochs50",
        epochs_esperadas=50,
        identidad_estricta=True,
        aviso=AVISO_CAMPANA_50EP,
    ),
}


def leer_train_args(ruta: Path) -> dict[str, Any] | None:
    """Lee los ``train_args`` que Ultralytics guarda en el checkpoint, o devuelve ``None`` si no los
    trae.
    """
    contenido = torch.load(ruta, map_location="cpu", weights_only=False)
    if not isinstance(contenido, dict):
        return None
    argumentos = contenido.get("train_args")
    return dict(argumentos) if isinstance(argumentos, dict) else None


def comprobar_identidad(
    eid: str,
    layout: SourceLayout,
    volcado: dict[str, Any] | None,
    fila_exp: dict[str, str],
    pesos: Path,
) -> list[str]:
    """Comprueba que volcado, fila y checkpoint son del mismo experimento y devuelve las
    discrepancias.

    Se comprueba que el volcado declara el ``experiment_id`` de su nombre, que ``config_path`` de la
    fila apunta al subdirectorio de la disposición, que el checkpoint se entrenó en un directorio
    con el nombre del experimento (``train_args.project``) y que los tres declaran las épocas
    esperadas.
    """
    problemas: list[str] = []
    if layout.identidad_estricta and volcado is not None and layout.raw_subdir is not None:
        declarado = volcado.get("experiment_id")
        if declarado != eid:
            problemas.append(
                f"[{eid}] el volcado dice ser de '{declarado}', no de este experimento."
            )
        sufijo = f"results/raw/{layout.raw_subdir}/{eid}.json"
        ruta_fila = fila_exp.get("config_path", "").replace("\\", "/")
        if not ruta_fila.endswith(sufijo):
            problemas.append(
                f"[{eid}] la fila apunta a su volcado en '{ruta_fila}', no en '.../{sufijo}'. "
                f"La sesión lo escribió en otro subdirectorio que el de la disposición "
                f"'{layout.name}': importarlo aquí lo dejaría donde la fila no lo encuentra."
            )

    argumentos = leer_train_args(pesos) if layout.identidad_estricta else None
    if layout.identidad_estricta:
        if argumentos is None:
            problemas.append(
                f"[{eid}] el checkpoint '{pesos}' no trae train_args: no hay forma de "
                "saber de qué entrenamiento es, y los parámetros no lo distinguen."
            )
        else:
            proyecto = str(argumentos.get("project", "")).replace("\\", "/").rstrip("/")
            if proyecto.rsplit("/", 1)[-1] != eid:
                problemas.append(
                    f"[{eid}] el checkpoint '{pesos}' se entrenó en '{proyecto}', no en el "
                    "directorio de este experimento. Son los pesos de otra ejecución; los "
                    "parámetros coinciden porque todos los alumnos tienen los mismos."
                )

    if layout.epochs_esperadas is not None:
        # El volcado y el checkpoint tienen que declararlas; en la fila basta con que no haya un
        # número distinto, porque vacío significa no medido.
        obligatorias: dict[str, Any] = {}
        if volcado is not None:
            obligatorias["volcado (effective_config.epochs)"] = volcado.get(
                "effective_config", {}
            ).get("epochs")
        if layout.identidad_estricta:
            obligatorias["checkpoint (train_args.epochs)"] = (argumentos or {}).get("epochs")
        en_la_fila = fila_exp.get("training_epochs", "").strip()
        fuentes = {**obligatorias, "fila (training_epochs)": en_la_fila or "(vacía)"}

        def _son_las_esperadas(valor: Any) -> bool:
            try:
                return float(valor) == layout.epochs_esperadas
            except (TypeError, ValueError):
                return False

        malas = {k: v for k, v in obligatorias.items() if not _son_las_esperadas(v)}
        if en_la_fila and not _son_las_esperadas(en_la_fila):
            malas["fila (training_epochs)"] = en_la_fila
        if malas:
            detalle = "; ".join(f"{k}={v}" for k, v in fuentes.items())
            problemas.append(
                f"[{eid}] la disposición '{layout.name}' es de {layout.epochs_esperadas} "
                f"épocas y no todas las fuentes lo dicen: {detalle}."
            )
    return problemas


#: Columnas en las que origen y registro local pueden diferir sin bloquear la importación: las rutas
#: (las del origen apuntan a ``/content/drive``), el ``timestamp`` (fin de la medición frente a
#: inscripción) y ``notes``. Estas diferencias se imprimen en el plan.
COLUMNAS_QUE_PUEDEN_DIFERIR: dict[str, tuple[str, ...]] = {
    "experiments": (*PATH_COLUMNS["experiments"], "timestamp", "notes"),
    "quality_metrics": (*PATH_COLUMNS["quality_metrics"], "notes"),
    "environments": ("timestamp",),
}


def experiment_dir(experiment_id: str, layout: SourceLayout, raiz: Path) -> Path:
    """Directorio local de un experimento según su disposición. Lanza InvalidExperimentIdError si
    hay que quitar el hash y el identificador no sigue la convención.
    """
    if not layout.dirname_sin_hash:
        return raiz / experiment_id
    partes = parse_experiment_id(experiment_id)
    return raiz / f"{partes['model_size']}_{partes['technique']}_{partes['variant']}"


def parametros_de_la_evidencia(directorio: Path, layout: SourceLayout) -> tuple[int | None, str]:
    """Extrae el número de parámetros de la evidencia junto a los pesos, cuando la disposición no
    trae volcado crudo. Devuelve ``(parametros, procedencia)``.
    """
    if layout.params_en_la_evidencia is None:
        return None, f"disposición '{layout.name}' (no declara de dónde salen los parámetros)"
    fichero, clave = layout.params_en_la_evidencia
    ruta = directorio / fichero
    if not ruta.is_file():
        return None, f"evidencia ({fichero} no está en '{directorio}')"
    with ruta.open("r", encoding="utf-8") as manejador:
        contenido = json.load(manejador)
    valor = contenido.get(clave)
    if valor is None:
        return None, f"evidencia ({fichero} no trae la clave '{clave}')"
    return int(valor), f"evidencia ({fichero}:{clave})"


def comparar_filas(
    tabla: str, local: dict[str, str], origen: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Compara una fila del origen con la inscrita y devuelve ``(bloqueantes, previstas)``.

    Una columna que solo existe en un lado cuenta como diferencia frente a la celda vacía.
    """
    tolerables = COLUMNAS_QUE_PUEDEN_DIFERIR.get(tabla, ())
    bloqueantes: list[str] = []
    previstas: list[str] = []
    for columna in sorted(set(local) | set(origen)):
        valor_local = normalize_value(local.get(columna, ""))
        valor_origen = normalize_value(origen.get(columna, ""))
        if valor_local == valor_origen:
            continue
        detalle = f"{columna}: local='{valor_local}' origen='{valor_origen}'"
        (previstas if columna in tolerables else bloqueantes).append(detalle)
    return bloqueantes, previstas


def clave_de_calidad(fila: dict[str, str]) -> tuple[str, ...]:
    """Clave primaria de una fila de quality_metrics, tomada de ``TABLE_PRIMARY_KEYS``."""
    return tuple(fila.get(columna, "") for columna in TABLE_PRIMARY_KEYS["quality_metrics"])


@dataclass
class ImportPlan:
    """Todo lo que se va a escribir, ya verificado."""

    experiment_ids: list[str] = field(default_factory=list)
    raw_copies: list[tuple[Path, Path]] = field(default_factory=list)
    experiment_rows: list[dict[str, str]] = field(default_factory=list)
    quality_rows: list[dict[str, str]] = field(default_factory=list)
    environment_rows: list[dict[str, str]] = field(default_factory=list)
    experiment_confirmations: list[tuple[str, list[str]]] = field(default_factory=list)
    quality_confirmations: list[tuple[tuple[str, ...], list[str]]] = field(default_factory=list)
    environment_confirmations: list[tuple[str, list[str]]] = field(default_factory=list)
    raw_confirmations: list[Path] = field(default_factory=list)


def read_csv_rows(ruta: Path) -> list[dict[str, str]]:
    """Lee un CSV del registro como lista de diccionarios. Lanza TFMEdgeError si no existe."""
    if not ruta.is_file():
        # El mensaje lista los ficheros que hay porque el error habitual es un prefijo (el cuaderno
        # de baselines deja 'l_experiments.csv').
        candidatos = sorted(p.name for p in ruta.parent.glob(f"*{ruta.name}"))
        pista = (
            f" En '{ruta.parent}' sí están: {', '.join(candidatos)}; si es lo que buscabas, "
            "pasa su prefijo en --csv-prefix."
            if candidatos
            else ""
        )
        raise TFMEdgeError(
            f"No existe '{ruta}'. El directorio de origen debe contener los CSV del "
            f"registro tal y como los dejó la sesión de Colab.{pista}"
        )
    with ruta.open("r", encoding="utf-8", newline="") as manejador:
        return [dict(fila) for fila in csv.DictReader(manejador)]


def count_checkpoint_parameters(ruta: Path) -> int:
    """Cuenta los parámetros del modelo guardado en un checkpoint."""
    contenido = torch.load(ruta, map_location="cpu", weights_only=False)
    modelo = contenido.get("model") if isinstance(contenido, dict) else contenido
    if modelo is None or not hasattr(modelo, "parameters"):
        raise TFMEdgeError(
            f"El checkpoint '{ruta}' no contiene un modelo con parámetros. "
            "No se puede comprobar que sea el que produjo la métrica."
        )
    return sum(p.numel() for p in modelo.parameters())


def rewrite_path(valor: str, source_root: str, destino_root: Path) -> str:
    """Reapunta una ruta del origen a la raíz local; si no empieza por ``source_root`` la deja
    igual.
    """
    if valor.startswith(source_root):
        return str(destino_root) + valor[len(source_root) :]
    return valor


def build_plan(
    args: argparse.Namespace,
    registro: ExperimentRegistry,
    rutas: Any,
) -> ImportPlan:
    """Verifica el origen contra el destino y construye el plan de escritura sin escribir nada.

    Si algo falla, lanza TFMEdgeError con todos los problemas encontrados.
    """
    origen = args.source_dir
    layout: SourceLayout = args.source_layout
    prefijo = args.csv_prefix
    origen_csv = origen / layout.csv_subdir
    exp_origen = read_csv_rows(origen_csv / f"{prefijo}experiments.csv")
    cal_origen = read_csv_rows(origen_csv / f"{prefijo}quality_metrics.csv")
    ent_origen = read_csv_rows(origen_csv / f"{prefijo}environments.csv")

    exp_local = {f["experiment_id"]: f for f in registro.read_rows("experiments")}
    ent_local = {f["environment_id"]: f for f in registro.read_rows("environments")}
    cal_local = {clave_de_calidad(f): f for f in registro.read_rows("quality_metrics")}

    if args.experiment_id:
        pedidos = list(args.experiment_id)
        conocidos = {fila["experiment_id"] for fila in exp_origen}
        desconocidos = [e for e in pedidos if e not in conocidos]
        if desconocidos:
            raise TFMEdgeError(
                f"No están en el origen: {', '.join(desconocidos)}. "
                f"El origen '{origen}' contiene: {', '.join(sorted(conocidos))}."
            )
    else:
        # Un experimento entra si le falta su fila o alguna de sus filas de calidad, como pasa con
        # los baselines.
        pedidos = [
            f["experiment_id"]
            for f in exp_origen
            if f["experiment_id"] not in exp_local
            or any(
                clave_de_calidad(c) not in cal_local
                for c in cal_origen
                if c["experiment_id"] == f["experiment_id"]
            )
        ]
        if not pedidos:
            raise TFMEdgeError(
                "El origen no aporta ninguna fila nueva. Para reimportar algo, indícalo con "
                "--experiment-id."
            )
        logger.info("El origen aporta filas nuevas de %d experimentos.", len(pedidos))

    plan = ImportPlan(experiment_ids=pedidos)
    problemas: list[str] = []

    for eid in pedidos:
        try:
            directorio = experiment_dir(eid, layout, args.weights_root)
        except TFMEdgeError as error:
            problemas.append(f"[{eid}] no se puede localizar su directorio local: {error}")
            continue

        # Evidencia de la medición.
        copia: tuple[Path, Path] | None = None
        volcado: dict[str, Any] | None = None
        if layout.raw_subdir is not None:
            volcado_origen = origen / layout.raw_source_subdir / f"{eid}.json"
            if not volcado_origen.is_file():
                problemas.append(
                    f"[{eid}] falta el volcado crudo '{volcado_origen.name}' en el origen. "
                    "Sin él no hay medición que respalde la fila."
                )
                continue
            volcado_destino = args.raw_dir / f"{eid}.json"
            if volcado_destino.exists():
                if volcado_destino.read_bytes() != volcado_origen.read_bytes():
                    problemas.append(
                        f"[{eid}] ya existe '{volcado_destino}' y su contenido no es el del "
                        "origen. Compáralos y aparta a mano el que sobre."
                    )
                    continue
                plan.raw_confirmations.append(volcado_destino)
            else:
                copia = (volcado_origen, volcado_destino)
            with volcado_origen.open("r", encoding="utf-8") as manejador:
                volcado = json.load(manejador)
            params_evidencia, procedencia_params = parametros_del_volcado(volcado)
        else:
            faltan = [f for f in layout.evidence_files if not (directorio / f).is_file()]
            if faltan:
                problemas.append(
                    f"[{eid}] falta evidencia en '{directorio}': {', '.join(faltan)}. La "
                    f"disposición '{layout.name}' no trae volcado crudo, así que estos "
                    "ficheros son lo único que respalda la fila."
                )
                continue
            params_evidencia, procedencia_params = parametros_de_la_evidencia(directorio, layout)

        if params_evidencia is None:
            problemas.append(
                f"[{eid}] la evidencia no declara el número de parámetros: "
                f"{procedencia_params}. Sin él no se puede cruzar con la fila y el "
                "checkpoint."
            )
            continue

        fila_exp = next(f for f in exp_origen if f["experiment_id"] == eid)
        params_fila = fila_exp.get("effective_params", "").strip()

        # Pesos locales.
        pesos = directorio / args.weights_relpath
        if not pesos.is_file():
            problemas.append(
                f"[{eid}] no hay pesos locales en '{pesos}'. Descárgalos de Drive antes "
                "de importar, o pasa --weights-relpath si están en otro sitio."
            )
            continue
        params_pesos = count_checkpoint_parameters(pesos)

        declarados = {
            procedencia_params: params_evidencia,
            "fila (effective_params)": int(params_fila) if params_fila else None,
            "checkpoint local (contados)": params_pesos,
        }
        distintos = {v for v in declarados.values() if v is not None}
        if len(distintos) != 1:
            detalle = "; ".join(f"{k}={v}" for k, v in declarados.items())
            problemas.append(
                f"[{eid}] el número de parámetros no coincide entre las tres fuentes: "
                f"{detalle}. Los pesos locales no son los que produjeron esa métrica."
            )
            continue
        logger.info("[%s] parámetros coherentes en las tres fuentes: %d", eid, params_pesos)

        discrepancias = comprobar_identidad(eid, layout, volcado, fila_exp, pesos)
        if discrepancias:
            problemas.extend(discrepancias)
            continue

        # El volcado crudo se copia aunque la fila ya esté inscrita.
        if copia is not None:
            plan.raw_copies.append(copia)

        # Fila de experimento: se inscribe o se confirma, nunca se sobrescribe.
        if eid in exp_local:
            bloqueantes, previstas = comparar_filas("experiments", exp_local[eid], fila_exp)
            if bloqueantes:
                problemas.append(
                    f"[{eid}] ya está inscrito y su fila NO coincide con la del origen en: "
                    + "; ".join(bloqueantes)
                    + ". No se reescribe ninguna de las dos: que dos fuentes digan cosas "
                    "distintas del mismo experimento es algo que hay que mirar, no un "
                    "conflicto que se resuelva eligiendo una."
                )
                continue
            plan.experiment_confirmations.append((eid, previstas))
        else:
            # Solo el nombre del directorio de origen, para no meter rutas absolutas en notes. La
            # sesión la identifica el environment_id de la fila de calidad.
            procedencia = (
                f" Importado desde el volcado '{Path(origen).name}' con "
                f"scripts/14_import_colab_registry.py el {utc_timestamp()}; medido en otra "
                "máquina (ver environment_id de su fila de calidad). weights_path reapunta "
                "al ejemplar local; el volcado crudo conserva las rutas originales de la "
                "máquina que midió."
            )
            fila_nueva = dict(fila_exp)
            for columna in PATH_COLUMNS["experiments"]:
                fila_nueva[columna] = rewrite_path(
                    fila_nueva.get(columna, ""), args.source_root, rutas.root
                )
            fila_nueva["weights_path"] = str(pesos)
            fila_nueva["notes"] = (fila_nueva.get("notes", "") + procedencia).strip()
            plan.experiment_rows.append(fila_nueva)

        # Filas de calidad.
        for fila_cal in cal_origen:
            if fila_cal["experiment_id"] != eid:
                continue
            clave = clave_de_calidad(fila_cal)
            if clave in cal_local:
                bloqueantes, previstas = comparar_filas(
                    "quality_metrics", cal_local[clave], fila_cal
                )
                if bloqueantes:
                    problemas.append(
                        f"[{eid}] la fila de calidad {clave} ya está inscrita y NO coincide "
                        "con la del origen en: " + "; ".join(bloqueantes) + "."
                    )
                    continue
                plan.quality_confirmations.append((clave, previstas))
                continue
            fila_cal_nueva = dict(fila_cal)
            for columna in PATH_COLUMNS["quality_metrics"]:
                fila_cal_nueva[columna] = rewrite_path(
                    fila_cal_nueva.get(columna, ""), args.source_root, rutas.root
                )
            plan.quality_rows.append(fila_cal_nueva)

    # Entornos citados.
    citados = {f["environment_id"] for f in plan.quality_rows if f.get("environment_id")}
    ya_planificados: set[str] = set()
    for env_id in sorted(citados):
        fila_env = next((f for f in ent_origen if f["environment_id"] == env_id), None)
        if env_id in ent_local:
            if fila_env is not None:
                bloqueantes, previstas = comparar_filas("environments", ent_local[env_id], fila_env)
                if bloqueantes:
                    problemas.append(
                        f"El entorno '{env_id}' ya está inscrito con un contenido distinto "
                        "del que trae el origen: " + "; ".join(bloqueantes) + ". El "
                        "environment_id es el hash de la huella, así que dos filas con el "
                        "mismo identificador y distinto contenido significan que el "
                        "identificador no es función de lo que describe, y las medidas que "
                        "lo citan dejan de saber en qué máquina se tomaron."
                    )
                    continue
                plan.environment_confirmations.append((env_id, previstas))
            continue
        if fila_env is None:
            problemas.append(
                f"El environment_id '{env_id}' que citan las filas de calidad no está "
                "en el registro local ni en environments.csv del origen. Sin él se "
                "pierden la versión de torch y el commit con los que se midió."
            )
            continue
        if env_id not in ya_planificados:
            plan.environment_rows.append(fila_env)
            ya_planificados.add(env_id)

    if problemas:
        raise TFMEdgeError(
            "La importación no se ha hecho. Comprobaciones fallidas:\n  - "
            + "\n  - ".join(problemas)
        )
    return plan


def apply_plan(plan: ImportPlan, registro: ExperimentRegistry) -> None:
    """Ejecuta el plan: copia los volcados e inscribe entornos, experimentos y calidad, en ese
    orden, para no dejar referencias colgando.
    """
    for origen, destino in plan.raw_copies:
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origen, destino)
        logger.info("Volcado crudo copiado: %s", destino)

    for fila in plan.environment_rows:
        registro.append("environments", fila, on_duplicate="fail")
        logger.info("Entorno inscrito: %s", fila["environment_id"])

    for fila in plan.experiment_rows:
        registro.append("experiments", fila, on_duplicate="fail")
        logger.info("Experimento inscrito: %s", fila["experiment_id"])

    for fila in plan.quality_rows:
        registro.append("quality_metrics", fila, on_duplicate="fail")
        logger.info(
            "Calidad inscrita: %s (%s, map_50_95=%s)",
            fila["experiment_id"],
            fila["device_id"],
            fila.get("map_50_95", ""),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define y analiza los argumentos de la CLI."""
    rutas = get_project_paths()
    analizador = argparse.ArgumentParser(
        description=(
            "Incorpora al registro local los resultados medidos en otra máquina (normalmente "
            "Colab) y comprueba que los pesos descargados son los que produjeron las métricas."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  # Ver qué haría, sin escribir nada: es lo que hace sin banderas\n"
            "  python scripts/14_import_colab_registry.py --source-dir /tmp/colab_registro\n\n"
            "  # Lo mismo, pero escribiéndolo\n"
            "  python scripts/14_import_colab_registry.py --source-dir /tmp/colab_registro "
            "--apply\n\n"
            "  # Importar solo dos experimentos concretos\n"
            "  python scripts/14_import_colab_registry.py --source-dir /tmp/colab_registro "
            "--experiment-id s_pruning_l1global_r50_9419b26f --apply\n\n"
            "  # Completar las filas de calidad de los baselines entrenados en Colab\n"
            "  python scripts/14_import_colab_registry.py --layout baselines "
            "--source-dir models/baselines/l_baseline_default --csv-prefix l_\n\n"
            "  # La campaña de destilación a 50 épocas, con la carpeta de Drive bajada\n"
            "  python scripts/14_import_colab_registry.py --layout distillation-50ep "
            "--source-dir /tmp/resultados_distillation_50ep\n"
        ),
    )
    analizador.add_argument(
        "--layout",
        choices=sorted(LAYOUTS),
        default="pruning",
        help=(
            "Disposición de lo que se importa (por defecto: %(default)s). 'pruning' "
            "espera un volcado crudo {experiment_id}.json por experimento en el origen; "
            "'baselines' no espera ninguno y busca complexity.json, schedule.json y "
            "training_cost.json junto al checkpoint, que es donde los deja el "
            "entrenamiento. 'distillation-50ep' espera la carpeta "
            "resultados_distillation_50ep/ de Drive (registry/ y raw_epochs50/) y "
            "comprueba además que volcado, fila y checkpoint son del mismo experimento "
            "y de 50 épocas."
        ),
    )
    analizador.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directorio con los CSV del registro de origen y los volcados crudos.",
    )
    analizador.add_argument(
        "--experiment-id",
        nargs="+",
        default=None,
        help=(
            "Experimentos a importar. Si se omite, se toman los del origen que aporten alguna "
            "fila nueva, sea de experimento o de calidad."
        ),
    )
    analizador.add_argument(
        "--csv-prefix",
        default="",
        help=(
            "Prefijo de los CSV en el directorio de origen (por defecto, ninguno). Para el "
            "cuaderno de baselines, --csv-prefix l_."
        ),
    )
    analizador.add_argument(
        "--source-root",
        default="/content/tfm-edge-object-detection",
        help=(
            "Raíz del proyecto en la máquina de origen, para reapuntar las rutas del "
            "registro a esta máquina (por defecto: %(default)s)."
        ),
    )
    analizador.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Destino de los volcados crudos. Por defecto, results/raw/pruning con --layout "
            "pruning y results/raw/distillation/epochs50 con --layout distillation-50ep. No se "
            "admite con --layout baselines."
        ),
    )
    analizador.add_argument(
        "--weights-root",
        type=Path,
        default=None,
        help=(
            "Directorio bajo el que buscar los pesos locales. Por defecto, models/pruned "
            "con --layout pruning, models/baselines con --layout baselines y "
            "models/distillation con --layout distillation-50ep."
        ),
    )
    analizador.add_argument(
        "--weights-relpath",
        type=Path,
        default=None,
        help=(
            "Ruta de los pesos dentro del directorio del experimento. Por defecto, "
            "'finetune/train/weights/best.pt' con --layout pruning (el modelo podado sin "
            "reajustar está junto a 'train/', no dentro), 'best.pt' con --layout baselines y "
            "'distill/weights/best.pt' con --layout distillation-50ep."
        ),
    )
    analizador.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Directorio del registro de destino (por defecto, el del proyecto).",
    )
    analizador.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Escribe el plan. Sin esta opción solo se verifica y se muestra, igual que en "
            "15_normalize_pruned_layout.py."
        ),
    )
    analizador.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Obsoleta: ahora es el comportamiento por defecto. Se sigue aceptando "
            "para no romper una invocación escrita antes del cambio, y avisa."
        ),
    )
    analizador.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    args = analizador.parse_args(argv)

    # Las tres rutas dependen de la disposición; lo que se pase explícitamente tiene prioridad.
    layout = LAYOUTS[args.layout]
    args.source_layout = layout
    if layout.raw_subdir is None and args.raw_dir is not None:
        analizador.error(
            f"--raw-dir no aplica a la disposición '{layout.name}': no trae volcado crudo. "
            "Su evidencia son los ficheros que acompañan al checkpoint "
            f"({', '.join(layout.evidence_files)})."
        )
    if args.raw_dir is None and layout.raw_subdir is not None:
        args.raw_dir = rutas.results_raw / layout.raw_subdir
    if args.weights_root is None:
        args.weights_root = rutas.root / "models" / layout.weights_subdir
    if args.weights_relpath is None:
        args.weights_relpath = layout.weights_relpath
    return args


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada."""
    args = parse_args(argv)
    setup_logging("14_import_colab_registry", level=args.log_level)
    rutas = get_project_paths()

    try:
        clases = load_class_names(rutas.root / "configs" / "dataset" / "sh17_classes.json")
        registro = ExperimentRegistry(registry_dir=args.registry_dir, class_names=clases)
        plan = build_plan(args, registro, rutas)
    except TFMEdgeError as error:
        logger.error("%s", error)
        return 2

    print("\n=== Plan de importación ===")
    print(f"  origen      : {args.source_dir}")
    print(f"  destino     : {registro.registry_dir}")
    print(f"  disposición : {args.layout}")
    for entorno in plan.environment_rows:
        print(f"  + entorno    {entorno['environment_id']} ({entorno.get('device_id', '')})")
    for fila in plan.experiment_rows:
        print(f"  + experimento {fila['experiment_id']}  params={fila.get('effective_params', '')}")
        print(f"      pesos: {fila['weights_path']}")
    for fila in plan.quality_rows:
        print(
            f"  + calidad     {fila['experiment_id']}  device={fila['device_id']}  "
            f"map_50={fila.get('map_50', '')}  map_50_95={fila.get('map_50_95', '')}"
        )
    for _, destino in plan.raw_copies:
        print(f"  + volcado     {destino}")

    # Lo que ya estaba y coincide también se imprime.
    for eid, previstas in plan.experiment_confirmations:
        print(f"  = experimento {eid}  ya inscrito y coincide; no se reescribe")
        for detalle in previstas:
            print(f"      difiere lo previsto: {detalle}")
    for clave, previstas in plan.quality_confirmations:
        print(f"  = calidad     {'/'.join(clave)}  ya inscrita y coincide")
        for detalle in previstas:
            print(f"      difiere lo previsto: {detalle}")
    for env_id, previstas in plan.environment_confirmations:
        print(f"  = entorno     {env_id}  ya inscrito y coincide")
        for detalle in previstas:
            print(f"      difiere lo previsto: {detalle}")
    for destino in plan.raw_confirmations:
        print(f"  = volcado     {destino}  ya estaba, idéntico al del origen")
    if args.source_layout.aviso:
        print(f"\nAVISO ({args.layout}): {args.source_layout.aviso}")

    if args.dry_run:
        # Por defecto no escribe; solo se avisa de que la opción sobra.
        logger.warning(
            "--dry-run es obsoleta: no escribir es ahora el comportamiento por "
            "defecto, y escribir se pide con --apply."
        )

    if not args.apply:
        print("\nNo se ha escrito nada, ni volcados ni filas.")
        print("Vuelve a lanzarlo con --apply para aplicar el plan.")
        logger.info("Plan mostrado. No se ha escrito nada (falta --apply).")
        return 0

    try:
        apply_plan(plan, registro)
    except TFMEdgeError as error:
        logger.error("Fallo al escribir: %s", error)
        return 2

    informe = registro.validate()
    print("\n=== Validación del registro tras la importación ===")
    print(informe.summary())
    if not informe.is_valid:
        logger.error("El registro no valida tras la importación.")
        return 2

    print(
        f"\nImportados {len(plan.experiment_rows)} experimentos, "
        f"{len(plan.quality_rows)} filas de calidad y "
        f"{len(plan.environment_rows)} entornos. Confirmados sin reescribir: "
        f"{len(plan.experiment_confirmations)} experimentos, "
        f"{len(plan.quality_confirmations)} filas de calidad y "
        f"{len(plan.environment_confirmations)} entornos."
    )
    logger.info(
        "Importación completada: %d experimentos, %d calidad, %d entornos; "
        "confirmados sin reescribir %d/%d/%d.",
        len(plan.experiment_rows),
        len(plan.quality_rows),
        len(plan.environment_rows),
        len(plan.experiment_confirmations),
        len(plan.quality_confirmations),
        len(plan.environment_confirmations),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
