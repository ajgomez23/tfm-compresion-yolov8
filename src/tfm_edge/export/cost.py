"""Coste de producir cada artefacto de despliegue a partir de los tiempos ya medidos.

La exportación guarda export_seconds y la cuantización quantize_seconds; el paso 31 los lleva a
export_time_s. El coste de un artefacto suma toda su cadena (por ejemplo ONNX FP32, IR FP32 e IR
INT8), que se sigue por los enlaces de procedencia guardados en cada paso y no por el nombre del
fichero. Si dos volcados dicen haber producido el mismo artefacto con tiempos distintos, se lanza
ExportCostError.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import (
    ProjectPaths,
    get_project_paths,
    relative_to_root,
    relocate_under_results,
)

logger = get_logger(__name__)

#: Grupos del volcado de exportación con su tipo de paso y la clave del tiempo.
STEP_GROUPS: dict[str, tuple[str, str]] = {
    "fp32_exports": ("export", "export_seconds"),
    "int8_artifacts": ("quantize", "quantize_seconds"),
}

#: Claves de procedencia por orden de preferencia: source_path (cuantización) y from_onnx
#: (conversión a OpenVINO).
SOURCE_KEYS: tuple[str, ...] = ("source_path",)
METADATA_SOURCE_KEYS: tuple[str, ...] = ("from_onnx",)

#: Solo cuentan los pasos que terminaron bien.
OK_STATUS: str = "ok"

#: Profundidad máxima de una cadena; las reales tienen tres pasos.
MAX_CHAIN_DEPTH: int = 32


class ExportCostError(TFMEdgeError):
    """Error al reconstruir el coste de un artefacto."""


@dataclass(frozen=True)
class ExportStep:
    """Paso medido que produjo un artefacto."""

    artifact: str
    kind: str
    target: str
    seconds: float | None
    source: str | None
    status: str
    dump: str

    def to_dict(self) -> dict[str, Any]:
        """Serializa el paso."""
        return {
            "artifact": self.artifact,
            "kind": self.kind,
            "target": self.target,
            "seconds": self.seconds,
            "source": self.source,
            "status": self.status,
            "dump": self.dump,
        }


@dataclass(frozen=True)
class ArtifactCost:
    """Coste de un artefacto con la cadena de pasos que lo justifica."""

    artifact: str
    seconds: float | None
    steps: tuple[ExportStep, ...]
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializa el coste."""
        return {
            "artifact": self.artifact,
            "seconds": self.seconds,
            "steps": [paso.to_dict() for paso in self.steps],
            "reason": self.reason,
        }


def _normalize(path: str, paths: ProjectPaths | None = None) -> str:
    """Ruta relativa a la raíz del proyecto.

    Si cae fuera de la raíz, se intenta reubicar por el tramo a partir de results/; si tampoco
    cuelga de ahí, se devuelve absoluta.
    """
    rutas = paths or get_project_paths()
    normalizada = relative_to_root(Path(path), paths=rutas)
    if not Path(normalizada).is_absolute():
        return normalizada
    reubicada = relocate_under_results(normalizada, rutas.results)
    if reubicada is None:
        return normalizada
    return relative_to_root(reubicada, paths=rutas)


def _source_of(entry: Mapping[str, Any]) -> str | None:
    """Artefacto del que parte un paso, o None si parte del checkpoint."""
    for clave in SOURCE_KEYS:
        valor = entry.get(clave)
        if valor:
            return str(valor)
    metadatos = entry.get("metadata") or {}
    if isinstance(metadatos, Mapping):
        for clave in METADATA_SOURCE_KEYS:
            valor = metadatos.get(clave)
            if valor:
                return str(valor)
    return None


def _seconds_of(entry: Mapping[str, Any], key: str) -> float | None:
    """Segundos de un paso, o None si faltan. Lanza ExportCostError si el valor no es numérico."""
    bruto = entry.get(key, "")
    if bruto == "" or bruto is None:
        return None
    try:
        return float(bruto)
    except (TypeError, ValueError) as error:
        raise ExportCostError(
            f"El paso declara {key}={bruto!r}, que no es un número. Corrige el "
            f"volcado: dar por ausente un valor mal escrito ocultaría el error."
        ) from error


def index_export_steps(
    exports_dir: Path | str,
    paths: ProjectPaths | None = None,
) -> dict[str, ExportStep]:
    """Indexa por artefacto los pasos medidos de los volcados de exportación."""
    raiz = Path(exports_dir)
    indice: dict[str, ExportStep] = {}

    for volcado in sorted(raiz.glob("**/*.json")):
        try:
            datos = json.loads(volcado.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("No se pudo leer el volcado %s: %s", volcado, error)
            continue
        if not isinstance(datos, dict):
            continue

        for grupo, (clase, clave_tiempo) in STEP_GROUPS.items():
            entradas = datos.get(grupo) or {}
            if not isinstance(entradas, Mapping):
                continue
            for destino, entrada in entradas.items():
                if not isinstance(entrada, Mapping):
                    continue
                ruta = entrada.get("path")
                if not ruta:
                    continue
                artefacto = _normalize(str(ruta), paths=paths)
                origen = _source_of(entrada)
                paso = ExportStep(
                    artifact=artefacto,
                    kind=clase,
                    target=str(destino),
                    seconds=_seconds_of(entrada, clave_tiempo),
                    source=_normalize(origen, paths=paths) if origen else None,
                    status=str(entrada.get("status", "")),
                    dump=_normalize(str(volcado), paths=paths),
                )
                anterior = indice.get(artefacto)
                if anterior is not None and anterior.seconds != paso.seconds:
                    raise ExportCostError(
                        f"Dos volcados dicen haber producido '{artefacto}' con tiempos "
                        f"distintos: {anterior.seconds} s en '{anterior.dump}' y {paso.seconds} "
                        f"s en '{paso.dump}'."
                    )
                indice.setdefault(artefacto, paso)

    logger.info("Pasos de exportación indexados: %d.", len(indice))
    return indice


def artifact_cost(artifact: str, steps: Mapping[str, ExportStep]) -> ArtifactCost:
    """Suma el coste de la cadena que produjo un artefacto.

    Si a un paso le falta el tiempo no se devuelve total. Lanza ExportCostError si la cadena es
    cíclica o demasiado larga.
    """
    cadena: list[ExportStep] = []
    visitados: set[str] = set()
    actual: str | None = artifact

    while actual is not None:
        if actual in visitados:
            raise ExportCostError(
                f"La cadena de '{artifact}' pasa dos veces por '{actual}'. Un volcado "
                f"declara una procedencia circular y no se puede sumar."
            )
        visitados.add(actual)
        paso = steps.get(actual)
        if paso is None:
            if not cadena:
                return ArtifactCost(
                    artifact=artifact,
                    seconds=None,
                    steps=(),
                    reason=(
                        f"Ningún volcado de exportación dice haber producido "
                        f"'{artifact}'. Si es un checkpoint de PyTorch no se exportó "
                        f"nada y no hay coste que apuntar; si es un artefacto, su "
                        f"volcado se ha perdido."
                    ),
                )
            return ArtifactCost(
                artifact=artifact,
                seconds=None,
                steps=tuple(cadena),
                reason=(
                    f"La cadena llega a '{actual}', del que ningún volcado dice cómo se "
                    f"produjo. Falta un eslabón, así que el total sería menor que el real."
                ),
            )
        if len(cadena) >= MAX_CHAIN_DEPTH:
            raise ExportCostError(
                f"La cadena de '{artifact}' supera los {MAX_CHAIN_DEPTH} eslabones. "
                f"Las reales tienen tres; esto es un volcado corrupto."
            )
        cadena.append(paso)
        actual = paso.source

    fallidos = [paso for paso in cadena if paso.status != OK_STATUS]
    if fallidos:
        return ArtifactCost(
            artifact=artifact,
            seconds=None,
            steps=tuple(cadena),
            reason=(
                f"La cadena incluye pasos con estado distinto de '{OK_STATUS}': "
                f"{[paso.artifact for paso in fallidos]}. Un paso que no terminó bien no "
                f"produjo lo que se midió, así que su tiempo no es el coste de nada."
            ),
        )

    sin_tiempo = [paso for paso in cadena if paso.seconds is None]
    if sin_tiempo:
        return ArtifactCost(
            artifact=artifact,
            seconds=None,
            steps=tuple(cadena),
            reason=(
                f"No hay tiempo apuntado de {[paso.artifact for paso in sin_tiempo]}. "
                f"Sumar solo los eslabones que sí lo tienen daría una cifra menor que la "
                f"real con aspecto de medida."
            ),
        )

    total = sum(paso.seconds for paso in cadena if paso.seconds is not None)
    return ArtifactCost(artifact=artifact, seconds=round(total, 2), steps=tuple(cadena))
