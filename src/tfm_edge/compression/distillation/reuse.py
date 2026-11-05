"""Lee de disco un entrenamiento de destilación terminado para completarlo sin repetirlo.

Se usa con --reuse-run cuando la campaña falla después de entrenar. Lo que el entrenamiento no
guarda (normas iniciales de los adaptadores, número de parámetros, gradiente paso a paso) queda en
None, para que no se confunda con una medida. Para el control basta la media por época del
gradiente: las normas no son negativas, así que media cero implica cero en todos los pasos.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.distillation.trainer import COMPONENTS_FILENAME, DistillationError
from tfm_edge.training.cost import TRAINING_COST_FILENAME

logger = get_logger(__name__)

ADAPTER_NORM_PREFIX = "adapter_norm_"

ADAPTER_GRAD_COLUMN = "adapter_grad_norm"

#: Columnas del desglose que no son de un adaptador.
COMPONENT_COLUMNS = (
    "epoch",
    "steps",
    "response_cls",
    "response_box",
    "features",
    "detection",
    "total",
    "marked_anchors",
)


@dataclass
class ReusedRun:
    """Entrenamiento de destilación leído de disco. Lo que no está en disco queda en None."""

    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path | None = None
    cost: dict[str, Any] | None = None
    components_csv: Path | None = None
    epoch_components: list[dict[str, float]] = field(default_factory=list)
    adapter_norms_first_epoch: dict[str, float] = field(default_factory=dict)
    adapter_norms_last_epoch: dict[str, float] = field(default_factory=dict)
    adapter_grad_norms_by_epoch: list[float] = field(default_factory=list)
    closed_epochs: int = 0
    steps_in_closed_epochs: int = 0
    missing: list[str] = field(default_factory=list)

    @property
    def adapter_grad_always_zero(self) -> bool | None:
        """True o False si el gradiente de los adaptadores fue cero en todas las épocas; None si no
        hay datos.
        """
        if not self.adapter_grad_norms_by_epoch:
            return None
        # Igual que en DistillationResult: un nan no pasa esta comprobación.
        return not any(norma != 0.0 for norma in self.adapter_grad_norms_by_epoch)

    @property
    def adapter_norms_moved(self) -> bool | None:
        """Si alguna norma de adaptador cambió entre la primera y la última época registradas.

        No equivale a adapters_learned, que compara con las normas iniciales.
        """
        if not self.adapter_norms_first_epoch or not self.adapter_norms_last_epoch:
            return None
        return any(
            self.adapter_norms_last_epoch.get(capa) != primera
            for capa, primera in self.adapter_norms_first_epoch.items()
        )

    def summary(self) -> str:
        """Resume por consola qué se ha recuperado y qué no."""
        lineas = [
            "ENTRENAMIENTO REUTILIZADO (no se reentrena)",
            f"  Directorio       : {self.run_dir}",
            f"  Checkpoint       : {self.best_checkpoint.name}",
            f"  Épocas cerradas  : {self.closed_epochs}"
            + (f" ({self.steps_in_closed_epochs} pasos)" if self.steps_in_closed_epochs else ""),
        ]
        if self.cost:
            lineas.append(
                f"  Coste            : {self.cost.get('epochs_completed', '?')} épocas, "
                f"{float(self.cost.get('wall_time_s', 0.0)) / 3600:.2f} h en "
                f"'{self.cost.get('gpu_name') or self.cost.get('device') or '?'}'"
            )
        cero = self.adapter_grad_always_zero
        if cero is not None:
            lineas.append(
                f"  Gradiente a los adaptadores: "
                f"{'CERO en todas las épocas' if cero else 'distinto de cero'}"
            )
        if self.missing:
            lineas.append("  NO se ha podido recuperar de disco:")
            lineas.extend(f"    - {falta}" for falta in self.missing)
        return "\n".join(lineas)

    def to_dict(self) -> dict[str, Any]:
        """Procedencia de cada dato, para el volcado."""
        return {
            "run_dir": str(self.run_dir),
            "best_checkpoint": str(self.best_checkpoint),
            "last_checkpoint": str(self.last_checkpoint) if self.last_checkpoint else None,
            "components_csv": str(self.components_csv) if self.components_csv else None,
            "closed_epochs": self.closed_epochs,
            "steps_in_closed_epochs": self.steps_in_closed_epochs,
            "adapter_grad_resolution": "media por época; el paso a paso no está en disco",
            "not_recoverable": self.missing,
        }


def _leer_coste(run_dir: Path, missing: list[str]) -> dict[str, Any] | None:
    """Lee ``training_cost.json`` del directorio de la ejecución."""
    ruta = run_dir / TRAINING_COST_FILENAME
    if not ruta.is_file():
        missing.append(f"el coste: no hay '{TRAINING_COST_FILENAME}' en el directorio")
        return None
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        missing.append(f"el coste: '{ruta.name}' no se ha podido leer ({error})")
        return None
    if not isinstance(datos, dict):
        missing.append(f"el coste: '{ruta.name}' no contiene un objeto JSON")
        return None
    return datos


def _a_numero(valor: str) -> float | int | str:
    """Convierte un campo del CSV a int o float, o lo devuelve tal cual."""
    try:
        return int(valor)
    except ValueError:
        pass
    try:
        return float(valor)
    except ValueError:
        return valor


def _leer_componentes(run_dir: Path, reutilizado: ReusedRun) -> None:
    """Rellena el desglose por época leyendo el CSV de componentes."""
    ruta = run_dir / COMPONENTS_FILENAME
    if not ruta.is_file():
        reutilizado.missing.append(
            f"el desglose por época: no hay '{COMPONENTS_FILENAME}' en el directorio"
        )
        return

    with ruta.open(newline="", encoding="utf-8") as fichero:
        filas = [{k: _a_numero(v) for k, v in fila.items()} for fila in csv.DictReader(fichero)]

    if not filas:
        reutilizado.missing.append(f"el desglose por época: '{ruta.name}' no tiene filas")
        return

    reutilizado.components_csv = ruta
    reutilizado.epoch_components = filas  # type: ignore[assignment]
    reutilizado.closed_epochs = len(filas)
    reutilizado.steps_in_closed_epochs = sum(
        int(fila["steps"]) for fila in filas if isinstance(fila.get("steps"), int)
    )

    normas = [
        {
            clave[len(ADAPTER_NORM_PREFIX) :]: valor
            for clave, valor in fila.items()
            if clave.startswith(ADAPTER_NORM_PREFIX) and isinstance(valor, (int, float))
        }
        for fila in filas
    ]
    if normas and normas[0]:
        reutilizado.adapter_norms_first_epoch = normas[0]
        reutilizado.adapter_norms_last_epoch = normas[-1]
    else:
        reutilizado.missing.append(
            f"las normas de los adaptadores: '{ruta.name}' no trae columnas "
            f"'{ADAPTER_NORM_PREFIX}*'"
        )

    gradientes = [
        fila[ADAPTER_GRAD_COLUMN]
        for fila in filas
        if isinstance(fila.get(ADAPTER_GRAD_COLUMN), (int, float))
    ]
    if len(gradientes) == len(filas):
        reutilizado.adapter_grad_norms_by_epoch = [float(v) for v in gradientes]
    else:
        reutilizado.missing.append(
            f"la norma del gradiente de los adaptadores: '{ruta.name}' no la trae en "
            f"todas las filas"
        )


def load_finished_run(run_dir: Path) -> ReusedRun:
    """Lee de disco un entrenamiento de destilación terminado, sin cargar modelos.

    Lo que falte se anota en ``missing``. Lanza DistillationError si no hay checkpoint.
    """
    run_dir = Path(run_dir)
    mejor = run_dir / "weights" / "best.pt"
    if not mejor.is_file():
        raise DistillationError(
            f"No hay nada que reutilizar en '{run_dir}': falta 'weights/best.pt'. --reuse-run "
            "completa un entrenamiento ya hecho; para entrenar, quita la opción."
        )

    ultimo = run_dir / "weights" / "last.pt"
    reutilizado = ReusedRun(
        run_dir=run_dir,
        best_checkpoint=mejor,
        last_checkpoint=ultimo if ultimo.is_file() else None,
    )
    if reutilizado.last_checkpoint is None:
        reutilizado.missing.append("el último checkpoint: no hay 'weights/last.pt'")

    reutilizado.cost = _leer_coste(run_dir, reutilizado.missing)
    _leer_componentes(run_dir, reutilizado)

    # Estos tres nunca están en disco; se anotan siempre.
    reutilizado.missing.extend(
        (
            "las normas de partida de los adaptadores: el CSV empieza al final de "
            "la primera época, no antes de la primera",
            "el número de parámetros de los adaptadores: se sueltan al terminar (9.4.2)",
            "la norma del gradiente paso a paso: el CSV guarda la media por época",
        )
    )

    logger.info(
        "Reutilizado '%s': %d épocas cerradas, %d pasos.",
        run_dir,
        reutilizado.closed_epochs,
        reutilizado.steps_in_closed_epochs,
    )
    return reutilizado


def epoch_components_are_finite(filas: list[dict[str, float]]) -> bool:
    """True si todos los valores numéricos del desglose son finitos."""
    return all(
        math.isfinite(valor)
        for fila in filas
        for valor in fila.values()
        if isinstance(valor, (int, float))
    )
