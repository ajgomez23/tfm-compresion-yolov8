"""Instrumentación del entrenamiento: tiempo, memoria y avance por época.

Complementa el results.csv de Ultralytics con el coste por época y lo vuelca a disco al terminar
cada época, porque una sesión de Colab se puede cortar. Se engancha con ``add_callback`` sin
importar Ultralytics, y assert_fired comprueba al final que los eventos se dispararon.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir
from tfm_edge.common.registry import utc_timestamp

logger = get_logger(__name__)

#: Eventos del trainer de Ultralytics que se usan.
TRAIN_START = "on_train_start"
EPOCH_END = "on_fit_epoch_end"
TRAIN_END = "on_train_end"


class CallbackError(TFMEdgeError):
    """Error en la instrumentación del entrenamiento."""


@dataclass
class EpochRecord:
    """Datos de una época."""

    epoch: int
    wall_seconds: float
    cumulative_hours: float
    metrics: dict[str, float] = field(default_factory=dict)
    learning_rates: list[float] = field(default_factory=list)
    gpu_peak_allocated_bytes: int | str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el registro como diccionario serializable."""
        return asdict(self)


class TrainingRecorder:
    """Registra cada época y vuelca el histórico a disco de forma atómica, sin esperar al final."""

    def __init__(self, output_path: Path | str, flush_every: int = 1) -> None:
        """Configura el registrador. ``flush_every`` indica cada cuántas épocas se vuelca."""
        self.output_path = ensure_parent_dir(output_path)
        self.flush_every = max(1, flush_every)
        self.records: list[EpochRecord] = []
        self.start_time: float = 0.0
        self.epoch_start_time: float = 0.0
        self.events_seen: dict[str, int] = {TRAIN_START: 0, EPOCH_END: 0, TRAIN_END: 0}

    def attach(self, trainer: Any) -> None:
        """Engancha el registrador a un trainer de Ultralytics."""
        if not hasattr(trainer, "add_callback"):
            raise CallbackError(
                f"El trainer {type(trainer).__name__} no expone 'add_callback'.",
                "La instrumentación depende del sistema de callbacks de "
                "Ultralytics. Comprueba la versión anclada.",
            )
        trainer.add_callback(TRAIN_START, self._on_train_start)
        trainer.add_callback(EPOCH_END, self._on_epoch_end)
        trainer.add_callback(TRAIN_END, self._on_train_end)
        logger.info("Instrumentación enganchada; histórico en %s", self.output_path)

    def _on_train_start(self, trainer: Any) -> None:
        """Arranca los cronómetros."""
        self.events_seen[TRAIN_START] += 1
        self.start_time = time.perf_counter()
        self.epoch_start_time = self.start_time
        self._reiniciar_pico_de_memoria(trainer)

    def _on_epoch_end(self, trainer: Any) -> None:
        """Registra la época que acaba de terminar y vuelca si toca."""
        self.events_seen[EPOCH_END] += 1
        ahora = time.perf_counter()
        registro = EpochRecord(
            epoch=int(getattr(trainer, "epoch", len(self.records))),
            wall_seconds=round(ahora - self.epoch_start_time, 3),
            cumulative_hours=round((ahora - self.start_time) / 3600.0, 4),
            metrics=self._extraer_metricas(trainer),
            learning_rates=self._extraer_lr(trainer),
            gpu_peak_allocated_bytes=self._pico_de_memoria(),
            timestamp=utc_timestamp(),
        )
        self.records.append(registro)
        self.epoch_start_time = ahora
        self._reiniciar_pico_de_memoria(trainer)

        if len(self.records) % self.flush_every == 0:
            self.flush()

    def _on_train_end(self, trainer: Any) -> None:
        """Vuelca el histórico completo."""
        self.events_seen[TRAIN_END] += 1
        self.flush()

    @staticmethod
    def _extraer_metricas(trainer: Any) -> dict[str, float]:
        """Extrae las métricas del trainer, tolerando que falten."""
        crudas = getattr(trainer, "metrics", None) or {}
        limpias: dict[str, float] = {}
        for clave, valor in dict(crudas).items():
            try:
                limpias[str(clave)] = float(valor)
            except (TypeError, ValueError):
                continue
        perdida = getattr(trainer, "tloss", None)
        if perdida is not None:
            try:
                limpias["train_loss"] = float(perdida.sum())
            except (TypeError, ValueError, AttributeError):
                pass
        return limpias

    @staticmethod
    def _extraer_lr(trainer: Any) -> list[float]:
        """Extrae las tasas de aprendizaje del optimizador."""
        optimizador = getattr(trainer, "optimizer", None)
        if optimizador is None:
            return []
        try:
            return [round(float(g["lr"]), 8) for g in optimizador.param_groups]
        except (KeyError, TypeError, ValueError):
            return []

    @staticmethod
    def _reiniciar_pico_de_memoria(trainer: Any) -> None:
        """Reinicia el contador de memoria pico, si hay GPU."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:  # pragma: no cover - sin CUDA
            pass

    @staticmethod
    def _pico_de_memoria() -> int | str:
        """Memoria de GPU pico de la época en bytes, o cadena vacía si no hay GPU."""
        try:
            import torch

            if not torch.cuda.is_available():
                return ""
            return int(torch.cuda.max_memory_allocated())
        except Exception:  # pragma: no cover - sin CUDA
            return ""

    def flush(self) -> Path:
        """Vuelca el histórico a disco de forma atómica."""
        contenido = {
            "num_epochs_recorded": len(self.records),
            "total_hours": self.records[-1].cumulative_hours if self.records else 0.0,
            "events_seen": dict(self.events_seen),
            "epochs": [r.to_dict() for r in self.records],
        }
        temporal = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporal.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")
        temporal.replace(self.output_path)
        return self.output_path

    def assert_fired(self, expected_epochs: int | None = None) -> None:
        """Comprueba que se registró al menos una época.

        Si cambian los nombres de los eventos de Ultralytics, el histórico quedaría vacío sin error.
        Si no coinciden las épocas esperadas solo se avisa, porque puede haber parada temprana.
        """
        if self.events_seen[EPOCH_END] == 0:
            raise CallbackError(
                "La instrumentación no registró ninguna época: no se disparó "
                f"'{EPOCH_END}' ni una sola vez.",
                "Probablemente Ultralytics ha cambiado el nombre del evento; revisa los "
                "callbacks de la versión instalada.",
            )
        if expected_epochs is not None and len(self.records) < expected_epochs:
            logger.warning(
                "Se registraron %d épocas de las %d previstas. Si hubo parada "
                "temprana es lo esperado; si no, la ejecución se cortó.",
                len(self.records),
                expected_epochs,
            )

    def summary(self) -> dict[str, Any]:
        """Resumen del coste: épocas, horas totales y segundos por época."""
        if not self.records:
            return {"epochs": 0, "total_hours": "", "mean_epoch_seconds": ""}
        duraciones = [r.wall_seconds for r in self.records]
        picos = [
            r.gpu_peak_allocated_bytes
            for r in self.records
            if isinstance(r.gpu_peak_allocated_bytes, int)
        ]
        return {
            "epochs": len(self.records),
            "total_hours": self.records[-1].cumulative_hours,
            "mean_epoch_seconds": round(sum(duraciones) / len(duraciones), 2),
            "min_epoch_seconds": round(min(duraciones), 2),
            "max_epoch_seconds": round(max(duraciones), 2),
            "gpu_peak_allocated_bytes": max(picos) if picos else "",
        }


def make_progress_logger(every: int = 1) -> Callable[[Any], None]:
    """Callback que escribe el avance en el log del proyecto cada ``every`` épocas."""
    paso = max(1, every)

    def informar(trainer: Any) -> None:
        """Registra una línea de avance."""
        epoca = int(getattr(trainer, "epoch", 0))
        if epoca % paso != 0:
            return
        metricas = TrainingRecorder._extraer_metricas(trainer)
        interesantes = {
            k: round(v, 4)
            for k, v in metricas.items()
            if "mAP" in k or "map" in k or k == "train_loss"
        }
        logger.info("Época %d: %s", epoca, interesantes or "sin métricas todavía")

    return informar
