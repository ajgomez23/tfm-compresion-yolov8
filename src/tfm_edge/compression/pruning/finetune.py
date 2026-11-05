"""Reajuste después de la poda.

Se registran dos métricas, antes y después del reajuste, y las dos se miden sobre el checkpoint en
disco: el best.pt de Ultralytics guarda la EMA de la mejor época, no los pesos en memoria. También
se comprueba que el checkpoint conserva la poda. La evaluación la pasa quien llama; sin evaluador la
métrica queda vacía.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tfm_edge.common.config import PruningConfig
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.pruning.dependency import PruningError
from tfm_edge.compression.pruning.schedule import FinetuneCallback

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Evaluador: recibe la ruta de un checkpoint y devuelve sus métricas. Vacío significa no medido.
MetricEvaluator = Callable[[Path], dict[str, float]]

#: mAP@50-95 porque la poda degrada sobre todo la localización, que mAP@50 apenas refleja.
DEFAULT_METRIC_KEY = "map_50_95"

#: Checkpoint del modelo podado sin reajustar.
PRUNED_CHECKPOINT_NAME = "pruned_before_finetune.pt"


@dataclass
class FinetuneReport:
    """Métricas alrededor de un reajuste posterior a la poda."""

    epochs: int = 0
    learning_rate: float = 0.0
    metric_key: str = DEFAULT_METRIC_KEY
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    metrics_baseline: dict[str, float] = field(default_factory=dict)
    checkpoint_before: str = ""
    checkpoint_after: str = ""
    parameters: int = 0
    cost: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def metric_before(self) -> float | str:
        """Métrica principal antes del reajuste, o celda vacía si no se midió."""
        return self.metrics_before.get(self.metric_key, "")

    @property
    def metric_after(self) -> float | str:
        """Métrica principal tras el reajuste, o celda vacía si no se midió."""
        return self.metrics_after.get(self.metric_key, "")

    @property
    def metric_baseline(self) -> float | str:
        """Métrica principal del modelo sin podar, o celda vacía si no se conoce."""
        return self.metrics_baseline.get(self.metric_key, "")

    @property
    def recovery(self) -> float | str:
        """Mejora de la métrica principal con el reajuste, o vacío si falta alguna medida."""
        antes, despues = self.metric_before, self.metric_after
        if isinstance(antes, float) and isinstance(despues, float):
            return despues - antes
        return ""

    @property
    def recovery_fraction(self) -> float | str:
        """Fracción recuperada de lo perdido al podar: ``(d - a) / (b - a)``.

        ``b`` es el baseline, ``a`` el podado sin reajustar y ``d`` el reajustado. Vacío si no hay
        baseline o si la poda no degradó nada.
        """
        base, antes = self.metric_baseline, self.metric_before
        ganancia = self.recovery
        if not (
            isinstance(base, float) and isinstance(antes, float) and isinstance(ganancia, float)
        ):
            return ""
        perdido = base - antes
        if perdido <= 0.0:
            return ""
        return ganancia / perdido

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        datos = asdict(self)
        datos["metric_before"] = self.metric_before
        datos["metric_after"] = self.metric_after
        datos["metric_baseline"] = self.metric_baseline
        datos["recovery"] = self.recovery
        datos["recovery_fraction"] = self.recovery_fraction
        return datos

    def summary(self) -> str:
        """Genera un resumen legible."""

        def cifra(valor: float | str) -> str:
            return f"{valor:.4f}" if isinstance(valor, float) else "—"

        lineas = [
            f"Reajuste tras la poda: {self.epochs} época(s), lr {self.learning_rate:g}",
            f"  métrica principal : {self.metric_key}",
            f"  sin podar         : {cifra(self.metric_baseline)}",
            f"  podado, sin reajustar: {cifra(self.metric_before)}",
            f"  podado y reajustado  : {cifra(self.metric_after)}",
            f"  recuperado        : {cifra(self.recovery)}",
        ]
        fraccion = self.recovery_fraction
        if isinstance(fraccion, float):
            lineas.append(f"  fracción recuperada de lo perdido: {fraccion:.1%}")
        return "\n".join(lineas)


def _medir(evaluator: MetricEvaluator | None, checkpoint: Path, momento: str) -> dict[str, float]:
    """Evalúa un checkpoint o deja constancia de que no se ha medido."""
    if evaluator is None:
        logger.warning(
            "No hay evaluador: la métrica %s se registra como AUSENTE, no como "
            "cero. La comparación del apartado 5.3.2.2 necesita las dos cifras.",
            momento,
        )
        return {}
    metricas = evaluator(checkpoint)
    logger.info(
        "Métrica %s: %s",
        momento,
        ", ".join(f"{k}={v:.4f}" for k, v in sorted(metricas.items())) or "(ninguna)",
    )
    return metricas


def _verificar_checkpoint_conserva_la_poda(
    checkpoint: Path, parametros_esperados: int, input_resolution: int
) -> "Detector":
    """Recarga un checkpoint y comprueba que conserva el número de parámetros de la poda."""
    from tfm_edge.models.detector import Detector

    recargado = Detector.load(checkpoint, input_resolution=input_resolution)
    parametros = recargado.count_parameters()
    if parametros != parametros_esperados:
        raise PruningError(
            f"El checkpoint '{checkpoint.name}' tiene {parametros:,} parámetros y "
            f"el modelo podado tenía {parametros_esperados:,}.",
            "La poda no se ha conservado al guardar. Revisa que el entrenador reciba el modelo "
            "por asignación directa y que el guardado no reconstruya desde el YAML.",
        )
    logger.info(
        "El checkpoint '%s' conserva la poda: %s parámetros",
        checkpoint.name,
        f"{parametros:,}",
    )
    return recargado


def finetune_pruned(
    detector: "Detector",
    config: PruningConfig,
    data_config: Path | str,
    output_dir: Path | str,
    evaluator: MetricEvaluator | None = None,
    baseline_metrics: dict[str, float] | None = None,
    run_name: str = "finetune",
    metric_key: str = DEFAULT_METRIC_KEY,
    trainer_kwargs: dict[str, Any] | None = None,
) -> tuple["Detector", FinetuneReport]:
    """Reajusta un modelo podado y mide antes y después.

    Con ``finetune_epochs=0`` solo mide el modelo podado. Épocas y tasa de aprendizaje salen de
    ``config`` y no se aceptan en ``trainer_kwargs``. Devuelve el detector recargado desde el
    checkpoint y el informe.
    """
    from tfm_edge.training.trainer import Trainer

    argumentos = dict(trainer_kwargs or {})
    conflictos = sorted({"epochs", "learning_rate", "optimizer"} & set(argumentos))
    if conflictos:
        raise PruningError(
            f"trainer_kwargs redefine {', '.join(conflictos)}, que salen de la "
            "configuración de poda.",
            "Cambia finetune_epochs o finetune_learning_rate en el YAML de poda.",
        )

    raiz = Path(output_dir) / run_name
    parametros = detector.count_parameters()

    # El antes también se mide sobre disco, como el después.
    ruta_antes = detector.save(
        raiz / PRUNED_CHECKPOINT_NAME,
        extra={"pruning_ratio_requested": config.ratio, "pruning_criterion": config.criterion},
    )
    _verificar_checkpoint_conserva_la_poda(ruta_antes, parametros, detector.input_resolution)
    metricas_antes = _medir(evaluator, ruta_antes, "antes del reajuste")

    informe = FinetuneReport(
        epochs=config.finetune_epochs,
        learning_rate=config.finetune_learning_rate,
        metric_key=metric_key,
        metrics_before=metricas_antes,
        metrics_baseline=dict(baseline_metrics or {}),
        checkpoint_before=str(ruta_antes),
        parameters=parametros,
    )

    if config.finetune_epochs == 0:
        informe.notes = (
            "finetune_epochs=0: no se ha reajustado. La métrica 'después' queda "
            "ausente a propósito, que no es lo mismo que un reajuste sin efecto."
        )
        logger.warning(
            "Reajuste desactivado (finetune_epochs=0). El modelo podado NO está "
            "recuperado y su métrica no es comparable con la del baseline."
        )
        return detector, informe

    entrenador = Trainer(
        detector=detector,
        data_config=data_config,
        output_dir=raiz,
        epochs=config.finetune_epochs,
        learning_rate=config.finetune_learning_rate,
        # Se pasa explícito para que Ultralytics no use su optimizador 'auto', que ignora
        # finetune_learning_rate. No está en PruningConfig para no cambiar el hash de configuración.
        optimizer="SGD",
        **argumentos,
    )
    resultado = entrenador.train(run_name="train")

    reajustado = _verificar_checkpoint_conserva_la_poda(
        resultado.best_checkpoint, parametros, detector.input_resolution
    )
    informe.checkpoint_after = str(resultado.best_checkpoint)
    informe.cost = resultado.cost.to_dict()
    informe.metrics_after = _medir(evaluator, resultado.best_checkpoint, "tras el reajuste")
    informe.notes = (
        "Las dos métricas salen de checkpoints en disco, no del modelo en "
        "memoria: Ultralytics entrena con media móvil exponencial y su best.pt "
        "no contiene los pesos que quedan en memoria al terminar."
    )
    logger.info("\n%s", informe.summary())
    return reajustado, informe


def make_partial_finetune_callback(
    config: PruningConfig,
    data_config: Path | str,
    output_dir: Path | str,
    evaluator: MetricEvaluator | None = None,
    metric_key: str = DEFAULT_METRIC_KEY,
    trainer_kwargs: dict[str, Any] | None = None,
) -> FinetuneCallback:
    """Construye el reajuste parcial que la poda iterativa aplica entre ciclos.

    Dura finetune_epochs_per_step. La métrica devuelta sale del best.pt, que no es el modelo en
    memoria que sigue podándose, así que solo sirve para seguir el avance.
    """
    parcial = config.model_copy(update={"finetune_epochs": config.finetune_epochs_per_step})

    def reajuste(detector: "Detector", ciclo: int) -> float | None:
        """Reajusta tras un ciclo de poda y devuelve la métrica, o None."""
        if parcial.finetune_epochs == 0:
            logger.warning(
                "Reajuste parcial del ciclo %d desactivado (finetune_epochs_per_step=0).",
                ciclo,
            )
            return None
        _, informe = finetune_pruned(
            detector,
            parcial,
            data_config=data_config,
            output_dir=output_dir,
            evaluator=evaluator,
            run_name=f"finetune_step{ciclo}",
            metric_key=metric_key,
            trainer_kwargs=trainer_kwargs,
        )
        metrica = informe.metric_after
        return metrica if isinstance(metrica, float) else None

    return reajuste
