"""Afinado QAT completo: preparar, afinar, quitar la simulación y medir.

Mide el baseline FP32 y el modelo afinado ya sin simulación, también en FP32, sobre los checkpoints
en disco. El INT8 lo materializa cada runtime después, por el mismo camino que PTQ. No se mide el
modelo simulado antes de afinar: sin la simulación es el baseline exacto, y con ella sería una PTQ
en eager distinta de la del estudio. Comprueba que hay observadores, que han visto datos, que
congelar BatchNorm tiene efecto si se pide y que el número de parámetros no cambia. Sin evaluador
las métricas quedan vacías.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir
from tfm_edge.compression.quantization.qat import (
    QatConfig,
    QatError,
    freeze_observers,
    prepare_for_qat,
    strip_fake_quantization,
)

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from torch import nn

    from tfm_edge.common.config import QuantizationConfig
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

MetricEvaluator = Callable[[Path], dict[str, float]]

DEFAULT_METRIC_KEY = "map_50_95"

#: Checkpoint con los pesos FP32 afinados y sin simulación, que pasa a la materialización INT8.
STRIPPED_CHECKPOINT_NAME = "qat_stripped_fp32.pt"


@dataclass
class QatFinetuneReport:
    """Resultado de un afinado QAT.

    ``integrity`` recoge lo que vieron las comprobaciones durante el entrenamiento.
    """

    epochs: int = 0
    learning_rate: float = 0.0
    metric_key: str = DEFAULT_METRIC_KEY
    preparation: dict[str, Any] = field(default_factory=dict)
    metrics_baseline: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)
    checkpoint_after: str = ""
    parameters: int = 0
    observers_exercised: bool | str = ""
    frozen: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def metric_baseline(self) -> float | str:
        """Métrica principal del baseline, o ``""``."""
        return self.metrics_baseline.get(self.metric_key, "")

    @property
    def metric_after(self) -> float | str:
        """Métrica principal tras el afinado, o ``""``."""
        return self.metrics_after.get(self.metric_key, "")

    @property
    def fp32_delta(self) -> float | str:
        """Cambio de la métrica en FP32 por afinar con la simulación (``después - baseline``), o
        ``""``.
        """
        base, despues = self.metric_baseline, self.metric_after
        if isinstance(base, float) and isinstance(despues, float):
            return despues - base
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        datos = asdict(self)
        datos["fp32_delta"] = self.fp32_delta
        return datos

    def summary(self) -> str:
        """Devuelve un resumen legible del afinado."""

        def fmt(valor: float | str) -> str:
            return f"{valor:.4f}" if isinstance(valor, float) else "—"

        lineas = [
            f"Afinado QAT: {self.epochs} época(s), lr {self.learning_rate:g}",
            f"  métrica principal   : {self.metric_key}",
            f"  baseline FP32       : {fmt(self.metric_baseline)}",
            f"  afinado, FP32       : {fmt(self.metric_after)}",
            f"  diferencia en FP32  : {fmt(self.fp32_delta)}",
            f"  observadores        : "
            f"{self.preparation.get('num_weight_observers', '—')} en pesos, "
            f"{self.preparation.get('num_activation_observers', '—')} en activaciones",
            # Se imprime "no comprobado" en lugar de dejar la línea en blanco.
            f"  vieron datos        : "
            f"{self.observers_exercised if self.observers_exercised != '' else '—'}",
        ]
        if self.frozen:
            lineas.append(f"  congelado           : {self.frozen}")
        if self.integrity:
            lineas.append(f"  guardas             : {self.integrity}")
        lineas.append(
            "  El INT8 no se mide aquí: lo materializa cada runtime por el "
            "mismo camino que el baseline."
        )
        if self.notes:
            lineas.append(f"  nota                : {self.notes}")
        return "\n".join(lineas)


def _medir(evaluador: MetricEvaluator | None, checkpoint: Path, etiqueta: str) -> dict[str, float]:
    """Evalúa un checkpoint, o devuelve ``{}`` si no hay evaluador."""
    if evaluador is None:
        logger.info("Sin evaluador: la métrica '%s' queda AUSENTE, no a cero.", etiqueta)
        return {}
    logger.info("Midiendo %s sobre '%s'...", etiqueta, checkpoint)
    return dict(evaluador(checkpoint))


def wrap_prepared_model(model: "nn.Module", reference: "Detector") -> "Detector":
    """Envuelve un modelo preparado para QAT en un Detector, sin validar la estructura.

    prepare_qat cambia las convoluciones por las de torch.ao.nn.qat, así que la comprobación de
    estructura de Ultralytics no se cumple. El número de parámetros se comprueba aparte.
    """
    from tfm_edge.models.detector import Detector

    return Detector(
        model=model,
        size=reference.size,
        num_classes=reference.num_classes,
        input_resolution=reference.input_resolution,
        validate_structure=False,
    )


#: Parte de la ruta que identifica a un observador de pesos; el resto observa activaciones.
WEIGHT_OBSERVER_PATH = "weight_fake_quant"


def observers_have_been_exercised(model: "nn.Module") -> bool:
    """True si algún observador de activación tiene un rango finito.

    Un observador sin datos se queda en +inf/-inf. Los de pesos no cuentan, porque se asientan al
    preparar sin necesidad de datos.
    """
    import torch

    from tfm_edge.compression.quantization import qat as qat_module

    qao = qat_module._importar_quantization()
    for nombre, modulo in model.named_modules():
        if not isinstance(modulo, qao.ObserverBase):
            continue
        if WEIGHT_OBSERVER_PATH in nombre:
            continue
        minimo = getattr(modulo, "min_val", None)
        maximo = getattr(modulo, "max_val", None)
        if minimo is None or maximo is None:
            continue
        if minimo.numel() == 0 or maximo.numel() == 0:
            continue
        if bool(torch.isfinite(minimo).all()) and bool(torch.isfinite(maximo).all()):
            return True
    return False


def count_fake_quant_modules(model: "nn.Module") -> int:
    """Número de módulos FakeQuantizeBase. prepare_qat no añade parámetros, así que hay que
    contarlos.
    """
    from tfm_edge.compression.quantization import qat as qat_module

    qao = qat_module._importar_quantization()
    return sum(1 for modulo in model.modules() if isinstance(modulo, qao.FakeQuantizeBase))


def count_active_fake_quant_modules(model: "nn.Module") -> int:
    """Número de nodos de simulación con la simulación activa.

    El checkpoint qat_stripped_fp32.pt de s_qat_int8-noexcl_d4d403fa conserva los nodos
    desactivados: no es exportable, pero sus pesos son los FP32 afinados. Esta cuenta lo distingue
    de unos pesos que aún simulan.
    """
    from tfm_edge.compression.quantization import qat as qat_module

    qao = qat_module._importar_quantization()
    return sum(
        1
        for modulo in model.modules()
        if isinstance(modulo, qao.FakeQuantizeBase) and int(modulo.fake_quant_enabled) != 0
    )


def make_qat_integrity_callbacks(
    report: QatFinetuneReport,
) -> list[tuple[str, Callable[[Any], None]]]:
    """Comprobaciones de invariante del afinado QAT.

    QAT no cambia parámetros ni canales, así que la verificación de estructura del Trainer no
    detecta nada aquí. Se comprueba: 1) que el modelo que entrena lleva simulación; 2) que la EMA se
    rehace tras el primer lote, cuando las formas per-channel ya están asentadas; 3) que no se usa
    AMP, porque fused_moving_avg_obs_fake_quant exige Float; 4) que ningún búfer queda con forma
    provisional.
    """
    estado = {"ema_rehecha": False}

    def comprobar_que_entrena_el_modelo_preparado(trainer: Any) -> None:
        """Lanza QatError si el modelo que va a entrenar Ultralytics no lleva simulación."""
        simulacion = count_fake_quant_modules(trainer.model)
        report.integrity["fake_quant_at_train_start"] = simulacion
        if simulacion == 0:
            raise QatError(
                "El modelo que Ultralytics va a entrenar no lleva ningún módulo "
                "de cuantización simulada.",
                "Revisa que prepare_for_qat se ejecute antes de construir el Trainer y que su "
                "resultado sea el detector que se le pasa.",
            )
        logger.info(
            "Guarda de integridad: el modelo que entrena lleva %d módulos de " "simulación.",
            simulacion,
        )

        # La EMA se copió con formas provisionales; se desactiva en el primer lote (``enabled =
        # False``, como indica la propia clase) y se reconstruye después.
        if getattr(trainer, "ema", None) is not None:
            trainer.ema.enabled = False

    def comprobar_que_el_afinado_corre_en_fp32(trainer: Any) -> None:
        """Lanza QatError si Ultralytics va a entrenar con AMP. Se mira ``trainer.amp`` ya
        resuelto."""
        precision_mixta = bool(getattr(trainer, "amp", False))
        report.integrity["amp"] = precision_mixta
        if precision_mixta:
            raise QatError(
                "El afinado QAT va a correr con precisión mixta (AMP), y el "
                "núcleo de la cuantización simulada solo acepta FP32.",
                "'torch.fused_moving_avg_obs_fake_quant' exige 'Float' y con AMP las "
                "convoluciones devuelven 'Half'. Quita 'amp' de trainer_kwargs y de su "
                "'extra_args'.",
            )
        logger.info("El afinado QAT corre en FP32, sin AMP.")

    def comprobar_que_ninguna_forma_esta_por_decidir(trainer: Any) -> None:
        """Lanza QatError si queda algún búfer sin forma, mirando el modelo y su EMA por
        separado."""
        from tfm_edge.compression.quantization.qat import count_empty_buffers

        pendientes = count_empty_buffers(trainer.model)
        ema = getattr(trainer, "ema", None)
        if ema is not None:
            pendientes += count_empty_buffers(ema.ema)
        report.integrity["empty_buffers_at_train_start"] = pendientes
        if pendientes:
            raise QatError(
                f"Al arrancar el afinado quedan {pendientes} búferes sin forma "
                "real, entre el modelo y su EMA.",
                "Si el modelo y su EMA fijan formas distintas, ModelEMA.update falla al empezar "
                "la segunda época. Revisa settle_weight_observers.",
            )
        logger.info("Ninguna forma queda pendiente entre el modelo y su EMA.")

    def rehacer_la_ema_tras_el_primer_lote(trainer: Any) -> None:
        """Reconstruye la EMA cuando las formas de la simulación ya están asentadas."""
        if estado["ema_rehecha"] or getattr(trainer, "ema", None) is None:
            return
        estado["ema_rehecha"] = True

        # Se usa type(...) para no importar ModelEMA de Ultralytics y reutilizar la misma clase.
        # Ultralytics la construye sin argumentos.
        trainer.ema = type(trainer.ema)(trainer.model)
        report.integrity["ema_rebuilt_after_first_batch"] = True
        report.integrity["fake_quant_in_ema"] = count_fake_quant_modules(trainer.ema.ema)
        logger.info("EMA reconstruida tras el primer lote, con las formas per-channel asentadas.")

    return [
        ("on_train_start", comprobar_que_entrena_el_modelo_preparado),
        ("on_train_start", comprobar_que_el_afinado_corre_en_fp32),
        ("on_train_start", comprobar_que_ninguna_forma_esta_por_decidir),
        ("on_train_batch_end", rehacer_la_ema_tras_el_primer_lote),
    ]


def make_qat_epoch_callbacks(
    config: QatConfig, report: QatFinetuneReport
) -> list[tuple[str, Callable[[Any], None]]]:
    """Callbacks que congelan los observadores en la época configurada.

    Lanza QatError si se pide congelar BatchNorm, porque en YOLOv8 sin fusionar no hay ConvBn2d.
    """
    if config.freeze_bn_epoch is not None:
        raise QatError(
            f"Se ha pedido congelar la BatchNorm en la época {config.freeze_bn_epoch}, pero en "
            "YOLOv8 sin fusionar no hay módulos ConvBn2d.",
            "Fusiona Conv y BN antes de preparar el QAT o deja freeze_bn_epoch a None.",
        )

    if config.freeze_observers_epoch is None:
        return []

    epoca_objetivo = int(config.freeze_observers_epoch)

    def congelar_observadores(trainer: Any) -> None:
        """Congela los observadores al llegar a la época configurada."""
        if report.frozen.get("observers_frozen") is not None:
            return
        epoca = int(getattr(trainer, "epoch", 0))
        if epoca < epoca_objetivo:
            return
        congelados = freeze_observers(trainer.model)
        report.frozen["observers_epoch"] = epoca
        report.frozen["observers_frozen"] = congelados
        logger.info("Época %d: congelados %d observadores de rango.", epoca, congelados)

    return [("on_train_epoch_start", congelar_observadores)]


def finetune_qat(
    detector: "Detector",
    config: QuantizationConfig,
    data_config: Path | str,
    output_dir: Path | str,
    evaluator: MetricEvaluator | None = None,
    baseline_metrics: dict[str, float] | None = None,
    qat_config: QatConfig | None = None,
    metric_key: str = DEFAULT_METRIC_KEY,
    trainer_kwargs: dict[str, Any] | None = None,
) -> tuple["Detector", QatFinetuneReport]:
    """Prepara un detector para QAT, lo afina y devuelve el detector sin simulación y el informe.

    Los pesos resultantes siguen el mismo camino de materialización INT8 que el baseline. Falla si
    ``trainer_kwargs`` contradice la configuración, si los observadores no ven datos o si cambia el
    número de parámetros.
    """
    from tfm_edge.training.trainer import Trainer

    argumentos = dict(trainer_kwargs or {})
    conflictos = sorted(
        {"epochs", "learning_rate", "optimizer", "callbacks", "amp"} & set(argumentos)
    )
    if conflictos:
        raise QatError(
            f"trainer_kwargs no puede traer {conflictos}: salen de la "
            "configuración de cuantización y de este módulo.",
            "Quítalos de trainer_kwargs y ajústalos en el YAML de configuración.",
        )
    # extra_args se fusiona después en _construir_overrides y podría cambiar amp; se rechaza aquí
    # antes de usar la GPU.
    if "amp" in dict(argumentos.get("extra_args") or {}):
        raise QatError(
            "trainer_kwargs['extra_args'] no puede incluir 'amp': el afinado QAT corre en FP32.",
            "El núcleo de la cuantización simulada solo acepta 'Float'. No es un "
            "parámetro del estudio, es una condición de la técnica.",
        )

    raiz = ensure_dir(output_dir)
    parametros = detector.count_parameters()

    informe = QatFinetuneReport(
        epochs=config.qat_epochs,
        learning_rate=config.qat_learning_rate,
        metric_key=metric_key,
        metrics_baseline=dict(baseline_metrics or {}),
        parameters=parametros,
    )

    configuracion_qat = qat_config or QatConfig(
        excluded_modules=tuple(config.excluded_layers),
        freeze_observers_epoch=config.freeze_observers_epoch,
        freeze_bn_epoch=config.freeze_bn_stats_epoch,
    )

    # Los callbacks se construyen antes de preparar nada, para fallar pronto si la configuración no
    # es posible. Las comprobaciones de invariante van primero.
    callbacks = make_qat_integrity_callbacks(informe) + make_qat_epoch_callbacks(
        configuracion_qat, informe
    )

    preparado, preparacion = prepare_for_qat(detector, configuracion_qat)
    informe.preparation = preparacion.to_dict()

    if config.qat_epochs == 0:
        informe.notes = (
            "qat_epochs=0: no se ha afinado. La métrica de después queda ausente "
            "a propósito, que no es lo mismo que un afinado sin efecto."
        )
        logger.warning("Afinado QAT desactivado (qat_epochs=0).")
        return detector, informe

    entrenando = wrap_prepared_model(preparado, detector)
    entrenador = Trainer(
        detector=entrenando,
        data_config=data_config,
        output_dir=raiz,
        epochs=config.qat_epochs,
        learning_rate=config.qat_learning_rate,
        # Optimizador explícito: con 'auto' Ultralytics usaría AdamW a 4,76e-4, casi cinco veces la
        # tasa configurada.
        optimizer="SGD",
        # AMP desactivado porque fused_moving_avg_obs_fake_quant exige Float. No está en el YAML y
        # no afecta al hash de configuración.
        amp=False,
        callbacks=callbacks,
        **argumentos,
    )
    resultado = entrenador.train(run_name="train")
    informe.cost = resultado.cost.to_dict()

    # Comprueba que los observadores llegaron a ver datos.
    informe.observers_exercised = observers_have_been_exercised(entrenando.model)
    if not informe.observers_exercised:
        raise QatError(
            "Los observadores de cuantización no han visto datos: sus rangos "
            "siguen en el valor inicial tras el afinado.",
            "Comprueba que prepare_qat insertó los observadores en el modelo que se ha entrenado.",
        )

    # Se parte del checkpoint (best.pt guarda la EMA de la mejor época), que es lo que se exporta.
    afinado = _cargar_preparado(resultado.best_checkpoint, detector)
    limpio = wrap_prepared_model(strip_fake_quantization(afinado.model), detector)

    if limpio.count_parameters() != parametros:
        raise QatError(
            f"El modelo afinado tiene {limpio.count_parameters():,} parámetros y "
            f"el de partida tenía {parametros:,}.",
            "QAT no cambia el número de parámetros; si cambia, el modelo se ha reconstruido por "
            "el camino.",
        )

    ruta = limpio.save(
        raiz / STRIPPED_CHECKPOINT_NAME,
        extra={
            "qat": {
                "stage": "after_finetune",
                "preparation": preparacion.to_dict(),
                "frozen": dict(informe.frozen),
            }
        },
    )
    informe.checkpoint_after = str(ruta)
    informe.metrics_after = _medir(evaluator, ruta, "el afinado sin simulación")

    logger.info("%s", informe.summary())
    return limpio, informe


def _cargar_preparado(checkpoint: Path | str, reference: "Detector") -> "Detector":
    """Carga un checkpoint con cuantización simulada sin validar la estructura de Ultralytics."""
    import torch

    contenido = torch.load(checkpoint, map_location="cpu", weights_only=False)
    modelo = contenido.get("ema") or contenido.get("model") if isinstance(contenido, dict) else None
    if modelo is None:
        raise QatError(
            f"'{checkpoint}' no contiene ningún modelo bajo las claves " "'ema' ni 'model'.",
            "Comprueba que el checkpoint lo ha escrito el Trainer del proyecto.",
        )
    return wrap_prepared_model(modelo.float(), reference)
