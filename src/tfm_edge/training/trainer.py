"""Bucle de entrenamiento con la maquinaria de Ultralytics, sin pasar por ``YOLO.train()``.

``Model.train`` reconstruye el DetectionModel desde el YAML de arquitectura y perdería una poda o
los nodos de QAT, mientras que ``BaseTrainer.setup_model`` respeta un nn.Module ya asignado (no es
API pública; lo comprueba tests/test_trainer_integration.py). Por eso se instancia DetectionTrainer
y se le asigna el modelo.

``BaseTrainer.save_model`` guarda los checkpoints en FP16, así que se sustituye por una versión que
conserva FP32, verificada por dtype al recargar.

El entrenamiento no se puede reanudar: con el modelo ya asignado, setup_model devuelve None y
resume_training no hace nada, y cargar last.pt reconstruiría desde el YAML. Pedir reanudar con un
last.pt presente es un error.
"""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from tfm_edge.common.exceptions import DependencyNotAvailableError, TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir
from tfm_edge.common.registry import utc_timestamp
from tfm_edge.models.detector import (
    Detector,
    drop_cached_loss,
    drop_forward_hooks,
    drop_quantization_config,
)

logger = get_logger(__name__)

#: El modelo del estudio es la EMA, igual para todas las variantes.
STUDY_MODEL: str = "ema"

#: Optimizadores admitidos, los mismos que TrainingConfig.optimizer. Un test comprueba que las dos
#: listas coinciden.
OPTIMIZADORES_ADMITIDOS: frozenset[str] = frozenset({"SGD", "Adam", "AdamW", "auto"})

#: Traducción de TrainingConfig.scheduler a cos_lr. Falta ``"step"`` porque Ultralytics no lo
#: implementa.
COS_LR_POR_PLANIFICADOR: dict[str, bool] = {"linear": False, "cosine": True}

#: close_mosaic por defecto de Ultralytics 8.3.253; se pasa siempre de forma explícita.
CLOSE_MOSAIC_POR_DEFECTO = 10

#: Claves que el Trainer pasa desde sus propios parámetros y que no se admiten en extra_args, que se
#: fusiona después y las sobrescribiría.
#:
#: warmup_epochs por defecto de Ultralytics, igual al de TrainingConfig.warmup_epochs.
WARMUP_EPOCHS_POR_DEFECTO = 3.0

CLAVES_DE_RECETA_GESTIONADAS: frozenset[str] = frozenset(
    {"cos_lr", "lrf", "momentum", "weight_decay", "close_mosaic", "warmup_epochs", "deterministic"}
)


def mosaic_regime(epochs: int, close_mosaic: int) -> tuple[str, str]:
    """Clasifica el régimen de mosaico que aplicará Ultralytics.

    El cierre se dispara con ``epoch == epochs - close_mosaic`` en épocas de 0 a ``epochs - 1``.
    Devuelve ``(régimen, descripción)``: ``"sin_cierre"``, ``"cierra"``, ``"sin_mosaico"`` o
    ``"cierre_inalcanzable"``.
    """
    if close_mosaic <= 0:
        return "sin_cierre", "el mosaico no se cierra: todas las épocas llevan mosaico"
    if close_mosaic < epochs:
        return (
            "cierra",
            f"el mosaico se cierra en la época {epochs - close_mosaic}: las "
            f"{close_mosaic} últimas de {epochs} van sin mosaico",
        )
    if close_mosaic == epochs:
        return (
            "sin_mosaico",
            f"close_mosaic={close_mosaic} con {epochs} épocas: el cierre salta en la "
            f"época 0, antes del primer lote, y NI UN LOTE lleva mosaico",
        )
    return (
        "cierre_inalcanzable",
        f"close_mosaic={close_mosaic} > {epochs} épocas: la época del cierre sería "
        f"negativa y no llega nunca, así que TODAS las épocas llevan mosaico y el "
        f"cierre pedido no ocurre",
    )


class TrainerError(TFMEdgeError):
    """Error del bucle de entrenamiento."""


class PrecisionError(TFMEdgeError):
    """Un checkpoint no está en la precisión requerida."""

    def __init__(self, ruta: Path, detalle: str) -> None:
        super().__init__(
            f"El checkpoint '{ruta}' no está en FP32: {detalle}",
            (
                "Ultralytics guarda en media precisión con .half(). Este proyecto "
                "intercepta save_model para conservar FP32, porque una referencia "
                "'FP32' que ha pasado por FP16 invalida la comparación del "
                "capítulo 5.2.\n"
                "     Si ves este error, la interceptación no se ha aplicado: NO "
                "uses este checkpoint como referencia."
            ),
        )


@dataclass
class TrainingCost:
    """Coste computacional del entrenamiento."""

    wall_time_s: float = 0.0
    gpu_hours: float = 0.0
    device: str = ""
    gpu_name: str = ""
    epochs_completed: int = 0
    peak_gpu_memory_mb: float | str = ""
    execution_context: str = ""
    interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el coste como diccionario serializable."""
        return asdict(self)


@dataclass
class TrainingResult:
    """Resultado de un entrenamiento."""

    best_checkpoint: Path
    last_checkpoint: Path
    cost: TrainingCost
    metrics: dict[str, Any] = field(default_factory=dict)
    effective_batch_size: int = 0
    run_dir: Path = Path()


# Guardado en FP32


def _guardar_checkpoint_fp32(trainer: Any) -> None:
    """Sustituye a ``BaseTrainer.save_model`` conservando FP32 en modelo y optimizador."""
    import io

    from ultralytics.utils.torch_utils import unwrap_model

    buffer = io.BytesIO()
    ema = copy.deepcopy(unwrap_model(trainer.ema.ema)).float() if trainer.ema else None
    modelo = copy.deepcopy(unwrap_model(trainer.model)).float()
    # Al disco va sin la pérdida cacheada, cuyos tensores están ligados a este dispositivo. Se quita
    # de la copia, no del modelo que sigue entrenando.
    for copia in (ema, modelo):
        if copia is not None:
            drop_cached_loss(copia)
            # El qconfig de prepare_qat lleva un objeto local que pickle no serializa; también se
            # quita de la copia.
            drop_quantization_config(copia)
            # Lo mismo con los forward hooks de la destilación.
            drop_forward_hooks(copia)

    contenido = {
        "epoch": trainer.epoch,
        "best_fitness": trainer.best_fitness,
        # Ultralytics guarda 'model': None y deriva de la EMA. Aquí se guardan los dos en FP32.
        "model": modelo,
        "ema": ema,
        "updates": trainer.ema.updates if trainer.ema else 0,
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
        "scaler": trainer.scaler.state_dict() if hasattr(trainer, "scaler") else {},
        "train_args": vars(trainer.args),
        "train_metrics": {**trainer.metrics, **{"fitness": trainer.fitness}},
        "date": utc_timestamp(),
        "tfm_edge": {
            "saved_dtype": "float32",
            "study_model": STUDY_MODEL,
            "note": (
                "Checkpoint del proyecto: guardado en FP32 a proposito. "
                "Ultralytics guardaria en FP16, lo que invalidaria la referencia "
                "del capitulo 5.2."
            ),
        },
    }
    torch.save(contenido, buffer)
    serializado = buffer.getvalue()

    trainer.wdir.mkdir(parents=True, exist_ok=True)
    trainer.last.write_bytes(serializado)
    if trainer.best_fitness == trainer.fitness:
        trainer.best.write_bytes(serializado)
    if trainer.args.save_period > 0 and trainer.epoch % trainer.args.save_period == 0:
        (trainer.wdir / f"epoch{trainer.epoch}.pt").write_bytes(serializado)
    logger.debug("Checkpoint FP32 guardado en la época %d", trainer.epoch)


#: Sufijo de la copia FP32 que se aparta antes de que Ultralytics haga .half().
FP32_BACKUP_SUFFIX = ".fp32bak"


def _envolver_final_eval(trainer: Any) -> Any:
    """Envuelve ``final_eval`` para conservar la versión FP32.

    final_eval llama a strip_optimizer, que reescribe last.pt y best.pt en FP16. Convertir después a
    FP32 no recupera la precisión, así que se aparta una copia antes.
    """
    original = trainer.final_eval

    def final_eval_conservando_fp32() -> Any:
        """Aparta las copias FP32, deja actuar a Ultralytics y las restaura."""
        respaldos: list[tuple[Path, Path]] = []
        for ruta in (Path(trainer.last), Path(trainer.best)):
            if ruta.is_file():
                copia = ruta.with_suffix(ruta.suffix + FP32_BACKUP_SUFFIX)
                copia.write_bytes(ruta.read_bytes())
                respaldos.append((ruta, copia))
        try:
            return original()
        finally:
            # Ultralytics valida sobre best.pt en final_eval, así que se restaura después.
            for ruta, copia in respaldos:
                try:
                    _finalizar_checkpoint_fp32(copia, destino=ruta)
                finally:
                    copia.unlink(missing_ok=True)

    return final_eval_conservando_fp32


def _finalizar_checkpoint_fp32(path: Path, destino: Path | None = None) -> bool:
    """Hace lo mismo que strip_optimizer (la EMA pasa a ser el modelo y se quita el optimizador)
    pero en FP32.
    """
    if not path.is_file():
        return False
    salida = destino or path
    contenido = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(contenido, dict):
        return False

    # El modelo del estudio es la EMA; se promueve conservando la precisión.
    modelo = contenido.get("ema") or contenido.get("model")
    if modelo is None:
        return False
    modelo = modelo.float()
    drop_cached_loss(modelo)  # Tampoco en el checkpoint final.
    for parametro in modelo.parameters():
        parametro.requires_grad = False

    final = {
        "epoch": contenido.get("epoch", -1),
        "best_fitness": contenido.get("best_fitness"),
        "model": modelo,
        "ema": None,
        "updates": contenido.get("updates", 0),
        "optimizer": None,
        "train_args": contenido.get("train_args", {}),
        "train_metrics": contenido.get("train_metrics", {}),
        "date": utc_timestamp(),
        "tfm_edge": {
            **contenido.get("tfm_edge", {}),
            "saved_dtype": "float32",
            "study_model": STUDY_MODEL,
            "finalized": True,
            "note": (
                "Finalizado por el proyecto en FP32. Ultralytics lo habria dejado "
                "en FP16 via strip_optimizer(), que corre en final_eval() DESPUES "
                "del entrenamiento."
            ),
        },
    }
    torch.save(final, salida)
    logger.info("Checkpoint finalizado en FP32 (EMA promovido a modelo): %s", salida)
    return True


def verify_checkpoint_is_fp32(path: Path | str) -> dict[str, Any]:
    """Recarga un checkpoint y comprueba que todos los parámetros son float32."""
    ruta = Path(path)
    if not ruta.is_file():
        raise TrainerError(
            f"No existe el checkpoint '{ruta}'.", "Comprueba la ruta del entrenamiento."
        )
    contenido = torch.load(ruta, map_location="cpu", weights_only=False)
    if not isinstance(contenido, dict):
        raise TrainerError(f"'{ruta}' no tiene el formato esperado.", "Se esperaba un diccionario.")

    informe: dict[str, Any] = {"keys": sorted(contenido), "dtypes": {}}
    problemas: list[str] = []
    for clave in ("model", "ema"):
        modelo = contenido.get(clave)
        if not isinstance(modelo, torch.nn.Module):
            continue
        conteo: dict[str, int] = {}
        for nombre, parametro in modelo.named_parameters():
            conteo[str(parametro.dtype)] = conteo.get(str(parametro.dtype), 0) + 1
            if parametro.dtype != torch.float32:
                problemas.append(f"{clave}.{nombre}: {parametro.dtype}")
        informe["dtypes"][clave] = conteo

    if problemas:
        raise PrecisionError(
            ruta, f"{len(problemas)} parámetros no FP32. Primeros: {problemas[:5]}"
        )
    logger.info("Checkpoint verificado en FP32: %s", ruta)
    return informe


# Trainer


class Trainer:
    """Entrena un detector sin reconstruir el modelo que recibe."""

    def __init__(
        self,
        detector: Detector,
        data_config: Path | str,
        output_dir: Path | str,
        epochs: int = 100,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        amp: bool = True,
        learning_rate: float = 0.01,
        optimizer: str = "SGD",
        checkpoint_every_epochs: int = 1,
        resume_if_possible: bool = True,
        workers: int = 8,
        device: str = "",
        patience: int = 30,
        seed: int | None = None,
        extra_args: dict[str, Any] | None = None,
        callbacks: Sequence[tuple[str, Callable[[Any], None]]] | None = None,
        scheduler: str = "linear",
        final_learning_rate_factor: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        close_mosaic_epochs: int = CLOSE_MOSAIC_POR_DEFECTO,
        allow_mosaic_free: bool = False,
        warmup_epochs: float = WARMUP_EPOCHS_POR_DEFECTO,
        deterministic: bool = True,
    ) -> None:
        """Configura el entrenamiento.

        ``optimizer`` se pasa siempre a Ultralytics: con su valor por defecto ``"auto"``,
        build_optimizer ignora ``learning_rate``. ``seed`` no tiene valor por defecto; si faltara,
        Ultralytics usaría su 0 (siembra con seed + 1 + RANK). ``resume_if_possible`` con un last.pt
        presente lanza error. ``callbacks`` son pares ``(evento, función)`` que se validan contra
        los eventos de Ultralytics. scheduler, final_learning_rate_factor, momentum, weight_decay,
        close_mosaic_epochs, warmup_epochs y deterministic se pasan siempre. ``allow_mosaic_free``
        permite entrenar sin ningún lote con mosaico.
        """
        if gradient_accumulation_steps < 1:
            raise TrainerError(
                f"gradient_accumulation_steps debe ser >= 1 y es {gradient_accumulation_steps}.",
                "Usa 1 para no acumular.",
            )
        self.callbacks = list(callbacks or [])
        self._validar_eventos_de_callback()
        self.detector = detector
        self.data_config = Path(data_config)
        self.output_dir = ensure_dir(output_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.amp = amp
        self.learning_rate = learning_rate
        self.optimizer = self._validar_optimizador(optimizer)
        self.checkpoint_every_epochs = checkpoint_every_epochs
        self.resume_if_possible = resume_if_possible
        self.workers = workers
        self.device = device
        self.patience = patience
        self.seed = seed
        self.extra_args = dict(extra_args or {})
        if scheduler not in COS_LR_POR_PLANIFICADOR:
            raise TrainerError(
                f"El planificador '{scheduler}' no tiene traducción a Ultralytics.",
                f"Usa uno de {sorted(COS_LR_POR_PLANIFICADOR)}. Ultralytics solo ofrece lineal y "
                "coseno (cos_lr).",
            )
        self.scheduler = scheduler
        self.final_learning_rate_factor = final_learning_rate_factor
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.close_mosaic_epochs = close_mosaic_epochs
        self.allow_mosaic_free = allow_mosaic_free
        self.warmup_epochs = warmup_epochs
        self.deterministic = deterministic
        pisadas = sorted(CLAVES_DE_RECETA_GESTIONADAS & set(self.extra_args))
        if pisadas:
            raise TrainerError(
                f"extra_args trae {pisadas}, que el Trainer ya pasa desde sus "
                "propios parámetros.",
                "Pásalas como scheduler, final_learning_rate_factor, momentum, weight_decay o "
                "close_mosaic_epochs, no por extra_args.",
            )

    @staticmethod
    def _validar_optimizador(nombre: str) -> str:
        """Comprueba que el optimizador es uno de los admitidos por el esquema."""
        if nombre not in OPTIMIZADORES_ADMITIDOS:
            raise TrainerError(
                f"Optimizador '{nombre}' desconocido.",
                f"Usa uno de {sorted(OPTIMIZADORES_ADMITIDOS)}. Distingue "
                "mayúsculas: 'SGD', no 'sgd'.",
            )
        return nombre

    def _callback_de_verificacion_del_optimizador(
        self,
    ) -> tuple[str, Callable[[Any], None]]:
        """Comprobación del optimizador que Ultralytics ha montado de verdad.

        Se ejecuta en on_train_start, cuando trainer.optimizer ya existe, y compara la tasa real con
        la pedida.
        """

        def verificar(trainer: Any) -> None:
            """Contrasta el optimizador montado con el que se pidió."""
            optimizador = getattr(trainer, "optimizer", None)
            if optimizador is None:  # pragma: no cover - Ultralytics siempre lo monta
                raise TrainerError(
                    "Ultralytics ha llegado a 'on_train_start' sin optimizador.",
                    "Es un cambio de su ciclo de vida interno: revisa la versión "
                    "anclada antes de seguir midiendo.",
                )

            real = type(optimizador).__name__
            lr_real = float(optimizador.param_groups[0]["lr"])
            logger.info(
                "Optimizador montado por Ultralytics: %s con lr=%.6g (pedido: %s con lr=%.6g)",
                real,
                lr_real,
                self.optimizer,
                self.learning_rate,
            )

            if self.optimizer == "auto":
                # Solo se llega aquí si se pidió 'auto'; no se falla, pero se deja constancia.
                logger.warning(
                    "optimizer='auto': Ultralytics ignora learning_rate=%.6g y entrena %s con "
                    "lr=%.6g.",
                    self.learning_rate,
                    real,
                    lr_real,
                )
                return

            if real.lower() != self.optimizer.lower():
                raise TrainerError(
                    f"Se pidió el optimizador '{self.optimizer}' y Ultralytics ha "
                    f"montado '{real}'.",
                    "El argumento no ha llegado al optimizador de Ultralytics.",
                )
            if abs(lr_real - self.learning_rate) > 1e-12:
                raise TrainerError(
                    f"Se pidió lr={self.learning_rate:.6g} y el optimizador "
                    f"arranca con lr={lr_real:.6g}.",
                    "La tasa configurada no ha llegado al optimizador. Comprueba que no se "
                    "sobrescriben 'lr0' ni 'optimizer' por extra_args.",
                )

        return ("on_train_start", verificar)

    def _callback_de_verificacion_de_la_semilla(
        self,
    ) -> tuple[str, Callable[[Any], None]]:
        """Comprobación de la semilla que Ultralytics ha resuelto de verdad.

        Se mira trainer.args, la configuración ya resuelta, porque si la semilla no llega todos los
        entrenamientos con semillas distintas salen idénticos.
        """

        def verificar(trainer: Any) -> None:
            """Contrasta la semilla resuelta por Ultralytics con la pedida."""
            argumentos = getattr(trainer, "args", None)
            real = getattr(argumentos, "seed", None) if argumentos is not None else None
            if real is None:  # pragma: no cover - Ultralytics siempre la resuelve
                raise TrainerError(
                    "Ultralytics ha llegado a 'on_train_start' sin 'args.seed'.",
                    "Revisa la versión anclada de Ultralytics.",
                )
            logger.info(
                "Semilla resuelta por Ultralytics: %s (pedida: %s; siembra con %s)",
                real,
                self.seed,
                int(real) + 1,
            )
            if int(real) != int(self.seed):
                raise TrainerError(
                    f"Se pidio seed={self.seed} y Ultralytics entrena con seed={real}.",
                    "La semilla no ha llegado. Comprueba que no se sobrescribe 'seed' por "
                    "extra_args; sin ella, ejecuciones con semillas distintas dan pesos "
                    "idénticos.",
                )

        return ("on_train_start", verificar)

    def _callback_de_verificacion_de_la_receta(
        self,
    ) -> tuple[str, Callable[[Any], None]]:
        """Comprobación de scheduler, lrf, momentum, weight_decay y close_mosaic tal como los ha
        resuelto Ultralytics.
        """
        pedida: dict[str, Any] = {
            "cos_lr": COS_LR_POR_PLANIFICADOR[self.scheduler],
            "lrf": self.final_learning_rate_factor,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "close_mosaic": self.close_mosaic_epochs,
        }

        def verificar(trainer: Any) -> None:
            """Contrasta la receta resuelta por Ultralytics con la pedida."""
            argumentos = getattr(trainer, "args", None)
            discrepancias = []
            for clave, valor in pedida.items():
                real = getattr(argumentos, clave, None) if argumentos is not None else None
                if isinstance(valor, bool):
                    coincide = real is not None and bool(real) == valor
                else:
                    coincide = real is not None and abs(float(real) - float(valor)) <= 1e-12
                if not coincide:
                    discrepancias.append(f"{clave}: pedido {valor!r}, resuelto {real!r}")
            if discrepancias:
                raise TrainerError(
                    "La receta resuelta por Ultralytics no es la pedida: "
                    + "; ".join(discrepancias)
                    + ".",
                    "El argumento no ha llegado o se ha sobrescrito.",
                )

        return ("on_train_start", verificar)

    @property
    def effective_batch_size(self) -> int:
        """Tamaño de lote efectivo, contando la acumulación de gradientes."""
        return self.batch_size * self.gradient_accumulation_steps

    def _construir_overrides(self, run_name: str) -> dict[str, Any]:
        """Traduce la configuración del proyecto a argumentos de Ultralytics."""
        if self.seed is None:
            raise TrainerError(
                "El entrenamiento no tiene semilla y Ultralytics usaría la suya.",
                "Pasa seed=... al Trainer. Sin ella Ultralytics siembra con su 0 por defecto y "
                "todas las ejecuciones salen iguales.",
            )

        overrides: dict[str, Any] = {
            "model": f"yolov8{self.detector.size}.yaml",
            # seed se pasa siempre: BaseTrainer.__init__ llama a init_seeds(self.args.seed + 1 +
            # RANK) con seed=0 por defecto después de set_global_seed().
            "seed": self.seed,
            "data": str(self.data_config),
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.detector.input_resolution,
            "lr0": self.learning_rate,
            # optimizer se pasa siempre: con 'auto' Ultralytics descarta lr0.
            "optimizer": self.optimizer,
            # Estas cinco se pasan siempre, aunque coincidan con el valor por defecto de
            # Ultralytics.
            "cos_lr": COS_LR_POR_PLANIFICADOR[self.scheduler],
            "lrf": self.final_learning_rate_factor,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "close_mosaic": self.close_mosaic_epochs,
            # warmup_epochs también se pasa siempre.
            "warmup_epochs": self.warmup_epochs,
            # deterministic gobierna init_seeds() de Ultralytics.
            "deterministic": self.deterministic,
            "amp": self.amp,
            "workers": self.workers,
            "patience": self.patience,
            "project": str(self.output_dir),
            "name": run_name,
            "exist_ok": True,
            "save_period": self.checkpoint_every_epochs,
            "verbose": True,
            # Ultralytics acumula gradientes hasta alcanzar 'nbs' (nominal batch size).
            "nbs": self.effective_batch_size,
        }
        if self.device:
            overrides["device"] = self.device
        overrides.update(self.extra_args)
        self._comprobar_regimen_de_mosaico(overrides)
        logger.info(
            "Receta efectiva: cos_lr=%s (%s), lrf=%s, momentum=%s, weight_decay=%s, "
            "close_mosaic=%s, workers=%s",
            overrides["cos_lr"],
            self.scheduler,
            overrides["lrf"],
            overrides["momentum"],
            overrides["weight_decay"],
            overrides["close_mosaic"],
            overrides["workers"],
        )
        return overrides

    def _comprobar_regimen_de_mosaico(self, overrides: dict[str, Any]) -> None:
        """Comprueba el régimen de mosaico sobre los argumentos ya fusionados.

        Está aquí porque todos los caminos de entrenamiento pasan por este punto. Falla si ningún
        lote va a llevar mosaico y no se ha pedido ``allow_mosaic_free``.
        """
        regimen, descripcion = mosaic_regime(
            int(overrides["epochs"]), int(overrides["close_mosaic"])
        )
        if regimen == "sin_mosaico":
            if not self.allow_mosaic_free:
                raise TrainerError(
                    f"Régimen de mosaico no pedido: {descripcion}.",
                    "Esta receta no es comparable con la de los baselines. Si es la que "
                    "quieres, pídela con allow_mosaic_free=True; si no, baja "
                    "close_mosaic_epochs por debajo de las épocas.",
                )
            logger.warning("Régimen de mosaico pedido explícitamente: %s.", descripcion)
        elif regimen == "cierre_inalcanzable":
            logger.warning("Régimen de mosaico: %s.", descripcion)
        else:
            logger.info("Régimen de mosaico: %s.", descripcion)

    def train(self, run_name: str = "train") -> TrainingResult:
        """Entrena el detector y devuelve rutas, coste y métricas."""
        try:
            from ultralytics.models.yolo.detect import DetectionTrainer
        except ImportError as error:
            raise DependencyNotAvailableError("ultralytics", "entrenar un detector") from error

        from tfm_edge.common.environment import detect_execution_context

        self._rechazar_reanudacion_imposible(run_name)

        canales_antes = self.detector.channels_per_conv()
        params_antes = self.detector.count_parameters()

        overrides = self._construir_overrides(run_name)
        logger.info(
            "Entrenando yolov8%s: %d épocas, lote %d x %d acumulaciones = %d efectivo, AMP=%s",
            self.detector.size,
            self.epochs,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.effective_batch_size,
            self.amp,
        )

        trainer = DetectionTrainer(overrides=overrides)

        # Se quita la pérdida que Ultralytics cachea en el modelo, con tensores ligados al
        # dispositivo donde se creó. Puede venir de un checkpoint o de un reajuste anterior.
        drop_cached_loss(self.detector.model)

        # setup_model() no reconstruye si self.model ya es un nn.Module, así que asignarlo aquí
        # conserva la poda o la QAT.
        trainer.model = self.detector.model

        # La comprobación del optimizador va primero.
        evento_guarda, funcion_guarda = self._callback_de_verificacion_del_optimizador()
        trainer.add_callback(evento_guarda, funcion_guarda)
        evento_semilla, funcion_semilla = self._callback_de_verificacion_de_la_semilla()
        trainer.add_callback(evento_semilla, funcion_semilla)
        evento_receta, funcion_receta = self._callback_de_verificacion_de_la_receta()
        trainer.add_callback(evento_receta, funcion_receta)

        for evento, funcion in self.callbacks:
            trainer.add_callback(evento, funcion)
        if self.callbacks:
            logger.info(
                "Registrados %d callbacks: %s",
                len(self.callbacks),
                ", ".join(sorted({evento for evento, _ in self.callbacks})),
            )

        # Guardado en FP32.
        trainer.save_model = lambda: _guardar_checkpoint_fp32(trainer)
        # final_eval() llama a strip_optimizer(), que pasa a FP16 los ficheros escritos; se aparta
        # antes una copia FP32.
        trainer.final_eval = _envolver_final_eval(trainer)

        contexto = detect_execution_context()
        inicio = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        try:
            trainer.train()
        except Exception as error:
            raise TrainerError(
                f"El entrenamiento ha fallado: {type(error).__name__}: {error}",
                "Revisa el log de la ejecución. Si es un error de memoria, reduce batch_size y "
                "aumenta gradient_accumulation_steps para mantener el lote efectivo.",
            ) from error

        transcurrido = time.perf_counter() - inicio
        coste = TrainingCost(
            wall_time_s=round(transcurrido, 2),
            gpu_hours=round(transcurrido / 3600.0, 4) if torch.cuda.is_available() else 0.0,
            device=str(getattr(trainer, "device", self.device or "desconocido")),
            gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            epochs_completed=int(getattr(trainer, "epoch", 0)) + 1,
            peak_gpu_memory_mb=(
                round(torch.cuda.max_memory_reserved() / 2**20, 2)
                if torch.cuda.is_available()
                else ""
            ),
            execution_context=contexto,
        )

        mejor = Path(trainer.best)
        ultimo = Path(trainer.last)

        # Comprobación por dtype del checkpoint guardado.
        for ruta in (ultimo, mejor):
            if ruta.is_file():
                verify_checkpoint_is_fp32(ruta)

        # La poda debe mantenerse tras el entrenamiento.
        self._verificar_estructura_conservada(canales_antes, params_antes)

        self._volcar_coste(coste, Path(trainer.save_dir))
        logger.info(
            "Entrenamiento terminado en %.1f s (%.3f horas de GPU)",
            coste.wall_time_s,
            coste.gpu_hours,
        )
        return TrainingResult(
            best_checkpoint=mejor,
            last_checkpoint=ultimo,
            cost=coste,
            metrics=dict(getattr(trainer, "metrics", {}) or {}),
            effective_batch_size=self.effective_batch_size,
            run_dir=Path(trainer.save_dir),
        )

    def _validar_eventos_de_callback(self) -> None:
        """Comprueba que los eventos de ``callbacks`` existen en Ultralytics.

        add_callback acepta un nombre inexistente sin avisar y la función nunca se llamaría.
        """
        if not self.callbacks:
            return
        try:
            from ultralytics.utils.callbacks.base import default_callbacks
        except ImportError as error:  # pragma: no cover - depende del entorno
            raise DependencyNotAvailableError(
                "ultralytics", "registrar callbacks de entrenamiento"
            ) from error

        conocidos = set(default_callbacks)
        desconocidos = sorted({evento for evento, _ in self.callbacks} - conocidos)
        if desconocidos:
            raise TrainerError(
                f"Eventos de callback que Ultralytics no conoce: {desconocidos}.",
                f"Un evento inexistente no falla al registrarlo: simplemente no "
                f"se llama nunca. Los disponibles son: {sorted(conocidos)}.",
            )

    def _rechazar_reanudacion_imposible(self, run_name: str) -> None:
        """Falla si se ha pedido reanudar y existe un last.pt, porque se empezaría de cero
        sobre él."""
        if not self.resume_if_possible:
            return
        ultimo = self.output_dir / run_name / "weights" / "last.pt"
        if not ultimo.is_file():
            return
        raise TrainerError(
            f"Se ha pedido reanudar y existe '{ultimo}', pero este entrenador no puede reanudar "
            "porque conserva el modelo en memoria; empezaría de cero sobre esa ejecución.",
            "Para rehacer la ejecución pasa resume_if_possible=False (en los scripts, "
            f"'--no-resume') o mueve '{ultimo.parent.parent}' a otro nombre.",
        )

    def _verificar_estructura_conservada(
        self, canales_antes: dict[str, tuple[int, int]], params_antes: int
    ) -> None:
        """Comprueba que el entrenamiento no ha reconstruido el modelo (canales y parámetros
        iguales).
        """
        canales_despues = self.detector.channels_per_conv()
        params_despues = self.detector.count_parameters()

        if params_despues != params_antes:
            raise TrainerError(
                f"El número de parámetros ha cambiado durante el entrenamiento: {params_antes:,} "
                f"antes y {params_despues:,} después.",
                "El modelo se ha reconstruido desde el YAML y se ha perdido la modificación "
                "previa. Comprueba que el trainer recibe el modelo por asignación directa.",
            )
        distintos = [n for n, c in canales_despues.items() if canales_antes.get(n) != c]
        if distintos:
            raise TrainerError(
                f"{len(distintos)} convoluciones han cambiado de canales durante el "
                f"entrenamiento. Primeras: {distintos[:5]}",
                "El modelo se ha reconstruido y la poda se ha perdido.",
            )
        logger.info(
            "Estructura conservada tras entrenar: %d parámetros, %d convoluciones intactas",
            params_despues,
            len(canales_despues),
        )

    def _volcar_coste(self, coste: TrainingCost, run_dir: Path) -> Path:
        """Escribe el coste del entrenamiento en disco."""
        destino = ensure_parent_dir(run_dir / "training_cost.json")
        destino.write_text(
            json.dumps(coste.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("Coste de entrenamiento volcado en %s", destino)
        return destino
