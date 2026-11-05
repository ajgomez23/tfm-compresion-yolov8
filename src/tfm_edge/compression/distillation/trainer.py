"""Bucle de destilación: maestro congelado y alumno entrenando.

Usa tfm_edge.training.Trainer, que conserva el modelo recibido, así que el alumno puede venir
podado. La destilación se añade envolviendo el criterio de Ultralytics, que se instala en
on_train_start porque Trainer.train() descarta el criterio cacheado y porque así la EMA no lleva
dentro ni el envoltorio ni el maestro. Los adaptadores tienen optimizador propio, que avanza en
on_train_batch_end, porque el de Ultralytics ya está construido. Por eso la destilación corre sin
AMP: no se pueden desescalar sus gradientes con el mismo GradScaler, y en media precisión se pierde
la cola del término de Bernoulli (los logits de fondo llegan a -58).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from tfm_edge.common.config import DistillationConfig
from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir
from tfm_edge.compression.distillation.adapters import FeatureAdapters
from tfm_edge.compression.distillation.feature_hooks import FeatureCollector
from tfm_edge.compression.distillation.losses import DistillationLoss, head_geometry
from tfm_edge.models.detector import Detector
from tfm_edge.models.surgery import unfreeze_all
from tfm_edge.training.trainer import (
    CLOSE_MOSAIC_POR_DEFECTO,
    WARMUP_EPOCHS_POR_DEFECTO,
    Trainer,
    TrainingResult,
)

logger = get_logger(__name__)

#: Cambio relativo mínimo de la norma de un adaptador para considerar que ha aprendido.
ADAPTER_CHANGE_TOLERANCE: float = 1e-4

#: Nombre del volcado con el desglose por época.
COMPONENTS_FILENAME = "distillation_components.csv"


class DistillationError(TFMEdgeError):
    """Error del bucle de destilación."""


@dataclass
class DistillationResult:
    """Resultado de una destilación.

    ``adapter_grad_norms`` es la norma del gradiente en cada paso; en el control (alpha=0) debe ser
    cero. ``adapter_params`` se guarda antes de descartar los adaptadores.
    """

    training: TrainingResult
    components_csv: Path | None = None
    epoch_components: list[dict[str, float]] = field(default_factory=list)
    adapter_initial_norms: dict[str, float] = field(default_factory=dict)
    adapter_final_norms: dict[str, float] = field(default_factory=dict)
    adapter_grad_norms: list[float] = field(default_factory=list)
    teacher_size: str = ""
    steps: int = 0
    adapter_params: int = 0

    @property
    def adapter_grad_was_always_zero(self) -> bool:
        """True si el gradiente de los adaptadores fue cero en todos los pasos. Un nan no pasa la
        comprobación.
        """
        return bool(self.adapter_grad_norms) and not any(
            norma != 0.0 for norma in self.adapter_grad_norms
        )

    @property
    def adapters_learned(self) -> bool:
        """True si la norma de algún adaptador cambió más que el umbral relativo.

        Con un umbral absoluto, un adaptador prácticamente congelado podía darse por bueno.
        """
        if not self.adapter_initial_norms:
            return False
        return any(
            abs(self.adapter_final_norms.get(capa, 0.0) - inicial)
            > ADAPTER_CHANGE_TOLERANCE * max(inicial, 1e-12)
            for capa, inicial in self.adapter_initial_norms.items()
        )


class _CriterionWithDistillation:
    """Envuelve el criterio nativo y le suma los términos de destilación.

    Es una clase y no una función local porque acaba en el ``__dict__`` del modelo, que se
    serializa.
    """

    def __init__(
        self,
        model: Any,
        teacher: Any,
        loss: DistillationLoss,
        pairs: Sequence[tuple[str, str]],
        adapters: FeatureAdapters | None,
        teacher_collector: FeatureCollector | None,
        student_collector: FeatureCollector | None,
        history: list[dict[str, float]],
    ) -> None:
        """Prepara el envoltorio."""
        self._model = model
        self.teacher = teacher
        self.loss = loss
        self.pairs = list(pairs)
        self.adapters = adapters
        self.teacher_collector = teacher_collector
        self.student_collector = student_collector
        self.history = history
        self.base = None

    def __deepcopy__(self, memo: dict) -> "_CriterionWithDistillation":
        """Copia vacía, sin maestro ni activaciones, para cuando Ultralytics rehace la EMA tras un
        NaN.
        """
        vacio = _CriterionWithDistillation.__new__(_CriterionWithDistillation)
        vacio.__dict__.update(
            {
                "_model": None,
                "teacher": None,
                "loss": None,
                "pairs": [],
                "adapters": None,
                "teacher_collector": None,
                "student_collector": None,
                "history": [],
                "base": None,
            }
        )
        memo[id(self)] = vacio
        return vacio

    def __getstate__(self) -> dict:
        """Para pickle tampoco se incluye el maestro."""
        estado = dict(self.__dict__)
        for clave in (
            "_model",
            "teacher",
            "adapters",
            "teacher_collector",
            "student_collector",
            "base",
        ):
            estado[clave] = None
        estado["history"] = []
        return estado

    def __call__(self, preds: Any, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calcula la pérdida de detección y le suma la destilación.

        El vector lleva un componente más; ``loss_items`` se deja igual para no cambiar el registro
        por época de Ultralytics.
        """
        if self.base is None:
            self.base = self._model.init_criterion()
        deteccion, items = self.base(preds, batch)
        tam_lote = int(batch["img"].shape[0])

        cero = torch.zeros((), device=deteccion.device, dtype=deteccion.dtype)
        cls_kd, box_kd, marcados = cero, cero, 0
        feat_kd = cero

        if self.teacher_collector is not None:
            self.teacher_collector.clear()
        with torch.no_grad():
            salida_maestro = self.teacher(batch["img"])
        crudas_maestro = (
            salida_maestro[1] if isinstance(salida_maestro, (tuple, list)) else salida_maestro
        )

        if self.loss.uses_response:
            cls_kd, box_kd, marcados = self.loss.response(crudas_maestro, preds, batch)
        if self.loss.uses_features:
            feats_maestro = self.teacher_collector.require([m for m, _ in self.pairs])
            feats_alumno = self.student_collector.require([a for _, a in self.pairs])
            feat_kd = self.loss.features(
                feats_maestro, self.adapters(feats_alumno), batch, self.pairs
            )

        componentes = self.loss.combine(
            deteccion.sum(), cls_kd, box_kd, feat_kd, marcados, batch_size=tam_lote
        )
        self.history.append(componentes.as_floats())

        if self.teacher_collector is not None:
            self.teacher_collector.clear()
        if self.student_collector is not None:
            self.student_collector.clear()

        # La pérdida nativa ya viene multiplicada por el lote; el término añadido va en la misma
        # escala.
        alfa = self.loss.alpha
        destilacion = componentes.response_cls + componentes.response_box + componentes.features
        return (
            torch.cat([deteccion * (1.0 - alfa), (alfa * destilacion * tam_lote).view(1)]),
            items,
        )


class DistillationTrainer:
    """Destila un maestro grande en un alumno pequeño, entrenando el alumno tal como llega."""

    def __init__(
        self,
        student: Detector,
        teacher: Detector,
        data_config: Path | str,
        output_dir: Path | str,
        config: DistillationConfig,
        resolution: int = 640,
        batch_size: int = 16,
        workers: int = 8,
        device: str = "",
        patience: int = 30,
        optimizer: str = "SGD",
        seed: int | None = None,
        extra_args: dict[str, Any] | None = None,
        amp: bool = False,
        extra_callbacks: Sequence[tuple[str, Any]] | None = None,
        scheduler: str = "linear",
        final_learning_rate_factor: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        close_mosaic_epochs: int = CLOSE_MOSAIC_POR_DEFECTO,
        allow_mosaic_free: bool = False,
        warmup_epochs: float = WARMUP_EPOCHS_POR_DEFECTO,
        deterministic: bool = True,
    ) -> None:
        """Configura la destilación.

        ``config`` es obligatoria porque las parejas de capas dependen del par maestro-alumno.
        ``seed`` se pasa al Trainer, que falla si falta. ``amp`` debe ser False. ``extra_callbacks``
        se registran después de los propios, así un on_train_batch_end externo ve ya el paso de los
        adaptadores.
        """
        if amp:
            raise DistillationError(
                "La destilación no admite AMP.",
                (
                    "La destilación no admite AMP: el optimizador de los adaptadores avanza "
                    "después de scaler.update(), y en media precisión se pierde la cola del "
                    "término de Bernoulli."
                ),
            )
        self.config = config
        self.student = student
        self.teacher = teacher
        self.data_config = Path(data_config)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.batch_size = batch_size
        self.workers = workers
        self.device = device
        self.patience = patience
        self.optimizer = optimizer
        self.seed = seed
        self.extra_args = dict(extra_args or {})
        self.extra_callbacks = list(extra_callbacks or [])
        self.scheduler = scheduler
        self.final_learning_rate_factor = final_learning_rate_factor
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.close_mosaic_epochs = close_mosaic_epochs
        self.allow_mosaic_free = allow_mosaic_free
        self.warmup_epochs = warmup_epochs
        self.deterministic = deterministic

        self.pairs = [(p["teacher"], p["student"]) for p in self.config.feature_pairs]
        self.geometry = head_geometry(student.model)
        self._comprobar_geometrias()

        self.loss = DistillationLoss(
            geometry=self.geometry,
            resolution=self.resolution,
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            class_weight=self.config.response_cls_weight,
            box_weight=self.config.response_box_weight,
            feature_weight=self.config.feature_loss_weight,
            kd_type=self.config.kd_type,
            response_masking=self.config.region_masking,
            feature_masking=self.config.feature_masking,
            teacher_conf_threshold=self.config.teacher_conf_threshold,
            normalize_features=self.config.normalize_features,
        )

        self.adapters: FeatureAdapters | None = None
        self._adapter_optimizer: torch.optim.Optimizer | None = None
        self._teacher_collector: FeatureCollector | None = None
        self._student_collector: FeatureCollector | None = None
        self._step_history: list[dict[str, float]] = []
        self._epoch_components: list[dict[str, float]] = []
        self._epoch_offset = 0
        self._initial_norms: dict[str, float] = {}
        self._adapter_grad_norms: list[float] = []
        self._grad_offset = 0

    # Comprobaciones de invariantes

    def _comprobar_geometrias(self) -> None:
        """Exige que maestro y alumno tengan la misma cabeza (clases y reg_max)."""
        geometria_maestro = head_geometry(self.teacher.model)
        if geometria_maestro != self.geometry:
            raise DistillationError(
                f"Maestro y alumno no comparten geometría de cabeza: "
                f"{geometria_maestro} frente a {self.geometry}.",
                "Los dos modelos tienen que estar entrenados sobre el mismo "
                "dataset y con el mismo reg_max.",
            )

    def _congelar_maestro(self, device: torch.device | str) -> None:
        """Deja el maestro en evaluación, sin gradientes y en el dispositivo del alumno."""
        self.teacher.to(device)
        self.teacher.eval()
        for parametro in self.teacher.model.parameters():
            parametro.requires_grad_(False)

    def _verificar_maestro(self, device: torch.device) -> None:
        """Comprueba en cada época que el maestro sigue en evaluación, sin gradientes y en el
        dispositivo del alumno.

        En modo entrenamiento la BatchNorm del maestro se actualizaría con los lotes del alumno.
        """
        if self.teacher.model.training:
            raise DistillationError(
                "El maestro ha pasado a modo entrenamiento.",
                "Sus estadísticas de BatchNorm se actualizarían con los lotes del "
                "alumno y dejaría de ser una referencia fija.",
            )
        con_gradiente = [n for n, p in self.teacher.model.named_parameters() if p.requires_grad]
        if con_gradiente:
            raise DistillationError(
                f"El maestro tiene {len(con_gradiente)} parámetros con gradiente activo "
                f"(el primero, {con_gradiente[0]}).",
                "El maestro no se entrena: es la referencia.",
            )
        dispositivo_maestro = self.teacher.device
        if dispositivo_maestro.type != device.type:
            raise DistillationError(
                f"El maestro está en {dispositivo_maestro} y el alumno en {device}.",
                "Los dos tienen que estar en el mismo dispositivo.",
            )

    # Callbacks

    def _instalar(self, trainer: Any) -> None:
        """Monta la destilación sobre el entrenador ya arrancado."""
        modelo = trainer.model
        dispositivo = next(modelo.parameters()).device
        self._congelar_maestro(dispositivo)
        self._verificar_maestro(dispositivo)

        if self.loss.uses_features:
            # Los adaptadores se construyen antes de instalar los colectores, porque medir sus
            # canales hace un pase de prueba que quedaría capturado.
            self.adapters = FeatureAdapters.from_detectors(
                self.student, self.teacher, self.pairs, resolution=self.resolution
            )
            self.adapters.to(dispositivo)
            self._teacher_collector = FeatureCollector(owner="maestro")
            self._student_collector = FeatureCollector(owner="alumno")
            self._teacher_collector.attach(self.teacher, [m for m, _ in self.pairs])
            self._student_collector.attach(self.student, [a for _, a in self.pairs])
            self._initial_norms = self.adapters.weight_signature()
            # Adam y no SGD: el gradiente de los adaptadores es del orden de 1e-5 y con Adam el paso
            # no depende de esa escala. Con SGD a 0,01 la norma apenas se movía.
            self._adapter_optimizer = torch.optim.Adam(
                self.adapters.trainable_parameters(), lr=self.config.adapter_learning_rate
            )

        modelo.criterion = _CriterionWithDistillation(
            model=modelo,
            teacher=self.teacher.model,
            loss=self.loss,
            pairs=self.pairs,
            adapters=self.adapters,
            teacher_collector=self._teacher_collector,
            student_collector=self._student_collector,
            history=self._step_history,
        )
        logger.info(
            "Destilación instalada: kd_type=%s, T=%.2f, alpha=%.2f, máscara de "
            "respuesta=%s, %d emparejamientos",
            self.config.kd_type,
            self.config.temperature,
            self.config.alpha,
            self.config.region_masking,
            len(self.pairs),
        )

    def _paso_de_adaptadores(self, trainer: Any) -> None:
        """Avanza el optimizador de los adaptadores al cerrar cada lote."""
        if self._adapter_optimizer is None:
            return
        gradientes = [p.grad for p in self.adapters.parameters() if p.grad is not None]
        self._adapter_grad_norms.append(
            float(torch.stack([g.norm() for g in gradientes]).norm()) if gradientes else 0.0
        )
        self._adapter_optimizer.step()
        self._adapter_optimizer.zero_grad(set_to_none=True)

    def _cerrar_epoca(self, trainer: Any) -> None:
        """Acumula el desglose medio de la época y comprueba los invariantes."""
        self._verificar_maestro(next(trainer.model.parameters()).device)
        pasos = self._step_history[self._epoch_offset :]
        self._epoch_offset = len(self._step_history)
        if not pasos:
            return
        medias = {
            clave: sum(p[clave] for p in pasos) / len(pasos)
            for clave in (
                "response_cls",
                "response_box",
                "features",
                "detection",
                "total",
                "marked_anchors",
            )
        }
        fila = {"epoch": int(trainer.epoch), "steps": len(pasos), **medias}
        if self.adapters is not None:
            fila.update(
                {f"adapter_norm_{c}": v for c, v in self.adapters.weight_signature().items()}
            )
            normas = self._adapter_grad_norms[self._grad_offset :]
            self._grad_offset = len(self._adapter_grad_norms)
            # Cero aquí indica que el término de características no llega a los adaptadores.
            fila["adapter_grad_norm"] = sum(normas) / len(normas) if normas else 0.0
        self._epoch_components.append(fila)
        logger.info(
            "Época %d: detección=%.4f  clase=%.4f  caja=%.4f  caract.=%.4f  total=%.4f  "
            "anclajes=%.1f",
            fila["epoch"],
            medias["detection"],
            medias["response_cls"],
            medias["response_box"],
            medias["features"],
            medias["total"],
            medias["marked_anchors"],
        )

    def _desinstalar(self, trainer: Any) -> None:
        """Retira hooks y criterio al terminar."""
        for colector in (self._teacher_collector, self._student_collector):
            if colector is not None:
                colector.detach()
        if getattr(trainer.model, "criterion", None) is not None:
            trainer.model.criterion = None

    # Ejecución

    def train(self, run_name: str = "distill") -> DistillationResult:
        """Ejecuta la destilación.

        Falla si el alumno llega congelado o si los adaptadores no han aprendido.
        """
        reactivados = unfreeze_all(self.student.model)
        logger.info("Parámetros del alumno con gradiente reactivado: %d", reactivados)

        entrenador = Trainer(
            detector=self.student,
            data_config=self.data_config,
            output_dir=self.output_dir,
            epochs=self.config.epochs,
            batch_size=self.batch_size,
            amp=False,
            learning_rate=self.config.learning_rate,
            optimizer=self.optimizer,
            resume_if_possible=False,
            workers=self.workers,
            device=self.device,
            patience=self.patience,
            seed=self.seed,
            scheduler=self.scheduler,
            final_learning_rate_factor=self.final_learning_rate_factor,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            close_mosaic_epochs=self.close_mosaic_epochs,
            allow_mosaic_free=self.allow_mosaic_free,
            warmup_epochs=self.warmup_epochs,
            deterministic=self.deterministic,
            extra_args={"imgsz": self.resolution, **self.extra_args},
            callbacks=[
                ("on_train_start", self._instalar),
                ("on_train_batch_end", self._paso_de_adaptadores),
                ("on_train_epoch_end", self._cerrar_epoca),
                ("on_train_end", self._desinstalar),
                *self.extra_callbacks,
            ],
        )
        resultado = entrenador.train(run_name=run_name)

        finales = self.adapters.weight_signature() if self.adapters is not None else {}
        salida = DistillationResult(
            training=resultado,
            epoch_components=self._epoch_components,
            adapter_initial_norms=dict(self._initial_norms),
            adapter_final_norms=finales,
            adapter_grad_norms=list(self._adapter_grad_norms),
            teacher_size=self.teacher.size,
            steps=len(self._step_history),
            adapter_params=self.adapters.parameter_count() if self.adapters is not None else 0,
        )
        salida.components_csv = self._volcar_componentes(resultado.run_dir)
        self._comprobar_adaptadores(salida)

        # Los adaptadores no forman parte del alumno final; se descartan tras leer sus normas.
        self.adapters = None
        self._adapter_optimizer = None
        return salida

    def _comprobar_adaptadores(self, salida: DistillationResult) -> None:
        """Comprueba lo que corresponde a los adaptadores según la rama.

        Con ``alpha > 0`` tienen que haber aprendido. Con ``alpha = 0`` (control) tienen que quedar
        intactos y con gradiente cero en todos los pasos; si no, el control está contaminado.
        """
        if not self.loss.uses_features:
            return

        if self.loss.alpha > 0.0:
            if not salida.adapters_learned:
                raise DistillationError(
                    "Los adaptadores no han cambiado en todo el entrenamiento.",
                    (
                        "Los adaptadores no han llegado a ningún optimizador, así que el "
                        "término de características no ha enseñado nada. No uses este resultado."
                    ),
                )
            return

        if salida.adapters_learned:
            raise DistillationError(
                "Con alpha=0 los adaptadores se han movido, y no pueden moverse.",
                (
                    "En el control sin destilar los adaptadores no deberían cambiar; si "
                    "cambian, algo más los está entrenando y la comparación con el piloto deja "
                    "de valer."
                ),
            )
        if not salida.adapter_grad_was_always_zero:
            distintos = [n for n in salida.adapter_grad_norms if n != 0.0]
            raise DistillationError(
                f"Con alpha=0 ha llegado gradiente a los adaptadores en {len(distintos)} de "
                f"{len(salida.adapter_grad_norms)} pasos (el primero, "
                f"{distintos[0] if distintos else '?'}).",
                (
                    "Con alpha=0 la norma tiene que ser cero en todos los pasos; otro valor, "
                    "incluido nan, indica que el control no está aislado."
                ),
            )
        logger.info(
            "Control sin destilar (alpha=0): adaptadores sin cambios y gradiente cero en los %d "
            "pasos. Desglose registrado en '%s'.",
            len(salida.adapter_grad_norms),
            COMPONENTS_FILENAME,
        )

    def _volcar_componentes(self, run_dir: Path) -> Path | None:
        """Escribe el desglose por época. Devuelve None si no hay épocas."""
        if not self._epoch_components:
            return None
        destino = ensure_parent_dir(Path(run_dir) / COMPONENTS_FILENAME)
        columnas = list(self._epoch_components[0])
        with destino.open("w", newline="", encoding="utf-8") as fichero:
            escritor = csv.DictWriter(fichero, fieldnames=columnas)
            escritor.writeheader()
            for fila in self._epoch_components:
                escritor.writerow(fila)
        logger.info("Desglose de la pérdida por época en %s", destino)
        return destino
