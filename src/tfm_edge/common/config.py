"""Carga, validación, fusión y hash de las configuraciones YAML.

Una configuración puede heredar de otras con ``_base_`` (fusión en profundidad), se valida con
Pydantic v2 y ``extra="forbid"``, admite overrides con notación de punto (``--override
training.epochs=150``) y tiene un hash estable que forma parte del experiment_id. El hash solo
incluye dataset, model, training y la sección de compresión aplicada: medir el mismo modelo en otro
dispositivo no debe cambiar su identificador.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, Type, TypeVar

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from tfm_edge.common.exceptions import (
    ConfigFileNotFoundError,
    ConfigInheritanceError,
    ConfigOverrideError,
    ConfigSchemaError,
)
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir

logger = get_logger(__name__)

BASE_KEY = "_base_"

#: Secciones que definen la identidad del modelo y entran en el hash.
EXPERIMENT_IDENTITY_SECTIONS: tuple[str, ...] = (
    "seed",
    "model",
    "dataset",
    "training",
    "quantization",
    "pruning",
    "distillation",
)

SHORT_HASH_LENGTH = 8

T = TypeVar("T", bound=BaseModel)


# Esquemas


#: Opciones que el esquema declara pero ningún código lee, con lo que hace el código en su lugar. No
#: se quitan porque cambiaría el experiment_id de lo ya inscrito; pedir un valor distinto del de por
#: defecto lanza un error que lo explica.
OPCIONES_SIN_LECTOR: dict[str, dict[str, str]] = {
    "DatasetConfig": {
        "class_names_file": (
            "el mapeo de índice a nombre se carga con load_class_names(), al que cada "
            "guion le pasa la ruta que quiere"
        ),
        "split_seed": "data/splits.py siembra con su propio parámetro",
        "stratified": "data/splits.py decide el estratificado por su cuenta",
        "smoke_test_images": "cada guion trae escrito su propio subconjunto de humo",
    },
    "TrainingConfig": {
        "smoke_test_epochs": "04_train_baselines.py fija 2 épocas escritas en el código",
    },
    "QuantizationConfig": {
        "bias_correction": (
            "NO la aplica ningún módulo de compression/quantization/. Ponerla a True "
            "no cambiaría el modelo y sí dejaría escrito en 'effective_config' que se "
            "aplicó una técnica que no se aplicó. Implementarla de verdad y "
            "retirarla de aquí es la salida buena"
        ),
        "cross_layer_equalization": (
            "NO la aplica ningún módulo de compression/quantization/, igual que "
            "bias_correction, y con la misma salida buena"
        ),
        "scheme": "export/quantize.py usa su propio symmetric_weights, fijado a True",
        "weight_granularity": "export/quantize.py usa su propio per_channel, fijado a True",
        "activation_granularity": "las activaciones van por tensor, sin opción que lo gobierne",
        "calibration_percentile": "ningún observador lee un percentil de aquí",
    },
    "ExportConfig": {
        "precisions": "08_export.py toma las precisiones de su --precision",
        "dynamic_axes": "el exportador a ONNX no consulta esta clave",
        "validate_numerical_equivalence": (
            "la equivalencia se comprueba SIEMPRE; no hay forma de desactivarla, que "
            "es la parte segura de esta mentira"
        ),
        "tolerance_fp32": "equivalence.py usa BOX_RELATIVE_TOLERANCE y SCORE_ABSOLUTE_TOLERANCE",
        "tolerance_fp16": "equivalence.py usa sus propias constantes, iguales para las tres",
        "tolerance_int8": "equivalence.py usa sus propias constantes, iguales para las tres",
    },
    "EvaluationConfig": {
        "compute_error_analysis": "la descomposición de errores se calcula siempre",
        "smoke_test_images": "cada guion trae escrito su propio subconjunto de humo",
    },
    "BenchmarkConfig": {
        "throughput_window_s": (
            "protocol.DEFAULT_THROUGHPUT_WINDOW_S vale 10,0 s, no los 30,0 que dice "
            "aquí, y es la que llevan las 50 medidas del registro"
        ),
        "memory_sample_interval_s": "el muestreo de memoria lleva su intervalo escrito",
        "max_system_load_percent": "07_benchmark.py comprueba la carga con su propio umbral",
        "record_temperature": "la temperatura se registra siempre que se puede leer",
        "smoke_test_warmup_iters": "07_benchmark.py anuncia y usa 5/20/1, escrito en el guion",
        "smoke_test_measure_iters": "07_benchmark.py anuncia y usa 5/20/1, escrito en el guion",
        "smoke_test_num_repeats": (
            "07_benchmark.py anuncia y usa 5/20/1: una repetición, no las dos que dice aquí"
        ),
    },
    "ProjectConfig": {
        "project_name": "las rutas salen de common/paths.py y los logs del nombre del guion",
    },
}


def _rechazar_opciones_sin_lector(modelo: BaseModel) -> None:
    """Lanza ValueError si una opción sin lector trae un valor distinto del suyo por defecto.

    Se compara con el valor por defecto declarado en el modelo.
    """
    silenciosas = OPCIONES_SIN_LECTOR.get(type(modelo).__name__, {})
    pedidas = []
    for campo, en_su_lugar in silenciosas.items():
        defecto = type(modelo).model_fields[campo].get_default(call_default_factory=True)
        valor = getattr(modelo, campo)
        if valor != defecto:
            pedidas.append(f"  {campo}={valor!r} (por defecto {defecto!r}): {en_su_lugar}")
    if pedidas:
        raise ValueError(
            "Opciones declaradas en el esquema que NO lee ningún módulo, y que se "
            "han pedido con un valor distinto del suyo por defecto:\n"
            + "\n".join(pedidas)
            + "\nQué hacer: dejarlas como están, que es lo que el código hace de "
            "verdad; o implementarlas y retirarlas de OPCIONES_SIN_LECTOR. Lo que "
            "no puede ser es que la configuración efectiva afirme lo que no pasó "
            "."
        )


class StrictModel(BaseModel):
    """Modelo base de las configuraciones; una clave desconocida en el YAML es un error."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class SplitRatios(StrictModel):
    """Proporciones de las particiones del dataset."""

    train: float = Field(default=0.70, gt=0.0, lt=1.0)
    val: float = Field(default=0.15, gt=0.0, lt=1.0)
    test: float = Field(default=0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _validar_suma(self) -> "SplitRatios":
        """Comprueba que las proporciones suman uno."""
        suma = self.train + self.val + self.test
        if abs(suma - 1.0) > 1e-6:
            raise ValueError(
                f"Las proporciones de las particiones deben sumar 1.0, y suman {suma:.6f}."
            )
        return self


class DatasetConfig(StrictModel):
    """Configuración del dataset.

    ``split_seed`` es independiente de la semilla global para que el particionado no cambie al
    cambiar la de entrenamiento.
    """

    name: str = "sh17"
    num_classes: int = Field(default=17, gt=0)
    class_names_file: str = "configs/dataset/sh17_classes.json"
    split_ratios: SplitRatios = Field(default_factory=SplitRatios)
    stratified: bool = True
    split_seed: int = Field(default=42, ge=0)
    smoke_test_images: int = Field(default=64, gt=0)

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "DatasetConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


class ModelConfig(StrictModel):
    """Configuración del detector."""

    family: Literal["yolov8"] = "yolov8"
    size: Literal["n", "s", "m", "l"] = "s"
    pretrained: bool = True
    input_resolution: int = Field(default=640, gt=0)

    @field_validator("input_resolution")
    @classmethod
    def _validar_multiplo_de_32(cls, valor: int) -> int:
        """Comprueba que la resolución es múltiplo de 32."""
        if valor % 32 != 0:
            raise ValueError(
                f"La resolución de entrada debe ser múltiplo de 32 por el factor de "
                f"submuestreo de la red, y se recibió {valor}."
            )
        return valor


class TrainingConfig(StrictModel):
    """Configuración de entrenamiento con nombres propios del proyecto; el Trainer los traduce a
    Ultralytics.

    ``amp`` forma parte de la identidad porque cambia la aritmética. El lote efectivo es
    ``batch_size * gradient_accumulation_steps``.
    """

    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    final_learning_rate_factor: float = Field(default=0.01, gt=0.0)
    momentum: float = Field(default=0.937, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    optimizer: Literal["SGD", "Adam", "AdamW", "auto"] = "SGD"
    # "linear" es el planificador con el que han corrido todas las campañas. "step" se admite aquí
    # pero el Trainer lo rechaza porque Ultralytics no lo ofrece.
    scheduler: Literal["cosine", "step", "linear"] = "linear"
    patience: int = Field(default=30, ge=0)
    # 2 es el valor con el que corrieron todas las campañas.
    workers: int = Field(default=2, ge=0)
    amp: bool = True
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    # Checkpoints frecuentes porque las sesiones de Colab se cortan.
    checkpoint_every_epochs: int = Field(default=1, gt=0)
    resume_if_possible: bool = True
    smoke_test_epochs: int = Field(default=2, gt=0)

    @property
    def effective_batch_size(self) -> int:
        """Lote efectivo, ``batch_size * gradient_accumulation_steps``; es el que se compara entre
        experimentos.
        """
        return self.batch_size * self.gradient_accumulation_steps

    def trainer_recipe_kwargs(self) -> dict[str, Any]:
        """Argumentos del Trainer que salen de esta sección (scheduler, final_learning_rate_factor,
        momentum, weight_decay y workers). Lo usan poda, QAT y destilación.
        """
        return {
            "scheduler": self.scheduler,
            "final_learning_rate_factor": self.final_learning_rate_factor,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "workers": self.workers,
        }

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "TrainingConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


class QuantizationConfig(StrictModel):
    """Configuración de cuantización.

    ``bias_correction`` y ``cross_layer_equalization`` están declaradas pero no implementadas y solo
    admiten False.
    """

    mode: Literal["ptq_static", "ptq_dynamic", "qat"] = "ptq_static"
    precision: Literal["int8", "fp16"] = "int8"
    scheme: Literal["symmetric", "asymmetric"] = "symmetric"
    weight_granularity: Literal["per_tensor", "per_channel"] = "per_channel"
    activation_granularity: Literal["per_tensor", "per_channel"] = "per_tensor"
    calibration_method: Literal["minmax", "percentile", "mse", "entropy"] = "minmax"
    calibration_percentile: float = Field(default=99.99, gt=0.0, le=100.0)
    calibration_size: int = Field(default=256, gt=0)
    excluded_layers: list[str] = Field(default_factory=list)
    # Ningún módulo lee estas dos banderas; con True el volcado diría que se aplicaron sin ser
    # cierto. El validador lo impide.
    bias_correction: bool = False
    cross_layer_equalization: bool = False
    # Cero es válido: prepara el modelo y lo comprueba sin afinar; finetune_qat deja la métrica
    # posterior vacía.
    qat_epochs: int = Field(default=10, ge=0)
    qat_learning_rate: float = Field(default=0.0001, gt=0.0)
    freeze_observers_epoch: int | None = None
    freeze_bn_stats_epoch: int | None = None

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "QuantizationConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


class PruningConfig(StrictModel):
    """Configuración de poda estructurada.

    ``random`` es el criterio de control. La cabeza del detector está en ``protected_layers`` por
    defecto. ``iterative_steps=1`` es poda en una etapa.
    """

    criterion: Literal["l1", "l2", "bn_gamma", "taylor", "random"] = "l1"
    ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    distribution: Literal["global", "uniform", "per_block"] = "global"
    block_ratios: dict[str, float] = Field(default_factory=dict)
    min_channels: int = Field(default=8, gt=0)
    protected_layers: list[str] = Field(default_factory=lambda: ["model.22"])
    iterative_steps: int = Field(default=1, gt=0)
    finetune_epochs: int = Field(default=30, ge=0)
    finetune_learning_rate: float = Field(default=0.001, gt=0.0)
    finetune_epochs_per_step: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _validar_reparto_por_bloques(self) -> "PruningConfig":
        """Comprueba que ``block_ratios`` está definido solo con el reparto ``per_block``."""
        if self.distribution == "per_block" and not self.block_ratios:
            raise ValueError(
                "Con distribution='per_block' hay que definir 'block_ratios' "
                "(por ejemplo: {backbone: 0.5, neck: 0.3, head: 0.0})."
            )
        if self.distribution != "per_block" and self.block_ratios:
            raise ValueError(
                f"Se han definido 'block_ratios' pero distribution es "
                f"'{self.distribution}'. Esos ratios se ignorarían en silencio."
            )
        return self


class DistillationConfig(StrictModel):
    """Configuración de destilación de conocimiento.

    ``temperature`` es 2,0 por defecto, medida sobre los logits del maestro l: con 4,0 la
    distribución queda plana y con 1,0 es casi una etiqueta dura. ``region_masking`` es la máscara
    del término de respuesta (intersección de cajas y confianza del maestro) y ``feature_masking``
    la del de características (cajas). ``close_mosaic_epochs=None`` significa no declarado: el
    Trainer usa 10 y la clave no se serializa.
    """

    teacher_size: Literal["n", "s", "m", "l"] = "l"
    teacher_weights: str | None = None
    kd_type: Literal["logit", "feature", "combined"] = "combined"
    temperature: float = Field(default=2.0, gt=0.0)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    response_cls_weight: float = Field(default=0.1431, ge=0.0)
    response_box_weight: float = Field(default=0.0984, ge=0.0)
    feature_loss_weight: float = Field(default=0.0704, ge=0.0)
    feature_pairs: list[dict[str, str]] = Field(default_factory=list)
    region_masking: Literal["none", "gt_boxes", "teacher_attention", "gt_boxes_and_teacher"] = (
        "gt_boxes_and_teacher"
    )
    feature_masking: Literal["none", "gt_boxes", "teacher_attention", "gt_boxes_and_teacher"] = (
        "gt_boxes"
    )
    teacher_conf_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    normalize_features: bool = True
    adapter_learning_rate: float = Field(default=0.001, gt=0.0)
    student_from_pretrained: bool = True
    epochs: int = Field(default=50, gt=0)
    learning_rate: float = Field(default=0.005, gt=0.0)
    close_mosaic_epochs: int | None = Field(default=None, ge=0)

    @model_serializer(mode="wrap")
    def _omitir_el_cierre_no_declarado(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serializa sin ``close_mosaic_epochs`` cuando vale None.

        Así no cambia el hash de los experimentos que no la declaran, que se hashean con
        model_dump() completo.
        """
        datos = handler(self)
        if self.close_mosaic_epochs is None and isinstance(datos, dict):
            datos.pop("close_mosaic_epochs", None)
        return datos

    @model_validator(mode="after")
    def _validar_cierre_del_mosaico(self) -> "DistillationConfig":
        """Lanza ValueError si ``close_mosaic_epochs`` supera a ``epochs``, porque el cierre nunca
        ocurriría.
        """
        if self.close_mosaic_epochs is not None and self.close_mosaic_epochs > self.epochs:
            raise ValueError(
                f"close_mosaic_epochs={self.close_mosaic_epochs} supera a epochs={self.epochs}. "
                "Para no usar mosaico, iguálalas."
            )
        return self

    @model_validator(mode="after")
    def _validar_pares_de_caracteristicas(self) -> "DistillationConfig":
        """Comprueba que hay pares de capas si se destilan características."""
        if self.kd_type in {"feature", "combined"} and not self.feature_pairs:
            raise ValueError(
                f"kd_type='{self.kd_type}' necesita 'feature_pairs' con los pares "
                "de capas maestro-alumno a emparejar."
            )
        if self.kd_type == "logit" and self.feature_pairs:
            raise ValueError(
                "Se han definido 'feature_pairs' pero kd_type es 'logit' y se ignorarían."
            )
        for i, par in enumerate(self.feature_pairs):
            faltan = {"teacher", "student"} - set(par)
            if faltan or set(par) - {"teacher", "student"}:
                raise ValueError(
                    f"El par de capas {i} es {par!r} y debe tener exactamente las "
                    "claves 'teacher' y 'student'."
                )
        return self


class ExportConfig(StrictModel):
    """Configuración de exportación."""

    formats: list[Literal["onnx", "tensorrt", "openvino", "tflite"]] = Field(
        default_factory=lambda: ["onnx"]
    )
    precisions: list[Literal["fp32", "fp16", "int8"]] = Field(default_factory=lambda: ["fp32"])
    opset: int = Field(default=17, gt=0)
    dynamic_axes: bool = False
    simplify: bool = True
    batch_size: int = Field(default=1, gt=0)
    validate_numerical_equivalence: bool = True
    tolerance_fp32: float = Field(default=1e-4, gt=0.0)
    tolerance_fp16: float = Field(default=1e-2, gt=0.0)
    tolerance_int8: float = Field(default=1e-1, gt=0.0)

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "ExportConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


class EvaluationConfig(StrictModel):
    """Configuración de la evaluación de calidad.

    Los umbrales son los mismos para todas las variantes, para que las diferencias de mAP se deban a
    la compresión.
    """

    # 'val' es la partición de toda la campaña y el valor por defecto de --split en los scripts.
    split: Literal["train", "val", "test"] = "val"
    conf_threshold: float = Field(default=0.001, gt=0.0, lt=1.0)
    iou_threshold: float = Field(default=0.7, gt=0.0, lt=1.0)
    max_detections: int = Field(default=300, gt=0)
    # Criterio de tamaño primario. Los dos se calculan siempre. 'inference_res' mide el área a la
    # entrada del modelo tras el letterbox; 'original_res' es el estándar de pycocotools.
    size_criterion: Literal["inference_res", "original_res"] = "inference_res"
    compute_error_analysis: bool = True
    smoke_test_images: int = Field(default=32, gt=0)

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "EvaluationConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


class BenchmarkConfig(StrictModel):
    """Configuración del protocolo de medida.

    Por defecto: 50 iteraciones de calentamiento, 300 de medida y 3 repeticiones. Con
    ``valid_for_measurement=False`` (entornos compartidos) el protocolo rechaza medir, e
    ``invalid_reason`` es obligatorio. ``cpu_affinity`` y ``num_threads`` distinguen perfiles sobre
    la misma CPU.
    """

    device_id: str = "workstation_cpu"
    valid_for_measurement: bool = True
    invalid_reason: str = ""
    cpu_affinity: list[int] | None = None
    warmup_iters: int = Field(default=50, ge=0)
    measure_iters: int = Field(default=300, gt=0)
    num_repeats: int = Field(default=3, gt=0)
    batch_size: int = Field(default=1, gt=0)
    throughput_batch_sizes: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16])
    throughput_window_s: float = Field(default=30.0, gt=0.0)
    num_threads: int | None = None
    memory_sample_interval_s: float = Field(default=0.05, gt=0.0)
    check_system_load: bool = True
    max_system_load_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    record_temperature: bool = True
    smoke_test_warmup_iters: int = Field(default=5, ge=0)
    smoke_test_measure_iters: int = Field(default=20, gt=0)
    smoke_test_num_repeats: int = Field(default=2, gt=0)

    @model_validator(mode="after")
    def _validar_motivo_de_invalidez(self) -> "BenchmarkConfig":
        """Exige ``invalid_reason`` cuando el perfil no sirve para medir."""
        if not self.valid_for_measurement and not self.invalid_reason.strip():
            raise ValueError(
                "Un perfil con valid_for_measurement=false debe declarar 'invalid_reason'."
            )
        return self

    @field_validator("batch_size")
    @classmethod
    def _avisar_lote_distinto_de_uno(cls, valor: int) -> int:
        """Avisa si la latencia se va a medir con lote distinto de 1."""
        if valor != 1:
            logger.warning(
                "La latencia se va a medir con batch_size=%d; debería ser 1. Los lotes mayores "
                "son para el rendimiento.",
                valor,
            )
        return valor

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "BenchmarkConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


class ProjectConfig(StrictModel):
    """Configuración raíz del proyecto.

    Las secciones de compresión son opcionales: un baseline no declara ``pruning``.
    """

    project_name: str = "tfm-edge-object-detection"
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    device: str = "auto"

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    quantization: QuantizationConfig | None = None
    pruning: PruningConfig | None = None
    distillation: DistillationConfig | None = None

    export: ExportConfig = Field(default_factory=ExportConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    technique: Literal["baseline", "ptq", "qat", "pruning", "distillation", "combined"] = "baseline"
    technique_variant: str = "default"
    notes: str = ""

    @model_validator(mode="after")
    def _comprobar_opciones_sin_lector(self) -> "ProjectConfig":
        """Rechaza opciones declaradas que nadie lee."""
        _rechazar_opciones_sin_lector(self)
        return self


# Carga con herencia


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Fusiona dos diccionarios en profundidad. Las listas y demás tipos se reemplazan, no se
    concatenan.
    """
    resultado: dict[str, Any] = dict(base)
    for clave, valor in override.items():
        if (
            clave in resultado
            and isinstance(resultado[clave], Mapping)
            and isinstance(valor, Mapping)
        ):
            resultado[clave] = deep_merge(resultado[clave], valor)
        else:
            resultado[clave] = valor
    return resultado


def load_yaml_with_inheritance(path: Path | str, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Carga un YAML resolviendo la herencia de ``_base_``.

    Las bases se resuelven respecto al directorio del fichero que las declara. ``_chain`` es
    interno, para detectar ciclos.
    """
    fichero = Path(path).expanduser().resolve()
    if not fichero.is_file():
        origen = _chain[-1] if _chain else None
        raise ConfigFileNotFoundError(fichero, referenced_from=origen)

    if fichero in _chain:
        raise ConfigInheritanceError(
            f"Herencia cíclica detectada: '{fichero}' ya estaba en la cadena.",
            chain=(*_chain, fichero),
        )

    with fichero.open("r", encoding="utf-8") as manejador:
        contenido = yaml.safe_load(manejador)

    if contenido is None:
        contenido = {}
    if not isinstance(contenido, dict):
        raise ConfigSchemaError(
            fichero, f"la raíz del YAML debe ser un diccionario y es {type(contenido).__name__}."
        )

    bases = contenido.pop(BASE_KEY, None)
    if bases is None:
        return contenido

    if isinstance(bases, (str, Path)):
        bases = [bases]
    if not isinstance(bases, list):
        raise ConfigInheritanceError(
            f"La clave '{BASE_KEY}' de '{fichero}' debe ser una ruta o una lista de "
            f"rutas, y es {type(bases).__name__}."
        )

    acumulado: dict[str, Any] = {}
    for base in bases:
        ruta_base = (fichero.parent / str(base)).resolve()
        cargada = load_yaml_with_inheritance(ruta_base, _chain=(*_chain, fichero))
        acumulado = deep_merge(acumulado, cargada)

    return deep_merge(acumulado, contenido)


# Overrides de línea de comandos


def parse_override(override: str) -> tuple[list[str], Any]:
    """Descompone un override ``clave.anidada=valor``; el valor se interpreta como YAML.

    Para forzar una cadena, entrecomíllala: ``nombre='150'``.
    """
    if "=" not in override:
        raise ConfigOverrideError(override, "falta el signo '='")
    clave, _, valor_texto = override.partition("=")
    clave = clave.strip()
    if not clave:
        raise ConfigOverrideError(override, "la clave está vacía")
    partes = clave.split(".")
    if any(not parte for parte in partes):
        raise ConfigOverrideError(override, "la clave tiene un componente vacío")
    try:
        valor = yaml.safe_load(valor_texto)
    except yaml.YAMLError as error:
        raise ConfigOverrideError(override, f"el valor no es YAML válido ({error})") from error
    return partes, valor


def apply_overrides(data: Mapping[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    """Aplica una lista de overrides ``clave.anidada=valor`` sobre una configuración."""
    resultado: dict[str, Any] = json.loads(json.dumps(data, default=str)) if data else {}
    for override in overrides or []:
        partes, valor = parse_override(override)
        nodo: dict[str, Any] = resultado
        for parte in partes[:-1]:
            siguiente = nodo.get(parte)
            if siguiente is None:
                siguiente = {}
                nodo[parte] = siguiente
            elif not isinstance(siguiente, dict):
                raise ConfigOverrideError(
                    override,
                    f"la clave intermedia '{parte}' contiene un valor de tipo "
                    f"{type(siguiente).__name__}, no un diccionario",
                )
            nodo = siguiente
        nodo[partes[-1]] = valor
        logger.debug("Override aplicado: %s = %r", ".".join(partes), valor)
    return resultado


# Validación y carga completa


def _formatear_errores(error: ValidationError) -> str:
    """Una línea por error de validación, con la ruta de la clave."""
    lineas = []
    for detalle in error.errors():
        ruta = ".".join(str(parte) for parte in detalle["loc"]) or "(raíz)"
        lineas.append(f"  - {ruta}: {detalle['msg']}")
    return "\n".join(lineas)


def validate_config(
    data: Mapping[str, Any],
    schema: Type[T] = ProjectConfig,  # type: ignore[assignment]
    source: Path | None = None,
) -> T:
    """Valida un diccionario contra un esquema Pydantic."""
    try:
        return schema.model_validate(dict(data))
    except ValidationError as error:
        raise ConfigSchemaError(source, _formatear_errores(error)) from error


def load_config(
    path: Path | str,
    overrides: Sequence[str] | None = None,
    schema: Type[T] = ProjectConfig,  # type: ignore[assignment]
) -> T:
    """Carga, fusiona, aplica overrides y valida una configuración. Es el punto de entrada de los
    scripts.
    """
    fichero = Path(path)
    logger.debug("Cargando configuración desde %s", fichero)
    crudo = load_yaml_with_inheritance(fichero)
    con_overrides = apply_overrides(crudo, overrides)
    configuracion = validate_config(con_overrides, schema=schema, source=fichero)
    logger.info("Configuración cargada y validada: %s", fichero)
    return configuracion


# Hash y serialización de la configuración efectiva


def _canonicalizar(payload: Any) -> str:
    """JSON canónico (claves ordenadas, separadores fijos) para calcular el hash."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def compute_config_hash(payload: Mapping[str, Any], length: int = SHORT_HASH_LENGTH) -> str:
    """Primeros ``length`` caracteres del SHA-256 de la forma canónica."""
    if not 1 <= length <= 64:
        raise ValueError(f"La longitud del hash debe estar entre 1 y 64, y se pidió {length}.")
    canonico = _canonicalizar(payload)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:length]


def experiment_identity_payload(config: BaseModel) -> dict[str, Any]:
    """Secciones de la configuración que definen la identidad del modelo.

    No incluye benchmark, export ni evaluation. ``deterministic`` sí cambia el entrenamiento y no
    está en el hash, así que pedir ``deterministic: false`` lanza ConfigSchemaError.
    """
    if getattr(config, "deterministic", True) is not True:
        raise ConfigSchemaError(
            "deterministic: false cambia el entrenamiento y NO entra en el experiment_id.",
            "Dos ejecuciones que solo difieran en esta clave compartirían identificador. Para "
            "usarla, añade 'deterministic' a EXPERIMENT_IDENTITY_SECTIONS.",
        )
    completo = config.model_dump(mode="json")
    return {
        clave: completo[clave]
        for clave in EXPERIMENT_IDENTITY_SECTIONS
        if clave in completo and completo[clave] is not None
    }


def compute_experiment_hash(config: BaseModel, length: int = SHORT_HASH_LENGTH) -> str:
    """Hash de las secciones de identidad que se usa en el experiment_id."""
    return compute_config_hash(experiment_identity_payload(config), length=length)


def dump_effective_config(config: BaseModel, path: Path | str) -> Path:
    """Guarda la configuración efectiva junto a los resultados, con los overrides aplicados."""
    destino = ensure_parent_dir(path)
    contenido = config.model_dump(mode="json")
    with destino.open("w", encoding="utf-8") as manejador:
        yaml.safe_dump(contenido, manejador, allow_unicode=True, sort_keys=True, indent=2)
    logger.debug("Configuración efectiva volcada en %s", destino)
    return destino


def load_class_names(path: Path | str) -> list[str]:
    """Carga el mapeo índice-nombre de clase desde su JSON (``{"classes": [...]}``).

    Lo genera la preparación del dataset; si no existe, falla.
    """
    fichero = Path(path)
    if not fichero.is_file():
        raise ConfigFileNotFoundError(fichero)
    with fichero.open("r", encoding="utf-8") as manejador:
        contenido = json.load(manejador)
    if not isinstance(contenido, dict) or "classes" not in contenido:
        raise ConfigSchemaError(fichero, "  - se esperaba un objeto con la clave 'classes'")
    clases = contenido["classes"]
    if not isinstance(clases, list) or not all(isinstance(c, str) for c in clases):
        raise ConfigSchemaError(fichero, "  - 'classes' debe ser una lista de cadenas")
    if not clases:
        raise ConfigSchemaError(fichero, "  - 'classes' está vacía")
    return list(clases)
