"""Poda estructurada de canales.

Los canales se eliminan físicamente y la reducción de parámetros se mide después; no coincide con el
ratio pedido porque podar un canal de salida arrastra las entradas de las capas siguientes.
Repartos: ``global``, ``uniform`` (control del global) y ``per_block``. PruningSession conserva el
podador entre ciclos para la poda iterativa; prune_model la ejecuta de una vez.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from tfm_edge.common.config import PruningConfig
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.pruning import importance as importance_module
from tfm_edge.compression.pruning.dependency import (
    PruningError,
    ShapeValidation,
    SurgeryReport,
    _importar_torch_pruning,
    is_prunable,
    make_prunable,
    resolve_protected_modules,
    validate_after_pruning,
)

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Bloques del reparto per_block, que corresponden a los métodos de introspección del detector.
BLOCK_NAMES = ("backbone", "neck", "head")


@dataclass
class PruningResult:
    """Resultado de podar un modelo."""

    criterion: str = "l1"
    distribution: str = "global"
    ratio_requested: float = 0.0
    steps_applied: int = 1
    total_steps: int = 1
    parameters_before: int = 0
    parameters_after: int = 0
    channels_before: dict[str, int] = field(default_factory=dict)
    channels_after: dict[str, int] = field(default_factory=dict)
    protected_layers: list[str] = field(default_factory=list)
    surgery: SurgeryReport | None = None
    validation: ShapeValidation | None = None
    notes: str = ""

    @property
    def complete(self) -> bool:
        """Si se han aplicado todos los ciclos previstos."""
        return self.steps_applied >= self.total_steps

    @property
    def parameter_reduction(self) -> float:
        """Fracción de parámetros eliminados, medida."""
        if self.parameters_before <= 0:
            return 0.0
        return 1.0 - self.parameters_after / self.parameters_before

    @property
    def layers_changed(self) -> list[str]:
        """Convoluciones cuyo número de canales de salida ha cambiado."""
        return sorted(
            n for n, c in self.channels_before.items() if self.channels_after.get(n, c) != c
        )

    @property
    def head_intact(self) -> bool:
        """Si ninguna capa de la cabeza ha perdido canales de salida."""
        return not any(n.startswith("model.22") for n in self.layers_changed)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        datos = asdict(self)
        datos["parameter_reduction"] = self.parameter_reduction
        datos["num_layers_changed"] = len(self.layers_changed)
        datos["head_intact"] = self.head_intact
        datos["complete"] = self.complete
        return datos

    def summary(self) -> str:
        """Genera un resumen legible."""
        lineas = [
            f"Poda estructurada: criterio '{self.criterion}', reparto "
            f"'{self.distribution}', ciclo {self.steps_applied} de {self.total_steps}",
            f"  ratio PEDIDO      : {self.ratio_requested:.1%}",
            f"  reducción MEDIDA  : {self.parameter_reduction:.1%} "
            f"(de {self.parameters_before:,} a {self.parameters_after:,} parámetros)",
            f"  capas afectadas   : {len(self.layers_changed)} de {len(self.channels_before)}",
            f"  cabeza intacta    : {'sí' if self.head_intact else 'NO'}",
        ]
        if self.validation is not None:
            estado = "correcta" if self.validation.ok else f"FALLIDA ({self.validation.error})"
            lineas.append(f"  validación        : {estado}, salida {self.validation.output_shape}")
        return "\n".join(lineas)


def _ratios_por_bloque(
    detector: "Detector", config: PruningConfig
) -> dict[tuple[nn.Module, ...], float]:
    """Traduce los ratios por bloque al formato de torch-pruning."""
    obtener = {
        "backbone": detector.get_backbone_modules,
        "neck": detector.get_neck_modules,
        "head": detector.get_head_modules,
    }
    ratios: dict[tuple[nn.Module, ...], float] = {}
    for bloque, ratio in config.block_ratios.items():
        if bloque not in obtener:
            raise PruningError(
                f"Bloque desconocido en block_ratios: '{bloque}'.",
                f"Los bloques del modelo son: {', '.join(BLOCK_NAMES)}.",
            )
        modulos = tuple(obtener[bloque]().values())
        if not modulos:
            raise PruningError(
                f"El bloque '{bloque}' no contiene ninguna capa.",
                "La estructura del modelo no es la esperada; revisa "
                "detector.validate_structure().",
            )
        ratios[modulos] = ratio
        logger.info("Reparto por bloques: '%s' con ratio %.1f %%", bloque, 100 * ratio)
    return ratios


class PruningSession:
    """Campaña de poda de uno o varios ciclos sobre una copia del detector.

    El podador de torch-pruning lleva la cuenta de lo podado y tiene que mantenerse entre ciclos.
    """

    def __init__(
        self,
        detector: "Detector",
        config: PruningConfig,
        example_input: torch.Tensor | None = None,
        prepare: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """Prepara la campaña sin podar todavía.

        ``prepare`` aplica la transformación previa que necesita YOLOv8. ``random_seed`` es
        obligatoria con el criterio ``random``.
        """
        tp = _importar_torch_pruning()

        self.surgery: SurgeryReport | None = None
        if not is_prunable(detector.model):
            if not prepare:
                raise PruningError(
                    "El detector no está preparado para la poda y prepare=False.",
                    "Llama antes a make_prunable(), o usa prepare=True.",
                )
            detector, self.surgery = make_prunable(detector, example_input=example_input)

        importance_module.assert_gradients_available(detector.model, config.criterion)

        if example_input is None:
            generador = torch.Generator().manual_seed(0)
            example_input = torch.randn(
                1, 3, detector.input_resolution, detector.input_resolution, generator=generador
            )
        self.detector = detector
        self.config = config
        self.total_steps = config.iterative_steps
        self.steps_applied = 0
        self._example_input = example_input.to(detector.device)
        self._channels_before = {
            n: m.out_channels for n, m in detector.get_named_conv_layers().items()
        }
        self._parameters_before = detector.count_parameters()

        protegidos = resolve_protected_modules(detector, config.protected_layers)
        argumentos: dict[str, Any] = {
            "model": detector.model,
            "example_inputs": self._example_input,
            "importance": importance_module.build_importance(config.criterion, seed=random_seed),
            "pruning_ratio": config.ratio,
            "ignored_layers": protegidos,
            "iterative_steps": self.total_steps,
        }
        if config.distribution == "global":
            argumentos["global_pruning"] = True
        elif config.distribution == "uniform":
            argumentos["global_pruning"] = False
        elif config.distribution == "per_block":
            argumentos["global_pruning"] = False
            argumentos["pruning_ratio_dict"] = _ratios_por_bloque(detector, config)
        else:  # pragma: no cover - Pydantic ya restringe los valores
            raise PruningError(
                f"Reparto de poda desconocido: '{config.distribution}'.",
                "Los admitidos son: global, uniform, per_block.",
            )
        self._pruner = tp.MetaPruner(**argumentos)
        logger.info(
            "Campaña de poda preparada: criterio '%s', reparto '%s', ratio "
            "objetivo %.1f %% en %d ciclo(s)",
            config.criterion,
            config.distribution,
            100 * config.ratio,
            self.total_steps,
        )

    def step(self, validate: bool = True) -> PruningResult:
        """Aplica un ciclo de poda y lo valida. El último ciclo se valida siempre."""
        if self.steps_applied >= self.total_steps:
            raise PruningError(
                f"La campaña ya ha agotado sus {self.total_steps} ciclos.",
                "Crea otra sesión si quieres seguir podando, con el ratio "
                "objetivo que corresponda al modelo ya podado.",
            )
        self._pruner.step()
        self.steps_applied += 1

        resultado = self._resultado_actual()
        if resultado.parameters_after >= resultado.parameters_before:
            raise PruningError(
                f"El ciclo {self.steps_applied} no ha reducido nada: "
                f"{resultado.parameters_before:,} parámetros antes y "
                f"{resultado.parameters_after:,} después.",
                "Con todas las capas protegidas o un ratio demasiado bajo no "
                "queda nada que podar. Revisa protected_layers y ratio.",
            )
        if validate:
            resultado.validation = validate_after_pruning(
                self.detector,
                min_channels=self.config.min_channels,
                raise_on_failure=True,
                protected_layers=self.config.protected_layers,
            )
        logger.info(
            "Ciclo %d de %d: reducción acumulada %.1f %% (%d capas afectadas)",
            self.steps_applied,
            self.total_steps,
            100 * resultado.parameter_reduction,
            len(resultado.layers_changed),
        )
        return resultado

    def _resultado_actual(self) -> PruningResult:
        """Compone el resultado con el estado actual del modelo."""
        return PruningResult(
            criterion=self.config.criterion,
            distribution=self.config.distribution,
            ratio_requested=self.config.ratio,
            steps_applied=self.steps_applied,
            total_steps=self.total_steps,
            parameters_before=self._parameters_before,
            parameters_after=self.detector.count_parameters(),
            channels_before=dict(self._channels_before),
            channels_after={
                n: m.out_channels for n, m in self.detector.get_named_conv_layers().items()
            },
            protected_layers=list(self.config.protected_layers),
            surgery=self.surgery,
            notes=(
                "ratio_requested es lo que se pidió; parameter_reduction es lo "
                "que se midió. Difieren porque podar un canal de salida arrastra "
                "el canal de entrada correspondiente de las capas siguientes."
            ),
        )


def prune_model(
    detector: "Detector",
    config: PruningConfig,
    example_input: torch.Tensor | None = None,
    prepare: bool = True,
    random_seed: int | None = None,
) -> tuple["Detector", PruningResult]:
    """Poda un detector hasta el objetivo sin reajuste entre ciclos.

    Con varios ciclos los aplica seguidos; la poda iterativa con reajuste está en ``schedule``.
    """
    sesion = PruningSession(
        detector,
        config,
        example_input=example_input,
        prepare=prepare,
        random_seed=random_seed,
    )
    resultado = sesion.step()
    while not resultado.complete:
        resultado = sesion.step()
    return sesion.detector, resultado


def compare_against_random(
    detector: "Detector",
    config: PruningConfig,
    example_input: torch.Tensor | None = None,
    random_seed: int | None = None,
) -> tuple["Detector", PruningResult]:
    """Poda con el criterio aleatorio y el resto de la configuración igual, como control."""
    de_control = config.model_copy(update={"criterion": "random"})
    logger.info(
        "Poda de control con criterio aleatorio: mismo ratio (%.1f %%), mismo "
        "reparto ('%s') y mismas capas protegidas que la poda real",
        100 * config.ratio,
        config.distribution,
    )
    return prune_model(detector, de_control, example_input=example_input, random_seed=random_seed)
