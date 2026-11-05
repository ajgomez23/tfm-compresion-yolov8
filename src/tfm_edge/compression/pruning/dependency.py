"""Grafo de dependencias entre capas y la transformación previa que necesita YOLOv8.

El análisis de dependencias lo hace torch-pruning (DepGraph). El ``chunk`` seguido de ``cat`` de
C2f.forward hace que torch-pruning calcule mal los índices de la concatenación (IndexError con el
doble del ancho), así que antes se sustituye cada C2f por SplitFreeC2f, con la primera convolución
partida en dos. La equivalencia depende de copiar también eps y momentum de las BatchNorm, que no
viajan en el state_dict; make_prunable la mide y falla si no se cumple.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch
import torch.nn as nn

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Se compara por nombre de clase para no importar ultralytics fuera de models/detector.py.
C2F_TYPE_NAME = "C2f"

#: Atributos de enrutado que _predict_once lee en cada capa; el módulo sustituto debe tenerlos.
ULTRALYTICS_ROUTING_ATTRS = ("i", "f", "type", "np")

#: Tolerancia relativa de la equivalencia. En CPU la diferencia es 0; el margen es para GPU.
SURGERY_RELATIVE_TOLERANCE = 1e-5


class PruningError(TFMEdgeError):
    """Error genérico del submódulo de poda."""


class GraphSurgeryError(PruningError):
    """La transformación previa a la poda no se pudo aplicar o no es equivalente."""


class ShapeValidationError(PruningError):
    """El modelo podado no pasa la validación de formas."""


class SplitFreeC2f(nn.Module):
    """C2f con la primera convolución partida en dos, sin ``chunk``.

    Es numéricamente idéntico al original si también se reparten los parámetros de la normalización.
    """

    def __init__(self, original: nn.Module) -> None:
        """Construye el módulo equivalente a partir de un C2f de Ultralytics."""
        super().__init__()
        for atributo in ("cv1", "cv2", "m", "c"):
            if not hasattr(original, atributo):
                raise GraphSurgeryError(
                    f"El módulo a sustituir no tiene el atributo '{atributo}'.",
                    "Se esperaba un C2f de Ultralytics. Si su estructura ha cambiado, hay que "
                    "revisar esta transformación.",
                )
        # Se reutiliza la clase Conv del módulo existente para no importar ultralytics.
        tipo_conv = type(original.cv1)
        canales_entrada = original.cv1.conv.in_channels
        self.c = original.c
        # Las convoluciones nuevas se crean en el mismo dispositivo que el módulo original.
        referencia = original.cv1.conv.weight
        self.cv0 = tipo_conv(canales_entrada, self.c, 1, 1).to(
            device=referencia.device, dtype=referencia.dtype
        )
        self.cv1 = tipo_conv(canales_entrada, self.c, 1, 1).to(
            device=referencia.device, dtype=referencia.dtype
        )
        self.cv2 = original.cv2
        self.m = original.m
        self._transferir_pesos(original)
        for atributo in ULTRALYTICS_ROUTING_ATTRS:
            if hasattr(original, atributo):
                setattr(self, atributo, getattr(original, atributo))

    def _transferir_pesos(self, original: nn.Module) -> None:
        """Reparte los parámetros de ``cv1`` entre las dos convoluciones nuevas."""
        c = self.c
        mitades = ((self.cv0, slice(0, c)), (self.cv1, slice(c, None)))
        with torch.no_grad():
            for destino, corte in mitades:
                destino.conv.weight.copy_(original.cv1.conv.weight[corte])
                if original.cv1.conv.bias is not None and destino.conv.bias is not None:
                    destino.conv.bias.copy_(original.cv1.conv.bias[corte])
                # eps y momentum no están en el state_dict y Ultralytics los cambia tras construir
                # el modelo.
                destino.bn.eps = original.cv1.bn.eps
                destino.bn.momentum = original.cv1.bn.momentum
                destino.bn.weight.copy_(original.cv1.bn.weight[corte])
                destino.bn.bias.copy_(original.cv1.bn.bias[corte])
                destino.bn.running_mean.copy_(original.cv1.bn.running_mean[corte])
                destino.bn.running_var.copy_(original.cv1.bn.running_var[corte])
                destino.bn.num_batches_tracked.copy_(original.cv1.bn.num_batches_tracked)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ejecuta el módulo."""
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


@dataclass
class SurgeryReport:
    """Resultado de preparar un modelo para la poda. El número de parámetros no debe cambiar."""

    converted: list[str] = field(default_factory=list)
    max_abs_difference: float | str = ""
    output_scale: float | str = ""
    parameters_before: int = 0
    parameters_after: int = 0
    verified: bool = False
    notes: str = ""

    @property
    def relative_difference(self) -> float | str:
        """Diferencia máxima relativa a la escala de la salida."""
        if not isinstance(self.max_abs_difference, float):
            return ""
        if not isinstance(self.output_scale, float) or self.output_scale <= 0.0:
            return ""
        return self.max_abs_difference / self.output_scale

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return {
            "converted": list(self.converted),
            "num_converted": len(self.converted),
            "max_abs_difference": self.max_abs_difference,
            "output_scale": self.output_scale,
            "relative_difference": self.relative_difference,
            "parameters_before": self.parameters_before,
            "parameters_after": self.parameters_after,
            "verified": self.verified,
            "notes": self.notes,
        }


def count_c2f_modules(model: nn.Module) -> int:
    """Número de C2f sin transformar que quedan en el modelo."""
    return sum(1 for m in model.modules() if type(m).__name__ == C2F_TYPE_NAME)


def is_prunable(model: nn.Module) -> bool:
    """True si no queda ningún C2f con ``chunk``."""
    return count_c2f_modules(model) == 0


def _sustituir_c2f(model: nn.Module) -> list[str]:
    """Sustituye en el sitio todos los C2f por SplitFreeC2f y devuelve sus nombres."""
    sustituidos: list[str] = []
    for nombre_padre, padre in list(model.named_modules()):
        for nombre, hijo in list(padre.named_children()):
            if type(hijo).__name__ != C2F_TYPE_NAME:
                continue
            setattr(padre, nombre, SplitFreeC2f(hijo))
            completo = f"{nombre_padre}.{nombre}" if nombre_padre else nombre
            sustituidos.append(completo)
    return sustituidos


def make_prunable(
    detector: "Detector",
    example_input: torch.Tensor | None = None,
    tolerance: float = SURGERY_RELATIVE_TOLERANCE,
) -> tuple["Detector", SurgeryReport]:
    """Prepara una copia del detector para la poda y comprueba que la salida no cambia.

    La entrada de prueba puede ser aleatoria: se comprueba una identidad algebraica y en evaluación
    la normalización usa sus estadísticas acumuladas.
    """
    from tfm_edge.models.detector import Detector

    if is_prunable(detector.model):
        raise GraphSurgeryError(
            "El modelo no tiene ningún módulo C2f que transformar.",
            "O ya se preparó para la poda (en cuyo caso no hay que repetirlo), o "
            "la arquitectura no es YOLOv8. Comprueba con is_prunable().",
        )

    if example_input is None:
        generador = torch.Generator().manual_seed(0)
        example_input = torch.randn(
            1,
            3,
            detector.input_resolution,
            detector.input_resolution,
            generator=generador,
        )
    example_input = example_input.to(detector.device)

    original = detector.model.eval()
    with torch.no_grad():
        salida_antes = _tensor_principal(original(example_input))
    params_antes = sum(p.numel() for p in original.parameters())

    copia = copy.deepcopy(original)
    sustituidos = _sustituir_c2f(copia)
    copia.eval()
    with torch.no_grad():
        salida_despues = _tensor_principal(copia(example_input))
    params_despues = sum(p.numel() for p in copia.parameters())

    informe = SurgeryReport(
        converted=sustituidos,
        parameters_before=params_antes,
        parameters_after=params_despues,
        notes=(
            "Sustitución de C2f por SplitFreeC2f: el chunk desaparece y el grafo "
            "de dependencias se construye bien. La transformación es algebraica "
            "y no cambia el número de parámetros."
        ),
    )

    if params_despues != params_antes:
        raise GraphSurgeryError(
            f"La transformación ha cambiado el número de parámetros: {params_antes:,} antes y "
            f"{params_despues:,} después.",
            "No debería: solo reorganiza una convolución en dos. Revisa el "
            "reparto de pesos de SplitFreeC2f antes de podar nada.",
        )

    if salida_antes.shape != salida_despues.shape:
        raise GraphSurgeryError(
            f"La transformación ha cambiado la forma de la salida: {tuple(salida_antes.shape)} "
            f"antes y {tuple(salida_despues.shape)} después.",
            "La salida del modelo transformado no coincide con la del original.",
        )

    diferencia = (salida_antes - salida_despues).abs().max().item()
    escala = salida_antes.abs().max().item()
    informe.max_abs_difference = diferencia
    informe.output_scale = escala

    admitida = tolerance * max(escala, 1e-12)
    if diferencia > admitida:
        raise GraphSurgeryError(
            f"La transformación NO es equivalente: diferencia máxima {diferencia:.3e} "
            f"sobre una escala de {escala:.3e} (admitida {admitida:.3e}).",
            "La salida del modelo transformado no coincide con la del original. Comprueba que "
            "SplitFreeC2f copia eps y momentum de la normalización.",
        )

    informe.verified = True
    logger.info(
        "Modelo preparado para la poda: %d módulos C2f sustituidos, diferencia "
        "máxima %.3e sobre una escala de %.3e",
        len(sustituidos),
        diferencia,
        escala,
    )
    preparado = Detector(
        copia,
        size=detector.size,
        num_classes=detector.num_classes,
        input_resolution=detector.input_resolution,
    )
    return preparado, informe


def _tensor_principal(salida: Any) -> torch.Tensor:
    """Tensor de predicciones de la salida de YOLOv8: tupla en evaluación, lista de mapas en
    entrenamiento.
    """
    if isinstance(salida, torch.Tensor):
        return salida
    if isinstance(salida, (tuple, list)) and salida:
        primero = salida[0]
        if isinstance(primero, torch.Tensor):
            return primero
        if isinstance(primero, (tuple, list)) and primero:
            return torch.cat([t.flatten() for t in primero if isinstance(t, torch.Tensor)])
    raise GraphSurgeryError(
        f"No se reconoce la salida del modelo (tipo {type(salida).__name__}).",
        "Se esperaba un tensor, o una tupla o lista que empiece por uno.",
    )


def build_dependency_graph(detector: "Detector", example_input: torch.Tensor | None = None) -> Any:
    """Construye el grafo de dependencias de torch-pruning. El detector debe pasar antes por
    make_prunable.
    """
    tp = _importar_torch_pruning()
    if not is_prunable(detector.model):
        raise GraphSurgeryError(
            f"El detector todavía tiene {count_c2f_modules(detector.model)} "
            "módulos C2f sin transformar.",
            "Pasa antes por make_prunable(); si no, torch-pruning calcula mal los índices de las "
            "concatenaciones del C2f.",
        )
    if example_input is None:
        generador = torch.Generator().manual_seed(0)
        example_input = torch.randn(
            1, 3, detector.input_resolution, detector.input_resolution, generator=generador
        )
    example_input = example_input.to(detector.device)
    return tp.DependencyGraph().build_dependency(
        detector.model, example_inputs=example_input, verbose=False
    )


def resolve_protected_modules(detector: "Detector", patterns: Sequence[str]) -> list[nn.Module]:
    """Módulos que nunca se podan a partir de prefijos de nombre (``model.22`` protege la cabeza).

    Un patrón que no coincide con ninguna capa es un error, porque suele ser una errata.
    """
    modulos: list[nn.Module] = []
    sin_coincidencias: list[str] = []
    for patron in patterns:
        coincidentes = [
            m
            for n, m in detector.model.named_modules()
            if n.startswith(patron) and isinstance(m, (nn.Conv2d, nn.Linear))
        ]
        if not coincidentes:
            sin_coincidencias.append(patron)
        modulos.extend(coincidentes)
    if sin_coincidencias:
        raise PruningError(
            f"Estos patrones de capa protegida no coinciden con ninguna capa: "
            f"{sin_coincidencias}.",
            "Una errata aquí deja desprotegida la capa que se quería proteger. "
            "Consulta los nombres reales con detector.get_named_conv_layers().",
        )
    # Sin duplicados y en orden, por si dos patrones se solapan.
    unicos: list[nn.Module] = []
    vistos: set[int] = set()
    for modulo in modulos:
        if id(modulo) not in vistos:
            vistos.add(id(modulo))
            unicos.append(modulo)
    logger.debug("%d módulos protegidos por %d patrones", len(unicos), len(patterns))
    return unicos


@dataclass
class ShapeValidation:
    """Resultado de validar un modelo podado con un pase hacia delante."""

    forward_ok: bool = False
    output_shape: tuple[int, ...] | None = None
    expected_output_channels: int | None = None
    min_channels_found: int = 0
    layers_below_minimum: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        """Si el modelo podado ha pasado todas las comprobaciones."""
        return self.forward_ok and not self.layers_below_minimum and not self.error

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la validación como diccionario serializable."""
        return {
            "forward_ok": self.forward_ok,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "expected_output_channels": self.expected_output_channels,
            "min_channels_found": self.min_channels_found,
            "layers_below_minimum": list(self.layers_below_minimum),
            "error": self.error,
            "ok": self.ok,
        }


def validate_after_pruning(
    detector: "Detector",
    min_channels: int = 1,
    batch_size: int = 1,
    raise_on_failure: bool = True,
    protected_layers: Sequence[str] = (),
) -> ShapeValidation:
    """Validación obligatoria de un modelo podado: pase hacia delante y mínimo de canales.

    Comprueba que el modelo se ejecuta, no que conserve la precisión. A las capas protegidas no se
    les exige el mínimo de canales; ``model.22.dfl.conv`` tiene un solo canal de salida.
    """
    esperados = detector.num_classes + 4
    validacion = ShapeValidation(expected_output_channels=esperados)

    canales = {n: m.out_channels for n, m in detector.get_named_conv_layers().items()}
    candidatas = {
        n: c
        for n, c in canales.items()
        if not any(n.startswith(patron) for patron in protected_layers)
    }
    validacion.min_channels_found = min(candidatas.values()) if candidatas else 0
    validacion.layers_below_minimum = sorted(n for n, c in candidatas.items() if c < min_channels)

    generador = torch.Generator().manual_seed(0)
    entrada = torch.randn(
        batch_size,
        3,
        detector.input_resolution,
        detector.input_resolution,
        generator=generador,
    ).to(detector.device)

    try:
        detector.model.eval()
        with torch.no_grad():
            salida = _tensor_principal(detector.model(entrada))
        validacion.forward_ok = True
        validacion.output_shape = tuple(salida.shape)
        if salida.ndim >= 2 and salida.shape[1] != esperados:
            validacion.error = (
                f"la salida tiene {salida.shape[1]} canales y se esperaban "
                f"{esperados} ({detector.num_classes} clases + 4 de caja): la "
                "cabeza del detector ha sido podada"
            )
    except Exception as excepcion:  # noqa: BLE001 - se registra el motivo real
        validacion.error = f"{type(excepcion).__name__}: {excepcion}"

    if validacion.ok:
        logger.info(
            "Validación tras podar: pase hacia delante correcto, salida %s, "
            "mínimo de canales %d",
            validacion.output_shape,
            validacion.min_channels_found,
        )
        return validacion

    detalle = validacion.error or (
        f"{len(validacion.layers_below_minimum)} capas por debajo del mínimo de "
        f"{min_channels} canales: {validacion.layers_below_minimum[:5]}"
    )
    logger.error("Validación tras podar FALLIDA: %s", detalle)
    if raise_on_failure:
        raise ShapeValidationError(
            f"El modelo podado no pasa la validación: {detalle}",
            "Baja el ratio de poda, sube min_channels, o añade a "
            "protected_layers las capas que no deben adelgazar más.",
        )
    return validacion


def _importar_torch_pruning() -> Any:
    """Importa torch-pruning o lanza DependencyNotAvailableError."""
    from tfm_edge.common.exceptions import DependencyNotAvailableError

    try:
        import torch_pruning as tp
    except ImportError as excepcion:  # pragma: no cover - depende del entorno
        raise DependencyNotAvailableError(
            "torch-pruning",
            "el análisis de dependencias entre capas (DepGraph), que este "
            "proyecto NO reimplementa",
            # 1.6.1 es la versión anclada; su __version__ dice "1.6.0" por un descuido del paquete.
            "Instálalo con: pip install torch-pruning==1.6.1",
        ) from excepcion
    return tp
