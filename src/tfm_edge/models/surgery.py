"""Operaciones sobre la estructura del modelo: acceso por ruta, sustitución, congelación y huella.

Poda, cuantización y destilación las comparten, incluida la lista de capas protegidas. Cada
sustitución comprueba que ha surtido efecto. El análisis de dependencias está en
compression/pruning/dependency.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from torch import nn

    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)


class SurgeryError(TFMEdgeError):
    """Error al intervenir la estructura de un modelo."""


# Acceso a submódulos por ruta


def get_submodule(root: "nn.Module", path: str) -> "nn.Module":
    """Submódulo a partir de su ruta con puntos, por ejemplo ``"model.4.cv1.conv"``."""
    actual: Any = root
    recorrido: list[str] = []
    for segmento in path.split("."):
        recorrido.append(segmento)
        if segmento.isdigit() and hasattr(actual, "__getitem__"):
            try:
                actual = actual[int(segmento)]
                continue
            except (IndexError, KeyError, TypeError) as error:
                raise SurgeryError(
                    f"La ruta '{path}' falla en '{'.'.join(recorrido)}': "
                    f"índice fuera de rango.",
                    "Comprueba la ruta con detector.get_named_conv_layers().",
                ) from error
        if not hasattr(actual, segmento):
            disponibles = [n for n, _ in getattr(actual, "named_children", lambda: [])()]
            raise SurgeryError(
                f"La ruta '{path}' falla en '{'.'.join(recorrido)}': no existe.",
                f"Hijos disponibles en ese punto: {disponibles[:10]}",
            )
        actual = getattr(actual, segmento)
    return actual


def replace_submodule(root: "nn.Module", path: str, replacement: "nn.Module") -> "nn.Module":
    """Sustituye un submódulo y comprueba que el cambio ha tenido efecto. Devuelve el módulo
    anterior.
    """
    if "." not in path:
        padre: Any = root
        hoja = path
    else:
        ruta_padre, hoja = path.rsplit(".", 1)
        padre = get_submodule(root, ruta_padre)

    anterior = get_submodule(root, path)
    if hoja.isdigit() and hasattr(padre, "__setitem__"):
        padre[int(hoja)] = replacement
    else:
        setattr(padre, hoja, replacement)

    # Comprobación de que el cambio está aplicado.
    actual = get_submodule(root, path)
    if actual is not replacement:
        raise SurgeryError(
            f"La sustitución de '{path}' no ha surtido efecto: sigue habiendo "
            f"{type(actual).__name__}.",
            "Suele significar que el padre es una copia y no el módulo real. "
            "Comprueba que 'root' es el modelo vivo, no un deepcopy.",
        )
    logger.debug(
        "Sustituido '%s': %s por %s", path, type(anterior).__name__, type(replacement).__name__
    )
    return anterior


# Capas protegidas


def protected_layer_paths(detector: "Detector") -> set[str]:
    """Rutas de las capas que no se modifican por defecto: la cabeza de detección.

    Podar sus salidas cambiaría las clases o cajas que predice, y es la parte más sensible a la
    cuantización. En yolov8n a 640 px la cabeza supone el 37 % de las operaciones, lo que limita el
    ahorro posible con poda.
    """
    return set(detector.get_head_conv_names())


def assert_not_protected(detector: "Detector", paths: Sequence[str]) -> None:
    """Lanza SurgeryError si alguna de las rutas está protegida."""
    protegidas = protected_layer_paths(detector)
    invasoras = sorted(set(paths) & protegidas)
    if invasoras:
        raise SurgeryError(
            f"{len(invasoras)} de las capas a intervenir están protegidas: " f"{invasoras[:5]}",
            "La cabeza de detección no se poda ni se cuantiza por defecto. Si es intencionado, "
            "pásalo explícitamente.",
        )


# Congelación de parámetros


def freeze_parameters(root: "nn.Module", path_prefixes: Sequence[str]) -> int:
    """Congela los parámetros que empiezan por alguno de los prefijos y devuelve cuántos."""
    if not path_prefixes:
        return 0
    congelados = 0
    for nombre, parametro in root.named_parameters():
        if any(nombre.startswith(prefijo) for prefijo in path_prefixes):
            parametro.requires_grad_(False)
            congelados += parametro.numel()
    logger.info("Congelados %s parámetros bajo %s", f"{congelados:,}", list(path_prefixes)[:5])
    return congelados


def unfreeze_all(root: "nn.Module") -> int:
    """Reactiva el gradiente en todos los parámetros de coma flotante.

    Los checkpoints guardan la EMA con el gradiente desactivado. Ultralytics lo reactiva al
    entrenar, pero QAT, Taylor y la destilación lo necesitan antes. Se filtra por dtype porque un
    tensor entero no admite requires_grad. Devuelve el número de elementos reactivados.
    """
    descongelados = 0
    tensores = 0
    for parametro in root.parameters():
        if parametro.requires_grad or not parametro.dtype.is_floating_point:
            continue
        parametro.requires_grad_(True)
        descongelados += parametro.numel()
        tensores += 1
    if tensores:
        logger.info(
            "Reactivado el gradiente en %d parámetros (%s elementos) que venían congelados en "
            "la EMA.",
            tensores,
            f"{descongelados:,}",
        )
    return descongelados


def frozen_summary(root: "nn.Module") -> dict[str, int]:
    """Resume cuántos parámetros están congelados."""
    total = sum(p.numel() for p in root.parameters())
    entrenables = sum(p.numel() for p in root.parameters() if p.requires_grad)
    return {"total": total, "trainable": entrenables, "frozen": total - entrenables}


# Huella estructural


@dataclass
class StructuralFingerprint:
    """Huella de la estructura de un modelo para detectar cambios."""

    num_layers: int
    conv_channels: dict[str, tuple[int, int]] = field(default_factory=dict)
    module_types: dict[str, int] = field(default_factory=dict)
    num_parameters: int = 0

    def diff(self, otra: "StructuralFingerprint") -> dict[str, Any]:
        """Diferencias entre dos huellas; vacío si son iguales."""
        cambiadas = {
            nombre: (canales, otra.conv_channels[nombre])
            for nombre, canales in self.conv_channels.items()
            if nombre in otra.conv_channels and otra.conv_channels[nombre] != canales
        }
        return {
            "layers_before": self.num_layers,
            "layers_after": otra.num_layers,
            "params_before": self.num_parameters,
            "params_after": otra.num_parameters,
            "convs_removed": sorted(set(self.conv_channels) - set(otra.conv_channels)),
            "convs_added": sorted(set(otra.conv_channels) - set(self.conv_channels)),
            "convs_reshaped": cambiadas,
            "identical": (
                self.num_layers == otra.num_layers
                and self.conv_channels == otra.conv_channels
                and self.num_parameters == otra.num_parameters
            ),
        }


def structural_fingerprint(detector: "Detector") -> StructuralFingerprint:
    """Huella de la estructura actual del detector.

    Sirve para comprobar que un entrenamiento no cambia la estructura y que una poda sí la cambia.
    """
    from torch import nn

    tipos: dict[str, int] = {}
    for modulo in detector.model.modules():
        nombre = type(modulo).__name__
        tipos[nombre] = tipos.get(nombre, 0) + 1

    return StructuralFingerprint(
        num_layers=len(detector.layers),
        conv_channels=detector.channels_per_conv(),
        module_types=dict(sorted(tipos.items())),
        num_parameters=sum(p.numel() for p in detector.model.parameters()),
    )


def iter_prunable_convs(
    detector: "Detector", include_protected: bool = False
) -> Iterator[tuple[str, "nn.Conv2d"]]:
    """Recorre ``(ruta, convolución)`` de las convoluciones candidatas a poda."""
    protegidas = set() if include_protected else protected_layer_paths(detector)
    for nombre, conv in detector.get_named_conv_layers().items():
        if nombre in protegidas:
            continue
        # Las convoluciones agrupadas (depthwise) se dejan fuera: podarlas exige recalcular los
        # grupos.
        if conv.groups != 1:
            logger.debug("Se omite '%s': convolución agrupada (groups=%d)", nombre, conv.groups)
            continue
        yield nombre, conv


def assert_forward_pass_works(
    detector: "Detector", resolution: int | None = None, batch_size: int = 2
) -> dict[str, Any]:
    """Comprueba con un paso hacia delante que el modelo sigue funcionando. Es obligatorio tras
    podar.

    Se usa lote 2 porque con lote 1 algunos errores de dimensión no aparecen.
    """
    import torch

    resolucion = int(resolution or detector.input_resolution)
    if batch_size < 2:
        raise SurgeryError(
            f"batch_size debe ser >= 2 para que la validación sea útil, y es {batch_size}.",
            "Con lote 1 hay errores de dimensión que no se manifiestan.",
        )

    estaba_entrenando = detector.model.training
    detector.model.eval()
    try:
        entrada = torch.zeros(
            batch_size, 3, resolucion, resolucion, device=detector.device, dtype=detector.dtype
        )
        with torch.no_grad():
            salida = detector.model(entrada)
    except Exception as error:
        raise SurgeryError(
            f"El paso hacia delante falla tras la intervención: "
            f"{type(error).__name__}: {error}",
            "La estructura es incoherente. Si vienes de una poda, revisa el grafo de dependencias.",
        ) from error
    finally:
        detector.model.train(estaba_entrenando)

    def formas(objeto: Any) -> Any:
        """Extrae recursivamente las formas de una salida anidada."""
        if hasattr(objeto, "shape"):
            return tuple(objeto.shape)
        if isinstance(objeto, (list, tuple)):
            return [formas(o) for o in objeto]
        return type(objeto).__name__

    return {
        "input_shape": (batch_size, 3, resolucion, resolucion),
        "output_shapes": formas(salida),
        "ok": True,
    }
