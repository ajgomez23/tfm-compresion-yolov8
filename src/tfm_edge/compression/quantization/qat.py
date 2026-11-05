"""Entrenamiento consciente de la cuantización (QAT) sobre YOLOv8.

Solo se simula la cuantización: PyTorch no tiene kernel cuantizado para SiLU, así que convert()
falla, mientras que FakeQuantize funciona en FP32 y deja pasar el gradiente. La rama QAT prepara el
baseline con simulación, lo afina, quita los nodos y materializa INT8 por el mismo camino que PTQ.
torch.ao.quantization está obsoleto y desaparece en PyTorch 2.10; el proyecto usa la 2.8.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.models.surgery import unfreeze_all

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from torch import nn

    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Motor de cuantización: ``x86`` para la CPU de medida, ``qnnpack`` para ARM.
DEFAULT_QENGINE = "x86"

#: Tipos de módulo que se cuantizan; el resto queda en coma flotante.
QUANTIZABLE_TYPES = ("Conv2d", "Linear")


class QatError(TFMEdgeError):
    """Error al preparar o revertir el entrenamiento consciente de cuantización."""


@dataclass
class QatConfig:
    """Configuración del QAT.

    La cabeza no se excluye por defecto, para poder medir su sensibilidad.
    """

    qengine: str = DEFAULT_QENGINE
    excluded_modules: tuple[str, ...] = ()
    exclude_head: bool = False
    freeze_observers_epoch: int | None = None
    freeze_bn_epoch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la configuración como diccionario serializable."""
        datos = asdict(self)
        datos["excluded_modules"] = list(self.excluded_modules)
        return datos


@dataclass
class QatPreparation:
    """Resultado de preparar un modelo para QAT.

    ``empty_buffers_after_prepare`` tiene que ser cero (ver settle_weight_observers).
    """

    num_weight_observers: int = 0
    num_activation_observers: int = 0
    weight_observers_settled: int = 0
    empty_buffers_after_prepare: int = 0
    excluded_modules: list[str] = field(default_factory=list)
    qengine: str = DEFAULT_QENGINE
    gradient_flows: bool = False
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        return asdict(self)

    def summary(self) -> str:
        """Genera un resumen legible."""
        return "\n".join(
            [
                f"QAT preparado con motor '{self.qengine}':",
                f"  observadores en pesos       : {self.num_weight_observers}",
                f"  observadores en activaciones: {self.num_activation_observers}",
                f"  formas asentadas al preparar: {self.weight_observers_settled}",
                f"  búferes sin forma           : {self.empty_buffers_after_prepare}",
                f"  módulos excluidos           : {len(self.excluded_modules)}",
                f"  el gradiente fluye          : "
                f"{'sí' if self.gradient_flows else 'NO COMPROBADO'}",
            ]
        )


def _importar_quantization() -> Any:
    """Importa ``torch.ao.quantization``."""
    import torch.ao.quantization as qao

    return qao


@contextmanager
def _sin_aviso_de_obsolescencia() -> Iterator[None]:
    """Silencia el aviso de obsolescencia de torch.ao.quantization, que sale en cada llamada.

    Solo se filtra ese mensaje, para que pytest siga detectando otros usos obsoletos.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*torch.ao.quantization is deprecated.*",
            category=DeprecationWarning,
        )
        yield


def prepare_for_qat(
    detector: "Detector", config: QatConfig | None = None, check_gradient: bool = True
) -> tuple["nn.Module", QatPreparation]:
    """Devuelve una copia del modelo con cuantización simulada, lista para afinar.

    ``check_gradient`` comprueba con un paso que el gradiente atraviesa la simulación.
    """
    import torch

    qao = _importar_quantization()
    configuracion = config or QatConfig()

    if configuracion.qengine not in torch.backends.quantized.supported_engines:
        raise QatError(
            f"El motor de cuantización '{configuracion.qengine}' no está disponible. Disponibles: "
            f"{list(torch.backends.quantized.supported_engines)}.",
            "Usa 'x86' para la CPU de medida o 'qnnpack' para ARM.",
        )
    torch.backends.quantized.engine = configuracion.qengine

    # Se trabaja sobre una copia para conservar el baseline.
    modelo = copy.deepcopy(detector.model).train()
    # Los checkpoints son la EMA y llegan con el gradiente desactivado; sin reactivarlo falla la
    # comprobación de gradiente.
    unfreeze_all(modelo)

    excluidos = list(configuracion.excluded_modules)
    if configuracion.exclude_head:
        excluidos.extend(sorted(detector.get_head_conv_names()))

    with _sin_aviso_de_obsolescencia():
        modelo.qconfig = qao.get_default_qat_qconfig(configuracion.qengine)
    # Un qconfig a None deja el módulo en coma flotante.
    for ruta in excluidos:
        try:
            from tfm_edge.models.surgery import get_submodule

            get_submodule(modelo, ruta).qconfig = None
        except Exception as error:
            raise QatError(
                f"No se ha podido excluir el módulo '{ruta}': " f"{type(error).__name__}: {error}",
                "Comprueba la ruta con detector.get_named_conv_layers().",
            ) from error

    try:
        with _sin_aviso_de_obsolescencia():
            preparado = qao.prepare_qat(modelo, inplace=False)
    except Exception as error:
        raise QatError(
            f"prepare_qat ha fallado: {type(error).__name__}: {error}",
            "Si vienes de una poda, valida antes la estructura con un paso hacia " "delante.",
        ) from error

    pesos = sum(
        1
        for m in preparado.modules()
        if "PerChannel" in type(m).__name__ and "Observer" in type(m).__name__
    )
    activaciones = sum(
        1
        for m in preparado.modules()
        if "Observer" in type(m).__name__ and "PerChannel" not in type(m).__name__
    )
    if pesos + activaciones == 0:
        raise QatError(
            "prepare_qat no ha insertado ningún observador.",
            "No se ha insertado ningún observador: el modelo se afinaría sin simular "
            "cuantización. Comprueba que el qconfig se ha propagado.",
        )

    asentados = settle_weight_observers(preparado)
    vacios = count_empty_buffers(preparado)
    if vacios:
        raise QatError(
            f"Tras preparar quedan {vacios} búferes sin forma real, y no debería "
            "quedar ninguno.",
            "Un búfer de forma (0,) toma su tamaño la primera vez que se ejecuta su módulo; si "
            "eso pasa solo en la EMA, los state_dict dejan de coincidir. Si algún módulo solo "
            "se ejecuta en evaluación, hay que asentarlo aparte.",
        )

    informe = QatPreparation(
        num_weight_observers=pesos,
        num_activation_observers=activaciones,
        weight_observers_settled=asentados,
        empty_buffers_after_prepare=vacios,
        excluded_modules=excluidos,
        qengine=configuracion.qengine,
        config=configuracion.to_dict(),
    )

    if check_gradient:
        informe.gradient_flows = _comprobar_gradiente(preparado, detector.input_resolution)

    logger.info("%s", informe.summary())
    return preparado, informe


def count_empty_buffers(model: "nn.Module") -> int:
    """Número de búferes con cero elementos.

    Los observadores per-channel nacen con min_val y max_val de forma ``(0,)`` hasta su primera
    ejecución.
    """
    return sum(1 for _, buffer in model.named_buffers() if buffer.numel() == 0)


def settle_weight_observers(model: "nn.Module") -> int:
    """Ejecuta una vez cada observador de peso para fijar la forma de sus búferes.

    La convolución del DFL no se ejecuta en entrenamiento, así que su observador se asentaba solo en
    la EMA durante la validación y ModelEMA.update fallaba en la segunda época. Un observador de
    peso solo mira el parámetro, así que ejecutarlo aquí no cambia nada. Los de activación no se
    tocan.
    """
    import torch

    asentados = 0
    with torch.no_grad():
        for modulo in model.modules():
            simulador = getattr(modulo, "weight_fake_quant", None)
            if simulador is None:
                continue
            simulador(modulo.weight)
            asentados += 1
    logger.debug("Asentada la forma de %d observadores de peso.", asentados)
    return asentados


def _comprobar_gradiente(modelo: "nn.Module", resolution: int) -> bool:
    """Comprueba que el gradiente atraviesa la simulación, sobre una copia del modelo.

    La copia evita que el paso en modo entrenamiento con entrada aleatoria altere las estadísticas
    de BatchNorm del modelo bueno.
    """
    import torch

    copia = copy.deepcopy(modelo).train()
    copia.zero_grad(set_to_none=True)
    salida = copia(torch.randn(2, 3, resolution, resolution))
    tensor = salida[0] if isinstance(salida, (list, tuple)) else salida
    tensor.sum().backward()
    con_gradiente = sum(1 for p in copia.parameters() if p.grad is not None)
    del copia

    if con_gradiente == 0:
        logger.error(
            "NINGÚN parámetro recibe gradiente tras prepare_qat: el "
            "entrenamiento no aprendería nada."
        )
        return False
    logger.debug("%d parámetros reciben gradiente tras prepare_qat", con_gradiente)
    return True


def _devolver_modulos_a_flotante(modelo: "nn.Module") -> int:
    """Devuelve a su clase flotante los módulos de torch.ao.nn.qat con to_float().

    Desactivar la simulación no basta: la clase sigue siendo la de QAT y el trazador de ONNX recoge
    ``fused_moving_avg_obs_fake_quant``. Lanza QatError si algún módulo no tiene to_float.
    """
    convertidos = 0
    for nombre, hijo in list(modelo.named_children()):
        if type(hijo).__module__.startswith("torch.ao.nn.qat"):
            if not hasattr(hijo, "to_float"):
                raise QatError(
                    f"El módulo '{nombre}' es de tipo {type(hijo).__name__} "
                    f"({type(hijo).__module__}) y no ofrece to_float().",
                    "Sin volver a flotante el modelo no se puede exportar a ONNX. Añade el caso "
                    "aquí.",
                )
            setattr(modelo, nombre, hijo.to_float())
            convertidos += 1
        else:
            convertidos += _devolver_modulos_a_flotante(hijo)
    return convertidos


def _retirar_simulacion_suelta(modelo: "nn.Module") -> int:
    """Sustituye por nn.Identity los nodos de simulación que cuelgan sueltos en el grafo.

    Cubre lo que no quitan to_float() ni _remove_qconfig. Un observador devuelve su entrada, así que
    la sustitución es exacta.
    """
    from torch import nn as torch_nn

    qao = _importar_quantization()
    sueltos = (qao.FakeQuantizeBase, qao.ObserverBase)

    retirados = 0
    for nombre, hijo in list(modelo.named_children()):
        if isinstance(hijo, sueltos):
            setattr(modelo, nombre, torch_nn.Identity())
            retirados += 1
        else:
            retirados += _retirar_simulacion_suelta(hijo)
    return retirados


def assert_no_simulation_residue(model: "nn.Module") -> None:
    """Comprueba que no queda ningún módulo de torch.ao en el modelo."""
    restos: dict[str, int] = {}
    for modulo in model.modules():
        origen = type(modulo).__module__
        if origen.startswith("torch.ao"):
            clave = f"{origen}.{type(modulo).__name__}"
            restos[clave] = restos.get(clave, 0) + 1
    if restos:
        detalle = ", ".join(f"{n}x {t}" for t, n in sorted(restos.items()))
        raise QatError(
            f"Tras despojar el modelo siguen dentro módulos de torch.ao: {detalle}.",
            "El trazador de torch.onnx.export recoge la operación aunque su paso hacia delante "
            "sea la identidad, y la exportación falla.",
        )


def strip_fake_quantization(model: "nn.Module") -> "nn.Module":
    """Quita los nodos de cuantización simulada y devuelve un modelo nuevo con pesos FP32 afinados.

    Esos pesos siguen después el mismo camino de exportación y materialización INT8 que el baseline.
    Hay que quitar los nodos, no solo desactivarlos, porque fake_quant_enabled es un argumento
    tensorial y torch.onnx.export traza la operación igualmente.
    """
    import torch

    qao = _importar_quantization()

    modulos_simulacion = [m for m in model.modules() if isinstance(m, qao.FakeQuantizeBase)]
    if not modulos_simulacion:
        raise QatError(
            "El modelo no tiene cuantización simulada que quitar.",
            "Solo se puede aplicar a un modelo preparado con prepare_for_qat().",
        )

    limpio = copy.deepcopy(model)

    # 1. Los módulos QAT vuelven a su clase flotante con los mismos pesos.
    convertidos = _devolver_modulos_a_flotante(limpio)

    # 2. Los observadores de activación cuelgan también de módulos normales como hijo
    # 'activation_post_process' más un hook. Los quita PyTorch y abajo se comprueba el resultado.
    try:
        from torch.ao.quantization.quantize import _remove_qconfig
    except ImportError as error:  # pragma: no cover - solo al subir de torch
        raise QatError(
            f"torch.ao.quantization.quantize._remove_qconfig no existe: {error}",
            "Es una función privada disponible en torch==2.8.0. Al migrar a torchao hay que "
            "sustituirla por otra que quite el hijo 'activation_post_process' y su hook.",
        ) from error

    with _sin_aviso_de_obsolescencia():
        _remove_qconfig(limpio)

    # 3. Nodos sueltos en el grafo.
    sueltos = _retirar_simulacion_suelta(limpio)

    limpio.eval()

    for parametro in limpio.parameters():
        if parametro.dtype != torch.float32:
            raise QatError(
                f"Tras quitar la simulación queda un parámetro en "
                f"{parametro.dtype}, no en FP32.",
                "Los pesos afinados con QAT tienen que salir en FP32 para seguir el mismo "
                "camino que el baseline.",
            )

    assert_no_simulation_residue(limpio)

    logger.info(
        "Cuantización simulada retirada: %d nodos fuera, %d módulos devueltos a su clase "
        "flotante y %d nodos sueltos sustituidos por la identidad.",
        len(modulos_simulacion),
        convertidos,
        sueltos,
    )
    return limpio


def freeze_observers(model: "nn.Module") -> int:
    """Congela los observadores para que los rangos dejen de cambiar en las últimas épocas."""
    qao = _importar_quantization()
    with _sin_aviso_de_obsolescencia():
        model.apply(qao.disable_observer)
    return sum(1 for m in model.modules() if isinstance(m, qao.ObserverBase))


def freeze_batchnorm_stats(model: "nn.Module") -> int:
    """Congela las estadísticas de BatchNorm de los módulos ConvBn2d fusionados.

    En YOLOv8 sin fusionar, convolución y BN están separadas y la función devuelve 0; en ese caso se
    avisa.
    """
    import torch.ao.nn.intrinsic.qat as nniqat

    congelados = 0
    for modulo in model.modules():
        if isinstance(modulo, (nniqat.ConvBn2d, nniqat.ConvBnReLU2d)):
            modulo.freeze_bn_stats()
            congelados += 1

    if congelados == 0:
        logger.warning(
            "No se ha congelado ninguna BatchNorm: no hay módulos ConvBn2d fusionados. "
            "freeze_bn_epoch no tiene efecto sin fusionar antes de prepare_qat."
        )
    return congelados
