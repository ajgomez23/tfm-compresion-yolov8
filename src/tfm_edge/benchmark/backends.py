"""Interfaz común de los motores de inferencia.

Preprocesado y postprocesado son los mismos para todos los motores; solo cambia infer(), que es lo
que se cronometra y compara. Los tres tramos se miden por separado. Implementados: PyTorch, ONNX
Runtime y OpenVINO.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import numpy as np
    import torch

    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Módulos pesados importados una vez, para que la importación no caiga dentro de la medida.
_MODULOS: dict[str, Any] = {}


def _cargar_modulos() -> dict[str, Any]:
    """Importa y cachea los módulos que usa la ruta de inferencia."""
    if not _MODULOS:
        import cv2
        import numpy as np
        import torch
        from torchvision.ops import nms

        _MODULOS.update({"cv2": cv2, "np": np, "torch": torch, "nms": nms})
    return _MODULOS


class BackendError(TFMEdgeError):
    """Error de un motor de inferencia."""


class BackendNotAvailableError(TFMEdgeError):
    """Motor no disponible en este entorno. Se registra como no_disponible y se sigue con los
    demás."""

    def __init__(self, backend: str, motivo: str) -> None:
        super().__init__(
            f"El motor de inferencia '{backend}' no está disponible: {motivo}",
            "No es un error de la campaña. Se registrará con "
            "status=no_disponible y se continuará con los demás motores.",
        )
        self.backend = backend
        self.motivo = motivo


@dataclass
class InferenceOutput:
    """Salida de una inferencia con sus tiempos. ``inference_s`` es el que se compara entre
    motores."""

    num_detections: int = 0
    preprocess_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0
    raw: Any = None

    @property
    def total_s(self) -> float:
        """Segundos de la cadena completa."""
        return self.preprocess_s + self.inference_s + self.postprocess_s


class InferenceBackend(ABC):
    """Motor de inferencia con preprocesado y postprocesado comunes.

    Las subclases implementan load, infer y unload, y no deben sobrescribir preprocess ni
    postprocess.
    """

    name: str = "base"

    def __init__(self, input_resolution: int = 640, device: str = "cpu") -> None:
        """Configura el motor."""
        self.input_resolution = input_resolution
        self.device = device
        self._cargado = False

    @abstractmethod
    def load(self, artifact: Any) -> None:
        """Carga el modelo y lo deja listo para inferir."""

    @abstractmethod
    def infer(self, tensor: "torch.Tensor") -> Any:
        """Ejecuta solo el paso hacia delante, sin pre ni postprocesado."""

    def unload(self) -> None:
        """Libera el modelo. Por defecto no hace nada."""
        self._cargado = False

    @abstractmethod
    def synchronize(self) -> None:
        """Espera a que el dispositivo termine. En GPU, sin esto se mediría solo el encolado."""

    # Preprocesado y postprocesado comunes

    def preprocess(self, image: "np.ndarray") -> "torch.Tensor":
        """Letterbox a la resolución de entrada, normalización a [0, 1] y paso a NCHW. Igual para
        todos los motores.
        """
        modulos = _cargar_modulos()
        cv2, np, torch = modulos["cv2"], modulos["np"], modulos["torch"]

        alto, ancho = image.shape[:2]
        escala = min(self.input_resolution / ancho, self.input_resolution / alto)
        nuevo = (max(1, round(ancho * escala)), max(1, round(alto * escala)))
        redimensionada = cv2.resize(image, nuevo, interpolation=cv2.INTER_LINEAR)

        # Relleno centrado con el gris 114 de Ultralytics.
        lienzo = np.full((self.input_resolution, self.input_resolution, 3), 114, dtype=np.uint8)
        arriba = (self.input_resolution - nuevo[1]) // 2
        izquierda = (self.input_resolution - nuevo[0]) // 2
        lienzo[arriba : arriba + nuevo[1], izquierda : izquierda + nuevo[0]] = redimensionada

        tensor = torch.from_numpy(lienzo[:, :, ::-1].copy())  # BGR a RGB
        tensor = tensor.permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        return tensor.to(self.device)

    def postprocess(self, raw: Any, conf: float = 0.25, iou: float = 0.7) -> int:
        """Convierte la salida cruda en detecciones y devuelve cuántas hay.

        Solo interesa su coste; la calidad se evalúa en evaluation/.
        """
        modulos = _cargar_modulos()
        torch, nms = modulos["torch"], modulos["nms"]

        predicciones = raw[0] if isinstance(raw, (list, tuple)) else raw
        if not isinstance(predicciones, torch.Tensor):
            return 0
        # Salida de YOLOv8: (lote, 4 + clases, num_cajas).
        if predicciones.ndim != 3:
            return 0
        prediccion = predicciones[0].transpose(0, 1)  # (cajas, 4 + clases)
        cajas = prediccion[:, :4]
        puntuaciones, _ = prediccion[:, 4:].max(dim=1)
        mascara = puntuaciones > conf
        cajas, puntuaciones = cajas[mascara], puntuaciones[mascara]
        if cajas.numel() == 0:
            return 0
        # De centro y tamaño a esquinas, como espera nms.
        esquinas = torch.empty_like(cajas)
        esquinas[:, 0] = cajas[:, 0] - cajas[:, 2] / 2
        esquinas[:, 1] = cajas[:, 1] - cajas[:, 3] / 2
        esquinas[:, 2] = cajas[:, 0] + cajas[:, 2] / 2
        esquinas[:, 3] = cajas[:, 1] + cajas[:, 3] / 2
        return int(nms(esquinas, puntuaciones, iou).numel())

    def run(
        self, image: "np.ndarray", conf: float = 0.25, iou: float = 0.7, keep_raw: bool = False
    ) -> InferenceOutput:
        """Ejecuta la cadena completa y cronometra cada tramo."""
        inicio = time.perf_counter()
        tensor = self.preprocess(image)
        self.synchronize()
        tras_preproceso = time.perf_counter()

        crudo = self.infer(tensor)
        self.synchronize()
        tras_inferencia = time.perf_counter()

        detecciones = self.postprocess(crudo, conf=conf, iou=iou)
        self.synchronize()
        fin = time.perf_counter()

        return InferenceOutput(
            num_detections=detecciones,
            preprocess_s=tras_preproceso - inicio,
            inference_s=tras_inferencia - tras_preproceso,
            postprocess_s=fin - tras_inferencia,
            raw=crudo if keep_raw else None,
        )


class PyTorchBackend(InferenceBackend):
    """Motor de referencia: PyTorch en modo evaluación, sin torch.compile ni fusiones extra."""

    name = "pytorch"

    def __init__(self, input_resolution: int = 640, device: str = "cpu") -> None:
        """Configura el motor."""
        super().__init__(input_resolution=input_resolution, device=device)
        self._modelo: Any = None

    def load(self, artifact: Any) -> None:
        """Carga un Detector o un nn.Module."""
        from torch import nn

        modelo = getattr(artifact, "model", artifact)
        if not isinstance(modelo, nn.Module):
            raise BackendError(
                f"El artefacto es {type(artifact).__name__} y no un modelo de PyTorch.",
                "Pasa un Detector o un nn.Module.",
            )
        self._modelo = modelo.to(self.device).eval()
        self._cargado = True
        logger.info(
            "Motor '%s' cargado en %s a %d px", self.name, self.device, self.input_resolution
        )

    def infer(self, tensor: "torch.Tensor") -> Any:
        """Ejecuta el paso hacia delante."""
        import torch

        if self._modelo is None:
            raise BackendError("El motor no tiene modelo cargado.", "Llama antes a load().")
        with torch.inference_mode():
            return self._modelo(tensor)

    def synchronize(self) -> None:
        """Sincroniza CUDA si el dispositivo es una GPU."""
        import torch

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def unload(self) -> None:
        """Libera el modelo y vacía la caché de CUDA."""
        import torch

        self._modelo = None
        self._cargado = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OnnxRuntimeBackend(InferenceBackend):
    """Motor ONNX Runtime. Solo cambia infer() respecto a la clase base."""

    name = "onnxruntime"

    def __init__(self, input_resolution: int = 640, device: str = "cpu") -> None:
        """Configura el motor; ``cuda`` selecciona el proveedor de GPU."""
        super().__init__(input_resolution=input_resolution, device=device)
        self._sesion: Any = None
        self._nombre_entrada = ""
        self.providers: list[str] = []

    def load(self, artifact: Any) -> None:
        """Carga un modelo ``.onnx``."""
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise BackendNotAvailableError("onnxruntime", "no está instalado") from error

        ruta = Path(artifact)
        if not ruta.is_file():
            raise BackendError(
                f"No existe el artefacto ONNX '{ruta}'.",
                "Expórtalo antes con scripts/08_export.py.",
            )

        quiere_gpu = str(self.device).startswith("cuda")
        deseado = "CUDAExecutionProvider" if quiere_gpu else "CPUExecutionProvider"
        disponibles = ort.get_available_providers()
        if deseado not in disponibles:
            # Si falta el proveedor de GPU se falla en vez de caer a CPU sin avisar.
            raise BackendError(
                f"El proveedor '{deseado}' no está disponible en esta "
                f"instalación de onnxruntime. Disponibles: {disponibles}.",
                "Para GPU hace falta el paquete 'onnxruntime-gpu'.",
            )

        self.providers = [deseado]
        self._sesion = ort.InferenceSession(str(ruta), providers=self.providers)
        self._nombre_entrada = self._sesion.get_inputs()[0].name
        self._cargado = True
        logger.info("Motor '%s' cargado con %s", self.name, self.providers)

    def infer(self, tensor: "torch.Tensor") -> Any:
        """Ejecuta el paso hacia delante y devuelve la salida como tensor de PyTorch."""
        import torch

        if self._sesion is None:
            raise BackendError("El motor no tiene modelo cargado.", "Llama antes a load().")
        entrada = tensor.detach().cpu().numpy()
        salida = self._sesion.run(None, {self._nombre_entrada: entrada})[0]
        return torch.from_numpy(salida)

    def synchronize(self) -> None:
        """ONNX Runtime es síncrono: no hay nada que esperar."""

    def unload(self) -> None:
        """Libera la sesión."""
        self._sesion = None
        self._cargado = False


class OpenVinoBackend(InferenceBackend):
    """Motor OpenVINO para CPU Intel."""

    name = "openvino"

    def __init__(self, input_resolution: int = 640, device: str = "cpu") -> None:
        """Configura el motor; el dispositivo se traduce a ``CPU`` o ``GPU``."""
        super().__init__(input_resolution=input_resolution, device=device)
        self._compilado: Any = None
        self._salida: Any = None

    def load(self, artifact: Any) -> None:
        """Carga un artefacto de OpenVINO a partir de su ``.xml``."""
        try:
            import openvino as ov
        except ImportError as error:
            raise BackendNotAvailableError("openvino", "no está instalado") from error

        ruta = Path(artifact)
        if not ruta.is_file():
            raise BackendError(
                f"No existe el artefacto de OpenVINO '{ruta}'.",
                "Expórtalo antes con scripts/08_export.py.",
            )

        objetivo = "GPU" if str(self.device).startswith("cuda") else "CPU"
        nucleo = ov.Core()
        if objetivo not in nucleo.available_devices:
            raise BackendError(
                f"OpenVINO no ve el dispositivo '{objetivo}'. "
                f"Disponibles: {nucleo.available_devices}.",
                "No se cambia de dispositivo en silencio: la medida quedaría mal " "etiquetada.",
            )
        self._compilado = nucleo.compile_model(nucleo.read_model(str(ruta)), objetivo)
        self._salida = self._compilado.output(0)
        self._cargado = True
        logger.info("Motor '%s' cargado sobre %s", self.name, objetivo)

    def infer(self, tensor: "torch.Tensor") -> Any:
        """Ejecuta el paso hacia delante y devuelve la salida como tensor de PyTorch."""
        import torch

        if self._compilado is None:
            raise BackendError("El motor no tiene modelo cargado.", "Llama antes a load().")
        salida = self._compilado(tensor.detach().cpu().numpy())[self._salida]
        return torch.from_numpy(salida)

    def synchronize(self) -> None:
        """La llamada síncrona de OpenVINO ya devuelve el resultado completo."""

    def unload(self) -> None:
        """Libera el modelo compilado."""
        self._compilado = None
        self._salida = None
        self._cargado = False


#: Motores implementados.
AVAILABLE_BACKENDS: dict[str, type[InferenceBackend]] = {
    "pytorch": PyTorchBackend,
    "onnxruntime": OnnxRuntimeBackend,
    "openvino": OpenVinoBackend,
}


def build_backend(name: str, input_resolution: int = 640, device: str = "cpu") -> InferenceBackend:
    """Construye un motor a partir de su nombre."""
    if name not in AVAILABLE_BACKENDS:
        raise BackendNotAvailableError(
            name,
            f"no está implementado. Disponibles: {sorted(AVAILABLE_BACKENDS)}.",
        )
    return AVAILABLE_BACKENDS[name](input_resolution=input_resolution, device=device)
