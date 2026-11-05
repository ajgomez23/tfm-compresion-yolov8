"""Capa de abstracción sobre Ultralytics; es el único módulo que importa ``ultralytics``.

Así la poda, la QAT y la destilación pueden modificar el modelo sin depender de la API de
Ultralytics. ``DetectionModel.model`` es un nn.Sequential de 23 capas (verificado con
ultralytics==8.3.253): 0 a 9 backbone, 10 a 21 cuello y 22 la cabeza Detect. El YAML de Ultralytics
llama head a todo el bloque 10 a 22; aquí se separa el cuello porque se poda y la cabeza no. Estos
índices no son API pública y tests/test_detector_structure.py los comprueba.
"""

from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch import nn

from tfm_edge.common.exceptions import DependencyNotAvailableError, TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_parent_dir

logger = get_logger(__name__)

#: Versión de Ultralytics con la que se ha verificado la estructura.
ANCHORED_ULTRALYTICS_VERSION = "8.3.253"

VALID_SIZES = ("n", "s", "m", "l")

BACKBONE_END_INDEX = 10

#: Frontera entre cuello y cabeza; la cabeza es la última capa.
NECK_END_INDEX = 22

EXPECTED_NUM_LAYERS = 23

#: Opset de ONNX por defecto. Se elige el 17 por compatibilidad con los motores de destino, no por
#: ser el más reciente.
DEFAULT_ONNX_OPSET = 17


class DetectorError(TFMEdgeError):
    """Error de la capa de abstracción del detector."""


class UnexpectedModelStructureError(TFMEdgeError):
    """La estructura interna del modelo no es la esperada (normalmente, por un cambio de
    Ultralytics).
    """

    def __init__(self, detalle: str, version_actual: str) -> None:
        super().__init__(
            f"La estructura interna del modelo no es la esperada: {detalle}",
            (
                f"El proyecto ancla su abstracción a ultralytics=="
                f"{ANCHORED_ULTRALYTICS_VERSION} y la versión instalada es "
                f"{version_actual}.\n"
                "     La estructura interna de Ultralytics NO es API pública y puede "
                "cambiar entre versiones.\n"
                "     Revisa src/tfm_edge/models/detector.py y los tests de "
                "estructura antes de continuar: la poda y la destilación dependen "
                "de estos índices."
            ),
        )


def _importar_ultralytics() -> Any:
    """Importa Ultralytics o lanza DependencyNotAvailableError."""
    try:
        import ultralytics
    except ImportError as error:
        raise DependencyNotAvailableError(
            "ultralytics",
            "construir o cargar un detector",
            f"Instálalo con: pip install ultralytics=={ANCHORED_ULTRALYTICS_VERSION}",
        ) from error
    return ultralytics


def installed_ultralytics_version() -> str:
    """Versión de Ultralytics instalada, o ``"desconocida"``."""
    return getattr(_importar_ultralytics(), "__version__", "desconocida")


#: Gris de las bandas del letterbox de Ultralytics.
LETTERBOX_PADDING_VALUE = 114


@dataclass(frozen=True)
class LetterboxTransform:
    """Geometría del letterbox: escala ``min(R/W, R/H)``, igual en los dos ejes, y relleno."""

    scale: float
    pad_x: int
    pad_y: int
    resolution: int

    def map_box(self, x: float, y: float, width: float, height: float) -> tuple[float, ...]:
        """Lleva una caja de la imagen original al lienzo, conservando su relación de aspecto."""
        return (
            x * self.scale + self.pad_x,
            y * self.scale + self.pad_y,
            width * self.scale,
            height * self.scale,
        )


def letterbox_image(
    image: np.ndarray,
    resolution: int = 640,
    padding_value: int = LETTERBOX_PADDING_VALUE,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Aplica el letterbox de Ultralytics (LetterBox) y devuelve el lienzo y su geometría.

    Se usa ``auto=False`` para que el lienzo salga cuadrado. El relleno se mide pasando una imagen
    uniforme por la misma transformación y se contrasta con la fórmula analítica, para detectar un
    cambio de redondeo o centrado en Ultralytics.
    """
    _importar_ultralytics()
    try:
        from ultralytics.data.augment import LetterBox
    except ImportError as error:  # pragma: no cover - depende de la versión
        raise DependencyNotAvailableError(
            "ultralytics.data.augment.LetterBox",
            "aplicar el letterbox de Ultralytics",
            f"Anclado a ultralytics=={ANCHORED_ULTRALYTICS_VERSION}.",
        ) from error

    if image.ndim != 3:
        raise DetectorError(
            f"Se esperaba una imagen HxWxC y ha llegado una de {image.ndim} dimensiones.",
            "Convierte la imagen a RGB o BGR antes de llamar.",
        )

    alto, ancho = image.shape[:2]
    operacion = LetterBox(
        new_shape=(resolution, resolution),
        auto=False,
        scale_fill=False,
        scaleup=True,
        center=True,
        padding_value=padding_value,
    )
    lienzo = operacion(image=image)
    if lienzo.shape[:2] != (resolution, resolution):
        raise DetectorError(
            f"El letterbox ha devuelto un lienzo de {lienzo.shape[:2]} en lugar de "
            f"({resolution}, {resolution}).",
            "Revisa los argumentos de LetterBox: con auto=True el relleno no es cuadrado.",
        )

    # Imagen sonda uniforme con un valor distinto del relleno; tras el letterbox, lo que no es
    # relleno es el contenido.
    valor_sonda = 0 if padding_value != 0 else 255
    sonda = operacion(image=np.full_like(image, valor_sonda))
    contenido = np.any(sonda != padding_value, axis=2)
    filas = np.flatnonzero(contenido.any(axis=1))
    columnas = np.flatnonzero(contenido.any(axis=0))
    if filas.size == 0 or columnas.size == 0:
        raise DetectorError(
            "La imagen sonda no ha dejado contenido reconocible tras el letterbox.",
            "Comprueba el valor de padding_value y la versión de Ultralytics.",
        )
    pad_x, pad_y = int(columnas[0]), int(filas[0])
    medido_ancho = int(columnas[-1] - columnas[0] + 1)
    medido_alto = int(filas[-1] - filas[0] + 1)

    escala = min(resolution / ancho, resolution / alto)
    esperado_ancho, esperado_alto = round(ancho * escala), round(alto * escala)
    if abs(medido_ancho - esperado_ancho) > 1 or abs(medido_alto - esperado_alto) > 1:
        raise DetectorError(
            "La geometría del letterbox medida sobre la salida de Ultralytics "
            f"({medido_ancho}x{medido_alto} px de contenido) no coincide con la analítica "
            f"({esperado_ancho}x{esperado_alto} px).",
            f"Revisa letterbox_image (verificado con {ANCHORED_ULTRALYTICS_VERSION}).",
        )
    return lienzo, LetterboxTransform(scale=escala, pad_x=pad_x, pad_y=pad_y, resolution=resolution)


#: Atributo donde Ultralytics cachea la función de pérdida dentro del modelo. No es submódulo ni
#: buffer, así que .to(device) no lo mueve y torch.save sí lo guarda.
CACHED_LOSS_ATTR = "criterion"

#: Atributo donde prepare_qat deja el qconfig de cada módulo; torch.save no lo puede serializar.
QCONFIG_ATTR = "qconfig"


def drop_cached_loss(model: nn.Module) -> bool:
    """Quita la función de pérdida que Ultralytics cachea en el modelo.

    v8DetectionLoss fija su dispositivo al construirse y ``self.proj`` es un tensor suelto que
    ``model.to()`` no mueve. Un checkpoint guardado durante el entrenamiento la incluye; al
    recargarlo y entrenar en GPU, falla con "Expected all tensors to be on the same device" en la
    proyección DFL de bbox_decode. Devuelve True si había una pérdida cacheada.
    """
    perdida = getattr(model, CACHED_LOSS_ATTR, None)
    if perdida is None:
        return False
    delattr(model, CACHED_LOSS_ATTR)
    logger.info(
        "Quitada la pérdida cacheada (%s); Ultralytics la reconstruirá en el dispositivo correcto.",
        type(perdida).__name__,
    )
    return True


def drop_quantization_config(model: nn.Module) -> int:
    """Quita el qconfig que prepare_qat pone en cada módulo, en la copia que va a disco.

    Contiene una función local que pickle no puede serializar, así que el checkpoint no se podría
    escribir. Los rangos aprendidos están en los módulos FakeQuantize, que se conservan. Devuelve el
    número de módulos afectados.
    """
    quitados = 0
    for modulo in model.modules():
        if getattr(modulo, QCONFIG_ATTR, None) is not None:
            delattr(modulo, QCONFIG_ATTR)
            quitados += 1
    if quitados:
        logger.info(
            "Quitado el qconfig de %d módulos de la copia que va a disco; los nodos de "
            "simulación se conservan.",
            quitados,
        )
    return quitados


def drop_forward_hooks(model: nn.Module) -> int:
    """Quita los forward hooks de todos los submódulos, en la copia que va a disco.

    Los hooks se guardan en el ``__dict__`` del módulo y uno definido como función local impide
    serializar el checkpoint. Devuelve el número de hooks quitados.
    """
    total = 0
    for modulo in model.modules():
        total += len(modulo._forward_hooks)
        modulo._forward_hooks.clear()
        total += len(modulo._forward_pre_hooks)
        modulo._forward_pre_hooks.clear()
    if total:
        logger.info(
            "Quitados %d hooks de la copia que va a disco.",
            total,
        )
    return total


@lru_cache(maxsize=1)
def _ultralytics_scale_table() -> dict[str, tuple[float, float]]:
    """Multiplicadores de profundidad y anchura de cada escala, leídos del yolov8.yaml de
    Ultralytics.

    Devuelve ``escala: (depth_multiple, width_multiple)``, vacío si no encuentra el YAML.
    """
    _importar_ultralytics()
    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    ruta = check_yaml("yolov8.yaml", hard=False)
    if not ruta:
        logger.warning(
            "No se encuentra 'yolov8.yaml' en el Ultralytics instalado: los "
            "multiplicadores no podrán usarse para deducir el tamaño del modelo."
        )
        return {}
    tabla: dict[str, tuple[float, float]] = {}
    for nombre, valores in (YAML.load(ruta).get("scales") or {}).items():
        if isinstance(valores, (list, tuple)) and len(valores) >= 2:
            tabla[str(nombre)] = (float(valores[0]), float(valores[1]))
    return tabla


def infer_model_size(model: nn.Module) -> str | None:
    """Deduce el tamaño de un modelo a partir de su arquitectura.

    Los checkpoints del proyecto no guardan el tamaño. Se prueba, en orden, ``model.yaml['scale']``,
    ``model.yaml['yaml_file']`` y los multiplicadores de profundidad y anchura. Devuelve None si no
    hay pistas; nunca un valor por defecto.
    """
    yaml_modelo = getattr(model, "yaml", None)
    if not isinstance(yaml_modelo, dict):
        return None

    escala = str(yaml_modelo.get("scale") or "").strip()
    if escala:
        return escala

    fichero = yaml_modelo.get("yaml_file")
    if fichero:
        _importar_ultralytics()
        from ultralytics.nn.tasks import guess_model_scale

        deducida = guess_model_scale(fichero)
        if deducida:
            return deducida

    profundidad = yaml_modelo.get("depth_multiple")
    anchura = yaml_modelo.get("width_multiple")
    if profundidad is None or anchura is None:
        return None
    coincidencias = [
        nombre
        for nombre, (d, w) in _ultralytics_scale_table().items()
        if math.isclose(float(profundidad), d) and math.isclose(float(anchura), w)
    ]
    # Si dos escalas comparten multiplicadores no se puede distinguir.
    return coincidencias[0] if len(coincidencias) == 1 else None


class Detector:
    """Envoltorio sobre un detector YOLOv8 de Ultralytics con acceso a sus módulos por nombre."""

    def __init__(
        self,
        model: nn.Module,
        size: str,
        num_classes: int,
        input_resolution: int = 640,
        validate_structure: bool = True,
    ) -> None:
        """Envuelve un modelo ya construido; normalmente se usa from_architecture o load."""
        self._model = model
        self.size = size
        self.num_classes = num_classes
        self.input_resolution = input_resolution
        if validate_structure:
            self.validate_structure()

    @classmethod
    def from_architecture(
        cls,
        size: str,
        num_classes: int,
        input_resolution: int = 640,
        pretrained: bool = False,
    ) -> "Detector":
        """Construye un detector desde su YAML de arquitectura, con o sin pesos de COCO."""
        if size not in VALID_SIZES:
            raise DetectorError(
                f"Tamaño de modelo no válido: '{size}'.",
                f"Los tamaños admitidos son: {', '.join(VALID_SIZES)}.",
            )
        _importar_ultralytics()
        from ultralytics import YOLO

        # Con pesos preentrenados se parte del .pt; sin ellos, del YAML, que no necesita red.
        referencia = f"yolov8{size}.pt" if pretrained else f"yolov8{size}.yaml"
        logger.info("Construyendo yolov8%s desde '%s'", size, referencia)
        yolo = YOLO(referencia)
        modelo = yolo.model

        if num_classes != getattr(modelo, "nc", None):
            modelo = cls._reconfigurar_clases(modelo, num_classes, size, pretrained)

        return cls(modelo, size=size, num_classes=num_classes, input_resolution=input_resolution)

    @staticmethod
    def _reconfigurar_clases(
        modelo: nn.Module, num_classes: int, size: str, pretrained: bool
    ) -> nn.Module:
        """Reconstruye el modelo con el número de clases del estudio."""
        from ultralytics.nn.tasks import DetectionModel

        logger.info("Reconfigurando la cabeza para %d clases", num_classes)
        nuevo = DetectionModel(f"yolov8{size}.yaml", nc=num_classes, verbose=False)
        if pretrained:
            # load() copia solo los parámetros con la misma forma: troncal y cuello conservan los
            # pesos de COCO y la cabeza se reinicializa.
            nuevo.load(modelo)
            logger.info("Pesos preentrenados transferidos salvo en la cabeza de detección")
        return nuevo

    @classmethod
    def load(
        cls,
        weights_path: Path | str,
        input_resolution: int = 640,
        map_location: str = "cpu",
        size: str | None = None,
    ) -> "Detector":
        """Carga un detector desde un checkpoint del proyecto.

        El tamaño se deduce de la arquitectura; si además viene declarado (argumento ``size`` o
        metadatos) tiene que coincidir. Los pesos oficiales de Ultralytics necesitan ``size`` porque
        no declaran la escala.
        """
        ruta = Path(weights_path)
        if not ruta.is_file():
            raise DetectorError(
                f"No existe el checkpoint '{ruta}'.",
                "Comprueba la ruta, o entrena el modelo con scripts/04_train_baselines.py.",
            )
        _importar_ultralytics()
        # weights_only=False porque el checkpoint guarda el módulo entero; lo genera el propio
        # proyecto.
        contenido = torch.load(ruta, map_location=map_location, weights_only=False)

        if isinstance(contenido, dict):
            modelo = contenido.get("model") or contenido.get("ema")
            metadatos = contenido.get("tfm_edge", {})
        else:
            modelo = contenido
            metadatos = {}
        if not isinstance(modelo, nn.Module):
            raise DetectorError(
                f"El checkpoint '{ruta}' no contiene un modelo utilizable.",
                "Se esperaba una clave 'model' o 'ema' con un nn.Module.",
            )

        modelo = modelo.float()  # los checkpoints propios ya son FP32
        # Se quita la pérdida cacheada, que ataría el modelo al dispositivo donde se guardó.
        drop_cached_loss(modelo)
        return cls(
            modelo,
            size=cls._resolver_size(ruta, modelo, metadatos, declarado=size),
            num_classes=metadatos.get("num_classes", getattr(modelo, "nc", 0)),
            input_resolution=metadatos.get("input_resolution", input_resolution),
        )

    @staticmethod
    def _resolver_size(
        ruta: Path, modelo: nn.Module, metadatos: dict[str, Any], declarado: str | None
    ) -> str:
        """Decide el tamaño de un checkpoint o falla si no se puede saber o las fuentes se
        contradicen.
        """
        deducido = infer_model_size(modelo)
        declaraciones = {
            "el argumento 'size'": (declarado or "").strip() or None,
            "los metadatos del checkpoint": (str(metadatos.get("size") or "").strip() or None),
        }
        for origen, valor in declaraciones.items():
            if valor and deducido and valor != deducido:
                raise DetectorError(
                    f"'{ruta.name}' dice ser de tamaño '{valor}' según {origen}, "
                    f"pero su arquitectura es la de yolov8{deducido}.",
                    "Comprueba de dónde salió el fichero; una etiqueta equivocada contaminaría "
                    "el experiment_id.",
                )
        distintos = {valor for valor in declaraciones.values() if valor}
        if deducido is None and len(distintos) > 1:
            raise DetectorError(
                f"Sobre '{ruta.name}' se afirman dos tamaños distintos: "
                f"{sorted(distintos)}, y su arquitectura no permite decidir.",
                "Comprueba qué fichero es antes de seguir.",
            )

        size = deducido or next(iter(distintos), None)
        if size is None:
            raise DetectorError(
                f"No se puede saber de qué tamaño es el modelo de '{ruta.name}'.",
                "Ni sus metadatos ni su arquitectura lo indican. Pásalo explícitamente: "
                "Detector.load(ruta, size='s').",
            )
        if size not in VALID_SIZES:
            raise DetectorError(
                f"'{ruta.name}' es un yolov8{size} y el estudio solo trabaja con "
                f"los tamaños {', '.join(VALID_SIZES)}.",
                "Un experiment_id con otra escala lo rechazaría build_experiment_id.",
            )
        return size

    @classmethod
    def from_pretrained_file(
        cls,
        weights_path: Path | str,
        size: str,
        num_classes: int,
        input_resolution: int = 640,
        map_location: str = "cpu",
    ) -> "Detector":
        """Construye un detector desde un .pt local y adapta el número de clases.

        Es el camino del baseline: los pesos oficiales traen las 80 clases de COCO y se adaptan a
        17. A diferencia de from_architecture, el fichero lo aporta quien llama.
        """
        if size not in VALID_SIZES:
            raise DetectorError(
                f"Tamaño de modelo no válido: '{size}'.",
                f"Los tamaños admitidos son: {', '.join(VALID_SIZES)}.",
            )
        # size se pasa a load() para que contraste lo declarado con la arquitectura.
        original = cls.load(
            weights_path,
            input_resolution=input_resolution,
            map_location=map_location,
            size=size,
        )
        if original.num_classes == num_classes:
            return original

        _importar_ultralytics()
        logger.info(
            "'%s' trae %d clases y el estudio necesita %d.",
            Path(weights_path).name,
            original.num_classes,
            num_classes,
        )
        modelo = cls._reconfigurar_clases(original.model, num_classes, size, pretrained=True)
        return cls(modelo, size=size, num_classes=num_classes, input_resolution=input_resolution)

    def validate(
        self,
        data_config: Path | str,
        split: str = "val",
        batch_size: int = 16,
        device: str = "",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        save_dir: Path | str | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evalúa este modelo con el validador de Ultralytics y devuelve métricas y AP por clase.

        El umbral de confianza por defecto es 0,001, el del protocolo COCO; uno más alto recortaría
        la cola y subiría la precisión a costa del recall.
        """
        if split not in ("val", "test", "train"):
            raise DetectorError(
                f"Partición '{split}' no válida.",
                "Usa 'val' o 'test'. Evaluar sobre 'train' solo tiene sentido "
                "para diagnosticar sobreajuste, nunca para la memoria.",
            )
        _importar_ultralytics()
        from ultralytics.models.yolo.detect import DetectionValidator

        argumentos = build_validator_args(
            data_config=data_config,
            split=split,
            batch_size=batch_size,
            input_resolution=self.input_resolution,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            save_dir=save_dir,
        )
        argumentos.update(extra_args or {})

        estaba_entrenando = self._model.training
        self._model.eval()
        try:
            validador = DetectionValidator(args=argumentos)
            validador(model=self._model)
        except Exception as error:
            raise DetectorError(
                f"La evaluación ha fallado: {type(error).__name__}: {error}",
                "Comprueba que el YAML de dataset es válido y que la partición "
                f"'{split}' tiene imágenes.",
            ) from error
        finally:
            self._model.train(estaba_entrenando)

        return self._extraer_metricas(validador, split, conf_threshold, iou_threshold)

    def predict_coco_detections(
        self,
        data_config: Path | str,
        split: str = "val",
        batch_size: int = 16,
        device: str = "",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        save_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Genera las detecciones en formato COCO en coordenadas de la imagen original.

        El reescalado lo hace Ultralytics (``ops.scale_boxes``) antes de ``pred_to_json``; no se
        reimplementa.
        """
        _importar_ultralytics()
        from ultralytics.models.yolo.detect import DetectionValidator

        # save_json hace que el validador acumule las detecciones en jdict.
        argumentos = build_validator_args(
            data_config=data_config,
            split=split,
            batch_size=batch_size,
            input_resolution=self.input_resolution,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            save_dir=save_dir,
            max_detections=max_detections,
            save_json=True,
        )

        estaba_entrenando = self._model.training
        self._model.eval()
        try:
            validador = DetectionValidator(args=argumentos)
            validador(model=self._model)
        except Exception as error:
            raise DetectorError(
                f"La generación de detecciones ha fallado: " f"{type(error).__name__}: {error}",
                "Comprueba el YAML de dataset y que la partición tenga imágenes.",
            ) from error
        finally:
            self._model.train(estaba_entrenando)

        detecciones = self._normalizar_categorias(
            list(getattr(validador, "jdict", []) or []), validador
        )
        if not detecciones:
            raise DetectorError(
                "El validador no ha producido ninguna detección.",
                "Con conf=0.001 un modelo entrenado debería producir miles. Si no está sin "
                "entrenar, comprueba que 'save_json' sigue alimentando 'jdict' en esta versión "
                "de Ultralytics.",
            )

        return {
            "detections": detecciones,
            "num_detections": len(detecciones),
            # Se devuelven también las métricas del validador, calculadas sobre las mismas
            # detecciones.
            "ultralytics_metrics": extract_metrics(
                validador, split, conf_threshold, iou_threshold, self.input_resolution
            ),
            "split": split,
            "input_resolution": self.input_resolution,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
        }

    @contextmanager
    def export_mode(self) -> Iterator[nn.Module]:
        """Pone el modelo en modo exportación y lo restaura al salir, también si hay excepción.

        Con ``export = True`` la cabeza Detect devuelve un único tensor ``(lote, 4 + clases,
        cajas)``.
        """
        _importar_ultralytics()
        from ultralytics.nn.modules import Detect

        cabezas = [m for m in self._model.modules() if isinstance(m, Detect)]
        if not cabezas:
            raise UnexpectedModelStructureError(
                "no se ha encontrado ninguna cabeza Detect en el modelo",
                installed_ultralytics_version(),
            )

        estaba_entrenando = self._model.training
        previos = [(m, getattr(m, "export", False), getattr(m, "format", None)) for m in cabezas]
        try:
            self._model.eval()
            for modulo in cabezas:
                modulo.export = True
                modulo.format = "onnx"
            yield self._model
        finally:
            for modulo, export_previo, formato_previo in previos:
                modulo.export = export_previo
                if formato_previo is not None:
                    modulo.format = formato_previo
            self._model.train(estaba_entrenando)

    def export_onnx(
        self,
        output_path: Path | str,
        opset: int | None = None,
        dynamic_batch: bool = False,
        simplify: bool = True,
    ) -> dict[str, Any]:
        """Exporta este modelo a ONNX y devuelve la ruta, el opset y las formas.

        Por defecto el lote es fijo, porque el protocolo mide con lote 1.
        """
        try:
            import onnx
        except ImportError as error:
            raise DependencyNotAvailableError(
                "onnx",
                "exportar a ONNX",
                "Instálalo con: pip install onnx==1.22.0",
            ) from error

        destino = ensure_parent_dir(Path(output_path).with_suffix(".onnx"))

        if opset is None:
            opset = DEFAULT_ONNX_OPSET
            # Ultralytics elegiría el opset más alto que admite onnx (22), pero torch.onnx solo
            # soporta hasta 20 y TensorRT y TFLite van por detrás. Se fija uno compatible.
            try:
                from ultralytics.engine.exporter import best_onnx_opset

                sugerido = best_onnx_opset(onnx, cuda=str(self.device).startswith("cuda"))
                if sugerido != opset:
                    logger.info(
                        "Ultralytics sugiere el opset %d; se usa el %d por "
                        "portabilidad entre motores. Pasa 'opset' para forzarlo.",
                        sugerido,
                        opset,
                    )
            except Exception:  # pragma: no cover - depende de la versión
                pass

        entrada = torch.zeros(1, 3, self.input_resolution, self.input_resolution)
        ejes = {"images": {0: "batch"}, "output0": {0: "batch"}} if dynamic_batch else None

        with self.export_mode() as modelo:
            modelo = modelo.cpu().float()
            entrada = entrada.cpu()
            with torch.no_grad():
                salida_referencia = modelo(entrada)
            try:
                # Se usa el exportador basado en TorchScript y no el de torch.export (dynamo=True),
                # porque la cabeza Detect tiene flujo de control dependiente de la forma. Solo se
                # silencia ese aviso. TODO(torch-2.9): al cambiar de versión hay que reexportar y
                # volver a comprobar la equivalencia.
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*legacy TorchScript-based ONNX export.*",
                        category=DeprecationWarning,
                    )
                    torch.onnx.export(
                        modelo,
                        entrada,
                        str(destino),
                        opset_version=opset,
                        input_names=["images"],
                        output_names=["output0"],
                        dynamic_axes=ejes,
                        do_constant_folding=True,
                    )
            except Exception as error:
                raise DetectorError(
                    f"La exportación a ONNX ha fallado: {type(error).__name__}: {error}",
                    "Prueba con otro opset. Si vienes de una poda, valida antes "
                    "la estructura con un paso hacia delante.",
                ) from error

        modelo_onnx = onnx.load(str(destino))
        onnx.checker.check_model(modelo_onnx)

        simplificado = False
        if simplify:
            try:
                import onnxslim

                onnx.save(onnxslim.slim(modelo_onnx), str(destino))
                simplificado = True
            except Exception as error:  # pragma: no cover - onnxslim opcional
                logger.warning(
                    "No se ha podido simplificar el grafo (%s: %s). El ONNX sin "
                    "simplificar es válido; solo será algo más lento.",
                    type(error).__name__,
                    error,
                )

        forma_salida = tuple(salida_referencia.shape) if hasattr(salida_referencia, "shape") else ()
        logger.info("ONNX exportado en %s (opset %d, salida %s)", destino, opset, forma_salida)
        return {
            "path": str(destino),
            "opset": int(opset),
            "dynamic_batch": dynamic_batch,
            "simplified": simplificado,
            "input_shape": (1, 3, self.input_resolution, self.input_resolution),
            "output_shape": forma_salida,
            "onnx_version": onnx.__version__,
        }

    def reference_output(self, batch_size: int = 1) -> "torch.Tensor":
        """Salida de referencia en modo exportación, para comparar con la de cada motor."""
        with self.export_mode() as modelo:
            entrada = torch.zeros(
                batch_size, 3, self.input_resolution, self.input_resolution, device=self.device
            )
            with torch.no_grad():
                return modelo(entrada)

    def _normalizar_categorias(
        self, detecciones: list[dict[str, Any]], validador: Any
    ) -> list[dict[str, Any]]:
        """Traduce las categorías de Ultralytics a índices de clase del proyecto."""
        return normalize_categories(detecciones, validador, self.num_classes)

    def _extraer_metricas(
        self, validador: Any, split: str, conf_threshold: float, iou_threshold: float
    ) -> dict[str, Any]:
        """Extrae las métricas de un validador ya ejecutado."""
        return extract_metrics(
            validador, split, conf_threshold, iou_threshold, self.input_resolution
        )

    def save(self, path: Path | str, extra: dict[str, Any] | None = None) -> Path:
        """Guarda el detector en FP32 (Ultralytics guardaría en media precisión)."""
        destino = ensure_parent_dir(path)
        modelo = self._model.float()
        # No se guarda la pérdida cacheada; Ultralytics la reconstruye al volver a entrenar.
        drop_cached_loss(modelo)
        # Tampoco el qconfig, que impediría escribir el fichero.
        drop_quantization_config(modelo)
        contenido = {
            "model": modelo,
            "tfm_edge": {
                "size": self.size,
                "num_classes": self.num_classes,
                "input_resolution": self.input_resolution,
                "ultralytics_version": installed_ultralytics_version(),
                "saved_dtype": "float32",
                **(extra or {}),
            },
        }
        torch.save(contenido, destino)
        logger.info("Detector guardado en FP32 en %s", destino)
        return destino

    @property
    def model(self) -> nn.Module:
        """El modelo envuelto de Ultralytics."""
        return self._model

    @property
    def layers(self) -> nn.Sequential:
        """El nn.Sequential interno de capas."""
        capas = getattr(self._model, "model", None)
        if not isinstance(capas, nn.Sequential):
            raise UnexpectedModelStructureError(
                "DetectionModel.model no es un nn.Sequential",
                installed_ultralytics_version(),
            )
        return capas

    def validate_structure(self) -> None:
        """Comprueba que la estructura interna es la esperada."""
        version = installed_ultralytics_version()
        if version != ANCHORED_ULTRALYTICS_VERSION:
            logger.warning(
                "La versión de Ultralytics instalada (%s) no es la anclada (%s). "
                "La estructura interna puede haber cambiado.",
                version,
                ANCHORED_ULTRALYTICS_VERSION,
            )

        capas = self.layers
        if len(capas) != EXPECTED_NUM_LAYERS:
            raise UnexpectedModelStructureError(
                f"se esperaban {EXPECTED_NUM_LAYERS} capas y hay {len(capas)}", version
            )
        if not self.get_head_modules():
            raise UnexpectedModelStructureError(
                f"no se encuentra la cabeza Detect en el índice {NECK_END_INDEX}", version
            )
        logger.debug("Estructura del detector verificada (%d capas)", len(capas))

    def get_backbone_modules(self) -> dict[str, nn.Module]:
        """Capas del backbone (índices 0 a 9) por nombre."""
        return {f"model.{i}": capa for i, capa in enumerate(self.layers) if i < BACKBONE_END_INDEX}

    def get_neck_modules(self) -> dict[str, nn.Module]:
        """Capas del cuello (índices 10 a 21) por nombre."""
        return {
            f"model.{i}": capa
            for i, capa in enumerate(self.layers)
            if BACKBONE_END_INDEX <= i < NECK_END_INDEX
        }

    def get_head_modules(self) -> dict[str, nn.Module]:
        """Capas de la cabeza de detección (índice 22) por nombre. Son las que la poda protege por
        defecto.
        """
        return {f"model.{i}": capa for i, capa in enumerate(self.layers) if i >= NECK_END_INDEX}

    def get_named_conv_layers(self) -> dict[str, nn.Conv2d]:
        """Convoluciones del modelo por nombre."""
        return {
            nombre: modulo
            for nombre, modulo in self._model.named_modules()
            if isinstance(modulo, nn.Conv2d)
        }

    def get_head_conv_names(self) -> set[str]:
        """Nombres de las convoluciones de la cabeza, para protegerlas en la poda."""
        prefijo = f"model.{NECK_END_INDEX}."
        return {n for n in self.get_named_conv_layers() if n.startswith(prefijo)}

    def get_module(self, name: str) -> nn.Module:
        """Módulo interno por su nombre completo."""
        modulos = dict(self._model.named_modules())
        if name not in modulos:
            raise DetectorError(
                f"No existe el módulo '{name}' en el detector.",
                "Consulta los nombres disponibles con get_named_conv_layers().",
            )
        return modulos[name]

    def forward(self, x: torch.Tensor) -> Any:
        """Pase hacia delante con una entrada ``(N, 3, H, W)``."""
        return self._model(x)

    def __call__(self, x: torch.Tensor) -> Any:
        """Alias de :meth:`forward`."""
        return self.forward(x)

    @contextmanager
    def capture_features(self, layer_names: Sequence[str]) -> Iterator[dict[str, torch.Tensor]]:
        """Captura las activaciones de las capas indicadas y retira los hooks al salir, también si
        hay excepción.

        >>> with detector.capture_features(["model.4", "model.6"]) as feats:
        ...     detector.forward(x)
        ...     mapa = feats["model.4"]
        """
        capturadas: dict[str, torch.Tensor] = {}
        handles = []

        def _crear_hook(nombre: str):
            def _hook(_modulo, _entrada, salida):
                capturadas[nombre] = salida

            return _hook

        try:
            for nombre in layer_names:
                handles.append(self.get_module(nombre).register_forward_hook(_crear_hook(nombre)))
            yield capturadas
        finally:
            for handle in handles:
                handle.remove()
            capturadas.clear()

    def to(self, device: str | torch.device) -> "Detector":
        """Mueve el modelo a un dispositivo y devuelve el detector."""
        self._model = self._model.to(device)
        return self

    def eval(self) -> "Detector":
        """Pone el modelo en modo evaluación."""
        self._model.eval()
        return self

    def train(self, mode: bool = True) -> "Detector":
        """Pone el modelo en modo entrenamiento."""
        self._model.train(mode)
        return self

    @property
    def device(self) -> torch.device:
        """Dispositivo en el que residen los parámetros del modelo."""
        return next(self._model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Tipo de dato de los parámetros del modelo."""
        return next(self._model.parameters()).dtype

    def assert_float32(self) -> None:
        """Comprueba que todos los pesos son FP32."""
        malos = [
            nombre
            for nombre, parametro in self._model.named_parameters()
            if parametro.dtype != torch.float32
        ]
        if malos:
            raise DetectorError(
                f"{len(malos)} parámetros del modelo NO son float32. " f"Primeros: {malos[:5]}",
                "El checkpoint parece venir del guardado de Ultralytics, que usa media "
                "precisión; no sirve como referencia FP32.",
            )

    def count_parameters(self, only_trainable: bool = False) -> int:
        """Cuenta los parámetros del modelo."""
        return sum(
            p.numel() for p in self._model.parameters() if not only_trainable or p.requires_grad
        )

    def channels_per_conv(self) -> dict[str, tuple[int, int]]:
        """Canales de entrada y salida de cada convolución, para comprobar una poda."""
        return {
            nombre: (modulo.in_channels, modulo.out_channels)
            for nombre, modulo in self.get_named_conv_layers().items()
        }

    def summary(self) -> str:
        """Genera un resumen legible del detector."""
        return (
            f"yolov8{self.size} | {self.num_classes} clases | "
            f"{self.count_parameters():,} parámetros | "
            f"{self.input_resolution}px | {self.dtype} | {self.device}"
        )


def normalize_categories(
    detecciones: list[dict[str, Any]], validador: Any, num_classes: int
) -> list[dict[str, Any]]:
    """Traduce las categorías de Ultralytics a índices de clase del proyecto, desde 0.

    Ultralytics numera desde 1 en datasets que no son COCO (``class_map = list(range(1, n + 1))``).
    Se usa el class_map real del validador y se comprueba que el resultado cae en rango, porque un
    desfase de uno desplazaría todas las clases sin error visible.
    """
    if not detecciones:
        return detecciones

    mapeo = getattr(validador, "class_map", None)
    if not mapeo:
        raise DetectorError(
            "El validador no expone 'class_map'.",
            "Sin el class_map del validador no se puede traducir la numeración de categorías. "
            "Revisa la versión de Ultralytics.",
        )
    inverso = {int(valor): indice for indice, valor in enumerate(mapeo)}

    traducidas: list[dict[str, Any]] = []
    desconocidas: set[int] = set()
    for deteccion in detecciones:
        categoria = int(deteccion["category_id"])
        if categoria not in inverso:
            desconocidas.add(categoria)
            continue
        copia = dict(deteccion)
        copia["category_id"] = inverso[categoria]
        traducidas.append(copia)

    if desconocidas:
        raise DetectorError(
            f"Categorías que el class_map del validador no cubre: " f"{sorted(desconocidas)}.",
            "Indica que el modelo predice más clases de las que declara el "
            "dataset. Revisa el número de clases del checkpoint.",
        )
    fuera = {d["category_id"] for d in traducidas if not 0 <= d["category_id"] < num_classes}
    if fuera:
        raise DetectorError(
            f"Tras traducir quedan índices fuera de [0, {num_classes}): " f"{sorted(fuera)}.",
            "El mapeo de clases está desalineado.",
        )
    logger.debug(
        "Categorías normalizadas: %s a índices de clase 0..%d",
        f"{mapeo[0]}..{mapeo[-1]}",
        num_classes - 1,
    )
    return traducidas


def extract_metrics(
    validador: Any,
    split: str,
    conf_threshold: float,
    iou_threshold: float,
    input_resolution: int,
) -> dict[str, Any]:
    """Métricas de un validador de Ultralytics ya ejecutado."""
    metricas = getattr(validador, "metrics", None)
    caja = getattr(metricas, "box", None)
    if caja is None:
        raise DetectorError(
            "El validador no ha producido métricas de caja.",
            "Es probable que Ultralytics haya cambiado la estructura de "
            "DetMetrics. Revisa la versión anclada antes de usar estas cifras.",
        )

    por_clase: dict[int, dict[str, float]] = {}
    # ap_class_index es un array de numpy; ``or []`` lanzaría ValueError.
    crudos = getattr(caja, "ap_class_index", None)
    indices = [] if crudos is None else list(crudos)
    for posicion, indice in enumerate(indices):
        precision, recall, ap50, ap = caja.class_result(posicion)
        por_clase[int(indice)] = {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "ap_50": round(float(ap50), 6),
            "ap_50_95": round(float(ap), 6),
        }

    return {
        "split": split,
        "input_resolution": input_resolution,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "map_50": round(float(caja.map50), 6),
        "map_50_95": round(float(caja.map), 6),
        "mean_precision": round(float(caja.mp), 6),
        "mean_recall": round(float(caja.mr), 6),
        "num_classes_evaluated": len(por_clase),
        "per_class": por_clase,
    }


class ArtifactDetector:
    """Evalúa un artefacto exportado por el mismo camino que un checkpoint.

    No hereda de Detector porque un ONNX cuantizado no tiene capas que intervenir; comparte con él
    las funciones de evaluación y build_validator_args, para usar exactamente los mismos argumentos
    del validador. El artefacto debe venir preparado por export/staging.py.
    """

    def __init__(
        self,
        artifact: Path | str,
        runtime: str,
        num_classes: int,
        input_resolution: int = 640,
    ) -> None:
        """Envuelve un artefacto ya preparado por export.staging.stage_artifact."""
        ruta = Path(artifact)
        if not ruta.exists():
            raise DetectorError(
                f"No existe el artefacto '{ruta}'.",
                "Prepáralo antes con tfm_edge.export.staging.stage_artifact: un "
                "artefacto sin los metadatos de Ultralytics no se puede evaluar.",
            )
        self.artifact = ruta
        self.runtime = runtime
        self.num_classes = num_classes
        self.input_resolution = input_resolution

    def evaluate(
        self,
        data_config: Path | str,
        split: str = "val",
        batch_size: int = 1,
        device: str = "cpu",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        save_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Ejecuta el validador sobre el artefacto y devuelve las detecciones COCO y las métricas de
        Ultralytics de una sola pasada.

        Así la comparación entre evaluadores usa las mismas detecciones.
        """
        _importar_ultralytics()
        from ultralytics.models.yolo.detect import DetectionValidator

        argumentos = build_validator_args(
            data_config=data_config,
            split=split,
            batch_size=batch_size,
            input_resolution=self.input_resolution,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            save_dir=save_dir,
            max_detections=max_detections,
            save_json=True,
        )
        # AutoBackend elige el motor por el nombre del fichero o directorio.
        argumentos["model"] = str(self.artifact)

        try:
            validador = DetectionValidator(args=argumentos)
            validador(model=str(self.artifact))
        except Exception as error:
            raise DetectorError(
                f"La evaluación del artefacto '{self.artifact.name}' ha fallado: "
                f"{type(error).__name__}: {error}",
                "Comprueba que el artefacto está preparado con "
                "export/staging.py y que el YAML de dataset es válido.",
            ) from error

        detecciones = normalize_categories(
            list(getattr(validador, "jdict", []) or []), validador, self.num_classes
        )
        if not detecciones:
            raise DetectorError(
                f"El artefacto '{self.artifact.name}' no ha producido ninguna "
                "detección sobre la partición completa.",
                "Con conf=0.001 un modelo funcional produce miles. Revisa qué nodos han quedado "
                "cuantizados.",
            )

        return {
            "detections": detecciones,
            "num_detections": len(detecciones),
            "ultralytics_metrics": extract_metrics(
                validador, split, conf_threshold, iou_threshold, self.input_resolution
            ),
            "artifact": str(self.artifact),
            "runtime": self.runtime,
            "split": split,
            "input_resolution": self.input_resolution,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "batch_size": batch_size,
        }


def build_validator_args(
    data_config: Path | str,
    split: str,
    batch_size: int,
    input_resolution: int,
    conf_threshold: float,
    iou_threshold: float,
    device: str = "",
    save_dir: Path | str | None = None,
    max_detections: int | None = None,
    save_json: bool = False,
) -> dict[str, Any]:
    """Argumentos del validador de Ultralytics, comunes a checkpoints y artefactos exportados."""
    argumentos: dict[str, Any] = {
        "data": str(data_config),
        "split": split,
        "batch": batch_size,
        "imgsz": input_resolution,
        "conf": conf_threshold,
        "iou": iou_threshold,
        # Se evalúa en FP32 salvo que se indique lo contrario.
        "half": False,
        "plots": False,
        "verbose": False,
        # rect=False: si no, Ultralytics agrupa por relación de aspecto y el relleno cambia entre
        # imágenes.
        "rect": False,
    }
    if max_detections is not None:
        argumentos["max_det"] = max_detections
    if save_json:
        argumentos["save_json"] = True
    if device:
        argumentos["device"] = device
    if save_dir is not None:
        argumentos["project"] = str(Path(save_dir).parent)
        argumentos["name"] = Path(save_dir).name
        argumentos["exist_ok"] = True
    return argumentos
