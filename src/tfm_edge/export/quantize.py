"""Materialización INT8 con el mecanismo nativo de cada runtime.

PTQ y QAT tienen que seguir el mismo camino, con la misma calibración y los mismos parámetros; solo
cambian los pesos de partida. Para comprobarlo hay una única función, materialize_int8, los
parámetros están en Int8Config con un hash estable, y assert_same_materialization_path compara los
hashes de configuración y de calibración de dos artefactos. La calibración usa el mismo preprocesado
que la inferencia (InferenceBackend).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir
from tfm_edge.common.registry import utc_timestamp

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import numpy as np

logger = get_logger(__name__)

INT8_RUNTIMES = ("onnxruntime", "openvino")

#: Procedencias admitidas de los pesos: baseline (PTQ), qat y pruned (poda combinada con PTQ). Todas
#: siguen el mismo camino; el valor solo identifica de dónde salen los pesos.
VALID_SOURCE_KINDS = ("baseline", "qat", "pruned")

#: Pareja que acepta por defecto assert_same_materialization_path.
D018_SOURCE_KINDS = frozenset({"baseline", "qat"})

#: Pareja PTQ y QAT sobre el mismo modelo podado. Solo se usa si ptq.resolve_d018_pair ha comprobado
#: en el registro que comparten padre.
PRUNED_QAT_SOURCE_KINDS = frozenset({"pruned", "qat"})

#: Parejas que se pueden autorizar; la lista es cerrada.
ADMISSIBLE_SOURCE_PAIRS = (D018_SOURCE_KINDS, PRUNED_QAT_SOURCE_KINDS)

#: Tamaño de calibración por defecto; 300 es lo que recomienda NNCF.
DEFAULT_CALIBRATION_SIZE = 300


class QuantizationError(TFMEdgeError):
    """Error al materializar un modelo en INT8."""


class MaterializationMismatchError(TFMEdgeError):
    """Dos artefactos INT8 no se han materializado por el mismo camino; la comparación PTQ frente a
    QAT no es válida.
    """

    def __init__(self, diferencias: Sequence[str]) -> None:
        detalle = "\n".join(f"       - {d}" for d in diferencias)
        super().__init__(
            f"Los dos artefactos INT8 NO se han materializado por el mismo " f"camino:\n{detalle}",
            "PTQ y QAT tienen que compartir conjunto de calibración y "
            "parámetros, y que la única diferencia sean los pesos de partida.\n"
            "     Con caminos distintos, una diferencia de mAP entre las dos "
            "ramas sería atribuible tanto a la técnica como a la exportación, y "
            "la comparación del capítulo 5.2 no diría nada.\n"
            "     Vuelve a materializar las dos con la MISMA Int8Config y el "
            "MISMO conjunto de calibración.",
        )


@dataclass(frozen=True)
class Int8Config:
    """Parámetros de la materialización INT8, idénticos en PTQ y QAT. Es frozen para que el hash no
    cambie.

    La salida de YOLOv8 concatena coordenadas (rango de unos 640) con puntuaciones en [0, 1]; con
    una sola escala por tensor las puntuaciones quedan a cero y el modelo no detecta nada. Por eso
    la salida de los Concat no se cuantiza en ONNX Runtime (``onnx_exclude_output_quantization``);
    son operadores de movimiento de datos y no cuesta nada apreciable.
    """

    calibration_size: int = DEFAULT_CALIBRATION_SIZE
    per_channel: bool = True
    symmetric_weights: bool = True
    activation_type: str = "uint8"
    weight_type: str = "int8"
    calibration_method: str = "MinMax"
    excluded_nodes: tuple[str, ...] = ()
    onnx_exclude_output_quantization: tuple[str, ...] = ("Concat",)
    preset: str = "PERFORMANCE"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la configuración como diccionario serializable."""
        datos = asdict(self)
        datos["excluded_nodes"] = list(self.excluded_nodes)
        datos["onnx_exclude_output_quantization"] = list(self.onnx_exclude_output_quantization)
        return datos

    @property
    def config_hash(self) -> str:
        """Hash estable de los parámetros, sin rutas ni fechas."""
        canonico = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]


@dataclass
class QuantizationResult:
    """Resultado de materializar un modelo en INT8.

    ``source_kind`` es la única diferencia admisible entre las ramas. ``functional_check`` indica si
    el artefacto produce detecciones.
    """

    runtime: str
    status: str = "no_disponible"
    path: str = ""
    source_path: str = ""
    source_kind: str = "baseline"
    disk_bytes: int | str = ""
    size_reduction: float | str = ""
    quantize_seconds: float | str = ""
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    calibration_hash: str = ""
    num_calibration_images: int = 0
    functional_check: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = ""

    @property
    def ok(self) -> bool:
        """Si el artefacto INT8 se generó correctamente."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        return asdict(self)


def calibration_hash(image_paths: Sequence[Path | str]) -> str:
    """Hash del conjunto de calibración, que depende de las imágenes y de su orden."""
    if not image_paths:
        raise QuantizationError(
            "El conjunto de calibración está vacío.",
            "La cuantización estática necesita imágenes reales para estimar los "
            "rangos de activación. Genéralo con data/calibration_set.py.",
        )
    nombres = [Path(p).name for p in image_paths]
    canonico = "\n".join(nombres)
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()[:16]


def assert_artifact_is_functional(
    artifact_path: Path | str,
    runtime: str,
    images: Sequence[Path | str],
    input_resolution: int = 640,
) -> dict[str, Any]:
    """Comprueba que un artefacto cuantizado produce puntuaciones que no son todas iguales.

    Un INT8 de ONNX Runtime con todas las puntuaciones a cero se materializaba sin errores. Se mira
    que el tensor no sea constante y no un umbral absoluto, porque un modelo sin entrenar da
    puntuaciones legítimamente muy bajas. No mide calidad. Si no se puede comprobar, ``passed``
    queda vacío y no se lanza error.
    """
    import numpy as np

    if not images:
        return {"passed": "", "reason": "no se han dado imágenes con las que comprobar"}

    try:
        import cv2

        from tfm_edge.benchmark.backends import build_backend

        motor = build_backend(runtime, input_resolution, "cpu")
        motor.load(Path(artifact_path))
        maximos: list[float] = []
        distintos = 0
        for ruta in images:
            imagen = cv2.imread(str(ruta))
            if imagen is None:
                continue
            salida = motor.infer(motor.preprocess(imagen)).numpy()
            # (1, 4 + num_clases, N): las cuatro primeras filas son coordenadas.
            puntuaciones = salida[0][4:]
            maximos.append(float(puntuaciones.max()))
            distintos = max(distintos, int(np.unique(puntuaciones).size))
        motor.unload()
    except Exception as error:  # pragma: no cover - depende del entorno
        return {
            "passed": "",
            "reason": f"no se ha podido comprobar: {type(error).__name__}: {str(error)[:120]}",
        }

    if not maximos:
        return {"passed": "", "reason": "ninguna de las imágenes se pudo leer"}

    mejor = max(maximos)
    informe = {
        "passed": distintos > 1,
        "max_score": round(mejor, 6),
        "unique_scores": distintos,
        "num_images": len(maximos),
        "reason": "",
    }
    if not informe["passed"]:
        informe["reason"] = (
            f"la salida está degenerada: los canales de puntuación tienen "
            f"{distintos} valor(es) distinto(s) en las {len(maximos)} imágenes "
            f"probadas, con máximo {mejor:.6f}. El artefacto NO detecta nada."
        )
    return informe


def _preprocesar_calibracion(
    image_paths: Sequence[Path | str], input_resolution: int
) -> Iterator["np.ndarray"]:
    """Preprocesa las imágenes de calibración igual que en inferencia, con el preprocesado de
    InferenceBackend.
    """
    import cv2

    from tfm_edge.benchmark.backends import PyTorchBackend

    # Solo se usa su preprocesado; no se carga ningún modelo.
    preprocesador = PyTorchBackend(input_resolution=input_resolution, device="cpu")

    for ruta in image_paths:
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            raise QuantizationError(
                f"No se ha podido leer la imagen de calibración '{ruta}'.",
                "Comprueba que el conjunto de calibración apunta a imágenes que " "existen.",
            )
        yield preprocesador.preprocess(imagen).cpu().numpy()


def materialize_int8_onnx(
    source_path: Path | str,
    output_path: Path | str,
    calibration_images: Sequence[Path | str],
    config: Int8Config,
    input_resolution: int = 640,
) -> QuantizationResult:
    """Materializa INT8 con quantize_static de ONNX Runtime."""
    resultado = QuantizationResult(runtime="onnxruntime", timestamp=utc_timestamp())
    resultado.config = config.to_dict()
    resultado.config_hash = config.config_hash
    resultado.source_path = str(source_path)

    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as error:
        resultado.notes = f"onnxruntime.quantization no disponible: {error}"
        return resultado

    # Mapeo explícito al enum: el miembro es QUInt8, con I mayúscula.
    tipos = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}
    for etiqueta, valor in (
        ("activation_type", config.activation_type),
        ("weight_type", config.weight_type),
    ):
        if valor not in tipos:
            resultado.status = "fallido"
            resultado.notes = f"{etiqueta}='{valor}' no válido; admitidos: {sorted(tipos)}"
            return resultado

    origen = Path(source_path)
    if not origen.is_file():
        resultado.status = "fallido"
        resultado.notes = f"no existe el ONNX de partida '{origen}'"
        return resultado

    imagenes = list(calibration_images)[: config.calibration_size]
    resultado.calibration_hash = calibration_hash(imagenes)
    resultado.num_calibration_images = len(imagenes)

    class _Lector(CalibrationDataReader):
        """Alimenta al calibrador con las imágenes ya preprocesadas."""

        def __init__(self) -> None:
            """Prepara el iterador."""
            self._nombre = ""
            self._datos = _preprocesar_calibracion(imagenes, input_resolution)

        def get_next(self) -> dict[str, Any] | None:
            """Siguiente lote ``{nombre_entrada: tensor}`` o None al terminar."""
            try:
                return {self._nombre: next(self._datos)}
            except StopIteration:
                return None

    destino = ensure_dir(Path(output_path).parent) / Path(output_path).name
    preprocesado = destino.with_suffix(".preprocessed.onnx")

    inicio = time.perf_counter()
    try:
        import onnxruntime as ort

        # onnxruntime recomienda inferir formas antes; sin ello quantize_static deja nodos sin
        # cuantizar.
        quant_pre_process(str(origen), str(preprocesado), skip_symbolic_shape=False)

        lector = _Lector()
        lector._nombre = (
            ort.InferenceSession(str(preprocesado), providers=["CPUExecutionProvider"])
            .get_inputs()[0]
            .name
        )

        opciones: dict[str, Any] = {"WeightSymmetric": config.symmetric_weights}
        if config.onnx_exclude_output_quantization:
            # Sin esto el Concat final deja las puntuaciones a cero (ver Int8Config).
            opciones["OpTypesToExcludeOutputQuantization"] = list(
                config.onnx_exclude_output_quantization
            )

        quantize_static(
            model_input=str(preprocesado),
            model_output=str(destino),
            calibration_data_reader=lector,
            quant_format=QuantFormat.QDQ,
            per_channel=config.per_channel,
            activation_type=tipos[config.activation_type],
            weight_type=tipos[config.weight_type],
            calibrate_method=getattr(CalibrationMethod, config.calibration_method),
            nodes_to_exclude=list(config.excluded_nodes) or None,
            extra_options=opciones,
        )
    except Exception as error:
        resultado.status = "fallido"
        resultado.notes = f"{type(error).__name__}: {str(error)[:200]}"
        logger.error("La materialización INT8 en ONNX Runtime ha fallado: %s", error)
        return resultado
    finally:
        preprocesado.unlink(missing_ok=True)

    resultado.quantize_seconds = round(time.perf_counter() - inicio, 2)
    resultado.path = str(destino)
    resultado.disk_bytes = destino.stat().st_size
    tam_origen = origen.stat().st_size
    resultado.size_reduction = round(tam_origen / resultado.disk_bytes, 3)
    resultado.status = "ok"
    logger.info(
        "INT8 (ONNX Runtime) materializado en %s: %.2f MiB, %.2fx menor, "
        "calibrado con %d imágenes",
        destino,
        resultado.disk_bytes / 2**20,
        resultado.size_reduction,
        len(imagenes),
    )
    return resultado


def materialize_int8_openvino(
    source_path: Path | str,
    output_path: Path | str,
    calibration_images: Sequence[Path | str],
    config: Int8Config,
    input_resolution: int = 640,
) -> QuantizationResult:
    """Materializa INT8 con NNCF para OpenVINO."""
    resultado = QuantizationResult(runtime="openvino", timestamp=utc_timestamp())
    resultado.config = config.to_dict()
    resultado.config_hash = config.config_hash
    resultado.source_path = str(source_path)

    try:
        import nncf
        import openvino as ov
    except ImportError as error:
        resultado.notes = f"nncf u openvino no disponibles: {error}"
        return resultado

    origen = Path(source_path)
    if not origen.is_file():
        resultado.status = "fallido"
        resultado.notes = f"no existe el modelo de partida '{origen}'"
        return resultado

    imagenes = list(calibration_images)[: config.calibration_size]
    resultado.calibration_hash = calibration_hash(imagenes)
    resultado.num_calibration_images = len(imagenes)

    destino = ensure_dir(Path(output_path).parent) / Path(output_path).name
    inicio = time.perf_counter()
    try:
        modelo = ov.Core().read_model(str(origen))
        datos = nncf.Dataset(list(_preprocesar_calibracion(imagenes, input_resolution)))
        cuantizado = nncf.quantize(
            modelo,
            datos,
            preset=getattr(nncf.QuantizationPreset, config.preset),
            subset_size=len(imagenes),
            ignored_scope=(
                nncf.IgnoredScope(names=list(config.excluded_nodes))
                if config.excluded_nodes
                else None
            ),
        )
        # compress_to_fp16 no aplica a un modelo cuantizado, pero se pasa explícito por si cambia el
        # defecto.
        ov.save_model(cuantizado, str(destino), compress_to_fp16=False)
    except Exception as error:
        resultado.status = "fallido"
        resultado.notes = f"{type(error).__name__}: {str(error)[:200]}"
        logger.error("La materialización INT8 en OpenVINO ha fallado: %s", error)
        return resultado

    resultado.quantize_seconds = round(time.perf_counter() - inicio, 2)
    resultado.path = str(destino)
    resultado.disk_bytes = sum(
        f.stat().st_size for f in destino.parent.rglob(f"{destino.stem}*") if f.is_file()
    )
    tam_origen = sum(
        f.stat().st_size for f in origen.parent.rglob(f"{origen.stem}*") if f.is_file()
    )
    if tam_origen and resultado.disk_bytes:
        resultado.size_reduction = round(tam_origen / resultado.disk_bytes, 3)
    resultado.status = "ok"
    logger.info(
        "INT8 (OpenVINO) materializado en %s: %.2f MiB, calibrado con %d imágenes",
        destino,
        resultado.disk_bytes / 2**20,
        len(imagenes),
    )
    return resultado


def materialize_int8(
    source_path: Path | str,
    output_path: Path | str,
    calibration_images: Sequence[Path | str],
    runtime: str,
    config: Int8Config | None = None,
    input_resolution: int = 640,
    source_kind: str = "baseline",
) -> QuantizationResult:
    """Materializa INT8 en un runtime. Es el punto de entrada común de PTQ y QAT.

    Entre ramas solo cambian ``source_path`` y ``source_kind`` (baseline, qat o pruned), que queda
    registrado en el artefacto.
    """
    if runtime not in INT8_RUNTIMES:
        raise QuantizationError(
            f"Runtime '{runtime}' sin materialización INT8 implementada.",
            f"Los implementados son: {list(INT8_RUNTIMES)}.",
        )
    if source_kind not in VALID_SOURCE_KINDS:
        raise QuantizationError(
            f"source_kind '{source_kind}' no válido.",
            f"Los admitidos son: {list(VALID_SOURCE_KINDS)}.",
        )

    configuracion = config or Int8Config()
    funcion = {
        "onnxruntime": materialize_int8_onnx,
        "openvino": materialize_int8_openvino,
    }[runtime]
    resultado = funcion(
        source_path, output_path, calibration_images, configuracion, input_resolution
    )
    resultado.source_kind = source_kind

    # La comprobación funcional va aquí para que la hagan todas las ramas por igual.
    if resultado.ok:
        resultado.functional_check = assert_artifact_is_functional(
            resultado.path,
            runtime,
            list(calibration_images)[:4],
            input_resolution=input_resolution,
        )
        if resultado.functional_check.get("passed") is False:
            resultado.status = "fallido"
            resultado.notes = (
                f"{resultado.functional_check['reason']} "
                "El artefacto se ha materializado pero NO sirve: no se registra como ok."
            )
            logger.error(
                "El artefacto INT8 de %s no produce detecciones: %s",
                runtime,
                resultado.functional_check["reason"],
            )
        elif resultado.functional_check.get("passed") == "":
            logger.warning(
                "No se ha podido comprobar que el INT8 de %s detecte: %s",
                runtime,
                resultado.functional_check.get("reason", ""),
            )
    return resultado


def assert_same_materialization_path(
    ptq: QuantizationResult,
    qat: QuantizationResult,
    admitted_pair: frozenset[str] = D018_SOURCE_KINDS,
) -> dict[str, Any]:
    """Comprueba que dos artefactos INT8 comparten runtime, configuración y calibración y difieren
    en los pesos de partida.

    ``admitted_pair`` es por defecto (baseline, qat); PRUNED_QAT_SOURCE_KINDS solo tras comprobar
    que las dos ramas parten del mismo modelo podado.
    """
    if frozenset(admitted_pair) not in ADMISSIBLE_SOURCE_PAIRS:
        raise QuantizationError(
            f"Pareja de procedencias no reconocida: {sorted(admitted_pair)}.",
            f"Solo se pueden autorizar {[sorted(p) for p in ADMISSIBLE_SOURCE_PAIRS]}.",
        )
    admitted_pair = frozenset(admitted_pair)
    diferencias: list[str] = []
    if ptq.runtime != qat.runtime:
        diferencias.append(
            f"runtimes distintos: '{ptq.runtime}' y '{qat.runtime}'. Solo tiene "
            "sentido comparar dentro del mismo runtime"
        )
    if ptq.config_hash != qat.config_hash:
        diferencias.append(
            f"parámetros distintos: {ptq.config_hash} y {qat.config_hash}.\n"
            f"         PTQ: {ptq.config}\n"
            f"         QAT: {qat.config}"
        )
    if ptq.calibration_hash != qat.calibration_hash:
        diferencias.append(
            f"conjuntos de calibración distintos: {ptq.calibration_hash} "
            f"({ptq.num_calibration_images} imágenes) y {qat.calibration_hash} "
            f"({qat.num_calibration_images} imágenes)"
        )
    if diferencias:
        raise MaterializationMismatchError(diferencias)

    # Se exige la pareja exacta autorizada, no solo que las procedencias difieran: con tres valores,
    # (pruned, qat) también difiere y no es la comparación buscada.
    if {ptq.source_kind, qat.source_kind} != admitted_pair:
        raise MaterializationMismatchError(
            [
                f"los artefactos declaran source_kind='{ptq.source_kind}' y "
                f"'{qat.source_kind}'; la comparación admitida es exactamente "
                f"{sorted(admitted_pair)}, y esta pareja no lo es"
            ]
        )

    logger.info(
        "PTQ y QAT comparten camino (config %s, calibración %s) y solo difieren en los pesos de "
        "partida",
        ptq.config_hash,
        ptq.calibration_hash,
    )
    return {
        "same_path": True,
        "runtime": ptq.runtime,
        "config_hash": ptq.config_hash,
        "calibration_hash": ptq.calibration_hash,
        "num_calibration_images": ptq.num_calibration_images,
        "ptq_source": ptq.source_path,
        "qat_source": qat.source_path,
    }
