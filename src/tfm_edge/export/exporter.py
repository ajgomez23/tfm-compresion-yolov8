"""Exportación a los formatos de despliegue.

Un runtime ausente se registra como status=no_disponible con el motivo y la campaña sigue. ONNX y
OpenVINO están verificados; TensorRT no está implementado y TFLite sobre x86 no representa a un
dispositivo ARM. Cada exportación se compara numéricamente con el modelo de PyTorch.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir
from tfm_edge.common.registry import utc_timestamp
from tfm_edge.export import equivalence

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)

#: Formatos contemplados y el paquete que necesita cada uno.
EXPORT_FORMATS: dict[str, tuple[str, ...]] = {
    "onnx": ("onnx",),
    "openvino": ("openvino",),
    "tensorrt": ("tensorrt",),
    "tflite": ("tensorflow",),
}

#: Formatos verificados en el entorno de desarrollo.
VERIFIED_FORMATS = ("onnx", "openvino")


class ExportError(TFMEdgeError):
    """Error al exportar un modelo."""


@dataclass
class ExportResult:
    """Resultado de exportar a un formato."""

    format: str
    status: str = "no_disponible"
    path: str = ""
    disk_bytes: int | str = ""
    export_seconds: float | str = ""
    equivalence: dict[str, Any] = field(default_factory=dict)
    latency_representative: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    timestamp: str = ""

    @property
    def ok(self) -> bool:
        """True si el artefacto se generó y es equivalente."""
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        return asdict(self)


def check_format_available(formato: str) -> tuple[bool, str]:
    """Indica si un formato se puede exportar aquí. Devuelve ``(disponible, motivo)``."""
    import importlib.util

    if formato not in EXPORT_FORMATS:
        raise ExportError(
            f"Formato '{formato}' no contemplado.",
            f"Los contemplados son: {sorted(EXPORT_FORMATS)}.",
        )
    faltan = [
        paquete for paquete in EXPORT_FORMATS[formato] if importlib.util.find_spec(paquete) is None
    ]
    if faltan:
        return False, f"falta {', '.join(faltan)} en este entorno"

    if formato == "tensorrt":
        try:
            import torch

            if not torch.cuda.is_available():
                return False, "TensorRT necesita una GPU NVIDIA y aquí no hay CUDA"
        except ImportError:  # pragma: no cover
            return False, "PyTorch no está instalado"
    return True, ""


def _muestras_de_entrada(detector: "Detector", num_samples: int, seed: int = 42) -> list["Any"]:
    """Entradas deterministas para la comprobación de equivalencia; varias para cubrir más ramas del
    grafo.
    """
    import numpy as np

    generador = np.random.default_rng(seed)
    resolucion = detector.input_resolution
    return [
        generador.random((1, 3, resolucion, resolucion), dtype=np.float32)
        for _ in range(num_samples)
    ]


def _salidas_de_referencia(detector: "Detector", entradas: Sequence[Any]) -> list[Any]:
    """Calcula las salidas del modelo de PyTorch en modo exportación."""
    import torch

    salidas = []
    with detector.export_mode() as modelo:
        modelo = modelo.cpu().float()
        with torch.no_grad():
            for entrada in entradas:
                salidas.append(modelo(torch.from_numpy(entrada)).numpy())
    return salidas


def export_onnx(
    detector: "Detector",
    output_dir: Path | str,
    name: str,
    num_check_samples: int = 3,
    opset: int | None = None,
    simplify: bool = True,
) -> ExportResult:
    """Exporta a ONNX y verifica la equivalencia numérica. Si no es equivalente, devuelve
    status=fallido.
    """
    resultado = ExportResult(format="onnx", timestamp=utc_timestamp())
    disponible, motivo = check_format_available("onnx")
    if not disponible:
        resultado.notes = motivo
        logger.warning("ONNX no disponible: %s", motivo)
        return resultado

    destino = ensure_dir(output_dir) / f"{name}.onnx"
    inicio = time.perf_counter()
    try:
        info = detector.export_onnx(destino, opset=opset, simplify=simplify)
    except TFMEdgeError as error:
        resultado.status = "fallido"
        resultado.notes = str(error).splitlines()[0]
        logger.error("La exportación a ONNX ha fallado: %s", error)
        return resultado
    resultado.export_seconds = round(time.perf_counter() - inicio, 2)
    resultado.path = info["path"]
    resultado.disk_bytes = Path(info["path"]).stat().st_size
    resultado.metadata = info

    # Comprobación de equivalencia numérica.
    try:
        import onnxruntime as ort

        entradas = _muestras_de_entrada(detector, num_check_samples)
        referencias = _salidas_de_referencia(detector, entradas)
        sesion = ort.InferenceSession(info["path"], providers=["CPUExecutionProvider"])
        nombre_entrada = sesion.get_inputs()[0].name
        candidatos = [sesion.run(None, {nombre_entrada: e})[0] for e in entradas]
        informe = equivalence.compare_many(
            referencias, candidatos, input_resolution=detector.input_resolution
        )
        resultado.equivalence = informe.to_dict()
        resultado.status = "ok" if informe.equivalent else "fallido"
        if not informe.equivalent:
            resultado.notes = "; ".join(informe.failures)
    except ImportError:
        resultado.status = "ok"
        resultado.notes = (
            "artefacto generado pero NO verificado: falta onnxruntime para "
            "comprobar la equivalencia numérica"
        )
        logger.warning("%s", resultado.notes)
    except Exception as error:  # pragma: no cover - depende del entorno
        resultado.status = "fallido"
        resultado.notes = f"fallo al verificar: {type(error).__name__}: {error}"
        logger.error("%s", resultado.notes)
    return resultado


def export_openvino(
    detector: "Detector",
    output_dir: Path | str,
    name: str,
    onnx_path: Path | str | None = None,
    num_check_samples: int = 3,
) -> ExportResult:
    """Exporta a OpenVINO a partir del ONNX y verifica la equivalencia.

    Partir del mismo ONNX permite atribuir las diferencias al motor y no a la conversión.
    """
    resultado = ExportResult(format="openvino", timestamp=utc_timestamp())
    disponible, motivo = check_format_available("openvino")
    if not disponible:
        resultado.notes = motivo
        logger.warning("OpenVINO no disponible: %s", motivo)
        return resultado

    if onnx_path is None or not Path(onnx_path).is_file():
        resultado.status = "fallido"
        resultado.notes = (
            "hace falta un ONNX de partida; exporta primero a ONNX y pásalo en " "'onnx_path'"
        )
        return resultado

    destino = ensure_dir(Path(output_dir) / f"{name}_openvino")
    inicio = time.perf_counter()
    try:
        import openvino as ov

        modelo = ov.convert_model(str(onnx_path))
        fichero = destino / f"{name}.xml"
        # compress_to_fp16=False explícito: por defecto OpenVINO guarda los pesos en FP16 aunque el
        # artefacto se llame fp32.
        ov.save_model(modelo, str(fichero), compress_to_fp16=False)
    except Exception as error:
        resultado.status = "fallido"
        resultado.notes = f"{type(error).__name__}: {str(error)[:200]}"
        logger.error("La exportación a OpenVINO ha fallado: %s", error)
        return resultado

    resultado.export_seconds = round(time.perf_counter() - inicio, 2)
    resultado.path = str(fichero)
    resultado.disk_bytes = sum(f.stat().st_size for f in destino.rglob("*") if f.is_file())
    resultado.metadata = {
        "openvino_version": ov.__version__,
        "from_onnx": str(onnx_path),
        "compress_to_fp16": False,
    }

    # Se comprueba aparte porque la equivalencia numérica no distingue FP32 de FP16.
    precision = verify_openvino_precision(fichero)
    resultado.metadata["weights_dtype"] = precision["dominant_dtype"]
    resultado.metadata["dtype_counts"] = precision["counts"]
    if not precision["is_fp32"]:
        resultado.status = "fallido"
        resultado.notes = (
            f"el artefacto NO está en FP32: {precision['counts']}. Una referencia "
            "que ha pasado por FP16 invalidaría la comparación con INT8"
        )
        logger.error("%s", resultado.notes)
        return resultado

    try:
        nucleo = ov.Core()
        compilado = nucleo.compile_model(modelo, "CPU")
        entradas = _muestras_de_entrada(detector, num_check_samples)
        referencias = _salidas_de_referencia(detector, entradas)
        candidatos = [compilado(e)[compilado.output(0)] for e in entradas]
        informe = equivalence.compare_many(
            referencias, candidatos, input_resolution=detector.input_resolution
        )
        resultado.equivalence = informe.to_dict()
        resultado.status = "ok" if informe.equivalent else "fallido"
        if not informe.equivalent:
            resultado.notes = "; ".join(informe.failures)
    except Exception as error:  # pragma: no cover - depende del entorno
        resultado.status = "fallido"
        resultado.notes = f"fallo al verificar: {type(error).__name__}: {error}"
        logger.error("%s", resultado.notes)
    return resultado


def verify_openvino_precision(model_path: Path | str) -> dict[str, Any]:
    """Comprueba en qué precisión están los pesos de un artefacto de OpenVINO.

    ov.save_model comprime a FP16 por defecto y la comparación numérica no lo detecta, así que se
    miran los tipos de las constantes.
    """
    try:
        import openvino as ov

        modelo = ov.Core().read_model(str(model_path))
    except Exception as error:
        raise ExportError(
            f"No se ha podido leer el artefacto de OpenVINO '{model_path}': "
            f"{type(error).__name__}: {error}",
            "Comprueba que existen el .xml y su .bin.",
        ) from error

    recuento: dict[str, int] = {}
    for nodo in modelo.get_ordered_ops():
        if nodo.get_type_name() != "Constant":
            continue
        tipo = str(nodo.get_element_type()).replace("<Type: '", "").replace("'>", "")
        recuento[tipo] = recuento.get(tipo, 0) + 1

    # Los enteros son índices y formas, no pesos.
    flotantes = {k: v for k, v in recuento.items() if k.startswith("float")}
    dominante = max(flotantes, key=flotantes.get) if flotantes else "desconocido"
    return {
        "counts": recuento,
        "float_counts": flotantes,
        "dominant_dtype": dominante,
        "is_fp32": bool(flotantes) and set(flotantes) == {"float32"},
    }


def export_unavailable(formato: str, motivo: str = "") -> ExportResult:
    """Resultado con status=no_disponible para un formato que no se puede exportar aquí."""
    if not motivo:
        _, motivo = check_format_available(formato)
    resultado = ExportResult(
        format=formato,
        status="no_disponible",
        notes=motivo or "no disponible en este entorno",
        timestamp=utc_timestamp(),
    )
    if formato == "tflite":
        resultado.latency_representative = False
        resultado.notes += (
            ". Alcance reducido: aunque se "
            "exporte, su latencia sobre x86 NO representa a un dispositivo ARM"
        )
    return resultado


def export_all(
    detector: "Detector",
    output_dir: Path | str,
    name: str,
    formats: Sequence[str] = ("onnx", "openvino", "tensorrt", "tflite"),
    num_check_samples: int = 3,
) -> dict[str, ExportResult]:
    """Exporta a todos los formatos pedidos sin abortar por los que falten.

    Devuelve una entrada por formato, aunque sea no_disponible.
    """
    resultados: dict[str, ExportResult] = {}
    directorio = ensure_dir(output_dir)

    if "onnx" in formats:
        resultados["onnx"] = export_onnx(
            detector, directorio, name, num_check_samples=num_check_samples
        )

    if "openvino" in formats:
        onnx_previo = resultados.get("onnx")
        resultados["openvino"] = export_openvino(
            detector,
            directorio,
            name,
            onnx_path=onnx_previo.path if onnx_previo and onnx_previo.ok else None,
            num_check_samples=num_check_samples,
        )

    for formato in formats:
        if formato in resultados:
            continue
        resultados[formato] = export_unavailable(formato)

    return resultados


def summarize_exports(resultados: dict[str, ExportResult]) -> str:
    """Tabla de exportaciones con su estado y motivo."""
    lineas = [
        f"{'formato':12s} {'estado':14s} {'tamaño':>10s} {'coord.':>10s} "
        f"{'punt.':>10s}  motivo",
        "-" * 96,
    ]
    for formato, r in resultados.items():
        tamano = f"{r.disk_bytes / 2**20:.2f} MiB" if isinstance(r.disk_bytes, int) else "—"
        eq = r.equivalence or {}
        coord = f"{eq['box_max_abs_diff']:.1e}" if eq.get("box_max_abs_diff") else "—"
        punt = f"{eq['score_max_abs_diff']:.1e}" if eq.get("score_max_abs_diff") else "—"
        lineas.append(
            f"{formato:12s} {r.status:14s} {tamano:>10s} {coord:>10s} {punt:>10s}  "
            f"{r.notes[:34]}"
        )
    lineas += [
        "",
        "coord. = peor diferencia en píxeles frente a PyTorch; punt. = en puntuaciones.",
        "'no_disponible' NO es un fallo: es un dato sobre esta máquina.",
    ]
    return "\n".join(lineas)
