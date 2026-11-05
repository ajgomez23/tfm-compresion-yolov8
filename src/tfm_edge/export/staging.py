"""Prepara un artefacto exportado para evaluarlo con Ultralytics.

Así la evaluación de ONNX y OpenVINO usa el mismo cargador, letterbox, NMS y reescalado que el
baseline, en lugar de reimplementar el postprocesado. Hace falta añadir los metadatos que espera
AutoBackend (stride, task, batch, imgsz, names) y, en OpenVINO, un directorio terminado en
``_openvino_model``. Se trabaja sobre una copia y se comprueba que el grafo no cambia. Los nombres
de clase salen de configs/dataset/sh17_classes.json.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir

logger = get_logger(__name__)

STAGED_TASK = "detect"

#: YOLOv8 reduce hasta 1/32 en la escala más profunda.
STAGED_STRIDE = 32

#: Los artefactos se exportan con lote fijo 1.
STAGED_BATCH = 1


class ArtifactStagingError(TFMEdgeError):
    """Error al preparar un artefacto exportado para su evaluación."""


def _graph_fingerprint(model: Any) -> str:
    """Hash del grafo ONNX sin metadata_props ni nombre del productor."""
    import onnx

    copia = onnx.ModelProto()
    copia.CopyFrom(model)
    del copia.metadata_props[:]
    copia.producer_name = ""
    copia.producer_version = ""
    copia.doc_string = ""
    return hashlib.sha256(copia.SerializeToString()).hexdigest()


def assert_graph_unchanged(original: Path | str, staged: Path | str) -> str:
    """Comprueba que la copia preparada tiene el mismo grafo que el original y devuelve el hash."""
    import onnx

    huella_original = _graph_fingerprint(onnx.load(str(original)))
    huella_preparada = _graph_fingerprint(onnx.load(str(staged)))
    if huella_original != huella_preparada:
        raise ArtifactStagingError(
            f"La copia preparada de '{Path(original).name}' tiene un grafo distinto del original "
            f"({huella_original[:12]} frente a {huella_preparada[:12]}).",
            "La preparación solo debe añadir metadatos.",
        )
    return huella_original


def build_ultralytics_metadata(
    class_names: Sequence[str],
    input_resolution: int = 640,
    batch: int = STAGED_BATCH,
    stride: int = STAGED_STRIDE,
) -> dict[str, Any]:
    """Metadatos que espera AutoBackend, con tipos nativos de Python.

    Sirven para los dos formatos: OpenVINO los escribe tal cual en metadata.yaml y ONNX los pasa por
    metadata_as_onnx_strings.
    """
    if not class_names:
        raise ArtifactStagingError(
            "No hay nombres de clase con los que preparar el artefacto.",
            "Se leen de configs/dataset/sh17_classes.json.",
        )
    return {
        "stride": stride,
        "task": STAGED_TASK,
        "batch": batch,
        "imgsz": [input_resolution, input_resolution],
        "names": {indice: nombre for indice, nombre in enumerate(class_names)},
        "channels": 3,
    }


def metadata_as_onnx_strings(metadata: dict[str, Any]) -> dict[str, str]:
    """Pasa los metadatos a cadenas para ONNX.

    Se usa ``repr`` y no ``json.dumps`` porque las claves de ``names`` son enteros y Ultralytics las
    evalúa como literales de Python.
    """
    return {
        clave: valor if isinstance(valor, str) else repr(valor) for clave, valor in metadata.items()
    }


def stage_onnx_artifact(
    artifact: Path | str,
    dest_dir: Path | str,
    class_names: Sequence[str],
    input_resolution: int = 640,
    batch: int = STAGED_BATCH,
) -> Path:
    """Copia un ONNX y le añade los metadatos que necesita Ultralytics."""
    import onnx

    origen = Path(artifact)
    if not origen.is_file():
        raise ArtifactStagingError(
            f"No existe el artefacto ONNX '{origen}'.",
            "Exportalo antes con scripts/08_export.py o scripts/10_quantize_ptq.py.",
        )

    destino = ensure_dir(dest_dir) / origen.name
    modelo = onnx.load(str(origen))

    # Se vacía antes porque el cuantizador de ONNX Runtime deja entradas propias que hacen fallar la
    # lectura de metadatos de Ultralytics.
    del modelo.metadata_props[:]
    metadatos = metadata_as_onnx_strings(
        build_ultralytics_metadata(class_names, input_resolution=input_resolution, batch=batch)
    )
    for clave, valor in metadatos.items():
        entrada = modelo.metadata_props.add()
        entrada.key = clave
        entrada.value = valor

    onnx.save(modelo, str(destino))
    huella = assert_graph_unchanged(origen, destino)
    logger.info(
        "Artefacto ONNX preparado en '%s' (grafo %s, %d clases)",
        destino.name,
        huella[:12],
        len(class_names),
    )
    return destino


def stage_openvino_artifact(
    artifact: Path | str,
    dest_dir: Path | str,
    class_names: Sequence[str],
    input_resolution: int = 640,
    batch: int = STAGED_BATCH,
) -> Path:
    """Copia un IR de OpenVINO a un directorio ``*_openvino_model`` con su metadata.yaml."""
    import yaml

    origen = Path(artifact)
    if origen.is_dir():
        xmls = sorted(origen.glob("*.xml"))
        if len(xmls) != 1:
            raise ArtifactStagingError(
                f"El directorio '{origen}' tiene {len(xmls)} ficheros .xml y "
                "se esperaba exactamente uno.",
                "Un IR de OpenVINO es un .xml con su .bin del mismo nombre.",
            )
        origen = xmls[0]
    if origen.suffix != ".xml" or not origen.is_file():
        raise ArtifactStagingError(
            f"'{origen}' no es el .xml de un IR de OpenVINO.",
            "Pasa el .xml o el directorio que lo contiene.",
        )
    pesos = origen.with_suffix(".bin")
    if not pesos.is_file():
        raise ArtifactStagingError(
            f"Falta el fichero de pesos '{pesos.name}' junto a '{origen.name}'.",
            "El .xml de OpenVINO es solo el grafo: sin el .bin no hay modelo.",
        )

    # Ultralytics reconoce el formato por este sufijo (AutoBackend._model_type).
    destino = ensure_dir(Path(dest_dir) / f"{origen.stem}_openvino_model")
    shutil.copy2(origen, destino / origen.name)
    shutil.copy2(pesos, destino / pesos.name)

    # Tipos nativos: Ultralytics compara 'batch' con un entero antes de convertir tipos y un "1" en
    # cadena lanza TypeError.
    metadatos = build_ultralytics_metadata(
        class_names, input_resolution=input_resolution, batch=batch
    )
    (destino / "metadata.yaml").write_text(
        yaml.safe_dump(metadatos, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    logger.info("IR de OpenVINO preparado en '%s' (%d clases)", destino.name, len(class_names))
    return destino


def stage_artifact(
    artifact: Path | str,
    runtime: str,
    dest_dir: Path | str,
    class_names: Sequence[str],
    input_resolution: int = 640,
    batch: int = STAGED_BATCH,
) -> Path:
    """Prepara un artefacto según su runtime (onnxruntime u openvino)."""
    if runtime == "onnxruntime":
        return stage_onnx_artifact(
            artifact, dest_dir, class_names, input_resolution=input_resolution, batch=batch
        )
    if runtime == "openvino":
        return stage_openvino_artifact(
            artifact, dest_dir, class_names, input_resolution=input_resolution, batch=batch
        )
    raise ArtifactStagingError(
        f"No hay preparación implementada para el runtime '{runtime}'.",
        "Los runtimes con evaluación de calidad son: onnxruntime, openvino. "
        "TensorRT y TFLite no están disponibles en esta máquina.",
    )
