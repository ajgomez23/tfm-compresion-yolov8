"""Evaluación de un checkpoint hasta las columnas del registro.

Reúne en una sola secuencia el recorte del COCO, las detecciones, el emparejado de identificadores y
la evaluación, para que todos los scripts sigan el mismo camino.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.evaluation import coco_eval
from tfm_edge.export import staging
from tfm_edge.models.detector import ArtifactDetector, Detector

logger = get_logger(__name__)


class EvaluationRunnerError(TFMEdgeError):
    """Error al orquestar una evaluación."""


def assert_evaluation_inputs_available(
    coco_path: Path | str,
    splits_dir: Path | str,
    split: str,
    data_config: Path | str | None = None,
) -> dict[str, str]:
    """Comprueba que existen el COCO de referencia, las listas de partición y el YAML antes de usar
    la GPU.

    El YAML viaja con data/resized_640, pero sh17_coco.json y las listas de data/splits no, así que
    un clon nuevo puede tener uno sin lo otro. Se llama al construir el evaluador.
    """
    lista = Path(splits_dir) / f"{split}.txt"
    piezas: list[tuple[str, Path, str]] = [
        (
            "COCO de referencia",
            Path(coco_path),
            "Lo genera scripts/02_prepare_dataset.py a partir de data/raw. NO se "
            "puede regenerar desde data/resized_640: la validación de procedencia "
            "rechaza por diseño una copia redimensionada (criterios 1 y 2), y las "
            "áreas del COCO están en coordenadas ORIGINALES (D-049). En una "
            "máquina sin data/raw hay que COPIARLO, no rehacerlo.",
        ),
        (
            f"lista de la partición '{split}'",
            lista,
            "La genera scripts/02_prepare_dataset.py junto al COCO, y tiene que "
            "venir de la MISMA preparación que él.",
        ),
    ]
    if data_config is not None:
        piezas.append(
            (
                "YAML de dataset",
                Path(data_config),
                "Va dentro de data/resized_640/, que sí viaja en el .tar.gz.",
            )
        )

    faltan = [(nombre, ruta, remedio) for nombre, ruta, remedio in piezas if not ruta.is_file()]
    if faltan:
        detalle = "\n".join(f"  - {nombre}: no existe '{ruta}'" for nombre, ruta, _ in faltan)
        remedios = "\n".join(f"  - {nombre}: {remedio}" for nombre, _, remedio in faltan)
        raise EvaluationRunnerError(
            f"Falta{'n' if len(faltan) > 1 else ''} "
            f"{len(faltan)} pieza{'s' if len(faltan) > 1 else ''} para poder "
            f"evaluar:\n{detalle}",
            f"conseguirlas antes de seguir.\n{remedios}",
        )
    return {nombre: str(ruta) for nombre, ruta, _ in piezas}


def load_split_file_names(splits_dir: Path | str, split: str, coco: dict[str, Any]) -> list[str]:
    """Nombres de fichero de una partición tal como aparecen en el COCO, emparejados sin
    extensión."""
    lista = Path(splits_dir) / f"{split}.txt"
    if not lista.is_file():
        raise EvaluationRunnerError(
            f"No existe la lista de la partición '{split}' en '{lista}'.",
            "Genera las particiones con scripts/02_prepare_dataset.py.",
        )
    por_stem = {Path(i["file_name"]).stem: i["file_name"] for i in coco["images"]}
    nombres: list[str] = []
    faltan: list[str] = []
    for linea in lista.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        stem = Path(linea.strip()).stem
        if stem in por_stem:
            nombres.append(por_stem[stem])
        else:
            faltan.append(stem)
    if faltan:
        raise EvaluationRunnerError(
            f"{len(faltan)} imágenes de '{split}' no están en el COCO de "
            f"referencia. Primeras: {faltan[:3]}",
            "Las particiones y el COCO tienen que venir de la misma preparación. "
            "Regenéralos con scripts/02_prepare_dataset.py.",
        )
    return nombres


def evaluate_checkpoint(
    checkpoint: Path | str,
    data_config: Path | str,
    coco_path: Path | str,
    splits_dir: Path | str,
    split: str = "val",
    input_resolution: int = 640,
    batch_size: int = 16,
    device: str = "",
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    save_dir: Path | str | None = None,
    with_per_class: bool = True,
) -> dict[str, Any]:
    """Evalúa un checkpoint y devuelve el resultado COCO, las columnas del registro y el tiempo
    empleado.
    """
    ruta_coco = Path(coco_path)
    if not ruta_coco.is_file():
        raise EvaluationRunnerError(
            f"No existe el COCO de referencia '{ruta_coco}'.",
            "Genéralo con scripts/02_prepare_dataset.py.",
        )

    inicio = time.perf_counter()
    completo = json.loads(ruta_coco.read_text(encoding="utf-8"))
    nombres = load_split_file_names(splits_dir, split, completo)
    referencia = coco_eval.subset_ground_truth(completo, nombres)
    logger.info(
        "Partición '%s': %s imágenes, %s anotaciones",
        split,
        f"{len(referencia['images']):,}",
        f"{len(referencia['annotations']):,}",
    )

    detector = Detector.load(checkpoint, input_resolution=input_resolution)
    prediccion = detector.predict_coco_detections(
        data_config,
        split=split,
        batch_size=batch_size,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        save_dir=save_dir,
    )
    detecciones = coco_eval.map_detections_to_ground_truth(prediccion["detections"], referencia)

    # Ultralytics reescala a la imagen en disco, que es la copia de 640 px, y el COCO tiene las
    # dimensiones originales. Sin esta corrección el mAP sale cero.
    directorio_imagenes = Path(data_config).parent / "images"
    detecciones, reescalado = coco_eval.rescale_detections_to_original(
        detecciones, referencia, directorio_imagenes
    )

    resultado = coco_eval.evaluate_coco(
        referencia,
        detecciones,
        input_resolution=input_resolution,
        split=split,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        with_per_class=with_per_class,
    )
    transcurrido = time.perf_counter() - inicio

    return {
        "result": resultado,
        "registry_columns": {
            **resultado.to_registry_columns(),
            "input_resolution": input_resolution,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "split": split,
            "eval_time_s": round(transcurrido, 2),
        },
        "num_raw_detections": prediccion["num_detections"],
        "num_matched_detections": len(detecciones),
        "rescaling": reescalado,
        "eval_time_s": round(transcurrido, 2),
    }


def evaluator_gap(coco_result: Any, ultralytics_metrics: dict[str, Any]) -> dict[str, Any]:
    """Compara el mAP de pycocotools y el de Ultralytics sobre las mismas detecciones.

    Las dos cifras salen de una sola pasada del validador, así que la diferencia solo viene del
    evaluador. El signo es pycocotools menos Ultralytics.
    """
    primario = coco_result.primary
    brecha: dict[str, Any] = {}
    for nombre, atributo in (("map_50", "map_50"), ("map_50_95", "map_50_95")):
        pycoco = getattr(primario, atributo, "")
        ultra = ultralytics_metrics.get(nombre, "")
        brecha[f"pycocotools_{nombre}"] = pycoco
        brecha[f"ultralytics_{nombre}"] = ultra
        # Celda vacía y no 0: una diferencia nula y una desconocida no son lo mismo.
        brecha[f"gap_{nombre}"] = (
            round(float(pycoco) - float(ultra), 6)
            if isinstance(pycoco, (int, float)) and isinstance(ultra, (int, float))
            else ""
        )
    return brecha


def evaluate_artifact(
    artifact: Path | str,
    runtime: str,
    class_names: Sequence[str],
    data_config: Path | str,
    coco_path: Path | str,
    splits_dir: Path | str,
    staging_dir: Path | str,
    split: str = "val",
    input_resolution: int = 640,
    batch_size: int = 1,
    device: str = "cpu",
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    save_dir: Path | str | None = None,
    with_per_class: bool = True,
) -> dict[str, Any]:
    """Evalúa un artefacto exportado por el mismo camino que un checkpoint.

    A partir del validador todo es igual que en evaluate_checkpoint, así que la diferencia con el
    baseline es la degradación por compresión. El artefacto no se modifica; se prepara una copia en
    ``staging_dir``.
    """
    ruta_coco = Path(coco_path)
    if not ruta_coco.is_file():
        raise EvaluationRunnerError(
            f"No existe el COCO de referencia '{ruta_coco}'.",
            "Genéralo con scripts/02_prepare_dataset.py.",
        )

    inicio = time.perf_counter()
    completo = json.loads(ruta_coco.read_text(encoding="utf-8"))
    nombres = load_split_file_names(splits_dir, split, completo)
    referencia = coco_eval.subset_ground_truth(completo, nombres)
    logger.info(
        "Partición '%s': %s imágenes, %s anotaciones",
        split,
        f"{len(referencia['images']):,}",
        f"{len(referencia['annotations']):,}",
    )

    preparado = staging.stage_artifact(
        artifact,
        runtime,
        staging_dir,
        class_names,
        input_resolution=input_resolution,
        batch=batch_size,
    )
    detector = ArtifactDetector(
        preparado,
        runtime=runtime,
        num_classes=len(class_names),
        input_resolution=input_resolution,
    )
    prediccion = detector.evaluate(
        data_config,
        split=split,
        batch_size=batch_size,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        save_dir=save_dir,
    )
    detecciones = coco_eval.map_detections_to_ground_truth(prediccion["detections"], referencia)

    directorio_imagenes = Path(data_config).parent / "images"
    detecciones, reescalado = coco_eval.rescale_detections_to_original(
        detecciones, referencia, directorio_imagenes
    )

    resultado = coco_eval.evaluate_coco(
        referencia,
        detecciones,
        input_resolution=input_resolution,
        split=split,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        with_per_class=with_per_class,
    )
    transcurrido = time.perf_counter() - inicio

    return {
        "result": resultado,
        "registry_columns": {
            **resultado.to_registry_columns(),
            "input_resolution": input_resolution,
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "split": split,
            "eval_time_s": round(transcurrido, 2),
        },
        "ultralytics_metrics": prediccion["ultralytics_metrics"],
        "evaluator_gap": evaluator_gap(resultado, prediccion["ultralytics_metrics"]),
        "artifact": str(Path(artifact)),
        "staged_artifact": str(preparado),
        "runtime": runtime,
        "num_raw_detections": prediccion["num_detections"],
        "num_matched_detections": len(detecciones),
        "rescaling": reescalado,
        "eval_time_s": round(transcurrido, 2),
    }
