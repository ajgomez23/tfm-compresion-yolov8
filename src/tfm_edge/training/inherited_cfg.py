"""Claves de configuración de Ultralytics que el Trainer no pasa, clasificadas con su motivo.

El Trainer pasa unas pocas claves y hereda el resto con los valores por defecto de Ultralytics. Cada
clave heredada está en IRRELEVANT_WHEN_TRAINING (no interviene al entrenar un detector) o en
INHERITED_ON_PURPOSE (interviene y se acepta su valor). Si Ultralytics añade una clave nueva,
tests/test_inherited_cfg.py falla hasta que se clasifique.
"""

from __future__ import annotations

#: Claves que no intervienen al entrenar un detector.
IRRELEVANT_WHEN_TRAINING: dict[str, str] = {
    # Fijadas por la clase del entrenador
    "task": (
        "La fija DetectionTrainer, que es la clase que el Trainer instancia. No "
        "hay camino por el que este proyecto entrene otra tarea."
    ),
    "mode": (
        "La fija el propio entrenador al arrancar. El valor del YAML no se "
        "consulta para decidir qué se hace."
    ),
    # Tarea distinta de 'detect'
    "overlap_mask": "Segmentación. DetectionTrainer no construye máscaras.",
    "mask_ratio": "Segmentación. DetectionTrainer no construye máscaras.",
    "retina_masks": "Segmentación, y además solo en predicción.",
    "dropout": "Clasificación: es la cabeza de clasificar, que aquí no existe.",
    "pose": "Ganancia de la pérdida de pose. No hay pérdida de pose en detect.",
    "kobj": "Ganancia de objetividad de keypoints. Tarea pose.",
    "copy_paste": "Aumento de segmentación: necesita máscaras de instancia.",
    "copy_paste_mode": "Parámetro del anterior; sin el anterior no hace nada.",
    "auto_augment": "Política de aumento de clasificación, no de detección.",
    "erasing": "Borrado aleatorio de clasificación, no de detección.",
    # Predicción e inferencia interactiva
    "source": "Predicción: de dónde salen las imágenes a predecir.",
    "vid_stride": "Predicción sobre vídeo. Aquí no hay vídeo.",
    "stream_buffer": "Predicción sobre flujo de vídeo.",
    "visualize": "Predicción: vuelca mapas de activación.",
    "augment": (
        "Aumento en tiempo de PREDICCIÓN (TTA). No es el aumento de "
        "entrenamiento, que son las claves de hsv_* en adelante."
    ),
    "agnostic_nms": "NMS de predicción. La validación de entrenamiento usa el suyo.",
    "classes": "Filtro de clases en predicción.",
    "embed": "Devuelve embeddings en vez de detecciones. Modo predict.",
    "show": "Abre una ventana. No hay pantalla en Colab ni en la medida.",
    "save_frames": "Vuelca fotogramas de un vídeo predicho.",
    "save_txt": "Vuelca las predicciones como .txt.",
    "save_conf": "Añade la confianza a ese .txt.",
    "save_crop": "Recorta y guarda cada detección.",
    "show_labels": "Dibujo sobre la imagen de salida de predict.",
    "show_conf": "Dibujo sobre la imagen de salida de predict.",
    "show_boxes": "Dibujo sobre la imagen de salida de predict.",
    "line_width": "Grosor del trazo al dibujar predicciones.",
    # Exportación. El proyecto exporta con 08_export.py, que no usa el Trainer.
    "format": "Exportación. El paso 8 exporta por su cuenta y con sus propios argumentos.",
    "keras": "Exportación a SavedModel de TensorFlow.",
    "optimize": "Exportación a TorchScript móvil.",
    "int8": "Exportación cuantizada. La cuantización del estudio la hacen los pasos 10 y 16.",
    "dynamic": "Ejes dinámicos al exportar.",
    "simplify": "Simplificación del grafo ONNX al exportar.",
    "opset": "Versión de opset al exportar a ONNX.",
    "workspace": "Memoria de trabajo de TensorRT al construir el motor.",
    "nms": "Fusiona NMS dentro del modelo exportado.",
    "dnn": "Inferencia ONNX con el módulo DNN de OpenCV.",
    # Otras
    "tracker": "Seguimiento multiobjeto. El estudio no sigue objetos.",
    "cfg": (
        "Ruta a un YAML que pisaría TODOS los defectos de Ultralytics. Vale "
        "None y tiene que seguir valiendo None: con un fichero ahí, la receta "
        "efectiva dejaría de ser la que el proyecto declara."
    ),
}

#: Claves que sí intervienen al entrenar y se heredan a propósito.
INHERITED_ON_PURPOSE: dict[str, str] = {
    # Ajustes de entrenamiento
    "pretrained": (
        "Inerte aquí, y comprobado: BaseTrainer.setup_model() solo la lee si es "
        "una ruta (str o Path), y además sale antes cuando el modelo ya es un "
        "nn.Module, que es justo lo que hace el Trainer al asignar "
        "trainer.model = detector.model para no perder la poda ni el QAT. "
        "Con True no carga nada."
    ),
    "single_cls": (
        "False: las 17 clases de SH17 se entrenan como 17. Colapsarlas cambiaría "
        "el estudio entero, no una receta."
    ),
    "rect": (
        "False: lotes cuadrados de 640x640. True agruparía por relación de "
        "aspecto y cambiaría el relleno de cada imagen, o sea la entrada del "
        "modelo. Es la resolución fijada para el estudio."
    ),
    "resume": (
        "False, y tiene que quedarse en False: este Trainer no puede reanudar. "
        "La reanudación se gobierna con resume_if_possible, que ante un "
        "last.pt falla con el motivo en vez de empezar de cero en silencio."
    ),
    "fraction": (
        "1.0: se entrena con el conjunto entero. Cualquier otro valor mediría "
        "sobre menos datos sin que la fila del registro lo dijera."
    ),
    "profile": "False: perfilar ONNX/TensorRT durante el entrenamiento solo añadiría coste.",
    "freeze": (
        "None: no se congela ninguna capa. Congelar cambiaría qué se adapta al "
        "podar o al destilar, que es precisamente lo que el estudio mide."
    ),
    "multi_scale": (
        "False: la resolución no varía entre lotes. Con True la entrada cambiaría "
        "de tamaño durante el entrenamiento y la latencia medida a 640 dejaría de "
        "corresponder a la receta."
    ),
    "compile": (
        "False: torch.compile cambiaría el grafo que se entrena y se mide. El "
        "estudio compara compresiones, no compiladores."
    ),
    "cache": (
        "False: no se cachean imágenes en RAM. En Colab la RAM es el recurso "
        "escaso y es lo que decide si la campaña cabe."
    ),
    "save": "True: sin esto no habría checkpoints, que son la salida del entrenamiento.",
    "time": (
        "None: el entrenamiento lo limitan las épocas, no un reloj. Con un valor "
        "aquí, Ultralytics recorta las épocas y la fila diría unas épocas que no "
        "se corrieron."
    ),
    "device": (
        "Se pasa CUANDO el script declara dispositivo; vacío, que es el defecto "
        "del Trainer, deja elegir a Ultralytics. Es lo que corrió en Colab, donde "
        "hay una sola GPU y elegirla es inequívoco."
    ),
    # Validación de cada época. No cambian las cifras publicadas, que salen de pycocotools en el
    # paso 6, pero sí deciden qué época se guarda como best.pt.
    "val": (
        "True: sin validación por época no hay best.pt ni parada temprana, y "
        "patience -que sí se pasa- dejaría de significar nada."
    ),
    "split": (
        "'val': la validación de cada época mira la partición de validación, no "
        "la de prueba. Es lo correcto y coincide con lo que mide el paso 6."
    ),
    "save_json": (
        "False: el JSON de COCO que usa el estudio lo produce el paso 6 con su "
        "propio camino, no el validador de Ultralytics."
    ),
    "conf": (
        "None, que el validador resuelve a 0.001 (engine/validator.py). Es el "
        "umbral bajo habitual para calcular mAP: recortar aquí sesgaría la "
        "elección de best.pt hacia el modelo más conservador."
    ),
    "iou": (
        "0.7 para el NMS de la validación de cada época. Solo interviene en la "
        "elección de best.pt; las cifras del estudio las produce pycocotools con "
        "sus propios parámetros."
    ),
    "max_det": (
        "300 detecciones por imagen en la validación de cada época. Importa "
        "decirlo porque SH17 tiene escenas pobladas: una imagen con más de 300 "
        "detecciones se trunca AL ELEGIR best.pt. Las cifras publicadas no lo "
        "sufren, porque pycocotools usa su maxDets de COCO (100)."
    ),
    "half": (
        "False: la validación de cada época corre en FP32. Es coherente con "
        "el Trainer, que aparta la copia FP32 antes de que Ultralytics la degrade."
    ),
    "plots": (
        "True: genera las curvas de la ejecución. No toca los pesos; es "
        "evidencia que se conserva junto al run."
    ),
    # Calentamiento. warmup_epochs no está aquí porque el Trainer la pasa siempre.
    "warmup_momentum": (
        "0.8 durante el calentamiento, antes de llegar al momentum configurado. "
        "Uniforme en las cuatro campañas; cambiarlo movería el arranque de todos "
        "los entrenamientos ya medidos."
    ),
    "warmup_bias_lr": (
        "0.1 para los sesgos durante el calentamiento. Mismo motivo que el "
        "anterior: es parte de la receta con la que se midió todo."
    ),
    # Ganancias de la pérdida: son las de la receta de YOLOv8 y se mantienen fijas.
    "box": "7.5, ganancia de la pérdida de caja. Receta de YOLOv8, igual en todas las ramas.",
    "cls": "0.5, ganancia de la pérdida de clasificación. Igual en todas las ramas.",
    "dfl": "1.5, ganancia de la pérdida focal distribuida. Igual en todas las ramas.",
    # Aumento de datos: se usa la política por defecto de Ultralytics, igual en todas las técnicas.
    "hsv_h": "0.015 de tono. Aumento activo, heredado, uniforme en las cuatro campañas.",
    "hsv_s": "0.7 de saturación. Aumento activo, heredado, uniforme.",
    "hsv_v": "0.4 de brillo. Aumento activo, heredado, uniforme.",
    "translate": "0.1 de traslación. Aumento activo, heredado, uniforme.",
    "scale": "0.5 de escala. Aumento activo, heredado, uniforme.",
    "fliplr": "0.5 de volteo horizontal. Aumento activo, heredado, uniforme.",
    "mosaic": (
        "1.0, y es la única de este grupo que el proyecto sí vigila: close_mosaic "
        "se pasa siempre y _comprobar_regimen_de_mosaico rechaza entrenar sin un "
        "solo lote mosaicado si no se pide a sabiendas."
    ),
    "degrees": "0.0: sin rotación. El defecto es no hacer nada.",
    "shear": "0.0: sin cizalla. El defecto es no hacer nada.",
    "perspective": "0.0: sin perspectiva. El defecto es no hacer nada.",
    "flipud": "0.0: sin volteo vertical. El defecto es no hacer nada.",
    "bgr": "0.0: sin intercambio de canales. El defecto es no hacer nada.",
    "mixup": "0.0: sin MixUp. El defecto es no hacer nada.",
    "cutmix": "0.0: sin CutMix. El defecto es no hacer nada.",
}


def classify(inherited: set[str]) -> tuple[set[str], set[str], set[str]]:
    """Reparte las claves heredadas entre los dos grupos. Devuelve ``(irrelevantes, a_sabiendas,
    sin_clasificar)``; la última debe quedar vacía.
    """
    irrelevantes = inherited & set(IRRELEVANT_WHEN_TRAINING)
    a_sabiendas = inherited & set(INHERITED_ON_PURPOSE)
    return irrelevantes, a_sabiendas, inherited - irrelevantes - a_sabiendas
