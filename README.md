# Optimización de modelos de detección de objetos bajo restricciones de recursos: cuantización, poda y destilación

Trabajo Fin de Máster. Máster en Ingeniería de Telecomunicación, Universidad de Sevilla, 2026.
Autor: Antonio Jesús Gómez Blanco. Tutor: Rafael Boloix Tortosa.

Estudio comparativo de tres técnicas de compresión (cuantización PTQ y QAT, poda estructurada y
destilación de conocimiento) aplicadas a detectores YOLOv8 (`n`, `s`, `m`, `l`) para la detección de
equipos de protección individual con el dataset SH17. Todas las variantes se miden con el mismo
protocolo, y los resultados se guardan en un registro único en `results/registry/`.

## Estructura del repositorio

| Directorio | Contenido |
|---|---|
| `configs` | Configuración del conjunto de datos (`dataset`), de las técnicas de compresión (`compression`) y del protocolo de medida (`benchmark`), y configuración por defecto del proyecto (`default.yaml`) |
| `data` | No versionado. Conjunto de datos original (`raw`), convertido (`processed`) y redimensionado (`resized_640`), particiones (`splits`), conjunto de calibración (`calibration`) y pesos preentrenados (`weights`) |
| `docs` | Protocolo de medida (`PROTOCOLO_MEDIDA.md`), esquema de los ficheros de resultados (`ESQUEMA_RESULTADOS.md`), atribuciones de las imágenes de SH17 (`SH17_ATTRIBUTIONS.md`) y ficheros de referencia que lee el propio código (`evidencia`) |
| `models` | No se versionan los pesos. Modelos de referencia (`baselines`), podados (`pruned`), afinados con cuantización simulada (`qat`) y destilados (`distillation`); solo se versionan la configuración, las curvas de entrenamiento y el coste de cada ejecución |
| `notebooks` | Cuadernos de Google Colab: preparación del entorno (`00_colab_setup`), poda con reajuste (`01_colab_prune_finetune`), afinado con cuantización simulada (`02_colab_qat`) y destilación (`03_colab_distill`) |
| `results` | Registro de experimentos y medidas (`registry`), resultados en bruto (`raw`), artefactos exportados y cuantizados con sus volcados de configuración (`exports`), registros de ejecución (`logs`, no versionados), tablas (`tables`), figuras (`figures`) y campañas de latencia archivadas (`archive`) |
| `scripts` | Scripts numerados del flujo experimental y tres auxiliares sin número |
| `src/tfm_edge` | Código reutilizable, en los paquetes `analysis`, `benchmark`, `common`, `compression`, `data`, `evaluation`, `export`, `models`, `reporting` y `training` |
| `tests` | Pruebas automatizadas |

## Instalación

El entorno de referencia es Linux (Ubuntu bajo WSL2) con Python 3.11.

```bash
conda env create -f environment.yml
conda activate tfm-edge
```

PyTorch se instala aparte, con la variante que corresponda a la máquina, antes del resto de
dependencias. Por ejemplo, para CPU:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

Para GPU se usa el índice de la variante CUDA correspondiente (por ejemplo `.../whl/cu126`).
Después:

```bash
pip install -r requirements.txt
pip install -e .
python scripts/00_setup_check.py --create-dirs
```

`00_setup_check.py` informa de qué componentes están disponibles. ONNX Runtime, OpenVINO, TensorRT y
TFLite son opcionales: si faltan, las medidas correspondientes se registran como `no_disponible`.
Las pruebas se ejecutan con:

```bash
pytest tests/
```

## Dataset SH17

El dataset no se incluye en el repositorio. Se obtiene de su fuente oficial:

- Repositorio: https://github.com/ahmadmughees/SH17dataset
- Descarga: https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection

Con el archivo descargado de Kaggle:

```bash
python scripts/01_download_dataset.py --archive ruta/al/archivo.zip
python scripts/02_prepare_dataset.py
python scripts/02b_resize_dataset.py --target-size 640
```

El paso 01 comprueba la integridad del archivo y lo descomprime en `data/raw/`; el 02 valida la
procedencia, convierte las anotaciones a COCO y genera las particiones; el 02b crea la copia a 640 px
que se usa para entrenar.

## Orden de los scripts

Cada script admite `--help`. Los entrenamientos, la poda con reajuste, QAT y la destilación se
ejecutaron en Colab con los cuadernos de `notebooks/`; la exportación, la cuantización y las medidas
de latencia y memoria, en local.

| Paso | Script | Qué hace |
|---|---|---|
| Entorno | `00_setup_check.py` | Comprueba el entorno |
| Dataset | `01_download_dataset.py` | Verifica y descomprime SH17 |
| | `02_prepare_dataset.py` | Valida, convierte a COCO y particiona |
| | `02b_resize_dataset.py` | Copia redimensionada para entrenar |
| | `03_dataset_eda.py` | Análisis exploratorio del dataset |
| Baselines | `04_train_baselines.py` | Entrena yolov8n/s/m/l |
| | `05_evaluate_baselines.py` | Evaluación con el validador de Ultralytics |
| | `06_evaluate_coco.py` | Evaluación COCO con los dos criterios de tamaño |
| Medida | `07_benchmark.py` | Latencia, memoria, rendimiento y tamaño sobre un perfil |
| Exportación y PTQ | `08_export.py` | Exporta a los formatos de despliegue |
| | `09_sensitivity.py` | Sensibilidad a la cuantización por capa |
| | `10_quantize_ptq.py` | PTQ a INT8 en cada runtime |
| | `11_evaluate_artifact.py` | Calidad de un artefacto exportado |
| | `12_score_resolution.py` | Coste en mAP de la resolución de puntuación INT8 |
| Poda | `13_prune.py` | Poda estructurada sobre pesos entrenados |
| | `14_import_colab_registry.py` | Incorpora al registro local lo medido en Colab |
| | `15_normalize_pruned_layout.py` | Ordena los checkpoints podados |
| QAT | `16_qat.py` | Afinado con cuantización simulada |
| | `17_register_quality.py` | Inscribe la calidad de un checkpoint existente |
| | `18_quantize_qat.py` | Materializa el INT8 de la rama QAT |
| Mantenimiento | `19_normalize_registry_paths.py`, `21_backfill_evaluator.py` | Normalizan filas ya inscritas |
| Destilación | `20_distill.py` | Destilación de yolov8l a yolov8n |
| Combinaciones | `22_combine.py`, `benchmark_grid.py` | Poda con PTQ y su medida de latencia |
| Análisis | `23_operating_curve.py`, `24_operating_analysis.py`, `25_matching_ranking.py` | Curvas a umbral fijo y reparto de la pérdida |
| | `26_analyze_pareto.py`, `27_build_pareto_figures.py` | Frontera de Pareto y sus figuras |
| | `29_analyze_results.py`, `30_build_results_figures.py` | Análisis de cobertura, degradación y selección, y sus figuras |
| | `31_register_export_cost.py` | Coste de exportación en el registro |
| | `32_distill_seed_variance.py` | Campaña de semillas de la destilación |
| | `34_register_error_decomposition.py` | Descomposición de los tipos de error |
| Memoria | `28_build_report.py` | Tablas y figuras de la memoria |
| | `33_build_design_figures.py`, `35_build_intro_figures.py` | Figuras de los capítulos 4 y 1 |
| Auxiliares | `measure_memory_envelope.py`, `measure_training_throughput.py` | Memoria de GPU y cuello de botella del entrenamiento |

## Datos y modelos

Los datos y los pesos de los modelos entrenados no se incluyen por su tamaño. De los artefactos
exportados solo se incluyen los volcados de configuración y los del modelo small. Los resultados
medidos, las tablas y las figuras están en `results/`.

## Licencia del dataset

SH17 se distribuye bajo la licencia CC BY-NC-SA 4.0 (atribución, uso no comercial y compartir igual)
y sus imágenes proceden de Pexels. La cita del dataset y la atribución de cada imagen están en
`docs/SH17_ATTRIBUTIONS.md`.
