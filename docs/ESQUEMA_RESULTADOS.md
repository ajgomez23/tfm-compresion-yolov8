# Esquema del registro de resultados

El registro (`results/registry/*.csv`) es, junto al protocolo de medida, una de
las dos piezas centrales del proyecto. Todo número que llegue a la memoria sale
de aquí.

Este documento fija qué significa cada columna y quién puede escribirla. El
esquema en sí vive en `src/tfm_edge/common/registry.py`, que es quien lo valida;
esto es la explicación, no la fuente de verdad.

---

## Índice

1. [Reglas que aplican a las cuatro tablas](#1-reglas-que-aplican-a-las-cuatro-tablas)
2. [Qué significa una celda vacía](#2-qué-significa-una-celda-vacía)
3. [Las doce columnas de desglose por tamaño](#3-las-doce-columnas-de-desglose-por-tamaño)
4. [De qué evaluador viene cada fila](#4-de-qué-evaluador-viene-cada-fila)
5. [Desviaciones deliberadas respecto a la sección 16](#5-desviaciones-deliberadas-respecto-a-la-sección-16)

---

## 1. Reglas que aplican a las cuatro tablas

- Se escribe solo a través de `ExperimentRegistry`. Nunca a mano, nunca con
  `pandas.to_csv`. El registro valida el esquema, la clave primaria y las
  referencias entre tablas antes de escribir, y la escritura es atómica
  (temporal + `os.replace`).
- Ninguna columna se añade ni se quita sin anotarlo. Un esquema que deriva sin
  registro convierte el CSV en algo que hay que arqueologizar.
- `status` solo admite `ok`, `fallido` o `no_disponible`. Un fallo se
  registra con su motivo en `notes`; no se omite la fila.
- Toda fila de métricas referencia un `environment_id`, que a su vez identifica
  la máquina y las versiones con las que se midió. Sin eso, un número no es
  interpretable.

---

## 2. Qué significa una celda vacía

Vacío significa "no medido". No `0`, no `-1`, no el texto `NaN`.

La distinción no es cosmética:

| Valor | Significado |
|---|---|
| *(vacío)* | No se midió, o el runtime no estaba disponible |
| `0` | Se midió y salió cero |

Un `ap_small = 0` dice que el modelo no detecta nada pequeño, que es un
resultado. Un `ap_small` vacío dice que nadie lo ha calculado todavía. Confundir
los dos falsearía la memoria en direcciones opuestas según la columna.

---

## 3. Las doce columnas de desglose por tamaño

```
ap_small_inference_res    ap_medium_inference_res    ap_large_inference_res
ap_small_original_res     ap_medium_original_res     ap_large_original_res
ar_small_inference_res    ar_medium_inference_res    ar_large_inference_res
ar_small_original_res     ar_medium_original_res     ar_large_original_res
```

### Regla, y es innegociable

> Estas doce columnas se pueblan únicamente desde la evaluación COCO propia
> del proyecto (`evaluation/coco_eval.py`), nunca desde las métricas nativas de
> Ultralytics ni de ninguna otra biblioteca.

### Por qué

Ultralytics, y cualquier otro validador, reporta un desglose *small / medium /
large* que parece el de estas columnas y no lo es:

1. Aplica un solo criterio, el de COCO sobre las coordenadas con las que
   está evaluando, y no declara cuál es. Este estudio reporta los dos: el de
   resolución de inferencia (primario) y el de resolución original
   (secundario).
2. En SH17 los dos criterios no coinciden. El mismo objeto cambia de rango
   según cuál se aplique, y el reparto entre pequeño, mediano y grande cambia
   por completo. No es un matiz: es la diferencia entre dos lecturas distintas
   del mismo experimento.
3. Un número que parece el correcto y no lo es no se detecta mirando el CSV.
   Sería un `0,41` en la celda que toca, con el formato que toca. Nadie lo
   notaría nunca.

Por eso, hasta que la evaluación COCO propia exista y se ejecute, estas celdas
quedan vacías. `scripts/05_evaluate_baselines.py` las deja así a propósito y
lo dice en `notes`.

### Las dos columnas que dan sentido a las otras doce

| Columna | Qué guarda |
|---|---|
| `size_criterion_primary` | Cuál de los dos criterios se declaró primario |
| `size_criterion_inference_resolution` | Con qué resolución se calculó el criterio de inferencia |

Sin la segunda, dos ejecuciones a 640 y a 416 producirían números incomparables
sin que nada lo delatara: el área de un objeto a la entrada del modelo depende
de la resolución, así que también su rango de tamaño.

---

## 4. De qué evaluador viene cada fila

Ultralytics y pycocotools no dan el mismo mAP, y la brecha depende de la
clase, así que no se puede corregir con un factor. Medido sobre el
baseline real: −0,0077 en mAP@50, con una dispersión por clase de −0,0264 a
+0,0012.

### Regla

> Toda cifra de calidad que entre en una tabla comparativa sale de
> `06_evaluate_coco.py`. Una fila que solo haya pasado por
> `05_evaluate_baselines.py` se reevalúa antes de compararla con nada, aunque sea
> un baseline ya medido.

Mezclar las dos fuentes en una tabla atribuiría a la compresión hasta dos puntos
porcentuales que son del evaluador.

### Cómo se comprueba al leer el registro

El campo `notes` de cada fila de `quality_metrics` empieza por `evaluator=`:

| Prefijo | Quién la escribió | Sirve para el capítulo 5 |
|---|---|---|
| `evaluator=pycocotools` | `06_evaluate_coco.py` | Sí |
| `evaluator=pycocotools` | `11_evaluate_artifact.py` | Sí |
| `evaluator=pycocotools` | `13_prune.py` | Sí, leyendo el resto de `notes` |
| `evaluator=pycocotools` | `16_qat.py` | Sí, leyendo el resto de `notes`: mide FP32, no INT8 |
| `evaluator=pycocotools` | `17_register_quality.py` | Sí; su nota dice además que la fila se inscribió reevaluando el checkpoint |
| `evaluator=ultralytics` | `05_evaluate_baselines.py` | No, hay que reevaluar |

Ninguna columna del esquema de la sección 16.3 guarda el evaluador, y añadir una
habría sido una desviación más del esquema. El prefijo en `notes` deja la regla
comprobable al leer el registro en lugar de confiada a la memoria del autor.

Las filas de `11_evaluate_artifact.py` anotan además en `notes` la brecha
`pycocotools − ultralytics` de esa misma pasada. No hay columnas para ella en el
esquema y el número no puede perderse: es lo que permite comprobar si la brecha
entre evaluadores aguanta bajo compresión sin volver a evaluar.

### Una fila puede haberla escrito otro script que el que produjo el modelo

`17_register_quality.py` puede inscribir la fila de calidad de un
checkpoint que produjo otra campaña. Es lo que permite recuperar una fila perdida
sin repetir el entrenamiento, y la fila lo dice en su propio texto:

```
Fila inscrita por 17_register_quality.py reevaluando 'qat_stripped_fp32.pt': la
evaluación es determinista dados los pesos, la partición, la resolución y los
umbrales, así que es una REMEDICIÓN del mismo artefacto [...].
```

Para el capítulo 5 esa fila vale igual que si la hubiera escrito su campaña: sale
del mismo evaluador, sobre los mismos pesos y con los mismos umbrales. La única
columna que difiere de lo que habría escrito la campaña es `eval_time_s`, que es
el tiempo de *esa* evaluación y no una propiedad del modelo.

Lo que ese script no puede escribir es una fila de `experiments.csv`: la
procedencia de unos pesos no está en el fichero de pesos.

### Qué distingue una fila de checkpoint de una de artefacto

Las dos salen de pycocotools y las dos son comparables entre sí, pero no miden lo
mismo:

| | `06_evaluate_coco.py` | `11_evaluate_artifact.py` |
|---|---|---|
| Qué evalúa | El checkpoint `.pt` | El artefacto exportado |
| `runtime` | `pytorch` | `onnxruntime`, `openvino` |
| Se empareja con | (ninguna) | La fila de `deployment_metrics` de la misma clave |

Una fila de artefacto y su fila de despliegue comparten `experiment_id`,
`runtime`, `precision` y `device_id`, así que describen el mismo objeto: el
mAP y la latencia de esa combinación se pueden llevar a la misma fila de una
tabla del capítulo 5 sin más justificación. Antes de que existiera
`11_evaluate_artifact.py` solo se conocía la mitad de cada par (la latencia), que
es precisamente la mitad que un artefacto roto mejora.

Y el INT8 de un motor se compara con el FP32 de ese mismo motor, no con el
checkpoint: el paso a ONNX o a IR ya introduce una diferencia propia, y restarla
junto con la de la cuantización se la atribuiría a esta.

### La fila de un modelo podado dice si está reajustada, y hay que mirarlo

`13_prune.py` escribe filas con `runtime=pytorch`, igual que `06_evaluate_coco.py`,
y por su clave son indistinguibles de la del baseline. La diferencia está en
`notes`, y es grande: un modelo podado sin reajustar no es comparable con el
baseline como resultado de la técnica. Su mAP mide otra cosa (cuánto degrada la
poda por sí sola, que es la primera mitad del apartado 5.3.2.2) y llevarlo a una
tabla de "poda frente a baseline" exageraría la degradación hasta lo absurdo:
medido en este proyecto, al 67,5 % de reducción esa cifra es cero.

Por eso la fila lo dice en su propio texto:

```
evaluator=pycocotools. MODELO PODADO SIN REAJUSTAR: esta cifra mide lo que
degrada la poda por sí sola (apartado 5.3.2.2) y NO es comparable con el
baseline como resultado de la técnica.
```

Las dos variantes, con y sin reajuste, tienen además `experiment_id` distintos,
porque el hash de configuración incluye si se reajustó. No es posible
sobrescribir una con la otra por accidente.

En `experiments.csv`, `compression_ratio` es el ratio pedido (sección 16.1) y
`effective_params_ratio` el medido. El apartado 5.3 usa siempre el segundo:
pedir un 50 % da un 67,5 % de reducción real, y confundirlos es un error de bulto
que no da ningún síntoma.

### La fila de una destilación dice si es el control, y hay que mirarlo

El mismo problema con otra técnica. `20_distill.py` inscribe dos filas por
campaña: la variante destilada y su control sin destilar, una ejecución con
`alpha=0` que recorre el mismo código con el término de destilación multiplicado
por cero, es decir un afinado normal. Por su clave las dos son filas de
`technique=distillation` del mismo tamaño de modelo, y solo se distinguen por
`notes`:

```
evaluator=pycocotools. CONTROL SIN DESTILAR (alpha=0): esta cifra NO es el
resultado de destilar. […] No la uses como resultado de la técnica.
```

La marca la deriva `build_distillation_note(alpha)` de la `alpha` que de verdad se
ejecutó, no la elige quien inscribe: un control sin marca sería indistinguible de
una variante candidata en la tabla del 5.4.

La fila de la variante destilada lleva su propia advertencia, y también hace
falta: el alumno parte del baseline `n` ya entrenado
(`student_from_pretrained: true`), así que la comparación que aísla el efecto de
destilar es contra el control, no contra ese baseline. Contra el baseline se
mezclan tres cosas: destilar, las épocas adicionales y el cambio de tasa.

Y el suelo de lectura: dos cifras de este estudio que difieran en menos de
0,004579 de mAP@50-95 no son distinguibles, porque eso es lo que se mueve el
mismo checkpoint de `n` entre dos dispositivos de evaluación del registro. Por
debajo se escribe "no distinguible del ruido de dispositivo", nunca "sin efecto".

### Las tres columnas del punto de operación

`precision_at_best_f1`, `recall_at_best_f1` y `f1_best` se calculan sobre las
curvas P-R de pycocotools a IoU 0,50, con el umbral de confianza único y común
a todas las clases que maximiza el F1 medio. Las cifras equivalentes de
Ultralytics no son comparables: monotoniza distinto y suaviza la curva de F1
antes de buscar el máximo.

---

## 5. Desviaciones deliberadas respecto a la sección 16

La especificación (16.3) pedía `ap_small`, `ap_medium`, `ap_large` y sus
equivalentes de AR. Se han sustituido por las doce columnas explícitas de arriba
más las dos de criterio.

Motivo: con dos criterios de tamaño, un `ap_small` a secas sería ambiguo, y la ambigüedad
acabaría en la memoria. El nombre de cada columna dice qué criterio aplica.

Las desviaciones están declaradas en `QUALITY_SCHEMA_EXTENSIONS`, dentro de
`registry.py`, para que los tests distingan una extensión deliberada de una
deriva accidental.
