# Protocolo de medida homogéneo

Documento vinculante del protocolo con el que se miden todas las variantes
del estudio. La sección 12 de la especificación lo declara la pieza más
importante del proyecto: si el protocolo no es idéntico para todas las variantes,
las comparaciones del capítulo 6 no significan nada.

---

## Índice

- [1. Criterio de tamaño de objeto](#1-criterio-de-tamaño-de-objeto)
- [2. Perfiles válidos e inválidos para medir](#2-perfiles-válidos-e-inválidos-para-medir)
- [3. Materialización INT8: PTQ y QAT comparten camino](#3-materialización-int8-ptq-y-qat-comparten-camino)
- [4. Temperatura obligatoria en perfiles con GPU](#4-temperatura-obligatoria-en-perfiles-con-gpu)
- [5. Procedimiento implementado](#5-procedimiento-implementado)
- [6. Evaluación de calidad](#6-evaluación-de-calidad)

---

## 1. Criterio de tamaño de objeto

Material del apartado 4.3.1 de la memoria.

### Qué se hace

Se calculan y se registran siempre los dos criterios, en columnas separadas
y con nombres que dicen cuál es cuál. Nunca `ap_small` a secas.

| | Criterio | Cómo se calcula |
|---|---|---|
| Primario | Resolución de inferencia | Los umbrales COCO se aplican sobre el área que el objeto ocupa a la entrada del modelo, tras el redimensionado *letterbox* a `model.input_resolution` |
| Secundario | Resolución original | Los umbrales se aplican sobre el área en la imagen original. Es el comportamiento estándar de `pycocotools` |

Los umbrales son los de COCO en ambos casos: pequeño < 32², mediano ≤ 96²,
grande > 96² píxeles cuadrados.

### Por qué el primario es el de resolución de inferencia

La pregunta de investigación del TFM no es *qué tamaño físico tiene el objeto en
la fotografía*, sino qué le pasa a la capacidad del modelo de detectar objetos
que ocupan pocos píxeles cuando se comprime. Esa es la magnitud que mide el
criterio de inferencia y que el de resolución original no mide.

El detector nunca ve la imagen de 8192 px: ve un redimensionado a 640×640.
Un objeto de 100×100 px en el original ocupa unos 8×8 px a la entrada de la red.
Para el detector, eso es un objeto pequeño, con independencia de que en la
fotografía midiera 100×100.

### La evidencia empírica que lo respalda

Medición sobre las 75.994 anotaciones de SH17:

| Criterio | small | medium | large |
|---|---:|---:|---:|
| Resolución de inferencia (640, letterbox) | 35,7 % | 36,6 % | 27,6 % |
| Resolución original | 2,1 % | 9,5 % | 88,3 % |

Los umbrales COCO se definieron para imágenes de ~640×480. Las de SH17 van de
4000×6000 a 8192×5461, entre 50 y 100 veces más píxeles, así que sobre la
resolución original casi cualquier objeto supera los 96² = 9.216 px².

El efecto sobre la tabla cruzada clase × tamaño, que es la que sostiene el
análisis del apartado 6.5.2:

| | Criterio de inferencia | Criterio original |
|---|---:|---:|
| Clases con cero instancias pequeñas | 1 | 4 |
| Clases con menos de 50 instancias pequeñas | 3 | 10 |

Con el criterio original, el AP para objetos pequeños se calcularía sobre 1.631
instancias repartidas entre 17 clases, y en cuatro clases sobre ninguna: no
tendría potencia estadística para comparar técnicas de compresión, que es
precisamente para lo que se quiere.

### El redimensionado es *letterbox*, no estirado

Ultralytics preserva la relación de aspecto y rellena los bordes. El factor de
escala es por tanto

```
s = min(R/W, R/H)
```

aplicado a ambas dimensiones, y el área escala con s².

Usar `(R/W)·(R/H)`, que es lo que correspondería a un estirado, sobreestima el
área de toda imagen no cuadrada y desplaza objetos hacia rangos mayores. Sobre
SH17 la diferencia no es marginal: con el factor del estirado el reparto sale
29,7 / 36,6 / 33,7, y con el correcto 35,7 / 36,6 / 27,6.

### La resolución de inferencia es un parámetro

`model.input_resolution` puede acabar siendo 512 o 416 en lugar de 640. El
criterio primario depende de ella, así que:

- Cada valor calculado registra con qué resolución se obtuvo, en las columnas
  `size_criterion_primary` y `size_criterion_inference_resolution` de
  `quality_metrics.csv`.
- Sin ese dato, dos ejecuciones a 640 y a 416 producirían números incomparables
  sin que nada lo delatara.

Nada en el código asume 640. El valor viaja siempre por configuración.

### Columnas del registro

La sección 16.3 de la especificación pide `ap_small`, `ap_medium`, `ap_large` y
sus equivalentes de AR. Se sustituyen por columnas explícitas, porque sobre SH17
el mismo objeto es "pequeño" bajo un criterio y "grande" bajo el otro:

```
size_criterion_primary
size_criterion_inference_resolution
ap_small_inference_res    ap_small_original_res
ap_medium_inference_res   ap_medium_original_res
ap_large_inference_res    ap_large_original_res
ar_small_inference_res    ar_small_original_res
ar_medium_inference_res   ar_medium_original_res
ar_large_inference_res    ar_large_original_res
```

La extensión está declarada en `registry.QUALITY_SCHEMA_EXTENSIONS` y la
correspondencia con los nombres de la especificación en
`registry.QUALITY_SPEC_COLUMN_MAPPING`, de modo que los tests distinguen una
extensión deliberada de una deriva accidental del esquema.

### Requisito operativo

Las detecciones se reescalan a coordenadas de la imagen original antes de
evaluar con `pycocotools`. El criterio de inferencia se obtiene aplicando el
factor `s²` al área, no evaluando en coordenadas redimensionadas: así ambos
criterios salen de la misma evaluación y no hay dos caminos que puedan
divergir.

---

## 2. Perfiles válidos e inválidos para medir

> Google Colab y SageMaker entrenan. El hardware físico del autor mide.

SageMaker no llegó a usarse: la validación del pipeline sobre servicios
gestionados quedó fuera del alcance, y lo único que se entrenó en la nube fue en
Colab. La regla se conserva porque el rechazo sigue en el código como guarda:
impide medir en cualquier instancia de nube, se use o no.

`BenchmarkConfig.valid_for_measurement` marca si un perfil produce medidas de
latencia interpretables. El protocolo debe rechazar con error explícito una
campaña de medida sobre un perfil marcado como no válido; no es un aviso.
`invalid_reason` es obligatorio cuando el perfil no vale, para que el rechazo se
pueda explicar.

Motivo: los percentiles de latencia (p95, p99) son la parte más sensible a la
contención. Medirlos en una máquina compartida, con hardware que cambia entre
sesiones, produce números que no son comparables ni entre sí, y contaminarían la
frontera de Pareto del apartado 6.4.

### Perfiles definidos y perfiles medidos

| Perfil | Descripción | En el registro |
|---|---|---|
| `workstation_cpu` | i7-10750H, todos los hilos disponibles | 50 medidas |
| `workstation_gpu` | GTX 1650 Max-Q, 4 GB, con registro de temperatura obligatorio | `no_disponible`: el PyTorch instalado es la compilación CPU (`torch 2.8.0+cpu`) |
| `constrained_cpu` | La misma CPU limitada a 2 hilos, con afinidad de núcleos fijada | Sin filas |
| `edge_cpu_arm`, `edge_gpu_jetson` | Dispositivos de borde | `no_disponible`: no hay hardware |

Todas las latencias del registro son, por tanto, de la CPU de la estación de
trabajo. TensorRT también figura como `no_disponible`.

`constrained_cpu` es una aproximación a un cómputo restringido, no una emulación
de ARM. No permite extrapolar a un dispositivo ARM real: la microarquitectura, la
jerarquía de caché y el conjunto de instrucciones son distintos. Lo que sí
permite es responder a una pregunta bien definida: cómo cambia la frontera de
Pareto al reducir el presupuesto de cómputo sobre el mismo hardware.

`num_threads` y `cpu_affinity` se registran en `deployment_metrics.csv`: son lo
único que distingue `workstation_cpu` de `constrained_cpu`, que corren sobre la
misma CPU física.

### TFLite: alcance reducido y declarado

La exportación a TFLite está implementada, pero su latencia sobre x86 no
representa a un dispositivo ARM. Si se midiera, esas filas llevarían
`latency_representative = false` y `ExperimentRegistry.latency_comparable_rows()`
las excluiría de las fronteras de Pareto de latencia, reportando cuántas
descartó. `tensorflow` no se instala en el entorno de referencia, así que el
registro no tiene filas de TFLite.

---

## 3. Materialización INT8: PTQ y QAT comparten camino

Condiciona la validez del capítulo 5.2.

QAT se entrena y se evalúa como técnica en PyTorch eager. Para cada runtime, el
INT8 se materializa con el mecanismo nativo de ese runtime, partiendo de los
pesos ya afinados por QAT.

Requisito: las ramas PTQ y QAT deben recorrer exactamente el mismo camino de
materialización INT8 en cada runtime, con el mismo conjunto de calibración y los
mismos parámetros. La única diferencia admisible entre
ambas es el punto de partida: pesos del baseline frente a pesos afinados con QAT.

Si se materializaran por caminos distintos, cualquier diferencia de mAP
observada sería atribuible tanto a la técnica como al camino de exportación, y la
comparación no diría nada.

---

## 4. Temperatura obligatoria en perfiles con GPU

La sección 12.1.8 de la especificación plantea registrar la temperatura "si es
accesible". En este proyecto es obligatorio en todo perfil
con GPU NVIDIA.

Motivo: la GPU de medida es una Max-Q de portátil con un tope de potencia
medido de 35,00 W, que en reposo y sin carga ya está a 52-59 °C. El
margen térmico es estrecho desde el primer instante, así que la limitación
térmica no es una posibilidad remota sino lo esperable (hipótesis H2).

Sin temperaturas al inicio y al final de cada repetición no se puede distinguir
una latencia p99 alta causada por el modelo de una causada por estrangulamiento
térmico, y el análisis de dispersión de la sección 12.1.7 sería ininterpretable.

Como ningún perfil con GPU llegó a medir, las columnas de temperatura y de
memoria de GPU están vacías en todas las filas del registro.

---

## 5. Procedimiento implementado

Implementado en `benchmark/protocol.py`, con los valores por defecto de la
sección 12 de la especificación.

| Medida | Cómo |
|---|---|
| Latencia | 50 iteraciones de calentamiento descartadas, 300 de medida, 3 repeticiones en bloques separados, lote 1, sobre imágenes reales del conjunto y en orden fijo |
| Dispersión | Media, desviación, mínimo, máximo, p50, p90, p95, p99 y el vector completo de tiempos, más la desviación entre repeticiones |
| Memoria | Pico de GPU (asignada y reservada), RSS en reposo tras cargar el modelo, y RSS pico muestreado en un hilo secundario. Los tres por separado, para separar pesos de activaciones |
| Rendimiento | Medida aparte de la latencia, con lotes 1, 2, 4, 8 y 16 sobre una ventana sostenida. Un fallo de memoria en un lote no aborta el barrido |
| Artefacto | Tamaño en disco y comprimido con gzip, como cota del coste de distribución por red |
| Carga previa | Se rechaza medir con más de 0,5 de carga por núcleo, salvo `--force`, y entonces queda anotado en el resultado |
| Temperatura | Al inicio y al final de cada repetición, obligatoria en perfiles con GPU |

### Lo que el protocolo rechaza, con error y no con aviso

1. Medir sobre un perfil no válido (`cloud_gpu`): Colab y SageMaker entrenan.
2. Medir con la máquina cargada, salvo `--force`.
3. Medir sobre una sola imagen: el coste del postprocesado depende del número
   de detecciones, y una imagen fija siempre da las mismas.

### Una comprobación que el protocolo se hace a sí mismo

Si ninguna iteración produce detecciones, la supresión de no máximos no llega a
ejecutarse y el tiempo de postprocesado no representa nada. Se marca en el
resultado (`postprocess_representative`) y se avisa. La latencia de inferencia
sigue siendo válida.

### Preprocesado y postprocesado, compartidos

`InferenceBackend` los implementa una sola vez y las subclases no los
sobrescriben. Si cada motor trajera los suyos, las diferencias de latencia
medirían también esas implementaciones y no el motor.

### Motores medidos

| Motor | Qué se mide | Precisiones en el registro |
|---|---|---|
| PyTorch | El checkpoint `.pt` (`--checkpoint`) | FP32 (los cuatro baselines) |
| ONNX Runtime | El `.onnx` exportado (`--artifact`), con `CPUExecutionProvider` | FP32 e INT8 |
| OpenVINO | El IR exportado (`--artifact`), compilado para CPU | FP32 e INT8 |

Si se pide GPU y falta su proveedor, ONNX Runtime falla en lugar de pasar a CPU.
En OpenVINO el tamaño
del artefacto se mide sobre el directorio del IR, que incluye el `.xml` y el
`.bin`.

Las 50 filas medidas del registro comparten condiciones: perfil
`workstation_cpu`, lote 1, 640 px, 6 hilos, 50 iteraciones de calentamiento, 300
de medida y 3 repeticiones. Cubren los baselines `n`, `s`, `m` y `l`, la PTQ de
los cuatro tamaños, los QAT, las podas y las combinaciones de poda y PTQ.

### Régimen de medida

Las 50 filas se midieron en una única sesión: todas comparten el mismo
`environment_id`. Cada fila declara en `notes`, y cada volcado en
`measurement_regime`, el régimen «campaña encadenada con presaturación»: las
medidas se ejecutan una tras otra en la misma sesión, precedidas de ejecuciones
de presaturación, y la opción `--regime` de `07_benchmark.py` lo guarda en el
volcado y en `notes` sin contarlo como aviso.

### Controles y suelo de lectura

En la sesión se midió el mismo artefacto de control al principio, a la mitad y
al final. Antes de medir se fijaron tres criterios para esos controles: una
banda de referencia de 53,8 a 58,0 ms, un margen de alarma de 52 a 60 ms y una
deriva máxima de 1,10 entre la latencia del último control y la del primero.

`analysis/floors.py` calcula, para cada eje, el suelo por debajo del cual una
diferencia no se presenta como resultado. Lo mide con pares de control: dos
medidas de coste equivalente cuya diferencia es error de medida. Hay pares entre
procesos de la misma sesión, entre ellos los tres controles de contorno, y pares
entre sesiones. Sobre el mismo par, `latency_p50_ms` varía un 9,3 % y
`latency_p95_ms` un 14,7 %. El suelo es una cota inferior del ruido, no una
estimación de la varianza.

---

## 6. Evaluación de calidad

`evaluation/coco_eval.py` produce los dos criterios de tamaño de una sola
generación de detecciones, escalando referencia y detecciones por el mismo factor
de letterbox. Como la IoU es invariante al escalado uniforme, las métricas
globales deben salir idénticas: se comprueba, y si no lo son se falla, porque
el desglose por tamaño no sería de fiar.

Las detecciones se llevan antes a coordenadas de la imagen original, midiendo el
factor a partir de la dimensión real del fichero.

### El evaluador canónico es pycocotools

Ultralytics y pycocotools no dan el mismo mAP, y la brecha depende de la
clase, así que no se puede corregir con un factor. Toda cifra de calidad
que entre en una tabla del capítulo 5 sale de `06_evaluate_coco.py`, incluidos los
baselines: una fila que solo haya pasado por `05_evaluate_baselines.py` se
reevalúa antes de compararla con nada. Cada fila declara su evaluador en el
prefijo `evaluator=` del campo `notes`, para que la regla sea comprobable al leer
el registro.

### Punto de operación

`precision_at_best_f1`, `recall_at_best_f1` y `f1_best` se calculan sobre las
curvas P-R de pycocotools a IoU 0,50, buscando el umbral de confianza único y
común a todas las clases que maximiza el F1 medio, promediado sobre las clases
con anotaciones en la partición. Un umbral por clase describiría un
sistema que no se puede desplegar.

### La calidad de un artefacto exportado se mide, no se hereda del checkpoint

Una variante comprimida se despliega como artefacto (ONNX, IR de OpenVINO), y su
mAP no es el del checkpoint del que salió: la exportación y la cuantización
lo cambian. `11_evaluate_artifact.py` lo mide por el mismo camino que el
checkpoint (el mismo validador de Ultralytics genera las detecciones y
pycocotools las evalúa), de modo que la diferencia entre los dos sea atribuible
al artefacto y no al protocolo.

Dos consecuencias operativas:

- El FP32 de cada motor se mide también, aunque no sea una variante del
  estudio: es el control contra el que se resta el INT8 de ese mismo motor. Sin
  él, la caída medida incluiría el coste de exportar.
- El artefacto se prepara, no se modifica. Los metadatos que Ultralytics
  necesita se inyectan en una copia; el artefacto que aparece en
  `deployment_metrics` conserva sus bytes y su tamaño, y una comprobación de
  igualdad del grafo garantiza que las dos filas hablan del mismo modelo.

### Un modelo podado se mide dos veces, y las dos cifras se registran

El apartado 5.3.2.2 compara cuánto degrada la poda con cuánto recupera el
reajuste, así que las dos medidas son resultado y ninguna sustituye a la otra.
Quedarse solo con la de después haría indistinguibles dos casos opuestos: que la
red aguantara el recorte de sobra, y que la red se hundiera y el reentrenamiento
la rescatara.

Tres reglas que hacen la comparación válida:

- Las dos se miden sobre ficheros en disco, no sobre el modelo en memoria.
  Ultralytics entrena con media móvil exponencial y su `best.pt` no contiene los
  pesos que quedan en memoria al terminar; medir el objeto vivo daría un número
  que nadie puede reproducir desde el checkpoint que sí se registra y se exporta.
- La reducción de parámetros que va a la memoria es la medida, no el ratio
  pedido. Pedir un 50 % da un 67,5 % real, porque podar un canal de salida
  arrastra el de entrada de todo lo que viene detrás.
- La poda tiene que sobrevivir al guardado, y se comprueba. El entrenador ya
  vigila el modelo en memoria; `finetune.py` vigila además el fichero, que es lo
  que sigue viaje hacia la exportación. Si la poda se perdiera ahí, todo seguiría
  funcionando y las tablas describirían un modelo sin podar.

Y el control aleatorio no es opcional para concluir nada: sin podar al azar al
mismo ratio y con las mismas protecciones, una caída pequeña de mAP no
distingue "el criterio acierta" de "la red aguanta cualquier recorte".
