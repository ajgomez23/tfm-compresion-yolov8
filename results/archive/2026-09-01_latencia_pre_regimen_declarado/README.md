# Campaña de latencia anterior al régimen declarado

Archivada el 01/09/2026, **antes** de volver a medir. No es una copia de
seguridad rutinaria: es la evidencia que la campaña nueva va a sobrescribir, y
se guarda porque una medición no vale menos por ser antigua.

> **Corregido el 02/09/2026.** La versión anterior de este README atribuía la
> anomalía del 31/08 a una saturación térmica sostenida con un factor de ×1,62.
> Esa explicación se puso a prueba y **es falsa**. Lo que la sustituye está abajo
> y en [D-136](../../../docs/DECISIONES.md). Se corrige en vez de reescribirse
> porque el texto original documenta qué se creía al archivar, y eso es parte de
> por qué se archivó.

## Qué hay aquí

| Fichero | Qué es |
|---|---|
| `raw_benchmark/` | Los **44** volcados de `results/raw/benchmark/`, con el vector completo de tiempos de cada medida. Sus nombres son deterministas (`<perfil>_<experimento>_<motor>_<precisión>_<resolución>.json`) y **remedir los pisa**: por eso se copian |
| `deployment_metrics.csv` | La tabla de despliegue del árbol de trabajo: 44 filas |
| `deployment_metrics_en_HEAD.csv` | La misma tabla en `HEAD`, 31 filas, sin las de la campaña `n`/`m`/`l` del 31/08 |

Reparto por fecha de medida: 2026-08-12 (5), 2026-08-17 (2), 2026-08-27 (24), 2026-08-31 (13).

Ninguno de estos volcados declara `measurement_regime`: el campo no existía
cuando se tomaron. Es la marca que los distingue de los de la campaña nueva, y
`tests/test_measurement_regime.py` comprueba que sigue siendo cierta —si algún
volcado de aquí lo declarase, significaría que se ha medido encima del archivo.

## Por qué se sustituyen

Ninguna de estas filas está mal medida en el sentido de tener un fallo de
cálculo: el protocolo se ejecutó como está escrito y los vectores de tiempos son
reales. Lo que les falta es una **condición de medida declarada**.

La latencia de esta máquina depende de cuánto tiempo lleve midiendo sin parar, y
**el efecto no es igual para todos los modelos**. Medido el 02/09/2026 sobre
`n`, `s` y `l` con el orden alternado entre pasadas:

| tamaño | en frío | en régimen | efecto |
|---|---|---|---|
| `n` | 48,54 ms | 54,87 – 55,32 | **+13,6 %** |
| `s` | 140,71 ms | 149,70 – 152,30 | **+7,5 %** |
| `l` | 715,82 ms | 721,61 – 730,06 | **+1,4 %** |

Estas filas se midieron en sesiones de duración muy distinta —de minutos a casi
dos horas—, así que **no comparten régimen**. Y como el efecto depende del
tamaño, mezclarlas no introduce un ruido simétrico sino un **sesgo direccional**
sobre la razón `l`/`n`, que se comprime un 10,7 % entre los dos extremos. Es
justo el eje del que cuelga la pregunta del capítulo 6.

La campaña que las sustituye se mide entera bajo un mismo régimen —**«campaña
encadenada con presaturación»**, declarado en la columna `notes` de cada fila y
en el campo `measurement_regime` de cada volcado—, con presaturación previa y un
control de contorno al principio, a la mitad y al final. Ver
[D-136](../../../docs/DECISIONES.md).

## La anomalía del 31/08, que es cosa aparte

Las tres filas de `n` de esta campaña (medidas entre las 17:20 y las 17:24) no
las explica el régimen. `n_baseline_default` con ONNX Runtime FP32 salió en
**82,73 ms**; sobre el mismo artefacto hay 23 medidas en disco y ninguna otra
pasa de 58, ni siquiera con la máquina a 1,01 de carga por núcleo —el doble que
aquel día—. Su suelo, 62,68 ms sobre 900 muestras, está por encima de la media
de todas las demás: la distribución entera está desplazada.

En la misma ventana la guarda de carga rechazó otras dos medidas por 0,51. Las
filas de `m` (17:34–17:57) y de `l` (18:12–18:55) caen fuera de ella y son
creíbles: la de `l` reproduce hoy dentro del 3 %.

La causa no se ha identificado —desde WSL no hay sensores térmicos ni frecuencia
real observable—, solo acotada en el tiempo. Se registra aquí para que quien lea
estas filas sepa que las tres de `n` del 31/08 son las únicas que, además de no
declarar régimen, están **contradichas por medidas posteriores del mismo
artefacto**.

## Qué se puede seguir citando de aquí

Las comparaciones **dentro de una misma sesión y próximas en el tiempo** siguen
siendo tan válidas como antes; lo que deja de sostenerse es comparar filas de
sesiones distintas sin declarar el régimen. El apartado 7 de
`docs/RESULTADOS_MEDIDOS.md` ya acotaba el suelo entre sesiones en un 10–12 %;
lo que añade la verificación del 02/09 es que, además de ese suelo simétrico, hay
un sesgo **direccional** de hasta el 13,6 % que **crece cuanto más pequeño es el
modelo**, y que por tanto no se cancela al comparar dos tamaños entre sí.
