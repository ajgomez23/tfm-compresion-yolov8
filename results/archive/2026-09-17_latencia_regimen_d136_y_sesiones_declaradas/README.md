# Latencia bajo el régimen de D-136, y las dos sesiones declaradas

Archivado el 17/09/2026, **antes** de la campaña única de latencia de
[D-162](../../../docs/DECISIONES.md). Esa campaña remide las 50 filas medibles de
`deployment_metrics` en una sola sesión, y los nombres de los volcados son
deterministas: **remedir los pisa**. Esto es lo que había, copiado byte a byte.
Una medición no vale menos por ser antigua.

## Qué hay aquí

| Fichero | Qué es |
|---|---|
| `raw_benchmark/` | Los **50** volcados de `results/raw/benchmark/`, con el vector completo de tiempos de cada medida |
| `deployment_metrics.csv` | La tabla de despliegue tal como estaba confirmada en git (54 filas: las 50 medidas y 4 `no_disponible`) |

Las 50 medidas son de tres sesiones, todas bajo el régimen declarado
«campaña encadenada con presaturación (D-136)»:

| Sesión | `environment_id` | Filas | Controles de contorno |
|---|---|---:|---|
| 02-03/09/2026 ([D-136](../../../docs/DECISIONES.md)) | `0f3e1bdeec35` | 46 | 54,43 / 53,60 / 54,53 ms; el segundo, fuera de banda |
| 15/09/2026 ([D-161](../../../docs/DECISIONES.md)) | `6484b9f04e71` | 2 (`s_qat_int8-noexcl_43300d34`) | 54,97 / 53,86 / 54,86 ms, en banda |
| 17/09/2026 ([D-164](../../../docs/DECISIONES.md)) | `89fc1928f7e9` | 2 (`s_qat_int8-noexcl_4d491880`) | 52,87 / 53,41 / 53,40 ms, los tres fuera de banda |

Los controles, las referencias y los resúmenes de cada sesión están en
`docs/evidencia/d136_*`, `d161_*` y `d164_*`.

## Por qué se archiva, además de por R8

Los 44 pares de control **entre sesiones** de `src/tfm_edge/analysis/floors.py`
comparan cada toma de agosto (`../2026-09-01_latencia_pre_regimen_declarado/`)
con su remedida de D-136. Hasta este archivo, esa remedida se leía de
`results/raw/benchmark/`, que la campaña única va a sobrescribir. Sin moverla, los
pares habrían pasado a comparar agosto contra la campaña nueva con su
justificación —«la del 02/09 bajo el régimen de D-136»— intacta y ya falsa, y el
suelo de las fronteras de Pareto habría cambiado sin que nada fallara. Es el daño
colateral 2 de D-136, y se evita apuntando los pares aquí **antes** de medir.

Como la copia es idéntica byte a byte, los suelos salen exactamente iguales con
los pares apuntando aquí o al directorio vivo. Se comprobó antes de confirmar el
cambio.

## Lo que este archivo no es

No es la fuente del suelo de la campaña única. Los contrastes entre estas tomas y
las de la campaña única se calculan y se reportan, pero **no alimentan el cálculo
de los suelos** (decisión del autor, 17/09/2026): sería medir la variación entre
sesiones con la misma campaña que se está evaluando.

`tests/test_measurement_regime.py` comprueba que los 50 volcados de aquí declaran
el régimen de D-136. Si alguno no lo hiciera, no sería esta evidencia.

## Fecha corregida

Renombrado el 17/09/2026 ([D-165](../../../docs/DECISIONES.md)). El directorio se
llamaba `2026-09-18_latencia_regimen_d136_y_sesiones_declaradas` y este README decía «Archivado el 18/09/2026» y «decisión del
autor, 18/09/2026»; se creó y se comitió el **17/09/2026** (`3fdcded`, 11:18). El
renombrado es un `git mv`: los 52 ficheros de datos son idénticos por sha256, y
solo cambian las dos fechas de este README y esta sección.
