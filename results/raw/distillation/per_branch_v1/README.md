# Volcados de la campaña `per_branch_v1`

Las cuatro ejecuciones de la campaña de semillas anterior al arreglo de
[D-139](../../../../docs/DECISIONES.md#d-139): semillas 42 y 43, piloto y
control, del 25/08/2026 y del 07/09/2026.

| Volcado | Semilla | Papel | Esquema |
|---|---|---|---|
| `..._combined_t2_pilot10_cf761acd.json` | 42 | piloto | `per_branch_v1` |
| `..._control_alpha0_pilot10_8826269c.json` | 42 | control | `per_branch_v1` |
| `..._combined_t2_pilot10_d97b62d6.json` | 43 | piloto | `per_branch_v1` |
| `..._control_alpha0_pilot10_b14f1f0f.json` | 43 | control | `per_branch_v1` |

## Por qué están aquí y no en el directorio de al lado

Porque **el directorio que lee la campaña contiene una sola campaña vigente**.
`load_distillation_runs` recorre `results/raw/distillation/*.json` sin bajar a
los subdirectorios, así que apartarlos aquí los saca de la resolución automática
sin sacarlos del repositorio.

Mientras estuvieron al lado de los de `per_branch_v2`, `find_run` encontraba dos
ejecuciones distintas en el papel de control de la semilla 42 y **fallaba**, que
es exactamente lo que debe hacer: dos ejecuciones con la misma semilla y el mismo
papel dicen que la campaña se lanzó dos veces, y ahí no se elige, se para. Lo que
no procedía era enseñarle a elegir. Filtrar por `seeding_scheme` habría sido
alargar la clave —arreglar este caso y dejar intacto el mecanismo, que es lo que
el proyecto lleva cuatro veces descartando— y además habría metido en una función
de búsqueda la definición de qué campaña es la vigente, como constante del código.
La ambigüedad no era entre dos ejecuciones de una campaña: era entre **dos
campañas compartiendo directorio**, y eso se arregla en el directorio.

## Por qué se conservan

- Son la evidencia de [D-140](../../../../docs/DECISIONES.md#d-140): las semillas
  42 y 43 salieron **bit a bit idénticas**, que es lo que destapó D-139.
- `cf761acd` y `8826269c` son la medida que sostiene el **+0,005758** del piloto,
  ya citado en la documentación de resultados.
- Una medición no vale menos por ser antigua (R8).

`tests/test_distillation_seed_variance.py::TestSobreLosVolcadosReales` los lee
**desde aquí**: leerlos tiene que seguir disparando la guarda de dispersión
degenerada, que es la mejor regresión que tiene el proyecto sobre datos reales.
