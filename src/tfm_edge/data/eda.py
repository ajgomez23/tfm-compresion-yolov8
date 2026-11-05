"""Figuras del análisis exploratorio del dataset (capítulo 3).

El prefijo de cada figura indica el apartado: fig_3_2 presentación, fig_3_3 particionado y
redimensionado, fig_3_4 análisis exploratorio. plot_size_distribution_both_criteria compara los dos
criterios de tamaño de objeto y muestra por qué el de resolución original no sirve como primario.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir
from tfm_edge.data.splits import SPLIT_NAMES
from tfm_edge.data.statistics import (
    CRITERION_LABELS,
    SIZE_CRITERIA,
    SIZE_RANGES,
    DatasetStatistics,
    compute_conditional_cooccurrence,
)
from tfm_edge.reporting.style import (
    SEQUENTIAL_COLORMAP,
    SIZE_STYLES,
    apply_style,
    comma_axis_formatter,
    format_integer,
    format_number,
    grayscale_safe_bars,
    save_figure,
    split_style_for,
)

logger = get_logger(__name__)

EDA_CHAPTER = 3

SECTION_DATASET = 2
SECTION_PREPARATION = 3
SECTION_ANALYSIS = 4

#: Títulos cortos de los paneles de tamaños; los de CRITERION_LABELS llevan aclaraciones para las
#: tablas.
SIZE_CRITERION_PANEL_TITLES: dict[str, str] = {
    "inference_res": "Resolución de inferencia",
    "original_res": "Resolución original",
}


def plot_class_distribution(stats: DatasetStatistics, output_dir: Path | str) -> dict[str, Path]:
    """Instancias por clase, ordenadas, para mostrar el desequilibrio de SH17."""
    import matplotlib.pyplot as plt

    por_clase = stats.payload["per_class"]
    ordenadas = sorted(por_clase.items(), key=lambda kv: kv[1]["instances"], reverse=True)
    nombres = [k for k, _ in ordenadas]
    valores = [v["instances"] for _, v in ordenadas]

    figura, ejes = plt.subplots(figsize=(7.0, 3.6))
    barras = ejes.bar(nombres, valores, color="#0072B2", edgecolor="black", linewidth=0.4)
    ejes.set_ylabel("Instancias")
    ejes.set_xlabel("Clase")
    ejes.set_title("Distribución de instancias por clase")
    # Escala logarítmica: hay dos órdenes de magnitud entre la clase más frecuente y la más rara.
    ejes.set_yscale("log")
    ejes.yaxis.set_major_formatter(comma_axis_formatter())
    ejes.tick_params(axis="x", rotation=60)
    for etiqueta in ejes.get_xticklabels():
        etiqueta.set_ha("right")
    for barra, valor in zip(barras, valores):
        ejes.text(
            barra.get_x() + barra.get_width() / 2,
            valor,
            format_integer(valor),
            ha="center",
            va="bottom",
            fontsize=6,
        )
    figura.tight_layout()
    return save_figure(figura, output_dir, EDA_CHAPTER, SECTION_ANALYSIS, "distribucion_de_clases")


def plot_size_distribution_both_criteria(
    stats: DatasetStatistics, output_dir: Path | str
) -> dict[str, Path]:
    """Reparto de tamaños con los dos criterios, en dos paneles con el mismo eje vertical."""
    import matplotlib.pyplot as plt

    reparto = stats.payload["size_distribution"]
    primario = stats.payload["size_criteria"]["primary"]
    resolucion = stats.inference_resolution

    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    for panel, criterio in zip(ejes, SIZE_CRITERIA):
        fracciones = [reparto[criterio]["fractions"][r] * 100 for r in SIZE_RANGES]
        conteos = [reparto[criterio]["counts"][r] for r in SIZE_RANGES]
        colores = [SIZE_STYLES[r].color for r in SIZE_RANGES]
        etiquetas = [SIZE_STYLES[r].label for r in SIZE_RANGES]

        contenedor = panel.bar(
            etiquetas, fracciones, color=colores, edgecolor="black", linewidth=0.4
        )
        # Esta figura va sin trama: son tres barras rotuladas con su porcentaje. plot_class_by_size
        # sí la lleva, porque apila los rangos.
        for barra, porcentaje, conteo in zip(contenedor, fracciones, conteos):
            panel.text(
                barra.get_x() + barra.get_width() / 2,
                porcentaje,
                f"{format_number(porcentaje, 1)} %\n({format_integer(conteo)})",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        # Título corto en negrita, sin las aclaraciones de CRITERION_LABELS.
        panel.set_title(SIZE_CRITERION_PANEL_TITLES[criterio], fontweight="bold", fontsize=10)
        panel.set_xlabel("Rango de tamaño")
        panel.set_ylim(0, 100)
        panel.yaxis.set_major_formatter(comma_axis_formatter())

    ejes[0].set_ylabel("Porcentaje de instancias")
    # El criterio primario y su resolución se indican aquí, en dos líneas para no ensanchar la
    # figura.
    figura.suptitle(
        "Reparto de tamaños de objeto según el criterio aplicado\n"
        f"primario: {SIZE_CRITERION_PANEL_TITLES[primario].lower()} "
        f"({resolucion}×{resolucion} px, letterbox)",
        fontsize=10,
    )
    figura.tight_layout()
    return save_figure(
        figura, output_dir, EDA_CHAPTER, SECTION_ANALYSIS, "distribucion_de_tamanos_dos_criterios"
    )


def plot_class_by_size(
    stats: DatasetStatistics, output_dir: Path | str, criterion: str = "inference_res"
) -> dict[str, Path]:
    """Composición por tamaño de cada clase en barras apiladas."""
    import matplotlib.pyplot as plt
    import numpy as np

    if criterion not in SIZE_CRITERIA:
        raise ValueError(f"Criterio desconocido: '{criterion}'.")

    tabla = stats.payload["class_by_size"][criterion]
    ordenadas = sorted(tabla.items(), key=lambda kv: kv[1]["total"], reverse=True)
    nombres = [k for k, _ in ordenadas]

    # Disposición constrained, porque tight_layout no reserva sitio a una leyenda fuera de los ejes.
    figura, ejes = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    acumulado = np.zeros(len(nombres))
    contenedores = []
    for rango in SIZE_RANGES:
        valores = np.array([v[f"{rango}_fraction"] * 100 for _, v in ordenadas])
        contenedor = ejes.bar(
            nombres,
            valores,
            bottom=acumulado,
            color=SIZE_STYLES[rango].color,
            label=SIZE_STYLES[rango].label,
        )
        contenedores.append(contenedor)
        acumulado += valores
    grayscale_safe_bars(ejes, contenedores)

    ejes.set_ylabel("Porcentaje de instancias de la clase")
    ejes.set_xlabel("Clase (ordenada por número de instancias)")
    ejes.set_ylim(0, 100)
    ejes.yaxis.set_major_formatter(comma_axis_formatter())
    ejes.tick_params(axis="x", rotation=60)
    for etiqueta in ejes.get_xticklabels():
        etiqueta.set_ha("right")
    # Leyenda arriba; debajo están los nombres de clase girados.
    ejes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    figura.suptitle(f"Composición por tamaño de cada clase — {CRITERION_LABELS[criterion]}")
    return save_figure(
        figura, output_dir, EDA_CHAPTER, SECTION_ANALYSIS, f"clase_por_tamano_{criterion}"
    )


def plot_cooccurrence(stats: DatasetStatistics, output_dir: Path | str) -> dict[str, Path]:
    """Matriz de coocurrencia entre clases como mapa de calor."""
    import matplotlib.pyplot as plt
    import numpy as np

    matriz_dict = stats.payload["cooccurrence"]
    nombres = list(matriz_dict)
    matriz = np.array([[matriz_dict[a][b] for b in nombres] for a in nombres], dtype=float)

    figura, ejes = plt.subplots(figsize=(6.4, 5.6))
    # Color en escala logarítmica; en lineal solo se vería la diagonal.
    with np.errstate(divide="ignore"):
        mostrada = np.where(matriz > 0, np.log10(matriz), np.nan)
    imagen = ejes.imshow(mostrada, cmap=SEQUENTIAL_COLORMAP, aspect="auto")
    ejes.set_xticks(range(len(nombres)))
    ejes.set_yticks(range(len(nombres)))
    ejes.set_xticklabels(nombres, rotation=60, ha="right")
    ejes.set_yticklabels(nombres)
    ejes.set_title("Coocurrencia de clases en la misma imagen")
    ejes.grid(False)
    barra = figura.colorbar(imagen, ax=ejes)
    barra.set_label("log₁₀(imágenes en común)")
    barra.ax.yaxis.set_major_formatter(comma_axis_formatter())
    figura.tight_layout()
    return save_figure(figura, output_dir, EDA_CHAPTER, SECTION_ANALYSIS, "coocurrencia_de_clases")


def plot_box_area_distribution(stats: DatasetStatistics, output_dir: Path | str) -> dict[str, Path]:
    """Distribución del área de las cajas, absoluta y relativa."""
    import matplotlib.pyplot as plt

    figura, ejes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    series = (
        ("box_area_px_original_res", "Área en resolución original (px²)", ejes[0]),
        ("box_area_px_inference_res", "Área en resolución de inferencia (px²)", ejes[1]),
    )
    for clave, titulo, panel in series:
        resumen = stats.payload.get(clave, {})
        if not resumen:
            continue
        etiquetas = ["mín", "p25", "mediana", "p75", "p95", "máx"]
        valores = [
            resumen["min"],
            resumen["p25"],
            resumen["median"],
            resumen["p75"],
            resumen["p95"],
            resumen["max"],
        ]
        contenedor = panel.bar(etiquetas, valores, color="#0072B2")
        grayscale_safe_bars(panel, [contenedor])
        panel.set_yscale("log")
        panel.yaxis.set_major_formatter(comma_axis_formatter())
        panel.set_title(titulo, fontsize=9)
        panel.set_ylabel("px²")
    # Umbrales COCO sobre la distribución.
    for panel in ejes:
        panel.axhline(32**2, color="#D55E00", linestyle=":", linewidth=1.0)
        panel.axhline(96**2, color="#009E73", linestyle="--", linewidth=1.0)
    ejes[1].text(
        0.02, 32**2, "32²", fontsize=6, va="bottom", transform=ejes[1].get_yaxis_transform()
    )
    ejes[1].text(
        0.02, 96**2, "96²", fontsize=6, va="bottom", transform=ejes[1].get_yaxis_transform()
    )
    figura.suptitle("Distribución del área de las cajas y umbrales COCO")
    figura.tight_layout()
    return save_figure(figura, output_dir, EDA_CHAPTER, SECTION_ANALYSIS, "distribucion_de_areas")


#: Imágenes del mosaico de ejemplos del apartado 3.2. Se eligieron con entre 3 y 6 cajas, al menos 3
#: clases, ninguna caja menor del 0,8 % del encuadre, alguna clase de EPI, rótulos separados y
#: formato 3:2. Cubren glasses, gloves, tool, ear-mufs, hands, person y shoes.
ANNOTATED_EXAMPLE_IMAGES: tuple[str, ...] = (
    "pexels-photo-6124242.jpeg",  # glasses, gloves, tool
    "pexels-photo-4820817.jpeg",  # ear-mufs, hands, person
    "pexels-photo-6835305.jpeg",  # gloves, person, shoes, tool
    "pexels-photo-6004889.jpeg",  # hands, person, shoes, tool
)


def plot_annotated_examples(
    coco: Mapping[str, Any],
    class_names: Sequence[str],
    images_dir: Path | str,
    output_dir: Path | str,
    file_names: Sequence[str] = ANNOTATED_EXAMPLE_IMAGES,
) -> dict[str, Path]:
    """Mosaico de ejemplos anotados con las imágenes de ANNOTATED_EXAMPLE_IMAGES y las cajas del
    COCO.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from PIL import Image

    directorio = Path(images_dir)
    por_imagen: dict[int, list[dict[str, Any]]] = {}
    for anotacion in coco.get("annotations", []):
        por_imagen.setdefault(anotacion["image_id"], []).append(anotacion)

    por_nombre = {img.get("file_name"): img for img in coco.get("images", [])}
    seleccion = []
    for nombre in file_names:
        imagen = por_nombre.get(nombre)
        if imagen is None or not por_imagen.get(imagen["id"]):
            # Si falta una imagen no se sustituye por otra.
            logger.warning(
                "La imagen de ejemplo '%s' no está en el COCO o no tiene "
                "anotaciones: se omite su panel.",
                nombre,
            )
            continue
        seleccion.append(imagen)

    if not seleccion:
        logger.warning("No hay imágenes anotadas para el mosaico de ejemplos.")
        return {}

    columnas = 2
    filas = (len(seleccion) + columnas - 1) // columnas
    figura, ejes = plt.subplots(filas, columnas, figsize=(7.2, 2.7 * filas))
    paneles = list(ejes.flat) if hasattr(ejes, "flat") else [ejes]

    for panel, imagen_info in zip(paneles, seleccion):
        ruta = directorio / imagen_info["file_name"]
        try:
            with Image.open(ruta) as imagen:
                # draft() abarata la decodificación; 1200 px para que no se note el remuestreo con
                # cuatro paneles.
                imagen.draft("RGB", (1200, 1200))
                miniatura = imagen.convert("RGB")
                escala = miniatura.width / imagen_info["width"]
                panel.imshow(miniatura)
        except Exception as error:  # pragma: no cover - imagen ilegible
            logger.warning("No se pudo dibujar %s: %s", ruta.name, error)
            panel.axis("off")
            continue

        for anotacion in por_imagen[imagen_info["id"]]:
            x, y, ancho, alto = (v * escala for v in anotacion["bbox"])
            indice = int(anotacion["category_id"])
            panel.add_patch(
                patches.Rectangle(
                    (x, y), ancho, alto, fill=False, edgecolor="#D55E00", linewidth=0.9
                )
            )
            if indice < len(class_names):
                # Rótulos de 6 pt con fondo blanco translúcido para que se lean sobre la foto.
                panel.text(
                    x,
                    y - 2,
                    class_names[indice],
                    fontsize=6,
                    color="#D55E00",
                    va="bottom",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 0.8},
                )
        panel.set_xticks([])
        panel.set_yticks([])
        panel.grid(False)

    for panel in paneles[len(seleccion) :]:
        panel.axis("off")
    figura.suptitle("Ejemplos anotados del dataset")
    figura.tight_layout()
    return save_figure(figura, output_dir, EDA_CHAPTER, SECTION_DATASET, "ejemplos_anotados")


#: Imagen del ejemplo de redimensionado (apartado 3.3.3): 3962x1625 px, la relación de aspecto más
#: extrema de las candidatas (2,44:1), sin cajas de cara ni cabeza, y con una caja de person que
#: ocupa el 43 % del encuadre.
LETTERBOX_EXAMPLE_IMAGE = "pexels-photo-5973974.jpeg"


def select_letterbox_example(
    coco: Mapping[str, Any], class_names: Sequence[str], file_name: str
) -> tuple[Path | None, list[tuple[float, float, float, float]], str | None]:
    """Busca en el COCO la imagen de ejemplo y todas sus cajas.

    Devuelve ``(nombre_de_fichero, cajas, etiqueta_de_la_mayor)``; el nombre es None si la imagen no
    está en el COCO.
    """
    imagen = next(
        (img for img in coco.get("images", []) if img.get("file_name") == file_name), None
    )
    if imagen is None:
        return None, [], None

    anotaciones = [a for a in coco.get("annotations", []) if a["image_id"] == imagen["id"]]
    if not anotaciones:
        return Path(file_name), [], None

    cajas = [tuple(float(v) for v in a["bbox"]) for a in anotaciones]
    mayor = max(anotaciones, key=lambda a: a["bbox"][2] * a["bbox"][3])
    indice = int(mayor["category_id"])
    etiqueta = class_names[indice] if 0 <= indice < len(class_names) else None
    return Path(file_name), cajas, etiqueta


def plot_split_class_distribution(
    counts: Mapping[str, Mapping[str, int]],
    class_names: Sequence[str],
    output_dir: Path | str,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, Path]:
    """Instancias por clase y partición en barras agrupadas (apartado 3.3.2).

    Muestra que el reparto 70/15/15 se mantiene por clase y que ninguna clase falta en validación o
    prueba. Las clases siguen el orden de sh17_classes.json y la escala es logarítmica.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    posiciones = np.arange(len(class_names))
    anchura = 0.8 / len(splits)

    figura, ejes = plt.subplots(figsize=(7.2, 3.8))
    contenedores = []
    for indice, particion in enumerate(splits):
        estilo = split_style_for(particion)
        valores = [counts[particion][nombre] for nombre in class_names]
        desplazamiento = (indice - (len(splits) - 1) / 2) * anchura
        contenedores.append(
            ejes.bar(
                posiciones + desplazamiento,
                valores,
                anchura,
                label=estilo.label,
                color=estilo.color,
                edgecolor="black",
                linewidth=0.4,
            )
        )
    # Color plano por partición; el filo negro separa barras contiguas.

    ejes.set_yscale("log")
    # El límite inferior del eje es la década por debajo del menor valor.
    minimo = min(counts[p][n] for p in splits for n in class_names)
    ejes.set_ylim(bottom=10 ** math.floor(math.log10(max(minimo, 1))))
    ejes.yaxis.set_major_formatter(comma_axis_formatter())
    ejes.set_xticks(posiciones)
    ejes.set_xticklabels(class_names, rotation=60, ha="right")
    ejes.set_xlabel("Clase")
    ejes.set_ylabel("Instancias anotadas (escala logarítmica)")
    # Hueco entre título y eje para la leyenda.
    ejes.set_title("Instancias anotadas por clase y partición", pad=26)
    ejes.set_xlim(-0.6, len(class_names) - 0.4)
    # Leyenda fuera del área de trazado para no tapar barras.
    ejes.legend(ncol=len(splits), loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0)
    figura.tight_layout()
    return save_figure(
        figura, output_dir, EDA_CHAPTER, SECTION_PREPARATION, "distribucion_clases_particion"
    )


def plot_letterbox_vs_stretch(
    image_path: Path | str,
    output_dir: Path | str,
    boxes: Sequence[Sequence[float]] | None = None,
    box_label: str | None = None,
    resolution: int = 640,
) -> dict[str, Path]:
    """Compara el letterbox de Ultralytics con un estirado simple (apartado 3.3.3).

    Tres paneles: original, letterbox real (detector.letterbox_image) y estirado. Se dibujan todas
    las cajas de la imagen transformadas en cada panel; la mayor va más gruesa y es la única
    rotulada.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # Import diferido porque detector importa torch.
    from tfm_edge.models.detector import letterbox_image

    ruta = Path(image_path)
    with Image.open(ruta) as imagen:
        original = np.asarray(imagen.convert("RGB"))
        estirada = np.asarray(
            imagen.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        )
    alto, ancho = original.shape[:2]
    lienzo, transformacion = letterbox_image(original, resolution=resolution)

    def _aspecto(w: float, h: float) -> str:
        """Formatea una relación de aspecto como ``a:1``."""
        return f"{format_number(w / h)}:1" if h else "—"

    cajas = [tuple(float(v) for v in caja) for caja in (boxes or ())]
    # Se rotula la caja mayor; las demás van sin rótulo.
    indice_mayor = max(range(len(cajas)), key=lambda i: cajas[i][2] * cajas[i][3]) if cajas else -1

    def _en_letterbox(caja: tuple[float, ...]) -> tuple[float, ...]:
        """Lleva una caja al lienzo del letterbox."""
        return transformacion.map_box(*caja)

    def _en_estirado(caja: tuple[float, ...]) -> tuple[float, ...]:
        """Lleva una caja al lienzo estirado con un factor por eje."""
        x, y, w, h = caja
        return (
            x * resolution / ancho,
            y * resolution / alto,
            w * resolution / ancho,
            h * resolution / alto,
        )

    paneles = [
        (original, "Original", f"{ancho}×{alto} px", list(cajas), True),
        (
            lienzo,
            "Letterbox de Ultralytics",
            f"{resolution}×{resolution} px",
            [_en_letterbox(c) for c in cajas],
            False,
        ),
        (
            estirada,
            "Estirado simple",
            f"{resolution}×{resolution} px",
            [_en_estirado(c) for c in cajas],
            False,
        ),
    ]

    figura, ejes = plt.subplots(1, 3, figsize=(7.2, 3.1))
    for panel, (imagen_panel, nombre, detalle, cajas_panel, es_original) in zip(ejes, paneles):
        panel.imshow(imagen_panel)
        for indice, (bx, by, bw, bh) in enumerate(cajas_panel):
            destacada = indice == indice_mayor
            panel.add_patch(
                patches.Rectangle(
                    (bx, by),
                    bw,
                    bh,
                    fill=False,
                    edgecolor="#D55E00",
                    linewidth=1.3 if destacada else 0.8,
                )
            )
            if destacada:
                etiqueta = f"{box_label} · " if box_label else ""
                detalle = f"{detalle}\n{etiqueta}caja {_aspecto(bw, bh)}"

        if es_original:
            # El original es apaisado: se le da al eje un recorrido vertical igual al horizontal
            # para que los tres títulos queden a la misma altura.
            centro = alto / 2.0
            panel.set_ylim(centro + ancho / 2.0, centro - ancho / 2.0)
            for lado in panel.spines.values():
                lado.set_visible(False)
            # Sin marco, la extensión de la imagen se marca con su propio rectángulo.
            panel.add_patch(
                patches.Rectangle(
                    (-0.5, -0.5),
                    ancho,
                    alto,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=5,
                )
            )
        else:
            # Marco en los paneles cuadrados para que se vean las bandas grises del letterbox.
            for lado in panel.spines.values():
                lado.set_visible(True)
                lado.set_linewidth(0.6)

        # Nombre del panel en negrita y detalle debajo, más pequeño.
        panel.set_title(nombre, fontweight="bold", fontsize=10, pad=26)
        panel.text(
            0.5,
            1.015,
            detalle,
            transform=panel.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            linespacing=1.35,
        )
        panel.set_xticks([])
        panel.set_yticks([])
        panel.grid(False)
        panel.set_anchor("N")
    figura.tight_layout()
    return save_figure(
        figura, output_dir, EDA_CHAPTER, SECTION_PREPARATION, "letterbox_vs_estirado"
    )


def plot_conditional_cooccurrence(
    cooccurrence: Mapping[str, Any], output_dir: Path | str
) -> dict[str, Path]:
    """Coocurrencia condicional dirigida entre clases.

    La celda (fila A, columna B) es el porcentaje de instancias de A en imágenes con alguna B; no es
    simétrica. La diagonal (100 % por construcción) va en gris y las clases sin instancias llevan un
    guion.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    nombres = list(cooccurrence["class_names"])
    matriz_dict = cooccurrence["matrix"]
    matriz = np.array(
        [
            [
                np.nan if matriz_dict[fila][columna] is None else matriz_dict[fila][columna] * 100
                for columna in nombres
            ]
            for fila in nombres
        ],
        dtype=float,
    )

    # La diagonal se enmascara solo al dibujar; los datos la conservan.
    dibujada = np.ma.masked_invalid(matriz)
    np.fill_diagonal(dibujada.mask, True)

    paleta = plt.get_cmap(SEQUENTIAL_COLORMAP).copy()
    paleta.set_bad("#D9D9D9")

    figura, ejes = plt.subplots(figsize=(7.2, 6.4))
    imagen = ejes.imshow(dibujada, cmap=paleta, vmin=0.0, vmax=100.0, aspect="auto")
    ejes.set_xticks(range(len(nombres)))
    ejes.set_yticks(range(len(nombres)))
    ejes.set_xticklabels(nombres, rotation=60, ha="right")
    ejes.set_yticklabels(nombres)
    ejes.set_xlabel("Clase B — presente en la imagen")
    ejes.set_ylabel("Clase A — de la que se cuentan las instancias")
    ejes.set_title(
        "Coocurrencia condicional entre clases\n"
        "porcentaje de instancias de A que comparten imagen con alguna B\n"
        "la diagonal vale 100 % por construcción y va en gris, fuera de la escala"
    )
    ejes.grid(False)

    for i in range(len(nombres)):
        for j in range(len(nombres)):
            valor = matriz[i, j]
            if i == j:
                # Diagonal en gris y sin rótulo.
                continue
            if not np.isfinite(valor):
                # Sin instancias en la fila no hay fracción; se marca con un guion.
                ejes.text(j, i, "—", ha="center", va="center", fontsize=4.5, color="#555555")
                continue
            # Texto claro sobre el extremo oscuro de 'viridis' y oscuro sobre el claro.
            ejes.text(
                j,
                i,
                format_number(valor, 0),
                ha="center",
                va="center",
                fontsize=4.5,
                color="white" if valor < 55.0 else "black",
            )

    barra = figura.colorbar(imagen, ax=ejes)
    barra.set_label("% de las instancias de A")
    barra.ax.yaxis.set_major_formatter(comma_axis_formatter())
    figura.tight_layout()
    return save_figure(
        figura, output_dir, EDA_CHAPTER, SECTION_ANALYSIS, "coocurrencia_condicional"
    )


def generate_all_figures(
    stats: DatasetStatistics,
    coco: Mapping[str, Any],
    class_names: Sequence[str],
    images_dir: Path | str,
    output_dir: Path | str,
    include_examples: bool = True,
    split_counts: Mapping[str, Mapping[str, int]] | None = None,
    conditional_cooccurrence: Mapping[str, Any] | None = None,
    include_letterbox: bool = True,
) -> dict[str, dict[str, Path]]:
    """Genera todas las figuras del capítulo 3 que salen del dataset.

    Sin ``split_counts`` no se dibuja la figura del particionado. ``conditional_cooccurrence`` se
    puede pasar ya calculada para dibujar los mismos números que se escriben a disco.
    """
    apply_style()
    ensure_dir(output_dir)

    generadas: dict[str, dict[str, Path]] = {
        "distribucion_de_clases": plot_class_distribution(stats, output_dir),
        "distribucion_de_tamanos_dos_criterios": plot_size_distribution_both_criteria(
            stats, output_dir
        ),
        "distribucion_de_areas": plot_box_area_distribution(stats, output_dir),
        "coocurrencia_de_clases": plot_cooccurrence(stats, output_dir),
        "coocurrencia_condicional": plot_conditional_cooccurrence(
            (
                conditional_cooccurrence
                if conditional_cooccurrence is not None
                else compute_conditional_cooccurrence(coco, class_names)
            ),
            output_dir,
        ),
    }
    for criterio in SIZE_CRITERIA:
        generadas[f"clase_por_tamano_{criterio}"] = plot_class_by_size(
            stats, output_dir, criterion=criterio
        )

    if split_counts is not None:
        generadas["distribucion_clases_particion"] = plot_split_class_distribution(
            split_counts, class_names, output_dir
        )
    else:
        logger.warning(
            "Sin recuentos por partición: se omite fig_%d_%d_distribucion_clases_particion.",
            EDA_CHAPTER,
            SECTION_PREPARATION,
        )

    if include_examples:
        ejemplos = plot_annotated_examples(coco, class_names, images_dir, output_dir)
        if ejemplos:
            generadas["ejemplos_anotados"] = ejemplos

    if include_letterbox:
        nombre, cajas_ejemplo, etiqueta = select_letterbox_example(
            coco, class_names, LETTERBOX_EXAMPLE_IMAGE
        )
        if nombre is None:
            logger.warning(
                "La imagen de ejemplo del redimensionado (%s) no está en el COCO: "
                "se omite fig_%d_%d_letterbox_vs_estirado.",
                LETTERBOX_EXAMPLE_IMAGE,
                EDA_CHAPTER,
                SECTION_PREPARATION,
            )
        else:
            generadas["letterbox_vs_estirado"] = plot_letterbox_vs_stretch(
                Path(images_dir) / nombre,
                output_dir,
                boxes=cajas_ejemplo,
                box_label=etiqueta,
                resolution=stats.inference_resolution,
            )
    return generadas
