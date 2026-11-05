"""Figuras de resultados del capítulo 5, la de pérdida por compresión (6.2) y la de validación
frente a prueba (6.7).

Curvas de entrenamiento (5.1), con las que faltan indicadas a partir del inventario de models/. mAP
frente al nivel de compresión (5.3), sin barras de error porque cada configuración se midió una vez;
en su lugar se dibuja el suelo de medida en cada punto. Comparativas por técnica (5.4). Degradación
por clase y por tamaño (5.5), con las compresiones y el orden de la figura del 6.2. Descomposición
de tipos de error (5.5), a partir de las celdas error_* del registro y del volcado del paso 34.
Coste frente a ganancia (5.6), solo con las configuraciones con coste registrado. Las del 5.5 no
llevan pie dentro de la imagen; el texto se calcula de los datos y queda en ``caption`` del
manifiesto.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.analysis.degradation import STAGE_FINETUNED, training_stage
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.registry import TABLE_PRIMARY_KEYS
from tfm_edge.reporting import style

logger = get_logger(__name__)

#: Capítulo y apartado de cada figura. El nombre de fichero indica dónde va la figura, no qué paso
#: la genera.
PLACEMENT: dict[str, tuple[int, int]] = {
    "curvas_entrenamiento": (5, 1),
    "map_vs_nivel_por_familia": (5, 3),
    "comparativa_por_tecnica": (5, 4),
    "degradacion_por_clase_onnxruntime": (5, 5),
    "degradacion_por_clase_openvino": (5, 5),
    "degradacion_por_clase_pytorch": (5, 5),
    "degradacion_por_tamano": (5, 5),
    "tipos_de_error": (5, 5),
    "coste_frente_a_ganancia": (5, 6),
    "perdida_map_por_compresion": (6, 2),
    "validacion_frente_a_prueba": (6, 7),
}

CURVE_FILENAME = "results.csv"

CURVE_COLUMNS: dict[str, str] = {
    "train/box_loss": "pérdida de caja",
    "train/cls_loss": "pérdida de clase",
    "train/dfl_loss": "pérdida DFL",
    "metrics/mAP50(B)": "mAP@50",
    "metrics/mAP50-95(B)": "mAP@50-95",
}

LOSS_COLUMNS: tuple[str, ...] = ("train/box_loss", "train/cls_loss", "train/dfl_loss")

#: Estilos de línea para distinguir los miembros de una familia; el color lo fija style.py.
WITHIN_FAMILY_LINESTYLES: tuple[Any, ...] = (
    "-",
    "--",
    ":",
    "-.",
    (0, (3, 1, 1, 1)),
    (0, (5, 1)),
)

#: Tramas para distinguir las familias en barras en escala de grises.
FAMILY_HATCHES: tuple[str, ...] = ("", "///", "...", "xxx", "|||", "+++")

#: Color de la banda del suelo, el mismo que en la figura de la frontera.
FLOOR_COLOR = "#7F0000"

SECONDS_PER_HOUR = 3600.0

#: Estados del factor de aceleración en la figura de coste. Los dos últimos dejan el punto fuera del
#: panel izquierdo por motivos distintos y el pie los cuenta por separado.
SPEEDUP_CERTIFIED = "certificada"
SPEEDUP_NOT_CERTIFIED = "sin_certificar"
SPEEDUP_NO_LATENCY = "sin_latencia"
SPEEDUP_STATUSES: tuple[str, ...] = (SPEEDUP_CERTIFIED, SPEEDUP_NOT_CERTIFIED, SPEEDUP_NO_LATENCY)

SIZE_RANGES: tuple[str, ...] = ("small", "medium", "large")

#: La única clase del reparto que cuenta anotaciones y no cajas.
MISSED_KIND = "missed"

#: Ruta del paso 34 cuyas cifras se contrastaron además con la curva de operación del paso 23; la
#: otra ruta se marca en la figura como más débil.
STRONG_ROUTE = "curva"

WEAK_ROUTE_MARK = "†"

#: Motivo corto de cada razón por la que una fila no tiene reparto. Una razón sin entrada hace
#: fallar la figura.
ERROR_REASON_LABELS: dict[str, str] = {
    "sin_detecciones": (
        "no dejaron detecciones por instancia en disco —la ruta de PyTorch solo "
        "vuelca un resumen— y recuperarlas exige volver a inferir"
    ),
    "otro_evaluador": "son del evaluador de Ultralytics, que no se mezcla con pycocotools",
    "otro_split": "son de otra partición",
}

RUNTIME_MARKERS: tuple[str, ...] = ("o", "s", "^", "D")

#: Por debajo de estas anotaciones en la partición, el AP de una clase no se interpreta y la figura
#: por clase la raya. El recuento sale de los volcados de evaluación.
MIN_INTERPRETABLE_INSTANCES = 150

SCARCE_CLASS_HATCH = "//"

#: Una figura por clase para cada motor y su ubicación; juntas medían unas veinte pulgadas y las
#: cifras salían a 2 pt.
CLASS_FIGURE_PLACEMENT: dict[str, str] = {
    "onnxruntime": "cuerpo",
    "openvino": "cuerpo",
    "pytorch": "anexo",
}

#: Ancho de la figura por clase (16 cm), dibujada a su tamaño final.
CLASS_FIGURE_WIDTH = 16.0 / 2.54

CLASS_CELL_FONTSIZE = 6.0
CLASS_LABEL_FONTSIZE = 6.5

#: Título legible de cada motor en las figuras del 5.5.
RUNTIME_TITLES: dict[str, str] = {
    "onnxruntime": "ONNX Runtime",
    "openvino": "OpenVINO",
    "pytorch": "PyTorch",
}

#: Orden de las familias en la figura de tipos de error, según la técnica de la fila INT8 de cada
#: pareja; el resto va detrás.
ERROR_FAMILY_ORDER: tuple[str, ...] = ("ptq", "qat", "combined")

#: Huecos verticales en filas: entre familias (con rótulo) y entre parejas.
ERROR_FAMILY_GAP = 1.1
ERROR_PAIR_GAP = 0.4

#: Ancho y cuerpos de letra de la figura de tipos de error, dibujada a su tamaño final.
ERROR_FIGURE_WIDTH = 16.0 / 2.54
ERROR_LABEL_FONTSIZE = 7.0
ERROR_TITLE_FONTSIZE = 7.5

#: Alto de fila y alto máximo (17 cm) de la figura de tipos de error. Si no cabe, se reduce la fila,
#: no el texto.
ERROR_ROW_HEIGHT = 0.19
ERROR_MAX_HEIGHT = 17.0 / 2.54

#: Rótulo en la leyenda de la marca de la ruta más débil.
WEAK_ROUTE_LEGEND = "verificación más débil"

#: Sufijo de la fila FP32 de un QAT, el mismo modelo antes de materializar el INT8.
QAT_FP32_SUFFIX = " en FP32"

#: Ancho, alto máximo, separación entre filas y cuerpo de texto de la figura por tamaño.
SIZE_FIGURE_WIDTH = 16.0 / 2.54
SIZE_MAX_HEIGHT = 12.0 / 2.54
SIZE_ROW_PITCH = 0.5 / 2.54
SIZE_FONTSIZE = 7.0
SIZE_TITLE_FONTSIZE = 7.5

#: Marcadores huecos de la figura por tamaño, para que uno no tape a otro cuando coinciden.
SIZE_MARKER_AREA = 13.0
SIZE_MARKER_EDGE = 0.8

#: Criterio de tamaño de la figura por tamaño, el primario. Se comprueba contra la columna de
#: origen.
SIZE_CRITERION = "inference_res"

#: Rótulo de cada fragmento de variante de una destilación, para la figura de PyTorch del anexo.
DISTILLATION_TOKENS: dict[str, str] = {"combined": "Destilación", "control": "Control α=0"}

#: Orden de las técnicas en la figura por clase de PyTorch del anexo.
ANNEX_TECHNIQUE_ORDER: tuple[str, ...] = ("distillation", "pruning", "qat")

#: Marca de la poda medida antes de reajustar.
RAW_PRUNING_SUFFIX = " sin reajuste"

#: Motores de la figura de pérdida por compresión en el orden de sus paneles. PyTorch no entra
#: porque allí no hay artefacto INT8.
COMPRESSION_RUNTIMES: tuple[str, ...] = ("onnxruntime", "openvino")
COMPRESSION_RUNTIME_TITLES: dict[str, str] = {
    "onnxruntime": "ONNX Runtime",
    "openvino": "OpenVINO",
}

#: Evaluador y partición de la figura de pérdida por compresión; se comprueban, no filtran.
COMPRESSION_EVALUATOR = "pycocotools"
COMPRESSION_SPLIT = "val"

#: Categorías de la figura de pérdida por compresión en el orden de la leyenda. Las combinaciones se
#: rotulan por lo que combinan (poda + PTQ, poda + QAT).
COMPRESSION_CATEGORIES: dict[str, tuple[style.FamilyStyle, str]] = {
    "pruning": (style.TECHNIQUE_STYLES["pruning"], style.TECHNIQUE_STYLES["pruning"].label),
    "ptq": (style.TECHNIQUE_STYLES["ptq"], style.TECHNIQUE_STYLES["ptq"].label),
    "qat": (style.TECHNIQUE_STYLES["qat"], style.TECHNIQUE_STYLES["qat"].label),
    "combined": (style.TECHNIQUE_STYLES["combined"], "Poda + PTQ"),
    "pruning_qat": (style.PRUNING_QAT_STYLE, style.PRUNING_QAT_STYLE.label),
}

#: Rótulo de cada método de poda según el primer fragmento de su variante.
PRUNING_METHOD_LABELS: dict[str, str] = {"l1global": "L1", "bnscale": "BN"}

#: Fragmento de variante del control aleatorio de una poda.
PRUNING_CONTROL_TOKEN = "randomctl"

#: Variantes de cuantización que se saben rotular.
QUANTIZATION_VARIANTS: tuple[str, ...] = ("int8-noexcl",)

#: Sufijo de tamaño en las etiquetas; las filas de s van sin sufijo.
COMPRESSION_SIZE_SUFFIX: dict[str, str] = {
    "s": "",
    "n": " nano",
    "m": " medium",
    "l": " large",
}

#: Ancho y cuerpo de texto de la figura de pérdida por compresión.
COMPRESSION_FIGURE_WIDTH = 16.0 / 2.54
COMPRESSION_FONTSIZE = 6.5

#: Alto de fila de la figura de pérdida por compresión, en pulgadas.
COMPRESSION_ROW_HEIGHT = 0.16

#: Separación entre el segundo panel y la columna de puestos, en fracción del ancho del panel.
RANK_COLUMN_OFFSET = 0.03

#: Motores de la figura de validación frente a prueba (6.7). PyTorch no entra.
SPLIT_CHECK_RUNTIMES: tuple[str, ...] = COMPRESSION_RUNTIMES

#: Evaluador y dispositivo de la figura del 6.7; se comprueban contra el registro.
SPLIT_CHECK_EVALUATOR = "pycocotools"
SPLIT_CHECK_DEVICE = "workstation_cpu"

#: Familias de la figura del 6.7 con el color y marcador de su técnica; la precisión la indica el
#: relleno.
SPLIT_CHECK_FAMILIES: dict[str, tuple[style.FamilyStyle, str]] = {
    "reference": (style.TECHNIQUE_STYLES["baseline"], "Referencias"),
    "quantization": (style.TECHNIQUE_STYLES["ptq"], "Cuantización sin podar"),
    "pruning": (style.TECHNIQUE_STYLES["pruning"], "Poda"),
    "pruning_quantization": (style.TECHNIQUE_STYLES["combined"], "Poda + cuantización"),
}

#: Configuraciones rotuladas en cada panel del 6.7. Si una no está en el volcado, la figura falla.
SPLIT_CHECK_LABELS: dict[str, tuple[str, ...]] = {
    "onnxruntime": (
        "n FP32",
        "m FP32",
        "l FP32",
        "L1 r50",
        "iterativa r50 + PTQ",
        "L1 r50 + PTQ",
        "L1 r25 + QAT",
    ),
    "openvino": ("n FP32", "m FP32", "l FP32", "L1 r50"),
}

#: Ancho y cuerpo de letra de la figura del 6.7, los mismos que la del 6.2.
SPLIT_CHECK_FIGURE_WIDTH = COMPRESSION_FIGURE_WIDTH
SPLIT_CHECK_FONTSIZE = COMPRESSION_FONTSIZE

#: Área de los marcadores de la figura del 6.7.
SPLIT_CHECK_MARKER_AREA = 20.0

#: Desplazamientos candidatos de un rótulo respecto a su punto, en puntos, por preferencia.
LABEL_OFFSETS: tuple[tuple[float, float], ...] = (
    (4, 3),
    (4, -3),
    (-4, 3),
    (-4, -3),
    (0, 6),
    (0, -6),
    (6, 0),
    (-6, 0),
    (12, 10),
    (12, -10),
    (-12, 10),
    (-12, -10),
    (0, 16),
    (0, -16),
    (18, 0),
    (-18, 0),
    (18, 18),
    (18, -18),
    (-18, 18),
    (-18, -18),
    (28, 0),
    (-28, 0),
    (0, 26),
    (0, -26),
    (26, 14),
    (26, -14),
    (-26, 14),
    (-26, -14),
)

#: A partir de qué desplazamiento un rótulo lleva línea guía.
LEADER_MIN_OFFSET = 10.0

#: Cuántas veces más lejos que el suyo debe quedar cualquier otro punto de un rótulo sin guía, para
#: que no se lea como de otro punto.
LABEL_AMBIGUITY_RATIO = 1.5


def stage_suffix(row: Mapping[str, Any]) -> str:
    """Marca de etapa para la etiqueta: vacía si está reajustada, que es lo normal.

    Falla si la fila no declara etapa.
    """
    etapa = (row.get("stage", "") or "").strip()
    if not etapa:
        raise ValueError(
            f"La fila de degradación de '{row.get('experiment_id', '?')}' no declara 'stage'. "
            "Vuelve a ejecutar 29_analyze_results.py antes de dibujar."
        )
    return "" if etapa == STAGE_FINETUNED else f" {etapa}"


def row_label(row: Mapping[str, Any]) -> str:
    """Etiqueta única de una fila de degradación, con motor, precisión y etapa."""
    return f"{row['experiment_id']}{stage_suffix(row)} [{row['runtime']}/{row['precision']}]"


@dataclass(frozen=True)
class TrainingCurve:
    """Curva por época de una ejecución de entrenamiento."""

    experiment_id: str
    label: str
    technique: str
    source: Path
    epochs: tuple[int, ...]
    series: Mapping[str, tuple[float, ...]]

    @property
    def total_loss(self) -> tuple[float, ...]:
        """Suma de las tres pérdidas por época, o una tupla vacía si falta alguna."""
        if not all(columna in self.series for columna in LOSS_COLUMNS):
            return ()
        return tuple(sum(valores) for valores in zip(*(self.series[c] for c in LOSS_COLUMNS)))


@dataclass(frozen=True)
class CurveInventory:
    """Curvas encontradas y experimentos que deberían tenerla y no la tienen."""

    curves: tuple[TrainingCurve, ...]
    missing: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serializa el inventario."""
        return {
            "found": [
                {
                    "experiment_id": c.experiment_id,
                    "label": c.label,
                    "epochs": len(c.epochs),
                    "source": c.source.name,
                }
                for c in self.curves
            ],
            "missing": [{"experiment_id": e, "reason": motivo} for e, motivo in self.missing],
        }


def _read_curve(path: Path) -> tuple[tuple[int, ...], dict[str, tuple[float, ...]]]:
    """Lee un results.csv de Ultralytics y devuelve épocas y series. Falla si no hay columna de
    época.
    """
    with path.open(encoding="utf-8") as fichero:
        filas = [
            {clave.strip(): valor for clave, valor in fila.items() if clave is not None}
            for fila in csv.DictReader(fichero)
        ]
    if not filas or "epoch" not in filas[0]:
        raise ValueError(
            f"{path} no tiene columna 'epoch'. Sin ella no se sabe en qué orden van "
            f"los puntos de la curva."
        )

    epocas = tuple(int(float(fila["epoch"])) for fila in filas)
    series: dict[str, tuple[float, ...]] = {}
    for columna in CURVE_COLUMNS:
        if columna not in filas[0]:
            continue
        valores = []
        completa = True
        for fila in filas:
            bruto = (fila.get(columna, "") or "").strip()
            if not bruto:
                completa = False
                break
            valores.append(float(bruto))
        if completa:
            series[columna] = tuple(valores)
    return epocas, series


def training_curve_inventory(
    models_dir: Path | str,
    experiment_rows: Sequence[Mapping[str, str]],
) -> CurveInventory:
    """Busca las curvas por época y lista las que faltan.

    Se espera curva de todo experimento con training_epochs mayor que cero.
    """
    raiz = Path(models_dir)
    encontradas: list[TrainingCurve] = []
    con_curva: set[str] = set()

    por_identificador = {fila["experiment_id"]: fila for fila in experiment_rows}
    # Los directorios de Ultralytics no llevan el hash, así que se ata por prefijo.
    por_prefijo: dict[str, str] = {}
    for identificador, fila in por_identificador.items():
        prefijo = identificador.rsplit("_", 1)[0]
        por_prefijo.setdefault(prefijo, identificador)

    for ruta in sorted(raiz.rglob(CURVE_FILENAME)):
        relativa = ruta.relative_to(raiz)
        nombre_directorio = relativa.parts[1] if len(relativa.parts) > 1 else relativa.parts[0]
        if nombre_directorio in por_identificador:
            identificador = nombre_directorio
        else:
            identificador = por_prefijo.get(nombre_directorio, "")
        try:
            epocas, series = _read_curve(ruta)
        except ValueError as error:
            logger.warning("Curva ilegible, fuera de la figura: %s", error)
            continue
        if not series:
            logger.warning("La curva %s no trae ninguna columna útil.", ruta)
            continue

        fila = por_identificador.get(identificador, {})
        tecnica = (fila.get("technique", "") or "").strip() or "baseline"
        etiqueta = identificador or nombre_directorio
        # Un experimento puede tener varias curvas (ciclos de la poda iterativa); se distinguen por
        # la ruta.
        tramo = "/".join(relativa.parts[2:-1])
        if tramo and tramo != "train":
            etiqueta = f"{etiqueta} ({tramo})"

        encontradas.append(
            TrainingCurve(
                experiment_id=identificador,
                label=etiqueta,
                technique=tecnica,
                source=ruta,
                epochs=epocas,
                series=series,
            )
        )
        if identificador:
            con_curva.add(identificador)

    faltan: list[tuple[str, str]] = []
    for identificador, fila in sorted(por_identificador.items()):
        bruto = (fila.get("training_epochs", "") or "").strip()
        if not bruto or float(bruto) <= 0 or identificador in con_curva:
            continue
        faltan.append(
            (
                identificador,
                f"Declara {int(float(bruto))} épocas en el registro y no hay "
                f"{CURVE_FILENAME} en models/: de esa ejecución volvieron los pesos "
                f"pero no la curva.",
            )
        )

    logger.info(
        "Curvas de entrenamiento: %d encontradas, %d experimentos entrenados sin curva.",
        len(encontradas),
        len(faltan),
    )
    return CurveInventory(curves=tuple(encontradas), missing=tuple(faltan))


def _style_cycle(curves: Sequence[TrainingCurve]) -> dict[str, Any]:
    """Trazo distinto para cada curva dentro de su familia."""
    contador: dict[str, int] = {}
    asignado: dict[str, Any] = {}
    for curva in curves:
        indice = contador.get(curva.technique, 0)
        asignado[curva.label] = WITHIN_FAMILY_LINESTYLES[indice % len(WITHIN_FAMILY_LINESTYLES)]
        contador[curva.technique] = indice + 1
    return asignado


def _footer(figure: Any, texto: str) -> None:
    """Escribe el pie de una figura."""
    figure.text(
        0.01,
        0.005,
        texto,
        fontsize=6.5,
        color="#333333",
        va="bottom",
        ha="left",
        wrap=True,
    )


def figure_training_curves(inventory: CurveInventory) -> tuple[Any, dict[str, Any]]:
    """Curvas de pérdida y de mAP por época."""
    import matplotlib.pyplot as plt

    if not inventory.curves:
        raise ValueError(
            "No hay ninguna curva de entrenamiento en models/."
        )

    style.apply_style()
    # La leyenda va en una franja bajo los paneles para no tapar las curvas.
    columnas_leyenda = 1 if len(inventory.curves) <= 5 else 2
    filas_leyenda = math.ceil(len(inventory.curves) / columnas_leyenda)
    # El margen fijo es necesario: tight_layout no ve una leyenda anclada fuera de los ejes.
    alto_franja = 0.55 + 0.16 * filas_leyenda
    figura = plt.figure(figsize=(9.6, 4.2 + alto_franja))
    rejilla = figura.add_gridspec(2, 2, height_ratios=[4.2, alto_franja], hspace=0.35)
    ejes = [figura.add_subplot(rejilla[0, 0]), figura.add_subplot(rejilla[0, 1])]
    franja = figura.add_subplot(rejilla[1, :])
    franja.axis("off")
    trazos = _style_cycle(inventory.curves)
    dibujadas: list[str] = []

    for curva in inventory.curves:
        estilo = style.style_for(curva.technique)
        perdida = curva.total_loss
        if perdida:
            ejes[0].plot(
                curva.epochs,
                perdida,
                color=estilo.color,
                linestyle=trazos[curva.label],
                linewidth=1.2,
                label=curva.label,
            )
        calidad = curva.series.get("metrics/mAP50-95(B)")
        if calidad:
            ejes[1].plot(
                curva.epochs,
                calidad,
                color=estilo.color,
                linestyle=trazos[curva.label],
                linewidth=1.2,
                label=curva.label,
            )
        dibujadas.append(curva.label)

    ejes[0].set_title("Pérdida de entrenamiento (caja + clase + DFL)")
    ejes[0].set_xlabel("época")
    ejes[0].set_ylabel("pérdida")
    ejes[1].set_title("mAP@50-95 en validación")
    ejes[1].set_xlabel("época")
    ejes[1].set_ylabel("mAP@50-95")
    for eje in ejes:
        eje.yaxis.set_major_formatter(style.comma_axis_formatter())

    apoderados, rotulos = ejes[1].get_legend_handles_labels()
    if not apoderados:
        # Si el panel de calidad no tiene curvas, la leyenda se toma del de pérdida.
        apoderados, rotulos = ejes[0].get_legend_handles_labels()
    franja.legend(
        apoderados,
        rotulos,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=columnas_leyenda,
        fontsize=6.0,
        frameon=False,
        handlelength=2.4,
        handletextpad=0.5,
        columnspacing=1.6,
    )

    pie = (
        f"Curvas por época de las {len(inventory.curves)} ejecuciones que dejaron "
        f"{CURVE_FILENAME} en models/. Las tres pérdidas se suman; una curva a la que "
        f"le falte alguna no se dibuja, porque su total no sería comparable."
    )
    if inventory.missing:
        faltan = ", ".join(identificador for identificador, _ in inventory.missing)
        pie += (
            f"\nNO están las de: {faltan}. De esas ejecuciones volvieron los pesos y el "
            f"coste, pero no la curva, así que este panel no cubre toda la campaña."
        )
    _footer(figura, pie)
    # El pie se calcula en líneas y se convierte a fracción de la figura.
    alto_pie = (0.42 if inventory.missing else 0.24) / figura.get_figheight()
    figura.tight_layout(rect=(0, alto_pie, 1, 1))

    return figura, {
        "curves_drawn": dibujadas,
        "legend_in_band": True,
        "missing": [identificador for identificador, _ in inventory.missing],
    }


def _floor_margins(valores: Sequence[float], suelo: Mapping[str, Any] | None) -> list[float] | None:
    """Margen del suelo de medida en cada punto, o None si el eje no tiene suelo."""
    if suelo is None or suelo.get("cross_session") is None:
        return None
    magnitud = float(suelo["cross_session"])
    if suelo.get("kind") == "relativo":
        return [magnitud * abs(v) for v in valores]
    return [magnitud for _ in valores]


def figure_quality_vs_level(
    curves: Sequence[Mapping[str, Any]],
    floors: Mapping[str, Any],
    quality_column: str = "map_50_95",
) -> tuple[Any, dict[str, Any]]:
    """mAP frente al nivel de compresión, una línea por método.

    Falla si ninguna curva tiene más de un nivel.
    """
    import matplotlib.pyplot as plt

    dibujables = [c for c in curves if c["num_levels"] >= 2]
    if not dibujables:
        raise ValueError(
            "Ninguna curva tiene más de un nivel de compresión: no hay ninguna "
            "tendencia que dibujar, solo configuraciones sueltas."
        )

    motores = sorted({c["runtime"] for c in dibujables})
    style.apply_style()
    figura, ejes = plt.subplots(
        1, len(motores), figsize=(3.7 * len(motores) + 1.0, 4.8), squeeze=False
    )

    suelo = floors.get("floors", {}).get(quality_column)
    resumen: dict[str, Any] = {"panels": {}, "floor_drawn": suelo is not None}
    #: Etiqueta, estilo de familia y trazo, para una sola leyenda.
    entradas: dict[str, tuple[Any, Any]] = {}

    # El color lo fija la familia y el trazo distingue los métodos dentro de ella. El tamaño del
    # modelo forma parte de la clave para no juntar la misma técnica sobre tamaños distintos.
    trazos: dict[tuple[str, ...], Any] = {}
    contador: dict[str, int] = {}
    for curva in sorted(
        dibujables, key=lambda c: (c["technique"], c["model_size"], c["method"], c["stage"])
    ):
        clave = (curva["technique"], curva["model_size"], curva["method"], curva["stage"])
        if clave not in trazos:
            indice_familia = contador.get(curva["technique"], 0)
            trazos[clave] = WITHIN_FAMILY_LINESTYLES[indice_familia % len(WITHIN_FAMILY_LINESTYLES)]
            contador[curva["technique"]] = indice_familia + 1

    for indice, motor in enumerate(motores):
        eje = ejes[0][indice]
        delmotor = [c for c in dibujables if c["runtime"] == motor]
        for curva in delmotor:
            puntos = [p for p in curva["points"] if quality_column in p["values"]]
            if len(puntos) < 2:
                continue
            estilo = style.style_for(curva["technique"])
            clave_trazo = (
                curva["technique"],
                curva["model_size"],
                curva["method"],
                curva["stage"],
            )
            valores = [p["values"][quality_column] for p in puntos]
            etapa = "" if curva["stage"] == "reajustado" else f" {curva['stage']}"
            etiqueta = (
                f"{curva['method']}{etapa} " f"(yolov8{curva['model_size']}, {curva['precision']})"
            )
            # Dos curvas distintas con el mismo rótulo serían indistinguibles; se falla.
            anterior = entradas.get(etiqueta)
            if anterior is not None and anterior != (estilo, trazos[clave_trazo]):
                raise ValueError(
                    f"Dos curvas distintas comparten el rótulo '{etiqueta}' con estilos distintos."
                )
            entradas.setdefault(etiqueta, (estilo, trazos[clave_trazo]))
            eje.errorbar(
                [p["level"] for p in puntos],
                valores,
                yerr=_floor_margins(valores, suelo),
                color=estilo.color,
                linestyle=trazos[clave_trazo],
                marker=estilo.marker,
                markersize=4,
                linewidth=1.3,
                elinewidth=1.0,
                ecolor=FLOOR_COLOR,
                capsize=2.5,
            )

        eje.set_title(motor)
        eje.set_xlabel("ratio de compresión declarado")
        if indice == 0:
            eje.set_ylabel("mAP@50-95" if quality_column == "map_50_95" else quality_column)
        eje.yaxis.set_major_formatter(style.comma_axis_formatter())
        eje.xaxis.set_major_formatter(style.comma_axis_formatter())
        resumen["panels"][motor] = [
            f"{c['model_size']}/{c['method']}/{c['stage']}" for c in delmotor
        ]

    # Una sola leyenda para los tres paneles.
    manejadores = [
        plt.Line2D(
            [],
            [],
            color=estilo.color,
            linestyle=trazo,
            marker=estilo.marker,
            markersize=4,
            linewidth=1.3,
        )
        for estilo, trazo in entradas.values()
    ]
    manejadores.append(
        plt.Line2D([], [], color=FLOOR_COLOR, linestyle="-", linewidth=1.0, marker="_")
    )
    figura.legend(
        manejadores,
        [*entradas, "resolución de la medida (no dispersión)"],
        loc="lower center",
        ncol=min(3, len(manejadores)),
        fontsize=6.5,
        frameon=False,
        handlelength=3.0,
        bbox_to_anchor=(0.5, 0.20),
    )

    formas: dict[str, int] = {}
    for curva in curves:
        formas[curva["shape"]] = formas.get(curva["shape"], 0) + 1
    pie = (
        "La especificación pide barras de error. Este estudio entrenó y evaluó cada "
        "configuración una sola vez (D-110), así que no hay dispersión que dibujar: "
        "unas barras de error sobre una única observación serían una invención con "
        "aspecto de rigor. Las barras rojas son otra cosa y lo dicen: la resolución "
        "medida del eje (D-127), por debajo de la cual dos valores son la misma medida."
        f"\nSolo se dibujan los métodos con dos niveles o más. De las {len(curves)} "
        f"curvas del análisis, {formas.get('punto', 0)} son un único punto y no aparecen. "
        f"Las marcadas «sin_reajuste» son la compresión en crudo, antes de reentrenar: "
        f"son otro procedimiento, no un punto peor del mismo."
    )
    _footer(figura, pie)
    figura.tight_layout(rect=(0, 0.34, 1, 1))
    return figura, resumen


def figure_technique_comparison(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[tuple[str, str, str]],
) -> tuple[Any, dict[str, Any]]:
    """Barras por configuración, agrupadas y coloreadas por familia de técnica, sin promediar."""
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No hay ninguna configuración que comparar.")

    ordenadas = sorted(rows, key=lambda f: (f["technique"], f["experiment_id"], f["runtime"]))
    etiquetas = [f"{f['experiment_id']} [{f['runtime']}/{f['precision']}]" for f in ordenadas]

    style.apply_style()
    altura = max(4.0, 0.22 * len(ordenadas) + 1.6)
    figura, ejes = plt.subplots(
        1, len(metrics), figsize=(3.6 * len(metrics), altura), squeeze=False, sharey=True
    )

    familias = sorted({f["technique"] for f in ordenadas})
    tramas = {
        familia: FAMILY_HATCHES[indice % len(FAMILY_HATCHES)]
        for indice, familia in enumerate(familias)
    }

    posiciones = list(range(len(ordenadas)))
    for indice, (columna, titulo, unidad) in enumerate(metrics):
        eje = ejes[0][indice]
        for posicion, fila in zip(posiciones, ordenadas):
            valor = fila.get(columna)
            if valor is None:
                continue
            eje.barh(
                posicion,
                valor,
                color=style.style_for(fila["technique"]).color,
                hatch=tramas[fila["technique"]],
                edgecolor="black",
                linewidth=0.3,
                height=0.72,
            )
        eje.set_title(f"{titulo} ({unidad})" if unidad else titulo)
        eje.xaxis.set_major_formatter(style.comma_axis_formatter())
        eje.invert_yaxis()

    ejes[0][0].set_yticks(posiciones)
    ejes[0][0].set_yticklabels(etiquetas, fontsize=5.5)

    manejadores = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=style.style_for(f).color,
            hatch=tramas[f],
            edgecolor="black",
            linewidth=0.3,
        )
        for f in familias
    ]
    figura.legend(
        manejadores,
        [style.style_for(f).label for f in familias],
        loc="lower center",
        ncol=min(len(familias), 6),
        fontsize=7,
        bbox_to_anchor=(0.5, 0.02),
    )
    _footer(
        figura,
        "Una barra por configuración medida, sin promediar dentro de cada familia: con "
        "dos o tres miembros, una media esconde el rango entero y presenta como "
        "resultado un número que nadie ha medido. Una configuración sin valor en una "
        "métrica no dibuja barra, en lugar de dibujar un cero. Cada familia lleva "
        "además su trama, para que la figura se lea impresa en blanco y negro.",
    )
    figura.tight_layout(rect=(0, 0.09, 1, 1))
    return figura, {"configurations": etiquetas, "metrics": [m[0] for m in metrics]}


@dataclass(frozen=True)
class ClassInstances:
    """Anotaciones por clase en la partición evaluada."""

    split: str
    by_index: Mapping[int, tuple[str, int]]
    sources: tuple[str, ...]


def class_instances(eval_dir: Path | str, split: str = "val") -> ClassInstances | None:
    """Número de anotaciones por clase según los volcados de evaluación de results/raw/coco_eval.

    Se usan los volcados porque están versionados, a diferencia del COCO. Todos los de la partición
    deben coincidir; los de humo se excluyen. Devuelve None si no hay ninguno.
    """
    import json

    vistos: dict[tuple[tuple[int, str, int], ...], list[str]] = {}
    for ruta in sorted(Path(eval_dir).glob("*.json")):
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        cuerpo = datos.get("coco", datos) if isinstance(datos, dict) else {}
        if not isinstance(cuerpo, dict) or cuerpo.get("split") != split:
            continue
        if datos.get("smoke_test") is True or ruta.stem.endswith("_smoke"):
            continue
        por_clase = cuerpo.get("per_class")
        if not isinstance(por_clase, dict) or not por_clase:
            continue
        firma = tuple(
            sorted(
                (int(indice), str(v["name"]), int(v["num_annotations"]))
                for indice, v in por_clase.items()
            )
        )
        vistos.setdefault(firma, []).append(ruta.name)
    if not vistos:
        return None
    if len(vistos) > 1:
        ejemplos = [fuentes[0] for fuentes in vistos.values()]
        raise ValueError(
            f"Los volcados de evaluación de '{split}' no dan el mismo número de anotaciones por "
            f"clase ({', '.join(ejemplos)})."
        )
    firma, fuentes = next(iter(vistos.items()))
    return ClassInstances(
        split=split,
        by_index={indice: (nombre, n) for indice, nombre, n in firma},
        sources=tuple(fuentes),
    )


def comparison_devices(
    rows: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, str]],
) -> dict[str, tuple[str, str]]:
    """Dispositivo de cada variante y de su referencia.

    Se reconstruye con degradation._pick_reference y se comprueba que el mAP de la referencia es el
    que usó el volcado del paso 29.
    """
    from tfm_edge.analysis.degradation import _pick_reference

    resultado: dict[str, tuple[str, str]] = {}
    for fila in rows:
        if not fila["per_class"]:
            continue
        clave = (fila["experiment_id"], fila["runtime"], fila["precision"])
        variantes = [
            q for q in quality_rows if (q["experiment_id"], q["runtime"], q["precision"]) == clave
        ]
        if len(variantes) != 1:
            raise ValueError(
                f"La variante {clave} tiene {len(variantes)} filas de calidad: no se sabe "
                f"en qué dispositivo se midió la que compara el paso 29."
            )
        variante = variantes[0]
        evaluador = (variante.get("evaluator", "") or "").strip()
        candidatas = [
            q
            for q in quality_rows
            if q["experiment_id"] == fila["reference"]["experiment_id"]
            and q["runtime"] == fila["runtime"]
            and (q.get("evaluator", "") or "").strip() == evaluador
        ]
        referencia = _pick_reference(
            candidatas, variante.get("device_id", ""), ("map_50", "map_50_95")
        )
        usada = fila["overall"].get("map_50_95", {}).get("reference")
        if (
            referencia is None
            or usada is None
            or abs(float(referencia["map_50_95"]) - usada) > 1e-9
        ):
            raise ValueError(
                f"La referencia reconstruida de {clave} no es la que usó el paso 29: el "
                f"registro y el volcado han cambiado por separado."
            )
        resultado[row_label(fila)] = (
            (variante.get("device_id", "") or "").strip(),
            (referencia.get("device_id", "") or "").strip(),
        )
    return resultado


def _normalized_class(name: str) -> str:
    """Nombre de clase en minúsculas y con espacios, para comparar COCO y registro."""
    return name.lower().replace("-", " ").replace("_", " ").strip()


def class_figure_rows(
    rows: Sequence[Mapping[str, Any]],
    runtime: str,
    compression: Sequence[Mapping[str, Any]] | None,
    experiments: Mapping[str, Mapping[str, str]] | None,
) -> list[tuple[str, Mapping[str, Any]]]:
    """Filas de la figura por clase de un motor, con su etiqueta y en orden.

    En ONNX Runtime y OpenVINO son las compresiones de la figura del 6.2, en su orden y sin el FP32
    de los QAT. En PyTorch (anexo), todas las que tienen desglose por clase.
    """
    if runtime in COMPRESSION_RUNTIMES:
        if not compression:
            raise ValueError(
                f"Sin las compresiones de la figura del 6.2 no se saben las filas de "
                f"{runtime}: son esas, en su orden."
            )
        seleccion = []
        for entrada in ordered_compressions(compression):
            fila = compression_row(rows, entrada, runtime)
            if not fila["per_class"]:
                raise ValueError(
                    f"'{entrada['label']}' no tiene AP por clase en {runtime}: la figura "
                    f"tendría una fila vacía."
                )
            seleccion.append((entrada["label"], fila))
        return seleccion

    if experiments is None:
        raise ValueError(
            f"Sin el registro de experimentos no se pueden rotular las filas de {runtime}."
        )
    seleccion = [
        (annex_label(f, experiments), f) for f in rows if f["per_class"] and f["runtime"] == runtime
    ]
    etiquetas = [e for e, _ in seleccion]
    repetidas = sorted({e for e in etiquetas if etiquetas.count(e) > 1})
    if repetidas:
        raise ValueError(
            f"En {runtime} hay filas distintas con la misma etiqueta ({repetidas}): "
            f"distínguelas antes de dibujarlas."
        )

    def clave(par: tuple[str, Mapping[str, Any]]) -> tuple[int, str, bool]:
        etiqueta, fila = par
        tecnica = (experiments[fila["experiment_id"]].get("technique", "") or "").strip()
        orden = ANNEX_TECHNIQUE_ORDER.index(tecnica) if tecnica in ANNEX_TECHNIQUE_ORDER else 99
        cruda = etiqueta.endswith(RAW_PRUNING_SUFFIX)
        base = etiqueta[: -len(RAW_PRUNING_SUFFIX)] if cruda else etiqueta
        return orden, base, cruda

    return sorted(seleccion, key=clave)


def figure_class_degradation(
    rows: Sequence[Mapping[str, Any]],
    instances: ClassInstances | None = None,
    devices: Mapping[str, tuple[str, str]] | None = None,
    runtime: str | None = None,
    *,
    compression: Sequence[Mapping[str, Any]] | None = None,
    experiments: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Mapa de calor de la retención de AP por clase y configuración en un motor.

    La retención es el AP de la variante dividido por el de su referencia. La escala de color es
    común a los tres motores. Las clases con pocas anotaciones se rayan. El texto del pie se
    devuelve en ``caption``.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import FixedLocator, FuncFormatter

    condesglose = [f for f in rows if f["per_class"]]
    if not condesglose:
        raise ValueError(
            "Ninguna configuración tiene AP por clase medido, así que no hay mapa de "
            "calor que dibujar."
        )
    if instances is None:
        raise ValueError(
            "No hay volcado de evaluación con el número de anotaciones por clase "
            "(results/raw/coco_eval)."
        )
    if devices is None:
        raise ValueError(
            "Faltan los dispositivos de cada comparación: sin ellos no se puede decir "
            "qué filas comparan dos máquinas distintas."
        )

    clases = sorted(condesglose[0]["per_class"])
    # La clase se ata a su recuento por el índice de la columna y se comprueba el nombre.
    recuento: dict[str, int] = {}
    for clase in clases:
        columnas = {f["per_class"][clase]["column"] for f in condesglose}
        if len(columnas) != 1:
            raise ValueError(f"La clase '{clase}' sale de columnas distintas: {sorted(columnas)}.")
        indice_clase = int(columnas.pop().split("_")[2])
        if indice_clase not in instances.by_index:
            raise ValueError(f"La clase '{clase}' no está en los volcados de evaluación.")
        nombre, n = instances.by_index[indice_clase]
        if _normalized_class(nombre) != _normalized_class(clase):
            raise ValueError(
                f"El índice {indice_clase} es '{clase}' en el registro y '{nombre}' en "
                f"los volcados de evaluación."
            )
        recuento[clase] = n
    escasas = [c for c in clases if recuento[c] < MIN_INTERPRETABLE_INSTANCES]

    motores = sorted({f["runtime"] for f in condesglose})
    if runtime not in motores:
        raise ValueError(
            f"El motor '{runtime}' no tiene ninguna fila con AP por clase; los que "
            f"tienen son {motores}."
        )
    por_motor = {m: class_figure_rows(rows, m, compression, experiments) for m in motores}

    def retencion(delta: Mapping[str, Any]) -> float:
        if not delta["reference"]:
            raise ValueError(
                f"La referencia de '{delta['column']}' es cero: la retención no está "
                f"definida y dibujarla como 0 o como 1 sería inventarla."
            )
        return float(delta["value"]) / float(delta["reference"])

    matrices = [
        np.array([[retencion(f["per_class"][c]) for c in clases] for _, f in por_motor[m]])
        for m in motores
    ]
    # Escala común a las tres figuras, con el 1 en blanco y el máximo redondeado a la décima.
    techo = max(1.1, math.ceil(max(float(m.max()) for m in matrices) * 10) / 10)
    escala = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=techo)

    # Comparaciones que cruzan de máquina, contadas de los datos.
    cruces: list[str] = []
    resumen_dispositivos: dict[str, dict[str, int]] = {}
    for motor in motores:
        parejas: dict[tuple[str, str], int] = {}
        for _, f in por_motor[motor]:
            if row_label(f) not in devices:
                raise ValueError(f"Falta el dispositivo de la fila {row_label(f)}.")
            par = devices[row_label(f)]
            parejas[par] = parejas.get(par, 0) + 1
        resumen_dispositivos[motor] = {f"{a} -> {b}": n for (a, b), n in sorted(parejas.items())}
        if any(a != b for a, b in parejas):
            # Primero las que cruzan de máquina.
            ordenadas = sorted(
                parejas.items(), key=lambda par_n: (par_n[0][0] == par_n[0][1], par_n[0])
            )
            (a0, b0), n0 = ordenadas[0]
            resto = "".join(f"; {n}, {a} con {b}" for (a, b), n in ordenadas[1:])
            cruces.append(
                f"En la figura de {RUNTIME_TITLES.get(motor, motor)}, {n0} de sus "
                f"{len(por_motor[motor])} filas comparan una variante medida en {a0} con una "
                f"referencia medida en {b0}{resto}. Las que cruzan de máquina comparan dos "
                f"dispositivos además de dos modelos."
            )

    escasas_texto = ", ".join(f"{c} ({style.format_integer(recuento[c])})" for c in escasas)
    titulos = ", ".join(RUNTIME_TITLES.get(m, m) for m in motores)
    pie = (
        "Color y cifra: retención, el AP de la variante dividido por el de su referencia "
        "—medida en el mismo motor y con el mismo evaluador—, en %; 100 es sin cambio. "
        f"Escala común a las {len(motores)} figuras, una por motor ({titulos}), de 0 a "
        f"{style.format_integer(round(100 * techo))}. Las clases van en orden alfabético, "
        f"con sus anotaciones en la partición {instances.split} entre paréntesis."
    )
    if escasas:
        pie += (
            f"\nRayadas, y sin interpretar: las {len(escasas)} clases con menos de "
            f"{MIN_INTERPRETABLE_INSTANCES} anotaciones, {escasas_texto}. Su AP se apoya "
            f"en tan pocas cajas que su retención no se lee como efecto de la técnica."
        )
    if cruces:
        pie += "\n" + " ".join(cruces)
    if any(e.endswith(RAW_PRUNING_SUFFIX) for m in motores for e, _ in por_motor[m]):
        pie += (
            f"\n«{RAW_PRUNING_SUFFIX.strip()}» marca las podas medidas antes de "
            "repararlas: un paso intermedio, y su caída no es la degradación del método."
        )

    filas_motor = por_motor[runtime]
    matriz = matrices[motores.index(runtime)]
    style.apply_style()
    # Ejes colocados a mano, en pulgadas, después de medir las etiquetas, para que la figura mida 16
    # cm.
    figura = plt.figure(figsize=(CLASS_FIGURE_WIDTH, 8.0))
    eje = figura.add_axes((0.0, 0.0, 1.0, 1.0))
    eje_barra = figura.add_axes((0.0, 0.0, 1.0, 1.0))

    imagen = eje.imshow(matriz, cmap=style.DIVERGING_COLORMAP, norm=escala, aspect="auto")
    # La cifra en % sobre el color, en blanco si el fondo es oscuro.
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = float(matriz[i, j])
            oscuro = valor < 0.35 or valor > 1.0 + 0.65 * (techo - 1.0)
            eje.text(
                j,
                i,
                f"{round(100 * valor)}",
                ha="center",
                va="center",
                fontsize=CLASS_CELL_FONTSIZE,
                color="white" if oscuro else "#111111",
            )
    for j, clase in enumerate(clases):
        if clase in escasas:
            eje.add_patch(
                Rectangle(
                    (j - 0.5, -0.5),
                    1.0,
                    float(matriz.shape[0]),
                    fill=False,
                    hatch=SCARCE_CLASS_HATCH,
                    edgecolor="#8A8A8A",
                    linewidth=0.0,
                )
            )
    # Las etiquetas son las de la figura del 6.2, sin hash.
    eje.set_yticks(range(len(filas_motor)))
    eje.set_yticklabels([e for e, _ in filas_motor], fontsize=CLASS_LABEL_FONTSIZE)
    eje.set_xticks(range(len(clases)))
    eje.set_xticklabels(
        [f"{c} ({style.format_integer(recuento[c])})" for c in clases],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=CLASS_LABEL_FONTSIZE,
    )
    eje.set_title(
        f"Retención de AP por clase: {RUNTIME_TITLES.get(runtime, runtime)}", fontsize=8
    )
    eje.grid(False)

    barra = figura.colorbar(imagen, cax=eje_barra)
    paso = 0.25
    barra.ax.yaxis.set_major_locator(
        FixedLocator([paso * k for k in range(int(round(techo / paso)) + 1)])
    )
    barra.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: style.format_integer(round(100 * v)))
    )
    barra.ax.tick_params(labelsize=CLASS_LABEL_FONTSIZE)
    barra.set_label("retención (%)", fontsize=CLASS_LABEL_FONTSIZE)

    # Medidas alrededor del mapa, en pulgadas.
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    ppp = figura.dpi
    ancho_y = max(t.get_window_extent(dibujo).width for t in eje.get_yticklabels()) / ppp
    alto_x = max(t.get_window_extent(dibujo).height for t in eje.get_xticklabels()) / ppp
    ancho_barra = (
        max(t.get_window_extent(dibujo).width for t in barra.ax.get_yticklabels()) / ppp + 0.3
    )
    izquierda = ancho_y + 0.12
    hueco, grosor = 0.1, 0.1
    ancho_mapa = CLASS_FIGURE_WIDTH - izquierda - hueco - grosor - ancho_barra
    alto_mapa = 0.28 * len(filas_motor)
    # Las etiquetas de clase cuelgan unas 0,1 pulgadas bajo el eje.
    abajo = alto_x + 0.18
    alto = abajo + alto_mapa + 0.3
    figura.set_size_inches(CLASS_FIGURE_WIDTH, alto)
    eje.set_position(
        (
            izquierda / CLASS_FIGURE_WIDTH,
            abajo / alto,
            ancho_mapa / CLASS_FIGURE_WIDTH,
            alto_mapa / alto,
        )
    )
    eje_barra.set_position(
        (
            (izquierda + ancho_mapa + hueco) / CLASS_FIGURE_WIDTH,
            abajo / alto,
            grosor / CLASS_FIGURE_WIDTH,
            alto_mapa / alto,
        )
    )
    return figura, {
        "runtime": runtime,
        "placement": CLASS_FIGURE_PLACEMENT.get(runtime, ""),
        "runtimes": motores,
        "classes": clases,
        "rows": [f["experiment_id"] for _, f in filas_motor],
        "labels": [e for e, _ in filas_motor],
        "row_precisions": [f["precision"] for _, f in filas_motor],
        "color": "retencion",
        "scale": [0.0, techo],
        "split": instances.split,
        "instances": {c: recuento[c] for c in clases},
        "scarce_classes": escasas,
        "devices": resumen_dispositivos,
        "caption": pie,
    }


def figure_size_degradation(
    rows: Sequence[Mapping[str, Any]],
    compression: Sequence[Mapping[str, Any]] | None,
) -> tuple[Any, dict[str, Any]]:
    """Retención de AP por tamaño de objeto de cada compresión, en ONNX Runtime y OpenVINO.

    Tres marcadores por fila (pequeño, mediano y grande). La retención es el AP de la variante
    dividido por el del FP32 de su tamaño en el mismo motor. El eje sale de los datos porque se
    puede superar el 100 %.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if not compression:
        raise ValueError(
            "No hay compresiones de la figura del 6.2: sin ellas no se saben las filas "
            "ni su orden."
        )
    entradas = ordered_compressions(compression)
    datos: list[dict[str, Any]] = []
    for entrada in entradas:
        por_motor: dict[str, dict[str, Any]] = {}
        for motor in COMPRESSION_RUNTIMES:
            fila = compression_row(rows, entrada, motor)
            rangos: dict[str, Any] = {}
            for rango in SIZE_RANGES:
                delta = (fila["per_size"] or {}).get(rango)
                if delta is None:
                    raise ValueError(
                        f"'{entrada['label']}' no tiene AP de objetos '{rango}' en {motor}."
                    )
                if SIZE_CRITERION not in delta["column"]:
                    raise ValueError(
                        f"El AP '{rango}' de '{entrada['label']}' sale de '{delta['column']}', "
                        f"que no es el criterio de {SIZE_CRITERION}."
                    )
                if not delta["reference"]:
                    raise ValueError(
                        f"La referencia de '{delta['column']}' de '{entrada['label']}' es "
                        f"cero: la retención no está definida."
                    )
                rangos[rango] = {
                    "value": delta["value"],
                    "reference": delta["reference"],
                    "retention": float(delta["value"]) / float(delta["reference"]),
                    "column": delta["column"],
                }
            por_motor[motor] = {"precision": fila["precision"], "sizes": rangos}
        datos.append({"entry": entrada, "runtimes": por_motor})

    valores = [
        100.0 * d["runtimes"][m]["sizes"][r]["retention"]
        for d in datos
        for m in COMPRESSION_RUNTIMES
        for r in SIZE_RANGES
    ]
    # Límite inferior: la decena por debajo del mínimo; superior: 100 o el máximo, con margen.
    izquierda_x = math.floor((min(valores) - 3.0) / 10.0) * 10.0
    derecha_x = math.ceil((max(100.0, max(valores)) + 3.0) / 5.0) * 5.0
    n = len(datos)

    style.apply_style()
    figura = plt.figure(figsize=(SIZE_FIGURE_WIDTH, 4.0))
    ejes = {m: figura.add_axes((0.0, 0.0, 1.0, 1.0)) for m in COMPRESSION_RUNTIMES}
    for motor, eje in ejes.items():
        for i, d in enumerate(datos):
            xs = [100.0 * d["runtimes"][motor]["sizes"][r]["retention"] for r in SIZE_RANGES]
            eje.plot([min(xs), max(xs)], [i, i], color="#B0B0B0", linewidth=0.5, zorder=1)
            for rango, x in zip(SIZE_RANGES, xs):
                estilo = style.SIZE_STYLES[rango]
                # Marcadores huecos con borde fino para que no se tapen.
                eje.scatter(
                    x,
                    i,
                    marker=estilo.marker,
                    s=SIZE_MARKER_AREA,
                    facecolor="none",
                    edgecolor=estilo.color,
                    linewidth=SIZE_MARKER_EDGE,
                    zorder=3,
                )
        eje.axvline(100.0, color="#333333", linewidth=0.7, zorder=0.8)
        eje.set_xlim(izquierda_x, derecha_x)
        eje.set_ylim(n - 0.4, -0.6)
        eje.grid(False)
        eje.grid(axis="x", color="#000000", alpha=0.08, linewidth=0.4)
        eje.xaxis.set_major_locator(MultipleLocator(10.0))
        eje.xaxis.set_major_formatter(FuncFormatter(lambda v, _: style.format_integer(round(v))))
        eje.set_yticks(range(n))
        eje.tick_params(labelsize=SIZE_FONTSIZE, length=2.5, pad=2)
        eje.set_xlabel("retención de AP (%)", fontsize=SIZE_FONTSIZE, labelpad=2)
        eje.set_title(RUNTIME_TITLES[motor], fontsize=SIZE_TITLE_FONTSIZE, pad=3)
    primero, segundo = COMPRESSION_RUNTIMES
    ejes[primero].set_yticklabels([d["entry"]["label"] for d in datos], fontsize=SIZE_FONTSIZE)
    ejes[segundo].tick_params(labelleft=False)

    manejadores = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=style.SIZE_STYLES[r].marker,
            markerfacecolor="none",
            markeredgecolor=style.SIZE_STYLES[r].color,
            markeredgewidth=SIZE_MARKER_EDGE,
            markersize=4.5,
        )
        for r in SIZE_RANGES
    ]
    leyenda = figura.legend(
        manejadores,
        [f"objeto {style.SIZE_STYLES[r].label.lower()}" for r in SIZE_RANGES],
        loc="upper center",
        ncol=len(SIZE_RANGES),
        fontsize=SIZE_FONTSIZE,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.5,
    )

    # Ejes colocados a mano: 16 cm de ancho y como mucho 12 de alto.
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    ppp = figura.dpi
    ancho_y = max(t.get_window_extent(dibujo).width for t in ejes[primero].get_yticklabels()) / ppp
    alto_leyenda = leyenda.get_window_extent(dibujo).height / ppp
    alto_titulo = ejes[primero].title.get_window_extent(dibujo).height / ppp
    izquierda = ancho_y + 0.09
    hueco = 0.16
    margen = 0.06
    ancho_panel = (SIZE_FIGURE_WIDTH - izquierda - hueco - margen) / 2.0
    abajo = 0.34
    arriba = alto_titulo + 0.06 + alto_leyenda + 0.04
    # Unos 0,5 cm entre filas si caben.
    paso_fila = min(SIZE_ROW_PITCH, (SIZE_MAX_HEIGHT - abajo - arriba) / (n + 0.2))
    alto_panel = paso_fila * (n + 0.2)
    alto = abajo + alto_panel + arriba
    figura.set_size_inches(SIZE_FIGURE_WIDTH, alto)
    for indice, motor in enumerate(COMPRESSION_RUNTIMES):
        x0 = izquierda + indice * (ancho_panel + hueco)
        ejes[motor].set_position(
            (x0 / SIZE_FIGURE_WIDTH, abajo / alto, ancho_panel / SIZE_FIGURE_WIDTH, alto_panel / alto)
        )
    leyenda.set_bbox_to_anchor((0.5, 1.0), transform=figura.transFigure)

    pie = (
        "Retención: AP de la variante dividido por el del FP32 de su tamaño, en el mismo "
        "motor y con el mismo evaluador, en %; la línea vertical es el 100, sin cambio. "
        "Rangos de tamaño según el criterio de resolución de inferencia. Filas en el "
        f"orden de {RUNTIME_TITLES[primero]} de la figura de pérdida por compresión."
    )
    # Se comprueba que cada fila sale igual en los dos paneles.
    iguales = [
        d["entry"]["label"]
        for d in datos
        if all(
            d["runtimes"][primero]["sizes"][r]["retention"]
            == d["runtimes"][segundo]["sizes"][r]["retention"]
            for r in SIZE_RANGES
        )
    ]
    if iguales:
        pie += (
            f" {len(iguales)} filas valen lo mismo en los dos motores, porque su AP por "
            f"tamaño es idéntico en ambos: {', '.join(iguales)}."
        )
    return figura, {
        "criterion": SIZE_CRITERION,
        "ranges": list(SIZE_RANGES),
        "runtimes": list(COMPRESSION_RUNTIMES),
        "row_order": primero,
        "x_range": [izquierda_x, derecha_x],
        "row_pitch_cm": round(paso_fila * 2.54, 3),
        "rows": [
            {
                "label": d["entry"]["label"],
                "experiment_id": d["entry"]["experiment_id"],
                "reference": d["entry"]["reference"],
                **d["runtimes"],
            }
            for d in datos
        ],
        "caption": pie,
    }


def figure_cost_vs_gain(
    rows: Sequence[Mapping[str, Any]],
    excluded: Sequence[tuple[str, str]],
) -> tuple[Any, dict[str, Any]]:
    """Coste de entrenamiento o compresión frente a lo que se gana.

    Falla si ninguna configuración tiene coste y ganancia, o si alguna no trae un estado de
    aceleración reconocible.
    """
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError(
            "Ninguna configuración tiene a la vez coste registrado y ganancia medida."
        )

    style.apply_style()
    sin_estado = [f["label"] for f in rows if f.get("speedup_status") not in SPEEDUP_STATUSES]
    if sin_estado:
        raise ValueError(
            f"{len(sin_estado)} configuraciones llegan sin un estado de aceleración reconocible "
                f"({sin_estado[:3]})."
        )

    figura, ejes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    familias: list[str] = []
    sin_latencia: list[str] = []
    sin_certificar: list[str] = []

    for fila in sorted(rows, key=lambda f: (f["technique"], f["label"])):
        estilo = style.style_for(fila["technique"])
        if fila["technique"] not in familias:
            familias.append(fila["technique"])
        if fila["speedup_status"] == SPEEDUP_CERTIFIED:
            ejes[0].scatter(
                fila["cost_hours"],
                fila["speedup"],
                color=estilo.color,
                marker=estilo.marker,
                s=42,
                edgecolor="black",
                linewidth=0.4,
            )
        elif fila["speedup_status"] == SPEEDUP_NOT_CERTIFIED:
            sin_certificar.append(fila["label"])
        else:
            sin_latencia.append(fila["label"])
        if fila.get("quality_delta") is not None:
            ejes[1].scatter(
                fila["cost_hours"],
                fila["quality_delta"],
                color=estilo.color,
                marker=estilo.marker,
                s=42,
                edgecolor="black",
                linewidth=0.4,
            )

    ejes[0].set_xlabel("coste registrado (h de reloj de pared)")
    ejes[0].set_ylabel("factor de aceleración en p95")
    ejes[0].set_title("Lo que se gana en latencia")
    ejes[0].axhline(1.0, color="#555555", linewidth=0.7, linestyle=":")
    ejes[1].set_xlabel("coste registrado (h de reloj de pared)")
    # En dos líneas para que quepa en el panel.
    ejes[1].set_ylabel("mAP@50-95 de la variante\nmenos el de su referencia")
    ejes[1].set_title("Lo que se paga en calidad")
    ejes[1].axhline(0.0, color="#555555", linewidth=0.7, linestyle=":")
    # Eje de coste logarítmico: va de medio minuto (cuantizar) a horas (podar y reajustar,
    # destilar).
    costes = [fila["cost_hours"] for fila in rows]
    minimo, maximo = min(costes), max(costes)
    for eje in ejes:
        eje.set_xscale("log")
        eje.set_xlim(minimo / 2.0, maximo * 2.0)
        eje.xaxis.set_major_formatter(style.comma_axis_formatter())
        eje.yaxis.set_major_formatter(style.comma_axis_formatter())

    # La leyenda se construye con todas las familias, aunque alguna no salga en el panel izquierdo.
    manejadores = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=style.style_for(f).marker,
            color=style.style_for(f).color,
            markeredgecolor="black",
            markeredgewidth=0.4,
        )
        for f in familias
    ]
    figura.legend(
        manejadores,
        [style.style_for(f).label for f in familias],
        loc="upper center",
        ncol=len(familias),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )

    # Las décadas del eje se cuentan de los datos.
    decadas = math.log10(maximo / minimo) if minimo > 0 else 0.0
    pie = (
        "El coste es el reloj de pared que quedó registrado —entrenar donde hubo "
        "entrenamiento, exportar y cuantizar donde no (D-132)—, que para una sesión "
        f"de Colab es lo que de verdad se consume (D-101). El eje va en escala "
        f"logarítmica porque abarca {decadas:.0f} décadas. El factor de aceleración "
        "solo se dibuja cuando supera la resolución medida del eje (D-127, D-186)."
    )
    # Sin latencia y sin certificar se cuentan por separado.
    solo_derecho = len(sin_latencia) + len(sin_certificar)
    if solo_derecho:
        motivo_latencia = (
            "no tienen latencia medida en su motor y precisión, en la variante o en su "
            "referencia"
        )
        motivo_certificar = (
            "tienen latencia medida, pero su aceleración no supera la resolución del eje"
        )
        if sin_latencia and sin_certificar:
            detalle = (
                f": {len(sin_latencia)} porque {motivo_latencia}, y "
                f"{len(sin_certificar)} porque {motivo_certificar}."
            )
        elif sin_latencia:
            detalle = f" porque {motivo_latencia}. Ninguna falta por no estar certificada."
        else:
            detalle = f": {motivo_certificar}."
        pie += f"\n{solo_derecho} configuraciones aparecen solo en el panel derecho{detalle}"
    if excluded:
        pie += (
            f"\nQuedan fuera {len(excluded)} configuraciones sin coste registrado: "
            f"{', '.join(identificador for identificador, _ in excluded[:6])}"
            f"{' y otras' if len(excluded) > 6 else ''}. Dibujarlas en el cero diría "
            f"que fueron gratis, y lo que pasa es que su coste no se apuntó."
        )
    _footer(figura, pie)
    figura.tight_layout(rect=(0, 0.16 if excluded else 0.08, 1, 0.94))
    return figura, {
        "drawn": [f["label"] for f in rows],
        "without_latency": sin_latencia,
        "not_certified": sin_certificar,
        "excluded": [identificador for identificador, _ in excluded],
    }


@dataclass(frozen=True)
class ErrorDecomposition:
    """Lo que dibuja la figura de tipos de error y lo que queda fuera."""

    kinds: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]
    quality_rows: int
    by_route: Mapping[str, int]
    missing_by_reason: Mapping[str, int]
    techniques_absent: tuple[str, ...]
    confidence: float
    split: str
    num_gt: int

    def coverage(self) -> dict[str, Any]:
        """La cobertura, para el manifiesto."""
        return {
            "quality_rows": self.quality_rows,
            "drawn": len(self.rows),
            "by_route": dict(sorted(self.by_route.items())),
            "missing_by_reason": dict(sorted(self.missing_by_reason.items())),
            "techniques_absent": list(self.techniques_absent),
            "confidence": self.confidence,
            "split": self.split,
            "num_gt": self.num_gt,
        }


def _clave_de_tabla(valores: Mapping[str, Any], campos: Sequence[str]) -> tuple[str, ...]:
    """Clave primaria de una fila como tupla de cadenas."""
    return tuple(str(valores.get(campo, "") or "").strip() for campo in campos)


def error_decomposition_rows(
    quality_rows: Sequence[Mapping[str, str]],
    experiments: Mapping[str, Mapping[str, str]],
    dump: Mapping[str, Any],
) -> ErrorDecomposition:
    """Cruza las celdas error_* del registro con el volcado del paso 34.

    Los números salen del registro y el volcado aporta la ruta de cada cifra y el motivo de cada
    hueco. El cruce usa la clave primaria completa, y la figura falla si registro y volcado no
    coinciden.
    """
    campos = TABLE_PRIMARY_KEYS["quality_metrics"]
    tipos = tuple(dump["kinds"])
    if set(tipos) != set(style.ERROR_KIND_STYLES) or MISSED_KIND not in tipos:
        raise ValueError(
            f"El volcado reparte en {tipos} y el estilo de la figura en "
                f"{tuple(style.ERROR_KIND_STYLES)}."
        )
    columnas = dump["registry_columns"]
    confianza = float(dump["registry_confidence"])
    razones: Mapping[str, str] = dump["reasons"]
    sin_etiqueta = sorted(set(razones) - set(ERROR_REASON_LABELS))
    if sin_etiqueta:
        raise ValueError(
            f"El volcado declara motivos sin etiqueta en la figura: {sin_etiqueta}. "
            f"Añádelos a ERROR_REASON_LABELS antes de dibujar."
        )

    evaluaciones: dict[tuple[str, ...], tuple[str, Mapping[str, Any], int]] = {}
    agregaciones: set[str] = set()
    for evaluacion in dump["evaluations"]:
        clave_volcado = evaluacion["primary_key"]
        if set(clave_volcado) != set(campos):
            raise ValueError(
                f"El volcado identifica sus evaluaciones con {sorted(clave_volcado)} y la "
                f"clave primaria de quality_metrics es {list(campos)}."
            )
        clave = _clave_de_tabla(clave_volcado, campos)
        if clave in evaluaciones:
            raise ValueError(f"El volcado trae dos evaluaciones con la clave {clave}.")
        reparto = evaluacion["decomposition"]
        puntos = [
            p for p in reparto["by_confidence"] if abs(float(p["confidence"]) - confianza) < 1e-9
        ]
        if len(puntos) != 1:
            raise ValueError(
                f"La evaluación {clave} tiene {len(puntos)} puntos a la confianza "
                f"{confianza} del registro."
            )
        evaluaciones[clave] = (evaluacion["route"], puntos[0], int(reparto["num_gt"]))
        agregaciones.add(reparto["aggregation"])
    # El pie dice «no medias por clase» y tiene que ser lo que declara el volcado.
    if len(agregaciones) > 1 or not all("no medias por clase" in a for a in agregaciones):
        raise ValueError(f"El volcado declara una agregación que el pie no dice: {agregaciones}")

    filas: list[dict[str, Any]] = []
    faltan: dict[str, int] = {}
    vistas: set[tuple[str, ...]] = set()
    tecnicas: set[str] = set()
    tecnicas_dibujadas: set[str] = set()
    anotaciones: set[int] = set()

    for fila in quality_rows:
        clave = _clave_de_tabla(fila, campos)
        experimento = experiments.get(fila["experiment_id"], {})
        tecnica = (experimento.get("technique", "") or "").strip()
        tecnicas.add(tecnica)
        celdas = {t: (fila.get(columnas[t], "") or "").strip() for t in tipos}
        llenas = [t for t in tipos if celdas[t]]

        if not llenas:
            notas = fila.get("notes", "") or ""
            motivos = [k for k, texto in razones.items() if texto in notas]
            if len(motivos) != 1:
                raise ValueError(
                    f"La fila {clave} no tiene reparto y su 'notes' trae {len(motivos)} motivos "
                    "del paso 34. Vuelve a ejecutar el paso 34."
                )
            faltan[motivos[0]] = faltan.get(motivos[0], 0) + 1
            continue
        if len(llenas) != len(tipos):
            raise ValueError(
                f"La fila {clave} tiene {len(llenas)} de las {len(tipos)} columnas del "
                f"reparto: un reparto a medias no suma nada."
            )
        if clave not in evaluaciones:
            raise ValueError(
                f"La fila {clave} tiene reparto en el registro y el volcado del paso 34 "
                f"no lo respalda: esa cifra no tiene evidencia versionada."
            )
        ruta, punto, num_gt = evaluaciones[clave]
        recuentos = {t: float(celdas[t]) for t in tipos}
        if any(v != int(v) for v in recuentos.values()):
            raise ValueError(f"La fila {clave} tiene recuentos no enteros: {recuentos}")
        if {t: int(v) for t, v in recuentos.items()} != {t: int(punto[t]) for t in tipos}:
            raise ValueError(
                f"Las celdas de {clave} no coinciden con el volcado del paso 34: el "
                f"registro o el volcado han cambiado por separado."
            )
        vistas.add(clave)
        anotaciones.add(num_gt)
        tecnicas_dibujadas.add(tecnica)
        # La raíz no tiene etapa; training_stage la daría por sin reajuste.
        padre = (experimento.get("parent_experiment_id", "") or "").strip()
        etapa = training_stage(fila["experiment_id"], experiments) if padre else STAGE_FINETUNED
        filas.append(
            {
                "experiment_id": fila["experiment_id"],
                "parent_experiment_id": padre,
                "technique": tecnica,
                "stage": etapa,
                "runtime": fila["runtime"],
                "precision": fila["precision"],
                "route": ruta,
                "counts": {t: int(recuentos[t]) for t in tipos},
            }
        )

    sobran = sorted(set(evaluaciones) - vistas)
    if sobran:
        raise ValueError(
            f"El volcado del paso 34 trae {len(sobran)} evaluaciones sin fila con reparto "
            f"en el registro, la primera {sobran[0]}."
        )
    if len(anotaciones) > 1:
        raise ValueError(
            "Las filas dibujadas se evaluaron sobre particiones con distinto número de "
            f"anotaciones ({sorted(anotaciones)})."
        )

    por_ruta: dict[str, int] = {}
    for fila_dibujada in filas:
        por_ruta[fila_dibujada["route"]] = por_ruta.get(fila_dibujada["route"], 0) + 1
    declarada = dump["coverage"]
    medida = {
        "quality_rows": len(quality_rows),
        "with_decomposition": len(filas),
        "by_route": dict(sorted(por_ruta.items())),
    }
    if medida != {c: declarada[c] for c in medida}:
        raise ValueError(
            f"La cobertura del cruce entre registro y volcado ({medida}) no es la que declara el "
            f"volcado ({ {c: declarada[c] for c in medida} })."
        )

    return ErrorDecomposition(
        kinds=tipos,
        rows=tuple(filas),
        quality_rows=len(quality_rows),
        by_route=por_ruta,
        missing_by_reason=faltan,
        techniques_absent=tuple(sorted(tecnicas - tecnicas_dibujadas)),
        confidence=confianza,
        split=str(dump["split"]),
        num_gt=anotaciones.pop() if anotaciones else 0,
    )


def error_decomposition_footer(data: ErrorDecomposition) -> str:
    """Pie de la figura de tipos de error con su cobertura, calculado de ``data``."""
    pie = (
        f"Recuentos a confianza {style.format_number(data.confidence, 2)} sobre la "
        f"partición {data.split} ({style.format_integer(data.num_gt)} anotaciones), "
        f"agregados sobre toda la partición y no medias por clase. Las cinco clases "
        f"apiladas reparten los falsos positivos sin resto; «no detectada» cuenta "
        f"anotaciones, no cajas, y por eso va en su propio panel."
    )
    faltan = sum(data.missing_by_reason.values())
    pie += f"\nCubre {len(data.rows)} de las {data.quality_rows} filas de calidad."
    if faltan:
        motivos = "; ".join(
            f"{n} {ERROR_REASON_LABELS[k]}" for k, n in sorted(data.missing_by_reason.items())
        )
        pie += f" Las otras {faltan} no tienen reparto: {motivos}."
    if data.techniques_absent:
        ausentes = ", ".join(style.style_for(t).label.lower() for t in data.techniques_absent)
        pie += f" Sin ninguna fila en la figura: {ausentes}."
    # Una poda sin reajustar sería un paso intermedio; si alguna entra, se avisa aquí.
    en_crudo = sorted({f["technique"] for f in data.rows if stage_suffix(f)} - {"ptq"})
    if en_crudo:
        pie += (
            f" Hay filas de {', '.join(style.style_for(t).label.lower() for t in en_crudo)} "
            f"medidas antes de repararlas: no son el resultado del método."
        )
    debiles = len(data.rows) - data.by_route.get(STRONG_ROUTE, 0)
    if debiles:
        pie += (
            f"\n{WEAK_ROUTE_MARK} y marcador hueco: cifra anclada solo contra el mAP "
            f"registrado, sin el contraste contra la curva de operación que tienen las "
            f"demás ({debiles} de {len(data.rows)}; D-183)."
        )
    return pie


def error_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[list[tuple[str, str, str]]]]]:
    """Agrupa las configuraciones de la figura de tipos de error en parejas FP32 e INT8.

    La pareja es el FP32 del mismo experimento (QAT) o el de su padre en el registro.
    """
    configuraciones = sorted({(f["technique"], f["experiment_id"], f["precision"]) for f in rows})
    padres = {f["experiment_id"]: (f.get("parent_experiment_id", "") or "") for f in rows}
    fp32 = {c[1]: c for c in configuraciones if c[2] == "fp32"}

    familias: dict[str, list[list[tuple[str, str, str]]]] = {}
    usadas: set[tuple[str, str, str]] = set()
    for c in configuraciones:
        if c[2] != "int8":
            continue
        socia = fp32.get(c[1]) or fp32.get(padres.get(c[1], ""))
        if socia is None or socia in usadas:
            continue
        familias.setdefault(c[0], []).append([socia, c])
        usadas.update((socia, c))

    orden = [t for t in ERROR_FAMILY_ORDER if t in familias]
    orden += sorted(t for t in familias if t not in ERROR_FAMILY_ORDER)
    bloques = [(t, sorted(familias[t], key=lambda g: g[-1][1])) for t in orden]
    sueltas = [[c] for c in configuraciones if c not in usadas]
    if sueltas:
        bloques.append(("", sueltas))
    return bloques


def error_configuration_label(
    experiment_id: str, precision: str, experiments: Mapping[str, Mapping[str, str]]
) -> str:
    """Etiqueta de una configuración en la figura de tipos de error, con el tamaño delante.

    El FP32 de un QAT lleva QAT_FP32_SUFFIX.
    """
    etiqueta, categoria, _ = configuration_label(
        experiment_id, experiment_chain(experiment_id, experiments), experiments
    )
    if categoria in ("qat", "pruning_qat") and precision == "fp32":
        etiqueta += QAT_FP32_SUFFIX
    return etiqueta


def figure_error_decomposition(
    data: ErrorDecomposition | None,
    experiments: Mapping[str, Mapping[str, str]],
) -> tuple[Any, dict[str, Any]]:
    """Descomposición de los tipos de error por configuración y motor.

    Barras apiladas por motor con los cinco tipos de falso positivo y un panel aparte con las
    anotaciones no detectadas. El texto del pie va en ``caption``.
    """
    import matplotlib.pyplot as plt

    if data is None:
        raise ValueError(
            "No está el volcado del paso 34 (results/raw/error_decomposition)."
        )
    if not data.rows:
        raise ValueError(
            "Ninguna fila de calidad tiene el reparto de errores: no hay nada que dibujar."
        )

    from matplotlib.ticker import FixedLocator
    from matplotlib.transforms import blended_transform_factory

    tipos_fp = [t for t in data.kinds if t != MISSED_KIND]
    motores = sorted({f["runtime"] for f in data.rows})
    bloques = error_pairs(data.rows)
    configuraciones = [c for _, grupos in bloques for grupo in grupos for c in grupo]
    # Hueco pequeño entre parejas y mayor, con rótulo, entre familias.
    posicion: dict[tuple[str, str, str], float] = {}
    cabeceras: list[tuple[float, str]] = []
    y = 0.0
    for familia, grupos in bloques:
        y += ERROR_FAMILY_GAP
        cabeceras.append((y - 0.5 - ERROR_FAMILY_GAP / 2, familia))
        for indice_grupo, grupo in enumerate(grupos):
            if indice_grupo:
                y += ERROR_PAIR_GAP
            for c in grupo:
                posicion[c] = y
                y += 1.0
    por_celda = {
        (f["technique"], f["experiment_id"], f["precision"], f["runtime"]): f for f in data.rows
    }
    etiquetas: dict[tuple[str, str, str], str] = {
        c: error_configuration_label(c[1], c[2], experiments) for c in configuraciones
    }
    valores_etiqueta = list(etiquetas.values())
    repetidas = sorted({e for e in valores_etiqueta if valores_etiqueta.count(e) > 1})
    if repetidas:
        raise ValueError(
            f"Dos configuraciones saldrían con la misma etiqueta ({repetidas}): "
            f"distínguelas antes de dibujarlas."
        )

    style.apply_style()
    figura = plt.figure(figsize=(ERROR_FIGURE_WIDTH, 8.0))
    ejes = [figura.add_axes((0.0, 0.0, 1.0, 1.0))]
    for _ in motores:
        ejes.append(figura.add_axes((0.0, 0.0, 1.0, 1.0), sharey=ejes[0]))
    debiles: list[str] = []
    leyenda: list[Any] = []

    for indice, motor in enumerate(motores):
        eje = ejes[indice]
        filas_motor = [
            (posicion[c], por_celda[(*c, motor)])
            for c in configuraciones
            if (*c, motor) in por_celda
        ]
        config_de = {posicion[c]: c for c in configuraciones}
        izquierda = [0.0] * len(filas_motor)
        contenedores = []
        for tipo in tipos_fp:
            estilo = style.ERROR_KIND_STYLES[tipo]
            valores = [float(f["counts"][tipo]) for _, f in filas_motor]
            contenedores.append(
                eje.barh(
                    [p for p, _ in filas_motor],
                    valores,
                    left=izquierda,
                    height=0.72,
                    color=estilo.color,
                    label=estilo.label,
                )
            )
            izquierda = [a + v for a, v in zip(izquierda, valores)]
        style.grayscale_safe_bars(eje, contenedores)
        if indice == 0:
            leyenda = contenedores
        for (pos, fila), total in zip(filas_motor, izquierda):
            if fila["route"] != STRONG_ROUTE:
                eje.annotate(
                    WEAK_ROUTE_MARK,
                    xy=(total, pos),
                    xytext=(2, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=ERROR_LABEL_FONTSIZE,
                )
                debiles.append(f"{etiquetas[config_de[pos]]} [{motor}]")
        eje.set_title(
            f"Falsos positivos: {RUNTIME_TITLES.get(motor, motor)}",
            fontsize=ERROR_TITLE_FONTSIZE,
            pad=3,
        )
        eje.set_xlabel("detecciones", fontsize=ERROR_LABEL_FONTSIZE, labelpad=2)
        eje.xaxis.set_major_formatter(style.comma_axis_formatter())
        # Hueco a la derecha para la marca de la ruta débil.
        eje.set_xlim(0, max(izquierda) * 1.08 if izquierda else 1)

    # No detectadas: un marcador por motor, desplazados en vertical para que no se tapen.
    eje_fn = ejes[-1]
    estilo_fn = style.ERROR_KIND_STYLES[MISSED_KIND]
    paso_vertical = 0.36 / max(1, len(motores) - 1) if len(motores) > 1 else 0.0
    manejadores_fn = []
    for indice, motor in enumerate(motores):
        desplazamiento = -0.18 + indice * paso_vertical if len(motores) > 1 else 0.0
        marcador = RUNTIME_MARKERS[indice % len(RUNTIME_MARKERS)]
        for c in configuraciones:
            fila = por_celda.get((*c, motor))
            if fila is None:
                continue
            fuerte = fila["route"] == STRONG_ROUTE
            eje_fn.scatter(
                fila["counts"][MISSED_KIND],
                posicion[c] + desplazamiento,
                marker=marcador,
                s=14,
                facecolors=estilo_fn.color if fuerte else "none",
                edgecolors=estilo_fn.color,
                linewidths=0.8,
            )
        manejadores_fn.append(
            plt.Line2D(
                [],
                [],
                linestyle="none",
                marker=marcador,
                color=estilo_fn.color,
                label=f"{estilo_fn.label} ({RUNTIME_TITLES.get(motor, motor)})",
            )
        )
    # En dos líneas para que no se salga.
    eje_fn.set_title(
        f"No detectadas\n(de {style.format_integer(data.num_gt)} anotaciones)",
        fontsize=ERROR_TITLE_FONTSIZE,
        pad=3,
    )
    eje_fn.set_xlabel("anotaciones", fontsize=ERROR_LABEL_FONTSIZE, labelpad=2)
    eje_fn.xaxis.set_major_formatter(style.comma_axis_formatter())
    # El eje de no detectadas empieza en el múltiplo de 500 bajo el mínimo; el pie lo indica.
    minimo_fn = min(float(f["counts"][MISSED_KIND]) for f in data.rows)
    maximo_fn = max(float(f["counts"][MISSED_KIND]) for f in data.rows)
    inicio_fn = math.floor(minimo_fn / 500.0) * 500.0
    eje_fn.set_xlim(inicio_fn, maximo_fn + 0.06 * max(1.0, maximo_fn - inicio_fn))
    # Sin marca en el origen, para que no se junte con la del panel vecino.
    eje_fn.xaxis.set_major_locator(
        FixedLocator([inicio_fn + 500.0 * k for k in range(1, int((maximo_fn - inicio_fn) // 500) + 2)])
    )

    ejes[0].set_yticks([posicion[c] for c in configuraciones])
    ejes[0].set_yticklabels([etiquetas[c] for c in configuraciones], fontsize=ERROR_LABEL_FONTSIZE)
    # Eje invertido a mano: la primera familia arriba.
    ejes[0].set_ylim(y - 1.0 + 0.6, -0.5)
    for eje in ejes:
        eje.tick_params(labelsize=ERROR_LABEL_FONTSIZE, length=2.5, pad=2)
    for eje in ejes[1:]:
        eje.tick_params(axis="y", labelleft=False)
    mezcla = blended_transform_factory(ejes[0].transAxes, ejes[0].transData)
    for altura, familia in cabeceras:
        ejes[0].text(
            0.005,
            altura,
            style.style_for(familia).label if familia else "Sin pareja",
            transform=mezcla,
            ha="left",
            va="center",
            fontsize=ERROR_LABEL_FONTSIZE,
            fontstyle="italic",
            fontweight="bold",
            color="#333333",
        )

    manejadores_extra: list[Any] = []
    if debiles:
        manejadores_extra.append(
            plt.Line2D(
                [],
                [],
                linestyle="none",
                marker=RUNTIME_MARKERS[0],
                markerfacecolor="none",
                markeredgecolor=estilo_fn.color,
                color=estilo_fn.color,
                label=f"hueco y {WEAK_ROUTE_MARK}: {WEAK_ROUTE_LEGEND}",
            )
        )
    manejadores = [*leyenda, *manejadores_fn, *manejadores_extra]
    figura_leyenda = figura.legend(
        manejadores,
        [
            *(style.ERROR_KIND_STYLES[t].label for t in tipos_fp),
            *(m.get_label() for m in manejadores_fn),
            *(m.get_label() for m in manejadores_extra),
        ],
        loc="upper center",
        ncol=math.ceil(len(manejadores) / 3),
        fontsize=ERROR_LABEL_FONTSIZE,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0,
    )

    # Ejes colocados a mano para que la figura mida 16 cm.
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    ppp = figura.dpi
    ancho_y = max(t.get_window_extent(dibujo).width for t in ejes[0].get_yticklabels()) / ppp
    alto_leyenda = figura_leyenda.get_window_extent(dibujo).height / ppp
    alto_titulo = max(e.title.get_window_extent(dibujo).height for e in ejes) / ppp
    izquierda_fig = ancho_y + 0.08
    hueco, margen = 0.14, 0.1
    # Panel de no detectadas algo más ancho para que quepa su título.
    proporciones = [3.0] * len(motores) + [2.0]
    disponible = ERROR_FIGURE_WIDTH - izquierda_fig - hueco * len(motores) - margen
    anchos = [disponible * p / sum(proporciones) for p in proporciones]
    abajo = 0.34
    arriba = alto_titulo + 0.08 + alto_leyenda + 0.04
    unidades = y - 1.0 + 0.6 + 0.5
    alto_fila = min(ERROR_ROW_HEIGHT, (ERROR_MAX_HEIGHT - abajo - arriba) / unidades)
    alto_panel = alto_fila * unidades
    alto = abajo + alto_panel + arriba
    figura.set_size_inches(ERROR_FIGURE_WIDTH, alto)
    x0 = izquierda_fig
    for eje, ancho_eje in zip(ejes, anchos):
        eje.set_position(
            (x0 / ERROR_FIGURE_WIDTH, abajo / alto, ancho_eje / ERROR_FIGURE_WIDTH, alto_panel / alto)
        )
        x0 += ancho_eje + hueco
    figura_leyenda.set_bbox_to_anchor((0.5, 1.0), transform=figura.transFigure)

    parejas = [g for _, grupos in bloques for g in grupos if len(g) == 2]
    pie = error_decomposition_footer(data)
    pie += (
        f"\nFilas por pares, {len(parejas)} en total: cada INT8 justo debajo de su FP32 "
        "de partida —el baseline del mismo tamaño para la PTQ, el mismo QAT en FP32 y "
        "el podado reajustado para las combinadas—, agrupadas por familia."
    )
    if inicio_fn > 0:
        pie += (
            f" El eje de «no detectadas» empieza en {style.format_integer(inicio_fn)}, no "
            f"en cero: las diferencias entre filas se leen; las proporciones, no."
        )

    return figura, {
        "configurations": [etiquetas[c] for c in configuraciones],
        "experiments": [list(c[1:]) for c in configuraciones],
        "pairs": [[etiquetas[c] for c in g] for g in parejas],
        "runtimes": motores,
        "weak_route": debiles,
        "missed_axis_start": inicio_fn,
        "row_height_in": round(alto_fila, 4),
        "coverage": data.coverage(),
        "caption": pie,
    }


def _experiment_variant(experiment_id: str) -> str:
    """Variante de un experiment_id (lo que hay entre la técnica y el hash)."""
    partes = experiment_id.split("_")
    if len(partes) < 4:
        raise ValueError(f"'{experiment_id}' no tiene la forma tamaño_técnica_variante_hash.")
    return "_".join(partes[2:-1])


def _pruning_label(experiment_id: str) -> tuple[str, bool]:
    """Etiqueta de una poda («L1 r25», «Aleatoria r50», «Iterativa r50») y si es control aleatorio.

    Un fragmento de variante desconocido hace fallar la etiqueta.
    """
    import re

    fragmentos = _experiment_variant(experiment_id).split("_")
    if (
        len(fragmentos) < 2
        or fragmentos[0] not in PRUNING_METHOD_LABELS
        or not re.fullmatch(r"r\d+", fragmentos[1])
    ):
        raise ValueError(
            f"La variante de '{experiment_id}' no empieza por un método de poda conocido "
                f"({sorted(PRUNING_METHOD_LABELS)}) y un ratio."
        )
    nombre = PRUNING_METHOD_LABELS[fragmentos[0]]
    modificadores = fragmentos[2:]
    control = PRUNING_CONTROL_TOKEN in modificadores
    iterativa = [m for m in modificadores if re.fullmatch(r"iter\d+", m)]
    desconocidos = [m for m in modificadores if m != PRUNING_CONTROL_TOKEN and m not in iterativa]
    if desconocidos or (control and iterativa) or len(iterativa) > 1:
        raise ValueError(
            f"La variante de '{experiment_id}' lleva fragmentos que no se saben rotular "
            f"({modificadores})."
        )
    if control:
        nombre = "Aleatoria"
    elif iterativa:
        nombre = "Iterativa"
    return f"{nombre} {fragmentos[1]}", control


def compression_label(
    experiment_id: str,
    chain: Sequence[str],
    experiments: Mapping[str, Mapping[str, str]],
) -> tuple[str, str, bool]:
    """Etiqueta, categoría y si es control de una compresión, a partir del registro.

    Las filas de s no llevan tamaño; las demás lo llevan al final («PTQ medium»).
    """
    fila = experiments[experiment_id]
    tecnica = (fila.get("technique", "") or "").strip()
    podas = [
        e for e in chain[1:-1] if (experiments[e].get("technique", "") or "").strip() == "pruning"
    ]
    tamano = (fila.get("model_size", "") or "").strip()
    if tamano not in COMPRESSION_SIZE_SUFFIX:
        raise ValueError(f"'{experiment_id}' es de un tamaño sin rótulo: '{tamano}'.")

    if tecnica in ("ptq", "qat"):
        variante = _experiment_variant(experiment_id)
        if variante not in QUANTIZATION_VARIANTS:
            raise ValueError(
                f"'{experiment_id}' es una cuantización con la variante '{variante}', que no se "
                "sabe rotular."
            )

    if tecnica == "pruning" and not podas:
        etiqueta, control = _pruning_label(experiment_id)
        categoria = "pruning"
    elif tecnica == "ptq" and not podas:
        etiqueta, control, categoria = "PTQ", False, "ptq"
    elif tecnica == "qat" and not podas:
        etiqueta, control, categoria = "QAT", False, "qat"
    elif tecnica in ("qat", "combined") and len(podas) == 1:
        base, control = _pruning_label(podas[0])
        if tecnica == "combined":
            esperada = f"prune-{_experiment_variant(podas[0]).replace('_', '-')}-ptqint8"
            if _experiment_variant(experiment_id) != esperada:
                raise ValueError(
                    f"La combinación '{experiment_id}' cuelga de '{podas[0]}', pero su variante "
                    f"no es '{esperada}'."
                )
            etiqueta, categoria = f"{base} + PTQ", "combined"
        else:
            etiqueta, categoria = f"{base} + QAT", "pruning_qat"
    else:
        raise ValueError(
            f"'{experiment_id}' es de la técnica '{tecnica}' y cuelga de {len(podas)} podas: "
            f"no es ninguna de las categorías de la figura."
        )
    return etiqueta + COMPRESSION_SIZE_SUFFIX[tamano], categoria, control


def configuration_label(
    experiment_id: str,
    chain: Sequence[str],
    experiments: Mapping[str, Mapping[str, str]],
) -> tuple[str, str, bool]:
    """Etiqueta de cualquier configuración, incluidos los baselines, con el tamaño delante («n
    FP32»).

    Delega en compression_label y solo recoloca el tamaño.
    """
    fila = experiments[experiment_id]
    tecnica = (fila.get("technique", "") or "").strip()
    tamano = (fila.get("model_size", "") or "").strip()
    if tamano not in COMPRESSION_SIZE_SUFFIX:
        raise ValueError(f"'{experiment_id}' es de un tamaño sin rótulo: '{tamano}'.")
    if tecnica == "baseline":
        if len(chain) != 1:
            raise ValueError(
                f"'{experiment_id}' es un baseline y cuelga de {chain[1:]}: no se sabe "
                f"qué referencia es."
            )
        return f"{tamano} FP32", "baseline", False

    etiqueta, categoria, control = compression_label(experiment_id, chain, experiments)
    sufijo = COMPRESSION_SIZE_SUFFIX[tamano]
    base = etiqueta[: len(etiqueta) - len(sufijo)] if sufijo else etiqueta
    if categoria in ("ptq", "qat") or tamano != "s":
        return f"{tamano} {base}", categoria, control
    return base, categoria, control


def experiment_chain(
    experiment_id: str, experiments: Mapping[str, Mapping[str, str]]
) -> tuple[str, ...]:
    """Cadena de ascendencia de un experimento hasta la raíz, a partir de parent_experiment_id."""
    if experiment_id not in experiments:
        raise ValueError(f"'{experiment_id}' no está en el registro de experimentos.")
    cadena = [experiment_id]
    padre = (experiments[experiment_id].get("parent_experiment_id", "") or "").strip()
    while padre:
        if padre in cadena:
            raise ValueError(f"La ascendencia de '{experiment_id}' tiene un ciclo en '{padre}'.")
        if padre not in experiments:
            raise ValueError(
                f"'{padre}', en la ascendencia de '{experiment_id}', no está en el registro."
            )
        cadena.append(padre)
        padre = (experiments[padre].get("parent_experiment_id", "") or "").strip()
    return tuple(cadena)


def _distillation_label(experiment_id: str) -> str:
    """Etiqueta de una destilación («Destilación l a n», «Control α=0 l a n, piloto 10 ép.»)."""
    import re

    fragmentos = _experiment_variant(experiment_id).split("_")
    par = re.fullmatch(r"([nslm])2([nslm])", fragmentos[0])
    tipos = [f for f in fragmentos[1:] if f in DISTILLATION_TOKENS]
    piloto = [f for f in fragmentos[1:] if re.fullmatch(r"pilot\d+", f)]
    # La temperatura y el alfa nulo del control no cambian la etiqueta.
    conocidos = {*tipos, *piloto, "alpha0"} | {
        f for f in fragmentos[1:] if re.fullmatch(r"t\d+", f)
    }
    desconocidos = [f for f in fragmentos[1:] if f not in conocidos]
    if par is None or len(tipos) != 1 or len(piloto) > 1 or desconocidos:
        raise ValueError(
            f"La variante de '{experiment_id}' no se sabe rotular ({fragmentos})."
        )
    if tipos[0] == "control" and "alpha0" not in fragmentos:
        raise ValueError(f"'{experiment_id}' es un control sin alfa nulo: no se sabe qué es.")
    etiqueta = f"{DISTILLATION_TOKENS[tipos[0]]} {par.group(1)}→{par.group(2)}"
    if piloto:
        etiqueta += f", piloto {piloto[0][len('pilot'):]} ép."
    return etiqueta


def annex_label(row: Mapping[str, Any], experiments: Mapping[str, Mapping[str, str]]) -> str:
    """Etiqueta de cualquier fila de degradación, para la figura del anexo.

    Incluye destilación, poda sin reajustar y el FP32 de un QAT.
    """
    from tfm_edge.analysis.degradation import STAGE_RAW

    experiment_id = row["experiment_id"]
    tecnica = (experiments[experiment_id].get("technique", "") or "").strip()
    if tecnica == "distillation":
        return _distillation_label(experiment_id)
    cadena = row["reference"]["chain"]
    etiqueta, categoria, _ = compression_label(experiment_id, cadena, experiments)
    poda = any(
        (experiments[e].get("technique", "") or "").strip() == "pruning" for e in cadena[:-1]
    )
    if row.get("stage") == STAGE_RAW and poda:
        etiqueta += RAW_PRUNING_SUFFIX
    # En el cuerpo «QAT» es el INT8; su FP32 se indica, como en la figura de tipos de error.
    if categoria in ("qat", "pruning_qat") and row["precision"] == "fp32":
        etiqueta += QAT_FP32_SUFFIX
    return etiqueta


def ordered_compressions(
    entries: Sequence[Mapping[str, Any]], runtime: str = COMPRESSION_RUNTIMES[0]
) -> list[Mapping[str, Any]]:
    """Compresiones de menor a mayor pérdida (y por experimento si empatan), el orden de la figura
    del 6.2.
    """
    return sorted(entries, key=lambda e: (-e["runtimes"][runtime]["delta"], e["experiment_id"]))


def compression_row(
    rows: Sequence[Mapping[str, Any]], entry: Mapping[str, Any], runtime: str
) -> Mapping[str, Any]:
    """Fila del volcado del paso 29 de una compresión en un motor, por experimento, motor y
    precisión.
    """
    precision = entry["runtimes"][runtime]["precision"]
    clave = (entry["experiment_id"], runtime, precision)
    filas = [f for f in rows if (f["experiment_id"], f["runtime"], f["precision"]) == clave]
    if len(filas) != 1:
        raise ValueError(f"La compresión {clave} tiene {len(filas)} filas en el volcado del paso "
                         "29.")
    fila = filas[0]
    volcado = fila["overall"].get("map_50_95", {}).get("value")
    if volcado is None or abs(float(volcado) - float(entry["runtimes"][runtime]["value"])) > 1e-9:
        raise ValueError(
            f"La fila {clave} del volcado no tiene el mAP de la compresión: no es la misma."
        )
    return fila


def compression_losses(
    rows: Sequence[Mapping[str, Any]],
    experiments: Mapping[str, Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], ...]:
    """Compresiones de la figura del 6.2 con su pérdida en cada motor.

    Solo ONNX Runtime y OpenVINO, sin destilación ni poda sin reajustar (mirando la cadena, porque
    la PTQ también figura como sin reajuste) y con la fila INT8 si la hay. Cada fila se reconstruye
    desde el registro con degradation._pick_reference y debe coincidir con el volcado.
    """
    from tfm_edge.analysis.degradation import STAGE_RAW, _pick_reference

    def tecnica(experiment_id: str) -> str:
        return (experiments[experiment_id].get("technique", "") or "").strip()

    grupos: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for fila in rows:
        if fila["runtime"] not in COMPRESSION_RUNTIMES:
            continue
        cadena = fila["reference"]["chain"]
        if tecnica(fila["experiment_id"]) == "distillation":
            continue
        if fila.get("stage") == STAGE_RAW and any(tecnica(e) == "pruning" for e in cadena[:-1]):
            continue
        grupos.setdefault((fila["experiment_id"], fila["runtime"]), []).append(fila)

    entradas: dict[str, dict[str, Any]] = {}
    for (experiment_id, motor), filas in sorted(grupos.items()):
        int8 = [f for f in filas if f["precision"] == "int8"]
        elegidas = int8 or filas
        if len(elegidas) != 1:
            raise ValueError(
                f"{experiment_id} tiene {len(elegidas)} filas en {motor} "
                f"({sorted(f['precision'] for f in elegidas)}) y ninguna regla elige una."
            )
        fila = elegidas[0]
        raiz = fila["reference"]["experiment_id"]
        tamano = (experiments[experiment_id].get("model_size", "") or "").strip()
        if (
            tecnica(raiz) != "baseline"
            or (experiments[raiz].get("model_size", "") or "").strip() != tamano
            or fila["reference"]["precision"] != "fp32"
        ):
            raise ValueError(
                f"La referencia de {experiment_id} en {motor} es {raiz} "
                f"[{fila['reference']['precision']}], que no es el FP32 del baseline de su "
                f"tamaño ('{tamano}')."
            )

        clave = (experiment_id, motor, fila["precision"])
        variantes = [
            q for q in quality_rows if (q["experiment_id"], q["runtime"], q["precision"]) == clave
        ]
        if len(variantes) != 1:
            raise ValueError(f"La variante {clave} tiene {len(variantes)} filas de calidad.")
        variante = variantes[0]
        evaluador = (variante.get("evaluator", "") or "").strip()
        referencia = _pick_reference(
            [
                q
                for q in quality_rows
                if q["experiment_id"] == raiz
                and q["runtime"] == motor
                and (q.get("evaluator", "") or "").strip() == evaluador
            ],
            variante.get("device_id", ""),
            ("map_50", "map_50_95"),
        )
        if referencia is None:
            raise ValueError(f"La referencia de {clave} no está en el registro.")
        for q in (variante, referencia):
            if (q.get("evaluator", "") or "").strip() != COMPRESSION_EVALUATOR or (
                q.get("split", "") or ""
            ).strip() != COMPRESSION_SPLIT:
                raise ValueError(
                    f"La fila de calidad de {q['experiment_id']} en {motor} es de "
                    f"'{q.get('evaluator')}' sobre '{q.get('split')}' y la figura compara "
                    f"{COMPRESSION_EVALUATOR} sobre {COMPRESSION_SPLIT}."
                )
        volcado = fila["overall"]["map_50_95"]
        valor = float(variante["map_50_95"])
        valor_referencia = float(referencia["map_50_95"])
        if (
            abs(valor - volcado["value"]) > 1e-9
            or abs(valor_referencia - volcado["reference"]) > 1e-9
            or abs((valor - valor_referencia) - volcado["absolute"]) > 1e-9
        ):
            raise ValueError(
                f"La reconstrucción de {clave} desde el registro no coincide con el volcado del "
                "paso 29."
            )

        if experiment_id not in entradas:
            etiqueta, categoria, control = compression_label(
                experiment_id, fila["reference"]["chain"], experiments
            )
            entradas[experiment_id] = {
                "experiment_id": experiment_id,
                "label": etiqueta,
                "category": categoria,
                "control": control,
                "model_size": tamano,
                "reference": raiz,
                "runtimes": {},
            }
        entradas[experiment_id]["runtimes"][motor] = {
            "precision": fila["precision"],
            "value": volcado["value"],
            "reference_value": volcado["reference"],
            "delta": volcado["absolute"],
        }

    incompletas = [e for e, d in entradas.items() if set(d["runtimes"]) != set(COMPRESSION_RUNTIMES)]
    if incompletas:
        raise ValueError(
            f"Hay compresiones sin fila en los dos motores ({incompletas})."
        )
    etiquetas = [d["label"] for d in entradas.values()]
    repetidas = sorted({e for e in etiquetas if etiquetas.count(e) > 1})
    if repetidas:
        raise ValueError(
            f"Dos compresiones saldrían con la misma etiqueta ({repetidas}): distínguelas en "
            f"la figura antes de dibujarla."
        )
    return tuple(entradas[e] for e in sorted(entradas))


def quality_floor(floors: Mapping[str, Any]) -> float:
    """Suelo absoluto del mAP@50-95 según el volcado del paso 29.

    Falla si no está medido, no es absoluto o difiere entre sesiones.
    """
    suelo = floors.get("floors", {}).get("map_50_95")
    if not suelo or suelo.get("kind") != "absoluto":
        raise ValueError("El volcado del paso 29 no trae un suelo absoluto de mAP@50-95.")
    misma, entre = suelo.get("same_session"), suelo.get("cross_session")
    if misma is None or entre is None or misma != entre:
        raise ValueError(
            f"El suelo de mAP@50-95 vale {misma} en la misma sesión y {entre} entre sesiones: "
            f"la figura no sabe cuál aplicar."
        )
    return float(misma)


def rank_comparison(entries: Sequence[Mapping[str, Any]], floor: float) -> dict[str, Any]:
    """Orden por pérdida en cada motor y qué cambia entre ellos.

    Lista los pares vecinos que no superan el suelo y, de los que invierten su orden, cuántos lo
    superan en los dos motores.
    """
    from itertools import combinations

    delta = {(e["label"], m): e["runtimes"][m]["delta"] for e in entries for m in e["runtimes"]}
    orden = {m: [e["label"] for e in ordered_compressions(entries, m)] for m in COMPRESSION_RUNTIMES}
    puestos = {m: {etiqueta: i + 1 for i, etiqueta in enumerate(orden[m])} for m in orden}
    primero, segundo = COMPRESSION_RUNTIMES
    cambian = [e for e in orden[primero] if puestos[primero][e] != puestos[segundo][e]]
    vecinos = {
        m: [
            {"pair": [a, b], "difference": round(abs(delta[(a, m)] - delta[(b, m)]), 6)}
            for a, b in zip(orden[m], orden[m][1:])
            if abs(delta[(a, m)] - delta[(b, m)]) <= floor
        ]
        for m in COMPRESSION_RUNTIMES
    }
    inversiones = []
    for a, b in combinations(orden[primero], 2):
        diferencias = {m: delta[(a, m)] - delta[(b, m)] for m in COMPRESSION_RUNTIMES}
        if diferencias[primero] * diferencias[segundo] < 0:
            inversiones.append(
                {
                    "pair": [a, b],
                    **{m: round(abs(d), 6) for m, d in diferencias.items()},
                    "above_floor_in_both": all(abs(d) > floor for d in diferencias.values()),
                }
            )
    return {
        "order": orden,
        "ranks": puestos,
        "positions_changed": cambian,
        "adjacent_within_floor": vecinos,
        "order_inversions": inversiones,
        "inversions_above_floor_in_both": sum(i["above_floor_in_both"] for i in inversiones),
    }


def figure_compression_loss(
    entries: Sequence[Mapping[str, Any]] | None,
    floors: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Pérdida de mAP@50-95 de cada compresión frente al FP32 de su tamaño, un panel por motor.

    Las filas siguen el orden de ONNX Runtime en los dos paneles y los cambios de puesto van en una
    columna a la derecha. El suelo se dibuja como segmento de escala.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    if not entries:
        raise ValueError(
            "No hay compresiones con fila en los dos motores: no hay pérdida que comparar."
        )
    suelo = quality_floor(floors)
    comparacion = rank_comparison(entries, suelo)
    primero, segundo = COMPRESSION_RUNTIMES
    orden = comparacion["order"][primero]
    por_etiqueta = {e["label"]: e for e in entries}
    n = len(orden)

    minimo = min(e["runtimes"][m]["delta"] for e in entries for m in COMPRESSION_RUNTIMES)
    paso = 0.05
    izquierda_x = math.floor(minimo / (paso / 2)) * (paso / 2)

    style.apply_style()
    figura = plt.figure(figsize=(COMPRESSION_FIGURE_WIDTH, 4.0))
    ejes = {m: figura.add_axes((0.0, 0.0, 1.0, 1.0)) for m in COMPRESSION_RUNTIMES}
    categorias_presentes: list[str] = []
    for motor, eje in ejes.items():
        for fila, etiqueta in enumerate(orden):
            entrada = por_etiqueta[etiqueta]
            estilo, _ = COMPRESSION_CATEGORIES[entrada["category"]]
            if entrada["category"] not in categorias_presentes:
                categorias_presentes.append(entrada["category"])
            eje.scatter(
                entrada["runtimes"][motor]["delta"],
                fila,
                marker=estilo.marker,
                s=20,
                facecolor="white" if entrada["control"] else estilo.color,
                edgecolor=estilo.color if entrada["control"] else "black",
                linewidth=0.9 if entrada["control"] else 0.4,
                zorder=3,
            )
        # Segmento de escala en una fila vacía bajo la última, a la derecha.
        fin = -0.01
        eje.plot(
            [fin - suelo, fin],
            [n, n],
            color=FLOOR_COLOR,
            linewidth=1.6,
            solid_capstyle="butt",
            marker="|",
            markersize=4,
            markeredgewidth=0.8,
        )
        eje.text(
            fin - suelo - 0.003,
            n,
            f"suelo de lectura ({style.format_number(suelo, 4)})",
            ha="right",
            va="center",
            fontsize=COMPRESSION_FONTSIZE,
            color=FLOOR_COLOR,
        )
        eje.set_xlim(izquierda_x, 0.0)
        eje.set_ylim(n + 0.7, -0.7)
        eje.xaxis.set_major_locator(MultipleLocator(paso))
        eje.xaxis.set_major_formatter(style.comma_axis_formatter())
        eje.set_yticks(range(n))
        eje.tick_params(labelsize=COMPRESSION_FONTSIZE, length=2.5, pad=2)
        eje.set_xlabel("Δ mAP@50-95 frente al FP32 de su tamaño", fontsize=COMPRESSION_FONTSIZE)
        eje.set_title(COMPRESSION_RUNTIME_TITLES[motor], fontsize=7.5, pad=3)
    ejes[primero].set_yticklabels(orden, fontsize=COMPRESSION_FONTSIZE)
    ejes[segundo].tick_params(labelleft=False)

    # Puesto en el segundo motor fuera del panel; en negrita si difiere del primero.
    transformacion = ejes[segundo].get_yaxis_transform()
    puestos = []
    for fila, etiqueta in enumerate(orden):
        puesto = comparacion["ranks"][segundo][etiqueta]
        puestos.append(
            ejes[segundo].text(
                1.0 + RANK_COLUMN_OFFSET,
                fila,
                str(puesto),
                transform=transformacion,
                ha="left",
                va="center",
                fontsize=COMPRESSION_FONTSIZE,
                fontweight="bold" if puesto != fila + 1 else "normal",
                clip_on=False,
            )
        )
    cabecera = ejes[segundo].text(
        1.0 + RANK_COLUMN_OFFSET,
        1.0,
        f"puesto\nen {COMPRESSION_RUNTIME_TITLES[segundo]}",
        transform=ejes[segundo].transAxes,
        ha="left",
        va="bottom",
        fontsize=COMPRESSION_FONTSIZE,
        linespacing=1.0,
    )

    manejadores = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=COMPRESSION_CATEGORIES[c][0].marker,
            markerfacecolor=COMPRESSION_CATEGORIES[c][0].color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=4.5,
        )
        for c in COMPRESSION_CATEGORIES
        if c in categorias_presentes
    ]
    rotulos = [COMPRESSION_CATEGORIES[c][1] for c in COMPRESSION_CATEGORIES if c in categorias_presentes]
    if any(e["control"] for e in entries):
        manejadores.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#333333",
                markeredgewidth=0.9,
                markersize=4.5,
            )
        )
        rotulos.append("control aleatorio (hueco)")
    leyenda = figura.legend(
        manejadores,
        rotulos,
        loc="upper center",
        ncol=math.ceil(len(manejadores) / 2),
        fontsize=COMPRESSION_FONTSIZE,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )

    # Ejes colocados a mano para que la figura mida 16 cm.
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    ppp = figura.dpi
    ancho_y = max(t.get_window_extent(dibujo).width for t in ejes[primero].get_yticklabels()) / ppp
    ancho_puesto = max(t.get_window_extent(dibujo).width for t in [*puestos, cabecera]) / ppp
    alto_leyenda = leyenda.get_window_extent(dibujo).height / ppp
    alto_cabecera = cabecera.get_window_extent(dibujo).height / ppp
    izquierda = ancho_y + 0.09
    hueco = 0.14
    margen = 0.02
    # La columna de puestos depende del ancho que se está calculando.
    ancho_panel = (
        COMPRESSION_FIGURE_WIDTH - izquierda - hueco - ancho_puesto - margen
    ) / (2.0 + RANK_COLUMN_OFFSET)
    alto_panel = COMPRESSION_ROW_HEIGHT * (n + 1.4)
    abajo = 0.36
    arriba = max(alto_cabecera, 0.16) + 0.06 + alto_leyenda + 0.04
    alto = abajo + alto_panel + arriba
    figura.set_size_inches(COMPRESSION_FIGURE_WIDTH, alto)
    for indice, motor in enumerate(COMPRESSION_RUNTIMES):
        x0 = izquierda + indice * (ancho_panel + hueco)
        ejes[motor].set_position(
            (
                x0 / COMPRESSION_FIGURE_WIDTH,
                abajo / alto,
                ancho_panel / COMPRESSION_FIGURE_WIDTH,
                alto_panel / alto,
            )
        )
    leyenda.set_bbox_to_anchor((0.5, 1.0), transform=figura.transFigure)

    filas = []
    for etiqueta in orden:
        entrada = por_etiqueta[etiqueta]
        filas.append(
            {
                "label": etiqueta,
                "experiment_id": entrada["experiment_id"],
                "category": entrada["category"],
                "control": entrada["control"],
                "model_size": entrada["model_size"],
                "reference": entrada["reference"],
                **{
                    m: {
                        "precision": entrada["runtimes"][m]["precision"],
                        "delta": entrada["runtimes"][m]["delta"],
                        "rank": comparacion["ranks"][m][etiqueta],
                    }
                    for m in COMPRESSION_RUNTIMES
                },
            }
        )
    return figura, {
        "metric": "map_50_95",
        "evaluator": COMPRESSION_EVALUATOR,
        "split": COMPRESSION_SPLIT,
        "runtimes": list(COMPRESSION_RUNTIMES),
        "row_order": primero,
        "floor": suelo,
        "rows": filas,
        "positions_changed": comparacion["positions_changed"],
        "adjacent_within_floor": comparacion["adjacent_within_floor"],
        "order_inversions": comparacion["order_inversions"],
        "inversions_above_floor_in_both": comparacion["inversions_above_floor_in_both"],
    }


def _split_check_family(
    experiment_id: str, precision: str, experiments: Mapping[str, Mapping[str, str]]
) -> str:
    """Familia de una configuración en la figura del 6.7, según su cadena y su precisión."""
    cadena = experiment_chain(experiment_id, experiments)
    tecnicas = [(experiments[e].get("technique", "") or "").strip() for e in cadena]
    podada = any(t in ("pruning", "combined") for t in tecnicas)
    if tecnicas[0] == "baseline" and precision == "fp32":
        return "reference"
    if precision == "int8":
        return "pruning_quantization" if podada else "quantization"
    if precision == "fp32" and podada:
        return "pruning"
    raise ValueError(
        f"{experiment_id} ({precision}, cadena {tecnicas}) no es referencia, cuantización, poda "
        "ni poda + cuantización."
    )


def split_check_entries(
    evidence: Mapping[str, Any],
    experiments: Mapping[str, Mapping[str, str]],
    quality_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Configuraciones de la figura del 6.7 contrastadas con el registro.

    Para cada fila del volcado de evaluación en test busca en el registro su fila de val y de test
    por clave primaria completa, y el mAP@50-95 debe coincidir exactamente.
    """
    tabla = evidence.get("tabla")
    suelo = evidence.get("suelo_map")
    if not tabla or suelo is None:
        raise ValueError("El volcado de evaluación en test no trae 'tabla' o 'suelo_map'.")

    indice: dict[tuple[str, ...], list[Mapping[str, str]]] = {}
    for fila in quality_rows:
        indice.setdefault(_clave_de_tabla(fila, TABLE_PRIMARY_KEYS["quality_metrics"]), []).append(
            fila
        )

    entradas: list[dict[str, Any]] = []
    for fila in tabla:
        motor = fila["runtime"]
        if motor not in SPLIT_CHECK_RUNTIMES:
            continue
        experiment_id, precision = fila["experiment_id"], fila["precision"]
        for particion, dispositivo in (("val", fila["val_source"]), ("test", SPLIT_CHECK_DEVICE)):
            if dispositivo != SPLIT_CHECK_DEVICE:
                raise ValueError(
                    f"{experiment_id} [{motor}/{precision}] tiene su {particion} en "
                    f"'{dispositivo}', no en '{SPLIT_CHECK_DEVICE}'."
                )
            clave = _clave_de_tabla(
                {
                    "experiment_id": experiment_id,
                    "runtime": motor,
                    "precision": precision,
                    "device_id": dispositivo,
                    "evaluator": SPLIT_CHECK_EVALUATOR,
                    "split": particion,
                },
                TABLE_PRIMARY_KEYS["quality_metrics"],
            )
            encontradas = indice.get(clave, [])
            if len(encontradas) != 1:
                raise ValueError(f"{clave} tiene {len(encontradas)} filas en el registro, no una.")
            registrado = float(encontradas[0]["map_50_95"])
            if registrado != float(fila[particion]["map_50_95"]):
                raise ValueError(
                    f"{clave}: el volcado de evaluación en test dice "
                    f"{fila[particion]['map_50_95']} y el registro {registrado}. Regenera el "
                    "volcado antes de la figura."
                )
        entradas.append(
            {
                "label": fila["label"],
                "experiment_id": experiment_id,
                "runtime": motor,
                "precision": precision,
                "family": _split_check_family(experiment_id, precision, experiments),
                "val": float(fila["val"]["map_50_95"]),
                "test": float(fila["test"]["map_50_95"]),
            }
        )

    configuraciones = {}
    for motor in SPLIT_CHECK_RUNTIMES:
        etiquetas = [e["label"] for e in entradas if e["runtime"] == motor]
        if len(etiquetas) != len(set(etiquetas)):
            raise ValueError(f"Hay etiquetas repetidas en {motor}: dos puntos se rotularían igual.")
        configuraciones[motor] = {
            (e["experiment_id"], e["precision"]) for e in entradas if e["runtime"] == motor
        }
    primero, segundo = SPLIT_CHECK_RUNTIMES
    if configuraciones[primero] != configuraciones[segundo]:
        raise ValueError(
            f"{primero} y {segundo} no tienen las mismas configuraciones en el volcado de "
            "evaluación en test."
        )
    return {"decision": evidence.get("decision", ""), "floor": float(suelo), "entries": entradas}


def _label_alignment(dx: float, dy: float) -> tuple[str, str]:
    """Alineación de un rótulo para que crezca alejándose de su punto."""
    ha = "left" if dx > 0 else "right" if dx < 0 else "center"
    va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
    return ha, va


def _box_point_distance(box: Any, x: float, y: float) -> float:
    """Distancia de un punto a una caja, cero si está dentro."""
    return math.hypot(max(box.x0 - x, 0.0, x - box.x1), max(box.y0 - y, 0.0, y - box.y1))


def _nearest_point(box: Any, x: float, y: float) -> tuple[float, float]:
    """Punto de una caja más cercano a otro punto."""
    return min(max(x, box.x0), box.x1), min(max(y, box.y0), box.y1)


def _segment_crossings(
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
    names: Sequence[str],
    boxes: Mapping[str, Any],
    samples: int = 40,
) -> list[str]:
    """Marcadores que cruza una línea guía, muestreando el segmento."""
    cruzados: set[str] = set()
    for k in range(samples + 1):
        x = start[0] + (end[0] - start[0]) * k / samples
        y = start[1] + (end[1] - start[1]) * k / samples
        if math.hypot(x - start[0], y - start[1]) <= radius:
            continue
        cruzados.update(n for n in names if boxes[n].contains(x, y))
    return sorted(cruzados)


def figure_validation_vs_test(
    split_check: Mapping[str, Any] | None,
    floors: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """mAP@50-95 en validación frente a prueba, un panel por motor.

    Los dos paneles comparten límites y la diagonal lleva la banda del suelo. Relleno es FP32 y
    hueco INT8. Cada rótulo de SPLIT_CHECK_LABELS se coloca en el primer desplazamiento de
    LABEL_OFFSETS que no pisa nada y no se confunde con otro punto; si ninguno cumple, se usa el que
    menos falla y se indica en el resumen.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MultipleLocator
    from matplotlib.transforms import Bbox

    if not split_check or not split_check.get("entries"):
        raise ValueError("No está el volcado de evaluación en test.")
    suelo = quality_floor(floors)
    if split_check["floor"] != suelo:
        raise ValueError(
            f"El volcado de evaluación en test usa un suelo de {split_check['floor']} y el paso "
            f"29 de {suelo}."
        )
    entradas = split_check["entries"]
    por_motor = {m: [e for e in entradas if e["runtime"] == m] for m in SPLIT_CHECK_RUNTIMES}
    for motor, etiquetas in SPLIT_CHECK_LABELS.items():
        presentes = {e["label"] for e in por_motor[motor]}
        faltan = [x for x in etiquetas if x not in presentes]
        if faltan:
            raise ValueError(
                f"En {motor} faltan las configuraciones que hay que rotular: {faltan}."
            )

    valores = [e[p] for e in entradas for p in ("val", "test")]
    limite_bajo = math.floor((min(valores) - 0.01) * 100) / 100
    limite_alto = math.ceil((max(valores) + 0.01) * 100) / 100
    diagonal = [limite_bajo, limite_alto]

    style.apply_style()
    figura = plt.figure(figsize=(SPLIT_CHECK_FIGURE_WIDTH, 4.0))
    ejes = {m: figura.add_axes((0.0, 0.0, 1.0, 1.0)) for m in SPLIT_CHECK_RUNTIMES}
    for motor, eje in ejes.items():
        eje.fill_between(
            diagonal,
            [v - suelo for v in diagonal],
            [v + suelo for v in diagonal],
            color=FLOOR_COLOR,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        eje.plot(diagonal, diagonal, color="#555555", linewidth=0.6, zorder=2)
        for entrada in por_motor[motor]:
            estilo, _ = SPLIT_CHECK_FAMILIES[entrada["family"]]
            int8 = entrada["precision"] == "int8"
            eje.scatter(
                entrada["val"],
                entrada["test"],
                marker=estilo.marker,
                s=SPLIT_CHECK_MARKER_AREA,
                facecolor="white" if int8 else estilo.color,
                edgecolor=estilo.color if int8 else "black",
                linewidth=0.9 if int8 else 0.4,
                zorder=3,
            )
        eje.set_xlim(limite_bajo, limite_alto)
        eje.set_ylim(limite_bajo, limite_alto)
        for eje_xy in (eje.xaxis, eje.yaxis):
            eje_xy.set_major_locator(MultipleLocator(0.05))
            eje_xy.set_major_formatter(style.comma_axis_formatter())
        eje.tick_params(labelsize=SPLIT_CHECK_FONTSIZE, length=2.5, pad=2)
        eje.set_xlabel("mAP@50-95 en validación", fontsize=SPLIT_CHECK_FONTSIZE)
        eje.set_title(COMPRESSION_RUNTIME_TITLES[motor], fontsize=7.5, pad=3)
    primero, segundo = SPLIT_CHECK_RUNTIMES
    ejes[primero].set_ylabel("mAP@50-95 en prueba", fontsize=SPLIT_CHECK_FONTSIZE)
    ejes[segundo].tick_params(labelleft=False)

    # Leyenda en dos filas; Matplotlib llena por columnas, así que se intercalan.
    familias = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=estilo.marker,
            markerfacecolor=estilo.color,
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=4.5,
            label=rotulo,
        )
        for estilo, rotulo in SPLIT_CHECK_FAMILIES.values()
    ]
    resto = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="#999999",
            markeredgecolor="black",
            markeredgewidth=0.4,
            markersize=4.5,
            label="FP32 (relleno)",
        ),
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markeredgewidth=0.9,
            markersize=4.5,
            label="INT8 (hueco)",
        ),
        Line2D([], [], color="#555555", linewidth=0.6, label="y = x"),
        Patch(
            facecolor=FLOOR_COLOR,
            alpha=0.18,
            linewidth=0,
            label=f"± suelo de lectura ({style.format_number(suelo, 4)})",
        ),
    ]
    manejadores = [h for par in zip(familias, resto) for h in par]
    leyenda = figura.legend(
        handles=manejadores,
        loc="lower center",
        ncol=len(familias),
        fontsize=SPLIT_CHECK_FONTSIZE,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )

    # Ejes colocados a mano: 16 cm y paneles cuadrados para que se lea la diagonal.
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    ppp = figura.dpi
    marco = ejes[primero].get_window_extent(dibujo)
    ajustado = ejes[primero].get_tightbbox(dibujo)
    izquierda = (marco.x0 - ajustado.x0) / ppp + 0.02
    abajo_ejes = (marco.y0 - ajustado.y0) / ppp
    arriba_ejes = (ajustado.y1 - marco.y1) / ppp
    alto_leyenda = leyenda.get_window_extent(dibujo).height / ppp
    hueco, margen = 0.16, 0.08
    lado = (SPLIT_CHECK_FIGURE_WIDTH - izquierda - hueco - margen) / 2.0
    abajo = 0.02 + alto_leyenda + 0.08 + abajo_ejes
    alto = abajo + lado + arriba_ejes + 0.02
    figura.set_size_inches(SPLIT_CHECK_FIGURE_WIDTH, alto)
    for indice, motor in enumerate(SPLIT_CHECK_RUNTIMES):
        x0 = izquierda + indice * (lado + hueco)
        ejes[motor].set_position(
            (
                x0 / SPLIT_CHECK_FIGURE_WIDTH,
                abajo / alto,
                lado / SPLIT_CHECK_FIGURE_WIDTH,
                lado / alto,
            )
        )
    leyenda.set_bbox_to_anchor((0.5, 0.02 / alto), transform=figura.transFigure)

    # Rótulos con los paneles ya colocados, midiendo en píxeles.
    figura.canvas.draw()
    dibujo = figura.canvas.get_renderer()
    radio = math.sqrt(SPLIT_CHECK_MARKER_AREA) / 2.0 * ppp / 72.0
    rotulos: list[dict[str, Any]] = []
    for motor, eje in ejes.items():
        marco = eje.get_window_extent(dibujo)
        por_etiqueta = {e["label"]: e for e in por_motor[motor]}
        centros = {
            n: tuple(eje.transData.transform((e["val"], e["test"])))
            for n, e in por_etiqueta.items()
        }
        marcadores = {
            n: Bbox.from_extents(x - radio, y - radio, x + radio, y + radio)
            for n, (x, y) in centros.items()
        }
        # El marcador con medio radio de holgura para la guía.
        holgados = {
            n: Bbox.from_extents(x - 1.5 * radio, y - 1.5 * radio, x + 1.5 * radio, y + 1.5 * radio)
            for n, (x, y) in centros.items()
        }
        colocadas: list[Any] = []
        for etiqueta in SPLIT_CHECK_LABELS[motor]:
            punto = (por_etiqueta[etiqueta]["val"], por_etiqueta[etiqueta]["test"])
            propio = centros[etiqueta]
            otros = [n for n in centros if n != etiqueta]
            mejor: tuple[tuple[Any, ...], tuple[float, float], Any, dict[str, Any]] | None = None
            for dx, dy in LABEL_OFFSETS:
                ha, va = _label_alignment(dx, dy)
                prueba = eje.annotate(
                    etiqueta,
                    punto,
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=SPLIT_CHECK_FONTSIZE,
                )
                caja = prueba.get_window_extent(dibujo)
                prueba.remove()
                guia = math.hypot(dx, dy) >= LEADER_MIN_OFFSET
                pisados = sorted(n for n in otros if caja.overlaps(marcadores[n]))
                if guia:
                    # La guía indica de quién es el rótulo si no pasa sobre otro punto.
                    cruzados = _segment_crossings(
                        propio, _nearest_point(caja, *propio), radio, otros, holgados
                    )
                    ambiguo = False
                else:
                    # Sin guía, el punto más cercano al rótulo tiene que ser el suyo.
                    cruzados = []
                    ambiguo = min(_box_point_distance(caja, *centros[n]) for n in otros) < (
                        LABEL_AMBIGUITY_RATIO * _box_point_distance(caja, *propio)
                    )
                fuera = not (
                    caja.x0 >= marco.x0
                    and caja.x1 <= marco.x1
                    and caja.y0 >= marco.y0
                    and caja.y1 <= marco.y1
                )
                choca = any(caja.overlaps(c) for c in colocadas)
                coste = (fuera, choca, len(pisados), len(cruzados), ambiguo)
                if mejor is None or coste < mejor[0]:
                    mejor = (
                        coste,
                        (dx, dy),
                        caja,
                        {
                            "leader": guia,
                            "outside_axes": fuera,
                            "overlaps_label": choca,
                            "overlaps_markers": pisados,
                            "leader_crosses_markers": cruzados,
                            "nearer_to_other_marker": ambiguo,
                        },
                    )
                if coste == (False, False, 0, 0, False):
                    break
            assert mejor is not None
            _, (dx, dy), caja, diagnostico = mejor
            ha, va = _label_alignment(dx, dy)
            eje.annotate(
                etiqueta,
                punto,
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=SPLIT_CHECK_FONTSIZE,
                zorder=4,
                arrowprops=(
                    {
                        "arrowstyle": "-",
                        "linewidth": 0.4,
                        "color": "#555555",
                        "shrinkA": 0,
                        "shrinkB": 2,
                    }
                    if diagnostico["leader"]
                    else None
                ),
            )
            colocadas.append(caja)
            rotulos.append(
                {"runtime": motor, "label": etiqueta, "offset_points": [dx, dy], **diagnostico}
            )

    filas = [
        {
            **{k: e[k] for k in ("label", "experiment_id", "runtime", "precision", "family")},
            "val": e["val"],
            "test": e["test"],
            "test_minus_val": round(e["test"] - e["val"], 6),
            "within_floor": abs(e["test"] - e["val"]) <= suelo,
        }
        for e in entradas
    ]
    return figura, {
        "metric": "map_50_95",
        "evaluator": SPLIT_CHECK_EVALUATOR,
        "device_id": SPLIT_CHECK_DEVICE,
        "decision": split_check.get("decision", ""),
        "runtimes": list(SPLIT_CHECK_RUNTIMES),
        "floor": suelo,
        "axis_limits": [limite_bajo, limite_alto],
        "rows": filas,
        "within_floor": {
            m: sum(f["within_floor"] for f in filas if f["runtime"] == m)
            for m in SPLIT_CHECK_RUNTIMES
        },
        "labels": rotulos,
        "labels_with_overlap": [
            r["runtime"] + "/" + r["label"]
            for r in rotulos
            if r["outside_axes"]
            or r["overlaps_label"]
            or r["overlaps_markers"]
            or r["leader_crosses_markers"]
            or r["nearer_to_other_marker"]
        ],
    }


def _iter_metrics() -> tuple[tuple[str, str, str], ...]:
    """Las tres métricas de la figura comparativa."""
    return (
        ("map_50_95", "mAP@50-95", ""),
        ("latency_p95_ms", "latencia p95", "ms"),
        ("peak_host_memory_mb", "memoria pico", "MiB"),
    )


def save_results_figures(
    analysis: Mapping[str, Any],
    comparison_rows: Sequence[Mapping[str, Any]],
    cost_rows: Sequence[Mapping[str, Any]],
    cost_excluded: Sequence[tuple[str, str]],
    inventory: CurveInventory,
    output_dir: Path | str,
    *,
    error_decomposition: ErrorDecomposition | None,
    instances: ClassInstances | None,
    devices: Mapping[str, tuple[str, str]] | None,
    compression: Sequence[Mapping[str, Any]] | None,
    experiments: Mapping[str, Mapping[str, str]],
    split_check: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Genera las figuras del volcado del paso 29 que se pueden generar y devuelve el manifiesto.

    error_decomposition, instances, devices, compression y split_check son obligatorios aunque
    valgan None; en ese caso la figura se omite con su motivo.
    """
    manifiesto: dict[str, Any] = {"figures": {}, "skipped": {}}
    tareas: tuple[tuple[str, Any], ...] = (
        ("curvas_entrenamiento", lambda: figure_training_curves(inventory)),
        (
            "map_vs_nivel_por_familia",
            lambda: figure_quality_vs_level(analysis["tradeoffs"]["curves"], analysis["floors"]),
        ),
        (
            "comparativa_por_tecnica",
            lambda: figure_technique_comparison(comparison_rows, _iter_metrics()),
        ),
        *(
            (
                f"degradacion_por_clase_{motor}",
                lambda motor=motor: figure_class_degradation(
                    analysis["degradation"]["rows"],
                    instances,
                    devices,
                    motor,
                    compression=compression,
                    experiments=experiments,
                ),
            )
            for motor in CLASS_FIGURE_PLACEMENT
        ),
        (
            "degradacion_por_tamano",
            lambda: figure_size_degradation(analysis["degradation"]["rows"], compression),
        ),
        (
            "tipos_de_error",
            lambda: figure_error_decomposition(error_decomposition, experiments),
        ),
        (
            "coste_frente_a_ganancia",
            lambda: figure_cost_vs_gain(cost_rows, cost_excluded),
        ),
        (
            "perdida_map_por_compresion",
            lambda: figure_compression_loss(compression, analysis["floors"]),
        ),
        (
            "validacion_frente_a_prueba",
            lambda: figure_validation_vs_test(split_check, analysis["floors"]),
        ),
    )

    for clave, construir in tareas:
        try:
            figura, resumen = construir()
        except ValueError as error:
            # Si los datos no dan para la figura, se anota el motivo y se sigue.
            logger.warning("Figura '%s' no generada: %s", clave, error)
            manifiesto["skipped"][clave] = str(error)
            continue
        rutas = style.save_figure(figura, output_dir, *PLACEMENT[clave], clave)
        resumen["files"] = style.manifest_files(rutas)
        manifiesto["figures"][clave] = resumen
        logger.info("Figura '%s' generada.", clave)

    manifiesto["training_curves"] = inventory.to_dict()
    return manifiesto


def technique_labels(techniques: Iterable[str]) -> list[str]:
    """Etiquetas legibles de un conjunto de familias."""
    return [style.style_for(t).label for t in techniques]
