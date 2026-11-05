"""Sensibilidad a la cuantización capa a capa.

Se cuantiza una sola capa cada vez, con el resto en coma flotante, y se mide cuánto cambia la
salida. No captura interacciones entre capas, pero da un orden de sensibilidad útil para decidir qué
excluir. Se mide sobre imágenes reales porque los rangos dependen de la distribución de las
activaciones.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.compression.quantization.qat import (
    QatConfig,
    _importar_quantization,
    _sin_aviso_de_obsolescencia,
)

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    import numpy as np
    from torch import nn

    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)


class SensitivityError(TFMEdgeError):
    """Error al medir la sensibilidad a la cuantización."""


@dataclass
class LayerSensitivity:
    """Sensibilidad de una capa a la cuantización."""

    layer: str
    output_mse: float | str = ""
    output_max_abs: float | str = ""
    box_max_abs: float | str = ""
    score_max_abs: float | str = ""
    num_parameters: int = 0
    in_head: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Devuelve la medida como diccionario serializable."""
        return asdict(self)


@dataclass
class SensitivityReport:
    """Resultado del análisis de sensibilidad."""

    layers: list[LayerSensitivity] = field(default_factory=list)
    num_images: int = 0
    input_resolution: int = 640
    qengine: str = "x86"
    baseline_output_scale: float = 0.0
    notes: str = ""

    @property
    def ranked(self) -> list[LayerSensitivity]:
        """Capas ordenadas de más a menos sensible."""
        medidas = [c for c in self.layers if isinstance(c.output_mse, float)]
        return sorted(medidas, key=lambda c: c.output_mse, reverse=True)

    def most_sensitive(self, n: int = 10) -> list[str]:
        """Las ``n`` capas más sensibles."""
        return [c.layer for c in self.ranked[:n]]

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return {
            "num_images": self.num_images,
            "input_resolution": self.input_resolution,
            "qengine": self.qengine,
            "baseline_output_scale": self.baseline_output_scale,
            "notes": self.notes,
            "layers": [c.to_dict() for c in self.layers],
            "ranked": [c.layer for c in self.ranked],
        }

    def summary(self, top: int = 12) -> str:
        """Resumen legible con las ``top`` capas más sensibles."""
        lineas = [
            f"SENSIBILIDAD A LA CUANTIZACIÓN: {len(self.layers)} capas, "
            f"{self.num_images} imágenes reales a {self.input_resolution} px",
            "",
            f"{'capa':28s} {'MSE salida':>12s} {'max |dif|':>11s} "
            f"{'coord (px)':>11s} {'punt.':>10s} {'params':>10s}  cabeza",
            "-" * 92,
        ]

        def cientifico(valor: float | str, ancho: int) -> str:
            """Formatea un valor que puede estar vacío (no medido) sin romper el resumen."""
            return f"{valor:>{ancho}.3e}" if isinstance(valor, float) else f"{'—':>{ancho}}"

        for capa in self.ranked[:top]:
            lineas.append(
                f"{capa.layer[:28]:28s} {cientifico(capa.output_mse, 12)} "
                f"{cientifico(capa.output_max_abs, 11)} "
                f"{cientifico(capa.box_max_abs, 11)} "
                f"{cientifico(capa.score_max_abs, 10)} {capa.num_parameters:>10,}  "
                f"{'sí' if capa.in_head else ''}"
            )
        fallidas = [c for c in self.layers if c.error]
        if fallidas:
            lineas += ["", f"No medidas ({len(fallidas)}):"]
            for capa in fallidas[:5]:
                lineas.append(f"  {capa.layer}: {capa.error[:60]}")
        lineas += [
            "",
            "Se cuantiza UNA capa cada vez; el resto queda en coma flotante.",
            "AVISO: no captura interacciones entre capas, así que la suma de las",
            "degradaciones individuales NO es la de cuantizarlas todas. Lo que da",
            "es un ORDEN de sensibilidad, que es para lo que se usa.",
        ]
        return "\n".join(lineas)


def _salida_de_referencia(
    modelo: "nn.Module", entradas: Sequence["np.ndarray"]
) -> list["np.ndarray"]:
    """Calcula las salidas del modelo sin cuantizar."""
    import torch

    salidas = []
    with torch.no_grad():
        for entrada in entradas:
            salida = modelo(torch.from_numpy(entrada))
            tensor = salida[0] if isinstance(salida, (list, tuple)) else salida
            salidas.append(tensor.numpy())
    return salidas


def _cuantizar_solo_una_capa(detector: "Detector", capa: str, qengine: str) -> "nn.Module":
    """Prepara un modelo con cuantización simulada solo en la capa indicada, poniendo qconfig a None
    en el resto.
    """
    from tfm_edge.models.surgery import get_submodule

    qao = _importar_quantization()
    modelo = copy.deepcopy(detector.model).train()

    for modulo in modelo.modules():
        modulo.qconfig = None
    try:
        objetivo = get_submodule(modelo, capa)
    except Exception as error:
        raise SensitivityError(
            f"La capa '{capa}' no existe: {type(error).__name__}: {error}",
            "Comprueba las rutas con detector.get_named_conv_layers().",
        ) from error

    with _sin_aviso_de_obsolescencia():
        objetivo.qconfig = qao.get_default_qat_qconfig(qengine)
        try:
            preparado = qao.prepare_qat(modelo, inplace=False)
        except Exception as error:
            raise SensitivityError(
                f"No se ha podido preparar la capa '{capa}': " f"{type(error).__name__}: {error}",
                "Puede que ese tipo de módulo no admita cuantización.",
            ) from error
    return preparado.eval()


def measure_layer_sensitivity(
    detector: "Detector",
    images: Sequence["np.ndarray"],
    layers: Sequence[str] | None = None,
    config: QatConfig | None = None,
) -> SensitivityReport:
    """Mide cuánto cambia la salida al cuantizar cada capa por separado.

    ``images`` son entradas reales preprocesadas de forma ``(1, 3, R, R)``. Por defecto se analizan
    todas las convoluciones.
    """
    import numpy as np
    import torch

    if len(images) < 2:
        raise SensitivityError(
            f"Se han pasado {len(images)} imágenes y hacen falta al menos 2.",
            "Con una sola, el orden de sensibilidad dependería de esa imagen "
            "concreta. Usa imágenes reales del conjunto, nunca ruido.",
        )

    configuracion = config or QatConfig()
    rutas = list(layers) if layers is not None else list(detector.get_named_conv_layers())
    if not rutas:
        raise SensitivityError(
            "No hay ninguna capa que analizar.",
            "Comprueba que el detector tiene convoluciones accesibles.",
        )

    cabeza = detector.get_head_conv_names()
    parametros = {
        nombre: sum(p.numel() for p in modulo.parameters())
        for nombre, modulo in detector.get_named_conv_layers().items()
    }

    original = copy.deepcopy(detector.model).eval()
    referencias = _salida_de_referencia(original, images)
    escala = float(np.mean([np.abs(r).mean() for r in referencias]))

    informe = SensitivityReport(
        num_images=len(images),
        input_resolution=detector.input_resolution,
        qengine=configuracion.qengine,
        baseline_output_scale=round(escala, 6),
        notes=(
            "Se cuantiza una capa cada vez; el resto queda en coma flotante. No "
            "captura interacciones entre capas: la suma de las degradaciones "
            "individuales NO es la de cuantizarlas todas."
        ),
    )

    torch.backends.quantized.engine = configuracion.qengine
    for indice, ruta in enumerate(rutas, start=1):
        medida = LayerSensitivity(
            layer=ruta,
            num_parameters=parametros.get(ruta, 0),
            in_head=ruta in cabeza,
        )
        try:
            preparado = _cuantizar_solo_una_capa(detector, ruta, configuracion.qengine)
            # Calibración: los observadores tienen que ver datos antes de medir.
            with torch.no_grad():
                for entrada in images:
                    preparado(torch.from_numpy(entrada))
            candidatas = _salida_de_referencia(preparado, images)

            errores, maximos, cajas, puntuaciones = [], [], [], []
            for referencia, candidata in zip(referencias, candidatas):
                if referencia.shape != candidata.shape:
                    raise SensitivityError(
                        f"la salida cambia de forma al cuantizar '{ruta}': de {referencia.shape} "
                        f"a {candidata.shape}",
                        "No debería ocurrir; revisa el módulo.",
                    )
                diferencia = np.abs(referencia - candidata)
                errores.append(float(np.mean(diferencia**2)))
                maximos.append(float(diferencia.max()))
                if referencia.ndim == 3 and referencia.shape[1] > 4:
                    cajas.append(float(diferencia[:, :4, :].max()))
                    puntuaciones.append(float(diferencia[:, 4:, :].max()))

            medida.output_mse = float(np.mean(errores))
            medida.output_max_abs = max(maximos)
            medida.box_max_abs = max(cajas) if cajas else ""
            medida.score_max_abs = max(puntuaciones) if puntuaciones else ""
        except Exception as error:
            medida.error = f"{type(error).__name__}: {str(error)[:120]}"
            logger.warning("No se ha podido medir '%s': %s", ruta, medida.error)

        informe.layers.append(medida)
        if indice % 10 == 0 or indice == len(rutas):
            logger.info("Sensibilidad: %d de %d capas medidas", indice, len(rutas))

    medidas = [c for c in informe.layers if isinstance(c.output_mse, float)]
    if not medidas:
        raise SensitivityError(
            "No se ha podido medir la sensibilidad de ninguna capa.",
            "Revisa los motivos en el informe: probablemente el motor de "
            "cuantización no admite estos módulos.",
        )
    logger.info("Capas más sensibles: %s", informe.most_sensitive(5))
    return informe


def suggest_exclusions(
    report: SensitivityReport, num_layers: int = 5, only_head: bool = False
) -> tuple[str, ...]:
    """Propone capas a excluir de la cuantización según lo medido.

    Devuelve las rutas de más a menos sensible, listas para Int8Config.excluded_nodes o
    QatConfig.excluded_modules.
    """
    candidatas = [c for c in report.ranked if not only_head or c.in_head]
    propuestas = tuple(c.layer for c in candidatas[:num_layers])
    logger.info(
        "Se proponen %d capas para excluir: %s. Excluir mejora la precisión pero aumenta tamaño y "
        "latencia.",
        len(propuestas),
        list(propuestas),
    )
    return propuestas
