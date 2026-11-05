"""Cuantización posterior al entrenamiento (PTQ).

PTQ es la materialización INT8 nativa de cada runtime (export/quantize.py) a partir de los pesos del
baseline. QAT usa exactamente el mismo camino, cambiando solo los pesos de partida; por eso las dos
ramas llaman a materialize_int8_branch y solo difieren en ``source_kind``. Este módulo añade el
experiment_id, la complejidad del artefacto y la fila del registro.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir
from tfm_edge.export import exporter, quantize

if TYPE_CHECKING:  # pragma: no cover - solo para anotaciones
    from tfm_edge.models.detector import Detector

logger = get_logger(__name__)


class PtqError(TFMEdgeError):
    """Error al aplicar cuantización posterior al entrenamiento."""


@dataclass
class Int8BranchResult:
    """Resultado de materializar INT8 en una rama, por runtime. ``source_kind`` distingue PTQ de
    QAT.
    """

    source_kind: str = "baseline"
    variant: str = "ptq-int8"
    fp32_exports: dict[str, Any] = field(default_factory=dict)
    int8_artifacts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    calibration_images: int = 0
    excluded_nodes: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def runtimes_ok(self) -> list[str]:
        """Runtimes cuyo artefacto INT8 se generó correctamente."""
        return [r for r, a in self.int8_artifacts.items() if a.get("status") == "ok"]

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el resultado como diccionario serializable."""
        return asdict(self)

    def summary(self) -> str:
        """Genera un resumen legible."""
        lineas = [
            f"INT8: variante '{self.variant}', partiendo de " f"'{self.source_kind}'",
            f"  calibración: {self.calibration_images} imágenes",
            f"  capas excluidas: {len(self.excluded_nodes)}",
            "",
            f"{'runtime':14s} {'estado':14s} {'tamaño INT8':>13s} {'reducción':>10s}",
            "-" * 56,
        ]
        for runtime, artefacto in self.int8_artifacts.items():
            tamano = artefacto.get("disk_bytes")
            texto = f"{tamano / 2**20:.2f} MiB" if isinstance(tamano, int) else "—"
            lineas.append(
                f"{runtime:14s} {artefacto.get('status', '?'):14s} {texto:>13s} "
                f"{str(artefacto.get('size_reduction', '—')):>10s}"
            )
        return "\n".join(lineas)


def materialize_int8_branch(
    detector: "Detector",
    output_dir: Path | str,
    name: str,
    calibration_images: Sequence[Path | str],
    runtimes: Sequence[str] = ("onnxruntime", "openvino"),
    config: quantize.Int8Config | None = None,
    variant: str = "ptq-int8",
    source_kind: str = "baseline",
) -> Int8BranchResult:
    """Materializa INT8 desde un detector FP32 en los runtimes indicados.

    La llaman PTQ (``source_kind="baseline"``) y QAT (``source_kind="qat"``) con las mismas imágenes
    de calibración y la misma configuración. Se exporta una sola vez a FP32 y de ahí parten todos
    los runtimes. En la rama QAT el detector llega ya sin nodos de simulación.
    """
    if not calibration_images:
        raise PtqError(
            "No se han indicado imágenes de calibración.",
            "La cuantización estática necesita imágenes reales de calibración. Genera el "
            "conjunto con data/calibration_set.py y usa el mismo en las dos ramas.",
        )

    configuracion = config or quantize.Int8Config()
    directorio = ensure_dir(output_dir)
    resultado = Int8BranchResult(
        source_kind=source_kind,
        variant=variant,
        config=configuracion.to_dict(),
        calibration_images=len(calibration_images),
        excluded_nodes=list(configuracion.excluded_nodes),
    )

    # 1. Una sola exportación FP32 de la que parten todos los runtimes.
    fp32_onnx = exporter.export_onnx(detector, directorio, f"{name}_fp32")
    resultado.fp32_exports["onnx"] = fp32_onnx.to_dict()
    if not fp32_onnx.ok:
        raise PtqError(
            f"La exportación FP32 de partida ha fallado: {fp32_onnx.notes}",
            "Sin un FP32 verificado no se puede materializar INT8.",
        )

    origenes: dict[str, str] = {"onnxruntime": fp32_onnx.path}
    if "openvino" in runtimes:
        fp32_ov = exporter.export_openvino(
            detector, directorio, f"{name}_fp32", onnx_path=fp32_onnx.path
        )
        resultado.fp32_exports["openvino"] = fp32_ov.to_dict()
        if fp32_ov.ok:
            origenes["openvino"] = fp32_ov.path
        else:
            logger.warning("No hay FP32 de OpenVINO del que partir: %s", fp32_ov.notes)

    # 2. Materialización INT8 por el camino común con QAT.
    for runtime in runtimes:
        if runtime not in origenes:
            resultado.int8_artifacts[runtime] = {
                "status": "no_disponible",
                "notes": f"no hay artefacto FP32 de partida para '{runtime}'",
            }
            continue
        destino = (
            directorio / f"{name}_int8_{runtime}.onnx"
            if runtime == "onnxruntime"
            else directorio / f"{name}_int8_openvino" / f"{name}_int8.xml"
        )
        artefacto = quantize.materialize_int8(
            origenes[runtime],
            destino,
            calibration_images,
            runtime=runtime,
            config=configuracion,
            input_resolution=detector.input_resolution,
            # Lo único que distingue una rama de la otra.
            source_kind=source_kind,
        )
        resultado.int8_artifacts[runtime] = artefacto.to_dict()

    logger.info(
        "Materialización INT8 desde '%s' completada: %d de %d runtimes con artefacto",
        source_kind,
        len(resultado.runtimes_ok),
        len(runtimes),
    )
    return resultado


def resolve_d018_pair(
    ptq_report: dict[str, Any],
    qat_experiment_id: str,
    experiments: Sequence[dict[str, str]],
) -> frozenset[str]:
    """Decide qué pareja de procedencias se puede comparar, consultando el registro.

    Con un informe ``baseline``, el QAT indicado debe partir de un baseline. Con un informe
    ``pruned``, el QAT debe partir del mismo modelo podado que el informe. Cualquier otra
    procedencia se rechaza.
    """
    procedencia = ptq_report.get("source_kind")
    por_id = {fila["experiment_id"]: fila for fila in experiments}

    def comprobar_el_informe_contra_su_fila(tecnica_esperada: str) -> None:
        """Comprueba que el padre que declara el informe coincide con el de su fila en el
        registro."""
        propio = (ptq_report.get("experiment_id") or "").strip()
        if not propio:
            # Los informes antiguos de combined/ no llevan experiment_id; en ese caso se avisa de
            # que no se ha podido contrastar el padre.
            logger.warning(
                "El informe PTQ no declara 'experiment_id': la pareja se autoriza sin "
                "contrastar el padre con su fila. Regenera el informe para comprobarlo."
            )
            return

        fila = por_id.get(propio)
        if fila is None:
            raise PtqError(
                f"El informe dice ser de '{propio}' y ese experimento no está en "
                f"experiments.csv.",
                "Un informe cuyo experimento no está inscrito no puede autorizar ninguna pareja.",
            )
        if fila.get("technique") != tecnica_esperada:
            raise PtqError(
                f"El informe dice ser de '{propio}', cuya fila declara "
                f"technique='{fila.get('technique')}' y no '{tecnica_esperada}'.",
                "El informe y su fila describen experimentos distintos: uno de los "
                "dos está mal.",
            )
        declarado = (ptq_report.get("parent_experiment_id") or "").strip()
        inscrito = (fila.get("parent_experiment_id") or "").strip()
        if declarado != inscrito:
            raise PtqError(
                f"El informe de '{propio}' dice partir de '{declarado}' y su fila "
                f"del registro dice '{inscrito}'.",
                "El informe y el registro declaran padres distintos.",
            )

    def fila_qat() -> dict[str, str]:
        """La fila del experimento QAT, comprobada."""
        fila = por_id.get(qat_experiment_id)
        if fila is None:
            raise PtqError(
                f"El experimento '{qat_experiment_id}' no está en experiments.csv.",
                "La pareja se autoriza contra el registro: importa antes el afinado "
                "(14_import_colab_registry.py) o corrige --experiment-id.",
            )
        if fila.get("technique") != "qat":
            raise PtqError(
                f"'{qat_experiment_id}' tiene technique='{fila.get('technique')}', no 'qat'.",
                "--experiment-id tiene que ser el afinado QAT cuyos pesos se materializan.",
            )
        return fila

    if procedencia == "baseline":
        comprobar_el_informe_contra_su_fila("ptq")
        if qat_experiment_id:
            padre = fila_qat().get("parent_experiment_id", "")
            tecnica_padre = por_id.get(padre, {}).get("technique")
            if tecnica_padre != "baseline":
                raise PtqError(
                    f"El QAT '{qat_experiment_id}' parte de '{padre}' "
                    f"(technique='{tecnica_padre}'), no de un baseline, y el informe es la PTQ "
                    "del baseline.",
                    "La referencia de un QAT sobre un modelo podado es la PTQ de ese mismo modelo.",
                )
        return quantize.D018_SOURCE_KINDS

    if procedencia == "pruned":
        comprobar_el_informe_contra_su_fila("combined")
        if not qat_experiment_id:
            raise PtqError(
                "El informe es la PTQ de un podado y no se ha dado --experiment-id.",
                "La pareja (pruned, qat) exige indicar el experimento QAT para comprobar el "
                "modelo podado de partida.",
            )
        padre_informe = ptq_report.get("parent_experiment_id") or ""
        if not padre_informe:
            raise PtqError(
                "El informe de la PTQ del podado no declara 'parent_experiment_id'.",
                "Sin él no hay forma de saber de qué podado parte.",
            )
        padre_qat = fila_qat().get("parent_experiment_id", "")
        if padre_qat != padre_informe:
            raise PtqError(
                f"La PTQ parte de '{padre_informe}' y el QAT '{qat_experiment_id}' de "
                f"'{padre_qat}'.",
                "Usa el informe de la PTQ de ese modelo podado.",
            )
        tecnica_padre = por_id.get(padre_qat, {}).get("technique")
        if tecnica_padre != "pruning":
            raise PtqError(
                f"El padre común '{padre_qat}' tiene technique='{tecnica_padre}', no 'pruning'.",
                "La pareja (pruned, qat) solo se usa para la rama de poda seguida de QAT.",
            )
        return quantize.PRUNED_QAT_SOURCE_KINDS

    raise PtqError(
        f"El informe declara source_kind='{procedencia}'.",
        "La referencia tiene que ser una rama PTQ: 'baseline', o 'pruned' para un QAT sobre "
        "podado.",
    )


def compare_with_qat(
    ptq: Int8BranchResult,
    qat_artifacts: dict[str, Any],
    runtime: str,
    admitted_pair: frozenset[str] = quantize.D018_SOURCE_KINDS,
) -> dict:
    """Comprueba que las ramas PTQ y QAT siguieron el mismo camino de materialización en un runtime.

    ``admitted_pair`` es por defecto (baseline, qat); la pareja para modelos podados sale de
    resolve_d018_pair.
    """
    if runtime not in ptq.int8_artifacts:
        raise PtqError(
            f"La rama PTQ no tiene artefacto para '{runtime}'.",
            "Materialízalo antes de comparar.",
        )
    if runtime not in qat_artifacts:
        raise PtqError(
            f"La rama QAT no tiene artefacto para '{runtime}'.",
            "Materialízalo con la MISMA Int8Config y el MISMO conjunto de "
            "calibración que la rama PTQ.",
        )

    def reconstruir(datos: dict[str, Any]) -> quantize.QuantizationResult:
        """Rehace el resultado desde su forma serializada."""
        return quantize.QuantizationResult(
            **{k: v for k, v in datos.items() if k in quantize.QuantizationResult.__annotations__}
        )

    return quantize.assert_same_materialization_path(
        reconstruir(ptq.int8_artifacts[runtime]),
        reconstruir(qat_artifacts[runtime]),
        admitted_pair=admitted_pair,
    )
