"""Comprueba que un volcado de análisis cuenta las mismas filas que el registro.

Los pasos 27, 28 y 30 dibujan desde el volcado de los pasos 26 y 29. Si el registro ha cambiado
después, se comparan todos los contadores y se informa de todos los que no cuadran. Solo se miran
recuentos, no valores.
"""

from __future__ import annotations

from typing import Any, Mapping

from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry

__all__ = ["recount_from_registry", "assert_analysis_matches_registry"]


def recount_from_registry(registry: ExperimentRegistry) -> dict[str, Any]:
    """Recalcula desde el registro los contadores que los volcados declaran."""
    experimentos, experimentos_descartados = registry.valid_rows("experiments")
    calidad, calidad_descartada = analysis_quality_rows(registry)
    despliegue, descartes = registry.latency_comparable_rows()
    return {
        "experiment_rows_used": len(experimentos),
        "experiment_rows_discarded_status": experimentos_descartados,
        "quality_rows_used": len(calidad),
        "quality_rows_discarded_status": calidad_descartada,
        "deployment_rows_used": len(despliegue),
        "deployment_rows_discarded": dict(sorted(descartes.items())),
    }


def assert_analysis_matches_registry(
    analysis: Mapping[str, Any], registry: ExperimentRegistry, *, step: str
) -> None:
    """Lanza ValueError si algún contador que declara el volcado no coincide con el registro."""
    declarados = analysis.get("inputs", {})
    if not declarados:
        return

    recontados = recount_from_registry(registry)
    diferencias = [
        f"  - {clave}: el volcado dice {declarados[clave]!r} y el registro da {valor!r}"
        for clave, valor in recontados.items()
        if clave in declarados and declarados[clave] != valor
    ]
    if not diferencias:
        return

    raise ValueError(
        f"El volcado de análisis y el registro no cuentan lo mismo:\n"
        + "\n".join(diferencias)
        + f"\n  Uno de los dos está desactualizado, y con esa diferencia la tabla y "
        f"la figura del mismo apartado contarían campañas distintas. Vuelve a "
        f"ejecutar {step}."
    )
