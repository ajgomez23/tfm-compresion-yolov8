"""Ningún texto del código repite a mano cuántas filas tiene el registro."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]

#: Un número delante de «filas», «huellas» o «entornos», en cifra o en las
#: formas en letra que llegaron a usarse para estos recuentos.
PATRON_AMPLIO = re.compile(
    r"\b(\d+|quince|diecinueve|veintitr[eé]s|treinta\s+y\s+un[ao]?|cuarenta\s+y\s+seis|"
    r"cincuenta)\s+(filas|huellas|entornos)\b",
    re.IGNORECASE,
)

#: Solo la forma exacta del fallo. pareto.py lleva otros dos recuentos que el patrón amplio no
#: detecta y que se dejan como están.
PATRON_DESPLIEGUE = re.compile(r"\b\d+\s+(filas\s+de\s+despliegue|parejas)\b", re.IGNORECASE)

FICHEROS = {
    "configs/benchmark/workstation_gpu.yaml": PATRON_AMPLIO,
    "scripts/07_benchmark.py": PATRON_AMPLIO,
    "scripts/26_analyze_pareto.py": PATRON_AMPLIO,
    "scripts/31_register_export_cost.py": PATRON_AMPLIO,
    "src/tfm_edge/export/cost.py": PATRON_AMPLIO,
    "src/tfm_edge/analysis/pareto.py": PATRON_DESPLIEGUE,
}


@pytest.mark.parametrize("relativa", sorted(FICHEROS))
def test_ningun_texto_escribe_a_mano_cuantas_filas_hay(relativa: str) -> None:
    """El fichero no repite a mano un recuento de filas del registro."""
    texto = (RAIZ / relativa).read_text(encoding="utf-8")
    encontrado = FICHEROS[relativa].search(texto)
    assert encontrado is None, encontrado


@pytest.mark.parametrize(
    "frase",
    [
        "consta en 15 filas de environments.csv",
        "en las 15 filas de environments.csv de esta maquina",
        "medido (15 filas de environments.csv)",
        "vacío en las 31 filas de despliegue",
        "era el paso de «31 filas» a",
        "vacías en las 31 filas. La consecuencia",
        "tensorrt_version vacío en las 23 filas",
        "instalado en 6 de 23 entornos",
        "consta en quince filas",
    ],
)
def test_el_patron_amplio_reconoce_las_formas_que_tuvo_el_fallo(frase: str) -> None:
    """Un patrón que no reconoce el fallo original no guarda nada."""
    assert PATRON_AMPLIO.search(frase)


@pytest.mark.parametrize(
    "frase",
    [
        "incomparables dos filas que parecerían comparables",
        "las cuatro filas que hay se inscribieron con un guion",
        "parte una sesión en dos huellas",
    ],
)
def test_el_patron_amplio_deja_pasar_frases_sin_recuento_del_registro(frase: str) -> None:
    """Frases legítimas de esos mismos ficheros, que no fijan un recuento."""
    assert not PATRON_AMPLIO.search(frase)


def test_el_patron_de_pareto_reconoce_el_fallo_y_respeta_lo_historico() -> None:
    """Coge «31 filas de despliegue, 31 parejas» y no los dos recuentos que se dejan."""
    assert PATRON_DESPLIEGUE.search("uno a uno: 31 filas de despliegue, 31 parejas")
    assert not PATRON_DESPLIEGUE.search("en una sola sesión y las 46 filas comparten huella")
    assert not PATRON_DESPLIEGUE.search("Hoy las 50 filas medidas comparten")
