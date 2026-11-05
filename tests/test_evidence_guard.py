"""Tests de la protección de volcados y del inventario de productores."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tfm_edge.common.evidence import (
    BANDERA,
    DUMP_GUARD_BY_SCRIPT,
    OVERWRITE_DUMP_FLAG,
    refuse_to_overwrite_dump,
)
from tfm_edge.common.paths import get_project_paths

#: Un script "toca results/raw" si nombra el directorio de alguna de las dos
#: formas en que se escribe en el proyecto.
TOCA_RAW = re.compile(r"results_raw|results\s*/\s*[\"']raw[\"']")


def _scripts_que_tocan_raw() -> dict[str, str]:
    """Los scripts que mencionan ``results/raw`` y su código fuente."""
    directorio = get_project_paths().root / "scripts"
    encontrados = {}
    for ruta in sorted(directorio.glob("*.py")):
        texto = ruta.read_text(encoding="utf-8")
        if TOCA_RAW.search(texto):
            encontrados[ruta.name] = texto
    return encontrados


class TestLaGuardaSeNiegaCuandoToca:
    """Los tres casos de la función, que son todos los que tiene."""

    def test_un_destino_que_existe_se_niega(self, tmp_path: Path) -> None:
        destino = tmp_path / "medida.json"
        destino.write_text("{}", encoding="utf-8")

        with pytest.raises(FileExistsError, match="ya existe"):
            refuse_to_overwrite_dump(destino, overwrite=False, what="el volcado")

    def test_un_destino_que_no_existe_pasa(self, tmp_path: Path) -> None:
        refuse_to_overwrite_dump(tmp_path / "nuevo.json", overwrite=False, what="el volcado")

    def test_con_la_bandera_se_permite(self, tmp_path: Path) -> None:
        destino = tmp_path / "medida.json"
        destino.write_text("{}", encoding="utf-8")

        refuse_to_overwrite_dump(destino, overwrite=True, what="el volcado")

    def test_el_mensaje_dice_que_hacer(self, tmp_path: Path) -> None:
        """Una excepción que no dice qué puede hacer el usuario es media excepción."""
        destino = tmp_path / "medida.json"
        destino.write_text("{}", encoding="utf-8")

        with pytest.raises(FileExistsError) as error:
            refuse_to_overwrite_dump(destino, overwrite=False, what="el volcado")

        assert OVERWRITE_DUMP_FLAG in str(error.value)
        assert str(destino) in str(error.value)


class TestElInventarioCubreLoQueHay:
    """La parte que envejece sola si no se comprueba."""

    def test_todo_script_que_toca_raw_esta_clasificado(self) -> None:
        sin_clasificar = set(_scripts_que_tocan_raw()) - set(DUMP_GUARD_BY_SCRIPT)

        assert not sin_clasificar, (
            "Estos scripts nombran results/raw y no están en DUMP_GUARD_BY_SCRIPT. "
            "Clasifícalos: o se niegan con la bandera, o su salida es derivada, o "
            f"no escriben ahí dentro. {sorted(sin_clasificar)}"
        )

    def test_no_se_clasifica_un_script_que_no_existe(self) -> None:
        """Un script retirado deja su entrada, y la entrada miente en silencio."""
        fantasmas = set(DUMP_GUARD_BY_SCRIPT) - set(_scripts_que_tocan_raw())

        assert (
            not fantasmas
        ), f"Estos están clasificados y no escriben en results/raw: {sorted(fantasmas)}"

    def test_el_que_dice_tener_bandera_la_tiene(self) -> None:
        fuentes = _scripts_que_tocan_raw()
        mienten = [
            nombre
            for nombre, motivo in DUMP_GUARD_BY_SCRIPT.items()
            if motivo == BANDERA and "add_overwrite_dump_argument" not in fuentes[nombre]
        ]

        assert (
            not mienten
        ), f"Dicen negarse con {OVERWRITE_DUMP_FLAG} y no la declaran: {sorted(mienten)}"

    def test_el_que_no_dice_tener_bandera_no_la_tiene(self) -> None:
        """El control simétrico, que es el que detecta un inventario a medias."""
        fuentes = _scripts_que_tocan_raw()
        mienten = [
            nombre
            for nombre, motivo in DUMP_GUARD_BY_SCRIPT.items()
            if motivo != BANDERA and "add_overwrite_dump_argument" in fuentes[nombre]
        ]

        assert not mienten, f"Declaran la bandera y el inventario dice otra cosa: {sorted(mienten)}"

    def test_cada_motivo_dice_algo(self) -> None:
        for nombre, motivo in DUMP_GUARD_BY_SCRIPT.items():
            assert len(motivo) > 25, f"'{nombre}' tiene un motivo demasiado corto: {motivo!r}"

    def test_la_comprobacion_ve_scripts_de_verdad(self) -> None:
        encontrados = _scripts_que_tocan_raw()

        assert len(encontrados) >= 20, f"Solo se han visto {len(encontrados)} scripts."
        assert "07_benchmark.py" in encontrados
