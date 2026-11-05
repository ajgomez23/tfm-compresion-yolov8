"""Tests del rotulado de ejes con coma decimal."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from tfm_edge.reporting.style import (  # noqa: E402
    MAX_AXIS_DECIMALS,
    axis_decimals,
    comma_axis_formatter,
    format_grouped,
    format_integer,
    format_number,
)


def _rotulos(preparar, eje: str = "y") -> list[str]:
    """Rotula un eje de verdad y devuelve las etiquetas de las marcas visibles."""
    figura, ejes = plt.subplots()
    try:
        preparar(ejes)
        objeto = ejes.yaxis if eje == "y" else ejes.xaxis
        objeto.set_major_formatter(comma_axis_formatter())
        figura.canvas.draw()
        limites = ejes.get_ylim() if eje == "y" else ejes.get_xlim()
        inferior, superior = min(limites), max(limites)
        formateador = objeto.get_major_formatter()
        marcas = [t for t in objeto.get_ticklocs() if inferior <= t <= superior]
        return [formateador(t) for t in marcas]
    finally:
        plt.close(figura)


class TestCifrasDelEje:
    """Cuántos decimales toca, decidido para el eje y no para cada valor."""

    def test_un_eje_entero_no_lleva_decimales(self) -> None:
        assert _rotulos(lambda ax: ax.set_ylim(0, 100)) == ["0", "20", "40", "60", "80", "100"]

    def test_las_marcas_de_un_eje_llevan_todas_las_mismas_cifras(self) -> None:
        """El fallo que motiva este módulo."""
        rotulos = _rotulos(lambda ax: ax.set_ylim(0, 1))

        assert rotulos == ["0,0", "0,2", "0,4", "0,6", "0,8", "1,0"]
        decimales = {len(r.split(",")[1]) if "," in r else 0 for r in rotulos}
        assert len(decimales) == 1

    def test_el_cero_no_es_un_caso_aparte(self) -> None:
        """En un eje de diferencias, «0» entre «-0,05» y «0,05» desentonaba."""
        rotulos = _rotulos(lambda ax: ax.set_ylim(-0.3, 0.1))

        assert "0,00" in rotulos
        assert "0" not in rotulos

    def test_no_recorta_cifras_aunque_las_etiquetas_salgan_distintas(self) -> None:
        """Distinguibles no es lo mismo que fieles."""
        rotulos = _rotulos(lambda ax: (ax.set_ylim(0, 1), ax.set_yticks([0, 0.25, 0.5, 0.75, 1])))

        assert rotulos == ["0,00", "0,25", "0,50", "0,75", "1,00"]


class TestEjeLogaritmico:
    """El eje de coste de la figura 5.6 es logarítmico y abarca dos décadas."""

    def test_las_marcas_de_fuera_de_la_vista_no_deciden_las_cifras(self) -> None:
        """Matplotlib entrega marcas fuera de los límites y hay que filtrarlas."""
        rotulos = _rotulos(lambda ax: (ax.set_yscale("log"), ax.set_ylim(0.005, 1.5)))

        assert rotulos == ["0,01", "0,10", "1,00"]

    def test_un_eje_log_de_enteros_se_queda_en_enteros(self) -> None:
        """El de áreas del capítulo 3 va de 1 a un millón, con millares."""
        rotulos = _rotulos(lambda ax: (ax.set_yscale("log"), ax.set_ylim(1, 1e6)))

        assert rotulos == ["1", "10", "100", "1.000", "10.000", "100.000", "1.000.000"]


class TestAxisDecimals:
    """La regla, aislada de Matplotlib."""

    @pytest.mark.parametrize(
        ("marcas", "esperado"),
        [
            ([0.0, 20.0, 100.0], 0),
            ([0.0, 0.2, 0.4, 1.0], 1),
            ([0.0, 0.25, 0.5], 2),
            ([0.01, 0.1, 1.0], 2),
            ([-0.05, 0.0, 0.05], 2),
            ([0.30000000000000004, 0.6], 1),  # la coma flotante no añade cifras
        ],
    )
    def test_las_menos_cifras_que_no_mienten(self, marcas: list[float], esperado: int) -> None:
        assert axis_decimals(marcas) == esperado

    def test_un_eje_sin_marcas_no_revienta(self) -> None:
        """Puede pasar antes del primer dibujado."""
        assert axis_decimals([]) == 0

    def test_hay_un_tope_y_se_respeta(self) -> None:
        """Con marcas que pidieran más cifras que el tope, se devuelve el tope."""
        assert axis_decimals([0.000001, 0.000002]) == MAX_AXIS_DECIMALS

    def test_llegar_al_tope_deja_rastro_en_el_log(self, caplog: Any) -> None:
        """Ahí las etiquetas quedan redondeadas y eso no puede ser silencioso."""
        with caplog.at_level(logging.WARNING, logger="tfm_edge.reporting.style"):
            axis_decimals([0.000001, 0.000002])

        assert any("exactamente" in r.message for r in caplog.records)

    def test_el_caso_normal_no_avisa_de_nada(self) -> None:
        """El aviso solo sirve si no salta en los ejes de todos los días."""
        import logging as _logging

        registrador = _logging.getLogger("tfm_edge.reporting.style")
        avisos: list[str] = []
        handler = _logging.Handler()
        handler.emit = lambda registro: avisos.append(registro.getMessage())  # type: ignore[method-assign]
        registrador.addHandler(handler)
        try:
            for marcas in ([0.0, 0.2, 1.0], [0, 25, 50], [0.01, 0.1, 1.0], [0.0, -0.025]):
                axis_decimals(marcas)
        finally:
            registrador.removeHandler(handler)

        assert avisos == []

    def test_el_infinito_no_cuenta(self) -> None:
        """Un eje vacío puede traer límites infinitos."""
        assert axis_decimals([float("inf"), 0.5, float("nan")]) == 1


class TestMillares:
    """Los millares van con punto, que es lo que deja libre la coma."""

    def test_el_eje_agrupa_igual_que_los_rotulos_de_las_barras(self) -> None:
        """El eje decía «10000» debajo de una barra que decía «15.850»."""
        rotulos = _rotulos(lambda ax: ax.set_ylim(0, 20000))

        assert "10.000" in rotulos
        assert "10000" not in rotulos
        assert format_integer(15850) == "15.850"

    @pytest.mark.parametrize(
        ("valor", "decimales", "esperado"),
        [
            (999, 0, "999"),
            (1000, 0, "1.000"),
            (1234567, 0, "1.234.567"),
            (1234.5, 1, "1.234,5"),
            (-1234.5, 1, "-1.234,5"),
            (0.05, 2, "0,05"),
        ],
    )
    def test_formatos(self, valor: float, decimales: int, esperado: str) -> None:
        assert format_grouped(valor, decimales) == esperado

    def test_no_hay_cero_con_signo(self) -> None:
        """Un valor negativo que se redondea a cero no se rotula «-0,00»."""
        assert format_grouped(-0.001, 2) == "0,00"
        assert format_grouped(-0.0, 2) == "0,00"


class TestFormatNumber:
    """Formato de números que siguen usando las tablas."""

    def test_pone_coma(self) -> None:
        """La convención decimal del documento entero."""
        assert format_number(0.5, 2) == "0,50"
        assert format_number(36.07, 1) == "36,1"

    def test_no_agrupa_millares(self) -> None:
        """Las tablas no tienen enteros de cuatro cifras y no se tocan."""
        assert format_number(1234.5, 1) == "1234,5"


class TestEstiloDePodaMasQat:
    """El QAT sobre un podado se separa del otro QAT solo por el marcador."""

    def test_lleva_el_color_de_qat(self) -> None:
        """En el registro es QAT, y el color lo dice."""
        from tfm_edge.reporting.style import PRUNING_QAT_STYLE, TECHNIQUE_STYLES

        assert PRUNING_QAT_STYLE.color == TECHNIQUE_STYLES["qat"].color

    def test_su_marcador_no_lo_usa_ninguna_tecnica(self) -> None:
        """En escala de grises, un marcador repetido lo confundiría con otra familia."""
        from tfm_edge.reporting.style import PRUNING_QAT_STYLE, TECHNIQUE_STYLES

        assert PRUNING_QAT_STYLE.marker not in {e.marker for e in TECHNIQUE_STYLES.values()}

    def test_no_es_una_tecnica_del_registro(self) -> None:
        from tfm_edge.reporting.style import PRUNING_QAT_STYLE, TECHNIQUE_STYLES

        assert PRUNING_QAT_STYLE not in TECHNIQUE_STYLES.values()
