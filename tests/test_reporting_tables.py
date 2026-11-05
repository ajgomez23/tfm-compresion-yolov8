"""La exportación de tablas: lo que se redondea, lo que se marca y lo que no."""

from __future__ import annotations

import csv
import unicodedata

import pytest

from tfm_edge.reporting import tables as t


def _columnas() -> tuple[t.Column, ...]:
    """Tres columnas típicas: texto, calidad y latencia con suelo."""
    return (
        t.text_column("experiment_id", "Experimento"),
        t.quality_column("map_50", "mAP@50"),
        t.latency_column("latency_p50_ms", "latencia p50", floor=1.2, floor_note="suelo de prueba"),
    )


def _tabla(rows: tuple[dict, ...], columnas: tuple[t.Column, ...] | None = None) -> t.Table:
    """Monta una tabla de prueba."""
    return t.Table(
        name="prueba",
        caption="Tabla de prueba",
        chapter=6,
        section=4,
        columns=columnas or _columnas(),
        rows=rows,
    )


class TestRedondeoYUnidades:
    """Redondeo a cuatro y dos decimales y unidad en la cabecera (sección 15.2)."""

    def test_calidad_cuatro_decimales_y_latencia_dos(self) -> None:
        """Los redondeos son los de la especificación, con coma decimal."""
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.578213, "latency_p50_ms": 138.4},))
        tex = t.render_latex(tabla)
        assert "0,5782" in tex
        assert "138,40" in tex

    def test_la_unidad_va_en_la_cabecera(self) -> None:
        """Una latencia sin unidad no se puede citar."""
        assert t.latency_column("latency_p50_ms", "latencia p50").header_text == (
            "latencia p50 (ms)"
        )
        assert t.text_column("experiment_id", "Experimento").header_text == "Experimento"

    def test_el_csv_no_redondea(self, tmp_path) -> None:
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5782134, "latency_p50_ms": 138.4},))
        ruta = t.write_csv(tabla, tmp_path / "x.csv")
        filas = list(csv.DictReader(ruta.read_text(encoding="utf-8").splitlines()))
        assert filas[0]["map_50"] == "0.5782134"


class TestAusentes:
    """Un valor que no se midió es una celda vacía, nunca un cero."""

    def test_la_celda_vacia_se_queda_vacia(self) -> None:
        """Ni cero, ni guion, ni «NaN»."""
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5, "latency_p50_ms": ""},))
        tex = t.render_latex(tabla)
        cuerpo = [linea for linea in tex.splitlines() if linea.strip().startswith("a &")]
        assert cuerpo == ["    a & 0,5000 &  \\\\"], cuerpo
        assert "textbf" not in tex, "con una sola fila no hay mejor valor que marcar"

    def test_un_valor_ilegible_es_un_error_y_no_una_celda_vacia(self) -> None:
        """Silenciarlo lo convertiría en un hueco indistinguible de «no se midió»."""
        tabla = _tabla(({"experiment_id": "a", "map_50": "no disponible"},))
        with pytest.raises(ValueError, match="no es un número"):
            t.render_latex(tabla)

    def test_nan_tambien_es_un_error(self) -> None:
        tabla = _tabla(({"experiment_id": "a", "map_50": float("nan")},))
        with pytest.raises(ValueError, match="NaN"):
            t.render_latex(tabla)


class TestElMejorValor:
    """La negrita es una afirmación, y tiene que sostenerse."""

    def test_marca_el_maximo_y_el_minimo(self) -> None:
        """Cada columna en su dirección."""
        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.50, "latency_p50_ms": 100.0},
                {"experiment_id": "b", "map_50": 0.60, "latency_p50_ms": 200.0},
            )
        )
        marcas, notas = t.best_marks(tabla)
        assert marcas["map_50"] == frozenset({1})
        assert marcas["latency_p50_ms"] == frozenset({0})
        assert notas == ()

    def test_el_suelo_marca_el_conjunto_indistinguible(self) -> None:
        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.50, "latency_p50_ms": 100.0},
                {"experiment_id": "b", "map_50": 0.51, "latency_p50_ms": 100.8},
                {"experiment_id": "c", "map_50": 0.52, "latency_p50_ms": 140.0},
            )
        )
        marcas, _ = t.best_marks(tabla)
        assert marcas["latency_p50_ms"] == frozenset({0, 1})

    def test_sin_suelo_solo_empatan_los_exactos(self) -> None:
        """Sin resolución medida, la tabla no puede tolerar nada."""
        columna = t.quality_column("map_50", "mAP@50")
        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.5000},
                {"experiment_id": "b", "map_50": 0.5000},
                {"experiment_id": "c", "map_50": 0.4999},
            ),
            columnas=(t.text_column("experiment_id", "Experimento"), columna),
        )
        marcas, _ = t.best_marks(tabla)
        assert marcas["map_50"] == frozenset({0, 1})

    def test_si_no_se_distingue_nada_no_se_marca_nada_y_se_dice(self) -> None:
        """Una columna entera en negrita se lee como un fallo de formato."""
        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.50, "latency_p50_ms": 100.0},
                {"experiment_id": "b", "map_50": 0.51, "latency_p50_ms": 100.5},
                {"experiment_id": "c", "map_50": 0.52, "latency_p50_ms": 100.9},
            )
        )
        marcas, notas = t.best_marks(tabla)
        assert "latency_p50_ms" not in marcas
        assert any("ninguna diferencia supera la resolución" in nota for nota in notas)
        assert any("suelo de prueba" in nota for nota in notas)
        assert "latency_p50_ms" in t.table_summary(tabla)["columns_without_distinction"]

    def test_una_columna_con_una_sola_medida_no_se_marca(self) -> None:
        """Con un único valor medido no hay comparación, así que no hay mejor."""
        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.50, "latency_p50_ms": 100.0},
                {"experiment_id": "b", "map_50": 0.51, "latency_p50_ms": ""},
            )
        )
        marcas, notas = t.best_marks(tabla)
        assert "latency_p50_ms" not in marcas
        assert notas == ()

    def test_el_suelo_relativo_se_calcula_como_en_la_frontera(self) -> None:
        """El umbral relativo va sobre el menor de los dos valores."""
        from tfm_edge.analysis import floors as mf
        from tfm_edge.analysis import pareto as mp

        columna = t.latency_column(
            "latency_p50_ms",
            "latencia p50",
            floor_relative=0.093,
            floor_note="suelo relativo de prueba",
        )
        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.50, "latency_p50_ms": 100.0},
                {"experiment_id": "b", "map_50": 0.51, "latency_p50_ms": 109.0},
                {"experiment_id": "c", "map_50": 0.52, "latency_p50_ms": 140.0},
            ),
            columnas=(t.text_column("experiment_id", "Experimento"), columna),
        )
        marcas, _ = t.best_marks(tabla)

        suelos = mf.FloorSet(
            floors={
                "latency_p50_ms": mf.AxisFloor(
                    axis="latency_p50_ms",
                    kind="relativo",
                    same_session=0.093,
                    cross_session=0.093,
                    provenance="el mismo del test",
                )
            }
        )
        objetivo = mp.Objective("latencia p50", "latency_p50_ms", "deployment", "min")

        def punto(nombre: str, valor: float) -> mp.Point:
            return mp.Point(
                experiment_id=f"s_baseline_default_{nombre * 8}",
                runtime="a",
                precision="fp32",
                device_id="dev",
                environment_id="e1",
                timestamp="2026-08-27T00:00:00",
                values={"latency_p50_ms": valor},
            )

        mejor = punto("a", 100.0)
        for indice, valor in ((1, 109.0), (2, 140.0)):
            veredicto = mp.compare(mejor, punto("b", valor), objetivo, suelos, True).verdict
            assert (indice in marcas["latency_p50_ms"]) == (
                veredicto == "indistinguible"
            ), f"la tabla y pareto.compare discrepan sobre {valor} ms"
        assert 0 in marcas["latency_p50_ms"]

    def test_no_se_admiten_los_dos_suelos_a_la_vez(self) -> None:
        """Con los dos declarados no se sabría cuál se aplica."""
        with pytest.raises(ValueError, match="absoluto y relativo a la vez"):
            t.Column(
                key="x",
                header="X",
                decimals=2,
                better="min",
                floor=1.0,
                floor_relative=0.1,
                floor_note="da igual",
            )

    def test_el_suelo_relativo_tambien_exige_procedencia(self) -> None:
        """Un porcentaje inventado se cita igual de mal que un absoluto."""
        with pytest.raises(ValueError, match="sin floor_note"):
            t.Column(key="x", header="X", decimals=2, better="min", floor_relative=0.1)


class TestElSueloLoDecideCadaPareja:
    """El suelo se decide comparando las huellas de las dos filas."""

    @staticmethod
    def _tabla_sesiones(huellas: tuple[str, str, str], environment_key: str | None) -> t.Table:
        """Tres latencias: la mejor, una a un 20 % y otra a un 60 %."""
        columna = t.latency_column(
            "latency_p95_ms",
            "latencia p95",
            floor_relative=0.47,
            floor_same_session=0.15,
            floor_note="suelos de prueba",
        )
        return t.Table(
            name="prueba",
            caption="Tabla de prueba",
            chapter=6,
            section=4,
            columns=(t.text_column("experiment_id", "Experimento"), columna),
            rows=tuple(
                {"experiment_id": nombre, "latency_p95_ms": valor, "environment_id": huella}
                for nombre, valor, huella in zip("abc", (100.0, 120.0, 160.0), huellas)
            ),
            environment_key=environment_key,
        )

    @pytest.mark.parametrize(
        "huellas, environment_key, esperado",
        [
            (("e1", "e1", "e1"), "environment_id", {0}),
            (("e1", "e2", "e1"), "environment_id", {0, 1}),
            (("e1", "", "e1"), "environment_id", {0, 1}),
            (("e1", "e1", "e1"), None, {0, 1}),
        ],
        ids=["misma-sesion", "otra-sesion", "huella-vacia", "sin-clave-de-huella"],
    )
    def test_misma_sesion_suelo_pequeno_y_si_no_el_grande(
        self, huellas, environment_key, esperado
    ) -> None:
        """Sin huella, o sin saber dónde está, se trata como sesión distinta."""
        marcas, _ = t.best_marks(self._tabla_sesiones(huellas, environment_key))
        assert marcas["latency_p95_ms"] == frozenset(esperado)

    def test_con_varios_mejores_basta_no_distinguirse_de_uno(self) -> None:
        columna = t.latency_column(
            "latency_p95_ms",
            "latencia p95",
            floor_relative=0.47,
            floor_same_session=0.15,
            floor_note="suelos por sesión",
        )
        tabla = t.Table(
            name="prueba",
            caption="Tabla de prueba",
            chapter=6,
            section=4,
            columns=(t.text_column("experiment_id", "Experimento"), columna),
            rows=(
                {"experiment_id": "a", "latency_p95_ms": 100.0, "environment_id": "e1"},
                {"experiment_id": "b", "latency_p95_ms": 100.0, "environment_id": "e2"},
                {"experiment_id": "c", "latency_p95_ms": 120.0, "environment_id": "e1"},
                {"experiment_id": "d", "latency_p95_ms": 200.0, "environment_id": "e1"},
            ),
            environment_key="environment_id",
        )
        marcas, _ = t.best_marks(tabla)
        assert marcas["latency_p95_ms"] == frozenset({0, 1, 2})

    def test_coincide_con_pareto_compare_en_las_dos_sesiones(self) -> None:
        """La misma pareja, en la tabla y en la frontera, con el mismo veredicto."""
        from tfm_edge.analysis import floors as mf
        from tfm_edge.analysis import pareto as mp

        suelos = mf.FloorSet(
            floors={
                "latency_p95_ms": mf.AxisFloor(
                    axis="latency_p95_ms",
                    kind="relativo",
                    same_session=0.15,
                    cross_session=0.47,
                    provenance="el mismo del test",
                )
            }
        )
        objetivo = mp.Objective("latencia p95", "latency_p95_ms", "deployment", "min")

        def punto(nombre: str, valor: float, huella: str) -> mp.Point:
            return mp.Point(
                experiment_id=f"s_baseline_default_{nombre * 8}",
                runtime="a",
                precision="fp32",
                device_id="dev",
                environment_id=huella,
                timestamp="2026-10-02T00:00:00",
                values={"latency_p95_ms": valor},
            )

        for huellas in (("e1", "e1", "e1"), ("e1", "e2", "e1"), ("e1", "", "e1")):
            marcas, _ = t.best_marks(self._tabla_sesiones(huellas, "environment_id"))
            mejor = punto("a", 100.0, huellas[0])
            for indice, valor in ((1, 120.0), (2, 160.0)):
                otro = punto("b", valor, huellas[indice])
                veredicto = mp.compare(
                    mejor, otro, objetivo, suelos, mp.same_session(mejor, otro)
                ).verdict
                assert (indice in marcas["latency_p95_ms"]) == (
                    veredicto == "indistinguible"
                ), f"la tabla y pareto.compare discrepan sobre {valor} ms con huellas {huellas}"

    def test_el_pie_cita_los_dos_suelos(self) -> None:
        """Si la columna entera empata, el lector tiene que saber qué suelo se aplicó."""
        columna = t.latency_column(
            "latency_p95_ms",
            "latencia p95",
            floor_relative=0.4729,
            floor_same_session=0.1472,
            floor_note="suelo entre sesiones",
        )
        assert columna.floor_text == "14,7 % en la misma sesión, 47,3 % entre sesiones"

    def test_sin_suelo_de_misma_sesion_el_pie_no_cambia(self) -> None:
        columna = t.latency_column(
            "latency_p95_ms",
            "latencia p95",
            floor_relative=0.4729,
            floor_note="suelo entre sesiones",
        )
        assert columna.floor_text == "47,3 %"

    def test_suelo_de_misma_sesion_sin_el_base(self) -> None:
        with pytest.raises(ValueError, match="sin el de sesiones distintas"):
            t.Column(
                key="x",
                header="X",
                decimals=2,
                better="min",
                floor_same_session=0.1,
                floor_note="suelos por sesión",
            )


class TestLoQueNoSeAdmite:
    """Errores que se detectan al construir la tabla."""

    def test_un_suelo_sin_procedencia(self) -> None:
        """Un umbral inventado acabaría citado en la memoria como si se hubiera medido."""
        with pytest.raises(ValueError, match="sin floor_note"):
            t.Column(key="x", header="X", decimals=2, better="min", floor=1.0)

    def test_una_columna_de_texto_con_mejor_valor(self) -> None:
        """No se puede ordenar lo que no es un número."""
        with pytest.raises(ValueError, match="no puede tener un mejor valor"):
            t.Column(key="x", header="X", better="max")

    def test_una_direccion_desconocida(self) -> None:
        """Un typo en ``better`` daría una tabla sin negritas y sin avisar."""
        with pytest.raises(ValueError, match="solo se admiten"):
            t.Column(key="x", header="X", decimals=2, better="mayor")

    def test_una_tabla_sin_filas(self) -> None:
        """Una tabla vacía en results/tables se lee como un resultado."""
        with pytest.raises(ValueError, match="no tiene filas"):
            _tabla(())

    def test_dos_columnas_con_la_misma_clave(self) -> None:
        """La segunda taparía a la primera en el CSV sin decir nada."""
        with pytest.raises(ValueError, match="repite las claves"):
            _tabla(
                ({"map_50": 0.5},),
                columnas=(t.quality_column("map_50", "A"), t.quality_column("map_50", "B")),
            )


class TestLaTeX:
    """El fichero que va a la memoria."""

    def test_escapa_los_guiones_bajos_del_identificador(self) -> None:
        """Sin escapar, un ``experiment_id`` sale con medio nombre en subíndice."""
        tabla = _tabla(({"experiment_id": "s_baseline_default_745e3d08", "map_50": 0.5},))
        tex = t.render_latex(tabla)
        assert r"s\_baseline\_default\_745e3d08" in tex
        assert "s_baseline" not in tex

    @pytest.mark.parametrize(
        ("dato", "escrito"),
        [
            ("a--b", "a-{}-b"),
            ("a---b", "a-{}-{}-b"),
            ("<<x>>", "<{}<x>{}>"),
            ("1,,2", ",{},"),
            ("l'x", r"l\textquotesingle{}x"),
            ("!`", r"!\textasciigrave{}"),
        ],
    )
    def test_no_deja_formar_ligaduras(self, dato: str, escrito: str) -> None:
        assert escrito in t._escape_latex(dato)

    def test_un_guion_suelto_no_se_toca(self) -> None:
        assert t._escape_latex("prune50-ptqint8") == "prune50-ptqint8"

    def test_lleva_caption_y_label(self) -> None:
        """Sin etiqueta no se puede referenciar desde el texto."""
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5},))
        contenido = t.render_latex(tabla)
        assert r"\caption{Tabla de prueba}" in contenido
        assert r"\label{tab:6_4_prueba}" in contenido

    def test_avisa_de_que_se_regenera(self) -> None:
        """Editar a mano un fichero que se regenera es perder el cambio."""
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5},))
        assert t.render_latex(tabla).startswith("% Generado por")


def _requisitos(tex: str) -> str:
    """La línea de la cabecera que dice qué paquetes necesita el fichero."""
    return next(linea for linea in tex.splitlines() if linea.startswith("% Requiere"))


class TestLoQueLaCabeceraDeclara:
    """Lo que el .tex dice necesitar tiene que bastar para compilarlo."""

    def test_con_comillas_latinas_pide_t1(self) -> None:
        """«» fallan en OT1, que es lo que hay sin ``fontenc``."""
        tabla = t.Table(
            name="prueba",
            caption="Vista «plano»",
            chapter=6,
            section=4,
            columns=_columnas(),
            rows=({"experiment_id": "a", "map_50": 0.5},),
        )
        assert r"\usepackage[T1]{fontenc}" in _requisitos(t.render_latex(tabla))

    def test_las_comillas_de_una_nota_tambien_cuentan(self) -> None:
        tabla = t.Table(
            name="prueba",
            caption="Sin comillas",
            chapter=6,
            section=4,
            columns=_columnas(),
            rows=({"experiment_id": "a", "map_50": 0.5},),
            notes=("«estricta»: no dominada por nadie.",),
        )
        assert r"\usepackage[T1]{fontenc}" in _requisitos(t.render_latex(tabla))

    def test_los_acentos_no_piden_t1(self) -> None:
        tabla = _tabla(({"experiment_id": "tamaño_ó", "map_50": 0.5},))
        assert _requisitos(t.render_latex(tabla)) == r"% Requiere \usepackage{booktabs}."

    @pytest.mark.parametrize("caracter", ["<", ">", "|", '"', "^", "~"])
    def test_los_que_compilan_cambiados_piden_t1(self, caracter: str) -> None:
        tabla = _tabla(({"experiment_id": f"a{caracter}b", "map_50": 0.5},))
        assert r"\usepackage[T1]{fontenc}" in _requisitos(t.render_latex(tabla))

    def test_el_marcado_propio_no_cuenta(self) -> None:
        """La apaisada lleva un ``>`` en su marcado; eso no es texto impreso."""
        tabla = t.Table(
            name="prueba",
            caption="Sin nada raro",
            chapter=6,
            section=4,
            columns=_columnas(),
            rows=({"experiment_id": "a", "map_50": 0.5},),
            latex_layout="landscape",
        )
        assert "fontenc" not in _requisitos(t.render_latex(tabla))


class TestCaracteresQueNoCompilan:
    """Un carácter que pdflatex no compone se rechaza al generar."""

    @pytest.mark.parametrize("simbolo", ["Δ", "≤", "≥", "○", "✓", "✕"])
    def test_se_rechaza_con_su_codigo(self, simbolo: str) -> None:
        """Los seis que aparecen en el código de reporting y fallan incluso en T1."""
        tabla = _tabla(({"experiment_id": f"a{simbolo}", "map_50": 0.5},))
        with pytest.raises(ValueError, match=f"U\\+{ord(simbolo):04X}"):
            t.render_latex(tabla)

    def test_tambien_en_el_titulo_y_en_las_notas(self) -> None:
        """No solo las celdas llegan al .tex."""
        for campos in ({"caption": "Δ mAP"}, {"notes": ("mAP ≥ 0,5",)}):
            tabla = t.Table(
                name="prueba",
                caption=campos.get("caption", "x"),
                chapter=6,
                section=4,
                columns=_columnas(),
                rows=({"experiment_id": "a", "map_50": 0.5},),
                notes=campos.get("notes", ()),
            )
            with pytest.raises(ValueError, match="no compone"):
                t.render_latex(tabla)

    def test_los_de_t1_estan_admitidos(self) -> None:
        assert t.T1_REQUIRED_CHARACTERS <= t.LATEX_SAFE_CHARACTERS


class TestApaisada:
    """Una tabla que no cabe en vertical sale en página apaisada, reducida si hace falta."""

    def _apaisada(self) -> t.Table:
        """Una tabla apaisada mínima."""
        return t.Table(
            name="prueba",
            caption="Tabla de prueba",
            chapter=6,
            section=4,
            columns=_columnas(),
            rows=({"experiment_id": "a", "map_50": 0.5},),
            latex_layout="landscape",
        )

    def test_va_en_una_pagina_apaisada_y_lo_declara(self) -> None:
        """Sin ``pdflscape`` y ``adjustbox`` en el preámbulo no compila."""
        tex = t.render_latex(self._apaisada())
        assert r"\begin{landscape}" in tex.splitlines()
        assert tex.rstrip().endswith(r"\end{landscape}")
        requisitos = _requisitos(tex)
        assert r"\usepackage{pdflscape}" in requisitos
        assert r"\usepackage{adjustbox}" in requisitos

    def test_reduce_a_lo_ancho_y_a_lo_alto_y_nunca_amplia(self) -> None:
        """Con cincuenta filas manda el alto; ``max`` no agranda nunca."""
        tex = t.render_latex(self._apaisada())
        assert r"\begin{adjustbox}{max width=\linewidth, max totalheight=" in tex
        assert "resizebox" not in tex

    def test_la_vertical_no_cambia(self) -> None:
        """Ni página apaisada, ni reducción, ni paquetes de más."""
        tex = t.render_latex(_tabla(({"experiment_id": "a", "map_50": 0.5},)))
        assert "landscape" not in tex
        assert "resizebox" not in tex
        assert _requisitos(tex) == r"% Requiere \usepackage{booktabs}."

    def test_una_orientacion_inventada_es_un_error(self) -> None:
        with pytest.raises(ValueError, match="orientación"):
            t.Table(
                name="prueba",
                caption="x",
                chapter=6,
                section=4,
                columns=_columnas(),
                rows=({"experiment_id": "a"},),
                latex_layout="apaisada",
            )


class TestCompilaConLoQueDeclara:
    """El consumidor del .tex es pdflatex, así que el test lo compila."""

    @staticmethod
    def _compilar(tex: str, directorio) -> tuple[int, str]:
        """Compila un .tex con el preámbulo que declara su propia cabecera."""
        import shutil
        import subprocess

        if shutil.which("pdflatex") is None:
            pytest.skip("pdflatex no está instalado en este entorno.")
        paquetes = _requisitos(tex).removeprefix("% Requiere ").rstrip(".").split(", ")
        (directorio / "tabla.tex").write_text(tex, encoding="utf-8")
        documento = directorio / "doc.tex"
        documento.write_text(
            "\\documentclass{article}\n"
            + "\n".join(paquetes)
            + "\n\\begin{document}\n\\input{tabla.tex}\n\\end{document}\n",
            encoding="utf-8",
        )
        completado = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", documento.name],
            cwd=directorio,
            capture_output=True,
            text=True,
            check=False,
        )
        log = (directorio / "doc.log").read_text(encoding="latin-1")
        return completado.returncode, log

    def test_con_comillas_latinas(self, tmp_path) -> None:
        """El caso que no compilaba."""
        tabla = t.Table(
            name="prueba",
            caption="Vista «plano»",
            chapter=6,
            section=4,
            columns=_columnas(),
            rows=({"experiment_id": "s_baseline_default_745e3d08", "map_50": 0.5},),
            notes=("«estricta»: no dominada por nadie.",),
        )
        codigo, log = self._compilar(t.render_latex(tabla), tmp_path)
        assert codigo == 0, log[-2000:]

    def test_una_apaisada_demasiado_ancha_se_reduce_y_cabe(self, tmp_path) -> None:
        """Doce identificadores largos por fila no caben ni apaisados sin reducir."""
        identificador = "s_combined_prune-l1global-r25-randomctl-ptqint8_e70dc4d6"
        columnas = tuple(t.text_column(f"c{i}", f"columna {i}") for i in range(12))
        tabla = t.Table(
            name="prueba",
            caption="Ancha",
            chapter=6,
            section=4,
            columns=columnas,
            rows=({f"c{i}": identificador for i in range(12)},),
            latex_layout="landscape",
        )
        codigo, log = self._compilar(t.render_latex(tabla), tmp_path)
        assert codigo == 0, log[-2000:]
        assert "Overfull \\hbox" not in log, "la reducción no ha hecho caber la tabla"

    def test_una_apaisada_demasiado_alta_se_reduce_y_cabe(self, tmp_path) -> None:
        tabla = t.Table(
            name="prueba",
            caption="Una tabla alta, con un título lo bastante largo como para ocupar "
            "más de una línea en una página apaisada y restarle alto al tabular " * 2,
            chapter=6,
            section=4,
            columns=_columnas(),
            rows=tuple(
                {"experiment_id": f"s_baseline_default_{i:08d}", "map_50": 0.5} for i in range(80)
            ),
            notes=("Una nota al pie que también ocupa alto. " * 12,),
            latex_layout="landscape",
        )
        codigo, log = self._compilar(t.render_latex(tabla), tmp_path)
        assert codigo == 0, log[-2000:]
        assert "Float too large" not in log, "la reducción no ha hecho caber la tabla"

    def test_todo_lo_admitido_compila(self, tmp_path) -> None:
        admitidos = "".join(sorted(t.LATEX_SAFE_CHARACTERS))
        tabla = _tabla(({"experiment_id": admitidos, "map_50": 0.5},))
        codigo, log = self._compilar(t.render_latex(tabla), tmp_path)
        assert codigo == 0, log[-2000:]
        assert "Missing character" not in log


class TestSeImprimeTalCual:
    """El texto impreso en el PDF coincide con el dato."""

    #: Filas por tabla: más no caben en una página vertical.
    _FILAS = 40

    @staticmethod
    def _como_se_extrae(texto: str, con_t1: bool) -> str:
        """Lo que ``pdftotext`` devuelve cuando lo impreso es correcto."""
        texto = texto.replace("…", "...")
        if not con_t1:
            texto = texto.replace("_", "")
            for vocal, acento in (("ì", "̀"), ("í", "́"), ("î", "̂"), ("ï", "̈")):
                texto = texto.replace(vocal, "ı" + acento)
        return texto

    def _imprimir(self, textos: list[str], con_t1: bool, directorio) -> list[str]:
        """Compila una columna con cada texto entre marcas y lee lo impreso."""
        import shutil
        import subprocess

        if shutil.which("pdftotext") is None:
            pytest.skip("pdftotext no está instalado en este entorno.")
        entradas: list[str] = []
        for inicio in range(0, len(textos), self._FILAS):
            tabla = t.Table(
                name=f"prueba{inicio}",
                caption="Con «T1»" if con_t1 else "Sin T1",
                chapter=6,
                section=4,
                columns=(t.text_column("x", "X"),),
                rows=tuple({"x": f"Q{x}Z"} for x in textos[inicio : inicio + self._FILAS]),
            )
            tex = t.render_latex(tabla)
            assert ("fontenc" in _requisitos(tex)) == con_t1
            entradas.append(tex)
        codigo, log = TestCompilaConLoQueDeclara._compilar(
            "\n\\clearpage\n".join(entradas), directorio
        )
        assert codigo == 0, log[-2000:]
        assert "Missing character" not in log
        extraido = subprocess.run(
            ["pdftotext", str(directorio / "doc.pdf"), "-"], capture_output=True, check=True
        ).stdout.decode("utf-8")
        lineas = [linea.strip() for linea in extraido.splitlines()]
        filas = [
            linea[1:-1].replace(" ", "")
            for linea in lineas
            if linea[:1] == "Q" and linea[-1:] == "Z"
        ]
        assert len(filas) == len(textos), "no se ha leído una fila por texto"
        return filas

    @pytest.mark.parametrize("con_t1", [False, True], ids=["ot1", "t1"])
    def test_cada_caracter_admitido(self, con_t1: bool, tmp_path) -> None:
        """Sin T1, los que no la piden; con T1, todos."""
        textos = sorted(
            c
            for c in t.LATEX_SAFE_CHARACTERS
            if c != " " and (con_t1 or c not in t.T1_REQUIRED_CHARACTERS)
        )
        impresos = self._imprimir(textos, con_t1, tmp_path)
        distintos = [
            (texto, impreso)
            for texto, impreso in zip(textos, impresos)
            if unicodedata.normalize("NFC", impreso) != self._como_se_extrae(texto, con_t1)
        ]
        assert not distintos

    @pytest.mark.parametrize("con_t1", [False, True], ids=["ot1", "t1"])
    def test_cada_par_de_puntuacion(self, con_t1: bool, tmp_path) -> None:
        import string

        pares = [a + b for a in string.punctuation for b in string.punctuation] + ["---"]
        textos = [p for p in pares if con_t1 or not set(p) & t.T1_REQUIRED_CHARACTERS]
        impresos = self._imprimir(textos, con_t1, tmp_path)
        distintos = [
            (texto, impreso)
            for texto, impreso in zip(textos, impresos)
            if impreso != self._como_se_extrae(texto, con_t1)
        ]
        assert not distintos


class TestXLSX:
    """La hoja de cálculo guarda números como números."""

    def test_los_numeros_no_se_escriben_como_texto(self, tmp_path) -> None:
        """Una hoja con números en texto no se puede ordenar ni graficar."""
        from openpyxl import load_workbook

        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5782134, "latency_p50_ms": 138.4},))
        ruta = t.write_xlsx(tabla, tmp_path / "x.xlsx")
        hoja = load_workbook(ruta).active
        celda = hoja.cell(row=2, column=2)
        assert isinstance(celda.value, float)
        assert celda.value == pytest.approx(0.5782134)
        assert celda.number_format == "0.0000"

    def test_la_celda_ausente_se_queda_sin_valor(self, tmp_path) -> None:
        from openpyxl import load_workbook

        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5, "latency_p50_ms": ""},))
        hoja = load_workbook(t.write_xlsx(tabla, tmp_path / "x.xlsx")).active
        assert hoja.cell(row=2, column=3).value is None

    def test_el_mejor_valor_va_en_negrita(self, tmp_path) -> None:
        """Es lo que pide la 15.2."""
        from openpyxl import load_workbook

        tabla = _tabla(
            (
                {"experiment_id": "a", "map_50": 0.50, "latency_p50_ms": 100.0},
                {"experiment_id": "b", "map_50": 0.60, "latency_p50_ms": 200.0},
            )
        )
        hoja = load_workbook(t.write_xlsx(tabla, tmp_path / "x.xlsx")).active
        assert hoja.cell(row=3, column=2).font.bold is True
        assert not hoja.cell(row=2, column=2).font.bold

    def test_las_dos_fechas_del_fichero_estan_fijadas(self, tmp_path) -> None:
        """Sin fecha fija, ``git status`` no distingue un número nuevo de una hora nueva."""
        import zipfile

        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5, "latency_p50_ms": 100.0},))
        ruta = t.write_xlsx(tabla, tmp_path / "a.xlsx")
        with zipfile.ZipFile(ruta) as fichero:
            nucleo = fichero.read(t.CORE_PROPERTIES_ENTRY).decode("utf-8")
            fechas_zip = {info.date_time for info in fichero.infolist()}
        assert (
            nucleo.count(t.FIXED_TIMESTAMP_TEXT) == 2
        ), f"creación y modificación tienen que ir las dos al instante fijo: {nucleo}"
        assert fechas_zip == {(2026, 1, 1, 0, 0, 0)}, fechas_zip

    def test_dos_escrituras_dan_el_mismo_fichero(self, tmp_path) -> None:
        """Y el fichero entero coincide, byte a byte."""
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5, "latency_p50_ms": 100.0},))
        primero = t.write_xlsx(tabla, tmp_path / "a.xlsx").read_bytes()
        segundo = t.write_xlsx(tabla, tmp_path / "b.xlsx").read_bytes()
        assert primero == segundo


class TestGuardado:
    """Los tres formatos, con el nombre que pide la convención del proyecto."""

    def test_escribe_los_tres_con_prefijo_de_capitulo(self, tmp_path) -> None:
        tabla = _tabla(({"experiment_id": "a", "map_50": 0.5, "latency_p50_ms": 100.0},))
        rutas = t.save_table(tabla, tmp_path)
        assert set(rutas) == {"tex", "xlsx", "csv"}
        for ruta in rutas.values():
            assert ruta.name.startswith("tab_6_4_prueba.")
            assert ruta.exists() and ruta.stat().st_size > 0
