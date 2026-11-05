"""Las figuras del capítulo 5, y lo que declaran de su propio alcance."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.analysis.splits import analysis_quality_rows
from tfm_edge.common.registry import ExperimentRegistry
from tfm_edge.reporting import figures_results as fr
from tfm_edge.reporting import style

RAIZ = Path(__file__).resolve().parent.parent

CABECERA = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),"
    "val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2\n"
)


def _escribir_curva(directorio: Path, epocas: int = 3) -> Path:
    """Escribe un ``results.csv`` mínimo con la forma de Ultralytics."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / fr.CURVE_FILENAME
    filas = [CABECERA]
    for epoca in range(1, epocas + 1):
        filas.append(
            f"{epoca},10.0,1.0,1.5,1.2,0.5,0.5,0.4{epoca},0.2{epoca},"
            f"1.0,0.8,1.1,0.01,0.01,0.01\n"
        )
    ruta.write_text("".join(filas), encoding="utf-8")
    return ruta


def _experimento(experiment_id: str, epocas: str, tecnica: str = "baseline") -> dict[str, str]:
    """Fila de experimento de prueba."""
    return {
        "experiment_id": experiment_id,
        "technique": tecnica,
        "training_epochs": epocas,
        "status": "ok",
    }


class TestElInventarioDiceLoQueFalta:
    """Una ausencia callada se lee como un descuido."""

    def test_encuentra_la_curva_y_la_ata_a_su_experimento(self, tmp_path: Path) -> None:
        """El directorio no lleva el hash, así que se ata por prefijo."""
        _escribir_curva(tmp_path / "baselines" / "s_baseline_default")

        inventario = fr.training_curve_inventory(
            tmp_path, [_experimento("s_baseline_default_745e3d08", "100")]
        )

        assert len(inventario.curves) == 1
        assert inventario.curves[0].experiment_id == "s_baseline_default_745e3d08"
        assert inventario.missing == ()

    def test_un_entrenamiento_sin_curva_se_declara(self, tmp_path: Path) -> None:
        _escribir_curva(tmp_path / "baselines" / "n_baseline_default")

        inventario = fr.training_curve_inventory(
            tmp_path,
            [
                _experimento("n_baseline_default_fab9d45c", "100"),
                _experimento("s_baseline_default_745e3d08", "100"),
            ],
        )

        assert [identificador for identificador, _ in inventario.missing] == [
            "s_baseline_default_745e3d08"
        ]

    def test_lo_que_no_entreno_no_cuenta_como_ausencia(self, tmp_path: Path) -> None:
        inventario = fr.training_curve_inventory(
            tmp_path, [_experimento("s_ptq_int8-noexcl_5bcd71f0", "", tecnica="ptq")]
        )

        assert inventario.missing == ()

    def test_varias_curvas_del_mismo_experimento_se_distinguen(self, tmp_path: Path) -> None:
        """La poda iterativa deja un results.csv por ciclo."""
        base = tmp_path / "pruned" / "s_pruning_l1global_r50_iter4_d8fe066f"
        _escribir_curva(base / "finetune" / "train")
        _escribir_curva(base / "finetune_step1" / "train")

        inventario = fr.training_curve_inventory(
            tmp_path, [_experimento("s_pruning_l1global_r50_iter4_d8fe066f", "30", "pruning")]
        )

        etiquetas = {c.label for c in inventario.curves}
        assert len(etiquetas) == 2

    def test_una_curva_sin_columna_de_epoca_no_entra(self, tmp_path: Path) -> None:
        """Dibujarla por su orden de aparición sería suponer el eje horizontal."""
        directorio = tmp_path / "baselines" / "s_baseline_default"
        directorio.mkdir(parents=True)
        (directorio / fr.CURVE_FILENAME).write_text("a,b\n1,2\n", encoding="utf-8")

        inventario = fr.training_curve_inventory(
            tmp_path, [_experimento("s_baseline_default_745e3d08", "100")]
        )

        assert inventario.curves == ()
        assert len(inventario.missing) == 1


class TestLaPerdidaTotalNoSeSumaAMedias:
    """Un total al que le falta un sumando no compara con otro que lo tenga."""

    def _curva(self, **series: tuple[float, ...]) -> fr.TrainingCurve:
        """Curva de prueba con las series indicadas."""
        return fr.TrainingCurve(
            experiment_id="s_baseline_default_745e3d08",
            label="prueba",
            technique="baseline",
            source=Path("results.csv"),
            epochs=(1, 2),
            series={clave.replace("__", "/"): valor for clave, valor in series.items()},
        )

    def test_con_las_tres_perdidas_se_suma(self) -> None:
        """El caso normal, para que el siguiente test no pase por no sumar nunca."""
        curva = self._curva(
            train__box_loss=(1.0, 0.9),
            train__cls_loss=(2.0, 1.9),
            train__dfl_loss=(3.0, 2.9),
        )

        assert curva.total_loss == pytest.approx((6.0, 5.7))

    def test_si_falta_una_no_se_suma_nada(self) -> None:
        """Sumar las que haya daría un total menor con aspecto de medida."""
        curva = self._curva(train__box_loss=(1.0, 0.9), train__cls_loss=(2.0, 1.9))

        assert curva.total_loss == ()


class TestCadaFilaTieneUnaEtiquetaSuya:
    """QAT tiene filas FP32 e INT8 en el mismo motor."""

    def test_la_etiqueta_lleva_motor_y_precision(self) -> None:
        """Sin la precisión, dos filas distintas salían con el mismo rótulo."""
        fp32 = fr.row_label(
            {
                "experiment_id": "s_qat_int8-noexcl_d4d403fa",
                "runtime": "onnxruntime",
                "precision": "fp32",
                "stage": "reajustado",
            }
        )
        int8 = fr.row_label(
            {
                "experiment_id": "s_qat_int8-noexcl_d4d403fa",
                "runtime": "onnxruntime",
                "precision": "int8",
                "stage": "reajustado",
            }
        )

        assert fp32 != int8
        assert "fp32" in fp32 and "int8" in int8

    def test_la_poda_en_crudo_se_distingue_de_la_reajustada(self) -> None:
        cruda = fr.row_label(
            {
                "experiment_id": "s_pruning_bnscale_r50_8895e5b2",
                "runtime": "pytorch",
                "precision": "fp32",
                "stage": "sin_reajuste",
            }
        )
        reajustada = fr.row_label(
            {
                "experiment_id": "s_pruning_bnscale_r50_f51fef4c",
                "runtime": "pytorch",
                "precision": "fp32",
                "stage": "reajustado",
            }
        )

        assert "sin_reajuste" in cruda
        # Lo reajustado no se marca: es el caso normal y marcarlo todo no
        # distingue nada. Es la convención de FamilyCurve.label.
        assert "reajustado" not in reajustada

    def test_una_fila_sin_etapa_no_se_etiqueta_a_medias(self) -> None:
        """Un volcado viejo no trae la etapa, y una etiqueta muda parece reajustada."""
        with pytest.raises(ValueError, match="29_analyze_results"):
            fr.row_label(
                {
                    "experiment_id": "s_pruning_bnscale_r50_8895e5b2",
                    "runtime": "pytorch",
                    "precision": "fp32",
                }
            )


class TestElSueloSeDibujaDondeVive:
    """No en una altura fija del eje, sino en cada punto."""

    def test_un_suelo_relativo_escala_con_el_valor(self) -> None:
        margenes = fr._floor_margins([0.40, 0.20], {"kind": "relativo", "cross_session": 0.10})

        assert margenes == pytest.approx([0.04, 0.02])

    def test_un_suelo_absoluto_no_escala(self) -> None:
        margenes = fr._floor_margins([0.40, 0.20], {"kind": "absoluto", "cross_session": 0.004579})

        assert margenes == pytest.approx([0.004579, 0.004579])

    def test_sin_suelo_medido_no_se_dibuja_ninguno(self) -> None:
        assert fr._floor_margins([0.40], None) is None
        assert fr._floor_margins([0.40], {"kind": "absoluto", "cross_session": None}) is None


class TestDosTamanosNoSeDibujanComoUnoSolo:
    """La cuarta aparición del índice más corto que su tabla."""

    @staticmethod
    def _curva(model_size: str) -> dict:
        """Una curva de poda con dos niveles para el tamaño dado."""
        return {
            "num_levels": 2,
            "runtime": "onnxruntime",
            "technique": "pruning",
            "model_size": model_size,
            "method": "l1global",
            "stage": "reajustado",
            "precision": "fp32",
            "shape": "curva",
            "points": [
                {"level": 0.25, "values": {"map_50_95": 0.35}},
                {"level": 0.50, "values": {"map_50_95": 0.30}},
            ],
        }

    def test_el_rotulo_de_la_leyenda_distingue_los_tamanos(self) -> None:
        """Sin el tamaño, la leyenda tendría dos entradas idénticas."""
        import matplotlib.pyplot as plt

        figura, _ = fr.figure_quality_vs_level([self._curva("n"), self._curva("s")], {})
        try:
            rotulos = [
                texto.get_text() for leyenda in figura.legends for texto in leyenda.get_texts()
            ]
            assert len(set(rotulos)) == len(rotulos), f"rótulos repetidos: {rotulos}"
            assert any("yolov8n" in r for r in rotulos), rotulos
            assert any("yolov8s" in r for r in rotulos), rotulos
        finally:
            plt.close(figura)

    def test_cada_tamano_recibe_su_propio_trazo(self) -> None:
        import matplotlib.pyplot as plt

        figura, _ = fr.figure_quality_vs_level([self._curva("n"), self._curva("s")], {})
        try:
            (panel,) = [ejes for ejes in figura.axes if ejes.get_xlabel()]
            trazos = {
                linea.get_linestyle() for linea in panel.lines if linea.get_xdata() is not None
            }
            assert (
                len(trazos) >= 2
            ), f"las dos curvas comparten trazo y color: serían indistinguibles ({trazos})"
        finally:
            plt.close(figura)

    def test_el_resumen_dice_de_que_tamano_es_cada_curva(self) -> None:
        """El manifiesto tiene que permitir comprobarlo sin abrir el PDF."""
        import matplotlib.pyplot as plt

        figura, resumen = fr.figure_quality_vs_level([self._curva("n"), self._curva("s")], {})
        try:
            dibujadas = resumen["panels"]["onnxruntime"]
            assert sorted(dibujadas) == ["n/l1global/reajustado", "s/l1global/reajustado"]
        finally:
            plt.close(figura)


class TestUnaFiguraSinDatosNoSale:
    """Una figura vacía dice que no pasó nada; la ausencia dice otra cosa."""

    def test_sin_curvas_no_hay_figura_de_entrenamiento(self) -> None:
        """Una figura vacía diría que el entrenamiento no ocurrió."""
        with pytest.raises(ValueError, match="ninguna curva de entrenamiento"):
            fr.figure_training_curves(fr.CurveInventory(curves=(), missing=()))

    def test_sin_curvas_de_dos_niveles_no_hay_figura_de_compresion(self) -> None:
        """Por un punto suelto no pasa ninguna tendencia."""
        curvas = [
            {
                "num_levels": 1,
                "runtime": "onnxruntime",
                "technique": "ptq",
                "method": "int8-noexcl",
                "stage": "sin_reajuste",
                "precision": "int8",
                "shape": "punto",
                "points": [{"level": 1.0, "values": {"map_50_95": 0.3}}],
            }
        ]

        with pytest.raises(ValueError, match="ninguna tendencia"):
            fr.figure_quality_vs_level(curvas, {})

    def test_sin_desglose_por_clase_no_hay_mapa_de_calor(self) -> None:
        """Sin AP por clase no hay nada que colorear."""
        with pytest.raises(ValueError, match="mapa de calor"):
            fr.figure_class_degradation([{"per_class": {}, "runtime": "pytorch"}])

    def test_sin_coste_registrado_no_hay_figura_de_coste(self) -> None:
        """Dibujarlas en el cero afirmaría que fueron gratis."""
        with pytest.raises(ValueError, match="coste registrado y ganancia medida"):
            fr.figure_cost_vs_gain([], [])


RAZONES = {
    "sin_detecciones": "sin descomposicion de errores: esta evaluacion no dejo detecciones",
    "otro_evaluador": "sin descomposicion de errores: la cifra es del evaluador ultralytics",
}

TIPOS = ("classification", "localization", "both", "duplicate", "background", "missed")


def _clave(experimento: str, motor: str, precision: str, evaluador: str = "pycocotools"):
    """Clave primaria completa de una fila de calidad de prueba."""
    return {
        "experiment_id": experimento,
        "runtime": motor,
        "precision": precision,
        "device_id": "workstation_cpu",
        "evaluator": evaluador,
        "split": "val",
    }


def _recuentos(base: int) -> dict[str, int]:
    """Seis recuentos distintos entre sí, para que un cruce equivocado se note."""
    return {t: base + i for i, t in enumerate(TIPOS)}


def _caso() -> tuple[list[dict[str, str]], dict[str, dict[str, str]], dict]:
    """Un registro de cuatro filas y el volcado que lo respalda."""
    experimentos = {
        "s_baseline_default_11111111": {
            "experiment_id": "s_baseline_default_11111111",
            "parent_experiment_id": "",
            "technique": "baseline",
            "model_size": "s",
            "training_epochs": "100",
            "status": "ok",
        },
        "s_ptq_int8-noexcl_22222222": {
            "experiment_id": "s_ptq_int8-noexcl_22222222",
            "parent_experiment_id": "s_baseline_default_11111111",
            "technique": "ptq",
            "model_size": "s",
            "training_epochs": "0",
            "status": "ok",
        },
        "n_distillation_kd_33333333": {
            "experiment_id": "n_distillation_kd_33333333",
            "parent_experiment_id": "",
            "technique": "distillation",
            "model_size": "n",
            "training_epochs": "50",
            "status": "ok",
        },
    }
    columnas = {t: f"error_{t}" for t in TIPOS}

    def fila(clave: dict[str, str], recuentos: dict[str, int] | None, notas: str) -> dict:
        salida = dict(clave)
        for t in TIPOS:
            salida[columnas[t]] = "" if recuentos is None else str(recuentos[t])
        salida["notes"] = notas
        return salida

    base = _clave("s_baseline_default_11111111", "openvino", "fp32")
    ptq = _clave("s_ptq_int8-noexcl_22222222", "openvino", "int8")
    calidad = [
        fila(base, _recuentos(10), "evaluador=pycocotools; ruta curva"),
        fila(ptq, _recuentos(100), "evaluador=pycocotools; ruta volcado"),
        fila(
            _clave("s_baseline_default_11111111", "pytorch", "fp32"),
            None,
            f"evaluador=pycocotools; {RAZONES['sin_detecciones']}",
        ),
        fila(
            _clave("n_distillation_kd_33333333", "pytorch", "fp32", "ultralytics"),
            None,
            f"evaluador=ultralytics; {RAZONES['otro_evaluador']}",
        ),
    ]

    def evaluacion(clave: dict[str, str], ruta: str, recuentos: dict[str, int]) -> dict:
        return {
            "primary_key": clave,
            "route": ruta,
            "decomposition": {
                "aggregation": "recuentos agregados, no medias por clase",
                "num_gt": 500,
                "by_confidence": [
                    {"confidence": 0.1, **_recuentos(999)},
                    {"confidence": 0.25, **recuentos},
                ],
            },
        }

    volcado = {
        "registry_confidence": 0.25,
        "split": "val",
        "kinds": list(TIPOS),
        "registry_columns": columnas,
        "reasons": dict(RAZONES),
        "coverage": {
            "quality_rows": 4,
            "with_decomposition": 2,
            "by_route": {"curva": 1, "volcado": 1},
            "without_decomposition": 2,
        },
        "evaluations": [
            evaluacion(base, "curva", _recuentos(10)),
            evaluacion(ptq, "volcado", _recuentos(100)),
        ],
    }
    return calidad, experimentos, volcado


class TestLaFiguraDeTiposDeErrorCruzaRegistroYVolcado:
    """La figura de tipos de error cruza registro y volcado."""

    def test_la_figura_va_al_apartado_5_5(self) -> None:
        """Con el mismo prefijo que la degradación por clase y por tamaño."""
        assert fr.PLACEMENT["tipos_de_error"] == (5, 5)

    def test_el_estilo_tiene_las_seis_clases_del_reparto(self) -> None:
        """Una clase sin estilo no se podría dibujar; una de más no existe."""
        from tfm_edge.evaluation.error_decomposition import ERROR_KINDS

        assert tuple(style.ERROR_KIND_STYLES) == ERROR_KINDS
        assert fr.MISSED_KIND in ERROR_KINDS

    def test_el_caso_bueno_da_la_cobertura_del_cruce(self) -> None:
        """Dos dibujadas, dos huecos con su motivo, y la destilación ausente."""
        calidad, experimentos, volcado = _caso()
        datos = fr.error_decomposition_rows(calidad, experimentos, volcado)

        assert len(datos.rows) == 2
        assert datos.by_route == {"curva": 1, "volcado": 1}
        assert datos.missing_by_reason == {"sin_detecciones": 1, "otro_evaluador": 1}
        assert datos.techniques_absent == ("distillation",)
        assert datos.num_gt == 500
        por_experimento = {f["experiment_id"]: f for f in datos.rows}
        assert por_experimento["s_ptq_int8-noexcl_22222222"]["counts"] == _recuentos(100)
        assert por_experimento["s_ptq_int8-noexcl_22222222"]["route"] == "volcado"

    def test_la_raiz_no_sale_marcada_sin_reajuste(self) -> None:
        calidad, experimentos, volcado = _caso()
        datos = fr.error_decomposition_rows(calidad, experimentos, volcado)
        por_experimento = {f["experiment_id"]: f for f in datos.rows}

        assert fr.stage_suffix(por_experimento["s_baseline_default_11111111"]) == ""
        assert fr.stage_suffix(por_experimento["s_ptq_int8-noexcl_22222222"]) == " sin_reajuste"

    def test_una_celda_que_no_coincide_con_el_volcado_aborta(self) -> None:
        """El registro o el volcado han cambiado por separado."""
        calidad, experimentos, volcado = _caso()
        calidad[0]["error_duplicate"] = "9999"
        with pytest.raises(ValueError, match="no coinciden con el volcado"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_una_cifra_sin_respaldo_en_el_volcado_aborta(self) -> None:
        """Una fila con reparto que el volcado no trae no tiene evidencia."""
        calidad, experimentos, volcado = _caso()
        volcado["evaluations"] = volcado["evaluations"][:1]
        with pytest.raises(ValueError, match="no lo respalda"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_una_evaluacion_del_volcado_sin_fila_aborta(self) -> None:
        calidad, experimentos, volcado = _caso()
        extra = dict(volcado["evaluations"][0])
        extra["primary_key"] = _clave("s_baseline_default_11111111", "onnxruntime", "fp32")
        volcado["evaluations"].append(extra)
        with pytest.raises(ValueError, match="sin fila con reparto"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_la_clave_corta_no_basta_para_casar(self) -> None:
        calidad, experimentos, volcado = _caso()
        volcado["evaluations"][0]["primary_key"] = _clave(
            "s_baseline_default_11111111", "openvino", "fp32", "ultralytics"
        )
        with pytest.raises(ValueError, match="no lo respalda"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_dos_evaluaciones_con_la_misma_clave_abortan(self) -> None:
        """Quedarse con una elegiría en silencio."""
        calidad, experimentos, volcado = _caso()
        volcado["evaluations"].append(volcado["evaluations"][0])
        with pytest.raises(ValueError, match="dos evaluaciones"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_un_hueco_sin_motivo_aborta(self) -> None:
        """Una fila que entró después del paso 34 no puede salir como hueco mudo."""
        calidad, experimentos, volcado = _caso()
        calidad[2]["notes"] = "evaluador=pycocotools"
        with pytest.raises(ValueError, match="motivos del paso 34"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_un_reparto_a_medias_aborta(self) -> None:
        """Cinco columnas de seis no son un reparto."""
        calidad, experimentos, volcado = _caso()
        calidad[0]["error_missed"] = ""
        with pytest.raises(ValueError, match="a medias"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_un_motivo_sin_etiqueta_aborta(self) -> None:
        """El pie no puede nombrar un hueco con una clave interna."""
        calidad, experimentos, volcado = _caso()
        volcado["reasons"]["motivo_nuevo"] = "otro texto"
        with pytest.raises(ValueError, match="sin etiqueta"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_la_cobertura_declarada_tiene_que_ser_la_medida(self) -> None:
        calidad, experimentos, volcado = _caso()
        volcado["coverage"]["by_route"] = {"curva": 2}
        with pytest.raises(ValueError, match="no es la que declara el volcado"):
            fr.error_decomposition_rows(calidad, experimentos, volcado)

    def test_el_pie_sale_de_los_datos(self) -> None:
        """Cada número del pie es de la cobertura, ninguno es una constante."""
        calidad, experimentos, volcado = _caso()
        pie = fr.error_decomposition_footer(
            fr.error_decomposition_rows(calidad, experimentos, volcado)
        )

        assert "Cubre 2 de las 4 filas de calidad" in pie
        assert "Las otras 2 no tienen reparto" in pie
        assert "destilación" in pie
        assert "(1 de 2; D-183)" in pie
        assert "0,25" in pie and "500 anotaciones" in pie
        # La PTQ no tiene reajuste porque no entrena; no se avisa.
        assert "sin_reajuste" not in pie and "antes de repararlas" not in pie

    def test_una_poda_sin_reajustar_se_avisa_aunque_la_etiqueta_no_lo_diga(self) -> None:
        """La marca de etapa salió de las etiquetas; el aviso no puede salir con ella."""
        calidad, experimentos, volcado = _caso()
        datos = fr.error_decomposition_rows(calidad, experimentos, volcado)
        fila = dict(datos.rows[0], technique="pruning", stage="sin_reajuste")
        pie = fr.error_decomposition_footer(
            fr.ErrorDecomposition(
                kinds=datos.kinds,
                rows=(fila, *datos.rows[1:]),
                quality_rows=datos.quality_rows,
                by_route=datos.by_route,
                missing_by_reason=datos.missing_by_reason,
                techniques_absent=datos.techniques_absent,
                confidence=datos.confidence,
                split=datos.split,
                num_gt=datos.num_gt,
            )
        )
        assert "antes de repararlas" in pie

    def test_sin_ruta_debil_el_pie_no_la_anuncia(self) -> None:
        """Una marca que la figura no dibuja no se explica en el pie."""
        calidad, experimentos, volcado = _caso()
        volcado["evaluations"][1]["route"] = "curva"
        volcado["coverage"]["by_route"] = {"curva": 2}
        pie = fr.error_decomposition_footer(
            fr.error_decomposition_rows(calidad, experimentos, volcado)
        )
        assert fr.WEAK_ROUTE_MARK not in pie

    def test_la_figura_marca_exactamente_las_de_la_ruta_debil(self) -> None:
        """Las veinte de la ruta débil no pueden parecer igual de fuertes."""
        import matplotlib.pyplot as plt

        calidad, experimentos, volcado = _caso()
        figura, resumen = fr.figure_error_decomposition(
            fr.error_decomposition_rows(calidad, experimentos, volcado), experimentos
        )
        textos = [t.get_text() for eje in figura.axes for t in eje.texts]
        pie = list(figura.texts)
        plt.close(figura)

        assert textos.count(fr.WEAK_ROUTE_MARK) == 1
        assert resumen["weak_route"] == ["s PTQ [openvino]"]
        assert resumen["coverage"]["drawn"] == 2
        # Sin pie dentro de la imagen; el calculado va al manifiesto.
        assert not pie
        assert "Cubre 2 de las 4 filas de calidad" in resumen["caption"]

    def test_sin_volcado_la_figura_no_sale_y_dice_por_que(self) -> None:
        with pytest.raises(ValueError, match="paso 34"):
            fr.figure_error_decomposition(None, {})

    def test_sin_filas_con_reparto_no_hay_figura(self) -> None:
        calidad, experimentos, volcado = _caso()
        datos = fr.error_decomposition_rows(calidad, experimentos, volcado)
        vacio = fr.ErrorDecomposition(
            kinds=datos.kinds,
            rows=(),
            quality_rows=4,
            by_route={},
            missing_by_reason={},
            techniques_absent=(),
            confidence=0.25,
            split="val",
            num_gt=500,
        )
        with pytest.raises(ValueError, match="nada que dibujar"):
            fr.figure_error_decomposition(vacio, experimentos)

    def test_olvidar_el_reparto_no_es_una_forma_de_no_dibujarlo(self) -> None:
        import inspect

        parametro = inspect.signature(fr.save_results_figures).parameters["error_decomposition"]
        assert parametro.kind is inspect.Parameter.KEYWORD_ONLY
        assert parametro.default is inspect.Parameter.empty

    def test_sobre_la_campana_real_la_cobertura_es_la_del_volcado(self) -> None:
        """El cruce real no aborta y cuenta lo que el paso 34 declaró."""
        from tfm_edge.evaluation.error_decomposition import DUMP_SUBDIR, dump_name
        from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE

        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        experimentos, _ = registro.valid_rows("experiments")
        volcado = json.loads(
            (RAIZ / "results" / "raw" / DUMP_SUBDIR / dump_name(REFERENCE_CONFIDENCE)).read_text(
                encoding="utf-8"
            )
        )
        datos = fr.error_decomposition_rows(
            calidad, {f["experiment_id"]: f for f in experimentos}, volcado
        )

        assert len(datos.rows) == volcado["coverage"]["with_decomposition"] > 0
        assert datos.by_route == volcado["coverage"]["by_route"]
        assert sum(datos.missing_by_reason.values()) == volcado["coverage"]["without_decomposition"]

    def test_el_manifiesto_versionado_trae_la_cobertura_de_hoy(self) -> None:
        """Si el registro o el volcado cambian sin regenerar la figura, falla aquí."""
        from tfm_edge.evaluation.error_decomposition import DUMP_SUBDIR, dump_name
        from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE

        ruta = RAIZ / "results" / "raw" / "analysis" / "results_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 30 no se ha ejecutado en este árbol.")
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        experimentos, _ = registro.valid_rows("experiments")
        volcado = json.loads(
            (RAIZ / "results" / "raw" / DUMP_SUBDIR / dump_name(REFERENCE_CONFIDENCE)).read_text(
                encoding="utf-8"
            )
        )
        datos = fr.error_decomposition_rows(
            calidad, {f["experiment_id"]: f for f in experimentos}, volcado
        )

        assert manifiesto["figures"]["tipos_de_error"]["coverage"] == datos.coverage()


#: Ficheros que repetían a mano el recuento de filas de calidad.
FICHEROS_CON_RECUENTO = (
    "scripts/30_build_results_figures.py",
    "src/tfm_edge/reporting/__init__.py",
    "src/tfm_edge/reporting/build_report.py",
    "src/tfm_edge/reporting/figures_results.py",
)


@pytest.mark.parametrize("relativa", FICHEROS_CON_RECUENTO)
def test_ningun_texto_escribe_a_mano_cuantas_filas_de_calidad_hay(relativa: str) -> None:
    """El «56» entró correcto, se quedó en cuatro ficheros y dejó de serlo sin avisar."""
    import re

    texto = (RAIZ / relativa).read_text(encoding="utf-8")
    patron = re.compile(
        r"\b(\d+\s+filas\s+de\s+calidad|cincuenta\s+y\s+seis\s+filas)", re.IGNORECASE
    )
    assert not patron.search(texto), patron.search(texto)


class TestElManifiestoNoGuardaRutasDeEstaMaquina:
    """Los dos manifiestos grababan la ruta absoluta de quien los generó."""

    def test_una_ruta_del_proyecto_sale_relativa(self) -> None:
        """Lo que el manifiesto tiene que guardar para que valga en otro clon."""
        destino = RAIZ / "results" / "figures" / "fig_5_1_ejemplo.pdf"

        assert style.manifest_files({"pdf": destino}) == {
            "pdf": "results/figures/fig_5_1_ejemplo.pdf"
        }

    def test_una_ruta_de_fuera_del_arbol_se_queda_absoluta(self, tmp_path: Path) -> None:
        """Falsear la relatividad sería peor que decir que el fichero está fuera."""
        fuera = tmp_path / "tablas" / "tabla_5_1.csv"

        assert style.manifest_files({"csv": fuera}) == {"csv": fuera.resolve().as_posix()}

    @pytest.mark.parametrize(
        "modulo",
        ["src/tfm_edge/reporting/figures_results.py", "src/tfm_edge/reporting/build_report.py"],
    )
    def test_los_dos_pasos_llaman_a_la_misma_regla(self, modulo: str) -> None:
        arbol = ast.parse((RAIZ / modulo).read_text(encoding="utf-8"))
        asignaciones = [
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Assign)
            and len(nodo.targets) == 1
            and isinstance(nodo.targets[0], ast.Subscript)
            and isinstance(nodo.targets[0].value, ast.Name)
            and nodo.targets[0].value.id == "resumen"
            and isinstance(nodo.targets[0].slice, ast.Constant)
            and nodo.targets[0].slice.value == "files"
        ]

        assert len(asignaciones) == 1, f"{len(asignaciones)} asignaciones a resumen['files']"
        valor = asignaciones[0].value
        assert isinstance(valor, ast.Call) and isinstance(
            valor.func, ast.Attribute
        ), "resumen['files'] no sale de una llamada a style.manifest_files"
        assert (
            valor.func.attr == "manifest_files"
        ), f"resumen['files'] se escribe con {valor.func.attr}, no con manifest_files"


class TestSobreLaCampanaReal:
    """Anclado a lo que hay en el repositorio."""

    def test_la_curva_del_baseline_principal_ya_esta(self) -> None:
        """Este test decía lo contrario, y por eso saltó cuando cambió el hecho."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")

        inventario = fr.training_curve_inventory(RAIZ / "models", experimentos)
        faltan = {identificador for identificador, _ in inventario.missing}

        assert inventario.curves, "No se ha encontrado ninguna curva en models/."
        assert "s_baseline_default_745e3d08" not in faltan

    def test_siguen_faltando_las_de_poda_y_qat(self) -> None:
        """El mecanismo tiene que seguir declarando lo que sí falta."""
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")

        inventario = fr.training_curve_inventory(RAIZ / "models", experimentos)
        faltan = {identificador for identificador, _ in inventario.missing}

        assert faltan, "Sin ausencias, el pie de la figura dejaría de declararlas."
        assert all("pruning" in i or "qat" in i for i in faltan), sorted(faltan)

    def test_el_manifiesto_recoge_lo_generado_y_lo_omitido(self) -> None:
        """El paso 30 deja escrito qué salió y qué no, para poder comprobarlo."""
        ruta = RAIZ / "results" / "raw" / "analysis" / "results_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 30 no se ha ejecutado en este árbol.")
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))

        assert set(manifiesto["figures"]) <= set(fr.PLACEMENT)
        assert manifiesto["training_curves"]["missing"]

    def test_la_figura_versionada_cubre_la_campana_de_hoy(self) -> None:
        ruta = RAIZ / "results" / "raw" / "analysis" / "results_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 30 no se ha ejecutado en este árbol.")
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))

        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos, _ = registro.valid_rows("experiments")
        inventario = fr.training_curve_inventory(RAIZ / "models", experimentos)

        dibujadas = {curva["label"] for curva in manifiesto["training_curves"]["found"]}
        ausentes = {fila["experiment_id"] for fila in manifiesto["training_curves"]["missing"]}

        assert dibujadas == {curva.label for curva in inventario.curves}, (
            "El manifiesto y el registro no cuentan las mismas curvas: la figura "
            "versionada es de antes del último cambio del registro. Regenérala "
            "con 'python scripts/30_build_results_figures.py --on-existing "
            "overwrite', ábrela y mírala antes de comitearla."
        )
        assert ausentes == {
            identificador for identificador, _ in inventario.missing
        }, "El pie de la figura declara unas ausencias que no son las actuales."


def _cargar_paso_30() -> ModuleType:
    """Carga ``scripts/30_build_results_figures.py`` como módulo."""
    ruta = RAIZ / "scripts" / "30_build_results_figures.py"
    especificacion = importlib.util.spec_from_file_location("build_results_figures", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


class TestElIndiceDeCostesNoColapsaDosMedidas:
    """La clave del índice es más corta que la de la tabla, y eso se comprueba."""

    def test_dos_dispositivos_con_el_mismo_coste_no_son_un_conflicto(self) -> None:
        """El mismo artefacto se exportó una vez y se midió en dos sitios."""
        paso = _cargar_paso_30()
        filas = [
            {
                "experiment_id": "s_ptq_x_11111111",
                "runtime": "onnxruntime",
                "precision": "int8",
                "device_id": "workstation_cpu",
                "export_time_s": "36.07",
            },
            {
                "experiment_id": "s_ptq_x_11111111",
                "runtime": "onnxruntime",
                "precision": "int8",
                "device_id": "colab_t4_gpu",
                "export_time_s": "36.07",
            },
        ]

        indice = paso.costes_de_exportacion(filas)

        assert indice[("s_ptq_x_11111111", "onnxruntime", "int8")] == 36.07

    def test_dos_costes_distintos_bajo_la_misma_clave_abortan(self) -> None:
        """Quedarse con uno elegiría en silencio cuál de las dos medidas cuenta."""
        paso = _cargar_paso_30()
        filas = [
            {
                "experiment_id": "s_ptq_x_11111111",
                "runtime": "onnxruntime",
                "precision": "int8",
                "device_id": "workstation_cpu",
                "export_time_s": "36.07",
            },
            {
                "experiment_id": "s_ptq_x_11111111",
                "runtime": "onnxruntime",
                "precision": "int8",
                "device_id": "colab_t4_gpu",
                "export_time_s": "99.0",
            },
        ]

        with pytest.raises(ValueError, match="costes de exportación distintos"):
            paso.costes_de_exportacion(filas)

    def test_una_celda_vacia_no_entra_en_el_indice(self) -> None:
        paso = _cargar_paso_30()
        filas = [
            {
                "experiment_id": "s_baseline_default_00000000",
                "runtime": "pytorch",
                "precision": "fp32",
                "device_id": "workstation_cpu",
                "export_time_s": "",
            }
        ]

        assert paso.costes_de_exportacion(filas) == {}

    def test_sobre_el_registro_real_las_combinaciones_tienen_coste(self) -> None:
        paso = _cargar_paso_30()
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        despliegue, _ = registro.latency_comparable_rows()

        indice = paso.costes_de_exportacion(despliegue)
        combinadas = [clave for clave in indice if clave[0].startswith("s_combined_")]

        assert combinadas, "Ninguna combinación tiene coste de exportación registrado."
        assert all(indice[clave] > 0 for clave in combinadas)


def _fila_coste(etiqueta: str, estado: str, aceleracion: float | None) -> dict:
    """Fila de prueba de la figura de coste frente a ganancia."""
    return {
        "label": etiqueta,
        "technique": "pruning",
        "cost_hours": 0.9,
        "speedup": aceleracion,
        "speedup_status": estado,
        "quality_delta": -0.02,
    }


class TestLaFiguraDeCosteDistingueSinLatenciaDeSinCertificar:
    """«Sin latencia» y «sin certificar» son dos razones distintas."""

    def test_el_pie_y_el_manifiesto_cuentan_las_dos_por_separado(self) -> None:
        """Cada estado va a su lista, y las tres suman las filas dibujadas."""
        import matplotlib.pyplot as plt

        filas = [
            _fila_coste("a [onnxruntime/fp32]", fr.SPEEDUP_CERTIFIED, 2.0),
            _fila_coste("b [onnxruntime/fp32]", fr.SPEEDUP_NOT_CERTIFIED, None),
            _fila_coste("c [pytorch/fp32]", fr.SPEEDUP_NO_LATENCY, None),
            _fila_coste("d [pytorch/fp32]", fr.SPEEDUP_NO_LATENCY, None),
        ]

        figura, resumen = fr.figure_cost_vs_gain(filas, [])
        pie = " ".join(t.get_text() for t in figura.texts)
        plt.close(figura)

        assert resumen["without_latency"] == ["c [pytorch/fp32]", "d [pytorch/fp32]"]
        assert resumen["not_certified"] == ["b [onnxruntime/fp32]"]
        assert "3 configuraciones aparecen solo en el panel derecho" in pie
        assert "2 porque no tienen latencia medida" in pie
        assert "1 porque tienen latencia medida" in pie
        assert "ningún motor" not in pie

    def test_sin_ninguna_por_certificar_el_pie_lo_dice(self) -> None:
        """Que no falte ninguna por el suelo también es información."""
        import matplotlib.pyplot as plt

        filas = [
            _fila_coste("a [onnxruntime/fp32]", fr.SPEEDUP_CERTIFIED, 2.0),
            _fila_coste("c [pytorch/fp32]", fr.SPEEDUP_NO_LATENCY, None),
        ]

        figura, resumen = fr.figure_cost_vs_gain(filas, [])
        pie = " ".join(t.get_text() for t in figura.texts)
        plt.close(figura)

        assert resumen["not_certified"] == []
        assert "Ninguna falta por no estar certificada" in pie

    def test_una_fila_sin_estado_es_un_error(self) -> None:
        """Sin estado no se puede decir por qué falta del panel izquierdo."""
        fila = _fila_coste("a [onnxruntime/fp32]", fr.SPEEDUP_CERTIFIED, 2.0)
        del fila["speedup_status"]

        with pytest.raises(ValueError, match="estado de aceleración"):
            fr.figure_cost_vs_gain([fila], [])

    def test_sobre_el_volcado_real_las_etiquetas_son_unicas(self) -> None:
        ruta = RAIZ / "results" / "raw" / "analysis" / "results_analysis.json"
        if not ruta.is_file():
            pytest.skip("El paso 29 no se ha ejecutado en este árbol.")
        paso = _cargar_paso_30()
        analisis = json.loads(ruta.read_text(encoding="utf-8"))
        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        experimentos = {f["experiment_id"]: f for f in registro.valid_rows("experiments")[0]}
        despliegue, _ = registro.latency_comparable_rows()

        dibujables, excluidas = paso.filas_coste(
            analisis, experimentos, paso.costes_de_exportacion(despliegue)
        )
        etiquetas = [f["label"] for f in dibujables] + [e for e, _ in excluidas]

        assert len(etiquetas) == len(set(etiquetas))
        assert all(f["speedup_status"] in fr.SPEEDUP_STATUSES for f in dibujables)
        certificadas = [f for f in dibujables if f["speedup_status"] == fr.SPEEDUP_CERTIFIED]
        assert all(f["speedup"] is not None for f in certificadas)
        assert all(f["speedup"] is None for f in dibujables if f not in certificadas)

    def test_el_manifiesto_versionado_reparte_todas_las_dibujadas(self) -> None:
        """Panel izquierdo, sin latencia y sin certificar suman lo dibujado, sin solaparse."""
        ruta = RAIZ / "results" / "raw" / "analysis" / "results_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 30 no se ha ejecutado en este árbol.")
        entrada = json.loads(ruta.read_text(encoding="utf-8"))["figures"]["coste_frente_a_ganancia"]

        dibujadas = entrada["drawn"]
        sin_latencia = set(entrada["without_latency"])
        sin_certificar = set(entrada["not_certified"])

        assert len(dibujadas) == len(set(dibujadas))
        assert not sin_latencia & sin_certificar
        assert sin_latencia | sin_certificar <= set(dibujadas)


def _volcado_29() -> dict:
    """El volcado versionado del paso 29."""
    return json.loads(
        (RAIZ / "results" / "raw" / "analysis" / "results_analysis.json").read_text(
            encoding="utf-8"
        )
    )


def _campana_real() -> tuple[list[dict], dict[str, dict[str, str]], tuple[dict, ...]]:
    """Filas del paso 29, experimentos indexados y compresiones del 6.2, de la campaña real."""
    registro = ExperimentRegistry(RAIZ / "results" / "registry")
    calidad, _ = analysis_quality_rows(registro)
    experimentos, _ = registro.valid_rows("experiments")
    indice = {f["experiment_id"]: f for f in experimentos}
    filas = _volcado_29()["degradation"]["rows"]
    return filas, indice, fr.compression_losses(filas, indice, calidad)


def _figura_por_clase_real(motor: str = "onnxruntime"):
    """La figura por clase de un motor sobre la campaña real, con sus dos entradas."""
    registro = ExperimentRegistry(RAIZ / "results" / "registry")
    calidad, _ = analysis_quality_rows(registro)
    filas, indice, compresiones = _campana_real()
    instancias = fr.class_instances(RAIZ / "results" / "raw" / "coco_eval")
    dispositivos = fr.comparison_devices(filas, calidad)
    return (
        filas,
        instancias,
        fr.figure_class_degradation(
            filas, instancias, dispositivos, motor, compression=compresiones, experiments=indice
        ),
    )


def _normal(nombre: str) -> str:
    """Nombre de clase comparable entre el COCO y las columnas del registro."""
    return nombre.lower().replace("-", " ").replace("_", " ")


class TestRetencionPorClase:
    """La figura por clase colorea la retención y raya las clases escasas."""

    def test_las_rayadas_son_las_de_menos_de_150_contadas_en_el_coco(self) -> None:
        """Se cuenta el COCO directamente, sin pasar por pycocotools ni el volcado."""
        import matplotlib.pyplot as plt

        coco_ruta = RAIZ / "data" / "processed" / "sh17_coco.json"
        val_ruta = RAIZ / "data" / "splits" / "val.txt"
        if not (coco_ruta.is_file() and val_ruta.is_file()):
            pytest.skip("Sin data/processed/sh17_coco.json o data/splits/val.txt en este árbol.")
        coco = json.loads(coco_ruta.read_text(encoding="utf-8"))
        val = {
            Path(linea.strip()).stem
            for linea in val_ruta.read_text(encoding="utf-8").splitlines()
            if linea.strip()
        }
        imagenes = {im["id"] for im in coco["images"] if Path(im["file_name"]).stem in val}
        nombres = {c["id"]: c["name"] for c in coco["categories"]}
        cuenta: dict[str, int] = {}
        for anotacion in coco["annotations"]:
            if anotacion["image_id"] in imagenes:
                nombre = _normal(nombres[anotacion["category_id"]])
                cuenta[nombre] = cuenta.get(nombre, 0) + 1

        _, instancias, (figura, resumen) = _figura_por_clase_real()
        plt.close(figura)

        esperadas = sorted(n for n, k in cuenta.items() if k < fr.MIN_INTERPRETABLE_INSTANCES)
        assert sorted(_normal(c) for c in resumen["scarce_classes"]) == esperadas
        assert {_normal(c): k for c, k in resumen["instances"].items()} == cuenta
        assert {_normal(n): k for n, k in instancias.by_index.values()} == cuenta

    def test_la_retencion_dibujada_es_la_del_volcado(self) -> None:
        """Cada celda es valor / referencia, y coincide con 1 + relative del volcado."""
        import matplotlib.pyplot as plt

        for motor in fr.CLASS_FIGURE_PLACEMENT:
            filas, _, (figura, resumen) = _figura_por_clase_real(motor)
            matrices = [eje.images[0].get_array() for eje in figura.axes if eje.images]
            plt.close(figura)

            clases = resumen["classes"]
            assert len(matrices) == 1
            matriz = matrices[0]
            del_motor = [
                next(
                    f
                    for f in filas
                    if (f["experiment_id"], f["runtime"], f["precision"]) == (e, motor, p)
                )
                for e, p in zip(resumen["rows"], resumen["row_precisions"])
            ]
            assert matriz.shape == (len(del_motor), len(clases))
            for i, fila in enumerate(del_motor):
                for j, clase in enumerate(clases):
                    delta = fila["per_class"][clase]
                    assert matriz[i, j] == pytest.approx(delta["value"] / delta["reference"])
                    assert matriz[i, j] == pytest.approx(1.0 + delta["relative"])

    def test_el_pie_cuenta_de_los_datos_que_filas_cruzan_de_maquina(self) -> None:
        import matplotlib.pyplot as plt

        pies, escalas = {}, {}
        for motor in fr.CLASS_FIGURE_PLACEMENT:
            _, _, (figura, resumen) = _figura_por_clase_real(motor)
            assert not figura.texts, f"{motor}: el pie va en el texto, no en la imagen"
            pies[motor] = resumen["caption"]
            escalas[motor] = tuple(resumen["scale"])
            plt.close(figura)

        assert len(set(pies.values())) == 1
        assert len(set(escalas.values())) == 1
        pie = pies["pytorch"]
        parejas = resumen["devices"]["pytorch"]
        cruzan = sum(n for par, n in parejas.items() if len(set(par.split(" -> "))) == 2)
        assert cruzan, "Si ninguna fila cruza de máquina, el pie no tiene nada que avisar."
        assert f"En la figura de PyTorch, {cruzan} de sus {sum(parejas.values())} filas" in pie
        assert "Rayadas, y sin interpretar" in pie

    def test_la_escala_sigue_siendo_de_0_a_150(self) -> None:
        import matplotlib.pyplot as plt

        for motor in fr.CLASS_FIGURE_PLACEMENT:
            _, _, (figura, resumen) = _figura_por_clase_real(motor)
            plt.close(figura)
            assert resumen["scale"] == [0.0, 1.5], motor

    def test_el_cuerpo_tiene_las_filas_del_6_2_en_su_orden(self) -> None:
        """Las mismas compresiones, orden y etiquetas que la pérdida por compresión."""
        import matplotlib.pyplot as plt

        _, _, compresiones = _campana_real()
        esperadas = [e["label"] for e in fr.ordered_compressions(compresiones)]
        for motor in fr.COMPRESSION_RUNTIMES:
            _, _, (figura, resumen) = _figura_por_clase_real(motor)
            eje = next(e for e in figura.axes if e.images)
            dibujadas = [t.get_text() for t in eje.get_yticklabels()]
            plt.close(figura)

            assert resumen["labels"] == esperadas == dibujadas
            precision = {e["label"]: e["runtimes"][motor]["precision"] for e in compresiones}
            assert resumen["row_precisions"] == [precision[e] for e in esperadas]
            # Ningún QAT en FP32: es el modelo antes de materializar, no una compresión.
            qat = [p for e, p in zip(resumen["rows"], resumen["row_precisions"]) if "_qat_" in e]
            assert qat and set(qat) == {"int8"}

    def test_el_anexo_de_pytorch_se_lee_sin_el_registro_delante(self) -> None:
        """Sin hash, sin corchetes y sin guiones bajos; cada etiqueta, una fila."""
        import matplotlib.pyplot as plt

        _, _, (figura, resumen) = _figura_por_clase_real("pytorch")
        plt.close(figura)

        etiquetas = resumen["labels"]
        assert len(etiquetas) == len(set(etiquetas)) == len(resumen["rows"])
        for etiqueta in etiquetas:
            assert "[" not in etiqueta and "_" not in etiqueta, etiqueta
        crudas = [e for e in etiquetas if e.endswith(fr.RAW_PRUNING_SUFFIX)]
        assert len(crudas) == sum(1 for e in resumen["rows"] if "_pruning_" in e) // 2
        assert {e for e in etiquetas if e.startswith(("Destilación", "Control"))} == {
            "Destilación l→n",
            "Destilación l→n, piloto 10 ép.",
            "Control α=0 l→n",
            "Control α=0 l→n, piloto 10 ép.",
        }
        assert all(e.endswith(fr.QAT_FP32_SUFFIX) for e in etiquetas if "QAT" in e)

    def test_las_cifras_caben_en_su_celda_a_16_cm(self) -> None:
        """La figura se dibuja a ancho de página, y cada cifra cabe en su celda."""
        import matplotlib.pyplot as plt

        for motor in fr.CLASS_FIGURE_PLACEMENT:
            _, _, (figura, resumen) = _figura_por_clase_real(motor)
            assert figura.get_figwidth() == pytest.approx(16.0 / 2.54)
            figura.canvas.draw()
            dibujo = figura.canvas.get_renderer()
            # El guardado recorta a lo dibujado: si algo se saliera del lienzo,
            # la figura impresa mediría más de 16 cm.
            contenido = figura.get_tightbbox(dibujo)
            assert contenido.x0 >= 0.0
            assert contenido.x1 <= figura.get_figwidth() + 1e-6
            eje = next(e for e in figura.axes if e.images)
            caja = eje.get_window_extent(dibujo)
            ancho_celda = caja.width / len(resumen["classes"])
            alto_celda = caja.height / len(resumen["rows"])
            cifras = [t for t in eje.texts if t.get_text().isdigit()]
            assert len(cifras) == len(resumen["classes"]) * len(resumen["rows"])
            for cifra in cifras:
                extension = cifra.get_window_extent(dibujo)
                assert extension.width < ancho_celda
                assert extension.height < alto_celda
            plt.close(figura)

    def test_un_motor_sin_filas_no_dibuja_figura(self) -> None:
        """Una figura vacía no dice nada; el paso la deja en ``skipped`` con el motivo."""
        fila = {
            "experiment_id": "s_ptq_x_11111111",
            "runtime": "onnxruntime",
            "precision": "int8",
            "per_class": {"ear": {"column": "ap_class_01_ear", "value": 0.3, "reference": 0.6}},
        }
        instancias = fr.ClassInstances(split="val", by_index={1: ("ear", 149)}, sources=("x",))
        with pytest.raises(ValueError, match="no tiene ninguna fila"):
            fr.figure_class_degradation([fila], instancias, {}, "openvino")

    def test_pytorch_va_al_anexo_y_hay_una_clave_por_motor(self) -> None:
        """Las tres figuras van al apartado 5.5; la de PyTorch, al anexo."""
        assert fr.CLASS_FIGURE_PLACEMENT["pytorch"] == "anexo"
        for motor in fr.CLASS_FIGURE_PLACEMENT:
            assert fr.PLACEMENT[f"degradacion_por_clase_{motor}"] == (5, 5)
        assert "degradacion_por_clase" not in fr.PLACEMENT

    def test_las_escasas_salen_del_umbral_y_no_de_una_lista(self) -> None:
        """Con recuentos inventados, las rayadas son las que quedan bajo el umbral."""
        import matplotlib.pyplot as plt

        fila = {
            "experiment_id": "s_ptq_x_11111111",
            "runtime": "onnxruntime",
            "precision": "int8",
            "stage": "sin_reajuste",
            "overall": {"map_50_95": {"value": 0.25}},
            "per_class": {
                "person": {"column": "ap_class_00_person", "value": 0.4, "reference": 0.5},
                "ear": {"column": "ap_class_01_ear", "value": 0.3, "reference": 0.6},
            },
        }
        compresion = {
            "label": "PTQ",
            "experiment_id": "s_ptq_x_11111111",
            "runtimes": {"onnxruntime": {"precision": "int8", "value": 0.25, "delta": -0.1}},
        }
        instancias = fr.ClassInstances(
            split="val", by_index={0: ("person", 151), 1: ("ear", 149)}, sources=("x.json",)
        )
        dispositivos = {fr.row_label(fila): ("workstation_cpu", "workstation_cpu")}
        figura, resumen = fr.figure_class_degradation(
            [fila], instancias, dispositivos, "onnxruntime", compression=[compresion]
        )
        plt.close(figura)

        assert resumen["scarce_classes"] == ["ear"]
        assert resumen["scale"][0] == 0.0
        assert resumen["labels"] == ["PTQ"]

    def test_sin_compresiones_el_cuerpo_no_sabe_sus_filas(self) -> None:
        """Elegirlas de otra forma volvería a meter los QAT en FP32."""
        fila = {
            "experiment_id": "s_ptq_x_11111111",
            "runtime": "onnxruntime",
            "precision": "int8",
            "per_class": {"ear": {"column": "ap_class_01_ear", "value": 0.3, "reference": 0.6}},
        }
        instancias = fr.ClassInstances(split="val", by_index={1: ("ear", 149)}, sources=("x",))
        with pytest.raises(ValueError, match="figura del 6.2"):
            fr.figure_class_degradation([fila], instancias, {}, "onnxruntime")

    def test_sin_recuentos_la_figura_no_sale(self) -> None:
        """Rayar sin el recuento sería escribirlo a mano."""
        fila = {
            "runtime": "pytorch",
            "per_class": {"ear": {"column": "ap_class_01_ear", "value": 0.3, "reference": 0.6}},
        }
        with pytest.raises(ValueError, match="anotaciones por clase"):
            fr.figure_class_degradation([fila], None, {})

    def test_dos_volcados_que_discrepan_abortan(self, tmp_path: Path) -> None:
        """Quedarse con uno elegiría en silencio contra qué anotaciones se cuenta."""
        for nombre, n in (("a_val_640.json", 19), ("b_val_640.json", 20)):
            contenido = {
                "split": "val",
                "per_class": {"4": {"name": "face-guard", "num_annotations": n}},
            }
            (tmp_path / nombre).write_text(json.dumps(contenido), encoding="utf-8")
        with pytest.raises(ValueError, match="no dan el mismo número"):
            fr.class_instances(tmp_path)

    def test_los_de_humo_no_cuentan(self, tmp_path: Path) -> None:
        """Un volcado de humo evalúa un subconjunto a propósito."""
        for nombre, n in (("a_val_640.json", 9), ("a_val_640_smoke.json", 1)):
            contenido = {
                "split": "val",
                "per_class": {"0": {"name": "person", "num_annotations": n}},
            }
            (tmp_path / nombre).write_text(json.dumps(contenido), encoding="utf-8")
        assert fr.class_instances(tmp_path).by_index == {0: ("person", 9)}


def _fila_tamano(motor: str, valores: dict[str, tuple[float, float]], criterio: str) -> dict:
    """Fila mínima del paso 29 con desglose por tamaño, para la figura por tamaño."""
    return {
        "experiment_id": "s_ptq_int8-noexcl_bbbbbbbb",
        "runtime": motor,
        "precision": "int8",
        "overall": {"map_50_95": {"value": 0.3}},
        "per_size": {
            r: {"value": v, "reference": ref, "column": f"ap_{r}_{criterio}"}
            for r, (v, ref) in valores.items()
        },
    }


def _compresion_tamano() -> dict:
    """Compresión de :func:`_fila_tamano`, como la devuelve ``compression_losses``."""
    return {
        "label": "PTQ",
        "experiment_id": "s_ptq_int8-noexcl_bbbbbbbb",
        "reference": "s_baseline_default_aaaaaaaa",
        "runtimes": {
            m: {"precision": "int8", "value": 0.3, "delta": -0.05} for m in fr.COMPRESSION_RUNTIMES
        },
    }


class TestRetencionPorTamano:
    """Figura por tamaño: las compresiones del 6.2 en retención, a 16 × 12 cm."""

    @staticmethod
    def _real():
        filas, _, compresiones = _campana_real()
        figura, resumen = fr.figure_size_degradation(filas, compresiones)
        return filas, compresiones, figura, resumen

    def test_las_filas_son_las_del_6_2_en_su_orden_y_solo_a_la_izquierda(self) -> None:
        import matplotlib.pyplot as plt

        _, compresiones, figura, resumen = self._real()
        izquierda, derecha = figura.axes[0], figura.axes[1]
        rotulos = [t.get_text() for t in izquierda.get_yticklabels()]
        a_la_derecha = [t for t in derecha.get_yticklabels() if t.get_visible() and t.get_text()]
        titulos = [e.get_title() for e in figura.axes]
        plt.close(figura)

        esperadas = [e["label"] for e in fr.ordered_compressions(compresiones)]
        assert [f["label"] for f in resumen["rows"]] == esperadas == rotulos
        assert not a_la_derecha
        assert titulos == ["ONNX Runtime", "OpenVINO"]

    def test_cada_retencion_es_la_del_volcado_con_el_criterio_de_inferencia(self) -> None:
        """Valor entre referencia de la fila del paso 29, y de la columna de inferencia."""
        import matplotlib.pyplot as plt

        filas, compresiones, figura, resumen = self._real()
        plt.close(figura)
        por_etiqueta = {e["label"]: e for e in compresiones}
        for fila in resumen["rows"]:
            entrada = por_etiqueta[fila["label"]]
            for motor in fr.COMPRESSION_RUNTIMES:
                volcado = fr.compression_row(filas, entrada, motor)
                assert fila[motor]["precision"] == entrada["runtimes"][motor]["precision"]
                for rango in fr.SIZE_RANGES:
                    celda = fila[motor]["sizes"][rango]
                    delta = volcado["per_size"][rango]
                    assert celda["retention"] == pytest.approx(delta["value"] / delta["reference"])
                    assert celda["column"].endswith("_inference_res")

    def test_la_ptq_de_s_en_ort_retiene_la_mitad_de_lo_pequeno(self) -> None:
        import matplotlib.pyplot as plt

        _, _, figura, resumen = self._real()
        plt.close(figura)
        ptq = next(f for f in resumen["rows"] if f["label"] == "PTQ")
        assert ptq["onnxruntime"]["sizes"]["small"]["retention"] == pytest.approx(0.4968, abs=1e-3)

    def test_cabe_en_16_por_12_cm_sin_pie_y_ningun_texto_baja_de_7_pt(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.text import Text

        _, _, figura, resumen = self._real()
        figura.canvas.draw()
        contenido = figura.get_tightbbox(figura.canvas.get_renderer())
        textos = [t for t in figura.findobj(Text) if t.get_visible() and t.get_text().strip()]
        horizontales = [
            linea for eje in figura.axes for linea in eje.get_ygridlines() if linea.get_visible()
        ]
        ancho, alto, sueltos = figura.get_figwidth(), figura.get_figheight(), figura.texts
        plt.close(figura)

        assert ancho == pytest.approx(16.0 / 2.54)
        assert alto <= 12.0 / 2.54 + 1e-9
        assert contenido.x0 >= 0.0 and contenido.x1 <= ancho + 1e-6
        assert contenido.y0 >= 0.0 and contenido.y1 <= alto + 1e-6
        assert min(t.get_fontsize() for t in textos) >= 7.0
        assert not sueltos
        assert not horizontales
        assert resumen["row_pitch_cm"] <= 0.5 + 1e-9

    def test_los_marcadores_son_huecos_para_que_no_se_tapen(self) -> None:
        """Dos tamaños que retienen casi lo mismo caen en el mismo sitio."""
        import matplotlib.pyplot as plt

        _, _, figura, _ = self._real()
        colecciones = [c for eje in figura.axes for c in eje.collections]
        plt.close(figura)

        assert len(colecciones) == 19 * len(fr.SIZE_RANGES) * len(fr.COMPRESSION_RUNTIMES)
        for coleccion in colecciones:
            caras = coleccion.get_facecolors()
            assert len(caras) == 0 or all(c[3] == 0 for c in caras)
            assert coleccion.get_sizes()[0] == fr.SIZE_MARKER_AREA

    def test_el_eje_contiene_todos_los_marcadores_y_el_100(self) -> None:
        import matplotlib.pyplot as plt

        _, _, figura, resumen = self._real()
        plt.close(figura)
        izquierda, derecha = resumen["x_range"]
        valores = [
            100 * f[m]["sizes"][r]["retention"]
            for f in resumen["rows"]
            for m in fr.COMPRESSION_RUNTIMES
            for r in fr.SIZE_RANGES
        ]
        assert izquierda < min(valores) and derecha > max(max(valores), 100.0)

    def test_una_retencion_por_encima_del_100_ensancha_el_eje(self) -> None:
        import matplotlib.pyplot as plt

        valores = {"small": (0.30, 0.20), "medium": (0.4, 0.5), "large": (0.5, 0.6)}
        filas = [_fila_tamano(m, valores, "inference_res") for m in fr.COMPRESSION_RUNTIMES]
        figura, resumen = fr.figure_size_degradation(filas, [_compresion_tamano()])
        limite = figura.axes[0].get_xlim()[1]
        plt.close(figura)

        assert resumen["x_range"][1] == limite > 150.0

    def test_un_rango_que_falta_o_de_otro_criterio_aborta(self) -> None:
        """Un marcador que falta sin explicación se leería como un cero."""
        incompleto = {"small": (0.1, 0.2), "medium": (0.4, 0.5)}
        filas = [_fila_tamano(m, incompleto, "inference_res") for m in fr.COMPRESSION_RUNTIMES]
        with pytest.raises(ValueError, match="no tiene AP de objetos 'large'"):
            fr.figure_size_degradation(filas, [_compresion_tamano()])

        completo = {"small": (0.1, 0.2), "medium": (0.4, 0.5), "large": (0.5, 0.6)}
        filas = [_fila_tamano(m, completo, "original_res") for m in fr.COMPRESSION_RUNTIMES]
        with pytest.raises(ValueError, match="no es el criterio"):
            fr.figure_size_degradation(filas, [_compresion_tamano()])

    def test_sin_compresiones_no_hay_figura(self) -> None:
        with pytest.raises(ValueError, match="figura del 6.2"):
            fr.figure_size_degradation([], None)

    def test_el_manifiesto_versionado_trae_la_figura_de_hoy(self) -> None:
        """Si el registro cambia sin regenerar la figura, falla aquí."""
        ruta = RAIZ / "results" / "raw" / "analysis" / "results_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 30 no se ha ejecutado en este árbol.")
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        _, _, compresiones = _campana_real()
        entrada = manifiesto["figures"]["degradacion_por_tamano"]
        assert [f["label"] for f in entrada["rows"]] == [
            e["label"] for e in fr.ordered_compressions(compresiones)
        ]
        assert entrada["criterion"] == fr.SIZE_CRITERION


class TestRotuladoresDelAnexo:
    """Las etiquetas legibles de lo que la figura del 6.2 no rotula."""

    @pytest.mark.parametrize(
        ("experimento", "etiqueta"),
        [
            ("n_distillation_l2n_combined_t2_ac9ff009", "Destilación l→n"),
            ("n_distillation_l2n_combined_t2_pilot10_cf761acd", "Destilación l→n, piloto 10 ép."),
            ("n_distillation_l2n_control_alpha0_f5c4a70e", "Control α=0 l→n"),
        ],
    )
    def test_las_destilaciones_se_rotulan_por_su_variante(
        self, experimento: str, etiqueta: str
    ) -> None:
        assert fr._distillation_label(experimento) == etiqueta

    def test_un_fragmento_desconocido_no_se_rotula(self) -> None:
        with pytest.raises(ValueError, match="no se sabe rotular"):
            fr._distillation_label("n_distillation_l2n_combined_t2_nuevo_aaaaaaaa")

    def test_la_fila_de_una_compresion_tiene_que_ser_una(self) -> None:
        """Dos filas con la misma clave no se eligen por orden de fichero."""
        fila = _fila_tamano("onnxruntime", {}, "inference_res")
        with pytest.raises(ValueError, match="2 filas"):
            fr.compression_row([fila, fila], _compresion_tamano(), "onnxruntime")


def _fila_reparto(tecnica: str, experimento: str, precision: str, padre: str) -> dict:
    """Fila mínima de la figura de tipos de error, para formar parejas."""
    return {
        "technique": tecnica,
        "experiment_id": experimento,
        "precision": precision,
        "parent_experiment_id": padre,
    }


class TestTiposDeErrorPorPares:
    """Cada INT8 junto a su FP32 de partida, por familia."""

    def test_las_parejas_se_atan_por_el_padre_o_por_el_mismo_experimento(self) -> None:
        """PTQ con su baseline, QAT con su FP32, combinada con su podado."""
        filas = [
            _fila_reparto("baseline", "s_base", "fp32", ""),
            _fila_reparto("ptq", "s_ptq", "int8", "s_base"),
            _fila_reparto("qat", "s_qat", "fp32", "s_base"),
            _fila_reparto("qat", "s_qat", "int8", "s_base"),
            _fila_reparto("pruning", "s_poda", "fp32", "s_base"),
            _fila_reparto("combined", "s_comb", "int8", "s_poda"),
        ]
        bloques = fr.error_pairs(filas)

        assert [familia for familia, _ in bloques] == ["ptq", "qat", "combined"]
        assert [[c[1:] for c in g] for _, grupos in bloques for g in grupos] == [
            [("s_base", "fp32"), ("s_ptq", "int8")],
            [("s_qat", "fp32"), ("s_qat", "int8")],
            [("s_poda", "fp32"), ("s_comb", "int8")],
        ]

    def test_una_fila_sin_pareja_va_aparte(self) -> None:
        """No se pega a una pareja que no es la suya."""
        filas = [
            _fila_reparto("baseline", "s_base", "fp32", ""),
            _fila_reparto("ptq", "s_ptq", "int8", "s_base"),
            _fila_reparto("baseline", "n_base", "fp32", ""),
        ]
        bloques = fr.error_pairs(filas)

        assert bloques[-1] == ("", [[("baseline", "n_base", "fp32")]])

    def test_la_campana_real_queda_toda_en_parejas(self) -> None:
        """Y el pie avisa si el eje de las no detectadas no empieza en cero."""
        import matplotlib.pyplot as plt

        from tfm_edge.evaluation.error_decomposition import DUMP_SUBDIR, dump_name
        from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE

        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        experimentos, _ = registro.valid_rows("experiments")
        volcado = json.loads(
            (RAIZ / "results" / "raw" / DUMP_SUBDIR / dump_name(REFERENCE_CONFIDENCE)).read_text(
                encoding="utf-8"
            )
        )
        indice = {f["experiment_id"]: f for f in experimentos}
        datos = fr.error_decomposition_rows(calidad, indice, volcado)
        figura, resumen = fr.figure_error_decomposition(datos, indice)
        plt.close(figura)

        assert all(familia for familia, _ in fr.error_pairs(datos.rows))
        assert 2 * len(resumen["pairs"]) == len(resumen["configurations"])
        for (fp32, int8), (e_fp32, e_int8) in zip(
            resumen["pairs"], zip(resumen["experiments"][::2], resumen["experiments"][1::2])
        ):
            assert e_fp32[1] == "fp32" and e_int8[1] == "int8"
            # La pareja del QAT es el mismo experimento: su FP32 lo dice.
            if e_fp32[0] == e_int8[0]:
                assert fp32 == int8 + fr.QAT_FP32_SUFFIX
        if resumen["missed_axis_start"] > 0:
            assert "no en cero" in resumen["caption"]

    def test_la_campana_real_cabe_a_16_cm_con_etiquetas_legibles(self) -> None:
        """Sin hash, sin precisión entre corchetes, sin etapa y sin pie."""
        import matplotlib.pyplot as plt
        from matplotlib.text import Text

        from tfm_edge.evaluation.error_decomposition import DUMP_SUBDIR, dump_name
        from tfm_edge.evaluation.operating_curve import REFERENCE_CONFIDENCE

        registro = ExperimentRegistry(RAIZ / "results" / "registry")
        calidad, _ = analysis_quality_rows(registro)
        experimentos, _ = registro.valid_rows("experiments")
        volcado = json.loads(
            (RAIZ / "results" / "raw" / DUMP_SUBDIR / dump_name(REFERENCE_CONFIDENCE)).read_text(
                encoding="utf-8"
            )
        )
        indice = {f["experiment_id"]: f for f in experimentos}
        figura, resumen = fr.figure_error_decomposition(
            fr.error_decomposition_rows(calidad, indice, volcado), indice
        )
        figura.canvas.draw()
        contenido = figura.get_tightbbox(figura.canvas.get_renderer())
        textos = [t for t in figura.findobj(Text) if t.get_visible() and t.get_text().strip()]
        leyenda = [t.get_text() for t in figura.legends[0].get_texts()]
        sin_pie = not figura.texts
        ancho, alto = figura.get_figwidth(), figura.get_figheight()
        plt.close(figura)

        assert ancho == pytest.approx(16.0 / 2.54)
        assert alto <= fr.ERROR_MAX_HEIGHT + 1e-9
        assert contenido.x0 >= 0.0 and contenido.x1 <= ancho + 1e-6
        assert min(t.get_fontsize() for t in textos) >= 6.5
        assert sin_pie
        assert not any("D-183" in t for t in leyenda)
        assert len(set(resumen["configurations"])) == len(resumen["configurations"])
        for etiqueta in resumen["configurations"]:
            assert "[" not in etiqueta and "_" not in etiqueta, etiqueta


def _fila_29(
    experiment_id: str,
    motor: str,
    precision: str,
    cadena: list[str],
    valor: float,
    referencia: float,
    tecnica: str,
    etapa: str = "reajustado",
) -> dict:
    """Fila de degradación con la forma del volcado del paso 29."""
    return {
        "experiment_id": experiment_id,
        "runtime": motor,
        "precision": precision,
        "technique": tecnica,
        "stage": etapa,
        "reference": {
            "experiment_id": cadena[-1],
            "chain": cadena,
            "precision": "fp32",
            "runtime": motor,
        },
        "overall": {
            "map_50_95": {
                "value": valor,
                "reference": referencia,
                "absolute": valor - referencia,
            }
        },
    }


def _calidad(experiment_id: str, motor: str, precision: str, valor: float) -> dict:
    """Fila de calidad del registro, de pycocotools sobre val."""
    return {
        "experiment_id": experiment_id,
        "runtime": motor,
        "precision": precision,
        "evaluator": "pycocotools",
        "split": "val",
        "device_id": "cpu",
        "map_50": "0.5",
        "map_50_95": str(valor),
    }


def _caso_compresion() -> tuple[list[dict], dict[str, dict], list[dict]]:
    """Una campaña mínima con las trampas de la selección."""
    base = "s_baseline_default_aaaaaaaa"
    experimentos = {
        base: {"technique": "baseline", "model_size": "s"},
        "s_ptq_int8-noexcl_bbbbbbbb": {"technique": "ptq", "model_size": "s"},
        "s_pruning_l1global_r25_cccccccc": {"technique": "pruning", "model_size": "s"},
        "s_pruning_l1global_r25_dddddddd": {"technique": "pruning", "model_size": "s"},
        "s_qat_int8-noexcl_eeeeeeee": {"technique": "qat", "model_size": "s"},
        "s_distillation_kd_ffffffff": {"technique": "distillation", "model_size": "s"},
    }
    valores = {
        ("s_ptq_int8-noexcl_bbbbbbbb", "int8"): (0.30, 0.32, "sin_reajuste"),
        ("s_pruning_l1global_r25_cccccccc", "fp32"): (0.38, 0.38, "reajustado"),
        ("s_pruning_l1global_r25_dddddddd", "fp32"): (0.01, 0.01, "sin_reajuste"),
        ("s_qat_int8-noexcl_eeeeeeee", "fp32"): (0.39, 0.39, "reajustado"),
        ("s_qat_int8-noexcl_eeeeeeee", "int8"): (0.31, 0.35, "reajustado"),
        ("s_distillation_kd_ffffffff", "fp32"): (0.36, 0.36, "reajustado"),
    }
    filas, calidad = [], []
    for motor in fr.COMPRESSION_RUNTIMES:
        calidad.append(_calidad(base, motor, "fp32", 0.40))
        for (experimento, precision), (ort, ov, etapa) in valores.items():
            valor = ort if motor == "onnxruntime" else ov
            filas.append(
                _fila_29(
                    experimento,
                    motor,
                    precision,
                    [experimento, base],
                    valor,
                    0.40,
                    experimentos[experimento]["technique"],
                    etapa,
                )
            )
            calidad.append(_calidad(experimento, motor, precision, valor))
    return filas, experimentos, calidad


def _entrada(etiqueta: str, ort: float, ov: float) -> dict:
    """Compresión mínima para comparar órdenes."""
    return {
        "label": etiqueta,
        "experiment_id": etiqueta,
        "runtimes": {"onnxruntime": {"delta": ort}, "openvino": {"delta": ov}},
    }


class TestPerdidaPorCompresion:
    """Figura del 6.2: qué entra, contra qué se compara y en qué orden."""

    def test_la_ptq_entra_aunque_el_paso_29_la_marque_sin_reajuste(self) -> None:
        filas, experimentos, calidad = _caso_compresion()
        etiquetas = {e["label"] for e in fr.compression_losses(filas, experimentos, calidad)}

        assert etiquetas == {"PTQ", "L1 r25", "QAT"}

    def test_fuera_la_poda_sin_reajustar_y_la_destilacion(self) -> None:
        """Una poda en crudo no es el método, y la destilación no comprime."""
        filas, experimentos, calidad = _caso_compresion()
        ids = {e["experiment_id"] for e in fr.compression_losses(filas, experimentos, calidad)}

        assert "s_pruning_l1global_r25_dddddddd" not in ids
        assert "s_distillation_kd_ffffffff" not in ids

    def test_del_qat_se_queda_la_int8(self) -> None:
        filas, experimentos, calidad = _caso_compresion()
        qat = next(
            e for e in fr.compression_losses(filas, experimentos, calidad) if e["label"] == "QAT"
        )

        assert {m: d["precision"] for m, d in qat["runtimes"].items()} == {
            "onnxruntime": "int8",
            "openvino": "int8",
        }
        assert qat["runtimes"]["onnxruntime"]["delta"] == pytest.approx(0.31 - 0.40)

    def test_un_registro_que_no_es_el_volcado_aborta(self) -> None:
        """Si la reconstrucción con _pick_reference no da lo volcado, la figura no sale."""
        filas, experimentos, calidad = _caso_compresion()
        for fila in calidad:
            if fila["experiment_id"].startswith("s_ptq") and fila["runtime"] == "openvino":
                fila["map_50_95"] = "0.33"

        with pytest.raises(ValueError, match="no coincide con el volcado del paso 29"):
            fr.compression_losses(filas, experimentos, calidad)

    def test_otro_evaluador_aborta_en_vez_de_filtrarse(self) -> None:
        filas, experimentos, calidad = _caso_compresion()
        for fila in calidad:
            fila["evaluator"] = "ultralytics"

        with pytest.raises(ValueError, match="pycocotools"):
            fr.compression_losses(filas, experimentos, calidad)

    def test_la_referencia_tiene_que_ser_de_su_tamano(self) -> None:
        """Comparar una PTQ de nano con el FP32 de small no es su pérdida."""
        filas, experimentos, calidad = _caso_compresion()
        experimentos["s_ptq_int8-noexcl_bbbbbbbb"]["model_size"] = "n"

        with pytest.raises(ValueError, match="FP32 del baseline de su"):
            fr.compression_losses(filas, experimentos, calidad)

    def test_dos_etiquetas_iguales_abortan(self) -> None:
        filas, experimentos, calidad = _caso_compresion()
        otra = "s_ptq_int8-noexcl_99999999"
        experimentos[otra] = {"technique": "ptq", "model_size": "s"}
        for motor in fr.COMPRESSION_RUNTIMES:
            filas.append(
                _fila_29(
                    otra,
                    motor,
                    "int8",
                    [otra, "s_baseline_default_aaaaaaaa"],
                    0.29,
                    0.40,
                    "ptq",
                    "sin_reajuste",
                )
            )
            calidad.append(_calidad(otra, motor, "int8", 0.29))

        with pytest.raises(ValueError, match="misma etiqueta"):
            fr.compression_losses(filas, experimentos, calidad)

    @pytest.mark.parametrize(
        ("experimento", "esperada"),
        [
            ("s_pruning_l1global_r25_4e91ddbc", ("L1 r25", "pruning", False)),
            ("s_pruning_l1global_r50_randomctl_d581bd22", ("Aleatoria r50", "pruning", True)),
            ("s_pruning_l1global_r50_iter4_d8fe066f", ("Iterativa r50", "pruning", False)),
            ("s_pruning_bnscale_r50_f51fef4c", ("BN r50", "pruning", False)),
        ],
    )
    def test_las_podas_se_rotulan_por_su_variante(
        self, experimento: str, esperada: tuple[str, str, bool]
    ) -> None:
        base = "s_baseline_default_aaaaaaaa"
        experimentos = {
            experimento: {"technique": "pruning", "model_size": "s"},
            base: {"technique": "baseline", "model_size": "s"},
        }

        assert fr.compression_label(experimento, [experimento, base], experimentos) == esperada

    def test_las_combinaciones_cuelgan_de_su_poda(self) -> None:
        """PTQ sobre poda es «+ PTQ»; QAT sobre poda es «+ QAT», con su propio estilo."""
        poda = "s_pruning_l1global_r25_randomctl_a658e5d4"
        combinada = "s_combined_prune-l1global-r25-randomctl-ptqint8_e70dc4d6"
        qat = "s_qat_int8-noexcl_11111111"
        base = "s_baseline_default_aaaaaaaa"
        experimentos = {
            poda: {"technique": "pruning", "model_size": "s"},
            combinada: {"technique": "combined", "model_size": "s"},
            qat: {"technique": "qat", "model_size": "s"},
            base: {"technique": "baseline", "model_size": "s"},
        }

        assert fr.compression_label(combinada, [combinada, poda, base], experimentos) == (
            "Aleatoria r25 + PTQ",
            "combined",
            True,
        )
        assert fr.compression_label(qat, [qat, poda, base], experimentos) == (
            "Aleatoria r25 + QAT",
            "pruning_qat",
            True,
        )

    def test_el_tamano_va_al_final_salvo_en_small(self) -> None:
        base = "m_baseline_default_aaaaaaaa"
        ptq = "m_ptq_int8-noexcl_bbbbbbbb"
        experimentos = {
            base: {"technique": "baseline", "model_size": "m"},
            ptq: {"technique": "ptq", "model_size": "m"},
        }

        assert fr.compression_label(ptq, [ptq, base], experimentos) == ("PTQ medium", "ptq", False)

    def test_un_fragmento_desconocido_no_se_rotula(self) -> None:
        """Una poda nueva sin traducir saldría con una etiqueta que no la describe."""
        experimento = "s_pruning_l1global_r25_taylor_12345678"
        base = "s_baseline_default_aaaaaaaa"
        experimentos = {
            experimento: {"technique": "pruning", "model_size": "s"},
            base: {"technique": "baseline", "model_size": "s"},
        }

        with pytest.raises(ValueError, match="no se saben rotular"):
            fr.compression_label(experimento, [experimento, base], experimentos)

    def test_un_suelo_que_depende_de_la_sesion_no_se_elige(self) -> None:
        """El de mAP no depende de la sesión; si el volcado dijera otra cosa, se para."""
        suelos = {
            "floors": {
                "map_50_95": {"kind": "absoluto", "same_session": 0.004, "cross_session": 0.005}
            }
        }

        with pytest.raises(ValueError, match="no sabe cuál aplicar"):
            fr.quality_floor(suelos)

    def test_los_cambios_de_puesto_y_el_suelo(self) -> None:
        """Una inversión por debajo del suelo se cuenta, pero no como distinguible."""
        entradas = [
            _entrada("a", -0.010, -0.030),
            _entrada("b", -0.020, -0.028),
            _entrada("c", -0.050, -0.010),
        ]
        comparacion = fr.rank_comparison(entradas, 0.005)

        assert comparacion["order"]["onnxruntime"] == ["a", "b", "c"]
        assert comparacion["order"]["openvino"] == ["c", "b", "a"]
        assert comparacion["positions_changed"] == ["a", "c"]
        assert len(comparacion["order_inversions"]) == 3
        # a frente a b: 0,010 en ORT, pero solo 0,002 en OV.
        assert comparacion["inversions_above_floor_in_both"] == 2
        assert comparacion["adjacent_within_floor"]["openvino"] == [
            {"pair": ["b", "a"], "difference": 0.002}
        ]

    def test_ninguna_entrada_lee_filas_de_despliegue(self) -> None:
        import inspect

        for funcion in (fr.compression_losses, fr.figure_compression_loss):
            parametros = set(inspect.signature(funcion).parameters)
            assert not any("deploy" in p or "latency" in p for p in parametros), parametros


def _perdida_real():
    """Las compresiones y la figura del 6.2 sobre la campaña real."""
    registro = ExperimentRegistry(RAIZ / "results" / "registry")
    calidad, _ = analysis_quality_rows(registro)
    experimentos, _ = registro.valid_rows("experiments")
    volcado = _volcado_29()
    indice = {f["experiment_id"]: f for f in experimentos}
    entradas = fr.compression_losses(volcado["degradation"]["rows"], indice, calidad)
    figura, resumen = fr.figure_compression_loss(entradas, volcado["floors"])
    return calidad, indice, entradas, figura, resumen


class TestPerdidaPorCompresionSobreLaCampanaReal:
    """La figura del 6.2 contra el registro y el volcado de hoy."""

    def test_son_19_y_cambian_de_puesto_11(self) -> None:
        import matplotlib.pyplot as plt

        _, _, entradas, figura, resumen = _perdida_real()
        plt.close(figura)

        assert len(entradas) == 19
        assert len(resumen["positions_changed"]) == 11

    def test_la_composicion_sale_del_registro(self) -> None:
        import matplotlib.pyplot as plt

        calidad, indice, entradas, figura, _ = _perdida_real()
        plt.close(figura)
        motores: dict[str, set[str]] = {}
        for fila in calidad:
            if fila["runtime"] in fr.COMPRESSION_RUNTIMES:
                motores.setdefault(fila["experiment_id"], set()).add(fila["runtime"])
        esperadas = {
            e
            for e, m in motores.items()
            if m == set(fr.COMPRESSION_RUNTIMES)
            and indice[e]["technique"] not in ("baseline", "distillation")
        }

        assert {e["experiment_id"] for e in entradas} == esperadas
        for entrada in entradas:
            referencia = indice[entrada["reference"]]
            assert referencia["technique"] == "baseline"
            assert referencia["model_size"] == indice[entrada["experiment_id"]]["model_size"]
            con_int8 = {
                f["runtime"]
                for f in calidad
                if f["experiment_id"] == entrada["experiment_id"] and f["precision"] == "int8"
            }
            for motor, datos in entrada["runtimes"].items():
                assert datos["precision"] == ("int8" if motor in con_int8 else "fp32")
            experimento = indice[entrada["experiment_id"]]
            assert experimento["technique"] != "distillation"
            if experimento["technique"] == "pruning":
                assert (experimento.get("training_epochs") or "").strip(), entrada["label"]

    def test_el_orden_es_el_reconstruido_desde_el_registro(self) -> None:
        """La Δ se recalcula del registro, sin la figura ni el volcado, y ordena igual."""
        import matplotlib.pyplot as plt

        calidad, indice, entradas, figura, resumen = _perdida_real()
        plt.close(figura)
        bases = {f["model_size"]: e for e, f in indice.items() if f["technique"] == "baseline"}

        def map_registro(experimento: str, motor: str, precision: str) -> float:
            filas = [
                f
                for f in calidad
                if (f["experiment_id"], f["runtime"], f["precision"])
                == (experimento, motor, precision)
                and f["evaluator"] == "pycocotools"
                and f["split"] == "val"
            ]
            assert len(filas) == 1, (experimento, motor, precision)
            return float(filas[0]["map_50_95"])

        for motor in fr.COMPRESSION_RUNTIMES:
            perdidas = []
            for entrada in entradas:
                e = entrada["experiment_id"]
                precision = entrada["runtimes"][motor]["precision"]
                base = bases[indice[e]["model_size"]]
                perdidas.append(
                    (map_registro(base, motor, "fp32") - map_registro(e, motor, precision), e)
                )
            esperado = [e for _, e in sorted(perdidas)]
            dibujado = [
                f["experiment_id"] for f in sorted(resumen["rows"], key=lambda f: f[motor]["rank"])
            ]
            assert dibujado == esperado, motor

    def test_los_dos_paneles_dibujan_las_mismas_filas_en_el_mismo_orden(self) -> None:
        """Cada punto está en la fila de su etiqueta, en los dos paneles."""
        import matplotlib.pyplot as plt

        _, _, _, figura, resumen = _perdida_real()
        ejes = [e for e in figura.axes if e.collections]
        assert len(ejes) == 2
        etiquetas = [t.get_text() for t in ejes[0].get_yticklabels()]
        assert etiquetas == [f["label"] for f in resumen["rows"]]
        for eje, motor in zip(ejes, fr.COMPRESSION_RUNTIMES):
            puntos = sorted(
                (float(y), float(x))
                for coleccion in eje.collections
                for x, y in coleccion.get_offsets()
            )
            assert [y for y, _ in puntos] == list(range(len(resumen["rows"])))
            assert [x for _, x in puntos] == pytest.approx(
                [f[motor]["delta"] for f in resumen["rows"]]
            )
        plt.close(figura)

    def test_cabe_en_16_cm_sin_pie_y_ningun_texto_baja_de_6_5_pt(self) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.text import Text

        _, _, _, figura, _ = _perdida_real()
        assert figura.get_figwidth() == pytest.approx(16.0 / 2.54)
        figura.canvas.draw()
        contenido = figura.get_tightbbox(figura.canvas.get_renderer())
        assert contenido.x0 >= 0.0
        assert contenido.x1 <= figura.get_figwidth() + 1e-6
        assert not figura.texts, "El pie va en el texto de la memoria, no en la imagen."
        textos = [t for t in figura.findobj(Text) if t.get_visible() and t.get_text().strip()]
        assert textos
        assert min(t.get_fontsize() for t in textos) >= 6.5
        plt.close(figura)

    def test_el_manifiesto_versionado_trae_la_figura_de_hoy(self) -> None:
        """Si el registro o el volcado cambian sin regenerar la figura, falla aquí."""
        import matplotlib.pyplot as plt

        ruta = RAIZ / "results" / "raw" / "analysis" / "results_figures_manifest.json"
        if not ruta.is_file():
            pytest.skip("El paso 30 no se ha ejecutado en este árbol.")
        manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        _, _, _, figura, resumen = _perdida_real()
        plt.close(figura)
        versionado = manifiesto["figures"]["perdida_map_por_compresion"]

        for clave in (
            "rows",
            "positions_changed",
            "adjacent_within_floor",
            "order_inversions",
            "inversions_above_floor_in_both",
            "floor",
        ):
            assert versionado[clave] == json.loads(json.dumps(resumen[clave])), clave
        assert versionado["files"]["pdf"] == (
            "results/figures/fig_6_2_perdida_map_por_compresion.pdf"
        )


def _experimentos_rotulador() -> dict[str, dict[str, str]]:
    """Un registro de experimentos de juguete con una configuración de cada clase."""
    filas = [
        ("n_baseline_default_aaaaaaaa", "", "baseline", "n"),
        ("s_baseline_default_bbbbbbbb", "", "baseline", "s"),
        ("n_ptq_int8-noexcl_cccccccc", "n_baseline_default_aaaaaaaa", "ptq", "n"),
        ("s_qat_int8-noexcl_dddddddd", "s_baseline_default_bbbbbbbb", "qat", "s"),
        ("s_pruning_l1global_r50_eeeeeeee", "s_baseline_default_bbbbbbbb", "pruning", "s"),
        (
            "s_pruning_l1global_r25_randomctl_ffffffff",
            "s_baseline_default_bbbbbbbb",
            "pruning",
            "s",
        ),
        (
            "s_combined_prune-l1global-r50-ptqint8_11111111",
            "s_pruning_l1global_r50_eeeeeeee",
            "combined",
            "s",
        ),
        ("s_qat_int8-noexcl_22222222", "s_pruning_l1global_r50_eeeeeeee", "qat", "s"),
    ]
    return {
        e: {"experiment_id": e, "parent_experiment_id": p, "technique": t, "model_size": s}
        for e, p, t, s in filas
    }


def _cadena(experimento: str, experimentos: dict[str, dict[str, str]]) -> list[str]:
    """La ascendencia de un experimento, de él a la raíz."""
    cadena = [experimento]
    while experimentos[cadena[-1]]["parent_experiment_id"]:
        cadena.append(experimentos[cadena[-1]]["parent_experiment_id"])
    return cadena


class TestRotuladorDeConfiguraciones:
    """Rotulador del 6.4: el del 6.2 extendido a los baselines y con el tamaño delante."""

    ESPERADAS: dict[str, tuple[str, str, bool]] = {
        "n_baseline_default_aaaaaaaa": ("n FP32", "baseline", False),
        "s_baseline_default_bbbbbbbb": ("s FP32", "baseline", False),
        "n_ptq_int8-noexcl_cccccccc": ("n PTQ", "ptq", False),
        "s_qat_int8-noexcl_dddddddd": ("s QAT", "qat", False),
        "s_pruning_l1global_r50_eeeeeeee": ("L1 r50", "pruning", False),
        "s_pruning_l1global_r25_randomctl_ffffffff": ("Aleatoria r25", "pruning", True),
        "s_combined_prune-l1global-r50-ptqint8_11111111": ("L1 r50 + PTQ", "combined", False),
        "s_qat_int8-noexcl_22222222": ("L1 r50 + QAT", "pruning_qat", False),
    }

    def test_cada_clase_sale_con_su_etiqueta(self) -> None:
        """Baselines y cuantizaciones con el tamaño delante; podas de s sin él."""
        experimentos = _experimentos_rotulador()
        for experimento, esperada in self.ESPERADAS.items():
            assert (
                fr.configuration_label(experimento, _cadena(experimento, experimentos), experimentos)
                == esperada
            )

    def test_la_etiqueta_del_6_2_no_cambia(self) -> None:
        """La misma PTQ de nano sigue siendo «PTQ nano» en la figura del 6.2."""
        experimentos = _experimentos_rotulador()
        experimento = "n_ptq_int8-noexcl_cccccccc"

        assert fr.compression_label(
            experimento, _cadena(experimento, experimentos), experimentos
        ) == ("PTQ nano", "ptq", False)

    def test_un_baseline_con_padre_no_se_rotula(self) -> None:
        """Un «baseline» que cuelga de otro no se sabe qué referencia es."""
        experimentos = _experimentos_rotulador()
        experimentos["s_baseline_default_bbbbbbbb"]["parent_experiment_id"] = (
            "n_baseline_default_aaaaaaaa"
        )

        with pytest.raises(ValueError, match="baseline"):
            fr.configuration_label(
                "s_baseline_default_bbbbbbbb",
                _cadena("s_baseline_default_bbbbbbbb", experimentos),
                experimentos,
            )

    def test_una_poda_de_otro_tamano_lleva_el_suyo_delante(self) -> None:
        experimentos = _experimentos_rotulador()
        experimentos["s_pruning_l1global_r50_eeeeeeee"]["model_size"] = "m"

        etiqueta, _, _ = fr.configuration_label(
            "s_pruning_l1global_r50_eeeeeeee",
            _cadena("s_pruning_l1global_r50_eeeeeeee", experimentos),
            experimentos,
        )
        assert etiqueta == "m L1 r50"


def _contraste_real() -> tuple[dict, dict[str, dict[str, str]], list[dict[str, str]]]:
    """Volcado de validación frente a test, experimentos y filas de calidad de las dos particiones.
    """
    registro = ExperimentRegistry(RAIZ / "results" / "registry")
    experimentos, _ = registro.valid_rows("experiments")
    calidad, _ = registro.valid_rows("quality_metrics")
    evidencia = json.loads(
        (RAIZ / "docs" / "evidencia" / "d193_evaluacion_test.json").read_text(encoding="utf-8")
    )
    return evidencia, {f["experiment_id"]: f for f in experimentos}, calidad


class TestValidacionFrenteAPrueba:
    """Figura del 6.7: validación frente a test, contrastada con el registro."""

    def test_va_en_el_6_7_y_se_nombra_asi(self) -> None:
        assert fr.PLACEMENT["validacion_frente_a_prueba"] == (6, 7)
        assert style.figure_filename(6, 7, "validacion_frente_a_prueba") == (
            "fig_6_7_validacion_frente_a_prueba"
        )

    def test_olvidar_el_contraste_no_es_una_forma_de_no_dibujarlo(self) -> None:
        import inspect

        parametro = inspect.signature(fr.save_results_figures).parameters["split_check"]
        assert parametro.kind is inspect.Parameter.KEYWORD_ONLY
        assert parametro.default is inspect.Parameter.empty

    def test_sobre_la_campana_real_son_23_por_motor_en_cuatro_familias(self) -> None:
        from collections import Counter

        evidencia, indice, calidad = _contraste_real()
        contraste = fr.split_check_entries(evidencia, indice, calidad)

        for motor in fr.SPLIT_CHECK_RUNTIMES:
            familias = Counter(e["family"] for e in contraste["entries"] if e["runtime"] == motor)
            assert familias == {
                "reference": 4,
                "quantization": 5,
                "pruning": 6,
                "pruning_quantization": 8,
            }
        por_etiqueta = {
            e["label"]: e for e in contraste["entries"] if e["runtime"] == "onnxruntime"
        }
        assert por_etiqueta["L1 r25 + QAT"]["family"] == "pruning_quantization"
        assert por_etiqueta["s QAT"]["family"] == "quantization"

    def test_sobre_la_campana_real_ningun_rotulo_se_lee_mal(self) -> None:
        """Ningún rótulo pisa, sale del panel o queda más cerca de otro punto."""
        import matplotlib.pyplot as plt

        evidencia, indice, calidad = _contraste_real()
        contraste = fr.split_check_entries(evidencia, indice, calidad)
        figura, resumen = fr.figure_validation_vs_test(contraste, _volcado_29()["floors"])
        limites = {(eje.get_xlim(), eje.get_ylim()) for eje in figura.axes}
        plt.close(figura)

        assert resumen["labels_with_overlap"] == []
        assert len(limites) == 1
        ((x, y),) = limites
        assert x == y
        assert {(r["runtime"], r["label"]) for r in resumen["labels"]} == {
            (m, etiqueta)
            for m, etiquetas in fr.SPLIT_CHECK_LABELS.items()
            for etiqueta in etiquetas
        }

    def test_una_cifra_que_no_es_la_del_registro_aborta(self) -> None:
        """Si el volcado y el registro han cambiado por separado, la figura no sale."""
        evidencia, indice, calidad = _contraste_real()
        fila = next(f for f in evidencia["tabla"] if f["runtime"] == "openvino")
        fila["test"]["map_50_95"] += 0.001

        with pytest.raises(ValueError, match="registro"):
            fr.split_check_entries(evidencia, indice, calidad)

    def test_sin_la_fila_de_test_en_el_registro_aborta(self) -> None:
        """Cada punto tiene que tener exactamente una fila por partición."""
        evidencia, indice, calidad = _contraste_real()
        fila = next(f for f in evidencia["tabla"] if f["runtime"] == "onnxruntime")
        sin_test = [
            f
            for f in calidad
            if not (
                f["experiment_id"] == fila["experiment_id"]
                and f["runtime"] == "onnxruntime"
                and f["split"] == "test"
            )
        ]

        with pytest.raises(ValueError, match="filas en el registro"):
            fr.split_check_entries(evidencia, indice, sin_test)

    def test_un_suelo_distinto_del_paso_29_aborta(self) -> None:
        """La banda tiene que ser la misma que usa el resto del capítulo."""
        evidencia, indice, calidad = _contraste_real()
        contraste = fr.split_check_entries(evidencia, indice, calidad)
        contraste["floor"] = contraste["floor"] * 2

        with pytest.raises(ValueError, match="suelo"):
            fr.figure_validation_vs_test(contraste, _volcado_29()["floors"])

    def test_sin_volcado_no_sale(self) -> None:
        with pytest.raises(ValueError, match="volcado de evaluación en test"):
            fr.figure_validation_vs_test(None, _volcado_29()["floors"])

    def test_un_rotulo_pedido_que_no_esta_aborta(self) -> None:
        """Un rótulo que cita el texto no puede desaparecer en silencio."""
        evidencia, indice, calidad = _contraste_real()
        contraste = fr.split_check_entries(evidencia, indice, calidad)
        contraste["entries"] = [
            e
            for e in contraste["entries"]
            if not (e["runtime"] == "openvino" and e["label"] == "L1 r50")
        ]

        with pytest.raises(ValueError, match="rotular"):
            fr.figure_validation_vs_test(contraste, _volcado_29()["floors"])
