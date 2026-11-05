"""El control sin destilar, la comprobación previa y la regla de lectura."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch", reason="requiere PyTorch")

from tfm_edge.common.exceptions import TFMEdgeError  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.common.registry import ExperimentRegistry  # noqa: E402
from tfm_edge.compression.distillation.losses import DistillationLoss, HeadGeometry  # noqa: E402
from tfm_edge.compression.distillation.preflight import (  # noqa: E402
    FAILURE_MEMORY,
    FAILURE_PATH,
    PreflightResult,
    classify_failure,
)
from tfm_edge.compression.distillation.reading import (  # noqa: E402
    EVALUATOR_READABILITY_FLOOR,
    STARTING_BASELINE_ID,
    read_pilot,
)
from tfm_edge.compression.distillation.reuse import load_finished_run  # noqa: E402
from tfm_edge.compression.distillation.trainer import (  # noqa: E402
    DistillationError,
    DistillationResult,
    DistillationTrainer,
)
from tfm_edge.evaluation.coco_eval import (  # noqa: E402
    CRITERIA,
    CocoEvaluationResult,
    CriterionMetrics,
)
from tfm_edge.evaluation.quality_notes import (  # noqa: E402
    CANONICAL_EVALUATOR,
    DISTILLATION_CONTROL_MARKER,
    build_distillation_note,
)
from tfm_edge.training.trainer import TrainingCost, TrainingResult  # noqa: E402

#: Geometría de cabeza de SH17 a 640 px, la del estudio.
GEOMETRIA = HeadGeometry(num_classes=17, reg_max=16, strides=(8, 16, 32))


def _cargar_script() -> ModuleType:
    """Carga ``scripts/20_distill.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "20_distill.py"
    especificacion = importlib.util.spec_from_file_location("distill_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _perdida(alpha: float) -> DistillationLoss:
    """Construye la pérdida del estudio con el alpha pedido."""
    return DistillationLoss(
        geometry=GEOMETRIA,
        resolution=640,
        temperature=2.0,
        alpha=alpha,
        class_weight=0.1431,
        box_weight=0.0984,
        feature_weight=0.0704,
        kd_type="combined",
    )


def _resultado(
    iniciales: dict[str, float],
    finales: dict[str, float],
    normas_grad: list[float],
) -> DistillationResult:
    """Compone un resultado de destilación con las normas que se quieran probar."""
    return DistillationResult(
        training=TrainingResult(
            best_checkpoint=Path("best.pt"), last_checkpoint=Path("last.pt"), cost=TrainingCost()
        ),
        adapter_initial_norms=iniciales,
        adapter_final_norms=finales,
        adapter_grad_norms=normas_grad,
    )


class TestQueAlphaCeroEsUnAfinadoNormalExacto:
    """El control tiene que ser numéricamente indistinguible de no destilar."""

    def test_el_total_es_la_deteccion_bit_a_bit(self) -> None:
        """Multiplicar por 1,0 en IEEE754 no cambia ni un bit; por 0,0 tampoco suma."""
        deteccion = torch.tensor(5.0312345678)
        componentes = _perdida(0.0).combine(
            detection=deteccion,
            response_cls=torch.tensor(3.51),
            response_box=torch.tensor(5.10),
            feature_term=torch.tensor(7.13),
            marked_anchors=250,
            batch_size=16,
        )
        assert componentes.total.item() == deteccion.item()

    def test_con_alpha_positivo_el_total_ya_no_es_la_deteccion(self) -> None:
        deteccion = torch.tensor(5.0312345678)
        componentes = _perdida(0.5).combine(
            detection=deteccion,
            response_cls=torch.tensor(3.51),
            response_box=torch.tensor(5.10),
            feature_term=torch.tensor(7.13),
            marked_anchors=250,
            batch_size=16,
        )
        assert componentes.total.item() != deteccion.item()

    def test_los_componentes_se_registran_aunque_no_intervengan(self) -> None:
        componentes = _perdida(0.0).combine(
            detection=torch.tensor(5.03),
            response_cls=torch.tensor(3.51),
            response_box=torch.tensor(5.10),
            feature_term=torch.tensor(7.13),
            marked_anchors=250,
            batch_size=16,
        )
        assert componentes.response_cls.item() > 0.0
        assert componentes.response_box.item() > 0.0
        assert componentes.features.item() > 0.0


class TestLaGuardaCambiaDeAfirmacionSegunLaRama:
    """Con alpha=0 se exige lo contrario que con alpha>0, y se exige de verdad."""

    @staticmethod
    def _comprobar(alpha: float, resultado: DistillationResult) -> None:
        """Ejecuta la guarda sin construir un entrenador completo."""
        falso = DistillationTrainer.__new__(DistillationTrainer)
        falso.loss = _perdida(alpha)
        DistillationTrainer._comprobar_adaptadores(falso, resultado)

    def test_control_alpha_cero_con_adaptadores_quietos_pasa(self) -> None:
        self._comprobar(0.0, _resultado({"model.15": 9.16}, {"model.15": 9.16}, [0.0, 0.0, 0.0]))

    def test_control_alpha_cero_con_adaptadores_movidos_falla(self) -> None:
        with pytest.raises(DistillationError, match="no pueden moverse"):
            self._comprobar(
                0.0, _resultado({"model.15": 9.16}, {"model.15": 9.30}, [0.0, 0.0, 0.0])
            )

    def test_control_alpha_cero_con_gradiente_no_nulo_falla(self) -> None:
        with pytest.raises(DistillationError, match="ha llegado gradiente"):
            self._comprobar(
                0.0, _resultado({"model.15": 9.16}, {"model.15": 9.16}, [0.0, 1e-7, 0.0])
            )

    def test_control_alpha_cero_con_gradiente_nan_falla(self) -> None:
        """Un nan sale de multiplicar por cero un término desbordado, y no es cero."""
        with pytest.raises(DistillationError, match="ha llegado gradiente"):
            self._comprobar(
                0.0, _resultado({"model.15": 9.16}, {"model.15": 9.16}, [0.0, float("nan")])
            )

    def test_piloto_con_adaptadores_quietos_falla(self) -> None:
        with pytest.raises(DistillationError, match="no han cambiado"):
            self._comprobar(0.5, _resultado({"model.15": 9.16}, {"model.15": 9.16}, [1e-5, 1e-5]))

    def test_piloto_con_adaptadores_movidos_pasa(self) -> None:
        """Control negativo de la anterior."""
        self._comprobar(0.5, _resultado({"model.15": 9.16}, {"model.15": 9.30}, [1e-5, 1e-5]))


class TestLaMarcaDelControlNoEstaDuplicada:
    """El buscador del registro y la nota tienen que compartir el mismo texto."""

    def test_la_nota_del_control_lleva_la_marca(self) -> None:
        """Sin esto, el buscador no encontraría el control que él mismo inscribió."""
        assert DISTILLATION_CONTROL_MARKER in build_distillation_note(0.0)

    def test_la_nota_de_una_variante_de_verdad_no_la_lleva(self) -> None:
        """O toda fila de destilación pasaría por control."""
        assert DISTILLATION_CONTROL_MARKER not in build_distillation_note(0.5)

    def test_el_script_usa_la_constante_y_no_una_copia(self) -> None:
        """Dos copias del texto que identifica una fila acaban divergiendo."""
        modulo = _cargar_script()
        assert modulo.DISTILLATION_CONTROL_MARKER is DISTILLATION_CONTROL_MARKER
        fuente = (get_project_paths().root / "scripts" / "20_distill.py").read_text(
            encoding="utf-8"
        )
        assert f'"{DISTILLATION_CONTROL_MARKER}"' not in fuente

    def test_las_dos_notas_empiezan_por_el_prefijo_del_evaluador(self) -> None:
        """La regla de lectura de ESQUEMA_RESULTADOS vale también para estas filas."""
        for alpha in (0.0, 0.5):
            assert build_distillation_note(alpha).startswith("evaluator=pycocotools.")


class TestQueUnaComprobacionSinGpuNoDaPermisoParaLanzar:
    """Una comprobación previa sin GPU no autoriza el lanzamiento."""

    def test_sin_memoria_medida_no_cabe(self) -> None:
        previa = PreflightResult(completed=True, memory_measured=False)
        assert previa.completed
        assert not previa.fits

    def test_con_memoria_medida_si_cabe(self) -> None:
        """Control negativo."""
        assert PreflightResult(completed=True, memory_measured=True).fits

    def test_un_bucle_que_falla_no_cabe_aunque_haya_gpu(self) -> None:
        assert not PreflightResult(completed=False, memory_measured=True).fits

    def test_el_resumen_lo_dice_con_todas_las_letras(self) -> None:
        """Quien lea la consola tiene que ver que no se ha comprobado nada."""
        texto = PreflightResult(completed=True, memory_measured=False).summary()
        assert "NO autoriza a lanzar" in texto
        assert "CABE" not in texto.replace("NO CABE", "")

    def test_las_condiciones_del_estudio_son_640_y_lote_16(self) -> None:
        base = {"completed": True, "memory_measured": True}
        assert PreflightResult(**base, resolution=640, batch_size=16).conditions_match_campaign
        assert not PreflightResult(**base, resolution=320, batch_size=16).conditions_match_campaign
        assert not PreflightResult(**base, resolution=640, batch_size=8).conditions_match_campaign
        assert not PreflightResult(
            completed=True, memory_measured=False, resolution=640, batch_size=16
        ).conditions_match_campaign

    def test_los_veredictos_derivados_entran_en_el_volcado(self) -> None:
        """``asdict`` no recoge las propiedades, y son las que dicen qué concluir."""
        volcado = PreflightResult(completed=True, memory_measured=True, resolution=640).to_dict()
        assert volcado["fits"] is True
        assert "conditions_match_campaign" in volcado

    def test_los_picos_no_medidos_van_vacios_y_no_a_cero(self) -> None:
        """Un 0.0 se leería como 'medido y no consume nada' (ESQUEMA_RESULTADOS, 2)."""
        previa = PreflightResult(completed=True, memory_measured=False)
        assert previa.peak_reserved_mb == ""
        assert previa.gpu_total_mb == ""


class TestQueUnFalloDelCaminoNoSeaFaltaDeMemoria:
    """El fallo que se hace pasar por otro, que es el que no se corrige."""

    #: Error real del log de Colab: un ValueError de matplotlib (MPLBACKEND heredado del kernel)
    #: envuelto por Ultralytics en un RuntimeError de "dataset error", sin relación con la memoria.
    ERROR_DE_COLAB = (
        "Dataset '/content/.../sh17_ultralytics.yaml' error ❌ Key backend: "
        "'module://matplotlib_inline.backend_inline' is not a valid value for "
        "backend; supported values are ['gtk3agg', 'gtk3cairo', 'gtk4agg'"
    )

    def test_el_error_que_lo_destapo_se_clasifica_como_camino(self) -> None:
        clase, motivo = classify_failure(RuntimeError(self.ERROR_DE_COLAB))
        assert clase == FAILURE_PATH
        assert "RuntimeError" in motivo

    def test_quedarse_sin_memoria_sigue_siendo_falta_de_memoria(self) -> None:
        clase, motivo = classify_failure(
            RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        )
        assert clase == FAILURE_MEMORY
        assert motivo.startswith("Sin memoria:")

    def test_el_oom_tipado_no_depende_de_como_este_redactado(self) -> None:
        """El tipo manda sobre el texto, y por eso se mira antes."""
        clase, _ = classify_failure(
            torch.cuda.OutOfMemoryError("Allocation on device 0 would exceed allowed memory")
        )
        assert clase == FAILURE_MEMORY

    def test_una_guarda_del_propio_bucle_no_es_falta_de_memoria(self) -> None:
        """Que salte la guarda de los adaptadores es un resultado, y no de memoria."""
        clase, _ = classify_failure(TFMEdgeError("los adaptadores no se han movido"))
        assert clase == FAILURE_PATH

    def test_el_resumen_de_un_fallo_del_camino_no_dice_no_cabe(self) -> None:
        """Lo que lee quien mira la consola, que es de donde salió el diagnóstico malo."""
        previa = PreflightResult(
            completed=False, failure_kind=FAILURE_PATH, reason="RuntimeError: lo que sea"
        )
        resumen = previa.summary()
        assert "NO CABE" not in resumen
        assert "SIN MEDIR" in resumen
        assert "no llegó a tomarse" in resumen

    def test_el_resumen_de_un_fallo_de_memoria_si_dice_no_cabe(self) -> None:
        previa = PreflightResult(
            completed=False, failure_kind=FAILURE_MEMORY, reason="Sin memoria: CUDA out of memory"
        )
        assert "NO CABE" in previa.summary()

    def test_la_traza_entera_entra_en_el_volcado(self) -> None:
        previa = PreflightResult(
            completed=False,
            failure_kind=FAILURE_PATH,
            reason="RuntimeError: recortado",
            failure_traceback="Traceback (most recent call last):\n  ...\nValueError: Key backend",
        )
        volcado = previa.to_dict()
        assert volcado["failure_kind"] == FAILURE_PATH
        assert "ValueError" in volcado["failure_traceback"]
        assert volcado["fits"] is False


class TestQueNoSePiseUnEntrenamientoYaHecho:
    """No se entrena encima de un entrenamiento existente."""

    def test_un_directorio_vacio_no_dispara_la_guarda(self, tmp_path: Path) -> None:
        assert _cargar_script().entrenamiento_previo(tmp_path / "distill") is None

    def test_unos_pesos_sin_results_csv_se_detectan_igual(self, tmp_path: Path) -> None:
        """Se para igual, y se dice que no se sabe cuántas épocas son."""
        pesos = tmp_path / "distill" / "weights"
        pesos.mkdir(parents=True)
        (pesos / "best.pt").write_bytes(b"no importa el contenido")
        previo = _cargar_script().entrenamiento_previo(tmp_path / "distill")
        assert previo is not None
        assert previo[1] == 0

    def test_las_epocas_salen_de_contar_filas(self, tmp_path: Path) -> None:
        """El número que se le enseña al usuario tiene que ser el de su fichero."""
        run = tmp_path / "distill"
        (run / "weights").mkdir(parents=True)
        (run / "weights" / "best.pt").write_bytes(b"x")
        (run / "results.csv").write_text(
            "epoch,loss\n" + "".join(f"{i},0.1\n" for i in range(1, 11)), encoding="utf-8"
        )
        previo = _cargar_script().entrenamiento_previo(run)
        assert previo is not None and previo[1] == 10


class TestQueElRegistroAcepteLaFilaDeCalidad:
    """La fila que compone el script, consumida por el registro de verdad."""

    @staticmethod
    def _informe() -> dict:
        """Compone un informe de evaluación como el que devuelve el runner."""
        resultado = CocoEvaluationResult(
            split="val",
            input_resolution=640,
            num_images=1213,
            num_annotations=8000,
            num_detections=168286,
            by_criterion={
                criterio: CriterionMetrics(criterion=criterio, map_50=0.52, map_50_95=0.32)
                for criterio in CRITERIA
            },
            per_class={indice: {"ap_50_95": 0.3} for indice in range(17)},
        )
        return {
            "result": resultado,
            "registry_columns": {
                **resultado.to_registry_columns(),
                "input_resolution": 640,
                "conf_threshold": 0.001,
                "iou_threshold": 0.7,
                "split": "val",
                "eval_time_s": 33.0,
            },
            "num_matched_detections": 168286,
        }

    def test_el_registro_acepta_la_fila(self, tmp_path: Path) -> None:
        """Sin esto, el fallo aparece al final de la campaña y no antes."""
        script = _cargar_script()
        clases = [f"c{i}" for i in range(17)]
        fila = script.construir_fila_calidad(
            experiment_id="n_distillation_l2n_control_alpha0_pilot10_8826269c",
            informe_eval=self._informe(),
            clases=clases,
            device_id="colab_t4_gpu",
            environment_id="abc123",
            ruta_detalle=tmp_path / "detalle.json",
            alpha=0.0,
            seed=42,
        )
        registro = ExperimentRegistry(registry_dir=tmp_path / "registro", class_names=clases)
        registro.append_quality_metrics(fila)

        escritas = list(csv.DictReader((tmp_path / "registro" / "quality_metrics.csv").open()))
        assert len(escritas) == 1
        # Ausencia de medida es celda vacía, no un cero (sección 2 de
        # ESQUEMA_RESULTADOS): el desglose de errores no se calcula aquí.
        assert escritas[0]["error_missed"] == ""


class TestQueSePuedaCompletarSinReentrenar:
    """``--reuse-run`` lee de disco lo ya medido y deja a ``None`` lo que falta."""

    @staticmethod
    def _montar(directorio: Path, normas_grad: list[float]) -> Path:
        """Monta un directorio de ejecución como el que deja una destilación."""
        run_dir = directorio / "distill"
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"no son pesos de verdad")
        (run_dir / "training_cost.json").write_text(
            json.dumps({"wall_time_s": 12276.0, "epochs_completed": 10, "gpu_name": "Tesla T4"}),
            encoding="utf-8",
        )
        columnas = ["epoch", "steps", "total", "adapter_norm_model.15", "adapter_grad_norm"]
        with (run_dir / "distillation_components.csv").open("w", newline="") as fichero:
            escritor = csv.DictWriter(fichero, fieldnames=columnas)
            escritor.writeheader()
            for epoca, norma in enumerate(normas_grad):
                escritor.writerow(
                    {
                        "epoch": epoca,
                        "steps": 407,
                        "total": 1.5,
                        "adapter_norm_model.15": 9.163902,
                        "adapter_grad_norm": norma,
                    }
                )
        return run_dir

    def test_sin_checkpoint_no_hay_nada_que_reutilizar(self, tmp_path: Path) -> None:
        """Y se dice, en vez de seguir con un camino a medias."""
        with pytest.raises(DistillationError, match="No hay nada que reutilizar"):
            load_finished_run(tmp_path / "distill")

    def test_lee_el_coste_y_las_epocas_de_disco(self, tmp_path: Path) -> None:
        run_dir = self._montar(tmp_path, [0.0] * 10)
        reutilizado = load_finished_run(run_dir)
        assert reutilizado.closed_epochs == 10
        assert reutilizado.steps_in_closed_epochs == 4070
        assert reutilizado.cost is not None
        assert reutilizado.cost["gpu_name"] == "Tesla T4"

    def test_el_gradiente_a_cero_se_sostiene_con_medias_por_epoca(self, tmp_path: Path) -> None:
        """La afirmación del control sobrevive a perder la resolución por paso."""
        assert load_finished_run(self._montar(tmp_path, [0.0] * 10)).adapter_grad_always_zero

    def test_un_gradiente_distinto_de_cero_se_ve(self, tmp_path: Path) -> None:
        reutilizado = load_finished_run(self._montar(tmp_path, [0.0] * 9 + [1e-8]))
        assert reutilizado.adapter_grad_always_zero is False

    def test_un_nan_no_pasa_por_cero(self, tmp_path: Path) -> None:
        """Un ``nan`` es lo que sale de un término desbordado multiplicado por cero."""
        reutilizado = load_finished_run(self._montar(tmp_path, [0.0] * 9 + [float("nan")]))
        assert reutilizado.adapter_grad_always_zero is False

    def test_sin_desglose_no_se_afirma_nada(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "distill"
        (run_dir / "weights").mkdir(parents=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"pesos")
        reutilizado = load_finished_run(run_dir)
        assert reutilizado.adapter_grad_always_zero is None
        assert reutilizado.adapter_norms_moved is None
        assert any("desglose por época" in falta for falta in reutilizado.missing)

    def test_lo_que_no_esta_en_disco_va_a_null_y_no_a_cero(self, tmp_path: Path) -> None:
        """El volcado no puede publicar un valor por defecto como si fuera medido."""
        script = _cargar_script()
        reutilizado = load_finished_run(self._montar(tmp_path, [0.0] * 10))
        campos = script.campos_del_entrenamiento(None, reutilizado)

        for clave in (
            "steps",
            "adapter_params",
            "adapter_initial_norms",
            "adapter_final_norms",
            "adapters_learned",
        ):
            assert campos[clave] is None, f"'{clave}' saldría como {campos[clave]!r}"

        assert campos["training_source"] == "reutilizado de disco (--reuse-run)"
        assert campos["steps_in_closed_epochs"] == 4070
        assert campos["adapter_grad_always_zero"] is True
        assert campos["reused_training"]["not_recoverable"]

    def test_no_se_admiten_los_dos_caminos_a_la_vez(self, tmp_path: Path) -> None:
        """Entrenar y reutilizar son excluyentes; pasar los dos es un error de uso."""
        script = _cargar_script()
        with pytest.raises(ValueError):
            script.campos_del_entrenamiento(None, None)


class TestLaReglaDeLectura:
    """Los cuatro veredictos, fijados antes de ver ningún número."""

    def test_destilado_por_encima_del_control_es_senal_positiva(self) -> None:
        """El único caso que sostiene extender la campaña."""
        lectura = read_pilot(0.3500, 0.3400, 0.3302)
        assert lectura.verdict == "destilar ayuda"
        assert lectura.readable

    def test_destilado_por_debajo_del_control_no_habla_de_destilar_en_general(self) -> None:
        """alpha no es un eje independiente de la tasa, y el texto lo dice."""
        lectura = read_pilot(0.3300, 0.3400, 0.3302)
        assert lectura.verdict == "destilar perjudica a esta configuración"
        assert "en general" in lectura.statement

    def test_empate_con_los_dos_por_encima_atribuye_a_las_epocas(self) -> None:
        lectura = read_pilot(0.3400, 0.3401, 0.3302)
        assert lectura.verdict == "las épocas extra ayudaron, destilar no se ve"
        assert "no se puede concluir" in lectura.statement.lower()

    def test_empate_general_solo_permite_decir_que_no_se_ve(self) -> None:
        lectura = read_pilot(0.3303, 0.3302, 0.3302)
        assert lectura.verdict == "10 épocas no bastan para verlo"
        assert "10 épocas no bastan para verlo" in lectura.statement
        assert "no funciona" in lectura.statement

    def test_una_diferencia_justo_por_debajo_del_suelo_no_es_un_resultado(self) -> None:
        """El límite exacto, por los dos lados."""
        casi = EVALUATOR_READABILITY_FLOOR - 1e-9
        assert read_pilot(0.3302 + casi, 0.3302, 0.3302).verdict != "destilar ayuda"
        justo = EVALUATOR_READABILITY_FLOOR
        assert read_pilot(0.3302 + justo, 0.3302, 0.3302).verdict == "destilar ayuda"

    def test_todo_veredicto_arrastra_la_advertencia_del_piloto(self) -> None:
        for cifras in ((0.35, 0.34, 0.33), (0.33, 0.34, 0.33), (0.34, 0.34, 0.33)):
            assert "una semilla por variante" in read_pilot(*cifras).caveat

    def test_todo_veredicto_arrastra_el_confusor_de_alpha(self) -> None:
        casos = (
            (0.35, 0.34, 0.33),
            (0.33, 0.34, 0.33),
            (0.34, 0.34, 0.33),
            (0.3303, 0.3302, 0.3302),
        )
        for cifras in casos:
            lectura = read_pilot(*cifras)
            assert "no son ejes independientes" in lectura.confound
            assert lectura.confound in lectura.summary()

    def test_un_suelo_de_cero_es_un_error(self) -> None:
        """Sin umbral, el ruido del evaluador se leería como efecto de la técnica."""
        with pytest.raises(ValueError, match="debe ser positivo"):
            read_pilot(0.35, 0.34, 0.33, floor=0.0)


class TestQueElSubconjuntoDeLaComprobacionEsElPeorCaso:
    """Sin esto, la comprobación de memoria mide un lote cualquiera."""

    @staticmethod
    def _dataset(raiz: Path, anotaciones: dict[str, int]) -> tuple[Path, Path]:
        """Materializa un dataset de mentira con las anotaciones pedidas."""
        import yaml as _yaml

        datos = raiz / "ds"
        (datos / "images").mkdir(parents=True)
        (datos / "labels").mkdir(parents=True)
        splits = raiz / "splits"
        splits.mkdir()

        for nombre, cuantas in anotaciones.items():
            (datos / "images" / f"{nombre}.jpg").write_bytes(b"")
            (datos / "labels" / f"{nombre}.txt").write_text(
                "\n".join("0 0.5 0.5 0.1 0.1" for _ in range(cuantas)), encoding="utf-8"
            )
        entradas = [f"./images/{nombre}.jpg" for nombre in anotaciones]
        (datos / "ultralytics_train.txt").write_text("\n".join(entradas) + "\n", encoding="utf-8")
        (splits / "train.txt").write_text("\n".join(entradas) + "\n", encoding="utf-8")

        yaml_ruta = datos / "data.yaml"
        yaml_ruta.write_text(
            _yaml.safe_dump({"path": ".", "train": "ultralytics_train.txt", "names": {0: "x"}}),
            encoding="utf-8",
        )
        return yaml_ruta, splits

    def test_most_labels_se_queda_con_las_mas_pobladas(self, tmp_path: Path) -> None:
        from tfm_edge.data.smoke_subset import prepare_smoke_subset

        yaml_ruta, splits = self._dataset(
            tmp_path, {"pocas": 1, "muchas": 90, "ninguna": 0, "medias": 50}
        )
        reducido, _ = prepare_smoke_subset(
            yaml_ruta, splits, "train", 2, tmp_path / "sub", order="most_labels"
        )
        elegidas = (reducido.parent / "ultralytics_train.txt").read_text(encoding="utf-8")
        assert "muchas" in elegidas and "medias" in elegidas
        assert "pocas" not in elegidas and "ninguna" not in elegidas

    def test_first_conserva_el_orden_del_fichero(self, tmp_path: Path) -> None:
        from tfm_edge.data.smoke_subset import prepare_smoke_subset

        yaml_ruta, splits = self._dataset(
            tmp_path, {"pocas": 1, "muchas": 90, "ninguna": 0, "medias": 50}
        )
        reducido, _ = prepare_smoke_subset(
            yaml_ruta, splits, "train", 2, tmp_path / "sub", order="first"
        )
        elegidas = (reducido.parent / "ultralytics_train.txt").read_text(encoding="utf-8")
        assert "pocas" in elegidas and "muchas" in elegidas
        assert "medias" not in elegidas

    def test_un_orden_desconocido_es_un_error(self, tmp_path: Path) -> None:
        """Una errata no puede caer en silencio al comportamiento por defecto."""
        from tfm_edge.common.exceptions import TFMEdgeError
        from tfm_edge.data.smoke_subset import prepare_smoke_subset

        yaml_ruta, splits = self._dataset(tmp_path, {"a": 1})
        with pytest.raises(TFMEdgeError, match="Orden de selección desconocido"):
            prepare_smoke_subset(
                yaml_ruta, splits, "train", 1, tmp_path / "sub", order="las-mas-bonitas"
            )


class TestQueElSueloDeLecturaSigueCuadrandoConElRegistro:
    """La constante sale de dos filas reales; si se mueven, hay que enterarse."""

    def _filas_del_baseline(self) -> list[dict[str, str]]:
        """Filas de calidad del punto de partida, leídas del CSV del repositorio."""
        ruta = get_project_paths().results / "registry" / "quality_metrics.csv"
        if not ruta.is_file():
            pytest.skip(f"No existe '{ruta}'.")
        with ruta.open(encoding="utf-8") as fichero:
            return [
                fila
                for fila in csv.DictReader(fichero)
                if fila["experiment_id"] == STARTING_BASELINE_ID
                and fila["runtime"] == "pytorch"
                and fila["split"] == "val"
                and fila["map_50_95"]
            ]

    def test_el_suelo_es_la_diferencia_entre_los_dos_evaluadores(self) -> None:
        filas = self._filas_del_baseline()
        por_evaluador: dict[str, float] = {}
        for fila in filas:
            por_evaluador[fila["evaluator"]] = float(fila["map_50_95"])
        if len(por_evaluador) < 2:
            pytest.skip(
                f"'{STARTING_BASELINE_ID}' no está medido con dos evaluadores: el "
                f"suelo de lectura no se puede recalcular."
            )
        medido = max(por_evaluador.values()) - min(por_evaluador.values())
        assert medido == pytest.approx(EVALUATOR_READABILITY_FLOOR, abs=1e-6), (
            f"El suelo de lectura vale {EVALUATOR_READABILITY_FLOOR} y el registro dice "
            f"{medido:.6f}. Si las filas han cambiado, la constante tiene que cambiar "
            f"con ellas: es una medida, no un valor de gusto."
        )

    def test_el_eje_del_dispositivo_esta_tres_ordenes_por_debajo(self) -> None:
        """Con el evaluador fijo, cambiar de máquina no mueve la cifra."""
        filas = [
            fila
            for fila in self._filas_del_baseline()
            if fila["evaluator"] == CANONICAL_EVALUATOR and fila["predictions_path"]
        ]
        if len({fila["device_id"] for fila in filas}) < 2:
            pytest.skip(
                f"'{STARTING_BASELINE_ID}' no está medido con {CANONICAL_EVALUATOR} en "
                f"dos dispositivos: el eje del dispositivo no se puede medir."
            )

        valores: dict[str, float] = {}
        for fila in filas:
            volcado = get_project_paths().root / fila["predictions_path"]
            if not volcado.is_file():
                pytest.skip(f"falta el volcado '{volcado}'.")
            datos = json.loads(volcado.read_text(encoding="utf-8"))
            valores[fila["device_id"]] = datos["by_criterion"]["inference_resolution"]["map_50_95"]

        medido = max(valores.values()) - min(valores.values())
        assert medido < EVALUATOR_READABILITY_FLOOR / 1000.0, (
            f"El mismo checkpoint con el mismo evaluador difiere {medido:.3e} entre "
            f"{sorted(valores)}. Se esperaba ruido numérico, no un eje comparable al "
            f"del evaluador ({EVALUATOR_READABILITY_FLOOR}). Si esto crece, el "
            f"dispositivo deja de ser un desempate y pasa a ser un filtro."
        )
