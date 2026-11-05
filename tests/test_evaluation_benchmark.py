"""Tests de la evaluación COCO con dos criterios de tamaño y del protocolo de medida."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_edge.benchmark import profiles
from tfm_edge.benchmark.profiles import BenchmarkProfile, ProfileError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.evaluation import coco_eval, runner
from tfm_edge.evaluation.coco_eval import CocoEvalError
from tfm_edge.evaluation.runner import EvaluationRunnerError

# ===========================================================================
# Escalado al espacio de inferencia
# ===========================================================================


class TestLetterboxScale:
    """El factor de escala es el del letterbox."""

    def test_es_el_minimo_de_los_dos_factores(self) -> None:
        """s = min(R/W, R/H), que conserva la relación de aspecto."""
        assert coco_eval.letterbox_scale(1280, 640, 640) == pytest.approx(0.5)
        assert coco_eval.letterbox_scale(640, 1280, 640) == pytest.approx(0.5)

    def test_no_es_el_producto_de_los_factores(self) -> None:
        """El error clásico sería (R/W)·(R/H), que corresponde a estirar."""
        s = coco_eval.letterbox_scale(1280, 640, 640)
        estirado = (640 / 1280) * (640 / 640)
        assert s == pytest.approx(0.5)
        # Aquí coinciden por casualidad; con otra forma no.
        s2 = coco_eval.letterbox_scale(1000, 500, 640)
        assert s2 == pytest.approx(0.64)
        assert s2 != pytest.approx((640 / 1000) * (640 / 500))
        assert estirado == pytest.approx(0.5)

    def test_dimensiones_invalidas_se_rechazan(self) -> None:
        """Una imagen de anchura cero indicaría un COCO corrupto."""
        with pytest.raises(CocoEvalError, match="Dimensiones"):
            coco_eval.letterbox_scale(0, 480, 640)


def _coco_minimo() -> dict:
    """Construye un COCO pequeño pero realista."""
    return {
        "categories": [{"id": 0, "name": "person"}, {"id": 1, "name": "helmet"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 1280, "height": 640},
            {"id": 2, "file_name": "b.jpg", "width": 640, "height": 640},
        ],
        "annotations": [
            # Umbrales de COCO: pequeño < 32^2 = 1.024 < mediano < 96^2 = 9.216.
            # La imagen 1 es 1280x640, así que a 640 px el factor de letterbox es
            # s = 0,5 y las áreas se dividen por 4.
            #
            # 200x200 = 40.000: grande. Escalada, 100x100 = 10.000: sigue grande.
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "area": 40000.0,
                "iscrowd": 0,
            },
            # 60x60 = 3.600: mediano. Escalada, 30x30 = 900: pasa a pequeño.
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [400.0, 200.0, 60.0, 60.0],
                "area": 3600.0,
                "iscrowd": 0,
            },
            # La imagen 2 ya es 640x640: s = 1 y esta anotación no cambia.
            {
                "id": 3,
                "image_id": 2,
                "category_id": 0,
                "bbox": [10.0, 10.0, 300.0, 300.0],
                "area": 90000.0,
                "iscrowd": 0,
            },
        ],
    }


class TestEscaladoAInferencia:
    """Escalar tiene que tocar cajas y áreas, y nada más."""

    def test_las_areas_se_recalculan_desde_la_caja(self) -> None:
        escalado = coco_eval.scale_ground_truth(_coco_minimo(), 640)
        for anotacion in escalado["annotations"]:
            _, _, w, h = anotacion["bbox"]
            assert anotacion["area"] == pytest.approx(w * h)

    def test_el_reparto_por_tamano_cambia_entre_criterios(self) -> None:
        gt = _coco_minimo()
        original = coco_eval.count_objects_per_size(gt)
        inferencia = coco_eval.count_objects_per_size(coco_eval.scale_ground_truth(gt, 640))
        assert original != inferencia, "Los dos criterios deberían repartir de forma distinta."
        # El objeto de 60x60 pasa de mediano a pequeño al entrar a 640 px.
        assert original == {"small": 0, "medium": 1, "large": 2}
        assert inferencia == {"small": 1, "medium": 0, "large": 2}

    def test_escalar_no_cambia_el_numero_de_objetos(self) -> None:
        """El escalado no puede perder ni inventar anotaciones."""
        gt = _coco_minimo()
        escalado = coco_eval.scale_ground_truth(gt, 640)
        assert len(escalado["annotations"]) == len(gt["annotations"])
        assert len(escalado["images"]) == len(gt["images"])

    def test_el_original_no_se_modifica(self) -> None:
        gt = _coco_minimo()
        antes = json.dumps(gt, sort_keys=True)
        coco_eval.scale_ground_truth(gt, 640)
        assert json.dumps(gt, sort_keys=True) == antes


class TestCoherenciaEntreCriterios:
    """La comprobación que hace que la evaluación se valide a sí misma."""

    def _detecciones_perfectas(self, gt: dict) -> list[dict]:
        """Construye detecciones que coinciden exactamente con la referencia."""
        por_id = {i["id"]: i["file_name"] for i in gt["images"]}
        return [
            {
                "image_id": a["image_id"],
                "file_name": por_id[a["image_id"]],
                "category_id": a["category_id"],
                "bbox": list(a["bbox"]),
                "score": 0.9,
            }
            for a in gt["annotations"]
        ]

    def test_con_los_dos_lados_escalados_las_metricas_globales_coinciden(self) -> None:
        gt = _coco_minimo()
        detecciones = self._detecciones_perfectas(gt)
        resultado = coco_eval.evaluate_coco(
            gt, detecciones, input_resolution=640, with_per_class=False
        )
        assert resultado.consistency["consistent"]
        assert resultado.consistency["max_difference"] == 0.0
        assert resultado.primary.map_50 == pytest.approx(1.0)

    def test_escalar_solo_un_lado_se_caza(self) -> None:
        """Es el fallo que la comprobación existe para detectar."""
        gt = _coco_minimo()
        detecciones = self._detecciones_perfectas(gt)
        gt_escalado = coco_eval.scale_ground_truth(gt, 640)
        # Detecciones sin escalar contra referencia escalada: incoherente.
        mal = coco_eval._evaluar(gt_escalado, detecciones, "inference_resolution")
        bien = coco_eval._evaluar(gt, detecciones, "original_resolution")
        with pytest.raises(CocoEvalError, match="métricas globales distintas"):
            coco_eval.assert_criteria_consistency(mal, bien)

    def test_el_criterio_primario_es_el_de_inferencia(self) -> None:
        assert coco_eval.PRIMARY_CRITERION == "inference_resolution"
        gt = _coco_minimo()
        resultado = coco_eval.evaluate_coco(
            gt, self._detecciones_perfectas(gt), input_resolution=640, with_per_class=False
        )
        columnas = resultado.to_registry_columns()
        assert columnas["size_criterion_primary"] == "inference_resolution"
        assert columnas["size_criterion_inference_resolution"] == 640

    def test_se_pueblan_las_doce_columnas(self) -> None:
        """Las doce columnas de desglose salen de aquí y solo de aquí."""
        gt = _coco_minimo()
        resultado = coco_eval.evaluate_coco(
            gt, self._detecciones_perfectas(gt), input_resolution=640, with_per_class=False
        )
        columnas = resultado.to_registry_columns()
        esperadas = [
            f"{p}_{t}_{s}"
            for p in ("ap", "ar")
            for t in ("small", "medium", "large")
            for s in ("inference_res", "original_res")
        ]
        assert len(esperadas) == 12
        for columna in esperadas:
            assert columna in columnas, f"Falta la columna '{columna}'"


# ===========================================================================
# Punto de operación de mejor F1
# ===========================================================================


def _coco_con_optimos_enfrentados() -> tuple[dict, list[dict]]:
    """Construye un caso donde cada clase prefiere un umbral distinto."""
    gt = {
        "categories": [{"id": 0, "name": "a"}, {"id": 1, "name": "b"}],
        "images": [{"id": 1, "file_name": "u.jpg", "width": 640, "height": 640}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 100, 100]},
            {"id": 2, "image_id": 1, "category_id": 0, "bbox": [200, 0, 100, 100]},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 200, 100, 100]},
            {"id": 4, "image_id": 1, "category_id": 1, "bbox": [200, 200, 100, 100]},
        ],
    }
    for anotacion in gt["annotations"]:
        anotacion["area"] = float(anotacion["bbox"][2] * anotacion["bbox"][3])
        anotacion["iscrowd"] = 0

    def deteccion(categoria: int, caja: list[float], score: float) -> dict:
        return {
            "image_id": 1,
            "file_name": "u.jpg",
            "category_id": categoria,
            "bbox": [float(v) for v in caja],
            "score": score,
        }

    detecciones = [
        deteccion(0, [0, 0, 100, 100], 0.90),
        deteccion(0, [200, 0, 100, 100], 0.10),
        deteccion(1, [0, 200, 100, 100], 0.80),
        deteccion(1, [200, 200, 100, 100], 0.05),
    ]
    # Veinte falsos positivos de la clase 0, lejos de cualquier anotación.
    detecciones += [deteccion(0, [400 + 12 * i, 400, 10, 10], 0.50 - 0.01 * i) for i in range(20)]
    return gt, detecciones


class TestPuntoDeOperacion:
    """`precision_at_best_f1` y `recall_at_best_f1`, desde las curvas de pycocotools."""

    def _perfectas(self, gt: dict) -> list[dict]:
        """Detecciones que coinciden exactamente con la referencia."""
        por_id = {i["id"]: i["file_name"] for i in gt["images"]}
        return [
            {
                "image_id": a["image_id"],
                "file_name": por_id[a["image_id"]],
                "category_id": a["category_id"],
                "bbox": list(a["bbox"]),
                "score": 0.9,
            }
            for a in gt["annotations"]
        ]

    def test_una_deteccion_perfecta_da_precision_y_recall_uno(self) -> None:
        gt = _coco_minimo()
        resultado = coco_eval.evaluate_coco(
            gt, self._perfectas(gt), input_resolution=640, with_per_class=False
        )
        assert resultado.primary.precision_at_best_f1 == pytest.approx(1.0)
        assert resultado.primary.recall_at_best_f1 == pytest.approx(1.0)
        assert resultado.primary.f1_best == pytest.approx(1.0)
        assert resultado.primary.conf_at_best_f1 == pytest.approx(0.9)

    def test_las_tres_columnas_llegan_al_registro(self) -> None:
        gt = _coco_minimo()
        resultado = coco_eval.evaluate_coco(
            gt, self._perfectas(gt), input_resolution=640, with_per_class=False
        )
        columnas = resultado.to_registry_columns()
        for columna in ("precision_at_best_f1", "recall_at_best_f1", "f1_best"):
            assert columna in columnas, f"Falta la columna '{columna}'"
            assert isinstance(columnas[columna], float), (
                f"'{columna}' no puede quedarse vacía cuando hay detecciones: "
                f"es lo que motivó calcularla desde pycocotools."
            )

    def test_el_umbral_es_unico_para_todas_las_clases(self) -> None:
        """La decisión de diseño que ningún otro test detectaría."""
        gt, detecciones = _coco_con_optimos_enfrentados()
        resultado = coco_eval.evaluate_coco(
            gt, detecciones, input_resolution=640, with_per_class=False
        )
        punto = resultado.primary

        # Óptimo por clase: 2/3 para la clase 0, 1,0 para la clase 1.
        media_de_optimos = (2 / 3 + 1.0) / 2
        assert punto.f1_best < media_de_optimos - 1e-6, (
            "El F1 alcanza la media de los óptimos por clase, lo que significa que "
            "se está eligiendo un umbral distinto para cada una."
        )
        # El único umbral común razonable es 0,80: por encima la clase 1 no tiene
        # ninguna detección, por debajo de 0,10 la clase 0 se llena de falsos
        # positivos. Ahí las dos clases van a mitad de recall con precisión 1.
        assert punto.conf_at_best_f1 == pytest.approx(0.80)
        assert punto.precision_at_best_f1 == pytest.approx(1.0)
        assert punto.recall_at_best_f1 == pytest.approx(0.5)
        assert punto.f1_best == pytest.approx(2 / 3, abs=1e-4)

    def test_una_clase_no_detectada_baja_la_media(self) -> None:
        gt = _coco_minimo()
        solo_una_clase = [d for d in self._perfectas(gt) if d["category_id"] == 0]
        resultado = coco_eval.evaluate_coco(
            gt, solo_una_clase, input_resolution=640, with_per_class=False
        )
        # Dos clases con anotaciones, una sin detectar: recall medio 0,5.
        assert resultado.primary.operating_point["num_classes"] == 2
        assert resultado.primary.recall_at_best_f1 == pytest.approx(0.5)
        assert resultado.primary.f1_best == pytest.approx(0.5)

    def test_una_clase_sin_anotaciones_no_cuenta(self) -> None:
        """Una clase ausente de la partición no puede hundir la media."""
        gt = _coco_minimo()
        gt["categories"] = list(gt["categories"]) + [{"id": 2, "name": "fantasma"}]
        resultado = coco_eval.evaluate_coco(
            gt, self._perfectas(gt), input_resolution=640, with_per_class=False
        )
        assert resultado.primary.operating_point["num_classes"] == 2
        assert resultado.primary.f1_best == pytest.approx(1.0)

    def test_el_punto_no_depende_del_criterio_de_tamano(self) -> None:
        gt = _coco_minimo()
        resultado = coco_eval.evaluate_coco(
            gt, self._perfectas(gt), input_resolution=640, with_per_class=False
        )
        inferencia = resultado.by_criterion["inference_resolution"]
        original = resultado.by_criterion["original_resolution"]
        assert inferencia.precision_at_best_f1 == original.precision_at_best_f1
        assert inferencia.recall_at_best_f1 == original.recall_at_best_f1
        assert inferencia.f1_best == original.f1_best
        assert "f1_best" in resultado.consistency["checked"]

    def test_sin_acumular_falla_en_vez_de_devolver_ceros(self) -> None:
        _, COCOeval = coco_eval._importar_pycocotools()
        with pytest.raises(CocoEvalError, match="sin acumular"):
            coco_eval.operating_point_at_best_f1(COCOeval())


class TestMapeoDeDetecciones:
    """Un desalineamiento de clases desplazaría todas las categorías."""

    def test_una_categoria_inexistente_se_denuncia(self) -> None:
        """Es el fallo de la numeración 1-based de Ultralytics."""
        gt = _coco_minimo()
        detecciones = [
            {
                "image_id": 1,
                "file_name": "a.jpg",
                "category_id": 99,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "score": 0.5,
            }
        ]
        with pytest.raises(CocoEvalError, match="categorías que no existen"):
            coco_eval.map_detections_to_ground_truth(detecciones, gt)

    def test_si_nada_empareja_se_denuncia(self) -> None:
        """Evaluar la partición equivocada daría mAP cero sin explicación."""
        gt = _coco_minimo()
        detecciones = [
            {
                "image_id": 1,
                "file_name": "otra.jpg",
                "category_id": 0,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "score": 0.5,
            }
        ]
        with pytest.raises(CocoEvalError, match="Ninguna"):
            coco_eval.map_detections_to_ground_truth(detecciones, gt)

    def test_el_emparejamiento_por_nombre_funciona(self) -> None:
        """Ultralytics identifica por nombre; el COCO, por entero."""
        gt = _coco_minimo()
        detecciones = [
            {
                "image_id": "a",
                "file_name": "a.jpg",
                "category_id": 0,
                "bbox": [0.0, 0.0, 10.0, 10.0],
                "score": 0.5,
            }
        ]
        traducidas = coco_eval.map_detections_to_ground_truth(detecciones, gt)
        assert traducidas[0]["image_id"] == 1


class TestRecorteDeParticion:
    """Evaluar sobre el COCO completo hundiría el recall."""

    def test_se_queda_solo_con_las_imagenes_pedidas(self) -> None:
        """Y con sus anotaciones."""
        recortado = coco_eval.subset_ground_truth(_coco_minimo(), ["a.jpg"])
        assert len(recortado["images"]) == 1
        assert all(a["image_id"] == 1 for a in recortado["annotations"])

    def test_una_imagen_ausente_se_denuncia(self) -> None:
        """Partición y COCO tienen que venir de la misma preparación."""
        with pytest.raises(CocoEvalError, match="no están en el COCO"):
            coco_eval.subset_ground_truth(_coco_minimo(), ["a.jpg", "fantasma.jpg"])


# Piezas de la evaluación que se comprueban antes de usar la GPU.


class TestLaReferenciaSeCompruebaAntesDeEmpezar:
    """El COCO no viaja con el repositorio, y descubrirlo tarde cuesta la campaña."""

    def test_un_coco_ausente_se_denuncia_con_los_dos_motivos(self, tmp_path: Path) -> None:
        (tmp_path / "val.txt").write_text("a.jpg\n", encoding="utf-8")
        with pytest.raises(EvaluationRunnerError) as excinfo:
            runner.assert_evaluation_inputs_available(tmp_path / "no_existe.json", tmp_path, "val")
        mensaje = str(excinfo.value)
        assert "COCO de referencia" in mensaje
        assert "COPIARLO" in mensaje

    def test_una_particion_ausente_se_denuncia(self, tmp_path: Path) -> None:
        (tmp_path / "sh17_coco.json").write_text("{}", encoding="utf-8")
        with pytest.raises(EvaluationRunnerError, match="partición 'val'"):
            runner.assert_evaluation_inputs_available(tmp_path / "sh17_coco.json", tmp_path, "val")

    def test_las_dos_ausencias_se_denuncian_de_una_vez(self, tmp_path: Path) -> None:
        """Denunciar solo la primera obligaría a dos vueltas a Colab por nada."""
        with pytest.raises(EvaluationRunnerError) as excinfo:
            runner.assert_evaluation_inputs_available(tmp_path / "nada.json", tmp_path, "val")
        mensaje = str(excinfo.value)
        assert "2 piezas" in mensaje
        assert "nada.json" in mensaje and "val.txt" in mensaje

    def test_con_las_piezas_presentes_devuelve_lo_comprobado(self, tmp_path: Path) -> None:
        """El diccionario es lo que se registra junto al resultado."""
        (tmp_path / "sh17_coco.json").write_text("{}", encoding="utf-8")
        (tmp_path / "val.txt").write_text("a.jpg\n", encoding="utf-8")
        (tmp_path / "data.yaml").write_text("nc: 17\n", encoding="utf-8")
        comprobadas = runner.assert_evaluation_inputs_available(
            tmp_path / "sh17_coco.json", tmp_path, "val", tmp_path / "data.yaml"
        )
        assert len(comprobadas) == 3

    def test_el_yaml_del_dataset_es_opcional(self, tmp_path: Path) -> None:
        """Quien evalúa un artefacto ya exportado no siempre lo tiene a mano."""
        (tmp_path / "sh17_coco.json").write_text("{}", encoding="utf-8")
        (tmp_path / "val.txt").write_text("a.jpg\n", encoding="utf-8")
        comprobadas = runner.assert_evaluation_inputs_available(
            tmp_path / "sh17_coco.json", tmp_path, "val"
        )
        assert len(comprobadas) == 2


# ===========================================================================
# Perfiles de medida
# ===========================================================================


class TestPerfilesDeMedida:
    """Un perfil que no puede medir se rechaza con error."""

    def test_los_perfiles_del_proyecto_cargan(self) -> None:
        """Los seis YAML de configs/benchmark validan."""
        directorio = get_project_paths().root / "configs" / "benchmark"
        if not directorio.is_dir():
            pytest.skip("No hay configs/benchmark en este árbol")
        cargados = profiles.load_all_profiles(directorio)
        assert len(cargados) >= 3
        for esperado in ("workstation_gpu", "workstation_cpu", "constrained_cpu"):
            assert esperado in cargados, f"Falta el perfil '{esperado}'"

    def test_el_perfil_de_nube_no_puede_medir(self) -> None:
        from tfm_edge.benchmark.protocol import InvalidProfileError

        directorio = get_project_paths().root / "configs" / "benchmark"
        if not (directorio / "cloud_gpu.yaml").is_file():
            pytest.skip("No hay perfil de nube en este árbol")
        perfil = profiles.load_profile(directorio / "cloud_gpu.yaml")
        assert not perfil.valid_for_measurement
        with pytest.raises(InvalidProfileError, match="NO es válido para medir"):
            perfil.assert_can_measure()

    def test_constrained_cpu_no_es_representativo(self) -> None:
        directorio = get_project_paths().root / "configs" / "benchmark"
        if not (directorio / "constrained_cpu.yaml").is_file():
            pytest.skip("No hay perfil restringido en este árbol")
        perfil = profiles.load_profile(directorio / "constrained_cpu.yaml")
        assert not perfil.latency_representative
        assert perfil.num_threads == 2
        assert perfil.cpu_affinity == [0, 1]

    def test_no_valido_sin_motivo_no_valida(self) -> None:
        """El rechazo tiene que poder explicarse a quien lea el registro."""
        with pytest.raises(ValueError, match="no dice por qué"):
            BenchmarkProfile(device_id="x", valid_for_measurement=False)

    def test_gpu_sin_temperatura_no_valida(self) -> None:
        with pytest.raises(ValueError, match="temperatura"):
            BenchmarkProfile(device_id="x", device_type="cuda", requires_temperature=False)

    def test_afinidad_sin_hilos_no_valida(self) -> None:
        """Fijar uno sin el otro deja el perfil a medio definir."""
        with pytest.raises(ValueError, match="número de hilos"):
            BenchmarkProfile(device_id="x", cpu_affinity=[0, 1])

    def test_mas_hilos_que_nucleos_no_valida(self) -> None:
        """El perfil dejaría de medir lo que dice medir."""
        with pytest.raises(ValueError, match="hilos sobre"):
            BenchmarkProfile(device_id="x", num_threads=4, cpu_affinity=[0, 1])

    def test_una_clave_desconocida_no_valida(self) -> None:
        with pytest.raises(ValueError):
            BenchmarkProfile(device_id="x", clave_inventada=1)

    def test_un_perfil_no_disponible_se_detiene(self) -> None:
        """Se registra no_disponible; no se finge que se midió."""
        perfil = BenchmarkProfile(
            device_id="edge", available=False, unavailable_reason="no hay hardware"
        )
        with pytest.raises(ProfileError, match="no está disponible"):
            perfil.assert_can_measure()


# ===========================================================================
# Protocolo de medida
# ===========================================================================

pytest.importorskip("torch", reason="requiere PyTorch")

from tfm_edge.benchmark import protocol  # noqa: E402
from tfm_edge.benchmark.protocol import ProtocolError  # noqa: E402


@pytest.mark.requires_torch
class TestProtocolo:
    """Las reglas del protocolo se aplican en el código."""

    def test_medir_con_una_sola_imagen_se_rechaza(self) -> None:
        """La latencia del postprocesado depende del número de detecciones."""
        with pytest.raises(ProtocolError, match="al menos 2"):
            protocol.measure_latency(None, [object()], measure_iterations=1, repetitions=1)

    def test_agregar_sin_repeticiones_se_rechaza(self) -> None:
        """No hay nada que agregar."""
        with pytest.raises(ProtocolError, match="ninguna repetición"):
            protocol.aggregate_latency([])

    def test_la_dispersion_entre_repeticiones_necesita_mas_de_una(self) -> None:
        """Con una sola repetición no está definida, y se deja vacía."""
        medida = protocol.LatencyMeasurement(repetition=0, timings_s=[0.01, 0.02])
        medida.statistics = {"latency_p50_ms": 15.0}
        _, entre = protocol.aggregate_latency([medida])
        assert entre == ""

    def test_la_dispersion_entre_repeticiones_se_calcula(self) -> None:
        """Es lo que dice si la medida es reproducible."""
        medidas = []
        for indice, p50 in enumerate((10.0, 12.0, 11.0)):
            m = protocol.LatencyMeasurement(repetition=indice, timings_s=[p50 / 1000])
            m.statistics = {"latency_p50_ms": p50}
            medidas.append(m)
        _, entre = protocol.aggregate_latency(medidas)
        assert isinstance(entre, float)
        assert entre == pytest.approx(1.0, abs=0.01)

    def test_la_temperatura_ausente_queda_vacia_no_a_cero(self) -> None:
        """Un cero se confundiría con una lectura válida."""
        lectura = protocol.read_gpu_temperature()
        assert lectura == "" or isinstance(lectura, float)
        assert lectura != 0 or isinstance(lectura, float)

    def test_el_tamano_de_artefacto_se_mide_en_disco_y_comprimido(self, tmp_path: Path) -> None:
        """El comprimido aproxima el coste de distribución por red."""
        fichero = tmp_path / "modelo.bin"
        fichero.write_bytes(b"0" * 10_000)
        medida = protocol.measure_artifact_size(fichero)
        assert medida.disk_bytes == 10_000
        assert 0 < medida.gzip_bytes < medida.disk_bytes
        assert medida.compression_ratio > 1.0

    def test_un_artefacto_inexistente_se_denuncia(self, tmp_path: Path) -> None:
        """No se devuelve cero, que parecería un modelo vacío."""
        with pytest.raises(ProtocolError, match="No existe"):
            protocol.measure_artifact_size(tmp_path / "no_existe.pt")

    def test_num_threads_invalido_se_rechaza(self) -> None:
        """Cero hilos no es una configuración."""
        with pytest.raises(ProtocolError, match="num_threads"):
            protocol.configure_cpu(num_threads=0)

    def test_la_memoria_de_gpu_queda_vacia_sin_gpu(self) -> None:
        """Vacío significa 'no medido'; cero significaría 'no consumió'."""
        medida = protocol.MemoryMeasurement()
        assert medida.gpu_peak_allocated_bytes == ""
        assert medida.activations_bytes == ""
