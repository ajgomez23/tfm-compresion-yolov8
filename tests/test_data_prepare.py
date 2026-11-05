"""Tests de la preparación del dataset y del doble criterio de tamaño."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tfm_edge.data.prepare import (
    SH17_EXPECTED_ANNOTATIONS,
    SH17_EXPECTED_IMAGES,
    AnnotationAnomalies,
    ProvenanceError,
    RawDatasetMissingError,
    build_coco_dataset,
    extract_attributions,
    validate_provenance,
    verify_metadata_dimensions,
    verify_raw_available,
    write_manifest,
    yolo_to_absolute,
)
from tfm_edge.data.statistics import (
    COCO_MEDIUM_MAX_AREA,
    COCO_SMALL_MAX_AREA,
    SIZE_CRITERIA,
    area_at_inference,
    classify_size,
    classify_size_both,
    compute_statistics,
    labels_by_image,
    letterbox_scale,
    sizes_by_image,
)

CLASES = [f"clase_{i:02d}" for i in range(17)]


def crear_dataset(
    tmp_path: Path,
    num_imagenes: int = 6,
    tamano: tuple[int, int] = (2000, 1000),
    anotaciones_por_imagen: int = 2,
    uniforme: bool = False,
    con_metadatos: bool = True,
) -> Path:
    """Construye un dataset sintético con la estructura del paquete oficial."""
    from PIL import Image

    raiz = tmp_path / "raw"
    (raiz / "images").mkdir(parents=True)
    (raiz / "labels").mkdir(parents=True)
    if con_metadatos:
        (raiz / "meta-data").mkdir(parents=True)

    for i in range(num_imagenes):
        ancho, alto = tamano if uniforme else (tamano[0] + i * 10, tamano[1] + i * 5)
        nombre = f"img_{i:03d}"
        Image.new("RGB", (ancho, alto), (i * 20 % 256, 100, 150)).save(
            raiz / "images" / f"{nombre}.jpeg"
        )
        lineas = [
            f"{(i * anotaciones_por_imagen + j) % 17} 0.5 0.5 0.2 0.2"
            for j in range(anotaciones_por_imagen)
        ]
        (raiz / "labels" / f"{nombre}.txt").write_text("\n".join(lineas), encoding="utf-8")
        if con_metadatos:
            (raiz / "meta-data" / f"{nombre}.json").write_text(
                json.dumps(
                    {
                        "width": ancho,
                        "height": alto,
                        "photographer": f"Autor {i % 3}",
                        "photographer_url": f"https://example.invalid/a{i % 3}",
                        "url": f"https://example.invalid/foto/{i}",
                    }
                ),
                encoding="utf-8",
            )
    return raiz


# ---------------------------------------------------------------------------
# Validación de procedencia
# ---------------------------------------------------------------------------


class TestValidateProvenance:
    """Los criterios que detectan una reexportación con preprocesado destructivo."""

    def test_estructura_incompleta_se_detecta_pronto(self, tmp_path: Path) -> None:
        """Sin images/ o labels/ no se sigue comprobando nada más."""
        (tmp_path / "vacio").mkdir()
        informe = validate_provenance(tmp_path / "vacio")
        assert not informe.passed
        assert informe.checks[0].number == 0

    def test_detecta_resolucion_uniforme(self, tmp_path: Path) -> None:
        """Todas las imágenes del mismo tamaño delata un redimensionado masivo."""
        raiz = crear_dataset(tmp_path, num_imagenes=8, tamano=(640, 640), uniforme=True)
        informe = validate_provenance(raiz, sample_size=8)
        criterio_1 = next(c for c in informe.checks if c.number == 1)
        assert not criterio_1.passed
        assert "1 resoluciones distintas" in criterio_1.evidence

    def test_detecta_resolucion_demasiado_pequena(self, tmp_path: Path) -> None:
        """Imágenes muy por debajo de la resolución original son sospechosas."""
        raiz = crear_dataset(tmp_path, num_imagenes=6, tamano=(640, 480))
        criterio_2 = next(c for c in validate_provenance(raiz).checks if c.number == 2)
        assert not criterio_2.passed

    def test_detecta_recuento_incoherente(self, tmp_path: Path) -> None:
        """Un número de imágenes distinto del oficial hace fallar el criterio 4."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        criterio_4 = next(c for c in validate_provenance(raiz).checks if c.number == 4)
        assert not criterio_4.passed
        assert f"{SH17_EXPECTED_IMAGES:,}" in criterio_4.criterion
        assert f"{SH17_EXPECTED_ANNOTATIONS:,}" in criterio_4.criterion

    def test_detecta_clase_fuera_de_rango(self, tmp_path: Path) -> None:
        """Una categoría espuria, como la del export de Roboflow, se detecta."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        (raiz / "labels" / "img_000.txt").write_text("17 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        criterio_5 = next(c for c in validate_provenance(raiz).checks if c.number == 5)
        assert not criterio_5.passed
        assert "FUERA DE RANGO" in criterio_5.evidence

    def test_detecta_etiqueta_huerfana(self, tmp_path: Path) -> None:
        """Una etiqueta sin imagen rompe la correspondencia del criterio 6."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        (raiz / "labels" / "fantasma.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        criterio_6 = next(c for c in validate_provenance(raiz).checks if c.number == 6)
        assert not criterio_6.passed

    def test_detecta_particion_prefabricada(self, tmp_path: Path) -> None:
        """Una estructura train/valid/test impuesta se detecta."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        (raiz / "train").mkdir()
        criterio_7 = next(c for c in validate_provenance(raiz).checks if c.number == 7)
        assert not criterio_7.passed

    def test_detecta_marcador_de_reexportacion(self, tmp_path: Path) -> None:
        """Un README de Roboflow delata la procedencia."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        (raiz / "README.roboflow.txt").write_text("exportado", encoding="utf-8")
        criterio_8 = next(c for c in validate_provenance(raiz).checks if c.number == 8)
        assert not criterio_8.passed

    def test_el_informe_es_serializable_y_legible(self, tmp_path: Path) -> None:
        """El informe se puede volcar a JSON y resumir por pantalla."""
        informe = validate_provenance(crear_dataset(tmp_path, num_imagenes=4))
        assert json.dumps(informe.to_dict())
        assert "Validación de procedencia" in informe.summary()
        assert informe.failures


class TestProvenanceError:
    """El error de procedencia explica qué hacer."""

    def test_el_mensaje_orienta_al_usuario(self) -> None:
        """El mensaje menciona la reexportación y la fuente oficial."""
        error = ProvenanceError(["[1] criterio: evidencia"])
        texto = str(error)
        assert "reexportación" in texto
        assert "Kaggle" in texto


# ---------------------------------------------------------------------------
# Conversión de coordenadas y a COCO
# ---------------------------------------------------------------------------


class TestYoloToAbsolute:
    """Conversión de coordenadas YOLO normalizadas a absolutas."""

    def test_caja_centrada(self) -> None:
        """Una caja centrada que ocupa media imagen sale donde debe."""
        assert yolo_to_absolute(0.5, 0.5, 0.5, 0.5, 100, 100) == (25.0, 25.0, 50.0, 50.0)

    def test_usa_las_dimensiones_de_la_imagen(self) -> None:
        """El resultado escala con las dimensiones reales."""
        ancha = yolo_to_absolute(0.5, 0.5, 0.5, 0.5, 8192, 5461)
        estrecha = yolo_to_absolute(0.5, 0.5, 0.5, 0.5, 4096, 2730)
        assert ancha[2] == pytest.approx(estrecha[2] * 2, rel=1e-3)


class TestBuildCocoDataset:
    """Conversión del dataset a COCO."""

    def test_convierte_imagenes_y_anotaciones(self, tmp_path: Path) -> None:
        """El COCO resultante tiene tantas imágenes y anotaciones como el origen."""
        raiz = crear_dataset(tmp_path, num_imagenes=5, anotaciones_por_imagen=3)
        coco, anomalias = build_coco_dataset(raiz, CLASES)
        assert len(coco["images"]) == 5
        assert len(coco["annotations"]) == 15
        assert anomalias.total == 0

    def test_las_categorias_conservan_el_indice_de_yolo(self, tmp_path: Path) -> None:
        """Los identificadores de categoría coinciden con el índice de clase."""
        coco, _ = build_coco_dataset(crear_dataset(tmp_path, num_imagenes=2), CLASES)
        assert [c["id"] for c in coco["categories"]] == list(range(17))
        assert coco["categories"][0]["name"] == CLASES[0]

    def test_el_area_se_calcula_sobre_la_resolucion_original(self, tmp_path: Path) -> None:
        raiz = crear_dataset(tmp_path, num_imagenes=1, tamano=(1000, 500), uniforme=True)
        coco, _ = build_coco_dataset(raiz, CLASES)
        # 0.2 x 0.2 normalizado sobre 1000x500 = 200 x 100 px = 20.000 px2
        assert coco["annotations"][0]["area"] == pytest.approx(20000.0, rel=1e-6)

    def test_recorta_las_cajas_fuera_de_limites(self, tmp_path: Path) -> None:
        """Una caja que se sale se recorta y se cuenta como anomalía."""
        raiz = crear_dataset(tmp_path, num_imagenes=1)
        (raiz / "labels" / "img_000.txt").write_text("0 0.95 0.5 0.4 0.2\n", encoding="utf-8")
        coco, anomalias = build_coco_dataset(raiz, CLASES)
        assert len(anomalias.out_of_bounds) == 1
        x, _, ancho, _ = coco["annotations"][0]["bbox"]
        assert x + ancho <= coco["images"][0]["width"] + 1e-6

    def test_descarta_las_clases_fuera_de_rango(self, tmp_path: Path) -> None:
        """Una clase inexistente se descarta y se cuenta."""
        raiz = crear_dataset(tmp_path, num_imagenes=1)
        (raiz / "labels" / "img_000.txt").write_text("99 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        coco, anomalias = build_coco_dataset(raiz, CLASES)
        assert coco["annotations"] == []
        assert len(anomalias.invalid_class) == 1

    def test_cuenta_las_imagenes_sin_anotaciones(self, tmp_path: Path) -> None:
        """Una etiqueta vacía se registra como imagen sin anotaciones."""
        raiz = crear_dataset(tmp_path, num_imagenes=2)
        (raiz / "labels" / "img_000.txt").write_text("", encoding="utf-8")
        _, anomalias = build_coco_dataset(raiz, CLASES)
        assert "img_000.jpeg" in anomalias.images_without_annotations

    def test_puede_filtrar_por_particion(self, tmp_path: Path) -> None:
        """Se puede generar un COCO de solo un subconjunto de imágenes."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        coco, _ = build_coco_dataset(raiz, CLASES, image_stems=["img_000", "img_003"])
        assert {Path(i["file_name"]).stem for i in coco["images"]} == {"img_000", "img_003"}

    def test_falta_el_dataset(self, tmp_path: Path) -> None:
        """Sin images/ ni labels/ se falla con un mensaje accionable."""
        with pytest.raises(RawDatasetMissingError):
            build_coco_dataset(tmp_path / "no_existe", CLASES)

    def test_el_informe_de_anomalias_declara_su_politica(self) -> None:
        """El informe dice qué se hace con cada tipo de anomalía."""
        politica = AnnotationAnomalies().to_dict()["policy"]
        assert "RECORTAN" in politica
        assert "DESCARTAN" in politica


# Dimensiones de los metadatos.


class TestVerifyMetadataDimensions:
    """Comprobación de las dimensiones declaradas en los metadatos de Pexels."""

    def test_metadatos_correctos(self, tmp_path: Path) -> None:
        """Si coinciden, se declaran utilizables."""
        resultado = verify_metadata_dimensions(crear_dataset(tmp_path, num_imagenes=5))
        assert resultado["usable"] is True
        assert resultado["mismatches"] == 0

    def test_detecta_dimensiones_falseadas(self, tmp_path: Path) -> None:
        """Un metadato que no corresponde al fichero se detecta."""
        raiz = crear_dataset(tmp_path, num_imagenes=5)
        ruta = raiz / "meta-data" / "img_000.json"
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        datos["width"] = datos["width"] * 2
        ruta.write_text(json.dumps(datos), encoding="utf-8")

        resultado = verify_metadata_dimensions(raiz, sample_size=5)
        assert resultado["usable"] is False
        assert resultado["mismatches"] >= 1

    def test_sin_metadatos(self, tmp_path: Path) -> None:
        raiz = crear_dataset(tmp_path, num_imagenes=3, con_metadatos=False)
        assert verify_metadata_dimensions(raiz)["usable"] is False


# Criterio de tamaño de objeto.


class TestClassifySize:
    """Umbrales COCO."""

    @pytest.mark.parametrize(
        ("area", "esperado"),
        [
            (0.0, "small"),
            (COCO_SMALL_MAX_AREA - 1, "small"),
            (COCO_SMALL_MAX_AREA, "medium"),
            (COCO_MEDIUM_MAX_AREA, "medium"),
            (COCO_MEDIUM_MAX_AREA + 1, "large"),
        ],
    )
    def test_umbrales(self, area: float, esperado: str) -> None:
        """Los límites exactos caen del lado que fija el criterio COCO."""
        assert classify_size(area) == esperado

    def test_area_negativa(self) -> None:
        with pytest.raises(ValueError):
            classify_size(-1.0)


class TestLetterbox:
    """El redimensionado preserva la relación de aspecto."""

    def test_escala_de_una_imagen_apaisada(self) -> None:
        """La escala la fija la dimensión mayor."""
        assert letterbox_scale(1280, 640, 640) == pytest.approx(0.5)

    def test_escala_de_una_imagen_cuadrada(self) -> None:
        """En una imagen cuadrada ambas dimensiones dan la misma escala."""
        assert letterbox_scale(1280, 1280, 640) == pytest.approx(0.5)

    def test_no_es_lo_mismo_que_estirar(self) -> None:
        """Letterbox y estirado dan áreas distintas en imágenes no cuadradas."""
        ancho, alto, resolucion = 6000, 4000, 640
        factor_letterbox = letterbox_scale(ancho, alto, resolucion) ** 2
        factor_estirado = (resolucion / ancho) * (resolucion / alto)
        assert factor_letterbox < factor_estirado

    @pytest.mark.parametrize("dimensiones", [(0, 100), (100, 0), (-1, 100)])
    def test_dimensiones_invalidas(self, dimensiones: tuple[int, int]) -> None:
        """Dimensiones imposibles son un error explícito."""
        with pytest.raises(ValueError):
            letterbox_scale(dimensiones[0], dimensiones[1], 640)

    def test_resolucion_invalida(self) -> None:
        """Una resolución de inferencia no positiva es un error."""
        with pytest.raises(ValueError):
            letterbox_scale(100, 100, 0)


class TestAreaAtInference:
    """Conversión del área a la resolución de entrada del modelo."""

    def test_el_area_escala_con_el_cuadrado(self) -> None:
        """Al reducir el lado a la mitad, el área queda en la cuarta parte."""
        assert area_at_inference(1000.0, 1280, 1280, 640) == pytest.approx(250.0)

    def test_un_objeto_grande_del_original_puede_ser_pequeno_para_el_modelo(self) -> None:
        area_original = 100.0 * 100.0
        ambos = classify_size_both(area_original, 8000, 6000, 640)
        assert ambos["original_res"] == "large"
        assert ambos["inference_res"] == "small"


class TestComputeStatistics:
    """Estadísticas con doble criterio."""

    @pytest.fixture
    def coco(self, tmp_path: Path) -> dict:
        """Genera un COCO sintético."""
        raiz = crear_dataset(tmp_path, num_imagenes=8, anotaciones_por_imagen=3)
        datos, _ = build_coco_dataset(raiz, CLASES)
        return datos

    def test_calcula_los_dos_criterios(self, coco: dict) -> None:
        """El reparto de tamaños está presente para ambos criterios."""
        estadisticas = compute_statistics(coco, CLASES)
        for criterio in SIZE_CRITERIA:
            assert criterio in estadisticas.payload["size_distribution"]
            assert criterio in estadisticas.payload["class_by_size"]

    def test_registra_la_resolucion_de_inferencia(self, coco: dict) -> None:
        """Sin la resolución, dos ejecuciones serían incomparables."""
        estadisticas = compute_statistics(coco, CLASES, inference_resolution=416)
        assert estadisticas.inference_resolution == 416
        assert estadisticas.payload["size_criteria"]["resize_mode"] == "letterbox"

    def test_el_criterio_primario_es_configurable(self, coco: dict) -> None:
        assert compute_statistics(coco, CLASES).payload["size_criteria"]["primary"] == (
            "inference_res"
        )
        otro = compute_statistics(coco, CLASES, primary_criterion="original_res")
        assert otro.payload["size_criteria"]["primary"] == "original_res"

    def test_rechaza_un_criterio_inexistente(self, coco: dict) -> None:
        with pytest.raises(ValueError, match="Criterio de tamaño desconocido"):
            compute_statistics(coco, CLASES, primary_criterion="inventado")  # type: ignore[arg-type]

    def test_la_resolucion_menor_desplaza_objetos_a_pequeno(self, coco: dict) -> None:
        """Bajar la resolución de inferencia hace que haya más objetos pequeños."""
        a640 = compute_statistics(coco, CLASES, inference_resolution=640)
        a320 = compute_statistics(coco, CLASES, inference_resolution=320)
        pequenos_640 = a640.payload["size_distribution"]["inference_res"]["counts"]["small"]
        pequenos_320 = a320.payload["size_distribution"]["inference_res"]["counts"]["small"]
        assert pequenos_320 >= pequenos_640

    def test_el_criterio_secundario_no_depende_de_la_resolucion(self, coco: dict) -> None:
        """El de resolución original es invariante, como debe ser."""
        a640 = compute_statistics(coco, CLASES, inference_resolution=640)
        a320 = compute_statistics(coco, CLASES, inference_resolution=320)
        assert (
            a640.payload["size_distribution"]["original_res"]["counts"]
            == a320.payload["size_distribution"]["original_res"]["counts"]
        )

    def test_la_coocurrencia_es_simetrica(self, coco: dict) -> None:
        """La matriz de coocurrencia es simétrica por construcción."""
        matriz = compute_statistics(coco, CLASES).payload["cooccurrence"]
        for a in CLASES:
            for b in CLASES:
                assert matriz[a][b] == matriz[b][a]

    def test_vuelca_a_json_y_csv(self, coco: dict, tmp_path: Path) -> None:
        """Se generan las tablas, una por criterio para la cruzada."""
        estadisticas = compute_statistics(coco, CLASES)
        assert estadisticas.to_json(tmp_path / "stats.json").is_file()
        escritos = estadisticas.to_csv(tmp_path)
        for criterio in SIZE_CRITERIA:
            assert escritos[f"class_by_size_{criterio}"].is_file()

    def test_coco_vacio(self) -> None:
        """Un COCO sin imágenes es un error explícito."""
        with pytest.raises(ValueError, match="ninguna imagen"):
            compute_statistics({"images": [], "annotations": []}, CLASES)


class TestSizesByImage:
    """Rangos de tamaño presentes en cada imagen."""

    def test_depende_del_criterio(self, tmp_path: Path) -> None:
        """El mismo dataset da rangos distintos según el criterio."""
        raiz = crear_dataset(tmp_path, num_imagenes=4, tamano=(8000, 6000), uniforme=True)
        for i in range(4):
            (raiz / "labels" / f"img_{i:03d}.txt").write_text(
                f"{i} 0.5 0.5 0.05 0.05\n", encoding="utf-8"
            )
        coco, _ = build_coco_dataset(raiz, CLASES)

        por_original = sizes_by_image(coco, criterion="original_res")
        por_inferencia = sizes_by_image(coco, inference_resolution=640, criterion="inference_res")
        assert por_original != por_inferencia
        assert all(rangos == {"large"} for rangos in por_original.values())
        assert all(rangos == {"small"} for rangos in por_inferencia.values())

    def test_labels_incluye_las_imagenes_sin_anotaciones(self, tmp_path: Path) -> None:
        """Una imagen sin anotaciones no se pierde del particionado."""
        raiz = crear_dataset(tmp_path, num_imagenes=3)
        (raiz / "labels" / "img_000.txt").write_text("", encoding="utf-8")
        coco, _ = build_coco_dataset(raiz, CLASES)
        etiquetas = labels_by_image(coco)
        assert "img_000" in etiquetas
        assert etiquetas["img_000"] == set()


# ---------------------------------------------------------------------------
# Manifiesto y dependencia de `raw`
# ---------------------------------------------------------------------------


class TestManifiesto:
    """El acoplamiento entre `processed` y `raw` es explícito y comprobable."""

    def test_verifica_que_raw_sigue_presente(self, tmp_path: Path) -> None:
        """Con el manifiesto y el raw correctos, la comprobación pasa."""
        raiz = crear_dataset(tmp_path, num_imagenes=4)
        procesado = tmp_path / "processed"
        informe = validate_provenance(raiz, sample_size=4)
        write_manifest(procesado, raiz, 4, 8, informe, {"usable": True})
        assert verify_raw_available(procesado, raiz) == raiz

    def test_falla_si_falta_el_manifiesto(self, tmp_path: Path) -> None:
        """Sin manifiesto no se puede saber contra qué raw se generó."""
        with pytest.raises(RawDatasetMissingError, match="manifiesto"):
            verify_raw_available(tmp_path / "processed", tmp_path / "raw")

    def test_falla_si_raw_ha_cambiado(self, tmp_path: Path) -> None:
        raiz = crear_dataset(tmp_path, num_imagenes=4)
        procesado = tmp_path / "processed"
        informe = validate_provenance(raiz, sample_size=4)
        write_manifest(procesado, raiz, 4, 8, informe, {"usable": True})
        (raiz / "images" / "img_000.jpeg").unlink()
        with pytest.raises(RawDatasetMissingError, match="no es el que generó"):
            verify_raw_available(procesado, raiz)

    def test_el_manifiesto_avisa_de_la_dependencia(self, tmp_path: Path) -> None:
        """El propio manifiesto documenta que processed no es autocontenido."""
        raiz = crear_dataset(tmp_path, num_imagenes=2)
        informe = validate_provenance(raiz, sample_size=2)
        ruta = write_manifest(tmp_path / "processed", raiz, 2, 4, informe, {})
        contenido = json.loads(ruta.read_text(encoding="utf-8"))
        assert "NO contiene copias" in contenido["warning"]


class TestAtribuciones:
    """Cumplimiento de la cláusula BY de la licencia."""

    def test_genera_el_fichero_con_los_autores(self, tmp_path: Path) -> None:
        """Se listan los fotógrafos y la cita del dataset."""
        raiz = crear_dataset(tmp_path, num_imagenes=6)
        destino = extract_attributions(raiz, tmp_path / "docs" / "ATTRIB.md")
        texto = destino.read_text(encoding="utf-8")
        assert "CC BY-NC-SA 4.0" in texto
        assert "Autor 0" in texto
        assert "AHMAD2024" in texto

    def test_falla_sin_metadatos(self, tmp_path: Path) -> None:
        """Sin metadatos no se puede cumplir la atribución, y se dice."""
        raiz = crear_dataset(tmp_path, num_imagenes=2, con_metadatos=False)
        with pytest.raises(RawDatasetMissingError):
            extract_attributions(raiz, tmp_path / "ATTRIB.md")
