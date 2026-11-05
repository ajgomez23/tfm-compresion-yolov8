"""Test de guarda: Ultralytics carga el dataset, y sigue haciéndolo tras moverlo."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="requiere PyTorch")
pytest.importorskip("ultralytics", reason="requiere Ultralytics")

import yaml  # noqa: E402

from tfm_edge.common.paths import get_project_paths  # noqa: E402
from tfm_edge.data.prepare import (  # noqa: E402
    ULTRALYTICS_DATASET_FILENAME,
    ULTRALYTICS_SPLIT_TEMPLATE,
)

pytestmark = [pytest.mark.requires_torch, pytest.mark.requires_dataset]

#: Particiones que declara el YAML.
PARTICIONES = ("train", "val", "test")


@pytest.fixture(scope="module")
def raiz_dataset() -> Path:
    """Devuelve la raíz del dataset preparado, u omite los tests si no existe."""
    raiz = get_project_paths().data_raw
    if not (raiz / ULTRALYTICS_DATASET_FILENAME).is_file():
        pytest.skip(
            f"No existe '{raiz / ULTRALYTICS_DATASET_FILENAME}'. "
            "Prepara el dataset con scripts/02_prepare_dataset.py."
        )
    return raiz


@pytest.fixture(scope="module")
def ruta_yaml(raiz_dataset: Path) -> Path:
    """Devuelve la ruta del YAML de dataset."""
    return raiz_dataset / ULTRALYTICS_DATASET_FILENAME


@pytest.fixture(scope="module")
def contenido_yaml(ruta_yaml: Path) -> dict:
    """Carga el YAML generado."""
    return yaml.safe_load(ruta_yaml.read_text(encoding="utf-8"))


class TestElArtefactoEsPortable:
    """Nada dentro del YAML ni de las listas puede atar el dataset a una máquina."""

    def test_el_yaml_no_declara_la_clave_path(self, contenido_yaml: dict) -> None:
        """El YAML omite ``path``, para que la raíz sea su propio directorio."""
        assert "path" not in contenido_yaml, (
            "El YAML declara 'path'. Si es absoluta, ata el dataset a esta "
            "máquina; si es relativa, Ultralytics la resuelve contra su "
            "DATASETS_DIR global. Omitirla es lo correcto."
        )

    def test_el_yaml_no_contiene_rutas_absolutas(self, ruta_yaml: Path) -> None:
        """Ninguna clave del YAML lleva una ruta absoluta."""
        contenido = yaml.safe_load(ruta_yaml.read_text(encoding="utf-8"))
        for clave in PARTICIONES:
            valor = contenido[clave]
            assert not Path(valor).is_absolute(), (
                f"La clave '{clave}' vale '{valor}', que es absoluta y deja de "
                "existir al mover el dataset a otra máquina."
            )

    def test_las_listas_usan_el_prefijo_que_ultralytics_resuelve(self, raiz_dataset: Path) -> None:
        """Cada línea empieza por ``./``, la única forma relativa que funciona."""
        for particion in PARTICIONES:
            lista = raiz_dataset / ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)
            lineas = [
                linea for linea in lista.read_text(encoding="utf-8").splitlines() if linea.strip()
            ]
            assert lineas, f"La lista de '{particion}' está vacía."
            malas = [linea for linea in lineas if not linea.startswith("./")]
            assert not malas, (
                f"La lista de '{particion}' tiene {len(malas)} líneas sin el "
                f"prefijo './'. Primeras: {malas[:3]}"
            )

    def test_ninguna_linea_tiene_dos_secuencias_de_prefijo(self, raiz_dataset: Path) -> None:
        """Ninguna línea contiene ``./`` más de una vez."""
        for particion in PARTICIONES:
            lista = raiz_dataset / ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)
            lineas = [
                linea for linea in lista.read_text(encoding="utf-8").splitlines() if linea.strip()
            ]
            malas = [linea for linea in lineas if linea.count("./") != 1]
            assert not malas, f"Líneas con más de un './': {malas[:3]}"

    def test_la_lista_esta_junto_al_directorio_de_imagenes(self, raiz_dataset: Path) -> None:
        """Las listas viven en la misma carpeta que ``images/``."""
        for particion in PARTICIONES:
            lista = raiz_dataset / ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)
            assert lista.is_file(), f"No existe '{lista}'."
            assert (
                lista.parent / "images"
            ).is_dir(), f"'{lista}' no está junto a un directorio 'images/'."


class TestUltralyticsConstruyeElDataloader:
    """Ultralytics construye el dataloader con las listas generadas."""

    def test_check_det_dataset_acepta_el_yaml(self, ruta_yaml: Path) -> None:
        """La validación de dataset de Ultralytics acepta nuestro YAML."""
        from ultralytics.data.utils import check_det_dataset

        datos = check_det_dataset(str(ruta_yaml), autodownload=False)
        assert datos["nc"] == 17
        assert len(datos["names"]) == 17
        assert datos["names"][0] == "person"

    def test_construye_un_dataset_real_y_lee_una_muestra(self, ruta_yaml: Path) -> None:
        """Ultralytics construye el dataset y devuelve una muestra utilizable."""
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        datos = check_det_dataset(str(ruta_yaml), autodownload=False)
        cfg = get_cfg(overrides={"imgsz": 320, "mode": "train"})
        # fraction reduce el dataset: no hace falta indexar miles de imagenes
        # para comprobar que las rutas se resuelven.
        cfg.fraction = 0.005

        dataset = build_yolo_dataset(cfg, datos["val"], batch=2, data=datos, mode="val")
        assert len(dataset) > 0, "Ultralytics no ha encontrado ninguna imagen."

        muestra = dataset[0]
        assert "img" in muestra
        assert "cls" in muestra
        assert "bboxes" in muestra
        assert muestra["img"].shape[0] == 3, "Se esperaban 3 canales de color."

    def test_las_clases_de_las_etiquetas_estan_en_rango(self, ruta_yaml: Path) -> None:
        """Las etiquetas que carga Ultralytics tienen índices válidos."""
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        datos = check_det_dataset(str(ruta_yaml), autodownload=False)
        cfg = get_cfg(overrides={"imgsz": 320, "mode": "train"})
        cfg.fraction = 0.005
        dataset = build_yolo_dataset(cfg, datos["val"], batch=2, data=datos, mode="val")

        for indice in range(min(5, len(dataset))):
            clases = dataset[indice]["cls"]
            if clases.numel() == 0:
                continue
            assert int(clases.min()) >= 0
            assert int(clases.max()) < 17


class TestSobreviveAMoverse:
    """Los tests que habrían cazado los dos fallos reales."""

    def test_carga_desde_otro_directorio_de_trabajo(self, ruta_yaml: Path, tmp_path: Path) -> None:
        """El dataset carga con el proceso arrancado en otro sitio."""
        from ultralytics.data.utils import check_det_dataset

        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            datos = check_det_dataset(str(ruta_yaml), autodownload=False)
            assert Path(datos["val"]).is_file()
        finally:
            os.chdir(original)

    def test_carga_tras_copiar_el_dataset_a_otra_ubicacion(
        self, raiz_dataset: Path, tmp_path: Path
    ) -> None:
        """El dataset carga desde una ruta distinta de la que lo generó."""
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset
        from ultralytics.data.utils import check_det_dataset

        destino = tmp_path / "otra_ubicacion" / "sh17"
        (destino / "images").mkdir(parents=True)
        (destino / "labels").mkdir(parents=True)
        shutil.copy2(raiz_dataset / ULTRALYTICS_DATASET_FILENAME, destino)

        # Se recorta la lista de val a las primeras imagenes, y se copian solo esas.
        nombre_lista = ULTRALYTICS_SPLIT_TEMPLATE.format(split="val")
        lineas = [
            linea
            for linea in (raiz_dataset / nombre_lista).read_text(encoding="utf-8").splitlines()
            if linea.strip()
        ][:8]
        for particion in PARTICIONES:
            (destino / ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)).write_text(
                "\n".join(lineas) + "\n", encoding="utf-8"
            )
        for linea in lineas:
            nombre = Path(linea).name
            shutil.copy2(raiz_dataset / "images" / nombre, destino / "images" / nombre)
            etiqueta = raiz_dataset / "labels" / f"{Path(nombre).stem}.txt"
            if etiqueta.is_file():
                shutil.copy2(etiqueta, destino / "labels" / etiqueta.name)

        # Y se carga desde un directorio de trabajo que tampoco es el suyo.
        original = os.getcwd()
        try:
            os.chdir(tmp_path)
            datos = check_det_dataset(
                str(destino / ULTRALYTICS_DATASET_FILENAME), autodownload=False
            )
            assert Path(datos["val"]).parent == destino, (
                "Ultralytics no ha tomado como raíz el directorio del YAML "
                f"movido: {datos['val']}"
            )
            cfg = get_cfg(overrides={"imgsz": 320, "mode": "train"})
            dataset = build_yolo_dataset(cfg, datos["val"], batch=2, data=datos, mode="val")
            assert len(dataset) == len(lineas), (
                f"Tras mover el dataset, Ultralytics encontró {len(dataset)} "
                f"imágenes de {len(lineas)}. Las rutas no son portables."
            )
            assert dataset[0]["img"].shape[0] == 3
        finally:
            os.chdir(original)


class TestDeteccionDeArtefactosNoPortables:
    """Corregir el generador no arregla los artefactos ya generados."""

    def test_el_dataset_del_proyecto_es_portable(self, raiz_dataset: Path) -> None:
        """El generado por el pipeline actual pasa la comprobación."""
        from tfm_edge.data.prepare import verify_dataset_portability

        informe = verify_dataset_portability(raiz_dataset)
        assert informe["portable"], f"Problemas: {informe[problems]}"
        assert informe["splits_checked"] == 3

    def test_un_yaml_con_path_absoluto_se_detecta(self, tmp_path: Path) -> None:
        """Es la forma exacta del artefacto antiguo que falló en Colab."""
        from tfm_edge.common.exceptions import TFMEdgeError
        from tfm_edge.data.prepare import assert_dataset_portable

        (tmp_path / "images").mkdir()
        (tmp_path / ULTRALYTICS_DATASET_FILENAME).write_text(
            "path: /home/otro/data/resized_640\n"
            "train: /home/otro/data/resized_640/ultralytics_splits/train.txt\n"
            "val: /home/otro/data/resized_640/ultralytics_splits/val.txt\n"
            "test: /home/otro/data/resized_640/ultralytics_splits/test.txt\n"
            "names:\n  0: person\n",
            encoding="utf-8",
        )
        with pytest.raises(TFMEdgeError, match="NO es portable"):
            assert_dataset_portable(tmp_path)

    def test_una_lista_con_rutas_absolutas_se_detecta(self, tmp_path: Path) -> None:
        """Aunque el YAML esté bien, las listas pueden estar atadas."""
        from tfm_edge.data.prepare import verify_dataset_portability

        (tmp_path / "images").mkdir()
        (tmp_path / ULTRALYTICS_DATASET_FILENAME).write_text(
            "train: ultralytics_train.txt\n"
            "val: ultralytics_val.txt\n"
            "test: ultralytics_test.txt\n"
            "names:\n  0: person\n",
            encoding="utf-8",
        )
        for particion in PARTICIONES:
            (tmp_path / ULTRALYTICS_SPLIT_TEMPLATE.format(split=particion)).write_text(
                "/home/otro/data/resized_640/images/a.jpg\n", encoding="utf-8"
            )
        informe = verify_dataset_portability(tmp_path)
        assert not informe["portable"]
        assert any("rutas absolutas" in p for p in informe["problems"])

    def test_el_directorio_obsoleto_se_detecta(self, tmp_path: Path) -> None:
        """ultralytics_splits/ es del diseño anterior y delata un artefacto viejo."""
        from tfm_edge.data.prepare import verify_dataset_portability

        (tmp_path / "images").mkdir()
        (tmp_path / "ultralytics_splits").mkdir()
        (tmp_path / ULTRALYTICS_DATASET_FILENAME).write_text(
            "train: ultralytics_train.txt\nval: ultralytics_val.txt\n"
            "test: ultralytics_test.txt\nnames:\n  0: person\n",
            encoding="utf-8",
        )
        informe = verify_dataset_portability(tmp_path)
        assert not informe["portable"]
        assert any("ultralytics_splits" in p for p in informe["problems"])
