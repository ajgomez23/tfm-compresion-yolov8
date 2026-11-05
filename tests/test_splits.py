"""Tests del particionado estratificado y de la detección de casi duplicados."""

from __future__ import annotations

from pathlib import Path

import pytest

from tfm_edge.data.splits import (
    DEFAULT_PHASH_THRESHOLD,
    PHASH_SIZE,
    DuplicateGroup,
    SplitError,
    SplitReport,
    build_smoke_split,
    compare_with_reference_split,
    compute_dhash,
    count_instances_per_class,
    find_near_duplicates,
    hamming_distance,
    load_split_file,
    stratified_multilabel_split,
    verify_all_classes_present,
    verify_no_overlap,
    write_split_files,
    write_split_report,
)


@pytest.fixture
def etiquetas_desequilibradas() -> dict[str, set[int]]:
    """Dataset sintético con una clase frecuente y otra muy rara."""
    etiquetas: dict[str, set[int]] = {}
    for i in range(100):
        etiquetas[f"comun_{i:03d}"] = {0}
    for i in range(10):
        etiquetas[f"raro_{i:02d}"] = {0, 1}
    for i in range(4):
        etiquetas[f"rarisimo_{i}"] = {2}
    return etiquetas


class TestHammingDistance:
    """Distancia entre hashes perceptuales."""

    def test_hashes_iguales(self) -> None:
        """Dos hashes idénticos están a distancia cero."""
        assert hamming_distance(0b1010, 0b1010) == 0

    def test_cuenta_bits_distintos(self) -> None:
        """La distancia es el número de bits en que difieren."""
        assert hamming_distance(0b0000, 0b1011) == 3


class TestComputeDhash:
    """Hash perceptual por diferencias."""

    def test_hash_de_una_imagen_sintetica(self, tmp_path: Path) -> None:
        """El hash se calcula y tiene el número de bits esperado."""
        from PIL import Image

        ruta = tmp_path / "gradiente.png"
        imagen = Image.new("RGB", (64, 64))
        imagen.putdata([(x * 4 % 256, 0, 0) for y in range(64) for x in range(64)])
        imagen.save(ruta)
        valor = compute_dhash(ruta)
        assert isinstance(valor, int)
        assert 0 <= valor < 2 ** (PHASH_SIZE * PHASH_SIZE)

    def test_la_misma_imagen_reescalada_da_hash_parecido(self, tmp_path: Path) -> None:
        """Reescalar una imagen no cambia apenas su hash perceptual."""
        from PIL import Image

        original = tmp_path / "a.png"
        imagen = Image.new("RGB", (128, 128))
        imagen.putdata([((x * 2) % 256, (y * 3) % 256, 0) for y in range(128) for x in range(128)])
        imagen.save(original)

        reescalada = tmp_path / "b.png"
        imagen.resize((96, 96)).save(reescalada)

        assert hamming_distance(compute_dhash(original), compute_dhash(reescalada)) <= (
            DEFAULT_PHASH_THRESHOLD
        )


class TestFindNearDuplicates:
    """Agrupación de imágenes casi idénticas."""

    def test_sin_duplicados(self) -> None:
        """Hashes muy distintos no producen ningún grupo."""
        hashes = {"a": 0x0000000000000000, "b": 0xFFFFFFFFFFFFFFFF}
        assert find_near_duplicates(hashes, threshold=5) == []

    def test_agrupa_hashes_cercanos(self) -> None:
        """Dos hashes a distancia 1 acaban en el mismo grupo."""
        hashes = {"a": 0b1010, "b": 0b1011, "lejos": 0xFFFFFFFFFFFFFFFF}
        grupos = find_near_duplicates(hashes, threshold=5)
        assert len(grupos) == 1
        assert set(grupos[0].members) == {"a", "b"}

    def test_transitividad(self) -> None:
        """Si A se parece a B y B a C, los tres van al mismo grupo."""
        hashes = {"a": 0b000000, "b": 0b000001, "c": 0b000011}
        grupos = find_near_duplicates(hashes, threshold=1)
        assert len(grupos) == 1
        assert set(grupos[0].members) == {"a", "b", "c"}

    def test_umbral_cero_solo_agrupa_identicos(self) -> None:
        """Con umbral 0 solo se agrupan hashes exactamente iguales."""
        hashes = {"a": 0b1010, "b": 0b1010, "c": 0b1011}
        grupos = find_near_duplicates(hashes, threshold=0)
        assert len(grupos) == 1
        assert set(grupos[0].members) == {"a", "b"}

    def test_rechaza_umbral_negativo(self) -> None:
        """Un umbral negativo no tiene sentido."""
        with pytest.raises(ValueError):
            find_near_duplicates({"a": 1}, threshold=-1)

    def test_entrada_trivial(self) -> None:
        """Con menos de dos imágenes no hay nada que agrupar."""
        assert find_near_duplicates({}, threshold=5) == []
        assert find_near_duplicates({"a": 1}, threshold=5) == []


class TestStratifiedSplit:
    """Particionado estratificado multietiqueta."""

    def test_reparte_todas_las_imagenes_una_vez(
        self, etiquetas_desequilibradas: dict[str, set[int]]
    ) -> None:
        """Ninguna imagen se pierde ni se duplica."""
        particiones = stratified_multilabel_split(
            etiquetas_desequilibradas, {"train": 0.7, "val": 0.15, "test": 0.15}
        )
        repartidas = [n for v in particiones.values() for n in v]
        assert len(repartidas) == len(etiquetas_desequilibradas)
        assert set(repartidas) == set(etiquetas_desequilibradas)

    def test_sin_solapamiento(self, etiquetas_desequilibradas: dict[str, set[int]]) -> None:
        """Ninguna imagen aparece en dos particiones."""
        particiones = stratified_multilabel_split(
            etiquetas_desequilibradas, {"train": 0.7, "val": 0.15, "test": 0.15}
        )
        assert verify_no_overlap(particiones) == {}

    def test_respeta_las_proporciones_aproximadamente(
        self, etiquetas_desequilibradas: dict[str, set[int]]
    ) -> None:
        """Los tamaños se acercan a las proporciones pedidas."""
        total = len(etiquetas_desequilibradas)
        particiones = stratified_multilabel_split(
            etiquetas_desequilibradas, {"train": 0.7, "val": 0.15, "test": 0.15}
        )
        assert abs(len(particiones["train"]) / total - 0.7) < 0.12
        assert abs(len(particiones["val"]) / total - 0.15) < 0.12

    def test_la_clase_rara_llega_a_todas_las_particiones(
        self, etiquetas_desequilibradas: dict[str, set[int]]
    ) -> None:
        """La clase minoritaria no se concentra en una sola partición."""
        particiones = stratified_multilabel_split(
            etiquetas_desequilibradas, {"train": 0.5, "val": 0.25, "test": 0.25}
        )
        presencia = {
            nombre: sum(1 for i in imagenes if 2 in etiquetas_desequilibradas[i])
            for nombre, imagenes in particiones.items()
        }
        assert sum(presencia.values()) == 4
        assert sum(1 for v in presencia.values() if v > 0) >= 2

    def test_la_clase_rara_llega_a_todas_las_particiones_pese_a_coocurrir(self) -> None:
        etiquetas: dict[str, set[int]] = {}
        # 400 imágenes con solo las dos clases frecuentes.
        for i in range(400):
            etiquetas[f"frecuente_{i:03d}"] = {0, 1}
        # 20 imágenes con la clase rara, siempre acompañada de las frecuentes.
        for i in range(20):
            etiquetas[f"rara_{i:02d}"] = {0, 1, 2}

        particiones = stratified_multilabel_split(
            etiquetas, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42
        )
        presencia = {
            nombre: sum(1 for img in imagenes if 2 in etiquetas[img])
            for nombre, imagenes in particiones.items()
        }
        assert sum(presencia.values()) == 20
        for nombre, cuantas in presencia.items():
            assert cuantas > 0, (
                f"La partición '{nombre}' se ha quedado sin ninguna instancia de la "
                f"clase rara. Reparto obtenido: {presencia}"
            )

    def test_los_grupos_no_se_separan(self) -> None:
        """Todas las variantes de una escena caen en la misma partición."""
        etiquetas = {f"img_{i:03d}": {i % 3} for i in range(60)}
        grupo = DuplicateGroup(members=["img_000", "img_001", "img_002"], max_distance=2)
        particiones = stratified_multilabel_split(
            etiquetas, {"train": 0.7, "val": 0.15, "test": 0.15}, groups=[grupo], seed=1
        )
        ubicacion = {
            miembro: nombre
            for nombre, imagenes in particiones.items()
            for miembro in grupo.members
            if miembro in imagenes
        }
        assert (
            len(set(ubicacion.values())) == 1
        ), f"El grupo de casi duplicados se ha repartido: {ubicacion}"

    def test_es_reproducible(self, etiquetas_desequilibradas: dict[str, set[int]]) -> None:
        """La misma semilla produce exactamente el mismo reparto."""
        proporciones = {"train": 0.7, "val": 0.15, "test": 0.15}
        primera = stratified_multilabel_split(etiquetas_desequilibradas, proporciones, seed=123)
        segunda = stratified_multilabel_split(etiquetas_desequilibradas, proporciones, seed=123)
        assert primera == segunda

    def test_rechaza_proporciones_que_no_suman_uno(self) -> None:
        """Unas proporciones incoherentes son un error explícito."""
        with pytest.raises(SplitError, match="sumar 1.0"):
            stratified_multilabel_split({"a": {0}}, {"train": 0.5, "val": 0.2})

    def test_rechaza_dataset_vacio(self) -> None:
        """Sin imágenes no hay nada que repartir."""
        with pytest.raises(SplitError, match="No hay imágenes"):
            stratified_multilabel_split({}, {"train": 1.0})


class TestVerifyNoOverlap:
    """Comprobación de solapamiento."""

    def test_detecta_solapamiento(self) -> None:
        """Una imagen en dos particiones se reporta."""
        solapes = verify_no_overlap({"train": ["a", "b"], "val": ["b", "c"]})
        assert solapes == {"train∩val": ["b"]}

    def test_sin_solapamiento(self) -> None:
        """Particiones disjuntas no producen nada."""
        assert verify_no_overlap({"train": ["a"], "val": ["b"]}) == {}


class TestSmokeSplit:
    """Subconjunto reducido de smoke test."""

    def test_cubre_todas_las_clases(self) -> None:
        """El subconjunto contiene instancias de todas las clases."""
        etiquetas = {f"img_{i:03d}": {i % 17} for i in range(200)}
        smoke = build_smoke_split(etiquetas, num_images=40)
        cubiertas = set()
        for nombre in smoke:
            cubiertas |= etiquetas[nombre]
        assert cubiertas == set(range(17))

    def test_cubre_los_tres_rangos_de_tamano(self) -> None:
        """El subconjunto contiene objetos pequeños, medianos y grandes."""
        etiquetas = {f"img_{i:03d}": {i % 17} for i in range(200)}
        tamanos = {f"img_{i:03d}": {["small", "medium", "large"][i % 3]} for i in range(200)}
        smoke = build_smoke_split(etiquetas, tamanos, num_images=40)
        cubiertos = set()
        for nombre in smoke:
            cubiertos |= tamanos[nombre]
        assert cubiertos == {"small", "medium", "large"}

    def test_es_reproducible(self) -> None:
        """La misma semilla produce el mismo subconjunto."""
        etiquetas = {f"img_{i:03d}": {i % 17} for i in range(200)}
        assert build_smoke_split(etiquetas, num_images=30, seed=5) == build_smoke_split(
            etiquetas, num_images=30, seed=5
        )

    def test_rechaza_dataset_vacio(self) -> None:
        """Sin imágenes no se puede construir el subconjunto."""
        with pytest.raises(SplitError):
            build_smoke_split({})


class TestFicherosDeParticion:
    """Volcado y lectura de las listas de partición."""

    def test_ida_y_vuelta(self, tmp_path: Path) -> None:
        """Lo escrito se vuelve a leer igual."""
        particiones = {"train": ["b", "a"], "val": ["c"]}
        rutas = write_split_files(particiones, tmp_path)
        assert load_split_file(rutas["train"]) == ["a", "b"]
        assert load_split_file(rutas["val"]) == ["c"]

    def test_escribe_con_extension(self, tmp_path: Path) -> None:
        """Se puede volcar el nombre de fichero completo."""
        rutas = write_split_files({"train": ["a"]}, tmp_path, extension_map={"a": ".jpeg"})
        assert rutas["train"].read_text(encoding="utf-8").strip() == "a.jpeg"

    def test_fichero_inexistente(self, tmp_path: Path) -> None:
        """Leer una partición que no existe falla con un mensaje claro."""
        with pytest.raises(SplitError, match="No existe"):
            load_split_file(tmp_path / "no_existe.txt")

    def test_informe_de_particionado(self, tmp_path: Path) -> None:
        """El informe se vuelca con la política de casi duplicados explícita."""
        informe = SplitReport(
            counts={"train": 2},
            duplicate_groups=[DuplicateGroup(members=["a", "b"], max_distance=1)],
            seed=42,
        )
        ruta = write_split_report(informe, tmp_path / "split_report.json")
        contenido = ruta.read_text(encoding="utf-8")
        assert "near_duplicates" in contenido
        assert "NO se eliminan" in contenido


class TestComparacionConParticionOficial:
    """Comparación con la partición de los autores del dataset."""

    def test_calcula_jaccard(self) -> None:
        """La coincidencia se cuantifica con el índice de Jaccard."""
        propio = {"train": ["a", "b", "c"], "val": ["d"], "test": ["e"]}
        comparacion = compare_with_reference_split(propio, ["a.jpg", "b.jpg"], ["d.jpg"])
        assert comparacion["reference_sizes"] == {"train": 2, "val": 1}
        acuerdo = comparacion["agreement"]["oficial:train vs propio:train"]
        assert acuerdo["intersection"] == 2
        assert acuerdo["fraction_of_official"] == 1.0

    def test_ignora_la_extension(self) -> None:
        """La comparación funciona aunque las listas lleven extensión."""
        comparacion = compare_with_reference_split({"train": ["a"]}, ["a.jpeg"], [])
        assert comparacion["agreement"]["oficial:train vs propio:train"]["intersection"] == 1


class TestRecuentoDeInstanciasPorParticion:
    """Recuento de instancias por clase y partición."""

    @staticmethod
    def _escribir_etiquetas(directorio: Path, contenido: dict[str, list[int]]) -> Path:
        """Escribe etiquetas YOLO sintéticas."""
        directorio.mkdir(parents=True, exist_ok=True)
        for nombre, indices in contenido.items():
            lineas = [f"{i} 0.5 0.5 0.1 0.1" for i in indices]
            (directorio / f"{nombre}.txt").write_text("\n".join(lineas), encoding="utf-8")
        return directorio

    def test_cuenta_instancias_y_no_imagenes(self, tmp_path: Path) -> None:
        etiquetas = self._escribir_etiquetas(
            tmp_path / "labels", {"a": [0, 0, 0, 1], "b": [1], "c": [0, 1]}
        )
        conteos = count_instances_per_class(
            {"train": ["a", "b"], "val": ["c"]}, etiquetas, ["persona", "mano"]
        )
        assert conteos["train"] == {"persona": 3, "mano": 2}
        assert conteos["val"] == {"persona": 1, "mano": 1}

    def test_acepta_nombres_con_extension(self, tmp_path: Path) -> None:
        """Las listas de partición traen la extensión y la etiqueta no."""
        etiquetas = self._escribir_etiquetas(tmp_path / "labels", {"a": [0]})
        conteos = count_instances_per_class({"train": ["a.jpg"]}, etiquetas, ["persona"])
        assert conteos["train"]["persona"] == 1

    def test_falla_si_falta_una_etiqueta(self, tmp_path: Path) -> None:
        etiquetas = self._escribir_etiquetas(tmp_path / "labels", {"a": [0]})
        with pytest.raises(SplitError, match="etiquetas"):
            count_instances_per_class({"train": ["a", "fantasma"]}, etiquetas, ["persona"])

    def test_falla_si_el_indice_esta_fuera_de_rango(self, tmp_path: Path) -> None:
        """Un índice que no existe en el mapeo canónico es un error."""
        etiquetas = self._escribir_etiquetas(tmp_path / "labels", {"a": [7]})
        with pytest.raises(SplitError, match="fuera del rango"):
            count_instances_per_class({"train": ["a"]}, etiquetas, ["persona", "mano"])

    def test_invariante_se_cumple(self) -> None:
        """Con todas las clases en las tres particiones no salta nada."""
        conteos = {
            "train": {"persona": 10, "raro": 2},
            "val": {"persona": 3, "raro": 1},
            "test": {"persona": 3, "raro": 1},
        }
        verify_all_classes_present(conteos, ["persona", "raro"])

    def test_invariante_detecta_la_clase_ausente(self) -> None:
        conteos = {
            "train": {"persona": 10, "raro": 4},
            "val": {"persona": 3, "raro": 0},
            "test": {"persona": 3, "raro": 0},
        }
        with pytest.raises(SplitError, match="sin instancias") as excepcion:
            verify_all_classes_present(conteos, ["persona", "raro"])
        # El mensaje enumera todos los casos, no solo el primero.
        assert "raro sin instancias en val" in str(excepcion.value)
        assert "raro sin instancias en test" in str(excepcion.value)

    def test_invariante_exige_las_tres_particiones(self) -> None:
        """Un recuento sin la partición de prueba no puede darse por bueno."""
        conteos = {"train": {"persona": 10}, "val": {"persona": 3}}
        with pytest.raises(SplitError, match="test"):
            verify_all_classes_present(conteos, ["persona"])
