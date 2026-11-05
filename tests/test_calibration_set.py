"""Tests del conjunto de calibración."""

from __future__ import annotations

import pytest

from tfm_edge.data.calibration_set import CalibrationSetError, resolve_calibration_images


class TestResolucionDeImagenes:
    """La lista guarda nombres base; el dataset, extensiones distintas."""

    def test_resuelve_por_nombre_base_con_cualquier_extension(self, tmp_path) -> None:
        """El dataset mezcla .jpg, .jpeg y .png."""
        for nombre in ("a.jpg", "b.jpeg", "c.png"):
            (tmp_path / nombre).write_bytes(b"")
        resueltas = resolve_calibration_images(["a", "b", "c"], tmp_path)
        assert [r.name for r in resueltas] == ["a.jpg", "b.jpeg", "c.png"]

    def test_se_conserva_el_orden_de_la_lista(self, tmp_path) -> None:
        """Los rangos de cuantización dependen de qué imágenes y en qué orden."""
        for nombre in ("a.jpg", "b.jpg", "c.jpg"):
            (tmp_path / nombre).write_bytes(b"")
        resueltas = resolve_calibration_images(["c", "a", "b"], tmp_path)
        assert [r.stem for r in resueltas] == ["c", "a", "b"]

    def test_una_imagen_que_falta_es_un_error(self, tmp_path) -> None:
        """Es el fallo silencioso que esta guarda existe para impedir."""
        for nombre in ("a.jpg", "b.jpg"):
            (tmp_path / nombre).write_bytes(b"")
        with pytest.raises(CalibrationSetError, match="no están en"):
            resolve_calibration_images(["a", "b", "fantasma"], tmp_path)

    @pytest.mark.parametrize(
        "lista",
        [
            pytest.param(["fantasma", "a", "b"], id="la-que-falta-va-primero"),
            pytest.param(["a", "b", "fantasma"], id="la-que-falta-va-tras-el-recorte"),
        ],
    )
    def test_el_limite_no_oculta_las_que_faltan(self, tmp_path, lista) -> None:
        """El límite recorta la lista; no puede convertir un error en un acierto."""
        for nombre in ("a.jpg", "b.jpg"):
            (tmp_path / nombre).write_bytes(b"")
        with pytest.raises(CalibrationSetError, match="no están en"):
            resolve_calibration_images(lista, tmp_path, limit=2)

    def test_el_limite_recorta(self, tmp_path) -> None:
        """Cuando no falta nada, el límite hace su trabajo."""
        for nombre in ("a.jpg", "b.jpg", "c.jpg"):
            (tmp_path / nombre).write_bytes(b"")
        assert len(resolve_calibration_images(["a", "b", "c"], tmp_path, limit=2)) == 2

    def test_menos_de_dos_imagenes_se_rechaza(self, tmp_path) -> None:
        """Con una sola, los rangos dependerían de esa imagen concreta."""
        (tmp_path / "a.jpg").write_bytes(b"")
        with pytest.raises(CalibrationSetError, match="al menos 2"):
            resolve_calibration_images(["a"], tmp_path)

    def test_un_directorio_inexistente_se_denuncia(self, tmp_path) -> None:
        """Antes de fallar imagen a imagen, se dice lo obvio."""
        with pytest.raises(CalibrationSetError, match="No existe el directorio"):
            resolve_calibration_images(["a"], tmp_path / "no_existe")
