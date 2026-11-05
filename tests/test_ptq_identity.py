"""Tests de la identidad y del destino de ``10_quantize_ptq.py``."""

from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tfm_edge.common.paths import get_project_paths
from tfm_edge.export.quantize import Int8Config

RAIZ = get_project_paths().root


class _Centinela(BaseException):
    """Detiene ``main`` al entrar en la materialización, sin pasar por sus ``except``."""


class _DetectorDoble:
    """Lo único que ``main`` consulta del detector antes de materializar."""

    size = "s"

    def assert_float32(self) -> None:
        """El doble siempre está en FP32."""


def _cargar_10_ptq() -> ModuleType:
    """Carga ``scripts/10_quantize_ptq.py`` como módulo nuevo, sin ejecutar su ``main``."""
    ruta = RAIZ / "scripts" / "10_quantize_ptq.py"
    especificacion = importlib.util.spec_from_file_location("ptq_script_identidad", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _checkpoint(directorio: Path, contenido: bytes) -> Path:
    """Escribe un fichero que hace de checkpoint de partida."""
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / "best.pt"
    ruta.write_bytes(contenido)
    return ruta


def _calibracion(directorio: Path) -> tuple[Path, Path]:
    """Crea una lista de calibración de dos imágenes y sus ficheros."""
    imagenes = directorio / "images"
    imagenes.mkdir(parents=True, exist_ok=True)
    for stem in ("img_a", "img_b"):
        (imagenes / f"{stem}.jpg").write_bytes(b"")
    lista = directorio / "calibration_test.txt"
    lista.write_text("img_a\nimg_b\n", encoding="utf-8")
    return lista, imagenes


def _ejecutar_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    *extra: str,
    padre: str = "",
) -> dict[str, Any]:
    """Ejecuta el ``main`` real de ``10_quantize_ptq.py`` hasta la materialización."""
    script = _cargar_10_ptq()
    anotado: dict[str, Any] = {}

    def _materializar_que_anota(detector: Any, destino: Path, nombre: str, **_: Any) -> None:
        anotado.update(destino=Path(destino), nombre=nombre)
        raise _Centinela()

    lista, imagenes = _calibracion(tmp_path / "calibracion")
    rutas = SimpleNamespace(
        results=tmp_path / "results",
        data=tmp_path / "data",
        data_calibration=tmp_path / "data" / "calibration",
    )
    monkeypatch.setattr(script, "materialize_int8_branch", _materializar_que_anota)
    monkeypatch.setattr(script.Detector, "load", staticmethod(lambda *a, **k: _DetectorDoble()))
    monkeypatch.setattr(script, "get_project_paths", lambda: rutas)
    monkeypatch.setattr(script, "setup_logging", lambda *args, **kwargs: None)

    argumentos = [
        "10_quantize_ptq.py",
        "--checkpoint",
        str(checkpoint),
        "--calibration-set",
        str(lista),
        "--images-dir",
        str(imagenes),
        "--registry-dir",
        str(tmp_path / "registro"),
        "--dry-run",
        *extra,
    ]
    if padre:
        argumentos += ["--parent-experiment-id", padre]
    monkeypatch.setattr(sys, "argv", argumentos)

    with pytest.raises(_Centinela):
        script.main()
    return anotado


class TestElPuntoDePartidaEntraEnLaIdentidad:
    """Dos PTQ de pesos distintos no comparten identificador ni destino."""

    def test_dos_checkpoints_del_mismo_directorio_no_comparten_destino(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        uno = _checkpoint(tmp_path / "uno" / "weights", b"pesos del uno")
        otro = _checkpoint(tmp_path / "otro" / "weights", b"pesos del otro")

        de_uno = _ejecutar_main(monkeypatch, tmp_path / "a", uno, padre="p")
        de_otro = _ejecutar_main(monkeypatch, tmp_path / "a", otro, padre="p")

        assert de_uno["nombre"] != de_otro["nombre"]
        assert de_uno["destino"] != de_otro["destino"]

    def test_el_destino_por_defecto_se_nombra_por_el_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pesos = _checkpoint(tmp_path / "s_baseline_default", b"pesos")

        anotado = _ejecutar_main(monkeypatch, tmp_path / "a", pesos)

        assert re.fullmatch(r"s_ptq_int8-noexcl_[0-9a-f]{8}", anotado["nombre"])
        assert anotado["destino"] == tmp_path / "a" / "results" / "exports" / anotado["nombre"]

    def test_el_padre_declarado_entra_en_la_identidad(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pesos = _checkpoint(tmp_path / "pesos", b"los mismos pesos")

        con_a = _ejecutar_main(monkeypatch, tmp_path / "a", pesos, padre="padre_a")
        con_b = _ejecutar_main(monkeypatch, tmp_path / "b", pesos, padre="padre_b")

        assert con_a["nombre"] != con_b["nombre"]

    def test_el_mismo_contenido_en_otra_ruta_da_el_mismo_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        en_colab = _checkpoint(tmp_path / "content" / "tfm-edge", b"mismo fichero")
        en_local = _checkpoint(tmp_path / "models" / "baselines", b"mismo fichero")

        desde_colab = _ejecutar_main(monkeypatch, tmp_path / "a", en_colab, padre="p")
        desde_local = _ejecutar_main(monkeypatch, tmp_path / "b", en_local, padre="p")

        assert desde_colab["nombre"] == desde_local["nombre"]

    def test_nada_de_esto_escribe_en_el_repositorio(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        exportaciones = RAIZ / "results" / "exports"
        antes = sorted(p.name for p in exportaciones.iterdir()) if exportaciones.is_dir() else []

        _ejecutar_main(monkeypatch, tmp_path / "a", _checkpoint(tmp_path / "w", b"pesos"))

        despues = sorted(p.name for p in exportaciones.iterdir()) if exportaciones.is_dir() else []
        assert antes == despues


class TestLoInscritoSoloCambiaPorElPuntoDePartida:
    """Sin las dos claves del punto de partida, el payload reproduce los cuatro PTQ inscritos."""

    #: Tamaño y ``config_hash`` de los cuatro PTQ inscritos que se reproducían.
    INSCRITOS = {"s": "5bcd71f0", "n": "a78f566a", "m": "b1a6e9e7", "l": "4a52a7e5"}

    def test_sin_el_punto_de_partida_salen_los_cuatro_inscritos(self) -> None:
        script = _cargar_10_ptq()
        with (RAIZ / "results" / "registry" / "experiments.csv").open(encoding="utf-8") as f:
            filas = {fila["experiment_id"]: fila for fila in csv.DictReader(f)}

        for talla, esperado in self.INSCRITOS.items():
            fila = filas[f"{talla}_ptq_int8-noexcl_{esperado}"]
            huella_calibracion = re.search(r"hash ([0-9a-f]{16})", fila["notes"]).group(1)
            payload, _, experiment_id = script.construir_identidad(
                model_size=talla,
                variante=script.DEFAULT_VARIANT,
                resolucion=640,
                configuracion=Int8Config(calibration_size=128, excluded_nodes=(), seed=42),
                nombre_calibracion="calibration_128.txt",
                huella_calibracion=huella_calibracion,
                runtimes=["onnxruntime", "openvino"],
                seed=42,
                huella_checkpoint="cualquiera",
                parent_experiment_id=fila["parent_experiment_id"],
            )
            del payload["starting_checkpoint_sha256"], payload["parent_experiment_id"]

            assert script.compute_config_hash(payload)[:8] == esperado
            assert not experiment_id.endswith(esperado)
