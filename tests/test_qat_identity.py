"""Tests de la identidad de un afinado QAT: el punto de partida entra en el hash."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.common.config import load_config
from tfm_edge.common.paths import get_project_paths
from tfm_edge.data.download import compute_sha256

RAIZ = get_project_paths().root
CONFIG_QAT = RAIZ / "configs" / "compression" / "quantization" / "qat_default.yaml"


def _cargar_16_qat() -> ModuleType:
    """Carga ``scripts/16_qat.py`` como módulo nuevo, sin ejecutar su ``main``."""
    ruta = RAIZ / "scripts" / "16_qat.py"
    especificacion = importlib.util.spec_from_file_location("qat_script_identidad", ruta)
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


def _identificador_de_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    checkpoint: Path,
    padre: str = "",
    extra: tuple[str, ...] = (),
) -> str:
    """Ejecuta el ``main`` real de ``16_qat.py`` y devuelve el identificador que calcula."""
    script = _cargar_16_qat()
    anotados: list[str] = []

    def _fila_previa_que_anota(registro: object, experiment_id: str) -> dict[str, str]:
        anotados.append(experiment_id)
        return {"experiment_id": experiment_id, "status": "ok"}

    monkeypatch.setattr(script, "_fila_previa_del_experimento", _fila_previa_que_anota)
    # setup_logging reconfigura el logging raíz y deja un fichero en results/logs:
    # ninguna de las dos cosas debe pasar dentro de la batería de tests.
    monkeypatch.setattr(script, "setup_logging", lambda *args, **kwargs: None)

    argumentos = [
        "16_qat.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(CONFIG_QAT),
        "--output-dir",
        str(tmp_path / "salida"),
        "--raw-dir",
        str(tmp_path / "crudo"),
        "--registry-dir",
        str(tmp_path / "registro"),
        "--no-evaluate",
        "--allow-cpu-finetune",
        "--allow-mosaic-free",
        "--on-duplicate",
        "fail",
        *extra,
    ]
    if padre:
        argumentos += ["--parent-experiment-id", padre]
    monkeypatch.setattr(sys, "argv", argumentos)

    codigo = script.main()

    assert codigo == 1, "main debía parar en la guarda de duplicados, antes de cargar el modelo"
    assert len(anotados) == 1, "main no llegó a buscar la fila previa con su identificador"
    return anotados[0]


class TestElPuntoDePartidaEntraEnLaIdentidad:
    """Dos afinados que parten de pesos distintos no pueden compartir identificador."""

    def test_dos_checkpoints_distintos_no_comparten_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        baseline = _checkpoint(tmp_path / "baseline", b"pesos del baseline")
        podado = _checkpoint(tmp_path / "podado", b"pesos del modelo podado")

        de_baseline = _identificador_de_main(monkeypatch, tmp_path / "a", baseline, padre="p")
        de_podado = _identificador_de_main(monkeypatch, tmp_path / "b", podado, padre="p")

        assert de_baseline != de_podado

    def test_el_padre_declarado_entra_en_la_identidad(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pesos = _checkpoint(tmp_path / "pesos", b"los mismos pesos")

        con_a = _identificador_de_main(monkeypatch, tmp_path / "a", pesos, padre="padre_a")
        con_b = _identificador_de_main(monkeypatch, tmp_path / "b", pesos, padre="padre_b")

        assert con_a != con_b

    def test_el_mismo_contenido_en_otra_ruta_da_el_mismo_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        en_colab = _checkpoint(tmp_path / "content" / "tfm-edge", b"mismo fichero")
        en_local = _checkpoint(tmp_path / "models" / "baselines", b"mismo fichero")

        desde_colab = _identificador_de_main(monkeypatch, tmp_path / "a", en_colab, padre="p")
        desde_local = _identificador_de_main(monkeypatch, tmp_path / "b", en_local, padre="p")

        assert desde_colab == desde_local

    def test_la_configuracion_sigue_entrando(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Añadir el punto de partida no puede sacar a la configuración del hash."""
        pesos = _checkpoint(tmp_path / "pesos", b"los mismos pesos")

        con_42 = _identificador_de_main(monkeypatch, tmp_path / "a", pesos, extra=("--seed", "42"))
        con_43 = _identificador_de_main(monkeypatch, tmp_path / "b", pesos, extra=("--seed", "43"))

        assert con_42 != con_43


class TestLaIdentidadSeRecalculaFueraDeMain:
    """El identificador de ``main`` se puede reconstruir con la función pública."""

    def test_construir_identidad_da_lo_mismo_que_main(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Si ``main`` volviera a construir el payload por su cuenta, esto fallaría."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos cualesquiera")
        de_main = _identificador_de_main(monkeypatch, tmp_path / "a", pesos, padre="padre_x")

        script = _cargar_16_qat()
        payload, config_hash, experiment_id = script.construir_identidad(
            load_config(CONFIG_QAT), compute_sha256(pesos), "padre_x"
        )

        assert experiment_id == de_main
        assert experiment_id.endswith(config_hash[:8])
        assert payload["starting_checkpoint_sha256"] == compute_sha256(pesos)
        assert payload["parent_experiment_id"] == "padre_x"
        assert payload["seeding_scheme"] == script.SEEDING_SCHEME
        assert payload["warmup_epochs"] == script.WARMUP_EPOCHS_QAT

    def test_nada_de_esto_escribe_en_el_repositorio(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pesos = _checkpoint(tmp_path / "pesos", b"pesos cualesquiera")
        crudo_qat = RAIZ / "results" / "raw" / "qat"
        antes = sorted(p.name for p in crudo_qat.iterdir()) if crudo_qat.is_dir() else []

        _identificador_de_main(monkeypatch, tmp_path / "a", pesos, padre="p")

        despues = sorted(p.name for p in crudo_qat.iterdir()) if crudo_qat.is_dir() else []
        assert antes == despues
        assert not any((tmp_path / "a" / "crudo").iterdir())


class TestElLoteEntraEnLaIdentidad:
    """Otro lote es otra receta, y el identificador tiene que decirlo."""

    def test_otro_lote_es_otro_experimento(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Antes, --batch-size 8 daba el mismo identificador con otra receta."""
        pesos = _checkpoint(tmp_path / "pesos", b"pesos cualesquiera")

        por_defecto = _identificador_de_main(monkeypatch, tmp_path / "a", pesos, padre="p")
        con_8 = _identificador_de_main(
            monkeypatch, tmp_path / "b", pesos, padre="p", extra=("--batch-size", "8")
        )

        assert con_8 != por_defecto

    def test_el_lote_del_yaml_explicito_no_mueve_el_identificador(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pesos = _checkpoint(tmp_path / "pesos", b"pesos cualesquiera")
        del_yaml = load_config(CONFIG_QAT).training.batch_size

        por_defecto = _identificador_de_main(monkeypatch, tmp_path / "a", pesos, padre="p")
        explicito = _identificador_de_main(
            monkeypatch, tmp_path / "b", pesos, padre="p", extra=("--batch-size", str(del_yaml))
        )

        assert explicito == por_defecto
