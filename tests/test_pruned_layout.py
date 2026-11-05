"""Tests de la colocación de checkpoints podados (scripts/15)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tfm_edge.common.paths import get_project_paths

EXPERIMENTO = "s_pruning_l1global_r50_9419b26f"
COLAB = "/content/drive/MyDrive/tfm-edge/pruned"


def cargar_script():
    """Carga scripts/15_normalize_pruned_layout.py como módulo."""
    import sys

    ruta = get_project_paths().root / "scripts" / "15_normalize_pruned_layout.py"
    especificacion = importlib.util.spec_from_file_location("layout_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules["layout_script"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


@pytest.fixture(scope="module")
def script():
    """El módulo del script, cargado una vez."""
    return cargar_script()


@pytest.fixture(autouse=True)
def preservar_logging():
    """Deja el logging como estaba tras cada test."""
    import logging

    from tfm_edge.common.logging_utils import LOGGER_NAME

    afectados = [logging.getLogger(), logging.getLogger(LOGGER_NAME)]
    estado = [(lg, list(lg.handlers), lg.level, lg.propagate) for lg in afectados]
    try:
        yield
    finally:
        for logger, manejadores, nivel, propaga in estado:
            logger.handlers = manejadores
            logger.setLevel(nivel)
            logger.propagate = propaga


def escribir_volcado(raw_dir: Path, experiment_id: str, con_reajuste: bool = True) -> Path:
    """Escribe un volcado crudo con las rutas que declararía el pipeline."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = f"{COLAB}/{experiment_id}/finetune"
    volcado = {
        "experiment_id": experiment_id,
        "finetune": {
            "checkpoint_before": f"{base}/pruned_before_finetune.pt",
            "checkpoint_after": f"{base}/train/weights/best.pt" if con_reajuste else "",
        },
    }
    destino = raw_dir / f"{experiment_id}.json"
    destino.write_text(json.dumps(volcado, indent=2), encoding="utf-8")
    return destino


def escribir_pesos(ruta: Path, contenido: bytes) -> None:
    """Crea un fichero de pesos simulado."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(contenido)


@pytest.fixture
def entorno(tmp_path):
    """Monta un árbol de trabajo vacío."""
    podados = tmp_path / "models" / "pruned"
    crudos = tmp_path / "results" / "raw" / "pruning"
    podados.mkdir(parents=True)
    crudos.mkdir(parents=True)
    return podados, crudos


def ejecutar(script, podados: Path, crudos: Path, *extra: str) -> int:
    """Lanza el script sobre un árbol de prueba."""
    return script.main(["--pruned-root", str(podados), "--raw-dir", str(crudos), *extra])


class TestLosDosErroresRealesDeCopia:
    """Regresión de las dos estructuras que llegaron mal de Drive."""

    def test_el_best_aplanado_vuelve_a_su_sitio(self, script, entorno) -> None:
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        mal = podados / EXPERIMENTO / "finetune" / "best.pt"
        escribir_pesos(mal, b"pesos del reajuste")
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt", b"podado")

        assert ejecutar(script, podados, crudos, "--apply") == 0

        bien = podados / EXPERIMENTO / "finetune" / "train" / "weights" / "best.pt"
        assert bien.is_file()
        assert not mal.exists()
        assert bien.read_bytes() == b"pesos del reajuste"

    def test_el_podado_dentro_de_train_sube_un_nivel(self, script, entorno) -> None:
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        pesos = podados / EXPERIMENTO / "finetune" / "train" / "weights"
        mal = pesos / "pruned_before_finetune.pt"
        escribir_pesos(mal, b"podado sin reajustar")
        escribir_pesos(pesos / "best.pt", b"pesos del reajuste")

        assert ejecutar(script, podados, crudos, "--apply") == 0

        bien = podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt"
        assert bien.is_file()
        assert not mal.exists()
        assert bien.read_bytes() == b"podado sin reajustar"


class TestGuardas:
    """Lo que el script no debe hacer."""

    def test_sin_apply_no_mueve_nada(self, script, entorno) -> None:
        """El plan se enseña primero; mover exige pedirlo."""
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        mal = podados / EXPERIMENTO / "finetune" / "best.pt"
        escribir_pesos(mal, b"pesos")
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt", b"podado")

        ejecutar(script, podados, crudos)

        assert mal.is_file()
        assert not (podados / EXPERIMENTO / "finetune" / "train").exists()

    def test_no_pisa_un_destino_que_ya_existe(self, script, entorno) -> None:
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        destino = podados / EXPERIMENTO / "finetune" / "train" / "weights" / "best.pt"
        escribir_pesos(destino, b"el bueno")
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "best.pt", b"otro distinto")
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt", b"podado")

        ejecutar(script, podados, crudos, "--apply")

        assert destino.read_bytes() == b"el bueno"

    def test_un_nombre_ambiguo_no_se_mueve(self, script, entorno) -> None:
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        uno = podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt"
        otro = podados / EXPERIMENTO / "copia" / "pruned_before_finetune.pt"
        # El destino canónico del 'before' es justo 'uno', así que para forzar la
        # ambigüedad se prueba sobre el 'after', que sí está desplazado.
        escribir_pesos(uno, b"a")
        escribir_pesos(otro, b"b")
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "best.pt", b"c")
        escribir_pesos(podados / EXPERIMENTO / "copia" / "best.pt", b"d")

        assert ejecutar(script, podados, crudos, "--apply") == 2

        assert (podados / EXPERIMENTO / "finetune" / "best.pt").is_file()
        assert (podados / EXPERIMENTO / "copia" / "best.pt").is_file()

    def test_un_checkpoint_que_falta_se_reporta(self, script, entorno) -> None:
        """Es el caso de r50, al que no le vino el podado-sin-reajustar."""
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        escribir_pesos(
            podados / EXPERIMENTO / "finetune" / "train" / "weights" / "best.pt", b"pesos"
        )

        assert ejecutar(script, podados, crudos, "--apply") == 2

    def test_sin_volcado_no_se_toca_el_experimento(self, script, entorno) -> None:
        podados, crudos = entorno
        mal = podados / EXPERIMENTO / "finetune" / "best.pt"
        escribir_pesos(mal, b"pesos")

        assert ejecutar(script, podados, crudos, "--apply") == 2

        assert mal.is_file()

    def test_una_campana_sin_reajuste_no_es_una_incidencia(self, script, entorno) -> None:
        """'checkpoint_after' vacío significa que no se reajustó, y es legítimo."""
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO, con_reajuste=False)
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt", b"podado")

        assert ejecutar(script, podados, crudos, "--apply") == 0


class TestLaConvencionSaleDelVolcado:
    """El script no lleva dentro ninguna estructura preferida."""

    def test_sigue_al_volcado_si_declara_otra_estructura(self, script, entorno) -> None:
        """Si el pipeline cambiara dónde escribe, el script lo seguiría."""
        podados, crudos = entorno
        crudos.mkdir(parents=True, exist_ok=True)
        (crudos / f"{EXPERIMENTO}.json").write_text(
            json.dumps(
                {
                    "experiment_id": EXPERIMENTO,
                    "finetune": {
                        "checkpoint_before": f"{COLAB}/{EXPERIMENTO}/finetune/podado.pt",
                        "checkpoint_after": f"{COLAB}/{EXPERIMENTO}/otra/ruta/best.pt",
                    },
                }
            ),
            encoding="utf-8",
        )
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "podado.pt", b"podado")
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "best.pt", b"pesos")

        assert ejecutar(script, podados, crudos, "--apply") == 0

        assert (podados / EXPERIMENTO / "otra" / "ruta" / "best.pt").read_bytes() == b"pesos"


class TestIntegridad:
    """El contenido tiene que llegar idéntico."""

    def test_el_contenido_se_conserva_byte_a_byte(self, script, entorno) -> None:
        """Se verifica con SHA-256 antes y después de mover."""
        podados, crudos = entorno
        escribir_volcado(crudos, EXPERIMENTO)
        contenido = bytes(range(256)) * 1000
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "best.pt", contenido)
        escribir_pesos(podados / EXPERIMENTO / "finetune" / "pruned_before_finetune.pt", b"podado")

        assert ejecutar(script, podados, crudos, "--apply") == 0

        destino = podados / EXPERIMENTO / "finetune" / "train" / "weights" / "best.pt"
        assert destino.read_bytes() == contenido

    def test_el_resumen_es_el_del_fichero(self, script, tmp_path) -> None:
        """sha256() no puede depender del tamaño de bloque de lectura."""
        import hashlib

        fichero = tmp_path / "x.pt"
        contenido = bytes(range(256)) * 5000
        fichero.write_bytes(contenido)
        assert script.sha256(fichero) == hashlib.sha256(contenido).hexdigest()
        assert script.sha256(fichero, bloque=7) == hashlib.sha256(contenido).hexdigest()
