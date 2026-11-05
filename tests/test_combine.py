"""Lo que decide ``scripts/22_combine.py`` antes de cuantizar nada."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.common.paths import build_experiment_id, get_project_paths


def _cargar_script() -> ModuleType:
    """Carga ``scripts/22_combine.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "22_combine.py"
    especificacion = importlib.util.spec_from_file_location("combine_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def _fila_poda(tmp_path: Path, **cambios: str) -> dict[str, str]:
    """Construye una fila de poda admisible, con los pesos de verdad en disco."""
    pesos = tmp_path / "best.pt"
    pesos.write_bytes(b"no son pesos de verdad, pero existen")
    fila = {
        "experiment_id": "s_pruning_l1global_r50_9419b26f",
        "technique": "pruning",
        "technique_variant": "l1global_r50",
        "config_hash": "abcdef0123456789",
        "compression_ratio": "0.5",
        "effective_params": "5580000",
        "effective_params_ratio": "0.32479",
        "training_epochs": "30",
        "weights_path": str(pesos),
        "status": "ok",
    }
    fila.update(cambios)
    return fila


class TestSeleccionDePadres:
    """De qué se puede partir y de qué no."""

    def test_un_podado_reajustado_entra(self, tmp_path: Path) -> None:
        script = _cargar_script()
        admitidos, motivos = script.seleccionar_padres(
            [_fila_poda(tmp_path)], ["s_pruning_l1global_r50_9419b26f"], False
        )
        assert len(admitidos) == 1
        assert motivos == []

    def test_un_podado_sin_reajustar_se_rechaza(self, tmp_path: Path) -> None:
        """El rechazo que justifica todo este fichero."""
        script = _cargar_script()
        admitidos, motivos = script.seleccionar_padres(
            [_fila_poda(tmp_path, training_epochs="")],
            ["s_pruning_l1global_r50_9419b26f"],
            False,
        )
        assert admitidos == []
        assert "SIN REAJUSTAR" in motivos[0]

    def test_los_sin_reajustar_tampoco_entran_por_all_pruned(self, tmp_path: Path) -> None:
        """La selección automática filtra por lo mismo que la explícita."""
        script = _cargar_script()
        otro = tmp_path / "otro"
        otro.mkdir()
        filas = [
            _fila_poda(tmp_path),
            _fila_poda(otro, experiment_id="s_pruning_l1global_r50_8db60bc4", training_epochs=""),
        ]
        admitidos, motivos = script.seleccionar_padres(filas, [], True)
        assert [f["experiment_id"] for f in admitidos] == ["s_pruning_l1global_r50_9419b26f"]
        # Y el descarte se explica: un filtro silencioso daria la misma lista y un
        # log en el que las filas que no entran no se distinguen de las que no
        # existen.
        assert len(motivos) == 1
        assert "SIN REAJUSTAR" in motivos[0]

    def test_una_fila_que_no_es_de_poda_se_rechaza(self, tmp_path: Path) -> None:
        """La rejilla parte de poda; un baseline no es una partida válida."""
        script = _cargar_script()
        admitidos, motivos = script.seleccionar_padres(
            [_fila_poda(tmp_path, technique="baseline")],
            ["s_pruning_l1global_r50_9419b26f"],
            False,
        )
        assert admitidos == []
        assert "pruning" in motivos[0]

    def test_un_padre_que_no_esta_en_el_registro_se_rechaza(self) -> None:
        """Sin fila de padre no hay linaje que inscribir."""
        script = _cargar_script()
        admitidos, motivos = script.seleccionar_padres([], ["s_pruning_inventado_00000000"], False)
        assert admitidos == []
        assert "no está en el registro" in motivos[0]

    def test_unos_pesos_que_no_estan_se_rechazan(self, tmp_path: Path) -> None:
        """La fila puede decir lo que quiera; el fichero tiene que estar."""
        script = _cargar_script()
        admitidos, motivos = script.seleccionar_padres(
            [_fila_poda(tmp_path, weights_path=str(tmp_path / "no_existe.pt"))],
            ["s_pruning_l1global_r50_9419b26f"],
            False,
        )
        assert admitidos == []
        assert "no hay pesos" in motivos[0]


class TestVariante:
    """La etiqueta que acaba en el ``experiment_id``."""

    @pytest.mark.parametrize(
        ("padre", "esperada"),
        [
            ("l1global_r50", "prune-l1global-r50-ptqint8"),
            ("l1global_r25", "prune-l1global-r25-ptqint8"),
            ("l1global_r50_iter4", "prune-l1global-r50-iter4-ptqint8"),
            ("l1global_r50_randomctl", "prune-l1global-r50-randomctl-ptqint8"),
            ("bnscale_r50", "prune-bnscale-r50-ptqint8"),
        ],
    )
    def test_la_variante_sale_de_la_del_padre(self, padre: str, esperada: str) -> None:
        """Cada poda del registro produce una etiqueta distinta y legible."""
        assert _cargar_script().construir_variante(padre) == esperada

    def test_las_cinco_variantes_dan_identificadores_validos(self) -> None:
        """Y el identificador resultante es un nombre de fichero portable."""
        script = _cargar_script()
        for padre in (
            "l1global_r50",
            "l1global_r25",
            "l1global_r50_iter4",
            "l1global_r50_randomctl",
            "bnscale_r50",
        ):
            identificador = build_experiment_id(
                model_size="s",
                technique="combined",
                variant=script.construir_variante(padre),
                config_hash="0123456789abcdef",
            )
            assert identificador.startswith("s_combined_prune-")


class TestNota:
    """Lo que la fila dice, y que no se puede deducir del artefacto."""

    def test_la_nota_declara_el_reajuste_sin_que_nadie_lo_teclee(self, tmp_path: Path) -> None:
        """El dato sale del registro, que es donde está."""
        script = _cargar_script()
        nota = script.construir_nota(_fila_poda(tmp_path))
        assert "REAJUSTADO 30 épocas" in nota
        assert "s_pruning_l1global_r50_9419b26f" in nota

    def test_la_nota_lleva_la_reduccion_medida_del_padre(self, tmp_path: Path) -> None:
        """67,5 % es lo medido para r50; no se redondea a lo pedido (50 %)."""
        nota = _cargar_script().construir_nota(_fila_poda(tmp_path))
        assert "67.5%" in nota

    def test_sin_reduccion_en_la_fila_la_nota_lo_dice(self, tmp_path: Path) -> None:
        """Una celda vacía no se convierte en un porcentaje inventado."""
        nota = _cargar_script().construir_nota(_fila_poda(tmp_path, effective_params_ratio=""))
        assert "no consta" in nota

    def test_la_nota_dice_contra_que_hay_que_comparar(self, tmp_path: Path) -> None:
        """Sin eso, la cifra se leería contra el baseline y mezclaría técnicas."""
        nota = _cargar_script().construir_nota(_fila_poda(tmp_path))
        assert "FP32" in nota and "baseline" in nota

    def test_el_texto_del_usuario_se_anade_sin_sustituir_lo_derivado(self, tmp_path: Path) -> None:
        """``--notes`` amplía la nota; no puede vaciar la parte que se deriva."""
        nota = _cargar_script().construir_nota(_fila_poda(tmp_path), "Repetición de agosto.")
        assert "REAJUSTADO 30 épocas" in nota
        assert nota.endswith("Repetición de agosto.")


class TestGuardaDeArtefactos:
    """Los artefactos existentes no se sobrescriben sin pedirlo."""

    def test_un_directorio_con_algo_dentro_cuenta_como_evidencia(self, tmp_path: Path) -> None:
        """Basta un fichero para que la partida se considere ya materializada."""
        script = _cargar_script()
        (tmp_path / "algo.onnx").write_bytes(b"x")
        assert script.hay_artefactos(tmp_path) is True

    def test_un_directorio_vacio_o_ausente_no_bloquea(self, tmp_path: Path) -> None:
        """Un directorio recién creado no es evidencia de nada."""
        script = _cargar_script()
        vacio = tmp_path / "vacio"
        vacio.mkdir()
        assert script.hay_artefactos(vacio) is False
        assert script.hay_artefactos(tmp_path / "no_existe") is False
