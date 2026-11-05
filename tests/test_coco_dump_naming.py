"""Tests del nombre y la procedencia del volcado de ``06_evaluate_coco.py``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tfm_edge.common.exceptions import InvalidPathComponentError
from tfm_edge.common.paths import get_project_paths
from tfm_edge.data.prepare import ULTRALYTICS_DATASET_FILENAME
from tfm_edge.evaluation.identity import build_baseline_evaluation_identity


@pytest.fixture(scope="module")
def script() -> ModuleType:
    """Carga ``scripts/06_evaluate_coco.py`` como módulo."""
    ruta = get_project_paths().root / "scripts" / "06_evaluate_coco.py"
    especificacion = importlib.util.spec_from_file_location("evaluate_coco_script", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


class TestQueElNombreDistingueElDispositivo:
    """Lo único que impedía que una medida borrase a la otra."""

    def test_el_dispositivo_va_en_el_nombre(self, script: ModuleType) -> None:
        nombre = script.build_dump_name("n", "val", 640, "colab_t4_gpu")
        assert nombre == "n_baseline_val_640_colab_t4_gpu.json"

    def test_dos_dispositivos_no_colisionan(self, script: ModuleType) -> None:
        cpu = script.build_dump_name("n", "val", 640, "workstation_cpu")
        t4 = script.build_dump_name("n", "val", 640, "colab_t4_gpu")
        assert cpu != t4

    def test_los_demas_ejes_siguen_distinguiendo(self, script: ModuleType) -> None:
        base = script.build_dump_name("n", "val", 640, "workstation_cpu")
        assert script.build_dump_name("s", "val", 640, "workstation_cpu") != base
        assert script.build_dump_name("n", "test", 640, "workstation_cpu") != base
        assert script.build_dump_name("n", "val", 416, "workstation_cpu") != base

    def test_un_dispositivo_con_caracteres_prohibidos_no_pasa(self, script: ModuleType) -> None:
        """El nombre se valida como componente de ruta en Linux Y en Windows."""
        with pytest.raises(InvalidPathComponentError):
            script.build_dump_name("n", "val", 640, "gpu/rara")
        with pytest.raises(InvalidPathComponentError):
            script.build_dump_name("n", "val", 640, "aux?")


class TestQueLosVolcadosAntiguosSiguenEnSuSitio:
    """Un arreglo del generador no renombra lo ya generado, y no debe."""

    def test_el_volcado_de_cpu_del_baseline_n_sigue_donde_estaba(self) -> None:
        antiguo = get_project_paths().results_raw / "coco_eval" / "n_baseline_val_640.json"
        if not antiguo.is_file():
            pytest.skip("esta copia del repositorio no trae los volcados antiguos")
        assert antiguo.stat().st_size > 0

    def test_las_filas_apuntan_a_ficheros_que_existen(self) -> None:
        """Si alguien renombrase los antiguos, esto lo detectaría."""
        import csv

        rutas = get_project_paths()
        csv_path = rutas.results_registry / "quality_metrics.csv"
        if not csv_path.is_file():
            pytest.skip("no hay registro en esta copia del repositorio")
        with csv_path.open(encoding="utf-8", newline="") as manejador:
            filas = list(csv.DictReader(manejador))

        rotas = []
        for indice, fila in enumerate(filas):
            ruta = (fila.get("predictions_path") or "").strip()
            # Solo los volcados de coco_eval: los de otras campañas viven en
            # Drive o los reapunta 14_import_colab_registry.py.
            if not ruta or "coco_eval" not in ruta:
                continue
            if not (rutas.root / ruta).is_file():
                rotas.append(f"L{indice + 2}: {ruta}")
        assert not rotas, f"predictions_path que no resuelven: {rotas}"


class TestQueLaGuardaImpidePisarUnVolcado:
    """El volcado no se sobrescribe sin pedirlo, ni siquiera con ``--dry-run``."""

    def test_la_bandera_existe_y_no_esta_activa_por_defecto(self, script: ModuleType) -> None:
        args = script.construir_parser().parse_args([])
        assert args.overwrite_dump is False

    def test_se_puede_pedir_explicitamente(self, script: ModuleType) -> None:
        args = script.construir_parser().parse_args(["--overwrite-dump"])
        assert args.overwrite_dump is True

    def test_el_dispositivo_por_defecto_sigue_siendo_el_de_la_estacion(
        self, script: ModuleType
    ) -> None:
        """Cambiar el nombre del volcado no debe cambiar dónde se mide."""
        args = script.construir_parser().parse_args([])
        assert args.device_id == "workstation_cpu"


def _identidad_de_pega() -> dict:
    """La identidad que ``volcar_detalle`` escribe dentro del volcado."""
    payload, _, experiment_id = build_baseline_evaluation_identity(
        model_size="n",
        num_classes=17,
        input_resolution=640,
        conf_threshold=0.001,
        iou_threshold=0.7,
        split="val",
        data=ULTRALYTICS_DATASET_FILENAME,
        seed=42,
    )
    return {"experiment_id": experiment_id, "identity_payload": payload}


class _ResultadoDePega:
    """Lo mínimo que volcar_detalle consume de un resultado real."""

    def __init__(self, map_50_95: float) -> None:
        self.map_50_95 = map_50_95

    def to_dict(self) -> dict:
        """Devuelve el resultado como diccionario."""
        return {"map_50_95": self.map_50_95, "criteria": {}}


class TestQueElVolcadoNoPisaNingunaMedida:
    """La guarda, ejecutada."""

    def _volcar(self, script: ModuleType, destino_dir: Path, **cambios):
        argumentos = {
            "output_dir": destino_dir,
            "size": "n",
            "split": "val",
            "resolution": 640,
            "device_id": "colab_t4_gpu",
            "environment_id": "abc123def456",
            "checkpoint": "models/baselines/n_baseline_default/best.pt",
            **_identidad_de_pega(),
        }
        argumentos.update(cambios)
        return script.volcar_detalle(_ResultadoDePega(0.3171), **argumentos)

    def test_escribe_y_deja_la_procedencia_dentro(self, script: ModuleType, tmp_path: Path) -> None:
        import json

        destino = self._volcar(script, tmp_path)
        assert destino.name == "n_baseline_val_640_colab_t4_gpu.json"
        datos = json.loads(destino.read_text(encoding="utf-8"))
        procedencia = datos["provenance"]
        assert procedencia["device_id"] == "colab_t4_gpu"
        assert procedencia["environment_id"] == "abc123def456"
        assert procedencia["evaluator"] == "pycocotools"
        assert procedencia["split"] == "val"
        assert procedencia["input_resolution"] == 640
        assert procedencia["measured_at_utc"]
        # La medida sigue entera: la procedencia se añade, no sustituye.
        assert datos["map_50_95"] == 0.3171

    def test_la_procedencia_no_lleva_rutas_absolutas(
        self, script: ModuleType, tmp_path: Path
    ) -> None:
        import json

        from tfm_edge.common.registry import is_absolute_path_text

        destino = self._volcar(script, tmp_path)
        procedencia = json.loads(destino.read_text(encoding="utf-8"))["provenance"]
        assert not is_absolute_path_text(procedencia["checkpoint"])

    def test_no_pisa_un_volcado_existente(self, script: ModuleType, tmp_path: Path) -> None:
        primero = self._volcar(script, tmp_path, device_id="workstation_cpu")
        contenido = primero.read_bytes()

        with pytest.raises(FileExistsError, match="ya existe"):
            self._volcar(script, tmp_path, device_id="workstation_cpu")

        assert primero.read_bytes() == contenido, "la medida anterior se ha modificado"

    def test_dos_dispositivos_conviven(self, script: ModuleType, tmp_path: Path) -> None:
        """Que es lo que el nombre corto impedía."""
        cpu = self._volcar(script, tmp_path, device_id="workstation_cpu")
        t4 = self._volcar(script, tmp_path, device_id="colab_t4_gpu")
        assert cpu.is_file() and t4.is_file()
        assert cpu != t4

    def test_se_puede_sobrescribir_si_se_pide(self, script: ModuleType, tmp_path: Path) -> None:
        import json

        primero = self._volcar(script, tmp_path)
        assert json.loads(primero.read_text(encoding="utf-8"))["map_50_95"] == 0.3171

        segundo = script.volcar_detalle(
            _ResultadoDePega(0.4000),
            output_dir=tmp_path,
            size="n",
            split="val",
            resolution=640,
            device_id="colab_t4_gpu",
            environment_id="abc123def456",
            checkpoint="models/baselines/n_baseline_default/best.pt",
            overwrite=True,
            **_identidad_de_pega(),
        )
        assert segundo == primero
        assert json.loads(segundo.read_text(encoding="utf-8"))["map_50_95"] == 0.4000
