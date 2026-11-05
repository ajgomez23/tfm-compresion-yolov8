"""Los bloques que escriben la evidencia cruda de cada técnica."""

from __future__ import annotations

import ast
import csv
import importlib.util
import inspect
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tfm_edge.common.config import ProjectConfig, compute_config_hash, load_class_names, load_config
from tfm_edge.common.paths import relativize_dump_paths
from tfm_edge.data.prepare import ULTRALYTICS_DATASET_FILENAME
from tfm_edge.evaluation.identity import build_baseline_evaluation_identity
from tfm_edge.export.quantize import Int8Config, calibration_hash

RAIZ = Path(__file__).resolve().parent.parent
CRUDOS = RAIZ / "results" / "raw"

#: Volcado real más reciente de cada familia, que es el que escribe el código actual.
VOLCADOS_DE_REFERENCIA = {
    "pruning": CRUDOS / "pruning" / "s_pruning_l1global_r50_iter4_d8fe066f.json",
    "qat": CRUDOS / "qat" / "s_qat_int8-noexcl_4d491880.json",
    "distillation": (
        CRUDOS / "distillation" / "epochs50" / "n_distillation_l2n_combined_t2_ac9ff009.json"
    ),
}


#: Huella de un checkpoint que no existe. Sirve igual: lo que se comprueba es
#: que la que entró en el hash sea la que queda escrita, no cuál era.
HUELLA_DE_PEGA = "0" * 64

#: La del maestro, distinta de la del alumno: en destilación entran las dos.
HUELLA_DEL_MAESTRO = "1" * 64


def _comprobar_que_el_payload_reproduce_el_identificador(ruta: Path) -> dict[str, Any]:
    """Recalcula el hash del payload volcado y lo compara con el identificador."""
    contenido = json.loads(ruta.read_text(encoding="utf-8"))
    payload = contenido["experiment_identity_payload"]
    sufijo = contenido["experiment_id"].rsplit("_", 1)[-1]

    assert compute_config_hash(payload)[: len(sufijo)] == sufijo, (
        "el payload volcado no reproduce el sufijo del experiment_id: lo que se "
        "guarda no es lo que se hashea"
    )
    return payload


def _cargar(nombre: str) -> ModuleType:
    """Carga un script de ``scripts/`` como módulo."""
    ruta = RAIZ / "scripts" / f"{nombre}.py"
    especificacion = importlib.util.spec_from_file_location(f"volcado_{nombre}", ruta)
    assert especificacion is not None and especificacion.loader is not None
    modulo = importlib.util.module_from_spec(especificacion)
    sys.modules[f"volcado_{nombre}"] = modulo
    especificacion.loader.exec_module(modulo)
    return modulo


def _claves_reales(familia: str) -> set[str]:
    """Claves de primer nivel del volcado real de esa familia."""
    ruta = VOLCADOS_DE_REFERENCIA[familia]
    if not ruta.is_file():
        pytest.skip(f"No está el volcado de referencia {ruta.name}.")
    return set(json.loads(ruta.read_text(encoding="utf-8")))


#: Volcado real de una evaluación COCO, el único de los cinco que lleva ``provenance``.
VOLCADO_REAL_DE_COCO = CRUDOS / "coco_eval" / "n_baseline_val_640_colab_t4_gpu.json"

#: Informe real de una rama combinada. Los seis de la campaña tienen el mismo
#: conjunto de claves, así que cualquiera sirve de referencia.
INFORME_REAL_DE_COMBINADA = (
    RAIZ
    / "results"
    / "exports"
    / "combined"
    / "s_combined_prune-l1global-r50-ptqint8_9067dba2"
    / "s_combined_prune-l1global-r50-ptqint8_9067dba2_combined.json"
)

#: Lo que el bloque de volcado añade por su cuenta en cada caso. El resto sale
#: del resultado de la medida, que aquí entra como doble.
ANADE_EL_BLOQUE = {
    "coco": {"provenance", "experiment_id", "experiment_identity_payload"},
    "combinada": {
        "checkpoint",
        "parent_experiment_id",
        "calibration_set",
        "calibration_image_names",
        "experiment_id",
        "experiment_identity_payload",
    },
}


def _volcado_real_de_coco() -> dict[str, Any]:
    """Lee el volcado real de la evaluación COCO."""
    if not VOLCADO_REAL_DE_COCO.is_file():
        pytest.skip(f"No está {VOLCADO_REAL_DE_COCO.name}.")
    return dict(json.loads(VOLCADO_REAL_DE_COCO.read_text(encoding="utf-8")))


def _informe_real_de_combinada() -> dict[str, Any]:
    """Lee el informe real de una rama combinada."""
    if not INFORME_REAL_DE_COMBINADA.is_file():
        pytest.skip(f"No está {INFORME_REAL_DE_COMBINADA.name}.")
    return dict(json.loads(INFORME_REAL_DE_COMBINADA.read_text(encoding="utf-8")))


def _contenido_real_de_coco() -> dict[str, Any]:
    """Lo que el resultado de una evaluación COCO aporta al volcado."""
    return {
        clave: valor
        for clave, valor in _volcado_real_de_coco().items()
        if clave not in ANADE_EL_BLOQUE["coco"]
    }


def _contenido_real_de_combinada() -> dict[str, Any]:
    """Lo que el resultado de una materialización aporta al informe."""
    return {
        clave: valor
        for clave, valor in _informe_real_de_combinada().items()
        if clave not in ANADE_EL_BLOQUE["combinada"]
    }


# ---------------------------------------------------------------------------
# Dobles con la forma que el volcado consume.
# ---------------------------------------------------------------------------


@dataclass
class _InformeConMetricas:
    """Informe de reajuste o de afinado, con el mínimo que el bloque mira."""

    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)

    @property
    def metrics_baseline(self) -> dict[str, float]:
        """Nombre que usa el informe de QAT para lo mismo."""
        return self.metrics_before

    def to_dict(self) -> dict[str, Any]:
        """Lo que el volcado escribe de este informe."""
        return {"metrics_before": self.metrics_before, "metrics_after": self.metrics_after}


@dataclass
class _ConToDict:
    """Cualquier pieza del volcado que solo aporte su ``to_dict()``."""

    contenido: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el contenido."""
        return dict(self.contenido)


@dataclass
class _ResultadoDeDestilacion:
    """Lo que ``campos_del_entrenamiento`` consume de un resultado."""

    steps: int = 120
    adapter_params: int = 4096
    adapter_initial_norms: dict[str, float] = field(default_factory=lambda: {"model.15": 1.0})
    adapter_final_norms: dict[str, float] = field(default_factory=lambda: {"model.15": 1.5})
    adapters_learned: bool = True
    adapter_grad_was_always_zero: bool = False
    epoch_components: list[dict[str, float]] = field(default_factory=list)
    training: Any = None


def _config(ruta_relativa: str) -> ProjectConfig:
    """Carga una configuración real del repositorio."""
    return load_config(RAIZ / "configs" / ruta_relativa)


class TestLaLlamadaYLaFirmaNoPuedenDivergir:
    """Extraer el bloque deja una costura que los tests de arriba no ven."""

    @pytest.mark.parametrize(
        ("nombre", "funcion"),
        [
            ("13_prune", "volcar_detalle"),
            ("16_qat", "volcar_detalle"),
            ("20_distill", "volcar_detalle"),
            ("10_quantize_ptq", "volcar_informe"),
            ("06_evaluate_coco", "volcar_detalle"),
            ("22_combine", "volcar_informe"),
        ],
    )
    def test_main_pasa_exactamente_los_argumentos_de_la_firma(
        self, nombre: str, funcion: str
    ) -> None:
        """Ni uno de menos, ni uno que no exista."""
        ruta = RAIZ / "scripts" / f"{nombre}.py"
        arbol = ast.parse(ruta.read_text(encoding="utf-8"))
        llamadas = [
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Name)
            and nodo.func.id == funcion
        ]
        assert len(llamadas) == 1, f"{nombre}: {len(llamadas)} llamadas a {funcion}"

        firma = inspect.signature(getattr(_cargar(nombre), funcion))
        esperados = {
            parametro.name
            for parametro in firma.parameters.values()
            if parametro.kind is inspect.Parameter.KEYWORD_ONLY
        }
        pasados = {palabra.arg for palabra in llamadas[0].keywords}

        assert len(llamadas[0].args) == 1, "el directorio de salida va posicional"
        assert pasados == esperados, sorted(pasados ^ esperados)


# ---------------------------------------------------------------------------
# Poda.
# ---------------------------------------------------------------------------


class TestElVolcadoDeLaPoda:
    """``13_prune.py``: el volcado de cada rama."""

    def _volcar(self, destino: Path, **cambios: Any) -> Any:
        """Llama a ``volcar_detalle`` con argumentos completos."""
        script = _cargar("13_prune")
        config = _config("compression/pruning/l1_global_r50.yaml")
        # La identidad se calcula con el código del script, así que el volcado lleva el payload que
        # se hasheó.
        payload, _, experiment_id = script.construir_identidad(
            config, "l1_global", "l1global_r50", True, HUELLA_DE_PEGA, "s_baseline_default_745e3d08"
        )
        argumentos: dict[str, Any] = {
            "experiment_id": experiment_id,
            "checkpoint": Path("models/baselines/s_baseline_default/best.pt"),
            "config_path": Path("configs/compression/pruning/l1_global_r50.yaml"),
            "criterio": "l1_global",
            "es_control": False,
            "smoke_test": False,
            "config": config,
            "seed_base": 42,
            "seed_branch": 1234,
            "identity_payload": payload,
            "schedule": _ConToDict({"steps": 1}),
            "resultado_poda": _ConToDict({"ratio": 0.5}),
            "informe_reajuste": _InformeConMetricas(metrics_after={"map_50_95": 0.3}),
            "dry_run": False,
        }
        argumentos.update(cambios)
        return script.volcar_detalle(destino, **argumentos)

    def test_escribe_el_json_con_el_nombre_del_experimento(self, tmp_path: Path) -> None:
        """El nombre del fichero es lo que ata el volcado a su fila."""
        ruta = self._volcar(tmp_path)

        assert ruta is not None and ruta.parent == tmp_path
        assert ruta.stem.startswith("s_pruning_l1global_r50_")
        assert json.loads(ruta.read_text(encoding="utf-8"))["criterion"] == "l1_global"

    def test_crea_el_directorio_si_no_existe(self, tmp_path: Path) -> None:
        """Toda función que escribe crea sus padres (convención del proyecto)."""
        ruta = self._volcar(tmp_path / "sin" / "crear")

        assert ruta is not None and ruta.is_file()

    def test_con_dry_run_no_escribe_nada(self, tmp_path: Path) -> None:
        assert self._volcar(tmp_path, dry_run=True) is None
        assert list(tmp_path.iterdir()) == []

    def test_no_pisa_una_medida_con_una_ejecucion_sin_medir(self, tmp_path: Path) -> None:
        """La guarda de la evidencia, ejercitada por primera vez."""
        primera = self._volcar(tmp_path)
        assert primera is not None
        antes = primera.read_bytes()

        segunda = self._volcar(tmp_path, informe_reajuste=_InformeConMetricas())

        assert segunda is None
        assert primera.read_bytes() == antes, "ha pisado una medida con una no medida"

    def test_una_campana_que_si_mide_sustituye_a_la_anterior(self, tmp_path: Path) -> None:
        """La regla no es «no sobrescribir», es «no sobrescribir con menos»."""
        primera = self._volcar(tmp_path)
        assert primera is not None

        segunda = self._volcar(
            tmp_path, informe_reajuste=_InformeConMetricas(metrics_after={"map_50_95": 0.4})
        )

        assert segunda == primera
        contenido = json.loads(primera.read_text(encoding="utf-8"))
        assert contenido["finetune"]["metrics_after"] == {"map_50_95": 0.4}

    def test_las_claves_son_las_del_volcado_real_mas_el_payload(self, tmp_path: Path) -> None:
        ruta = self._volcar(tmp_path)
        assert ruta is not None

        producidas = set(json.loads(ruta.read_text(encoding="utf-8")))
        reales = _claves_reales("pruning")

        assert producidas - reales == {"experiment_identity_payload"}
        assert reales - producidas == set(), sorted(reales - producidas)

    def test_el_payload_volcado_reproduce_el_identificador(self, tmp_path: Path) -> None:
        ruta = self._volcar(tmp_path)
        assert ruta is not None

        payload = _comprobar_que_el_payload_reproduce_el_identificador(ruta)

        assert payload["starting_checkpoint_sha256"] == HUELLA_DE_PEGA
        assert payload["pruning"]["criterion"] == "l1_global"

    def test_el_payload_lleva_el_punto_de_partida_sin_los_pesos_delante(
        self, tmp_path: Path
    ) -> None:
        """Lo que hace verificable la identidad de un experimento antiguo."""
        ruta = self._volcar(tmp_path)
        assert ruta is not None
        payload = json.loads(ruta.read_text(encoding="utf-8"))["experiment_identity_payload"]

        assert payload["parent_experiment_id"] == "s_baseline_default_745e3d08"
        assert payload["starting_checkpoint_sha256"] == HUELLA_DE_PEGA


# ---------------------------------------------------------------------------
# QAT.
# ---------------------------------------------------------------------------


class TestElVolcadoDelQat:
    """``16_qat.py``: mismo bloque y mismas guardas."""

    def _volcar(self, destino: Path, **cambios: Any) -> Any:
        """Llama a ``volcar_detalle`` con argumentos completos."""
        script = _cargar("16_qat")
        config = _config("compression/quantization/qat_default.yaml")
        payload, _, experiment_id = script.construir_identidad(
            config, HUELLA_DE_PEGA, "s_baseline_default_745e3d08"
        )
        argumentos: dict[str, Any] = {
            "experiment_id": experiment_id,
            "checkpoint": Path("models/baselines/s_baseline_default/best.pt"),
            "config_path": Path("configs/compression/quantization/qat_default.yaml"),
            "smoke_test": False,
            "config": config,
            "seed_branch": 1234,
            "identity_payload": payload,
            "informe": _InformeConMetricas(metrics_after={"map_50_95": 0.3}),
            "baseline_reference": {"map_50_95": 0.31, "evaluator": "pycocotools"},
            "dry_run": False,
        }
        argumentos.update(cambios)
        return script.volcar_detalle(destino, **argumentos)

    def test_escribe_el_json_con_el_nombre_del_experimento(self, tmp_path: Path) -> None:
        """Y con la referencia del baseline dentro, que es contra lo que compara."""
        ruta = self._volcar(tmp_path)

        assert ruta is not None and ruta.parent == tmp_path
        assert ruta.stem.startswith("s_qat_int8-noexcl_")
        contenido = json.loads(ruta.read_text(encoding="utf-8"))
        assert contenido["baseline_reference"]["evaluator"] == "pycocotools"

    def test_con_dry_run_no_escribe_nada(self, tmp_path: Path) -> None:
        assert self._volcar(tmp_path, dry_run=True) is None
        assert list(tmp_path.iterdir()) == []

    def test_no_pisa_una_medida_con_una_ejecucion_sin_medir(self, tmp_path: Path) -> None:
        primera = self._volcar(tmp_path)
        assert primera is not None
        antes = primera.read_bytes()

        segunda = self._volcar(tmp_path, informe=_InformeConMetricas())

        assert segunda is None
        assert primera.read_bytes() == antes

    def test_las_claves_son_las_del_volcado_real_mas_el_payload(self, tmp_path: Path) -> None:
        """Contraste contra el disco."""
        ruta = self._volcar(tmp_path)
        assert ruta is not None

        producidas = set(json.loads(ruta.read_text(encoding="utf-8")))
        reales = _claves_reales("qat")

        assert producidas - reales == {"experiment_identity_payload"}
        assert reales - producidas == set(), sorted(reales - producidas)

    def test_el_payload_volcado_reproduce_el_identificador(self, tmp_path: Path) -> None:
        ruta = self._volcar(tmp_path)
        assert ruta is not None

        payload = _comprobar_que_el_payload_reproduce_el_identificador(ruta)

        assert payload["starting_checkpoint_sha256"] == HUELLA_DE_PEGA
        assert payload["warmup_epochs"] == _cargar("16_qat").WARMUP_EPOCHS_QAT


# ---------------------------------------------------------------------------
# Destilación.
# ---------------------------------------------------------------------------


class TestElVolcadoDeLaDestilacion:
    """``20_distill.py``: el volcado de destilación."""

    def _volcar(self, destino: Path, **cambios: Any) -> Any:
        """Llama a ``volcar_detalle`` con argumentos completos."""
        script = _cargar("20_distill")
        resultado = _ResultadoDeDestilacion(training=_ConToDict())
        resultado.training = type("_Entrenamiento", (), {"cost": _ConToDict({"gpu_hours": 1.0})})()
        config = _config("compression/distillation/kd_combined.yaml")
        payload, _, experiment_id = script.construir_identidad(
            config, HUELLA_DE_PEGA, HUELLA_DEL_MAESTRO, "n_baseline_default_fab9d45c"
        )
        argumentos: dict[str, Any] = {
            "experiment_id": experiment_id,
            "es_control": False,
            "alumno_pt": Path("models/baselines/n_baseline_default/best.pt"),
            "maestro_pt": Path("models/baselines/l_baseline_default/best.pt"),
            "config_path": Path("configs/compression/distillation/kd_combined.yaml"),
            "smoke_test": False,
            "config": config,
            "seed_branch": 1234,
            "identity_payload": payload,
            "lote": 16,
            "cierre_mosaico": 10,
            "regimen": "sin mosaico",
            "previa": None,
            "metricas": {"map_50": 0.5, "map_50_95": 0.3},
            "resultado": resultado,
            "reutilizado": None,
            "dry_run": False,
        }
        argumentos.update(cambios)
        return script.volcar_detalle(destino, **argumentos)

    def test_escribe_el_json_con_la_receta_efectiva(self, tmp_path: Path) -> None:
        """La receta que no vive en la sección de destilación."""
        ruta = self._volcar(tmp_path)

        assert ruta is not None and ruta.parent == tmp_path
        assert ruta.stem.startswith("n_distillation_l2n_combined_t2_")
        receta = json.loads(ruta.read_text(encoding="utf-8"))["training_recipe"]
        assert receta["batch_size"] == 16
        assert receta["mosaic_regime"] == "sin mosaico"

    def test_con_dry_run_no_escribe_nada(self, tmp_path: Path) -> None:
        assert self._volcar(tmp_path, dry_run=True) is None
        assert list(tmp_path.iterdir()) == []

    def test_no_pisa_una_medida_con_una_ejecucion_sin_medir(self, tmp_path: Path) -> None:
        """Aquí la guarda mira ``metrics`` de primer nivel."""
        primera = self._volcar(tmp_path)
        assert primera is not None
        antes = primera.read_bytes()

        segunda = self._volcar(tmp_path, metricas={})

        assert segunda is None
        assert primera.read_bytes() == antes

    def test_sin_resultado_ni_reutilizado_falla_en_vez_de_escribir_a_medias(
        self, tmp_path: Path
    ) -> None:
        """``campos_del_entrenamiento`` exige exactamente uno de los dos."""
        with pytest.raises(ValueError, match="exactamente uno"):
            self._volcar(tmp_path, resultado=None, reutilizado=None)

        assert list(tmp_path.iterdir()) == [], "ha dejado un volcado a medias"

    def test_las_claves_son_las_del_volcado_real_mas_el_payload(self, tmp_path: Path) -> None:
        """Contraste contra el volcado de la campaña de 50 épocas."""
        ruta = self._volcar(tmp_path)
        assert ruta is not None

        producidas = set(json.loads(ruta.read_text(encoding="utf-8")))
        reales = _claves_reales("distillation")

        assert producidas - reales == {"experiment_identity_payload"}
        assert reales - producidas == set(), sorted(reales - producidas)

    def test_el_payload_volcado_reproduce_el_identificador(self, tmp_path: Path) -> None:
        """Con los dos puntos de partida dentro, que es lo propio de aquí."""
        ruta = self._volcar(tmp_path)
        assert ruta is not None

        payload = _comprobar_que_el_payload_reproduce_el_identificador(ruta)

        assert payload["student_checkpoint_sha256"] == HUELLA_DE_PEGA
        assert payload["teacher_checkpoint_sha256"] == HUELLA_DEL_MAESTRO


# ---------------------------------------------------------------------------
# PTQ.
# ---------------------------------------------------------------------------


class TestElInformeDeLaPtq:
    """``10_quantize_ptq.py``: el informe de PTQ, que se escribe siempre."""

    def _volcar(self, destino: Path, **cambios: Any) -> Path:
        """Llama a ``volcar_informe`` con argumentos completos."""
        script = _cargar("10_quantize_ptq")
        configuracion = Int8Config()
        payload, _, experiment_id = script.construir_identidad(
            model_size="s",
            variante=script.DEFAULT_VARIANT,
            resolucion=640,
            configuracion=configuracion,
            nombre_calibracion="calibration_512.txt",
            huella_calibracion=HUELLA_DEL_MAESTRO,
            runtimes=["onnxruntime", "openvino"],
            seed=42,
            huella_checkpoint=HUELLA_DE_PEGA,
            parent_experiment_id="s_baseline_default_745e3d08",
        )
        argumentos: dict[str, Any] = {
            "experiment_id": experiment_id,
            "resultado": _ConToDict({"runtimes": {}}),
            "checkpoint": Path("models/baselines/s_baseline_default/best.pt"),
            "calibration_set": Path("configs/dataset/calibration_512.txt"),
            "calibration_images": [Path("a.jpg"), Path("b.jpg")],
            "identity_payload": payload,
        }
        argumentos.update(cambios)
        return script.volcar_informe(destino, **argumentos)

    def test_escribe_el_informe_con_el_nombre_del_experimento(self, tmp_path: Path) -> None:
        ruta = self._volcar(tmp_path)

        assert ruta.parent == tmp_path
        assert ruta.name.endswith("_ptq.json")
        assert ruta.stem.startswith("s_ptq_int8-noexcl_")

    def test_el_informe_declara_dentro_a_que_experimento_pertenece(self, tmp_path: Path) -> None:
        contenido = json.loads(self._volcar(tmp_path).read_text(encoding="utf-8"))

        assert contenido["experiment_id"] == self._volcar(tmp_path).stem[: -len("_ptq")]

    def test_el_payload_volcado_reproduce_el_identificador(self, tmp_path: Path) -> None:
        """El sufijo del identificador sale del payload que está dentro."""
        ruta = self._volcar(tmp_path)

        payload = _comprobar_que_el_payload_reproduce_el_identificador(ruta)

        assert payload["starting_checkpoint_sha256"] == HUELLA_DE_PEGA
        assert payload["calibration_hash"] == HUELLA_DEL_MAESTRO

    def test_conserva_lo_que_ya_escribia(self, tmp_path: Path) -> None:
        contenido = json.loads(self._volcar(tmp_path).read_text(encoding="utf-8"))

        assert contenido["calibration_image_names"] == ["a.jpg", "b.jpg"]
        assert contenido["checkpoint"].endswith("best.pt")


# ---------------------------------------------------------------------------
# Evaluación de baselines: una sola fórmula para los pasos 05 y 06.
# ---------------------------------------------------------------------------


class TestLaIdentidadDeLaEvaluacionDeBaseline:
    """La fórmula que estaba escrita dos veces, y los cuatro identificadores reales."""

    def test_reproduce_los_cuatro_identificadores_del_registro(self) -> None:
        """Con los valores por defecto de los guiones, salen los del registro."""
        registro = RAIZ / "results" / "registry" / "experiments.csv"
        if not registro.is_file():
            pytest.skip("No está el registro.")
        with registro.open(encoding="utf-8") as fichero:
            reales = {
                fila["experiment_id"]
                for fila in csv.DictReader(fichero)
                if fila["technique"] == "baseline"
            }
        if not reales:
            pytest.skip("El registro no tiene baselines.")

        clases = load_class_names(RAIZ / "configs" / "dataset" / "sh17_classes.json")
        # Los valores por defecto se leen del parser del script.
        args = _cargar("05_evaluate_baselines").construir_parser().parse_args([])
        obtenidos = {
            build_baseline_evaluation_identity(
                model_size=size,
                num_classes=len(clases),
                input_resolution=args.resolution,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                split=args.split,
                data=ULTRALYTICS_DATASET_FILENAME,
                seed=args.seed,
            )[2]
            for size in ("n", "s", "m", "l")
        }

        assert reales <= obtenidos, sorted(reales - obtenidos)

    @pytest.mark.parametrize("nombre", ["05_evaluate_baselines", "06_evaluate_coco"])
    def test_ninguno_de_los_dos_vuelve_a_construir_el_payload(self, nombre: str) -> None:
        arbol = ast.parse((RAIZ / "scripts" / f"{nombre}.py").read_text(encoding="utf-8"))
        llamadas = [
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Name)
            and nodo.func.id in {"compute_config_hash", "build_experiment_id"}
        ]

        assert llamadas == [], f"{nombre} vuelve a construir la identidad por su cuenta"


class TestElVolcadoDeLaEvaluacionCoco:
    """``06_evaluate_coco.py``: el detalle de la evaluación COCO."""

    def _volcar(self, destino: Path, **cambios: Any) -> Path:
        """Llama a ``volcar_detalle`` con argumentos completos."""
        script = _cargar("06_evaluate_coco")
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
        argumentos: dict[str, Any] = {
            "output_dir": destino,
            "size": "n",
            "split": "val",
            "resolution": 640,
            "device_id": "colab_t4_gpu",
            "environment_id": "0123456789ab",
            "checkpoint": Path("models/baselines/n_baseline_default/best.pt"),
            "experiment_id": experiment_id,
            "identity_payload": payload,
        }
        argumentos.update(cambios)
        return script.volcar_detalle(_ConToDict(_contenido_real_de_coco()), **argumentos)

    def test_el_volcado_declara_dentro_a_que_experimento_pertenece(self, tmp_path: Path) -> None:
        """El nombre del fichero no lo dice, y hasta hoy el contenido tampoco."""
        ruta = self._volcar(tmp_path)
        contenido = json.loads(ruta.read_text(encoding="utf-8"))

        assert "n_baseline_default_" in contenido["experiment_id"]
        assert contenido["experiment_id"] not in ruta.name

    def test_el_payload_volcado_reproduce_el_identificador(self, tmp_path: Path) -> None:
        """Lo guardado es lo hasheado, recalculado sobre el fichero escrito."""
        payload = _comprobar_que_el_payload_reproduce_el_identificador(self._volcar(tmp_path))

        assert payload["technique"] == "baseline"
        assert payload["split"] == "val"

    def test_las_claves_son_las_del_volcado_real_mas_el_payload(self, tmp_path: Path) -> None:
        producidas = set(json.loads(self._volcar(tmp_path).read_text(encoding="utf-8")))
        reales = set(_volcado_real_de_coco())

        assert producidas - reales == {"experiment_id", "experiment_identity_payload"}
        assert reales - producidas == set(), sorted(reales - producidas)

    def test_no_pisa_un_volcado_anterior(self, tmp_path: Path) -> None:
        self._volcar(tmp_path)

        with pytest.raises(FileExistsError):
            self._volcar(tmp_path)

    def test_con_overwrite_si_lo_pisa(self, tmp_path: Path) -> None:
        """Y decirlo cuesta una bandera, que es lo que la hace una decisión."""
        primera = self._volcar(tmp_path)
        segunda = self._volcar(tmp_path, overwrite=True)

        assert primera == segunda


class TestElInformeDeLaRamaCombinada:
    """``22_combine.py``: el informe de cada rama de poda y PTQ."""

    def _identidad(self, **cambios: Any) -> tuple[dict[str, Any], str, str]:
        """Construye una identidad combinada con los argumentos del guion."""
        script = _cargar("22_combine")
        argumentos: dict[str, Any] = {
            "model_size": "s",
            "variante": "prune-l1global-r50-ptqint8",
            "padre": {
                "experiment_id": "s_pruning_l1global_r50_9419b26f",
                "config_hash": "9419b26f",
            },
            "input_resolution": 640,
            "int8_config": Int8Config().to_dict(),
            "calibration_set": "calibration_128.txt",
            "calibration_hash": HUELLA_DE_PEGA,
            "runtimes": ["onnxruntime", "openvino"],
            "seed": 42,
        }
        argumentos.update(cambios)
        return script.construir_identidad(**argumentos)

    def _volcar(self, destino: Path, **cambios: Any) -> Path:
        """Llama a ``volcar_informe`` con argumentos completos."""
        script = _cargar("22_combine")
        payload, _, experiment_id = self._identidad()
        argumentos: dict[str, Any] = {
            "experiment_id": experiment_id,
            "resultado": _ConToDict(_contenido_real_de_combinada()),
            "checkpoint": "models/pruned/s_pruning_l1global_r50_9419b26f/finetune/train/best.pt",
            "parent_experiment_id": "s_pruning_l1global_r50_9419b26f",
            "calibration_set": Path("configs/dataset/calibration_128.txt"),
            "calibration_images": [Path("a.jpg"), Path("b.jpg")],
            "identity_payload": payload,
        }
        argumentos.update(cambios)
        return script.volcar_informe(destino, **argumentos)

    def test_el_padre_entra_en_la_identidad(self) -> None:
        """Dos podas distintas con la misma variante no comparten identificador."""
        _, _, uno = self._identidad()
        _, _, otro = self._identidad(
            padre={"experiment_id": "s_pruning_bnscale_r50_f51fef4c", "config_hash": "f51fef4c"}
        )

        assert uno != otro

    def test_el_informe_declara_dentro_a_que_experimento_pertenece(self, tmp_path: Path) -> None:
        ruta = self._volcar(tmp_path)
        contenido = json.loads(ruta.read_text(encoding="utf-8"))

        assert contenido["experiment_id"] == ruta.stem[: -len("_combined")]

    def test_el_payload_volcado_reproduce_el_identificador(self, tmp_path: Path) -> None:
        """Lo guardado es lo hasheado, recalculado sobre el fichero escrito."""
        payload = _comprobar_que_el_payload_reproduce_el_identificador(self._volcar(tmp_path))

        assert payload["parent_experiment_id"] == "s_pruning_l1global_r50_9419b26f"
        assert payload["calibration_hash"] == HUELLA_DE_PEGA

    def test_las_claves_son_las_del_informe_real_mas_el_payload(self, tmp_path: Path) -> None:
        """Contraste contra los seis informes de la campaña, que son idénticos."""
        producidas = set(json.loads(self._volcar(tmp_path).read_text(encoding="utf-8")))
        reales = set(_informe_real_de_combinada())

        assert producidas - reales == {"experiment_id", "experiment_identity_payload"}
        assert reales - producidas == set(), sorted(reales - producidas)

    def test_los_seis_identificadores_de_la_campana_siguen_saliendo(self) -> None:
        registro = RAIZ / "results" / "registry" / "experiments.csv"
        if not registro.is_file():
            pytest.skip("No está el registro.")
        with registro.open(encoding="utf-8") as fichero:
            filas = {fila["experiment_id"]: fila for fila in csv.DictReader(fichero)}
        combinadas = [fila for fila in filas.values() if fila["technique"] == "combined"]
        if not combinadas:
            pytest.skip("El registro no tiene ramas combinadas.")

        script = _cargar("22_combine")
        args = script.construir_parser().parse_args(["--all-pruned"])
        for fila in combinadas:
            informe_path = RAIZ / fila["config_path"]
            if not informe_path.is_file():
                pytest.skip(f"No está el informe {fila['config_path']}.")
            informe = json.loads(informe_path.read_text(encoding="utf-8"))
            _, _, obtenido = script.construir_identidad(
                model_size=fila["model_size"],
                variante=informe["variant"],
                padre=filas[informe["parent_experiment_id"]],
                input_resolution=args.resolution,
                int8_config=informe["config"],
                calibration_set=Path(informe["calibration_set"]).name,
                calibration_hash=calibration_hash(informe["calibration_image_names"]),
                runtimes=sorted(informe["int8_artifacts"]),
                seed=args.seed,
            )

            assert obtenido == fila["experiment_id"]


# ---------------------------------------------------------------------------
# Sensibilidad por capa.
# ---------------------------------------------------------------------------


class TestLasRutasDelVolcadoDeSensibilidad:
    """``09_sensitivity.py``: su volcado se versiona y no lleva rutas absolutas."""

    @pytest.mark.parametrize("clave", ["checkpoint", "calibration_set"])
    def test_la_ruta_no_se_escribe_tal_cual(self, clave: str) -> None:
        arbol = ast.parse((RAIZ / "scripts" / "09_sensitivity.py").read_text(encoding="utf-8"))
        asignaciones = [
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Assign)
            and len(nodo.targets) == 1
            and isinstance(nodo.targets[0], ast.Subscript)
            and isinstance(nodo.targets[0].value, ast.Name)
            and nodo.targets[0].value.id == "datos"
            and isinstance(nodo.targets[0].slice, ast.Constant)
            and nodo.targets[0].slice.value == clave
        ]

        assert len(asignaciones) == 1, f"{len(asignaciones)} asignaciones a datos[{clave!r}]"
        valor = asignaciones[0].value
        assert isinstance(valor, ast.Call) and isinstance(
            valor.func, ast.Name
        ), f"datos[{clave!r}] no sale de una llamada"
        assert (
            valor.func.id == "relative_to_root"
        ), f"datos[{clave!r}] se escribe con {valor.func.id}, no con relative_to_root"


# ---------------------------------------------------------------------------
# Rutas absolutas en los volcados versionados.
# ---------------------------------------------------------------------------


#: Los cinco pasos que escriben un volcado versionado con rutas: el 11, el 23 y los informes de
#: exportación del 10, el 18 y el 22.
PRODUCTORES_CON_RUTAS = (
    "scripts/10_quantize_ptq.py",
    "scripts/11_evaluate_artifact.py",
    "scripts/18_quantize_qat.py",
    "scripts/22_combine.py",
    "scripts/23_operating_curve.py",
)


class TestLasRutasDeLosVolcadosVersionados:
    """Ningún volcado nuevo vuelve a llevar dentro dónde estaba esta máquina."""

    def test_una_ruta_del_proyecto_sale_relativa(self) -> None:
        volcado = {"calibration_set": str(RAIZ / "configs" / "dataset" / "calibration_512.txt")}

        assert relativize_dump_paths(volcado) == {
            "calibration_set": "configs/dataset/calibration_512.txt"
        }

    def test_una_ruta_de_otra_maquina_se_queda_absoluta(self) -> None:
        """Los volcados de Colab."""
        colab = "/content/drive/MyDrive/tfm-edge/runs/train/weights/best.pt"

        assert relativize_dump_paths({"checkpoint": colab}) == {"checkpoint": colab}

    def test_una_ruta_ya_relativa_no_se_resuelve_contra_el_cwd(self) -> None:
        """La función es idempotente, que es lo que la hace segura de aplicar."""
        volcado = {"path": "results/exports/x/m.onnx"}

        assert relativize_dump_paths(relativize_dump_paths(volcado)) == volcado

    def test_solo_se_tocan_las_claves_declaradas(self) -> None:
        """Una cadena que empieza por barra no es necesariamente una ruta."""
        volcado = {"notes": "/tensorrt///", "split": "/val"}

        assert relativize_dump_paths(volcado) == volcado

    def test_llega_a_las_rutas_anidadas(self) -> None:
        """Las de exportación viven tres niveles abajo, y dentro de listas."""
        onnx = str(RAIZ / "results" / "exports" / "x" / "m.onnx")
        volcado = {
            "fp32_exports": {"onnx": {"path": onnx, "metadata": {"from_onnx": onnx}}},
            "steps": [{"source_path": onnx}],
        }

        convertido = relativize_dump_paths(volcado)

        assert convertido["fp32_exports"]["onnx"]["path"] == "results/exports/x/m.onnx"
        assert convertido["fp32_exports"]["onnx"]["metadata"]["from_onnx"] == (
            "results/exports/x/m.onnx"
        )
        assert convertido["steps"][0]["source_path"] == "results/exports/x/m.onnx"

    def test_un_valor_que_no_es_cadena_no_estorba(self) -> None:
        """En los volcados de benchmark, ``artifact`` es un diccionario entero."""
        volcado = {"artifact": {"path": "/tmp/x.onnx", "sha256": "abc"}}

        assert relativize_dump_paths(volcado)["artifact"]["path"] == "/tmp/x.onnx"

    @pytest.mark.parametrize("guion", PRODUCTORES_CON_RUTAS)
    def test_el_productor_serializa_a_traves_de_la_regla(self, guion: str) -> None:
        arbol = ast.parse((RAIZ / guion).read_text(encoding="utf-8"))
        volcados = [
            nodo
            for nodo in ast.walk(arbol)
            if isinstance(nodo, ast.Call)
            and isinstance(nodo.func, ast.Attribute)
            and nodo.func.attr == "dumps"
        ]

        assert volcados, f"{guion} no escribe ningún volcado JSON"
        for llamada in volcados:
            assert llamada.args, f"{guion}: un json.dumps sin argumento posicional"
            primero = llamada.args[0]
            assert isinstance(primero, ast.Call) and isinstance(
                primero.func, ast.Name
            ), f"{guion}: json.dumps recibe el volcado sin pasarlo por relativize_dump_paths"
            assert primero.func.id == "relativize_dump_paths", (
                f"{guion}: json.dumps recibe {primero.func.id}(...), no "
                f"relativize_dump_paths(...)"
            )
