"""La evidencia que la campaña de baselines dejó, y los pares que quedan duplicados."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

RAIZ = Path(__file__).resolve().parent.parent
BASELINES = RAIZ / "models" / "baselines"
REGISTRO = RAIZ / "results" / "registry" / "experiments.csv"

TAMANOS = ("n", "s", "m", "l")

#: Lo que ``14_import_colab_registry.py`` llama ``evidence_files`` de un baseline.
FICHEROS_DE_EVIDENCIA = ("schedule.json", "complexity.json", "training_cost.json")

#: Calendario planificado (izquierda) frente a lo que Ultralytics dejó escrito
#: (derecha). Los nombres no coinciden, que es parte de por qué nadie los había
#: contrastado.
EQUIVALENCIAS_DEL_CALENDARIO = (
    ("epochs", "epochs"),
    ("warmup_epochs", "warmup_epochs"),
    ("close_mosaic_epochs", "close_mosaic"),
    ("initial_lr", "lr0"),
    ("final_lr_fraction", "lrf"),
    ("patience", "patience"),
    ("checkpoint_every_epochs", "save_period"),
)


def _directorio(size: str) -> Path:
    """Directorio del baseline de ese tamaño."""
    return BASELINES / f"{size}_baseline_default"


def _leer_json(ruta: Path) -> dict[str, Any]:
    """Lee un volcado, o salta si no está."""
    if not ruta.is_file():
        pytest.skip(f"No está {ruta.relative_to(RAIZ)}.")
    return dict(json.loads(ruta.read_text(encoding="utf-8")))


def _fila_del_registro(size: str) -> dict[str, str]:
    """Fila del baseline de ese tamaño en el registro."""
    if not REGISTRO.is_file():
        pytest.skip("No está el registro.")
    with REGISTRO.open(encoding="utf-8") as fichero:
        filas = [
            fila
            for fila in csv.DictReader(fichero)
            if fila["technique"] == "baseline" and fila["model_size"] == size
        ]
    if not filas:
        pytest.skip(f"El registro no tiene el baseline de {size}.")
    assert len(filas) == 1, f"{len(filas)} filas de baseline para {size}"
    return filas[0]


@pytest.mark.parametrize("size", TAMANOS)
class TestLaEvidenciaDeCadaBaselineEstaEnElRepositorio:
    """Los tres volcados que la importación declara como evidencia, más su receta."""

    def test_estan_los_tres_ficheros_de_evidencia(self, size: str) -> None:
        """Los que ``14_import_colab_registry.py`` nombra uno a uno."""
        directorio = _directorio(size)
        faltan = [nombre for nombre in FICHEROS_DE_EVIDENCIA if not (directorio / nombre).is_file()]

        assert faltan == [], f"{size}: faltan {faltan}"

    def test_el_calendario_planificado_es_el_que_quedo_escrito(self, size: str) -> None:
        """``schedule.json`` contra el ``args.yaml`` de Ultralytics."""
        calendario = _leer_json(_directorio(size) / "schedule.json")
        ruta_args = _directorio(size) / "args.yaml"
        if not ruta_args.is_file():
            pytest.skip(f"No está el args.yaml de {size}.")
        argumentos = yaml.safe_load(ruta_args.read_text(encoding="utf-8"))

        discrepancias = [
            (
                clave_propia,
                calendario[clave_propia],
                clave_ultralytics,
                argumentos[clave_ultralytics],
            )
            for clave_propia, clave_ultralytics in EQUIVALENCIAS_DEL_CALENDARIO
            if calendario[clave_propia] != argumentos[clave_ultralytics]
        ]

        assert discrepancias == [], f"{size}: {discrepancias}"

    def test_el_coste_medido_es_el_que_dice_el_registro(self, size: str) -> None:
        """``training_cost.json`` contra ``training_time_s`` y ``training_epochs``."""
        coste = _leer_json(_directorio(size) / "training_cost.json")
        fila = _fila_del_registro(size)

        assert float(fila["training_time_s"]) == pytest.approx(coste["wall_time_s"])
        assert int(fila["training_epochs"]) == coste["epochs_completed"]

    def test_los_parametros_medidos_son_los_que_dice_el_registro(self, size: str) -> None:
        """``complexity.json`` contra ``effective_params``."""
        complejidad = _leer_json(_directorio(size) / "complexity.json")
        fila = _fila_del_registro(size)

        assert int(fila["effective_params"]) == complejidad["params_total"]


# No se comprueba que la receta de los cuatro baselines ya entrenados coincida con la configuración
# actual.
