"""Raíz del proyecto, rutas de artefactos y convención de experiment_id.

Todas las rutas del proyecto se construyen aquí. El experiment_id se usa también como nombre de
directorio, así que tiene que ser válido en Linux y en Windows.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from tfm_edge.common.exceptions import (
    InvalidExperimentIdError,
    InvalidPathComponentError,
    ProjectRootNotFoundError,
)

ROOT_MARKER = "pyproject.toml"

# Variables de entorno para reubicar cada árbol.
ENV_ROOT = "TFM_EDGE_ROOT"
ENV_DATA_DIR = "TFM_EDGE_DATA_DIR"
ENV_MODELS_DIR = "TFM_EDGE_MODELS_DIR"
ENV_RESULTS_DIR = "TFM_EDGE_RESULTS_DIR"

#: Ancla de relocate_under_results: desde aquí empieza la parte relativa de una ruta grabada en otra
#: máquina.
RESULTS_ANCHOR = "results"

# Nombres de fichero válidos a la vez en Linux y en Windows

FORBIDDEN_CHARS = set('<>:"/\\|?*')

#: Nombres reservados por Windows, insensibles a mayúsculas, con o sin extensión.
WINDOWS_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)

#: Longitud máxima admitida para un componente de ruta.
MAX_COMPONENT_LENGTH = 120

# Convención de experiment_id

#: Tamaños de modelo contemplados.
VALID_MODEL_SIZES = ("n", "s", "m", "l")

#: Técnicas contempladas en la columna `technique` de experiments.csv.
VALID_TECHNIQUES = ("baseline", "ptq", "qat", "pruning", "distillation", "combined")

#: Longitud del hash de configuración embebido en el experiment_id.
CONFIG_HASH_LENGTH = 8

_HASH_RE = re.compile(r"^[0-9a-f]{%d}$" % CONFIG_HASH_LENGTH)
_VARIANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


# Raíz del proyecto


def find_project_root(start: Path | str | None = None) -> Path:
    """Localiza la raíz del proyecto.

    Usa TFM_EDGE_ROOT si está definida; si no, sube desde ``start`` (por defecto, el directorio de
    este fichero) hasta encontrar pyproject.toml.
    """
    entorno = os.environ.get(ENV_ROOT)
    if entorno:
        raiz = Path(entorno).expanduser()
        if not raiz.is_dir():
            raise ProjectRootNotFoundError(raiz)
        return raiz.resolve()

    inicio = Path(start).expanduser() if start is not None else Path(__file__).parent
    inicio = inicio.resolve()
    candidatos = [inicio, *inicio.parents] if inicio.is_dir() else list(inicio.parents)
    for candidato in candidatos:
        if (candidato / ROOT_MARKER).is_file():
            return candidato
    raise ProjectRootNotFoundError(inicio)


@dataclass(frozen=True)
class ProjectPaths:
    """Rutas absolutas del proyecto. Datos, modelos y resultados se pueden reubicar por variable de
    entorno.
    """

    root: Path
    configs: Path
    data: Path
    models: Path
    results: Path

    @property
    def data_raw(self) -> Path:
        """Dataset descargado tal cual, sin transformar."""
        return self.data / "raw"

    @property
    def data_processed(self) -> Path:
        """Dataset convertido al formato de entrenamiento y a COCO JSON."""
        return self.data / "processed"

    @property
    def data_calibration(self) -> Path:
        """Conjuntos de calibración para la cuantización PTQ."""
        return self.data / "calibration"

    @property
    def data_splits(self) -> Path:
        """Listas de ficheros de cada partición, en texto plano y auditables."""
        return self.data / "splits"

    @property
    def models_baselines(self) -> Path:
        """Modelos de referencia entrenados."""
        return self.models / "baselines"

    @property
    def models_compressed(self) -> Path:
        """Modelos resultantes de aplicar técnicas de compresión."""
        return self.models / "compressed"

    @property
    def models_exported(self) -> Path:
        """Artefactos exportados a los distintos runtimes."""
        return self.models / "exported"

    @property
    def results_registry(self) -> Path:
        """Los cuatro CSV del registro único de resultados (sección 16)."""
        return self.results / "registry"

    @property
    def results_raw(self) -> Path:
        """Volcados crudos: vectores de tiempos, predicciones COCO JSON, etc."""
        return self.results / "raw"

    @property
    def results_tables(self) -> Path:
        """Tablas generadas para la memoria (XLSX y LaTeX)."""
        return self.results / "tables"

    @property
    def results_figures(self) -> Path:
        """Figuras generadas para la memoria (PNG a 300 ppp y PDF vectorial)."""
        return self.results / "figures"

    @property
    def results_logs(self) -> Path:
        """Logs de ejecución de los scripts."""
        return self.results / "logs"

    def all_directories(self) -> tuple[Path, ...]:
        """Devuelve todos los directorios que el proyecto espera que existan."""
        return (
            self.configs,
            self.data,
            self.data_raw,
            self.data_processed,
            self.data_calibration,
            self.data_splits,
            self.models,
            self.models_baselines,
            self.models_compressed,
            self.models_exported,
            self.results,
            self.results_registry,
            self.results_raw,
            self.results_tables,
            self.results_figures,
            self.results_logs,
        )

    def create_all(self) -> None:
        """Crea los directorios del proyecto que falten."""
        for directorio in self.all_directories():
            ensure_dir(directorio)


@lru_cache(maxsize=1)
def get_project_paths() -> ProjectPaths:
    """Rutas del proyecto, cacheadas para que la raíz se detecte una sola vez por proceso."""
    raiz = find_project_root()

    def _dir_desde_entorno(variable: str, por_defecto: Path) -> Path:
        valor = os.environ.get(variable)
        return Path(valor).expanduser().resolve() if valor else por_defecto

    return ProjectPaths(
        root=raiz,
        configs=raiz / "configs",
        data=_dir_desde_entorno(ENV_DATA_DIR, raiz / "data"),
        models=_dir_desde_entorno(ENV_MODELS_DIR, raiz / "models"),
        results=_dir_desde_entorno(ENV_RESULTS_DIR, raiz / "results"),
    )


# Utilidades de escritura


def ensure_dir(path: Path | str) -> Path:
    """Crea un directorio y todos sus padres si no existen."""
    directorio = Path(path)
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


def ensure_parent_dir(path: Path | str) -> Path:
    """Crea el directorio padre de un fichero si no existe."""
    fichero = Path(path)
    fichero.parent.mkdir(parents=True, exist_ok=True)
    return fichero


# Validez de nombres de fichero


def validate_path_component(component: str) -> str:
    """Comprueba que una cadena sirve como nombre de fichero en Linux y en Windows."""
    if not component:
        raise InvalidPathComponentError(component, "está vacía")
    if len(component) > MAX_COMPONENT_LENGTH:
        raise InvalidPathComponentError(component, f"supera los {MAX_COMPONENT_LENGTH} caracteres")
    prohibidos = sorted(FORBIDDEN_CHARS.intersection(component))
    if prohibidos:
        raise InvalidPathComponentError(
            component, f"contiene caracteres prohibidos en Windows: {' '.join(prohibidos)}"
        )
    if any(ord(caracter) < 32 for caracter in component):
        raise InvalidPathComponentError(component, "contiene caracteres de control")
    if component[-1] in {".", " "}:
        raise InvalidPathComponentError(component, "termina en punto o en espacio")
    raiz_nombre = component.split(".")[0].upper()
    if raiz_nombre in WINDOWS_RESERVED_NAMES:
        raise InvalidPathComponentError(
            component, f"'{raiz_nombre}' es un nombre reservado en Windows"
        )
    return component


def slugify(text: str) -> str:
    """Pasa una cadena a minúsculas ASCII con guiones bajos. Se usa para las columnas
    ap_class_NN_nombre.
    """
    import unicodedata

    descompuesto = unicodedata.normalize("NFKD", text)
    sin_acentos = "".join(c for c in descompuesto if not unicodedata.combining(c))
    minusculas = sin_acentos.lower()
    con_guiones = re.sub(r"[^a-z0-9]+", "_", minusculas)
    return con_guiones.strip("_")


# Convención de experiment_id


def build_experiment_id(model_size: str, technique: str, variant: str, config_hash: str) -> str:
    """Construye el identificador ``{model_size}_{technique}_{variant}_{config_hash8}``."""
    if model_size not in VALID_MODEL_SIZES:
        raise InvalidExperimentIdError(
            f"{model_size}_{technique}_{variant}_...",
            f"'{model_size}' no es un tamaño válido (esperado: {', '.join(VALID_MODEL_SIZES)})",
        )
    if technique not in VALID_TECHNIQUES:
        raise InvalidExperimentIdError(
            f"{model_size}_{technique}_{variant}_...",
            f"'{technique}' no es una técnica válida (esperado: {', '.join(VALID_TECHNIQUES)})",
        )
    if not _VARIANT_RE.match(variant):
        raise InvalidExperimentIdError(
            f"{model_size}_{technique}_{variant}_...",
            f"la variante '{variant}' debe empezar por letra o dígito y contener "
            "solo letras, dígitos, guiones y guiones bajos",
        )
    hash_corto = config_hash[:CONFIG_HASH_LENGTH].lower()
    if not _HASH_RE.match(hash_corto):
        raise InvalidExperimentIdError(
            f"{model_size}_{technique}_{variant}_{config_hash}",
            f"el hash de configuración debe tener {CONFIG_HASH_LENGTH} caracteres "
            f"hexadecimales en minúscula, y se recibió '{config_hash}'",
        )
    experiment_id = f"{model_size}_{technique}_{variant}_{hash_corto}"
    return validate_path_component(experiment_id)


def parse_experiment_id(experiment_id: str) -> dict[str, str]:
    """Descompone un experiment_id en tamaño, técnica, variante y hash.

    El troceo es posicional, así que la variante puede llevar guiones bajos.
    """
    partes = experiment_id.split("_")
    if len(partes) < 4:
        raise InvalidExperimentIdError(
            experiment_id, "faltan campos (se esperan al menos cuatro separados por '_')"
        )
    model_size, technique, *medio, config_hash = partes
    variant = "_".join(medio)
    if model_size not in VALID_MODEL_SIZES:
        raise InvalidExperimentIdError(
            experiment_id, f"tamaño de modelo desconocido: '{model_size}'"
        )
    if technique not in VALID_TECHNIQUES:
        raise InvalidExperimentIdError(experiment_id, f"técnica desconocida: '{technique}'")
    if not _HASH_RE.match(config_hash):
        raise InvalidExperimentIdError(
            experiment_id,
            f"el último campo '{config_hash}' no es un hash hexadecimal de 8 caracteres",
        )
    return {
        "model_size": model_size,
        "technique": technique,
        "variant": variant,
        "config_hash": config_hash,
    }


# Rutas de artefactos derivadas del experiment_id


def experiment_weights_dir(experiment_id: str, paths: ProjectPaths | None = None) -> Path:
    """Directorio de pesos: models/baselines para los baselines y models/compressed para el
    resto."""
    componentes = parse_experiment_id(experiment_id)
    rutas = paths or get_project_paths()
    base = (
        rutas.models_baselines
        if componentes["technique"] == "baseline"
        else rutas.models_compressed
    )
    return base / experiment_id


def experiment_export_dir(experiment_id: str, paths: ProjectPaths | None = None) -> Path:
    """Directorio donde viven los artefactos exportados de un experimento."""
    parse_experiment_id(experiment_id)
    rutas = paths or get_project_paths()
    return rutas.models_exported / experiment_id


def experiment_raw_results_dir(experiment_id: str, paths: ProjectPaths | None = None) -> Path:
    """Directorio de volcados crudos (tiempos de latencia y predicciones COCO) de un experimento."""
    parse_experiment_id(experiment_id)
    rutas = paths or get_project_paths()
    return rutas.results_raw / experiment_id


def relative_to_root(path: Path | str, paths: ProjectPaths | None = None) -> str:
    """Ruta relativa a la raíz con separador ``/``. Si cae fuera del proyecto se devuelve
    absoluta."""
    rutas = paths or get_project_paths()
    absoluta = Path(path).expanduser().resolve()
    try:
        return absoluta.relative_to(rutas.root).as_posix()
    except ValueError:
        return absoluta.as_posix()


#: Claves de un volcado cuyo valor es una ruta. Se declaran a mano para no tocar por error una nota
#: o un mensaje que parezca una ruta.
DUMP_PATH_KEYS: frozenset[str] = frozenset(
    {
        "artifact",
        "staged_artifact",
        "validator_dir",
        "images_dir",
        "predictions",
        "recorded_dump",
        "calibration_set",
        "checkpoint",
        "ptq_report",
        "path",
        "source_path",
        "from_onnx",
        "ptq_source",
        "qat_source",
    }
)


def relativize_dump_paths(data: Any, paths: ProjectPaths | None = None) -> Any:
    """Convierte en relativas las rutas absolutas de un volcado antes de escribirlo.

    En memoria las rutas siguen absolutas porque se usan para abrir ficheros. Las de otra máquina se
    dejan como están.
    """
    rutas = paths or get_project_paths()
    if isinstance(data, dict):
        return {
            clave: (
                relative_to_root(valor, rutas)
                if clave in DUMP_PATH_KEYS and isinstance(valor, str) and Path(valor).is_absolute()
                else relativize_dump_paths(valor, rutas)
            )
            for clave, valor in data.items()
        }
    if isinstance(data, list):
        return [relativize_dump_paths(elemento, rutas) for elemento in data]
    return data


def relocate_under_results(path: Path | str, results: Path | str) -> Path | None:
    """Reubica bajo ``results`` de esta máquina una ruta grabada en otra.

    Se ancla en la primera aparición de ``results``. Devuelve None si la ruta no cuelga de results/.
    """
    partes = Path(path).parts
    if RESULTS_ANCHOR not in partes:
        return None
    desde = partes.index(RESULTS_ANCHOR)
    return Path(results).joinpath(*partes[desde + 1 :])
