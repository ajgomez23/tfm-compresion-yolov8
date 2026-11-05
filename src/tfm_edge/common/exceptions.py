"""Excepciones del proyecto.

Todas derivan de TFMEdgeError y su mensaje indica qué ha fallado y qué puede hacer el usuario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


class TFMEdgeError(Exception):
    """Excepción base. ``message`` describe el fallo y ``remedy`` cómo resolverlo."""

    def __init__(self, message: str, remedy: str | None = None) -> None:
        self.message = message
        self.remedy = remedy
        texto = message if remedy is None else f"{message}\n  Qué hacer: {remedy}"
        super().__init__(texto)


# Rutas y estructura del proyecto


class ProjectRootNotFoundError(TFMEdgeError):
    """No se ha podido localizar la raíz del proyecto."""

    def __init__(self, start: Path) -> None:
        super().__init__(
            f"No se encuentra la raíz del proyecto subiendo desde '{start}': "
            "no hay ningún pyproject.toml en esa ruta ni en sus directorios padre.",
            "Ejecuta el comando desde dentro del repositorio, o define la variable "
            "de entorno TFM_EDGE_ROOT con la ruta absoluta de la raíz.",
        )


class InvalidPathComponentError(TFMEdgeError):
    """Un identificador no es válido como nombre de fichero o de directorio."""

    def __init__(self, component: str, reason: str) -> None:
        super().__init__(
            f"El identificador '{component}' no es válido como nombre de fichero "
            f"o directorio: {reason}.",
            "Los identificadores deben ser portables entre Linux y Windows. "
            "Usa solo letras, dígitos, guiones y guiones bajos.",
        )


class InvalidExperimentIdError(TFMEdgeError):
    """Un experiment_id no sigue el formato ``{tamaño}_{técnica}_{variante}_{hash8}``."""

    def __init__(self, experiment_id: str, reason: str) -> None:
        super().__init__(
            f"El experiment_id '{experiment_id}' no sigue la convención: {reason}.",
            "El formato es {model_size}_{technique}_{variant}_{config_hash8}, "
            "por ejemplo 's_pruning_l1global-r50_9f8e7d6c'. Constrúyelo con "
            "tfm_edge.common.paths.build_experiment_id en lugar de a mano.",
        )


# Configuración


class ConfigError(TFMEdgeError):
    """Error genérico de configuración."""


class ConfigFileNotFoundError(ConfigError):
    """El fichero de configuración no existe."""

    def __init__(self, path: Path, referenced_from: Path | None = None) -> None:
        origen = "" if referenced_from is None else f" (referenciado desde '{referenced_from}')"
        super().__init__(
            f"No existe el fichero de configuración '{path}'{origen}.",
            "Comprueba la ruta. Las rutas de la clave '_base_' se resuelven "
            "respecto al directorio del fichero que las declara, no respecto al "
            "directorio de trabajo.",
        )


class ConfigInheritanceError(ConfigError):
    """La cadena de herencia de configuraciones es inválida (por ejemplo, cíclica)."""

    def __init__(self, message: str, chain: Iterable[Path] | None = None) -> None:
        cadena = ""
        if chain is not None:
            cadena = "\n  Cadena de herencia: " + ", ".join(str(p) for p in chain)
        super().__init__(
            message + cadena,
            "Revisa las claves '_base_'. Una configuración no puede heredar, "
            "ni directa ni indirectamente, de sí misma.",
        )


class ConfigSchemaError(ConfigError):
    """La configuración no valida contra su esquema."""

    def __init__(self, path: Path | None, details: str) -> None:
        origen = "la configuración" if path is None else f"la configuración '{path}'"
        super().__init__(
            f"Errores de validación en {origen}:\n{details}",
            "Corrige las claves indicadas. Recuerda que no se admiten claves "
            "desconocidas: si aparece 'Extra inputs are not permitted', esa clave "
            "no existe en el esquema (revisa si es una errata).",
        )


class ConfigOverrideError(ConfigError):
    """Un override de línea de comandos está mal formado o no es aplicable."""

    def __init__(self, override: str, reason: str) -> None:
        super().__init__(
            f"El override '{override}' no es válido: {reason}.",
            "El formato es clave.anidada=valor, por ejemplo "
            "--override training.epochs=150 pruning.ratio=0.6. El valor se "
            "interpreta como YAML, así que 'true' es booleano y '0.5' es float.",
        )


# Registro de resultados


class RegistryError(TFMEdgeError):
    """Error genérico del registro de resultados."""


class RegistrySchemaError(RegistryError):
    """El esquema de columnas de una tabla del registro no es el esperado."""

    def __init__(self, table: str, details: str) -> None:
        super().__init__(
            f"El esquema de la tabla '{table}' del registro no es válido: {details}",
            "El esquema está fijado en la sección 16 de la especificación y en "
            "docs/ESQUEMA_RESULTADOS.md. No añadas ni quites columnas a mano: "
            "escribe siempre a través de ExperimentRegistry.",
        )


class RegistryDuplicateError(RegistryError):
    """Fila duplicada con la política de fallar."""

    def __init__(self, table: str, key: str) -> None:
        super().__init__(
            f"Ya existe una fila con clave '{key}' en la tabla '{table}' del registro.",
            "Usa on_duplicate='skip' para no repetir el trabajo ya hecho, "
            "'overwrite' para sustituir la fila anterior, o 'version' para "
            "conservar ambas.",
        )


class RegistryIntegrityError(RegistryError):
    """El registro incumple alguna regla de integridad."""

    def __init__(self, details: str) -> None:
        super().__init__(
            f"El registro de resultados viola las reglas de integridad:\n{details}",
            "Ejecuta la validación completa del registro para ver el detalle. "
            "Un identificador huérfano suele significar que un experimento se "
            "interrumpió a medias: revísalo antes de usar el registro en el análisis.",
        )


class RegistryConflictError(RegistryError):
    """Se intenta rellenar una celda del registro que ya tiene otro valor.

    Se falla en vez de sobrescribir para no perder una medida ya inscrita.
    """

    def __init__(self, table: str, key: str, column: str, current: str, proposed: str) -> None:
        super().__init__(
            f"La fila '{key}' de la tabla '{table}' ya tiene "
            f"{column}={current!r} y se intentaba escribir {proposed!r}.",
            "Comprueba de dónde sale cada uno de los dos valores antes de "
            "elegir. Si el que ya está es el bueno, usa on_conflict='skip'; si "
            "de verdad hay que sustituirlo, on_conflict='overwrite', pero eso "
            "descarta una medida que estaba registrada.",
        )


class RegistryAmbiguousRowError(RegistryError):
    """Varias filas del registro responden a la misma consulta con valores distintos. Se falla en
    lugar de devolver la primera.
    """

    def __init__(self, query: str, candidates: str) -> None:
        super().__init__(
            f"La consulta al registro ({query}) devuelve varias filas que no "
            f"coinciden en su métrica:\n{candidates}",
            "Acota la consulta (normalmente con el dispositivo) o revisa por qué "
            "hay dos medidas distintas del mismo experimento con el mismo "
            "evaluador. No se elige una por ti: cuál de las dos vale es una "
            "decisión de quien mide, no del orden de las filas del CSV.",
        )


class AmbiguousDistillationControlError(RegistryError):
    """Hay varios controles sin destilar candidatos que no son la misma fila.

    Con un control por semilla, devolver el primero podría emparejar el piloto de una semilla con el
    control de otra.
    """

    def __init__(self, query: str, candidates: str) -> None:
        super().__init__(
            f"La búsqueda del control sin destilar ({query}) encuentra varios "
            f"candidatos distintos:\n{candidates}",
            "Elige uno con --control-experiment-id. No se toma el primero: con un "
            "control por semilla, la fila que va antes en el CSV es la de otra "
            "ejecución, y contrastar contra ella mide la diferencia entre "
            "semillas creyendo medir el efecto de destilar.",
        )


class RegistryChainError(RegistryError):
    """La copia de campaña del registro no es una continuación del registro local.

    Solo se adopta si el registro local es un prefijo exacto de ella; si no, se perderían filas.
    """


# Entorno, dependencias y hardware


class DependencyNotAvailableError(TFMEdgeError):
    """Falta una dependencia para la operación pedida.

    No se usa para los runtimes opcionales, que se registran como no_disponible.
    """

    def __init__(self, dependency: str, purpose: str, install_hint: str | None = None) -> None:
        super().__init__(
            f"Se necesita '{dependency}' para {purpose}, pero no está instalado.",
            install_hint or f"Instálalo con: pip install {dependency}",
        )


class HardwareNotAvailableError(TFMEdgeError):
    """Se ha solicitado hardware que no está presente."""

    def __init__(self, hardware: str, purpose: str) -> None:
        super().__init__(
            f"Se ha solicitado {hardware} para {purpose}, pero no está disponible "
            "en esta máquina.",
            "Ejecuta 'python scripts/00_setup_check.py' para ver qué hay "
            "realmente disponible. El proyecto NO simula medidas sobre hardware "
            "ausente: cambia de dispositivo o cambia la configuración.",
        )


class BinaryIncompatibilityError(TFMEdgeError):
    """Un paquete compilado no es compatible con la NumPy instalada (ver
    common/binary_compat.py)."""

    def __init__(self, details: str, remedy: str | None = None) -> None:
        super().__init__(
            "Hay paquetes compilados contra una versión de NumPy distinta de la "
            f"instalada:\n{details}",
            remedy
            or "Reinstala los paquetes afectados con "
            "'pip install --force-reinstall --no-cache-dir --no-deps <paquete>'. "
            "NO subas la versión de NumPy: está anclada por debajo de 2.0 a "
            "propósito.",
        )


class EnvironmentCaptureError(TFMEdgeError):
    """No se ha podido capturar la huella del entorno."""

    def __init__(self, details: str) -> None:
        super().__init__(
            f"No se ha podido capturar la huella del entorno: {details}",
            "Sin huella de entorno una medida no es interpretable, así que la "
            "ejecución se detiene aquí a propósito. Revisa el error concreto.",
        )
