"""Huella de hardware y software de la máquina.

Cada fila del registro apunta a un environment_id que identifica hardware, sistema, bibliotecas y
commit. Lo que no se puede obtener queda como cadena vacía, nunca como cero.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from tfm_edge.common.exceptions import EnvironmentCaptureError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Columnas de environments.csv en orden.
ENVIRONMENT_COLUMNS: tuple[str, ...] = (
    "environment_id",
    "device_id",
    "os",
    "os_version",
    "architecture",
    "cpu_model",
    "cpu_cores_physical",
    "cpu_cores_logical",
    "ram_total_gb",
    "gpu_model",
    "gpu_vram_gb",
    "driver_version",
    "cuda_version",
    "cudnn_version",
    "python_version",
    "torch_version",
    "ultralytics_version",
    "onnxruntime_version",
    "tensorrt_version",
    "openvino_version",
    "tflite_version",
    "git_commit",
    "git_dirty",
    "execution_context",
    "timestamp",
)

#: Columnas añadidas al esquema base. Los tests comprueban que las columnas reales son exactamente
#: las del esquema más estas.
ENVIRONMENT_SCHEMA_EXTENSIONS: tuple[str, ...] = ("execution_context",)

#: Campos que no entran en el environment_id. El timestamp cambia en cada captura.
_CAMPOS_EXCLUIDOS_DEL_HASH = frozenset({"environment_id", "timestamp"})

#: Candidatos para la versión de TFLite: tflite-runtime en ARM, tensorflow en x86 y ai-edge-litert
#: en las versiones recientes de Ultralytics.
_PAQUETES_TFLITE = ("tflite-runtime", "ai-edge-litert", "tensorflow")

_TIMEOUT_SUBPROCESO = 15


def _version_de_paquete(nombre: str) -> str:
    """Devuelve la versión instalada de un paquete, o cadena vacía si no está."""
    try:
        return package_version(nombre)
    except PackageNotFoundError:
        return ""


def _primera_version_disponible(nombres: tuple[str, ...]) -> str:
    """Versión del primer paquete instalado de la lista, como ``"paquete==version"``, o ``""``."""
    for nombre in nombres:
        version = _version_de_paquete(nombre)
        if version:
            return f"{nombre}=={version}"
    return ""


def _ejecutar(comando: list[str]) -> str:
    """Ejecuta un comando y devuelve su salida, o ``""`` si falla o no existe."""
    salida = _ejecutar_o_none(comando)
    return "" if salida is None else salida


def _ejecutar_o_none(comando: list[str]) -> str | None:
    """Como _ejecutar, pero devuelve None si el comando falla.

    Hace falta cuando una salida vacía es una respuesta válida, como en ``git status --porcelain``.
    """
    try:
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SUBPROCESO,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.debug("No se ha podido ejecutar %s: %s", " ".join(comando), error)
        return None
    if resultado.returncode != 0:
        logger.debug("El comando %s devolvió código %d", " ".join(comando), resultado.returncode)
        return None
    return resultado.stdout.strip()


def _info_cpu() -> dict[str, Any]:
    """Modelo de CPU y número de núcleos físicos y lógicos."""
    datos: dict[str, Any] = {"cpu_model": "", "cpu_cores_physical": "", "cpu_cores_logical": ""}

    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        datos["cpu_model"] = str(info.get("brand_raw", "")).strip()
    except Exception as error:  # pragma: no cover - depende del sistema
        logger.debug("py-cpuinfo no ha podido leer el modelo de CPU: %s", error)

    if not datos["cpu_model"]:
        # platform.processor() da poco en Linux, pero es mejor que nada.
        datos["cpu_model"] = platform.processor().strip()

    try:
        import psutil

        fisicos = psutil.cpu_count(logical=False)
        logicos = psutil.cpu_count(logical=True)
        datos["cpu_cores_physical"] = "" if fisicos is None else int(fisicos)
        datos["cpu_cores_logical"] = "" if logicos is None else int(logicos)
    except Exception as error:  # pragma: no cover - depende del sistema
        logger.debug("psutil no ha podido leer el número de núcleos: %s", error)

    return datos


def _info_ram() -> dict[str, Any]:
    """RAM total en GiB con dos decimales, o ``""``."""
    try:
        import psutil

        total_bytes = psutil.virtual_memory().total
        return {"ram_total_gb": round(total_bytes / (1024**3), 2)}
    except Exception as error:  # pragma: no cover - depende del sistema
        logger.debug("psutil no ha podido leer la RAM total: %s", error)
        return {"ram_total_gb": ""}


def _info_gpu() -> dict[str, Any]:
    """Modelo, VRAM y driver de las GPU NVIDIA según nvidia-smi.

    Con varias GPU los valores se concatenan con ``"; "``.
    """
    vacio = {"gpu_model": "", "gpu_vram_gb": "", "driver_version": ""}
    salida = _ejecutar(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not salida:
        return vacio

    modelos: list[str] = []
    vram: list[str] = []
    drivers: list[str] = []
    for linea in salida.splitlines():
        campos = [campo.strip() for campo in linea.split(",")]
        if len(campos) != 3:
            logger.debug("Línea inesperada de nvidia-smi: %r", linea)
            continue
        nombre, memoria_mib, driver = campos
        modelos.append(nombre)
        try:
            vram.append(f"{int(memoria_mib) / 1024:.2f}")
        except ValueError:
            logger.debug("No se ha podido interpretar la VRAM %r", memoria_mib)
        drivers.append(driver)

    if not modelos:
        return vacio
    return {
        "gpu_model": "; ".join(modelos),
        "gpu_vram_gb": "; ".join(vram) if vram else "",
        "driver_version": "; ".join(sorted(set(drivers))),
    }


def _info_cuda() -> dict[str, Any]:
    """Versiones de CUDA y cuDNN con las que se compiló PyTorch (no las del toolkit del sistema)."""
    try:
        import torch
    except ImportError:
        return {"cuda_version": "", "cudnn_version": ""}

    cuda = torch.version.cuda or ""
    cudnn = ""
    try:
        if torch.backends.cudnn.is_available():
            numero = torch.backends.cudnn.version()
            cudnn = "" if numero is None else str(numero)
    except Exception as error:  # pragma: no cover - depende de la instalación
        logger.debug("No se ha podido leer la versión de cuDNN: %s", error)
    return {"cuda_version": cuda, "cudnn_version": cudnn}


def _info_git(repo_dir: Path | None = None) -> dict[str, Any]:
    """Commit actual y si el código tiene cambios sin confirmar.

    ``git_dirty`` ignora results/, porque los scripts escriben ahí y si no una misma sesión
    cambiaría de environment_id a mitad. Queda vacío si no se ha podido consultar el estado.
    """
    base = ["git"] if repo_dir is None else ["git", "-C", str(repo_dir)]

    if _ejecutar([*base, "rev-parse", "--is-inside-work-tree"]) != "true":
        logger.debug("No se ha detectado un repositorio git; el commit quedará vacío.")
        return {"git_commit": "", "git_dirty": ""}

    commit = _ejecutar([*base, "rev-parse", "HEAD"])
    # ':(top)' ancla la ruta a la raíz del repositorio aunque el proceso corra en un subdirectorio.
    estado = _ejecutar_o_none(
        [*base, "status", "--porcelain", "--", ":(top)", ":(top,exclude)results/"]
    )
    if estado is None:
        logger.warning(
            "No se ha podido consultar git status; git_dirty quedará vacío en vez de "
            "declarar limpio un árbol que no se ha mirado."
        )
        return {"git_commit": commit, "git_dirty": ""}
    return {
        "git_commit": commit,
        # Con el árbol sucio el resultado no corresponde exactamente a ningún commit.
        "git_dirty": "true" if estado else "false",
    }


def detect_execution_context() -> str:
    """Detecta el tipo de entorno: google_colab, kaggle, sagemaker, wsl2, container o local.

    Si se cumplen varias condiciones se unen con ``+``. En Colab la GPU cambia entre sesiones, y
    ningún entorno compartido sirve para medir latencia.
    """
    etiquetas: list[str] = []

    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        etiquetas.append("google_colab")
    else:
        try:
            import google.colab  # noqa: F401

            etiquetas.append("google_colab")
        except ImportError:
            pass

    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        etiquetas.append("kaggle")
    if "SM_TRAINING_ENV" in os.environ or "SAGEMAKER_JOB_NAME" in os.environ:
        etiquetas.append("sagemaker")

    if platform.system() == "Linux":
        try:
            version_kernel = platform.uname().release.lower()
            if "microsoft" in version_kernel or "wsl" in version_kernel:
                etiquetas.append("wsl2")
        except Exception:  # pragma: no cover - depende del sistema
            pass
        if Path("/.dockerenv").exists():
            etiquetas.append("container")

    return "+".join(etiquetas) if etiquetas else "local"


def capture_environment(device_id: str, repo_dir: Path | None = None) -> dict[str, Any]:
    """Captura la huella completa de hardware y software.

    ``device_id`` forma parte de la identidad: la misma máquina forzada a CPU es otro entorno. Falla
    si no se obtienen al menos el sistema operativo, la arquitectura y la versión de Python.
    """
    try:
        huella: dict[str, Any] = {
            "device_id": device_id,
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        }
        if not huella["os"] or not huella["python_version"]:
            raise EnvironmentCaptureError(
                "platform no ha devuelto el sistema operativo o la versión de Python."
            )
    except EnvironmentCaptureError:
        raise
    except Exception as error:  # pragma: no cover - fallo muy improbable
        raise EnvironmentCaptureError(f"error inesperado al leer la plataforma: {error}") from error

    huella.update(_info_cpu())
    huella.update(_info_ram())
    huella.update(_info_gpu())
    huella.update(_info_cuda())
    huella.update(_info_git(repo_dir))

    huella["torch_version"] = _version_de_paquete("torch")
    huella["ultralytics_version"] = _version_de_paquete("ultralytics")
    huella["onnxruntime_version"] = _primera_version_disponible(("onnxruntime-gpu", "onnxruntime"))
    huella["tensorrt_version"] = _primera_version_disponible(("tensorrt", "tensorrt-libs"))
    huella["openvino_version"] = _version_de_paquete("openvino")
    huella["tflite_version"] = _primera_version_disponible(_PAQUETES_TFLITE)

    huella["execution_context"] = detect_execution_context()
    if "google_colab" in huella["execution_context"]:
        logger.warning(
            "Ejecución en Google Colab (GPU: %s). Sirve para entrenar, no para medir latencia.",
            huella.get("gpu_model") or "desconocida",
        )

    huella["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    huella["environment_id"] = compute_environment_id(huella)

    return {columna: huella.get(columna, "") for columna in ENVIRONMENT_COLUMNS}


def compute_environment_id(fingerprint: dict[str, Any], length: int = 12) -> str:
    """Identificador estable de un entorno; ignora environment_id y timestamp."""
    from tfm_edge.common.config import compute_config_hash

    relevante = {
        clave: valor
        for clave, valor in sorted(fingerprint.items())
        if clave not in _CAMPOS_EXCLUIDOS_DEL_HASH
    }
    return compute_config_hash(relevante, length=length)


def persist_environment(fingerprint: dict[str, Any], registry_dir: Path | str | None = None) -> str:
    """Añade la huella a environments.csv si su environment_id no está ya, y devuelve el
    identificador.
    """
    from tfm_edge.common.registry import ExperimentRegistry

    registro = ExperimentRegistry(registry_dir)
    registro.append_environment(fingerprint)
    return str(fingerprint["environment_id"])


def summarize_environment(fingerprint: dict[str, Any]) -> str:
    """Resumen legible de la huella."""

    def _mostrar(valor: Any) -> str:
        return str(valor) if valor not in (None, "") else "(no disponible)"

    lineas = [
        f"Entorno {_mostrar(fingerprint.get('environment_id'))} "
        f"(dispositivo: {_mostrar(fingerprint.get('device_id'))})",
        f"  Sistema : {_mostrar(fingerprint.get('os'))} "
        f"{_mostrar(fingerprint.get('os_version'))} ({_mostrar(fingerprint.get('architecture'))})",
        f"  CPU     : {_mostrar(fingerprint.get('cpu_model'))} "
        f"({_mostrar(fingerprint.get('cpu_cores_physical'))} núcleos físicos, "
        f"{_mostrar(fingerprint.get('cpu_cores_logical'))} lógicos)",
        f"  RAM     : {_mostrar(fingerprint.get('ram_total_gb'))} GB",
        f"  GPU     : {_mostrar(fingerprint.get('gpu_model'))} "
        f"({_mostrar(fingerprint.get('gpu_vram_gb'))} GB VRAM, "
        f"driver {_mostrar(fingerprint.get('driver_version'))})",
        f"  CUDA    : {_mostrar(fingerprint.get('cuda_version'))} "
        f"(cuDNN {_mostrar(fingerprint.get('cudnn_version'))})",
        f"  Python  : {_mostrar(fingerprint.get('python_version'))}",
        f"  Git     : {_mostrar(fingerprint.get('git_commit'))[:12]} "
        f"(árbol sucio: {_mostrar(fingerprint.get('git_dirty'))})",
    ]
    return "\n".join(lineas)


def python_executable_info() -> dict[str, str]:
    """Ejecutable, versión y prefijo del intérprete actual, para detectar un entorno virtual
    equivocado.
    """
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "in_virtualenv": str(sys.prefix != getattr(sys, "base_prefix", sys.prefix)),
    }
