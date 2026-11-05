"""Detección de incompatibilidades binarias con la API C de NumPy.

numpy está anclado por debajo de 2.0 y en Colab viene preinstalada la 2.x; al instalar
requirements.txt, pip baja NumPy pero no recompila lo que ya estaba, y pandas falla con "numpy.dtype
size changed". Un import correcto no basta para detectarlo, así que cada paquete se ejercita en su
propio subproceso, que además sobrevive a una caída dura del intérprete.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

from tfm_edge.common.exceptions import BinaryIncompatibilityError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

#: Mensajes que delatan un desajuste de ABI con NumPy, en minúsculas. Se buscan en la excepción y en
#: la salida de error, porque una caída dura no deja excepción.
NUMPY_ABI_SIGNATURES: tuple[str, ...] = (
    "numpy.dtype size changed",
    "numpy.ufunc size changed",
    "binary incompatibility",
    "compiled using numpy 1.x cannot be run in numpy 2",
    "numpy.core.multiarray failed to import",
    "_array_api not found",
)

#: Paquetes que usan la API C de NumPy: (módulo, distribución de pip, es_crítico). Si falla uno
#: crítico, falla la comprobación del entorno.
NUMPY_C_API_CONSUMERS: tuple[tuple[str, str, bool], ...] = (
    ("numpy", "numpy", True),
    ("pandas", "pandas", True),
    ("cv2", "opencv-python", False),
    ("pycocotools", "pycocotools", False),
    ("torch", "torch", False),
)

#: Segundos de espera por subproceso; importar torch es lo más lento.
_TIMEOUT_SUBPROCESO = 120

# Programa sonda. Se ejecuta con "python -c", recibe el módulo como argumento y escribe una línea
# JSON. No importa tfm_edge porque se usa en Colab antes de instalar el paquete.
_PROGRAMA_SONDA = r'''
import json, os, sys, tempfile


def _version_instalada(modulo, distribucion):
    """Version del paquete: su atributo __version__ o, si no lo tiene, pip."""
    import importlib

    version = getattr(importlib.import_module(modulo), "__version__", None)
    if version:
        return version
    try:
        from importlib.metadata import version as version_de_metadatos

        return version_de_metadatos(distribucion)
    except Exception:
        return "version desconocida"


def _ejercicio_numpy():
    """Comprueba la coherencia interna de NumPy consigo misma."""
    import numpy as np

    matriz = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert float(matriz.sum()) == 66.0, matriz.sum()
    assert np.dtype(np.float64).itemsize == 8, np.dtype(np.float64).itemsize
    return np.__version__


def _ejercicio_pandas():
    """Ida y vuelta por CSV: es exactamente donde falló en Colab."""
    import pandas as pd

    marco = pd.DataFrame({"a": [1, 2, 3], "b": [0.5, 1.5, 2.5]})
    with tempfile.TemporaryDirectory() as temporal:
        ruta = os.path.join(temporal, "sonda.csv")
        marco.to_csv(ruta, index=False)
        leido = pd.read_csv(ruta)
    assert leido.shape == (3, 2), leido.shape
    assert abs(float(leido["b"].sum()) - 4.5) < 1e-9, leido["b"].sum()
    return pd.__version__


def _ejercicio_cv2():
    """Redimensionar cruza la frontera C con un array de NumPy en ambos sentidos."""
    import cv2
    import numpy as np

    salida = cv2.resize(np.zeros((8, 8, 3), dtype=np.uint8), (4, 4))
    assert salida.shape == (4, 4, 3), salida.shape
    return cv2.__version__


def _ejercicio_pycocotools():
    """Codificar una mascara RLE es codigo C puro sobre un array de NumPy."""
    import numpy as np
    from pycocotools import mask as utilidades_mascara

    mascara = np.asfortranarray(np.zeros((4, 4), dtype=np.uint8))
    mascara[0, 0] = 1
    codificada = utilidades_mascara.encode(mascara)
    assert "counts" in codificada and "size" in codificada, sorted(codificada)
    return _version_instalada("pycocotools", "pycocotools")


def _ejercicio_torch():
    """from_numpy y .numpy() comparten memoria: si la ABI no cuadra, se ve aqui."""
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.arange(6, dtype=np.float32))
    assert float(tensor.sum().item()) == 15.0, tensor.sum()
    devuelto = tensor.numpy()
    assert devuelto.shape == (6,), devuelto.shape
    return torch.__version__


_EJERCICIOS = {
    "numpy": _ejercicio_numpy,
    "pandas": _ejercicio_pandas,
    "cv2": _ejercicio_cv2,
    "pycocotools": _ejercicio_pycocotools,
    "torch": _ejercicio_torch,
}

_nombre = sys.argv[1]
if _nombre not in _EJERCICIOS:
    print(json.dumps({"estado": "sin_ejercicio", "clase": "", "error": _nombre}))
    raise SystemExit(0)

try:
    _version = _EJERCICIOS[_nombre]()
except BaseException as _error:  # noqa: BLE001 - hay que clasificarlo, no propagarlo
    print(
        json.dumps(
            {
                "estado": "fallo",
                "clase": type(_error).__name__,
                "error": "{}: {}".format(type(_error).__name__, _error),
            }
        )
    )
    raise SystemExit(0)

print(json.dumps({"estado": "ok", "clase": "", "version": str(_version)}))
'''


@dataclass(frozen=True)
class BinaryCompatResult:
    """Resultado de ejercitar un paquete contra la NumPy instalada."""

    package: str
    module: str
    installed: bool
    healthy: bool
    abi_mismatch: bool
    detail: str
    critical: bool


def _parece_desajuste_de_abi(texto: str) -> bool:
    """Indica si un texto contiene la firma de un desajuste de ABI con NumPy."""
    minusculas = texto.lower()
    return any(firma in minusculas for firma in NUMPY_ABI_SIGNATURES)


def _ejercitar(modulo: str, distribucion: str, critico: bool, timeout: int) -> BinaryCompatResult:
    """Ejercita un paquete en un subproceso propio y clasifica el resultado."""
    # Se ejecuta en un directorio vacío para que un numpy.py suelto no interfiera.
    with tempfile.TemporaryDirectory() as neutral:
        try:
            proceso = subprocess.run(
                [sys.executable, "-c", _PROGRAMA_SONDA, modulo],
                capture_output=True,
                text=True,
                cwd=neutral,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return BinaryCompatResult(
                package=distribucion,
                module=modulo,
                installed=True,
                healthy=False,
                abi_mismatch=False,
                detail=(
                    f"la sonda no ha terminado en {timeout} s; el paquete se ha "
                    "quedado colgado al importarse o al ejecutarse"
                ),
                critical=critico,
            )

    salida = proceso.stdout.strip().splitlines()
    informe: dict[str, str] = {}
    if salida:
        try:
            informe = json.loads(salida[-1])
        except json.JSONDecodeError:
            informe = {}

    # Sin JSON de vuelta, el intérprete murió antes de responder.
    if not informe:
        error = proceso.stderr.strip()
        cola = error.splitlines()[-3:] if error else []
        return BinaryCompatResult(
            package=distribucion,
            module=modulo,
            installed=True,
            healthy=False,
            abi_mismatch=_parece_desajuste_de_abi(error),
            detail=(
                f"la sonda ha abortado el intérprete (código {proceso.returncode}) "
                f"sin llegar a informar. Es típico de una incompatibilidad binaria: "
                f"una caída dura no lanza excepción. "
                + (f"Últimas líneas: {' | '.join(cola)}" if cola else "Sin salida de error.")
            ),
            critical=critico,
        )

    if informe.get("estado") == "ok":
        return BinaryCompatResult(
            package=distribucion,
            module=modulo,
            installed=True,
            healthy=True,
            abi_mismatch=False,
            detail=f"{informe.get('version', 'versión desconocida')} (sonda superada)",
            critical=critico,
        )

    mensaje = informe.get("error", "sin detalle")
    # Primero se mira la ABI, porque un desajuste suele aparecer como ImportError.
    if _parece_desajuste_de_abi(mensaje):
        return BinaryCompatResult(
            package=distribucion,
            module=modulo,
            installed=True,
            healthy=False,
            abi_mismatch=True,
            detail=mensaje,
            critical=critico,
        )
    if informe.get("clase") == "ModuleNotFoundError":
        return BinaryCompatResult(
            package=distribucion,
            module=modulo,
            installed=False,
            healthy=False,
            abi_mismatch=False,
            detail=f"no instalado ({mensaje})",
            critical=critico,
        )
    return BinaryCompatResult(
        package=distribucion,
        module=modulo,
        installed=True,
        healthy=False,
        abi_mismatch=False,
        detail=mensaje,
        critical=critico,
    )


def check_numpy_binary_compat(
    packages: Iterable[tuple[str, str, bool]] | None = None,
    timeout: int = _TIMEOUT_SUBPROCESO,
) -> list[BinaryCompatResult]:
    """Ejercita cada paquete que usa la API C de NumPy en su propio subproceso."""
    seleccion = tuple(NUMPY_C_API_CONSUMERS if packages is None else packages)
    resultados: list[BinaryCompatResult] = []
    for modulo, distribucion, critico in seleccion:
        resultado = _ejercitar(modulo, distribucion, critico, timeout)
        if resultado.abi_mismatch:
            logger.error(
                "Incompatibilidad binaria con NumPy en '%s': %s",
                resultado.package,
                resultado.detail,
            )
        elif resultado.installed and not resultado.healthy:
            logger.warning(
                "El paquete '%s' está instalado pero su sonda falla: %s",
                resultado.package,
                resultado.detail,
            )
        resultados.append(resultado)
    return resultados


def format_binary_compat_report(results: Sequence[BinaryCompatResult]) -> str:
    """Formatea los resultados como texto, para imprimirlos en el cuaderno de Colab."""
    lineas = ["Compatibilidad binaria con la API C de NumPy"]
    lineas.append("-" * len(lineas[0]))
    for resultado in results:
        if resultado.healthy:
            marca = "[ OK ]"
        elif resultado.abi_mismatch:
            marca = "[ ABI]"
        elif not resultado.installed:
            marca = "[ -- ]"
        else:
            marca = "[FALLO]"
        lineas.append(f"  {marca} {resultado.package:16s} {resultado.detail}")
    rotos = [r for r in results if r.abi_mismatch]
    if rotos:
        lineas.append("")
        lineas.append(_remedio(rotos))
    return "\n".join(lineas)


def _remedio(rotos: Sequence[BinaryCompatResult]) -> str:
    """Texto con el comando para reparar los paquetes con la ABI rota."""
    nombres = " ".join(sorted({r.package for r in rotos}))
    return (
        f"Hay {len(rotos)} paquete(s) compilado(s) contra una versión de NumPy "
        f"distinta de la instalada. Reinstálalos SIN caché y sin tocar el anclaje "
        f"de NumPy:\n"
        f"    pip install --force-reinstall --no-cache-dir --no-deps {nombres}\n"
        f"y vuelve a ejecutar esta comprobación. En Colab, además, hay que "
        f"REINICIAR EL ENTORNO DE EJECUCIÓN después de reinstalar: el NumPy "
        f"anterior sigue cargado en memoria y el fallo reaparecería igual."
    )


def assert_numpy_binary_compat(
    packages: Iterable[tuple[str, str, bool]] | None = None,
) -> list[BinaryCompatResult]:
    """Lanza BinaryIncompatibilityError si algún paquete crítico falla. Pensada para el inicio de un
    script o cuaderno.
    """
    resultados = check_numpy_binary_compat(packages)
    fallos = [r for r in resultados if r.critical and r.installed and not r.healthy]
    if not fallos:
        return resultados
    detalle = "\n".join(f"  - {r.package}: {r.detail}" for r in fallos)
    rotos = [r for r in fallos if r.abi_mismatch]
    raise BinaryIncompatibilityError(detalle, _remedio(rotos) if rotos else None)
