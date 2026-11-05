#!/usr/bin/env python
"""Comprueba el entorno e informa de qué componentes están disponibles.

No instala ni modifica nada salvo, si se pide, crear los directorios del proyecto. Cada componente
se comprueba importándolo o ejecutándolo. Los runtimes de inferencia son opcionales y su ausencia
solo limita qué se puede medir en esta máquina.

Códigos de salida: 0 si están todos los componentes obligatorios, 1 si falta alguno.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# El proyecto puede no estar instalado todavía, así que se añade src/ al path.
_RAIZ = Path(__file__).resolve().parent.parent
_SRC = _RAIZ / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tfm_edge.common.environment import (  # noqa: E402
    capture_environment,
    detect_execution_context,
    python_executable_info,
    summarize_environment,
)
from tfm_edge.common.logging_utils import get_logger, setup_logging  # noqa: E402
from tfm_edge.common.paths import get_project_paths  # noqa: E402

logger = get_logger(__name__)

#: Versiones de Python soportadas. Por encima de 3.12 los pines de requirements.txt no tienen ruedas
#: (numpy 1.26.4 y pandas 2.2.2 llegan hasta cp312).
PYTHON_SOPORTADO_MIN = (3, 10)
PYTHON_SOPORTADO_MAX = (3, 12)

#: Dependencias imprescindibles, como pares (módulo importable, nombre de distribución para pip).
DEPENDENCIAS_OBLIGATORIAS: tuple[tuple[str, str], ...] = (
    ("pydantic", "pydantic"),
    ("yaml", "PyYAML"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("psutil", "psutil"),
    ("cpuinfo", "py-cpuinfo"),
)

#: Dependencias de entrenamiento y compresión. Sin ellas no se pueden producir modelos, pero sí
#: analizar un registro existente.
DEPENDENCIAS_ENTRENAMIENTO: tuple[tuple[str, str], ...] = (
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("ultralytics", "ultralytics"),
    ("torch_pruning", "torch-pruning"),
    ("pycocotools", "pycocotools"),
)

#: Runtimes de exportación e inferencia, todos opcionales. Si faltan se registra "no_disponible".
RUNTIMES_OPCIONALES: tuple[tuple[str, str], ...] = (
    ("onnx", "onnx"),
    ("onnxruntime", "onnxruntime"),
    ("openvino", "openvino"),
    ("nncf", "nncf"),
    ("tensorrt", "tensorrt"),
    ("tflite_runtime", "tflite-runtime"),
    ("ai_edge_litert", "ai-edge-litert"),
    ("tensorflow", "tensorflow"),
)

#: Dependencias de figuras y tablas.
DEPENDENCIAS_INFORMES: tuple[tuple[str, str], ...] = (
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("openpyxl", "openpyxl"),
)


@dataclass
class Comprobacion:
    """Resultado de una comprobación individual."""

    nombre: str
    disponible: bool
    detalle: str
    obligatorio: bool = False


def comprobar_modulo(modulo: str, distribucion: str, obligatorio: bool = False) -> Comprobacion:
    """Comprueba si un módulo se puede importar y devuelve su versión."""
    try:
        importado = importlib.import_module(modulo)
    except ImportError as error:
        return Comprobacion(
            nombre=distribucion,
            disponible=False,
            detalle=f"no instalado ({error.__class__.__name__})",
            obligatorio=obligatorio,
        )
    except Exception as error:  # pragma: no cover - importaciones que fallan por otra causa
        return Comprobacion(
            nombre=distribucion,
            disponible=False,
            detalle=f"instalado pero no se puede importar: {error}",
            obligatorio=obligatorio,
        )
    version = getattr(importado, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as metadata_version

            version = metadata_version(distribucion)
        except Exception:  # pragma: no cover - paquete sin metadatos
            version = "versión desconocida"
    return Comprobacion(
        nombre=distribucion, disponible=True, detalle=str(version), obligatorio=obligatorio
    )


def comprobar_compatibilidad_binaria() -> list[Comprobacion]:
    """Comprueba que los paquetes compilados funcionan con la NumPy instalada.

    Un desajuste de ABI puede pasar el import y fallar al usar el código C, así que cada paquete se
    ejercita en un subproceso (ver ``tfm_edge.common.binary_compat``).
    """
    from tfm_edge.common.binary_compat import check_numpy_binary_compat

    comprobaciones: list[Comprobacion] = []
    for resultado in check_numpy_binary_compat():
        if resultado.abi_mismatch:
            detalle = f"INCOMPATIBILIDAD BINARIA: {resultado.detail}"
        else:
            detalle = resultado.detail
        comprobaciones.append(
            Comprobacion(
                nombre=resultado.package,
                disponible=resultado.healthy,
                detalle=detalle,
                # Un paquete ausente ya aparece en la sección de dependencias.
                obligatorio=resultado.critical and resultado.installed,
            )
        )
    return comprobaciones


def comprobar_python() -> Comprobacion:
    """Comprueba que la versión de Python está dentro del rango soportado."""
    actual = sys.version_info[:2]
    banda = (
        f"{PYTHON_SOPORTADO_MIN[0]}.{PYTHON_SOPORTADO_MIN[1]}"
        f"-{PYTHON_SOPORTADO_MAX[0]}.{PYTHON_SOPORTADO_MAX[1]}"
    )
    if actual < PYTHON_SOPORTADO_MIN:
        motivo = "por debajo del mínimo soportado"
    elif actual > PYTHON_SOPORTADO_MAX:
        motivo = (
            "por encima del máximo soportado: los pines de requirements.txt no "
            "tienen rueda para esta versión y pip no podrá resolver el entorno"
        )
    else:
        motivo = ""
    detalle = f"{platform.python_version()} (banda soportada: {banda})"
    if motivo:
        detalle = f"{detalle}; {motivo}"
    return Comprobacion(
        nombre="Python",
        disponible=not motivo,
        detalle=detalle,
        obligatorio=True,
    )


def comprobar_cuda() -> list[Comprobacion]:
    """Comprueba CUDA ejecutando una operación real en la GPU, no solo
    ``torch.cuda.is_available()``.
    """
    try:
        import torch
    except ImportError:
        return [
            Comprobacion(
                "CUDA (vía PyTorch)", False, "PyTorch no está instalado; no se puede comprobar"
            )
        ]

    resultados: list[Comprobacion] = []
    disponible = bool(torch.cuda.is_available())
    resultados.append(
        Comprobacion(
            "CUDA disponible",
            disponible,
            (
                (f"sí, {torch.cuda.device_count()} dispositivo(s), " f"CUDA {torch.version.cuda}")
                if disponible
                else "no (se entrenará y medirá en CPU)"
            ),
        )
    )
    if not disponible:
        return resultados

    for indice in range(torch.cuda.device_count()):
        propiedades = torch.cuda.get_device_properties(indice)
        resultados.append(
            Comprobacion(
                f"GPU {indice}",
                True,
                (
                    f"{propiedades.name}, "
                    f"{propiedades.total_memory / (1024**3):.2f} GB, "
                    f"capacidad de cómputo {propiedades.major}.{propiedades.minor}"
                ),
            )
        )

    try:
        tensor = torch.randn(64, 64, device="cuda")
        _ = (tensor @ tensor).sum().item()
        torch.cuda.synchronize()
        resultados.append(Comprobacion("Operación real en GPU", True, "ejecutada correctamente"))
    except Exception as error:
        resultados.append(
            Comprobacion(
                "Operación real en GPU",
                False,
                f"CUDA se declara disponible pero la operación falla: {error}",
            )
        )
    return resultados


def comprobar_estructura(crear: bool) -> list[Comprobacion]:
    """Comprueba la estructura de directorios del proyecto y crea los que falten si ``crear`` es
    True.
    """
    resultados: list[Comprobacion] = []
    rutas = get_project_paths()
    resultados.append(Comprobacion("Raíz del proyecto", True, str(rutas.root), obligatorio=True))

    if crear:
        rutas.create_all()

    faltantes = [d for d in rutas.all_directories() if not d.is_dir()]
    resultados.append(
        Comprobacion(
            "Directorios del proyecto",
            not faltantes,
            (
                "todos presentes"
                if not faltantes
                else (
                    f"faltan {len(faltantes)} "
                    f"(ejecuta con --create-dirs): {', '.join(d.name for d in faltantes[:5])}"
                )
            ),
        )
    )

    prueba = rutas.results / ".escritura_de_prueba"
    try:
        prueba.parent.mkdir(parents=True, exist_ok=True)
        prueba.write_text("ok", encoding="utf-8")
        prueba.unlink()
        escribible = True
        detalle = f"se puede escribir en {rutas.results}"
    except OSError as error:
        escribible = False
        detalle = f"no se puede escribir en {rutas.results}: {error}"
    resultados.append(Comprobacion("Permiso de escritura", escribible, detalle, obligatorio=True))

    return resultados


def comprobar_espacio_en_disco(minimo_gb: float) -> list[Comprobacion]:
    """Comprueba el espacio libre en cada volumen que usa el proyecto.

    ``data``, ``models`` y ``results`` pueden estar en discos distintos (``TFM_EDGE_DATA_DIR``,
    ``TFM_EDGE_MODELS_DIR``), así que se mide cada volumen una vez.
    """
    rutas = get_project_paths()
    # Se agrupan los directorios por volumen para no repetir la misma medida.
    por_volumen: dict[str, list[str]] = {}
    for etiqueta, directorio in (
        ("raíz", rutas.root),
        ("data", rutas.data),
        ("models", rutas.models),
        ("results", rutas.results),
    ):
        # Si el directorio aún no existe, se mide su ancestro más cercano que sí.
        existente = directorio
        while not existente.exists() and existente != existente.parent:
            existente = existente.parent
        try:
            clave = str(Path(existente).resolve().anchor) or str(existente)
        except OSError:  # pragma: no cover - rutas inaccesibles
            clave = str(existente)
        por_volumen.setdefault(clave, []).append(etiqueta)

    resultados: list[Comprobacion] = []
    for volumen, etiquetas in sorted(por_volumen.items()):
        try:
            uso = shutil.disk_usage(volumen)
        except OSError as error:  # pragma: no cover - volumen no accesible
            resultados.append(
                Comprobacion(
                    f"Espacio libre ({', '.join(etiquetas)})",
                    False,
                    f"no se ha podido consultar '{volumen}': {error}",
                )
            )
            continue
        libre_gb = uso.free / (1024**3)
        resultados.append(
            Comprobacion(
                f"Espacio libre ({', '.join(etiquetas)})",
                libre_gb >= minimo_gb,
                (
                    f"{libre_gb:.1f} GB libres en {volumen} "
                    f"(recomendado al menos {minimo_gb:.0f} GB)"
                ),
            )
        )
    return resultados


def comprobar_sistema_de_ficheros() -> list[Comprobacion]:
    """Avisa si, bajo WSL, el proyecto o los datos están en el disco de Windows.

    El acceso a ``/mnt/c`` es muy lento con ficheros pequeños, ralentiza el entrenamiento y hace que
    las medidas de tiempo dejen de reflejar el cómputo.
    """
    contexto = detect_execution_context()
    if "wsl2" not in contexto:
        return [
            Comprobacion(
                "Sistema de ficheros",
                True,
                f"no se está bajo WSL (contexto: {contexto}); la comprobación no aplica",
            )
        ]

    rutas = get_project_paths()
    resultados: list[Comprobacion] = []
    for etiqueta, directorio in (
        ("repositorio", rutas.root),
        ("data", rutas.data),
        ("models", rutas.models),
        ("results", rutas.results),
    ):
        texto = str(directorio)
        en_windows = texto.startswith("/mnt/")
        resultados.append(
            Comprobacion(
                f"Ubicación de {etiqueta}",
                not en_windows,
                (
                    (
                        f"{texto} vive en el disco de Windows a través de /mnt/. "
                        "El acceso es lento con ficheros pequeños y puede convertir "
                        "la entrada/salida en el cuello de botella, INVALIDANDO las "
                        "medidas de tiempo. Muévelo al sistema de ficheros de Linux."
                    )
                    if en_windows
                    else f"{texto} (sistema de ficheros nativo de Linux, correcto)"
                ),
            )
        )
    return resultados


def comprobar_git() -> Comprobacion:
    """Comprueba si el directorio es un repositorio git; el hash del commit forma parte de la huella
    de cada resultado.
    """
    if shutil.which("git") is None:
        return Comprobacion("Repositorio git", False, "git no está instalado")
    huella = capture_environment(device_id="setup_check", repo_dir=get_project_paths().root)
    commit = huella.get("git_commit", "")
    if not commit:
        return Comprobacion(
            "Repositorio git",
            False,
            "el directorio no es un repositorio git; la columna git_commit del "
            "registro quedará vacía y los resultados no serán trazables",
        )
    sucio = huella.get("git_dirty") == "true"
    return Comprobacion(
        "Repositorio git",
        True,
        f"commit {commit[:12]}" + (" (ÁRBOL SUCIO: hay cambios sin confirmar)" if sucio else ""),
    )


def comprobar_paquete_importable() -> Comprobacion:
    """Comprueba que ``import tfm_edge`` funciona desde un intérprete limpio.

    Se ejecuta en un subproceso con el directorio de trabajo en la raíz del sistema, porque los
    scripts añaden ``src/`` a ``sys.path`` y en este proceso el import funcionaría aunque el paquete
    no estuviera instalado.
    """
    import subprocess
    import tempfile

    codigo = "import tfm_edge, sys; " "print(tfm_edge.__file__)"
    with tempfile.TemporaryDirectory() as neutral:
        resultado = subprocess.run(
            [sys.executable, "-c", codigo],
            capture_output=True,
            text=True,
            cwd=neutral,
            check=False,
        )

    if resultado.returncode == 0:
        return Comprobacion(
            "Paquete tfm_edge importable",
            True,
            f"desde un intérprete limpio: {resultado.stdout.strip()}",
        )

    # Diagnóstico: dónde debería estar y qué dice la instalación.
    pistas: list[str] = []
    try:
        import site

        for directorio in site.getsitepackages():
            for pth in Path(directorio).glob("*tfm*"):
                contenido = ""
                if pth.suffix == ".pth":
                    contenido = f" -> {pth.read_text(encoding='utf-8').strip()[:120]}"
                pistas.append(f"{pth.name}{contenido}")
    except Exception:  # pragma: no cover - depende del entorno
        pass

    detalle = (
        "'import tfm_edge' FALLA desde un intérprete limpio "
        f"({resultado.stderr.strip().splitlines()[-1][:100] if resultado.stderr else 'sin detalle'}). "
        f"Artefactos de instalación encontrados: {pistas or 'ninguno'}. Los scripts de scripts/ "
        "seguirán funcionando porque inyectan src/ en sys.path, pero un 'import tfm_edge' "
        "interactivo no. Reinstala con: pip install ."
    )
    return Comprobacion("Paquete tfm_edge importable", False, detalle)


def comprobar_portabilidad_datasets() -> list[Comprobacion]:
    """Comprueba que los datasets generados no contienen rutas absolutas de esta máquina."""
    from tfm_edge.data.prepare import ULTRALYTICS_DATASET_FILENAME, verify_dataset_portability

    rutas = get_project_paths()
    candidatos = [rutas.data_raw] + sorted(d for d in rutas.data.glob("resized_*") if d.is_dir())
    comprobaciones: list[Comprobacion] = []
    for directorio in candidatos:
        if not (directorio / ULTRALYTICS_DATASET_FILENAME).is_file():
            continue
        try:
            informe = verify_dataset_portability(directorio)
        except Exception as error:  # pragma: no cover - depende del entorno
            comprobaciones.append(
                Comprobacion(
                    f"Portabilidad de {directorio.name}",
                    False,
                    f"no se ha podido comprobar: {type(error).__name__}: {error}",
                )
            )
            continue
        if informe["portable"]:
            comprobaciones.append(
                Comprobacion(
                    f"Portabilidad de {directorio.name}",
                    True,
                    "sin rutas absolutas: el dataset se puede mover o empaquetar",
                )
            )
        else:
            comprobaciones.append(
                Comprobacion(
                    f"Portabilidad de {directorio.name}",
                    False,
                    "NO es portable: "
                    + "; ".join(informe["problems"][:2])
                    + ". Regenera con scripts/02b_resize_dataset.py y vuelve a "
                    "empaquetar el tar.gz",
                )
            )
    if not comprobaciones:
        comprobaciones.append(
            Comprobacion(
                "Portabilidad de datasets",
                True,
                "no hay ningún dataset generado todavía; nada que comprobar",
            )
        )
    return comprobaciones


def probar_compatibilidad_poda() -> list[Comprobacion]:
    """Comprueba que torch-pruning puede construir el grafo de dependencias de un YOLOv8.

    El ejemplo oficial de torch-pruning para YOLOv8 no funciona con versiones recientes de
    Ultralytics, así que se prueba con las versiones instaladas. El modelo se construye desde su
    YAML, sin descargar pesos.
    """
    resultados: list[Comprobacion] = []
    try:
        import torch
        import torch_pruning as tp
        from ultralytics import YOLO
    except ImportError as error:
        return [
            Comprobacion(
                "Sondeo de poda YOLOv8 + torch-pruning",
                False,
                f"falta alguna dependencia: {error}",
            )
        ]

    try:
        modelo = YOLO("yolov8n.yaml").model
        modelo.eval()
        entrada = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            modelo(entrada)
        resultados.append(
            Comprobacion("Construcción de yolov8n desde YAML", True, "pase hacia delante correcto")
        )
    except Exception as error:
        return [
            *resultados,
            Comprobacion(
                "Construcción de yolov8n desde YAML",
                False,
                f"{type(error).__name__}: {error}",
            ),
        ]

    try:
        parametros_antes = sum(p.numel() for p in modelo.parameters())
        grafo = tp.DependencyGraph().build_dependency(modelo, example_inputs=entrada)
        resultados.append(
            Comprobacion(
                "Grafo de dependencias (DepGraph)",
                True,
                f"construido sobre un modelo de {parametros_antes:,} parámetros",
            )
        )
    except Exception as error:
        resultados.append(
            Comprobacion(
                "Grafo de dependencias (DepGraph)",
                False,
                f"{type(error).__name__}: {error}. Es el riesgo conocido de "
                "compatibilidad entre Ultralytics y torch-pruning.",
            )
        )
        return resultados

    # No se poda; basta con que DepGraph rastree las dependencias de la primera convolución.
    try:
        primera_conv = next(
            modulo for modulo in modelo.modules() if isinstance(modulo, torch.nn.Conv2d)
        )
        grupo = grafo.get_pruning_group(primera_conv, tp.prune_conv_out_channels, idxs=[0, 1])
        resultados.append(
            Comprobacion(
                "Rastreo de dependencias de una convolución",
                True,
                f"el grupo de poda contiene {len(list(grupo))} operaciones dependientes",
            )
        )
    except Exception as error:
        resultados.append(
            Comprobacion(
                "Rastreo de dependencias de una convolución",
                False,
                f"{type(error).__name__}: {error}",
            )
        )
    return resultados


def imprimir_seccion(titulo: str, comprobaciones: list[Comprobacion]) -> None:
    """Imprime una sección del informe."""
    print(f"\n{titulo}")
    print("-" * len(titulo))
    for comprobacion in comprobaciones:
        if comprobacion.disponible:
            marca = "[ OK ]"
        elif comprobacion.obligatorio:
            marca = "[FALTA]"
        else:
            marca = "[ -- ]"
        print(f"  {marca} {comprobacion.nombre:38s} {comprobacion.detalle}")


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        prog="00_setup_check.py",
        description=(
            "Comprueba el entorno del proyecto e informa de qué componentes están disponibles. "
            "No instala nada. Los runtimes de inferencia son opcionales."
        ),
        epilog=(
            "Ejemplos:\n"
            "  python scripts/00_setup_check.py\n"
            "  python scripts/00_setup_check.py --create-dirs\n"
            "  python scripts/00_setup_check.py --probe-pruning --json-output "
            "results/logs/setup.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Crea los directorios del proyecto que falten en lugar de solo informar.",
    )
    parser.add_argument(
        "--probe-pruning",
        action="store_true",
        help=(
            "Prueba la compatibilidad entre Ultralytics y torch-pruning construyendo un yolov8n "
            "y su grafo de dependencias. Requiere torch, ultralytics y torch-pruning."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        metavar="RUTA",
        help="Vuelca el informe completo a un fichero JSON, además de imprimirlo.",
    )
    parser.add_argument(
        "--device-id",
        type=str,
        default="setup_check",
        help=(
            "Identificador del dispositivo con el que etiquetar la huella de "
            "entorno capturada (por defecto: %(default)s)."
        ),
    )
    parser.add_argument(
        "--min-disk-gb",
        type=float,
        default=50.0,
        help=(
            "Espacio libre en disco, en GB, por debajo del cual se avisa "
            "(por defecto: %(default)s)."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nivel de detalle del log (por defecto: %(default)s).",
    )
    return parser


def main() -> int:
    """Punto de entrada del script. Devuelve 0 si está todo lo obligatorio y 1 si no."""
    args = construir_parser().parse_args()
    setup_logging("00_setup_check", level=args.log_level, console=False)

    print("=" * 78)
    print("COMPROBACIÓN DEL ENTORNO: tfm-edge-object-detection")
    print("=" * 78)

    secciones: dict[str, list[Comprobacion]] = {}

    interprete = python_executable_info()
    secciones["Intérprete"] = [
        comprobar_python(),
        Comprobacion("Ejecutable", True, interprete["executable"]),
        Comprobacion(
            "Entorno virtual",
            interprete["in_virtualenv"] == "True",
            (
                "activo"
                if interprete["in_virtualenv"] == "True"
                else (
                    "NO se está usando un entorno virtual; se recomienda uno para que "
                    "las versiones ancladas no colisionen con otros proyectos"
                )
            ),
        ),
    ]

    secciones["Estructura del proyecto"] = [
        *comprobar_estructura(crear=args.create_dirs),
        *comprobar_espacio_en_disco(args.min_disk_gb),
        *comprobar_sistema_de_ficheros(),
        comprobar_git(),
    ]

    secciones["Instalación del proyecto"] = [
        comprobar_paquete_importable(),
        *comprobar_portabilidad_datasets(),
    ]

    secciones["Dependencias obligatorias"] = [
        comprobar_modulo(modulo, distribucion, obligatorio=True)
        for modulo, distribucion in DEPENDENCIAS_OBLIGATORIAS
    ]
    secciones["Compatibilidad binaria (API C de NumPy)"] = comprobar_compatibilidad_binaria()
    secciones["Entrenamiento y compresión"] = [
        comprobar_modulo(modulo, distribucion)
        for modulo, distribucion in DEPENDENCIAS_ENTRENAMIENTO
    ]
    secciones["Runtimes de inferencia (opcionales)"] = [
        comprobar_modulo(modulo, distribucion) for modulo, distribucion in RUNTIMES_OPCIONALES
    ]
    secciones["Figuras y tablas"] = [
        comprobar_modulo(modulo, distribucion) for modulo, distribucion in DEPENDENCIAS_INFORMES
    ]
    secciones["Cómputo en GPU"] = comprobar_cuda()

    if args.probe_pruning:
        secciones["Sondeo de compatibilidad de poda"] = probar_compatibilidad_poda()

    for titulo, comprobaciones in secciones.items():
        imprimir_seccion(titulo, comprobaciones)

    huella = capture_environment(device_id=args.device_id, repo_dir=get_project_paths().root)
    print("\nHuella del entorno")
    print("-" * len("Huella del entorno"))
    for linea in summarize_environment(huella).splitlines():
        print(f"  {linea}")

    todas = [c for lista in secciones.values() for c in lista]
    fallos_obligatorios = [c for c in todas if c.obligatorio and not c.disponible]
    opcionales_ausentes = [c for c in todas if not c.obligatorio and not c.disponible]

    print("\n" + "=" * 78)
    if fallos_obligatorios:
        print(f"RESULTADO: FALTAN {len(fallos_obligatorios)} COMPONENTE(S) OBLIGATORIO(S)")
        for comprobacion in fallos_obligatorios:
            print(f"  - {comprobacion.nombre}: {comprobacion.detalle}")
        print("\nInstala lo que falte con:  pip install -r requirements.txt")
    else:
        print("RESULTADO: todos los componentes obligatorios están disponibles.")
    if opcionales_ausentes:
        print(
            f"\n{len(opcionales_ausentes)} componente(s) opcional(es) ausente(s). "
            "Las combinaciones que los necesiten se registrarán con status 'no_disponible'."
        )
    print("=" * 78)

    if args.json_output is not None:
        informe: dict[str, Any] = {
            "secciones": {
                titulo: [asdict(c) for c in comprobaciones]
                for titulo, comprobaciones in secciones.items()
            },
            "environment": huella,
            "required_missing": [c.nombre for c in fallos_obligatorios],
            "optional_missing": [c.nombre for c in opcionales_ausentes],
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nInforme completo volcado en: {args.json_output}")

    return 1 if fallos_obligatorios else 0


if __name__ == "__main__":
    sys.exit(main())
