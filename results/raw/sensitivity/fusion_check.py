"""Mide si fusionar Conv+BN cambia el ORDEN de sensibilidad (riesgo D-063).

El analisis de 09 cuantiza la convolucion con la BatchNorm todavia separada, que
es como esta el modelo de Ultralytics. PTQ y QAT en el runtime SI fusionan. Si el
orden de sensibilidad cambia al fusionar, el que produce 09 no sirve para decidir
exclusiones.

Se mide sobre un subconjunto: las N mas sensibles segun 09 mas N repartidas por
el resto, para que la correlacion no salga inflada por medir solo los extremos.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "/home/gomez/tfm-edge-object-detection/src")
warnings.filterwarnings("ignore")

import copy  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.ao.quantization as qao  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from tfm_edge.models.detector import Detector  # noqa: E402
from tfm_edge.models.surgery import get_submodule  # noqa: E402

RAIZ = Path("/home/gomez/tfm-edge-object-detection")
CHECKPOINT = RAIZ / "models/baselines/s_baseline_default/best.pt"
INFORME = RAIZ / "results/raw/sensitivity/s_baseline_default_sensitivity.json"
QENGINE = "x86"


def entradas(n: int, resolucion: int):
    """Prepara las mismas imagenes que uso 09, con el preprocesado compartido."""
    import cv2

    from tfm_edge.benchmark.backends import PyTorchBackend

    datos = json.loads(INFORME.read_text(encoding="utf-8"))
    directorio = RAIZ / "data/resized_640/images"
    pre = PyTorchBackend(input_resolution=resolucion, device="cpu")
    salida = []
    for nombre in datos["calibration_images"][:n]:
        imagen = cv2.imread(str(directorio / nombre))
        assert imagen is not None, nombre
        salida.append(pre.preprocess(imagen).numpy())
    return salida, datos


def salidas(modelo, xs):
    """Salidas del modelo sobre las entradas."""
    out = []
    with torch.no_grad():
        for x in xs:
            y = modelo(torch.from_numpy(x))
            out.append((y[0] if isinstance(y, (list, tuple)) else y).numpy())
    return out


def preparar(detector, capa: str, fusionar: bool):
    """Cuantiza solo una capa, opcionalmente fusionando antes su Conv+BN."""
    modelo = copy.deepcopy(detector.model).train()
    for m in modelo.modules():
        m.qconfig = None

    ruta_objetivo = capa
    if fusionar:
        assert capa.endswith(".conv"), capa
        bloque = get_submodule(modelo, capa[: -len(".conv")])
        if not hasattr(bloque, "bn"):
            return None  # no tiene BN que fusionar
        qao.fuse_modules_qat(bloque, [["conv", "bn"]], inplace=True)

    objetivo = get_submodule(modelo, ruta_objetivo)
    objetivo.qconfig = qao.get_default_qat_qconfig(QENGINE)
    return qao.prepare_qat(modelo, inplace=False).eval()


def medir(detector, capas, xs, fusionar: bool):
    """Devuelve {capa: mse} midiendo con o sin fusion."""
    referencia = salidas(copy.deepcopy(detector.model).eval(), xs)
    torch.backends.quantized.engine = QENGINE
    resultado = {}
    for i, capa in enumerate(capas, 1):
        try:
            modelo = preparar(detector, capa, fusionar)
            if modelo is None:
                continue
            with torch.no_grad():
                for x in xs:
                    modelo(torch.from_numpy(x))
            candidatas = salidas(modelo, xs)
            resultado[capa] = float(
                np.mean([np.mean((a - b) ** 2) for a, b in zip(referencia, candidatas)])
            )
        except Exception as e:
            print(f"  {capa}: FALLA {type(e).__name__}: {str(e)[:70]}")
        print(f"  [{i}/{len(capas)}] {capa}", flush=True)
    return resultado


def main() -> int:
    num_imagenes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    num_extremos = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    xs, datos = entradas(num_imagenes, datos_resolucion := 640)
    orden = datos["ranked"]
    # Extremos mas una muestra repartida: si solo se midieran las mas sensibles,
    # la correlacion saldria inflada por el rango restringido.
    intermedias = orden[num_extremos:-num_extremos]
    paso = max(1, len(intermedias) // num_extremos)
    capas = orden[:num_extremos] + intermedias[::paso][:num_extremos] + orden[-num_extremos:]
    capas = list(dict.fromkeys(capas))
    print(f"{len(capas)} capas, {num_imagenes} imagenes a {datos_resolucion} px\n")

    detector = Detector.load(CHECKPOINT, input_resolution=datos_resolucion)

    print("SIN fusionar (como mide 09):")
    sin = medir(detector, capas, xs, fusionar=False)
    print("\nCON Conv+BN fusionado (como hace el runtime):")
    con = medir(detector, capas, xs, fusionar=True)

    comunes = [c for c in capas if c in sin and c in con]
    a = [sin[c] for c in comunes]
    b = [con[c] for c in comunes]
    rho, p = spearmanr(a, b)

    print("\n" + "=" * 78)
    print(f"{len(comunes)} capas comparables")
    print(f"Spearman entre los dos ordenes: rho = {rho:.4f}  (p = {p:.2e})")
    print()
    print(f"{'capa':34s} {'sin fusion':>12s} {'con fusion':>12s} {'factor':>9s}")
    print("-" * 78)
    for c in sorted(comunes, key=lambda k: sin[k], reverse=True):
        factor = con[c] / sin[c] if sin[c] > 0 else float("nan")
        print(f"{c[:34]:34s} {sin[c]:12.3e} {con[c]:12.3e} {factor:9.2f}")

    destino = Path("/tmp/fusion_check.json")
    destino.write_text(
        json.dumps(
            {"spearman": rho, "p_value": p, "sin_fusion": sin, "con_fusion": con},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nVolcado en {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
