"""Particionado estratificado multietiqueta del dataset, agrupando casi duplicados.

SH17 está desequilibrado y cada imagen tiene varias clases, así que se estratifica por clase a nivel
de imagen. Las imágenes de Pexels incluyen tomas casi iguales de la misma escena; se detectan con
dHash y cada grupo va entero a una partición, sin eliminar nada. La asignación es voraz: los grupos
se ordenan por su clase más rara y van a la partición con mayor déficit relativo de esa clase.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger
from tfm_edge.common.paths import ensure_dir, ensure_parent_dir

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")

SMOKE_SPLIT_NAME = "smoke"

#: Lado de la miniatura del hash perceptual.
PHASH_SIZE = 8

#: Distancia de Hamming máxima para considerar dos imágenes casi duplicadas. Con dHash de 64 bits, 5
#: agrupa recortes y reencuadres suaves sin juntar escenas distintas.
DEFAULT_PHASH_THRESHOLD = 5


class SplitError(TFMEdgeError):
    """Error en el particionado del dataset."""


@dataclass
class DuplicateGroup:
    """Grupo de imágenes casi idénticas."""

    members: list[str]
    max_distance: int = 0

    @property
    def size(self) -> int:
        """Número de imágenes del grupo."""
        return len(self.members)


@dataclass
class SplitReport:
    """Informe del particionado.

    ``class_distribution`` cuenta imágenes que contienen cada clase, no instancias; las instancias
    las cuenta count_instances_per_class. El nombre se mantiene por los split_report.json
    existentes.
    """

    counts: dict[str, int] = field(default_factory=dict)
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    duplicate_groups: list[DuplicateGroup] = field(default_factory=list)
    overlap: dict[str, list[str]] = field(default_factory=dict)
    seed: int = 0
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD

    @property
    def num_grouped_images(self) -> int:
        """Número de imágenes que pertenecen a algún grupo de casi duplicados."""
        return sum(g.size for g in self.duplicate_groups)

    def to_dict(self) -> dict[str, Any]:
        """Devuelve el informe como diccionario serializable."""
        return {
            "seed": self.seed,
            "phash_threshold": self.phash_threshold,
            "counts": self.counts,
            "class_distribution": self.class_distribution,
            "overlap": self.overlap,
            "near_duplicates": {
                "num_groups": len(self.duplicate_groups),
                "num_images_in_groups": self.num_grouped_images,
                "groups": [
                    {"members": g.members, "max_distance": g.max_distance}
                    for g in self.duplicate_groups
                ],
                "policy": (
                    "Los casi duplicados NO se eliminan: se agrupan para que todas "
                    "las variantes de una escena caigan en la misma particion. "
                    "Eliminarlos cambiaria el dataset y haria los resultados "
                    "incomparables con la literatura."
                ),
            },
        }


# Hash perceptual


def compute_dhash(image_path: Path | str, hash_size: int = PHASH_SIZE) -> int:
    """dHash de una imagen: compara cada píxel con el de su derecha en una miniatura en grises.

    Es robusto a cambios de escala, compresión y brillo.
    """
    from PIL import Image

    with Image.open(image_path) as imagen:
        # draft() descomprime el JPEG ya a escala reducida; sin él, procesar las 8.099 imágenes a
        # resolución completa lleva más de una hora. En PNG no hace nada.
        try:
            imagen.draft("L", (hash_size * 8, hash_size * 8))
        except (AttributeError, ValueError):  # pragma: no cover - formato sin draft
            pass
        # Un píxel más de ancho para tener hash_size al cuadrado comparaciones.
        miniatura = imagen.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixeles = list(miniatura.getdata())

    bits = 0
    for fila in range(hash_size):
        base = fila * (hash_size + 1)
        for columna in range(hash_size):
            bits = (bits << 1) | int(pixeles[base + columna] > pixeles[base + columna + 1])
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Número de bits en que difieren dos hashes."""
    return (a ^ b).bit_count()


def find_near_duplicates(
    hashes: Mapping[str, int], threshold: int = DEFAULT_PHASH_THRESHOLD
) -> list[DuplicateGroup]:
    """Agrupa las imágenes cuyos hashes están a distancia menor o igual que el umbral.

    Los grupos son componentes conexas. Para no comparar todos los pares se indexa por bandas de
    bits: dos hashes a distancia ``threshold`` comparten al menos una de ``threshold + 1`` bandas.
    """
    if threshold < 0:
        raise ValueError(f"El umbral debe ser no negativo y es {threshold}.")

    nombres = list(hashes)
    if len(nombres) < 2:
        return []

    numero_bandas = threshold + 1
    bits_totales = PHASH_SIZE * PHASH_SIZE
    ancho_banda = max(1, bits_totales // numero_bandas)

    cubos: dict[tuple[int, int], list[str]] = defaultdict(list)
    for nombre in nombres:
        valor = hashes[nombre]
        for banda in range(numero_bandas):
            desplazamiento = banda * ancho_banda
            trozo = (valor >> desplazamiento) & ((1 << ancho_banda) - 1)
            cubos[(banda, trozo)].append(nombre)

    # Union-find sobre los candidatos que comparten banda.
    padre: dict[str, str] = {n: n for n in nombres}

    def raiz(x: str) -> str:
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    def unir(a: str, b: str) -> None:
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[rb] = ra

    distancias: dict[tuple[str, str], int] = {}
    for candidatos in cubos.values():
        if len(candidatos) < 2:
            continue
        for i in range(len(candidatos)):
            for j in range(i + 1, len(candidatos)):
                a, b = candidatos[i], candidatos[j]
                if raiz(a) == raiz(b):
                    continue
                clave = (a, b) if a < b else (b, a)
                if clave not in distancias:
                    distancias[clave] = hamming_distance(hashes[a], hashes[b])
                if distancias[clave] <= threshold:
                    unir(a, b)

    componentes: dict[str, list[str]] = defaultdict(list)
    for nombre in nombres:
        componentes[raiz(nombre)].append(nombre)

    grupos: list[DuplicateGroup] = []
    for miembros in componentes.values():
        if len(miembros) < 2:
            continue
        miembros = sorted(miembros)
        maxima = 0
        for i in range(len(miembros)):
            for j in range(i + 1, len(miembros)):
                maxima = max(maxima, hamming_distance(hashes[miembros[i]], hashes[miembros[j]]))
        grupos.append(DuplicateGroup(members=miembros, max_distance=maxima))

    grupos.sort(key=lambda g: (-g.size, g.members[0]))
    return grupos


# Estratificación multietiqueta


def stratified_multilabel_split(
    labels_by_image: Mapping[str, set[int]],
    ratios: Mapping[str, float],
    groups: Sequence[DuplicateGroup] | None = None,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Reparte los grupos de imágenes en particiones estratificando por clase.

    Las imágenes sin grupo cuentan como grupos de uno. ``ratios`` debe sumar 1.
    """
    suma = sum(ratios.values())
    if abs(suma - 1.0) > 1e-6:
        raise SplitError(
            f"Las proporciones de partición suman {suma:.6f} y deben sumar 1.0.",
            "Revisa dataset.split_ratios en la configuración.",
        )
    if not labels_by_image:
        raise SplitError(
            "No hay imágenes que repartir.", "Comprueba que el dataset está preparado."
        )

    # Unidades a repartir
    pertenencia: dict[str, int] = {}
    unidades: list[list[str]] = []
    for indice, grupo in enumerate(groups or []):
        for miembro in grupo.members:
            if miembro in labels_by_image:
                pertenencia[miembro] = indice
        unidades.append([m for m in grupo.members if m in labels_by_image])
    unidades = [u for u in unidades if u]

    sueltas = [n for n in labels_by_image if n not in pertenencia]
    unidades.extend([[n] for n in sueltas])

    # Rareza de cada clase
    frecuencia: Counter[int] = Counter()
    for clases in labels_by_image.values():
        frecuencia.update(clases)

    def rareza(unidad: Sequence[str]) -> tuple[int, int]:
        clases: set[int] = set()
        for nombre in unidad:
            clases |= labels_by_image[nombre]
        if not clases:
            return (10**9, -len(unidad))
        return (min(frecuencia[c] for c in clases), -len(unidad))

    generador = random.Random(seed)
    generador.shuffle(unidades)
    # Primero las unidades con la clase más rara, cuando aún hay margen en todas las particiones.
    unidades.sort(key=rareza)

    total_imagenes = len(labels_by_image)
    objetivo = {p: ratios[p] * total_imagenes for p in ratios}
    asignadas: dict[str, list[str]] = {p: [] for p in ratios}
    conteo_clase: dict[str, Counter[int]] = {p: Counter() for p in ratios}
    objetivo_clase = {p: {c: frecuencia[c] * ratios[p] for c in frecuencia} for p in ratios}

    for unidad in unidades:
        clases: set[int] = set()
        for nombre in unidad:
            clases |= labels_by_image[nombre]

        # Estratificación iterativa (Sechidis et al.): la partición se elige por el déficit de la
        # clase más rara de la unidad. Sumar los déficits de todas sus clases dejaría que person y
        # hands, presentes casi siempre, mandaran a train todas las clases raras.
        if clases:
            clase_rara = min(clases, key=lambda c: (frecuencia[c], c))

            def coste(particion: str, rara: int = clase_rara) -> tuple[float, float]:
                # Se maximizan ambos déficits, de ahí el signo.
                deficit_rara = objetivo_clase[particion][rara] - conteo_clase[particion][rara]
                deficit_global = objetivo[particion] - len(asignadas[particion])
                return (-deficit_rara, -deficit_global)

        else:

            def coste(particion: str, rara: int = -1) -> tuple[float, float]:
                # Sin anotaciones solo cuenta el tamaño.
                return (0.0, -(objetivo[particion] - len(asignadas[particion])))

        elegida = min(ratios, key=coste)
        asignadas[elegida].extend(unidad)
        for nombre in unidad:
            conteo_clase[elegida].update(labels_by_image[nombre])

    return {p: sorted(v) for p, v in asignadas.items()}


def verify_no_overlap(splits: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Elementos comunes entre cada par de particiones; vacío si no hay solapamiento."""
    solapamientos: dict[str, list[str]] = {}
    nombres = list(splits)
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            comunes = sorted(set(splits[nombres[i]]) & set(splits[nombres[j]]))
            if comunes:
                solapamientos[f"{nombres[i]}∩{nombres[j]}"] = comunes
    return solapamientos


def compare_with_reference_split(
    splits: Mapping[str, Sequence[str]],
    train_files: Iterable[str],
    val_files: Iterable[str],
) -> dict[str, Any]:
    """Compara el particionado propio con train_files.txt y val_files.txt del paquete oficial.

    No se usan como partición del estudio, pero el informe indica si se podría medir sobre ellas.
    """

    def _stems(nombres: Iterable[str]) -> set[str]:
        return {Path(n.strip()).stem for n in nombres if n.strip()}

    referencia = {"train": _stems(train_files), "val": _stems(val_files)}
    propio = {p: {Path(n).stem for n in v} for p, v in splits.items()}

    comparacion: dict[str, Any] = {
        "reference_sizes": {k: len(v) for k, v in referencia.items()},
        "own_sizes": {k: len(v) for k, v in propio.items()},
        "note": (
            "La particion de los autores tiene dos conjuntos (train/val) y la del "
            "estudio tres (train/val/test), asi que no son directamente "
            "equiparables. La comparacion sirve para saber si seria viable medir "
            "tambien sobre la particion oficial."
        ),
        "agreement": {},
    }
    for nombre_ref, conjunto_ref in referencia.items():
        if not conjunto_ref:
            continue
        for nombre_propio, conjunto_propio in propio.items():
            interseccion = len(conjunto_ref & conjunto_propio)
            union = len(conjunto_ref | conjunto_propio)
            comparacion["agreement"][f"oficial:{nombre_ref} vs propio:{nombre_propio}"] = {
                "intersection": interseccion,
                "jaccard": round(interseccion / union, 4) if union else 0.0,
                "fraction_of_official": round(interseccion / len(conjunto_ref), 4),
            }
    return comparacion


def write_split_files(
    splits: Mapping[str, Sequence[str]],
    output_dir: Path | str,
    extension_map: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Escribe la lista de cada partición en texto plano."""
    directorio = ensure_dir(output_dir)
    rutas: dict[str, Path] = {}
    for particion, nombres in splits.items():
        destino = directorio / f"{particion}.txt"
        lineas = [f"{n}{extension_map.get(n, '')}" if extension_map else n for n in sorted(nombres)]
        ensure_parent_dir(destino).write_text("\n".join(lineas) + "\n", encoding="utf-8")
        rutas[particion] = destino
        logger.info("Partición '%s': %d imágenes en %s", particion, len(nombres), destino)
    return rutas


def build_smoke_split(
    labels_by_image: Mapping[str, set[int]],
    sizes_by_image: Mapping[str, set[str]] | None = None,
    num_images: int = 64,
    seed: int = 42,
) -> list[str]:
    """Subconjunto reducido para pruebas de humo que cubre las 17 clases y los tres rangos de
    tamaño.

    Primero cubre clases y tamaños de forma voraz y después completa al azar con la semilla. Se
    guarda como una partición más.
    """
    if not labels_by_image:
        raise SplitError(
            "No hay imágenes para construir el subconjunto de smoke test.",
            "Prepara antes el dataset con scripts/02_prepare_dataset.py.",
        )

    def elementos(nombre: str) -> set[str]:
        cubiertos = {f"c{c}" for c in labels_by_image.get(nombre, set())}
        if sizes_by_image:
            cubiertos |= {f"s{s}" for s in sizes_by_image.get(nombre, set())}
        return cubiertos

    universo: set[str] = set()
    for nombre in labels_by_image:
        universo |= elementos(nombre)

    generador = random.Random(seed)
    candidatos = sorted(labels_by_image)
    generador.shuffle(candidatos)

    seleccion: list[str] = []
    pendientes = set(universo)
    while pendientes:
        mejor = max(candidatos, key=lambda n: (len(elementos(n) & pendientes), n), default=None)
        if mejor is None or not (elementos(mejor) & pendientes):
            break
        seleccion.append(mejor)
        pendientes -= elementos(mejor)
        candidatos.remove(mejor)

    if pendientes:
        logger.warning(
            "El subconjunto de smoke test no cubre %d elementos: %s. "
            "Puede que alguna clase no aparezca en ninguna imagen.",
            len(pendientes),
            sorted(pendientes)[:10],
        )
    if len(seleccion) > num_images:
        logger.warning(
            "Cubrir todas las clases y tamaños exige %d imágenes, más de las %d pedidas. Se "
            "usan %d.",
            len(seleccion),
            num_images,
            len(seleccion),
        )
    for nombre in candidatos:
        if len(seleccion) >= num_images:
            break
        seleccion.append(nombre)

    return sorted(seleccion)


def load_split_file(path: Path | str) -> list[str]:
    """Carga una lista de partición sin extensiones."""
    fichero = Path(path)
    if not fichero.is_file():
        raise SplitError(
            f"No existe el fichero de partición '{fichero}'.",
            "Genera las particiones con scripts/02_prepare_dataset.py.",
        )
    return [
        Path(l.strip()).stem for l in fichero.read_text(encoding="utf-8").splitlines() if l.strip()
    ]


def count_instances_per_class(
    split_files: Mapping[str, Sequence[str]],
    labels_dir: Path | str,
    class_names: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Instancias anotadas por clase y partición, leídas de las etiquetas YOLO.

    Devuelve las 17 clases aunque valgan cero. Falla si falta el fichero de etiquetas de alguna
    imagen o si un índice de clase se sale de rango.
    """
    directorio = Path(labels_dir)
    if not directorio.is_dir():
        raise SplitError(
            f"No existe el directorio de etiquetas '{directorio}'.",
            "Prepara antes el dataset con scripts/02_prepare_dataset.py.",
        )

    conteos: dict[str, dict[str, int]] = {}
    ausentes: list[str] = []
    for particion, nombres in split_files.items():
        acumulado: Counter[int] = Counter()
        for nombre in nombres:
            fichero = directorio / f"{Path(nombre).stem}.txt"
            if not fichero.is_file():
                ausentes.append(fichero.name)
                continue
            for linea in fichero.read_text(encoding="utf-8").splitlines():
                if not linea.strip():
                    continue
                indice = int(linea.split()[0])
                if not 0 <= indice < len(class_names):
                    raise SplitError(
                        f"La etiqueta '{fichero}' contiene el índice de clase "
                        f"{indice}, fuera del rango 0..{len(class_names) - 1}.",
                        "Revisa configs/dataset/sh17_classes.json y la preparación del dataset.",
                    )
                acumulado[indice] += 1
        conteos[particion] = {n: acumulado.get(i, 0) for i, n in enumerate(class_names)}

    if ausentes:
        raise SplitError(
            f"Faltan {len(ausentes)} ficheros de etiquetas, entre ellos "
            f"{', '.join(ausentes[:3])}.",
            "Las particiones y las etiquetas no corresponden al mismo dataset. "
            "Vuelve a generar ambas con scripts/02_prepare_dataset.py.",
        )
    return conteos


def verify_all_classes_present(
    counts: Mapping[str, Mapping[str, int]],
    class_names: Sequence[str],
    splits: Sequence[str] = SPLIT_NAMES,
) -> None:
    """Comprueba que todas las clases tienen instancias en todas las particiones.

    Se ejecuta sobre el particionado real cada vez que se genera la figura, y el mensaje enumera
    todos los casos.
    """
    faltan_particiones = [s for s in splits if s not in counts]
    if faltan_particiones:
        raise SplitError(
            f"El recuento no incluye la partición {', '.join(faltan_particiones)}.",
            f"Se esperaban las particiones: {', '.join(splits)}.",
        )

    vacias = [
        (nombre, particion)
        for nombre in class_names
        for particion in splits
        if counts[particion].get(nombre, 0) == 0
    ]
    if vacias:
        detalle = "; ".join(f"{clase} sin instancias en {particion}" for clase, particion in vacias)
        raise SplitError(
            f"Hay {len(vacias)} combinaciones de clase y partición sin instancias: {detalle}.",
            "Sin instancias en prueba el AP de esas clases no se puede evaluar. Regenera las "
            "particiones con scripts/02_prepare_dataset.py y revisa stratified_multilabel_split.",
        )
    logger.info(
        "Las %d clases tienen instancias en %s.",
        len(class_names),
        ", ".join(splits),
    )


def write_split_report(report: SplitReport, output_path: Path | str) -> Path:
    """Vuelca el informe del particionado a JSON."""
    destino = ensure_parent_dir(output_path)
    destino.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Informe de particionado escrito en %s", destino)
    return destino
