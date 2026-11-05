"""Subconjunto reducido de una partición para las pruebas de humo.

Recorta a la vez la lista de Ultralytics y la del proyecto; si solo se recorta una, el COCO de
referencia trae imágenes que no se predicen y el recall baja. Imágenes y etiquetas se enlazan.
"""

from __future__ import annotations

from pathlib import Path

from tfm_edge.common.exceptions import TFMEdgeError
from tfm_edge.common.logging_utils import get_logger

logger = get_logger(__name__)

SMOKE_IMAGES = 40


def count_labels(data_yaml: Path, entry: str) -> int:
    """Cuenta las anotaciones de una entrada de partición. Sin fichero de etiquetas devuelve 0."""
    etiqueta = data_yaml.parent / "labels" / f"{Path(entry.strip()).stem}.txt"
    if not etiqueta.is_file():
        return 0
    return sum(1 for linea in etiqueta.read_text(encoding="utf-8").splitlines() if linea.strip())


def prepare_smoke_subset(
    data_yaml: Path,
    splits_dir: Path,
    split: str,
    limit: int,
    dest: Path,
    order: str = "first",
) -> tuple[Path, Path]:
    """Crea una copia reducida del dataset y devuelve ``(yaml, directorio de particiones)``.

    ``order="first"`` toma las primeras imágenes; ``"most_labels"`` las más pobladas, que son las
    que marcan el pico de memoria.
    """
    if order not in {"first", "most_labels"}:
        raise TFMEdgeError(
            f"Orden de selección desconocido: '{order}'.",
            "Los valores admitidos son 'first' y 'most_labels'.",
        )
    import yaml as _yaml

    lista_ultralytics = data_yaml.parent / f"ultralytics_{split}.txt"
    lista_proyecto = splits_dir / f"{split}.txt"
    for fichero in (lista_ultralytics, lista_proyecto):
        if not fichero.is_file():
            raise TFMEdgeError(
                f"No existe la lista '{fichero}'.",
                "Genera las particiones con scripts/02_prepare_dataset.py.",
            )

    dest.mkdir(parents=True, exist_ok=True)
    for nombre in ("images", "labels"):
        enlace = dest / nombre
        if enlace.is_symlink() or enlace.exists():
            enlace.unlink()
        enlace.symlink_to((data_yaml.parent / nombre).resolve(), target_is_directory=True)

    entradas = [
        linea
        for linea in lista_ultralytics.read_text(encoding="utf-8").splitlines()
        if linea.strip()
    ]
    if order == "most_labels":
        # Orden estable para que dos ejecuciones elijan las mismas imágenes.
        entradas.sort(key=lambda linea: -count_labels(data_yaml, linea))
    entradas = entradas[:limit]
    (dest / f"ultralytics_{split}.txt").write_text("\n".join(entradas) + "\n", encoding="utf-8")

    stems = {Path(entrada).stem for entrada in entradas}
    del_proyecto = [
        linea
        for linea in lista_proyecto.read_text(encoding="utf-8").splitlines()
        if linea.strip() and Path(linea.strip()).stem in stems
    ]
    splits_reducido = dest / "splits"
    splits_reducido.mkdir(parents=True, exist_ok=True)
    (splits_reducido / f"{split}.txt").write_text("\n".join(del_proyecto) + "\n", encoding="utf-8")

    contenido = _yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    contenido[split] = f"ultralytics_{split}.txt"
    yaml_reducido = dest / data_yaml.name
    yaml_reducido.write_text(
        _yaml.safe_dump(contenido, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    logger.info(
        "Prueba de humo: %d imágenes en '%s' (%d entradas de partición)",
        len(entradas),
        dest,
        len(del_proyecto),
    )
    return yaml_reducido, splits_reducido
