"""Términos de la pérdida de destilación y máscaras de región.

Funciones puras sobre tensores. La divergencia de clase es de Bernoulli por clase, porque YOLOv8 usa
un sigmoide por clase y un softmax perdería la información del fondo. La de caja es categórica sobre
las distribuciones DFL. Las dos se escalan por T² para que el gradiente no dependa de la
temperatura.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfm_edge.common.exceptions import TFMEdgeError

#: Estrategias de enmascarado, incluida la intersección de cajas y confianza del maestro.
MaskingStrategy = Literal["none", "gt_boxes", "teacher_attention", "gt_boxes_and_teacher"]


class DistillationLossError(TFMEdgeError):
    """Error de forma entre la pérdida de destilación y la cabeza del modelo."""


@dataclass(frozen=True)
class HeadGeometry:
    """Geometría de la cabeza de detección leída del modelo."""

    num_classes: int
    reg_max: int
    strides: tuple[int, ...]

    @property
    def channels_per_map(self) -> int:
        """Canales que tiene cada mapa crudo de la cabeza: ``4*reg_max + nc``."""
        return 4 * self.reg_max + self.num_classes


def head_geometry(model: nn.Module) -> HeadGeometry:
    """Lee nc, reg_max y stride de la cabeza del modelo en lugar de suponerlos."""
    capas = getattr(model, "model", None)
    cabeza = capas[-1] if capas is not None and len(capas) else None
    faltan = [a for a in ("nc", "reg_max", "stride") if not hasattr(cabeza, a)]
    if cabeza is None or faltan:
        raise DistillationLossError(
            f"La cabeza del modelo ({type(cabeza).__name__}) no expone {faltan or 'nada'}. Hacen "
            "falta nc, reg_max y stride."
        )
    zancadas = tuple(int(s) for s in cabeza.stride.tolist())
    return HeadGeometry(num_classes=int(cabeza.nc), reg_max=int(cabeza.reg_max), strides=zancadas)


def split_head_output(
    maps: Sequence[torch.Tensor], geometry: HeadGeometry
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separa los mapas crudos de la cabeza en ``(dist, cls)``, como la pérdida nativa.

    Las formas son ``(B, 4*reg_max, A)`` y ``(B, nc, A)``, con A la suma de posiciones de los tres
    niveles.
    """
    if not maps:
        raise DistillationLossError("No hay mapas de salida que repartir.")
    esperados = geometry.channels_per_map
    for i, mapa in enumerate(maps):
        if mapa.ndim != 4 or mapa.shape[1] != esperados:
            raise DistillationLossError(
                f"El mapa {i} tiene forma {tuple(mapa.shape)} y se esperaban {esperados} canales "
                f"(4*{geometry.reg_max} + {geometry.num_classes}). Comprueba que el modelo está "
                "en modo entrenamiento."
            )
    lote = maps[0].shape[0]
    plano = torch.cat([m.view(lote, esperados, -1) for m in maps], dim=2)
    return plano.split((4 * geometry.reg_max, geometry.num_classes), dim=1)


# Máscaras de región


def box_mask_grid(batch: dict[str, Any], cells_y: int, cells_x: int) -> torch.Tensor:
    """Marca las celdas de una rejilla cuyo centro cae dentro de alguna caja. Devuelve ``(B,
    cells_y, cells_x)``.

    Se calcula a la resolución de cada rejilla para no interpolar máscaras booleanas.
    """
    dispositivo = batch["bboxes"].device if "bboxes" in batch else batch["img"].device
    lote = int(batch["img"].shape[0])
    ys, xs = torch.meshgrid(
        torch.arange(cells_y, dtype=torch.float32, device=dispositivo),
        torch.arange(cells_x, dtype=torch.float32, device=dispositivo),
        indexing="ij",
    )
    centros = torch.stack(
        [((xs + 0.5) / cells_x).reshape(-1), ((ys + 0.5) / cells_y).reshape(-1)], dim=1
    )

    mascara = torch.zeros(lote, centros.shape[0], dtype=torch.bool, device=dispositivo)
    indices = batch["batch_idx"].to(dispositivo).long().view(-1)
    cajas_todas = batch["bboxes"].to(dispositivo)
    for imagen in range(lote):
        cajas = cajas_todas[indices == imagen]
        if cajas.numel() == 0:
            continue
        x1 = cajas[:, 0] - cajas[:, 2] / 2
        y1 = cajas[:, 1] - cajas[:, 3] / 2
        x2 = cajas[:, 0] + cajas[:, 2] / 2
        y2 = cajas[:, 1] + cajas[:, 3] / 2
        dentro = (
            (centros[:, None, 0] >= x1[None, :])
            & (centros[:, None, 0] <= x2[None, :])
            & (centros[:, None, 1] >= y1[None, :])
            & (centros[:, None, 1] <= y2[None, :])
        )
        mascara[imagen] = dentro.any(dim=1)
    return mascara.view(lote, cells_y, cells_x)


def ground_truth_box_mask(
    batch: dict[str, Any], geometry: HeadGeometry, resolution: int
) -> torch.Tensor:
    """Máscara de cajas sobre los anclajes aplanados de la cabeza, ``(B, A)``."""
    trozos = [
        box_mask_grid(batch, resolution // z, resolution // z).flatten(1) for z in geometry.strides
    ]
    return torch.cat(trozos, dim=1)


def teacher_confidence_mask(teacher_cls: torch.Tensor, threshold: float) -> torch.Tensor:
    """Marca los anclajes donde la probabilidad máxima del maestro supera el umbral, sin
    temperatura.
    """
    return torch.sigmoid(teacher_cls).amax(dim=1) > threshold


def build_mask(
    strategy: MaskingStrategy,
    batch: dict[str, Any],
    teacher_cls: torch.Tensor,
    geometry: HeadGeometry,
    resolution: int,
    threshold: float,
) -> torch.Tensor:
    """Construye la máscara de anclajes de la estrategia indicada."""
    if strategy == "none":
        return torch.ones_like(teacher_cls[:, 0, :], dtype=torch.bool)
    if strategy == "gt_boxes":
        return ground_truth_box_mask(batch, geometry, resolution)
    if strategy == "teacher_attention":
        return teacher_confidence_mask(teacher_cls, threshold)
    if strategy == "gt_boxes_and_teacher":
        return ground_truth_box_mask(batch, geometry, resolution) & teacher_confidence_mask(
            teacher_cls, threshold
        )
    raise DistillationLossError(f"Estrategia de enmascarado desconocida: {strategy!r}.")


# Divergencias


def class_kl_bernoulli(
    teacher_cls: torch.Tensor, student_cls: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL de Bernoulli por clase entre maestro y alumno, escalada por T² y sumada sobre clases."""
    log_p = F.logsigmoid(teacher_cls / temperature)
    log_1mp = F.logsigmoid(-teacher_cls / temperature)
    log_q = F.logsigmoid(student_cls / temperature)
    log_1mq = F.logsigmoid(-student_cls / temperature)
    p = log_p.exp()
    por_clase = p * (log_p - log_q) + (1.0 - p) * (log_1mp - log_1mq)
    return por_clase.sum(dim=1) * temperature**2


def dfl_kl(
    teacher_dist: torch.Tensor,
    student_dist: torch.Tensor,
    geometry: HeadGeometry,
    temperature: float,
) -> torch.Tensor:
    """KL categórica sobre las distribuciones DFL de los cuatro lados, escalada por T²."""
    lote, _, anclajes = teacher_dist.shape
    t = teacher_dist.view(lote, 4, geometry.reg_max, anclajes) / temperature
    s = student_dist.view(lote, 4, geometry.reg_max, anclajes) / temperature
    log_p = F.log_softmax(t, dim=2)
    log_q = F.log_softmax(s, dim=2)
    return (log_p.exp() * (log_p - log_q)).sum(dim=2).sum(dim=1) * temperature**2


def masked_mean(per_anchor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Media sobre los anclajes marcados. Con la máscara vacía devuelve cero y no nan."""
    marcados = mask.to(per_anchor.dtype)
    return (per_anchor * marcados).sum() / marcados.sum().clamp(min=1.0)


def feature_mse(
    student_feature: torch.Tensor,
    teacher_feature: torch.Tensor,
    mask: torch.Tensor | None = None,
    normalize_per_channel: bool = True,
) -> torch.Tensor:
    """Error cuadrático medio entre dos mapas de características ``(B, C, H, W)``.

    Con ``normalize_per_channel`` cada canal se lleva a media 0 y varianza 1, para que no domine la
    capa con activaciones más grandes. El mapa del alumno debe llegar ya adaptado.
    """
    if student_feature.shape != teacher_feature.shape:
        raise DistillationLossError(
            f"El mapa del alumno {tuple(student_feature.shape)} y el del maestro "
            f"{tuple(teacher_feature.shape)} no coinciden. Revisa el adaptador o el "
            "emparejamiento de capas."
        )

    def _normalizar(x: torch.Tensor) -> torch.Tensor:
        media = x.mean(dim=(2, 3), keepdim=True)
        desviacion = x.std(dim=(2, 3), keepdim=True)
        return (x - media) / (desviacion + 1e-6)

    alumno = _normalizar(student_feature) if normalize_per_channel else student_feature
    maestro = _normalizar(teacher_feature) if normalize_per_channel else teacher_feature
    por_elemento = (alumno - maestro.detach()) ** 2
    if mask is None:
        return por_elemento.mean()
    marcados = mask.to(por_elemento.dtype).unsqueeze(1)
    canales = por_elemento.shape[1]
    return (por_elemento * marcados).sum() / (marcados.sum() * canales).clamp(min=1.0)


# Pérdida completa


@dataclass
class DistillationComponents:
    """Desglose de la pérdida de un paso."""

    response_cls: torch.Tensor
    response_box: torch.Tensor
    features: torch.Tensor
    detection: torch.Tensor
    total: torch.Tensor
    marked_anchors: int

    def as_floats(self) -> dict[str, float]:
        """Devuelve el desglose como números, listo para volcar a CSV."""
        return {
            "response_cls": float(self.response_cls.detach()),
            "response_box": float(self.response_box.detach()),
            "features": float(self.features.detach()),
            "detection": float(self.detection.detach()),
            "total": float(self.total.detach()),
            "marked_anchors": float(self.marked_anchors),
        }


class DistillationLoss:
    """Combina los términos de destilación con la pérdida de detección. No guarda estado entre
    pasos.
    """

    def __init__(
        self,
        geometry: HeadGeometry,
        resolution: int,
        temperature: float = 2.0,
        alpha: float = 0.5,
        class_weight: float = 1.0,
        box_weight: float = 1.0,
        feature_weight: float = 1.0,
        kd_type: Literal["logit", "feature", "combined"] = "combined",
        response_masking: MaskingStrategy = "gt_boxes_and_teacher",
        feature_masking: MaskingStrategy = "gt_boxes",
        teacher_conf_threshold: float = 0.05,
        normalize_features: bool = True,
    ) -> None:
        """Configura la pérdida. El estudio usa T=2 y enmascara la respuesta con la intersección de
        cajas y confianza del maestro.
        """
        if temperature <= 0:
            raise DistillationLossError(f"La temperatura debe ser positiva y es {temperature}.")
        if not 0.0 <= alpha <= 1.0:
            raise DistillationLossError(f"alpha debe estar en [0, 1] y es {alpha}.")
        self.geometry = geometry
        self.resolution = resolution
        self.temperature = temperature
        self.alpha = alpha
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.feature_weight = feature_weight
        self.kd_type = kd_type
        self.response_masking = response_masking
        self.feature_masking = feature_masking
        self.teacher_conf_threshold = teacher_conf_threshold
        self.normalize_features = normalize_features

    @property
    def uses_response(self) -> bool:
        """Si el tipo de destilación configurado incluye el término de respuesta."""
        return self.kd_type in {"logit", "combined"}

    @property
    def uses_features(self) -> bool:
        """Si el tipo de destilación configurado incluye el término de características."""
        return self.kd_type in {"feature", "combined"}

    def response(
        self,
        teacher_maps: Sequence[torch.Tensor],
        student_maps: Sequence[torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Términos de clase y caja de la respuesta, sin ponderar, y anclajes marcados."""
        dist_m, cls_m = split_head_output(teacher_maps, self.geometry)
        dist_a, cls_a = split_head_output(student_maps, self.geometry)
        mascara = build_mask(
            self.response_masking,
            batch,
            cls_m,
            self.geometry,
            self.resolution,
            self.teacher_conf_threshold,
        )
        clase = masked_mean(class_kl_bernoulli(cls_m, cls_a, self.temperature), mascara)
        caja = masked_mean(dfl_kl(dist_m, dist_a, self.geometry, self.temperature), mascara)
        return clase, caja, int(mascara.sum())

    def features(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
        batch: dict[str, Any],
        pairs: Sequence[tuple[str, str]],
    ) -> torch.Tensor:
        """Término de características sumado sobre las parejas de capas, sin ponderar."""
        total = None
        for capa_maestro, capa_alumno in pairs:
            if capa_maestro not in teacher_features or capa_alumno not in student_features:
                raise DistillationLossError(
                    f"Falta la activación de {capa_maestro!r} o de {capa_alumno!r}. Maestro: "
                    f"{sorted(teacher_features)}; alumno: {sorted(student_features)}."
                )
            mapa_alumno = student_features[capa_alumno]
            mapa_maestro = teacher_features[capa_maestro]
            mascara = None
            if self.feature_masking in {"gt_boxes", "gt_boxes_and_teacher"}:
                mascara = box_mask_grid(batch, mapa_maestro.shape[2], mapa_maestro.shape[3])
            termino = feature_mse(mapa_alumno, mapa_maestro, mascara, self.normalize_features)
            total = termino if total is None else total + termino
        if total is None:
            raise DistillationLossError("No hay ningún emparejamiento de capas que destilar.")
        return total

    def combine(
        self,
        detection: torch.Tensor,
        response_cls: torch.Tensor,
        response_box: torch.Tensor,
        feature_term: torch.Tensor,
        marked_anchors: int,
        batch_size: int = 1,
    ) -> DistillationComponents:
        """Combina la detección con los términos de destilación según ``alpha``.

        La detección se multiplica por ``1 - alpha``, así que alpha también cambia la escala del
        gradiente. Los términos de destilación son medias por anclaje; la pérdida de detección de
        Ultralytics ya viene multiplicada por el lote, y el total se calcula en esa escala porque es
        lo que se retropropaga.
        """
        cls_pond = response_cls * self.class_weight
        box_pond = response_box * self.box_weight
        feat_pond = feature_term * self.feature_weight
        destilacion = cls_pond + box_pond + feat_pond
        total = (1.0 - self.alpha) * detection + self.alpha * destilacion * batch_size
        return DistillationComponents(
            response_cls=cls_pond,
            response_box=box_pond,
            features=feat_pond,
            detection=detection,
            total=total,
            marked_anchors=marked_anchors,
        )
