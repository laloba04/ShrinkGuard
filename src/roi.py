"""Zonas de interes (ROI) para acotar el analisis a ciertas areas (Fase 2).

En una tienda interesa analizar solo ciertas zonas (estanterias, probadores) y
no, por ejemplo, la fila de cajas o una zona de paso. Una ROI reduce falsos
positivos y carga de computo al descartar a quien esta fuera del area.

Pura geometria sobre coordenadas NORMALIZADAS [0,1], de modo que una misma ROI
vale para cualquier resolucion de captura. Solo numpy: testeable sin video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

L_HIP, R_HIP = 11, 12  # COCO-17


@dataclass
class ROIConfig:
    """Region de interes como poligono normalizado."""

    # Si False, no se filtra: se analiza todo el frame.
    enabled: bool = False
    # Poligono en coordenadas normalizadas [0,1]: lista de (x, y). Min 3 vertices.
    polygon: list[tuple[float, float]] = field(default_factory=list)
    # Confianza minima de keypoint para fiarnos al estimar la posicion.
    min_keypoint_conf: float = 0.30


def parse_roi_arg(items: list[str]) -> list[tuple[float, float]]:
    """Convierte ['x,y', 'x,y', ...] (normalizados [0,1]) en un poligono.

    Lanza ValueError si hay menos de 3 vertices o coordenadas fuera de [0,1].
    """
    pts: list[tuple[float, float]] = []
    for it in items:
        try:
            x_str, y_str = it.split(",")
            x, y = float(x_str), float(y_str)
        except ValueError:
            raise ValueError(f"vertice de ROI invalido: {it!r} (esperaba 'x,y')")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"coordenadas de ROI fuera de [0,1]: {it!r}")
        pts.append((x, y))
    if len(pts) < 3:
        raise ValueError("la ROI necesita al menos 3 vertices")
    return pts


def point_in_polygon(point: tuple[float, float],
                     polygon: list[tuple[float, float]]) -> bool:
    """Test punto-en-poligono por ray casting. numpy puro.

    point y polygon en el mismo sistema de coordenadas (aqui, normalizado).
    """
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)
    if n < 3:
        return False
    x, y = point
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
                (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def person_anchor(keypoints: np.ndarray, conf: np.ndarray,
                  min_conf: float = 0.30) -> np.ndarray | None:
    """Punto representativo de la persona en pixeles.

    Usa el punto medio de las caderas si son fiables (es el centro del cuerpo y
    el mismo ancla que la heuristica de ocultacion). Si no, cae a la media de
    los keypoints fiables. Devuelve None si no hay ningun keypoint fiable.
    """
    if conf[L_HIP] >= min_conf and conf[R_HIP] >= min_conf:
        return (keypoints[L_HIP] + keypoints[R_HIP]) / 2.0
    mask = conf >= min_conf
    if not mask.any():
        return None
    return keypoints[mask].mean(axis=0)


def in_roi(keypoints: np.ndarray, conf: np.ndarray,
           frame_w: float, frame_h: float, cfg: ROIConfig) -> bool:
    """True si la persona esta dentro de la ROI (o si la ROI esta desactivada).

    Conservador a proposito: si no se puede estimar la posicion de la persona,
    devuelve False (no la analizamos), en linea con el filtro de postura.
    """
    if not cfg.enabled or len(cfg.polygon) < 3:
        return True
    anchor = person_anchor(keypoints, conf, cfg.min_keypoint_conf)
    if anchor is None:
        return False
    nx = anchor[0] / max(frame_w, 1e-6)
    ny = anchor[1] / max(frame_h, 1e-6)
    return point_in_polygon((nx, ny), cfg.polygon)
