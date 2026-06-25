"""Dibujo de anotaciones sobre el frame (overlay para revision)."""

from __future__ import annotations

import cv2
import numpy as np

# Conexiones del esqueleto COCO-17 para dibujar lineas.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),      # brazos
    (5, 6), (5, 11), (6, 12), (11, 12),   # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # piernas
]

GREEN = (80, 200, 80)
AMBER = (40, 170, 240)   # BGR -> ambar
RED = (60, 60, 230)
WHITE = (245, 245, 245)


def _color_for(score: float) -> tuple[int, int, int]:
    if score >= 0.66:
        return RED
    if score >= 0.33:
        return AMBER
    return GREEN


def draw_person(frame: np.ndarray, person, score: float,
                min_conf: float = 0.3) -> None:
    color = _color_for(score)
    x1, y1, x2, y2 = (int(v) for v in person.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    kp, conf = person.keypoints, person.conf
    for a, b in SKELETON:
        if conf[a] >= min_conf and conf[b] >= min_conf:
            pa = tuple(int(v) for v in kp[a])
            pb = tuple(int(v) for v in kp[b])
            cv2.line(frame, pa, pb, color, 2)
    for i in range(len(kp)):
        if conf[i] >= min_conf:
            cv2.circle(frame, tuple(int(v) for v in kp[i]), 3, color, -1)

    label = f"ID {person.track_id}  sosp {score:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_alert_banner(frame: np.ndarray, track_id: int) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), RED, -1)
    cv2.putText(frame, f"SENAL DE SOSPECHA - ID {track_id} (revisar)",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)


def draw_footer(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    txt = "Senal automatica, NO acusacion. Decision: persona."
    cv2.putText(frame, txt, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
