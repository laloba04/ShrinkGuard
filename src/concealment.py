"""Heuristica de deteccion de gestos de ocultacion.

Esta es la "chicha" de la Fase 1 y, a proposito, NO depende de YOLO ni de
OpenCV: trabaja solo con keypoints (numpy). Asi se puede testear con datos
sinteticos sin necesidad de modelos ni video, y se puede sustituir mas
adelante por un clasificador temporal aprendido (LSTM/ST-GCN) sin tocar el
resto del pipeline.

IMPORTANTE: esto genera una SEÑAL DE SOSPECHA, nunca una acusacion. La
decision final es siempre de una persona (human-in-the-loop). Ver GUIA.md.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.posture import PostureConfig, is_standing

# Indices de keypoints en formato COCO-17 (el que devuelve YOLO-pose).
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12


@dataclass
class ConcealmentConfig:
    """Parametros ajustables del detector. Todos los umbrales estan
    normalizados por el tamaño del torso, por lo que funcionan a distintas
    distancias de la camara sin recalibrar."""

    # Distancia muñeca->cintura (normalizada por la longitud del torso) por
    # debajo de la cual consideramos que la mano esta "en la cintura".
    near_ratio: float = 0.40
    # Margen horizontal (fraccion del ancho de cadera) para considerar que la
    # muñeca esta DELANTE del cuerpo y no simplemente con el brazo caido al
    # costado. Esto es lo que distingue "rebuscar en la cinturilla" de "brazo
    # relajado al lado".
    column_margin_ratio: float = 0.25
    # Score minimo para que un frame cuente como "mano en la cintura". Sube
    # esto si tienes falsos positivos con gestos debiles (p. ej. 0.55).
    near_score_threshold: float = 0.45
    # Frames consecutivos con la mano en la cintura para disparar un evento.
    consecutive_frames: int = 8
    # Confianza minima del keypoint para fiarnos de el.
    min_keypoint_conf: float = 0.30
    # Tras un evento, frames de enfriamiento antes de volver a disparar para la
    # misma persona (evita spam de alertas).
    cooldown_frames: int = 30
    # Memoria de scores recientes por persona (para depurar / visualizar).
    history_len: int = 64


@dataclass
class ConcealmentEvent:
    """Una señal de sospecha lista para revision humana."""

    track_id: int
    frame_idx: int
    score: float  # 0..1, cuanto mas alto mas clara la postura de ocultacion


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def concealment_score(keypoints: np.ndarray, conf: np.ndarray,
                      cfg: ConcealmentConfig) -> float:
    """Devuelve un score 0..1 de "mano en la cinturilla / ocultacion".

    keypoints: array (17, 2) con coordenadas (x, y) en pixeles.
    conf:      array (17,) con la confianza de cada keypoint.

    Devuelve 0.0 si no hay datos fiables suficientes.
    """
    c = cfg.min_keypoint_conf
    # Necesitamos hombros y caderas para estimar la escala del cuerpo.
    needed = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if any(conf[i] < c for i in needed):
        return 0.0

    shoulder_mid = _midpoint(keypoints[L_SHOULDER], keypoints[R_SHOULDER])
    hip_mid = _midpoint(keypoints[L_HIP], keypoints[R_HIP])
    torso = _dist(shoulder_mid, hip_mid)
    if torso <= 1e-6:
        return 0.0

    hip_x = sorted([keypoints[L_HIP][0], keypoints[R_HIP][0]])
    hip_width = max(hip_x[1] - hip_x[0], 1e-6)
    margin = cfg.column_margin_ratio * hip_width
    col_min, col_max = hip_x[0] - margin, hip_x[1] + margin

    best = 0.0
    for wrist_idx in (L_WRIST, R_WRIST):
        if conf[wrist_idx] < c:
            continue
        wrist = keypoints[wrist_idx]
        # ¿Esta la muñeca DELANTE del cuerpo (dentro de la columna del torso)?
        if not (col_min <= wrist[0] <= col_max):
            continue
        d = _dist(wrist, hip_mid) / torso
        if d >= cfg.near_ratio:
            continue
        # Mapear distancia a score: cerca de la cintura -> score alto.
        score = 1.0 - (d / cfg.near_ratio)
        best = max(best, score)
    return best


@dataclass
class TrackState:
    """Estado temporal de UNA persona seguida (un track_id)."""

    consecutive: int = 0
    cooldown: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=64))

    def update(self, near: bool, score: float, cfg: ConcealmentConfig) -> bool:
        """Actualiza el estado y devuelve True si se debe disparar un evento."""
        self.history.append(score)
        if self.cooldown > 0:
            self.cooldown -= 1

        if near:
            self.consecutive += 1
        else:
            self.consecutive = 0

        if self.consecutive >= cfg.consecutive_frames and self.cooldown == 0:
            self.cooldown = cfg.cooldown_frames
            self.consecutive = 0
            return True
        return False


class ConcealmentDetector:
    """Orquesta el estado por persona y emite eventos de sospecha."""

    def __init__(self, cfg: ConcealmentConfig | None = None,
                 posture_cfg: PostureConfig | None = None,
                 require_standing: bool = True) -> None:
        self.cfg = cfg or ConcealmentConfig()
        self.posture_cfg = posture_cfg or PostureConfig()
        self.require_standing = require_standing
        self._states: dict[int, TrackState] = {}

    def update(self, frame_idx: int,
               people: list[tuple[int, np.ndarray, np.ndarray]]
               ) -> list[ConcealmentEvent]:
        """people: lista de (track_id, keypoints(17,2), conf(17,)).

        Devuelve la lista de eventos disparados en este frame.
        """
        seen = set()
        events: list[ConcealmentEvent] = []
        for track_id, kp, conf in people:
            seen.add(track_id)
            state = self._states.setdefault(track_id, TrackState())
            state.history = deque(state.history, maxlen=self.cfg.history_len)
            score = concealment_score(kp, conf, self.cfg)
            near = score >= self.cfg.near_score_threshold
            # Filtro de postura: en una tienda solo nos interesan personas de
            # pie. Si no se confirma postura erguida, no acumulamos (esto mata
            # los falsos positivos de gente sentada / sin piernas en plano).
            if self.require_standing and not is_standing(kp, conf, self.posture_cfg):
                near = False
            if state.update(near, score, self.cfg):
                events.append(ConcealmentEvent(track_id, frame_idx, score))

        # Limpiar tracks que ya no estan en escena (libera memoria).
        for gone in set(self._states) - seen:
            del self._states[gone]
        return events

    def current_score(self, track_id: int) -> float:
        st = self._states.get(track_id)
        return st.history[-1] if st and st.history else 0.0