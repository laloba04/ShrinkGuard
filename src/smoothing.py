"""Suavizado temporal de keypoints + tolerancia a oclusiones (Fase 2).

El estimador de pose es ruidoso: los keypoints "tiemblan" entre frames y a
veces desaparecen (oclusion). Este modulo limpia ese ruido ANTES de que la
heuristica puntue, manteniendo el principio de la Fase 1: solo numpy, sin YOLO
ni OpenCV, asi que se testea con datos sinteticos.

Dos mecanismos por keypoint y por persona (track_id):
  1. Media movil ponderada por confianza sobre las ultimas observaciones
     fiables -> reduce el jitter.
  2. Manejo de oclusion: si un keypoint deja de ser fiable, se mantiene su
     ultimo valor fiable durante unos frames con la confianza decayendo. Si la
     oclusion se prolonga, se da por ausente (confianza 0) y la heuristica lo
     ignora de forma natural por su propio umbral.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothingConfig:
    """Parametros del suavizado temporal y la tolerancia a oclusiones."""

    # Si False, el suavizador es un passthrough (devuelve la entrada tal cual).
    enabled: bool = True
    # Numero de observaciones fiables recientes para la media movil.
    window: int = 5
    # Umbral por debajo del cual un keypoint se considera no fiable (oclusion).
    min_keypoint_conf: float = 0.30
    # Frames que se reutiliza el ultimo valor fiable mientras dura una oclusion.
    max_hold_frames: int = 5
    # Factor de decaimiento de la confianza por cada frame mantenido. Cuando la
    # confianza mantenida cae por debajo del umbral aguas abajo, el keypoint deja
    # de contar de forma gradual.
    hold_conf_decay: float = 0.7


class _KeypointBuffer:
    """Historial temporal de UN keypoint de UNA persona."""

    def __init__(self, window: int) -> None:
        self._xy: deque = deque(maxlen=window)   # posiciones fiables recientes
        self._w: deque = deque(maxlen=window)    # confianzas (pesos) asociadas
        self._last_xy: np.ndarray | None = None  # ultimo valor fiable conocido
        self._last_conf: float = 0.0
        self._held: int = 0                      # frames consecutivos en oclusion

    def update(self, xy: np.ndarray, conf: float,
               cfg: SmoothingConfig) -> tuple[np.ndarray, float]:
        if conf >= cfg.min_keypoint_conf:
            self._xy.append(np.asarray(xy, dtype=float))
            self._w.append(float(conf))
            self._last_xy = self._weighted_mean()
            self._last_conf = float(conf)
            self._held = 0
            return self._last_xy.copy(), self._last_conf

        # Oclusion: mantenemos el ultimo valor fiable un numero limitado de
        # frames, con la confianza decayendo.
        if self._last_xy is not None and self._held < cfg.max_hold_frames:
            self._held += 1
            held_conf = self._last_conf * (cfg.hold_conf_decay ** self._held)
            return self._last_xy.copy(), held_conf

        # Oclusion prolongada (o nunca visto): se da por ausente.
        return np.asarray(xy, dtype=float), 0.0

    def _weighted_mean(self) -> np.ndarray:
        pts = np.stack(self._xy)
        ws = np.asarray(self._w)
        return (pts * ws[:, None]).sum(axis=0) / ws.sum()


class KeypointSmoother:
    """Suaviza keypoints por persona. Mantiene un buffer por (track_id, keypoint)."""

    def __init__(self, cfg: SmoothingConfig | None = None) -> None:
        self.cfg = cfg or SmoothingConfig()
        self._tracks: dict[int, list[_KeypointBuffer]] = {}

    def apply(self, track_id: int, keypoints: np.ndarray,
              conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (keypoints_suavizados, conf_suavizada) para una persona.

        keypoints: (K, 2) en pixeles. conf: (K,). No muta la entrada.
        """
        if not self.cfg.enabled:
            return keypoints, conf

        bufs = self._tracks.get(track_id)
        if bufs is None:
            bufs = [_KeypointBuffer(self.cfg.window) for _ in range(len(keypoints))]
            self._tracks[track_id] = bufs

        out_kp = np.array(keypoints, dtype=float, copy=True)
        out_conf = np.array(conf, dtype=float, copy=True)
        for i, buf in enumerate(bufs):
            xy, c = buf.update(keypoints[i], conf[i], self.cfg)
            out_kp[i] = xy
            out_conf[i] = c
        return out_kp, out_conf

    def drop(self, track_id: int) -> None:
        """Libera el estado de una persona que ya no esta en escena."""
        self._tracks.pop(track_id, None)

    def active_tracks(self) -> set[int]:
        return set(self._tracks)
