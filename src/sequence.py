"""Representacion de secuencias de pose para el clasificador aprendido (Fase 3).

El clasificador de la Fase 3 aprende el GESTO a partir de SECUENCIAS de pose, en
lugar de aplicar una regla geometrica fija. Para que generalice entre distancias
y posiciones de camara, no trabaja sobre pixeles crudos sino sobre keypoints
NORMALIZADOS: centrados en el cuerpo y divididos por la longitud del torso
(misma filosofia que la heuristica).

Solo numpy: se testea sin modelo ni video. Aqui vive SOLO la representacion
(features + ventanas temporales); el modelo en si (PyTorch) ira en
`src/classifier.py`, aislado como `pose.py` aisla Ultralytics.
"""

from __future__ import annotations

import numpy as np

# COCO-17
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
N_KEYPOINTS = 17
# Por frame: (x, y) normalizados de cada keypoint + su confianza como canal.
FEATURES_PER_FRAME = N_KEYPOINTS * 3  # 34 coords + 17 confs = 51


def normalize_pose(keypoints: np.ndarray, conf: np.ndarray,
                   min_conf: float = 0.30) -> np.ndarray | None:
    """Normaliza una pose a coordenadas invariantes a traslacion y escala.

    Centro = punto medio del torso (media de hombros y caderas); escala =
    longitud del torso. Devuelve (17, 2) o None si faltan hombros/caderas
    fiables (sin ellos no hay referencia de cuerpo).
    """
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(conf[i] < min_conf for i in needed):
        return None
    shoulder_mid = (keypoints[L_SHOULDER] + keypoints[R_SHOULDER]) / 2.0
    hip_mid = (keypoints[L_HIP] + keypoints[R_HIP]) / 2.0
    center = (shoulder_mid + hip_mid) / 2.0
    torso = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso <= 1e-6:
        return None
    return (keypoints - center) / torso


def pose_features(keypoints: np.ndarray, conf: np.ndarray,
                  min_conf: float = 0.30) -> np.ndarray:
    """Vector de features de UN frame: (51,) = [x,y]*17 normalizados + conf*17.

    Los keypoints poco fiables se ponen a 0 en coordenada (pero su confianza, 0,
    queda como canal para que el modelo sepa que no son fiables). Si la pose no
    se puede normalizar, devuelve un vector de ceros (frame no informativo).
    """
    norm = normalize_pose(keypoints, conf, min_conf)
    if norm is None:
        return np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    reliable = conf >= min_conf
    xy = np.where(reliable[:, None], norm, 0.0)
    c = np.where(reliable, conf, 0.0)
    return np.concatenate([xy.reshape(-1), c]).astype(np.float32)


def sequence_features(seq_kp: list[np.ndarray], seq_conf: list[np.ndarray],
                      min_conf: float = 0.30) -> np.ndarray:
    """Convierte una secuencia de poses en una matriz (T, 51)."""
    return np.stack([pose_features(kp, cf, min_conf)
                     for kp, cf in zip(seq_kp, seq_conf)]) if seq_kp \
        else np.zeros((0, FEATURES_PER_FRAME), dtype=np.float32)


def sliding_windows(frames: np.ndarray, window: int,
                    stride: int = 1) -> list[np.ndarray]:
    """Ventanas deslizantes de una secuencia (T, F) -> lista de (window, F).

    Devuelve [] si la secuencia es mas corta que la ventana.
    """
    t = len(frames)
    if window <= 0 or t < window:
        return []
    return [frames[s:s + window] for s in range(0, t - window + 1, stride)]
