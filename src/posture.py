"""Filtro de postura: analizar solo a personas DE PIE.

En una tienda los clientes estan de pie, asi que exigir postura erguida elimina
de golpe los falsos positivos de gente sentada (o con el tren inferior fuera de
plano, como en una webcam de escritorio). Pura geometria sobre keypoints: se
testea sin modelos ni video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class PostureConfig:
    min_keypoint_conf: float = 0.30
    # La pierna (cadera->tobillo, en vertical) debe medir al menos este factor
    # de la longitud del torso para considerar a la persona "de pie". Sentada o
    # con las piernas dobladas hacia delante, esta extension se acorta.
    min_leg_torso_ratio: float = 0.70


def _y(p) -> float:
    return float(p[1])


def _leg_span(keypoints, conf, hip_i, knee_i, ankle_i, c):
    """Extension vertical cadera->tobillo si la pierna es visible y apunta
    hacia abajo (de pie). Devuelve None si no se puede confirmar."""
    if conf[hip_i] < c or conf[knee_i] < c or conf[ankle_i] < c:
        return None
    hy, ky, ay = _y(keypoints[hip_i]), _y(keypoints[knee_i]), _y(keypoints[ankle_i])
    # En coordenadas de imagen, y crece hacia abajo: de pie -> cadera<rodilla<tobillo.
    if not (hy < ky < ay):
        return None
    return ay - hy


def is_standing(keypoints: np.ndarray, conf: np.ndarray,
                cfg: PostureConfig | None = None) -> bool:
    """True solo si podemos CONFIRMAR que la persona esta de pie.

    Conservador a proposito: si el tren inferior no se ve, devuelve False (no
    analizamos), que es justo lo que evita falsos positivos.
    """
    cfg = cfg or PostureConfig()
    c = cfg.min_keypoint_conf

    # Sin hombros y caderas no hay escala de torso: no decidimos.
    for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP):
        if conf[i] < c:
            return False

    shoulder_y = (_y(keypoints[L_SHOULDER]) + _y(keypoints[R_SHOULDER])) / 2
    hip_y = (_y(keypoints[L_HIP]) + _y(keypoints[R_HIP])) / 2
    torso = abs(hip_y - shoulder_y)
    if torso <= 1e-6:
        return False

    for hip_i, knee_i, ankle_i in ((L_HIP, L_KNEE, L_ANKLE),
                                   (R_HIP, R_KNEE, R_ANKLE)):
        span = _leg_span(keypoints, conf, hip_i, knee_i, ankle_i, c)
        if span is not None and span >= cfg.min_leg_torso_ratio * torso:
            return True
    return False