"""Tests de la representacion de secuencias de pose (Fase 3).

Solo numpy: sin modelo ni video.

Ejecutar:  python -m pytest tests/test_sequence.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sequence import (  # noqa: E402
    FEATURES_PER_FRAME,
    normalize_pose,
    pose_features,
    sequence_features,
    sliding_windows,
)

L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def _person(cx=150.0, cy=150.0, scale=1.0):
    """Persona centrada en (cx,cy) con torso=100*scale. Devuelve (kp, conf)."""
    kp = np.zeros((17, 2), dtype=float)
    conf = np.ones(17, dtype=float)
    kp[L_SHOULDER] = (cx - 20 * scale, cy - 50 * scale)
    kp[R_SHOULDER] = (cx + 20 * scale, cy - 50 * scale)
    kp[L_HIP] = (cx - 20 * scale, cy + 50 * scale)
    kp[R_HIP] = (cx + 20 * scale, cy + 50 * scale)
    kp[9] = (cx, cy)   # muñeca en el centro del torso
    return kp, conf


def test_normalize_centra_y_escala():
    kp, conf = _person(150, 150, 1.0)
    n = normalize_pose(kp, conf)
    # hombros y caderas quedan simetricos respecto al centro, a +-0.5 de torso
    sh = (n[L_SHOULDER] + n[R_SHOULDER]) / 2
    hp = (n[L_HIP] + n[R_HIP]) / 2
    assert np.allclose(sh, (0.0, -0.5), atol=1e-6)
    assert np.allclose(hp, (0.0, 0.5), atol=1e-6)


def test_normalize_invariante_a_traslacion_y_escala():
    a_kp, a_conf = _person(150, 150, 1.0)
    b_kp, b_conf = _person(600, 50, 3.0)   # otra posicion y tamaño
    na = normalize_pose(a_kp, a_conf)
    nb = normalize_pose(b_kp, b_conf)
    # comparar solo los keypoints colocados de forma consistente respecto al cuerpo
    idx = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, 9]
    assert np.allclose(na[idx], nb[idx], atol=1e-6), \
        "la pose normalizada debe ser invariante a traslacion/escala"


def test_normalize_none_sin_torso():
    kp, conf = _person()
    conf[L_HIP] = conf[R_HIP] = 0.05   # caderas no fiables
    assert normalize_pose(kp, conf) is None


def test_pose_features_shape_y_ceros():
    kp, conf = _person()
    f = pose_features(kp, conf)
    assert f.shape == (FEATURES_PER_FRAME,)
    # sin torso -> vector de ceros
    kp2, conf2 = _person()
    conf2[L_SHOULDER] = 0.0
    assert np.count_nonzero(pose_features(kp2, conf2)) == 0


def test_pose_features_keypoint_no_fiable_va_a_cero():
    kp, conf = _person()
    conf[9] = 0.1   # muñeca no fiable
    f = pose_features(kp, conf)
    # coordenadas de la muñeca (indices 18,19) a 0 y su conf (34+9=43) a 0
    assert f[18] == 0.0 and f[19] == 0.0
    assert f[34 + 9] == 0.0


def test_sequence_features_apila():
    kp, conf = _person()
    seq = sequence_features([kp, kp, kp], [conf, conf, conf])
    assert seq.shape == (3, FEATURES_PER_FRAME)


def test_sequence_features_vacia():
    seq = sequence_features([], [])
    assert seq.shape == (0, FEATURES_PER_FRAME)


def test_sliding_windows_cuenta():
    frames = np.arange(10 * 51, dtype=float).reshape(10, 51)
    w = sliding_windows(frames, window=4, stride=2)
    assert len(w) == 4               # s = 0,2,4,6
    assert all(x.shape == (4, 51) for x in w)


def test_sliding_windows_corta_devuelve_vacio():
    frames = np.zeros((3, 51))
    assert sliding_windows(frames, window=8) == []


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{ok}/{len(tests)} tests OK")
    sys.exit(0 if ok == len(tests) else 1)
