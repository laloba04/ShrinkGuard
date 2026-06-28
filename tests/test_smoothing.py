"""Tests del suavizado temporal y manejo de oclusiones (Fase 2).

Solo numpy: sin YOLO, OpenCV ni video.

Ejecutar:  python -m pytest tests/test_smoothing.py -q
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.smoothing import KeypointSmoother, SmoothingConfig  # noqa: E402

WRIST = 9  # un keypoint cualquiera para las pruebas


def _kp(x, y, conf=1.0):
    kp = np.zeros((17, 2), dtype=float)
    c = np.zeros(17, dtype=float)
    kp[WRIST] = (x, y)
    c[WRIST] = conf
    return kp, c


def test_suavizado_reduce_jitter():
    """La media movil reduce la dispersion del ruido del estimador."""
    sm = KeypointSmoother(SmoothingConfig(window=5))
    rng = np.random.default_rng(0)
    raw, out = [], []
    for _ in range(40):
        x = 100.0 + rng.normal(0, 5)   # posicion verdadera 100 + ruido
        kp, conf = _kp(x, 200.0)
        s_kp, _ = sm.apply(1, kp, conf)
        raw.append(x)
        out.append(s_kp[WRIST][0])
    # Tras llenar la ventana, la salida suavizada varia menos que la entrada.
    raw = np.array(raw[5:])
    out = np.array(out[5:])
    assert out.std() < raw.std(), f"suavizado no redujo jitter: {out.std()} vs {raw.std()}"


def test_oclusion_mantiene_ultimo_valor():
    """Un frame ocluido reutiliza la ultima posicion fiable con confianza decaida."""
    cfg = SmoothingConfig(window=3, max_hold_frames=3, hold_conf_decay=0.7)
    sm = KeypointSmoother(cfg)
    kp, conf = _kp(150, 198, 1.0)
    sm.apply(1, kp, conf)

    occ_kp, occ_conf = _kp(0, 0, 0.0)          # keypoint no fiable (oclusion)
    h_kp, h_conf = sm.apply(1, occ_kp, occ_conf)
    assert np.allclose(h_kp[WRIST], (150, 198)), "deberia mantener la ultima posicion fiable"
    assert 0.0 < h_conf[WRIST] < 1.0, f"la confianza mantenida deberia decaer, fue {h_conf[WRIST]}"


def test_oclusion_prolongada_da_conf_cero():
    """Superado max_hold_frames, el keypoint se da por ausente (conf 0)."""
    cfg = SmoothingConfig(window=3, max_hold_frames=2)
    sm = KeypointSmoother(cfg)
    kp, conf = _kp(150, 198, 1.0)
    sm.apply(1, kp, conf)

    occ_kp, occ_conf = _kp(0, 0, 0.0)
    sm.apply(1, occ_kp, occ_conf)              # hold 1
    sm.apply(1, occ_kp, occ_conf)              # hold 2 (== max)
    _, c = sm.apply(1, occ_kp, occ_conf)       # 3er frame -> ausente
    assert c[WRIST] == 0.0, f"oclusion prolongada deberia dar conf 0, fue {c[WRIST]}"


def test_desactivado_es_passthrough():
    """Con enabled=False la entrada sale intacta."""
    sm = KeypointSmoother(SmoothingConfig(enabled=False))
    kp, conf = _kp(123, 200, 0.9)
    s_kp, s_conf = sm.apply(1, kp, conf)
    assert np.array_equal(s_kp, kp)
    assert np.array_equal(s_conf, conf)


def test_drop_libera_estado():
    sm = KeypointSmoother()
    kp, conf = _kp(150, 198)
    sm.apply(5, kp, conf)
    assert 5 in sm.active_tracks()
    sm.drop(5)
    assert 5 not in sm.active_tracks()


def test_personas_independientes():
    """Cada track tiene su propio buffer: no se mezclan."""
    sm = KeypointSmoother(SmoothingConfig(window=5))
    a, ca = _kp(100, 200)
    b, cb = _kp(500, 200)
    for _ in range(5):
        sm.apply(1, a, ca)
        sm.apply(2, b, cb)
    sa, _ = sm.apply(1, a, ca)
    sb, _ = sm.apply(2, b, cb)
    assert np.allclose(sa[WRIST], (100, 200))
    assert np.allclose(sb[WRIST], (500, 200))


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
