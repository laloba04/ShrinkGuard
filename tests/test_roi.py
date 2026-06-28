"""Tests de las zonas de interes (ROI) (Fase 2).

Solo numpy: sin YOLO, OpenCV ni video.

Ejecutar:  python -m pytest tests/test_roi.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.roi import (  # noqa: E402
    ROIConfig,
    in_roi,
    parse_roi_arg,
    person_anchor,
    point_in_polygon,
)

L_HIP, R_HIP = 11, 12

# Rectangulo: mitad izquierda del frame normalizado.
LEFT_HALF = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]


def _person_at(hx, hy, conf=1.0):
    """Persona cuyas caderas se centran en (hx, hy) en pixeles."""
    kp = np.zeros((17, 2), dtype=float)
    c = np.zeros(17, dtype=float)
    kp[L_HIP] = (hx - 20, hy)
    kp[R_HIP] = (hx + 20, hy)
    c[L_HIP] = c[R_HIP] = conf
    return kp, c


# --- point_in_polygon ---

def test_punto_dentro_del_rectangulo():
    assert point_in_polygon((0.25, 0.5), LEFT_HALF)


def test_punto_fuera_del_rectangulo():
    assert not point_in_polygon((0.75, 0.5), LEFT_HALF)


def test_poligono_degenerado_es_falso():
    assert not point_in_polygon((0.1, 0.1), [(0.0, 0.0), (1.0, 1.0)])


# --- person_anchor ---

def test_anchor_usa_punto_medio_de_caderas():
    kp, conf = _person_at(300, 240)
    a = person_anchor(kp, conf)
    assert np.allclose(a, (300, 240))


def test_anchor_cae_a_media_si_no_hay_caderas():
    kp, conf = _person_at(300, 240)
    conf[L_HIP] = conf[R_HIP] = 0.05   # caderas no fiables
    kp[5] = (100, 100); conf[5] = 0.9  # un hombro fiable
    a = person_anchor(kp, conf)
    assert np.allclose(a, (100, 100))


def test_anchor_none_si_nada_fiable():
    kp = np.zeros((17, 2)); conf = np.zeros(17)
    assert person_anchor(kp, conf) is None


# --- in_roi ---

def test_roi_desactivada_siempre_dentro():
    kp, conf = _person_at(600, 240)
    cfg = ROIConfig(enabled=False, polygon=LEFT_HALF)
    assert in_roi(kp, conf, 640, 480, cfg)


def test_in_roi_persona_dentro():
    kp, conf = _person_at(100, 240)          # x=100/640=0.156 -> mitad izquierda
    cfg = ROIConfig(enabled=True, polygon=LEFT_HALF)
    assert in_roi(kp, conf, 640, 480, cfg)


def test_in_roi_persona_fuera():
    kp, conf = _person_at(500, 240)          # x=500/640=0.78 -> mitad derecha
    cfg = ROIConfig(enabled=True, polygon=LEFT_HALF)
    assert not in_roi(kp, conf, 640, 480, cfg)


def test_in_roi_sin_posicion_fiable_es_conservador():
    kp = np.zeros((17, 2)); conf = np.zeros(17)
    cfg = ROIConfig(enabled=True, polygon=LEFT_HALF)
    assert not in_roi(kp, conf, 640, 480, cfg), "sin posicion fiable -> fuera (no analizar)"


# --- parse_roi_arg ---

def test_parse_roi_valido():
    pts = parse_roi_arg(["0.1,0.2", "0.9,0.2", "0.5,0.9"])
    assert pts == [(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)]


def test_parse_roi_pocos_vertices():
    with pytest.raises(ValueError):
        parse_roi_arg(["0.1,0.2", "0.9,0.2"])


def test_parse_roi_fuera_de_rango():
    with pytest.raises(ValueError):
        parse_roi_arg(["0.1,0.2", "1.5,0.2", "0.5,0.9"])


def test_parse_roi_formato_invalido():
    with pytest.raises(ValueError):
        parse_roi_arg(["0.1", "0.9,0.2", "0.5,0.9"])


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
