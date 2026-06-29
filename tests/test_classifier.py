"""Tests del clasificador aprendido (Fase 3).

Requiere PyTorch (ya viene con Ultralytics). Verifica el modelo, la persistencia
y que LearnedConcealmentDetector cumple la MISMA interfaz que ConcealmentDetector
(es intercambiable en el pipeline).

Ejecutar:  python -m pytest tests/test_classifier.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

torch = pytest.importorskip("torch")

from src.classifier import (  # noqa: E402
    LearnedConcealmentDetector,
    PoseLSTM,
    load_model,
    save_model,
)
from src.concealment import ConcealmentConfig  # noqa: E402
from src.sequence import FEATURES_PER_FRAME  # noqa: E402


def _person(wrist_xy=(150, 198)):
    kp = np.zeros((17, 2), dtype=float)
    conf = np.ones(17, dtype=float)
    kp[5], kp[6] = (130, 100), (170, 100)
    kp[11], kp[12] = (130, 200), (170, 200)
    kp[9], kp[10] = wrist_xy, wrist_xy
    return kp, conf


def test_modelo_forward_shape():
    m = PoseLSTM(hidden=16)
    x = torch.zeros(4, 32, FEATURES_PER_FRAME)
    assert m(x).shape == (4, 2)


def test_guardar_cargar_roundtrip(tmp_path):
    m = PoseLSTM(hidden=16).eval()
    p = tmp_path / "m.pt"
    save_model(m, str(p), window=32)
    m2, window = load_model(str(p))
    assert window == 32
    x = torch.zeros(1, 32, FEATURES_PER_FRAME)
    with torch.no_grad():
        assert torch.allclose(m(x), m2(x), atol=1e-6)


def test_detector_aprendido_misma_interfaz(tmp_path):
    """LearnedConcealmentDetector.update(...) devuelve eventos como el heuristico."""
    p = tmp_path / "m.pt"
    save_model(PoseLSTM(hidden=16), str(p), window=8)
    det = LearnedConcealmentDetector(str(p), cfg=ConcealmentConfig(consecutive_frames=3))
    kp, conf = _person()
    out = []
    for f in range(12):                      # > window para que infiera
        out += det.update(f, [(1, kp, conf)])
    # interfaz: devuelve lista y expone current_score (no afirmamos disparo:
    # el modelo esta sin entrenar)
    assert isinstance(out, list)
    assert isinstance(det.current_score(1), float)


def test_detector_aprendido_limpia_tracks(tmp_path):
    p = tmp_path / "m.pt"
    save_model(PoseLSTM(hidden=16), str(p), window=8)
    det = LearnedConcealmentDetector(str(p))
    kp, conf = _person()
    det.update(0, [(7, kp, conf)])
    assert 7 in det._states
    det.update(1, [])                        # el track 7 desaparece
    assert 7 not in det._states


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        try:
            import inspect
            if "tmp_path" in inspect.signature(t).parameters:
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"PASS  {t.__name__}")
            ok += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{ok}/{len(tests)} tests OK")
    sys.exit(0 if ok == len(tests) else 1)
