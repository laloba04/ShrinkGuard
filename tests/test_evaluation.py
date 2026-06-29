"""Tests de la evaluacion precision/recall (Fase 2).

Solo logica pura: sin YOLO, OpenCV ni video.

Ejecutar:  python -m pytest tests/test_evaluation.py -q
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import (  # noqa: E402
    GTInterval,
    Prediction,
    evaluate,
)


def test_acierto_perfecto():
    gt = [GTInterval("a.mp4", 2.0, 4.0)]
    preds = [Prediction("a.mp4", 3.0)]
    r = evaluate(gt, preds)
    assert r.recall == 1.0
    assert r.precision == 1.0
    assert r.f1 == 1.0


def test_falsa_alarma_baja_precision():
    gt = [GTInterval("a.mp4", 2.0, 4.0)]
    preds = [Prediction("a.mp4", 3.0), Prediction("a.mp4", 10.0)]  # 2a fuera
    r = evaluate(gt, preds)
    assert r.recall == 1.0
    assert r.precision == 0.5     # 1 de 2 predicciones acierta


def test_intervalo_no_detectado_baja_recall():
    gt = [GTInterval("a.mp4", 2.0, 4.0), GTInterval("a.mp4", 8.0, 9.0)]
    preds = [Prediction("a.mp4", 3.0)]   # solo cubre el primero
    r = evaluate(gt, preds)
    assert r.recall == 0.5
    assert r.precision == 1.0


def test_varias_predicciones_en_un_intervalo():
    """Varias predicciones dentro del mismo intervalo: el intervalo cuenta una
    vez para recall, pero todas las predicciones cuentan como acierto."""
    gt = [GTInterval("a.mp4", 2.0, 6.0)]
    preds = [Prediction("a.mp4", 3.0), Prediction("a.mp4", 4.0), Prediction("a.mp4", 5.0)]
    r = evaluate(gt, preds)
    assert r.tp_intervals == 1
    assert r.recall == 1.0
    assert r.tp_preds == 3
    assert r.precision == 1.0


def test_video_negativo_solo_genera_falsas_alarmas():
    gt = [GTInterval("robo.mp4", 1.0, 2.0)]
    preds = [Prediction("normal.mp4", 5.0)]   # disparo en un video sin positivos
    r = evaluate(gt, preds)
    assert r.recall == 0.0       # el intervalo de robo.mp4 no se detecto
    assert r.precision == 0.0    # la unica prediccion es falsa alarma


def test_separa_por_video():
    """Una prediccion en a.mp4 no debe puntuar contra un intervalo de b.mp4."""
    gt = [GTInterval("b.mp4", 2.0, 4.0)]
    preds = [Prediction("a.mp4", 3.0)]
    r = evaluate(gt, preds)
    assert r.recall == 0.0
    assert r.precision == 0.0


def test_tolerancia():
    gt = [GTInterval("a.mp4", 2.0, 4.0)]
    preds = [Prediction("a.mp4", 4.5)]      # 0.5s despues del fin
    assert evaluate(gt, preds, tol=0.0).precision == 0.0
    assert evaluate(gt, preds, tol=1.0).precision == 1.0


def test_sin_datos_no_revienta():
    r = evaluate([], [])
    assert r.recall == 0.0
    assert r.precision == 0.0
    assert r.f1 == 0.0


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
