"""Tests de la heuristica de ocultacion + filtro de postura (datos sinteticos).

No necesita YOLO, OpenCV ni video: solo numpy. Demuestra que la logica es
correcta de forma reproducible.

Ejecutar:  python -m pytest -q   (o)   python tests/test_concealment.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.concealment import (  # noqa: E402
    ConcealmentConfig,
    ConcealmentDetector,
    concealment_score,
)
from src.posture import PostureConfig, is_standing  # noqa: E402
from src.smoothing import SmoothingConfig  # noqa: E402
from src.roi import ROIConfig  # noqa: E402

_NO_SMOOTH = SmoothingConfig(enabled=False)


# ---------------------------------------------------------------------------
# Helpers de datos sinteticos
# ---------------------------------------------------------------------------

def _person(wrist_xy):
    """Persona DE PIE con ambas muñecas en wrist_xy.

    Geometria:
      hombros  y=100  (x=130 izq, x=170 der)
      caderas  y=200  (x=130 izq, x=170 der)  -> torso=100
      rodillas y=310  (drop=1.10*torso >= 0.70 -> de pie segun posture.py)
      tobillos y=420  (span cadera->tobillo = 220 >= 0.70*100 -> confirmado)
    """
    kp = np.zeros((17, 2), dtype=float)
    conf = np.ones(17, dtype=float)
    kp[5]  = (130, 100)   # hombro izq
    kp[6]  = (170, 100)   # hombro der
    kp[11] = (130, 200)   # cadera izq
    kp[12] = (170, 200)   # cadera der
    kp[13] = (130, 310)   # rodilla izq
    kp[14] = (170, 310)   # rodilla der
    kp[15] = (130, 420)   # tobillo izq
    kp[16] = (170, 420)   # tobillo der
    kp[9]  = wrist_xy     # muñeca izq
    kp[10] = wrist_xy     # muñeca der
    return kp, conf


def _person_sitting(wrist_xy):
    """Persona SENTADA: rodillas por ENCIMA de las caderas (piernas hacia arriba).

    La condicion hip_y < knee_y < ankle_y falla -> is_standing devuelve False.
    """
    kp, conf = _person(wrist_xy)
    kp[13] = (130, 185)   # rodilla izq encima de la cadera (y=200)
    kp[14] = (170, 185)   # rodilla der
    kp[15] = (130, 195)   # tobillo por encima de la cadera tambien
    kp[16] = (170, 195)
    return kp, conf


def _person_no_ankles(wrist_xy):
    """Persona con el tren inferior fuera de plano (baja confianza en tobillos)."""
    kp, conf = _person(wrist_xy)
    conf[15] = 0.05   # tobillo izq no fiable
    conf[16] = 0.05   # tobillo der no fiable
    return kp, conf


# ---------------------------------------------------------------------------
# Tests de la heuristica de ocultacion (concealment_score / ConcealmentDetector)
# ---------------------------------------------------------------------------

def test_mano_en_cinturilla_da_score_alto():
    kp, conf = _person((150, 198))   # mano centrada, justo en la cintura
    s = concealment_score(kp, conf, ConcealmentConfig())
    assert s > 0.8, f"esperaba score alto, fue {s}"


def test_mano_en_pecho_da_score_alto():
    # Ocultacion ALTA (meter algo en la chaqueta): muñeca a la altura del pecho,
    # centrada. Con _person el ancla de pecho cae en y~145.
    kp, conf = _person((150, 145))
    s = concealment_score(kp, conf, ConcealmentConfig())
    assert s > 0.8, f"esperaba score alto para mano en el pecho, fue {s}"


def test_brazo_al_costado_no_dispara():
    kp, conf = _person((105, 190))   # muñeca fuera de la columna del torso
    s = concealment_score(kp, conf, ConcealmentConfig())
    assert s == 0.0, f"brazo al costado no deberia puntuar, fue {s}"


def test_keypoints_poco_fiables_devuelven_cero():
    kp, conf = _person((150, 198))
    conf[11] = 0.05   # cadera izq con baja confianza
    s = concealment_score(kp, conf, ConcealmentConfig())
    assert s == 0.0


def test_evento_tras_frames_consecutivos():
    cfg = ConcealmentConfig(consecutive_frames=8)
    det = ConcealmentDetector(cfg)
    kp, conf = _person((150, 198))

    fired = []
    for f in range(7):                         # 7 frames: aun no debe disparar
        fired += det.update(f, [(1, kp, conf)])
    assert fired == []

    fired += det.update(7, [(1, kp, conf)])    # frame 8 -> dispara
    assert len(fired) == 1
    assert fired[0].track_id == 1
    assert fired[0].score > 0.8


def test_cooldown_evita_spam():
    cfg = ConcealmentConfig(consecutive_frames=4, cooldown_frames=10)
    det = ConcealmentDetector(cfg)
    kp, conf = _person((150, 198))

    total = []
    for f in range(20):
        total += det.update(f, [(1, kp, conf)])
    assert 1 <= len(total) <= 3, f"cooldown no limita: {len(total)} eventos"


def test_dos_personas_independientes():
    det = ConcealmentDetector(ConcealmentConfig(consecutive_frames=4))
    sospechoso = _person((150, 198))   # mano en cintura
    inocente   = _person((105, 190))   # brazo al costado

    fired = []
    for f in range(6):
        fired += det.update(f, [(1, *sospechoso), (2, *inocente)])
    ids = {e.track_id for e in fired}
    assert ids == {1}, f"solo la persona 1 deberia disparar, fueron {ids}"


# ---------------------------------------------------------------------------
# Tests del filtro de postura (is_standing / require_standing)
# ---------------------------------------------------------------------------

def test_is_standing_de_pie():
    """is_standing devuelve True para una persona de pie con piernas visibles."""
    kp, conf = _person((150, 198))
    assert is_standing(kp, conf, PostureConfig()), \
        "persona de pie deberia pasar el filtro"


def test_is_standing_sentado():
    """is_standing devuelve False cuando las rodillas estan por encima de la cadera."""
    kp, conf = _person_sitting((150, 198))
    assert not is_standing(kp, conf, PostureConfig()), \
        "persona sentada no deberia pasar el filtro"


def test_piernas_no_visibles_no_analiza():
    """Sin tobillos visibles is_standing es conservador: devuelve False
    (no podemos confirmar que este de pie -> no analizamos)."""
    kp, conf = _person_no_ankles((150, 198))
    assert not is_standing(kp, conf, PostureConfig()), \
        "sin tobillos no se puede confirmar postura -> False"


def test_persona_sentada_no_dispara():
    """Con require_standing=True (default), persona sentada no genera eventos
    aunque la mano este en la cinturilla."""
    cfg = ConcealmentConfig(consecutive_frames=4)
    det = ConcealmentDetector(cfg, require_standing=True)
    kp, conf = _person_sitting((150, 198))

    fired = []
    for f in range(10):
        fired += det.update(f, [(1, kp, conf)])
    assert fired == [], \
        f"persona sentada no debe disparar, pero disparo {len(fired)} veces"


def test_require_standing_false_permite_sentados():
    """Con --no-posture-filter (require_standing=False) una persona sentada
    con la mano en la cinturilla SI puede disparar una señal."""
    cfg = ConcealmentConfig(consecutive_frames=4)
    det = ConcealmentDetector(cfg, require_standing=False)
    kp, conf = _person_sitting((150, 198))

    fired = []
    for f in range(6):
        fired += det.update(f, [(1, kp, conf)])
    assert len(fired) >= 1, \
        "con filtro desactivado, persona sentada deberia poder disparar"


# ---------------------------------------------------------------------------
# Tests de robustez Fase 2: tolerancia a huecos + manejo de oclusiones
# ---------------------------------------------------------------------------

def test_gap_tolerante_a_hueco_breve():
    """Un hueco de 1 frame (no near) no reinicia el contador de consecutivos.

    Suavizado desactivado para aislar la politica de max_gap_frames.
    """
    cfg = ConcealmentConfig(consecutive_frames=5, max_gap_frames=2)
    det = ConcealmentDetector(cfg, require_standing=False, smoothing_cfg=_NO_SMOOTH)
    sus = _person((150, 198))    # mano en cintura -> near
    fuera = _person((105, 190))  # brazo al costado -> no near

    fired = []
    for f in range(4):                          # 4 frames near
        fired += det.update(f, [(1, *sus)])
    fired += det.update(4, [(1, *fuera)])       # 1 frame de hueco
    fired += det.update(5, [(1, *sus)])         # vuelve near -> 5o consecutivo
    assert len(fired) == 1, f"el hueco breve no deberia romper la secuencia, fueron {len(fired)}"


def test_sin_tolerancia_el_hueco_reinicia():
    """Con max_gap_frames=0 un solo frame no-near reinicia el contador."""
    cfg = ConcealmentConfig(consecutive_frames=5, max_gap_frames=0)
    det = ConcealmentDetector(cfg, require_standing=False, smoothing_cfg=_NO_SMOOTH)
    sus = _person((150, 198))
    fuera = _person((105, 190))

    fired = []
    for f in range(4):
        fired += det.update(f, [(1, *sus)])
    fired += det.update(4, [(1, *fuera)])       # hueco -> reinicia
    fired += det.update(5, [(1, *sus)])
    assert fired == [], f"sin tolerancia el hueco deberia reiniciar, fueron {len(fired)}"


def test_oclusion_hold_puentea_dropout():
    """Con suavizado, un frame con keypoints no fiables (oclusion) se puentea:
    el hold mantiene la ultima pose fiable y la senal no se interrumpe."""
    cfg = ConcealmentConfig(consecutive_frames=6, max_gap_frames=0)
    det = ConcealmentDetector(cfg, require_standing=False)  # suavizado on (default)
    sus = _person((150, 198))
    drop_kp, drop_conf = _person((150, 198))
    drop_conf[:] = 0.0                          # dropout total de confianza

    fired = []
    for f in range(5):
        fired += det.update(f, [(1, *sus)])
    fired += det.update(5, [(1, drop_kp, drop_conf)])   # ocluido -> hold puentea
    assert len(fired) == 1, f"el hold deberia puentear el dropout, fueron {len(fired)}"


def test_sin_hold_el_dropout_interrumpe():
    """Sin suavizado, el mismo dropout reinicia el contador (no hay hold)."""
    cfg = ConcealmentConfig(consecutive_frames=6, max_gap_frames=0)
    det = ConcealmentDetector(cfg, require_standing=False, smoothing_cfg=_NO_SMOOTH)
    sus = _person((150, 198))
    drop_kp, drop_conf = _person((150, 198))
    drop_conf[:] = 0.0

    fired = []
    for f in range(5):
        fired += det.update(f, [(1, *sus)])
    fired += det.update(5, [(1, drop_kp, drop_conf)])
    fired += det.update(6, [(1, *sus)])
    assert fired == [], f"sin hold el dropout deberia interrumpir, fueron {len(fired)}"


# ---------------------------------------------------------------------------
# Tests de integracion de la ROI en el detector
# ---------------------------------------------------------------------------

def test_roi_excluye_persona_fuera():
    """Con ROI activa, una persona fuera del area no dispara aunque tenga la
    mano en la cinturilla. _person tiene las caderas en x~150; con un frame de
    640 px de ancho cae en la mitad izquierda, asi que una ROI en la mitad
    DERECHA la deja fuera."""
    roi = ROIConfig(enabled=True,
                    polygon=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])
    det = ConcealmentDetector(ConcealmentConfig(consecutive_frames=3),
                              require_standing=False, roi_cfg=roi)
    kp, conf = _person((150, 198))
    fired = []
    for f in range(6):
        fired += det.update(f, [(1, kp, conf)], frame_wh=(640, 480))
    assert fired == [], f"persona fuera de la ROI no deberia disparar, fueron {len(fired)}"


def test_roi_incluye_persona_dentro():
    """La misma persona con una ROI en la mitad IZQUIERDA cae dentro y dispara."""
    roi = ROIConfig(enabled=True,
                    polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])
    det = ConcealmentDetector(ConcealmentConfig(consecutive_frames=3),
                              require_standing=False, roi_cfg=roi)
    kp, conf = _person((150, 198))
    fired = []
    for f in range(6):
        fired += det.update(f, [(1, kp, conf)], frame_wh=(640, 480))
    assert len(fired) >= 1, "persona dentro de la ROI deberia disparar"


# ---------------------------------------------------------------------------
# Runner manual (sin pytest)
# ---------------------------------------------------------------------------

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
