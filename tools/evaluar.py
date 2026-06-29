"""Calibracion de umbrales: precision / recall del detector sobre videos
etiquetados, con barrido de umbral (Fase 2).

Estrategia: la pose (lo caro) se ejecuta UNA sola vez por video y se cachea en
memoria; despues se corre el detector sobre esa cache con cada umbral candidato.
Asi el barrido es rapido y los resultados son comparables entre umbrales.

Formato de etiquetas (CSV con cabecera):
    video,inicio,fin
    robo1.mp4,3.5,6.0
    robo1.mp4,12.0,14.5
Cada fila es un intervalo POSITIVO (segundos). Los videos de --videos-dir que
no aparezcan en el CSV se tratan como 100% negativos (solo pueden dar falsas
alarmas).

Uso:
    python tools/evaluar.py --videos-dir datos/clips --labels datos/labels.csv \
        --sweep 0.30 0.40 0.45 0.50 0.55 0.60 --device 0
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.concealment import ConcealmentDetector  # noqa: E402
from src.config import AppConfig  # noqa: E402
from src.evaluation import GTInterval, Prediction, evaluate  # noqa: E402
from src.pose import PoseEstimator  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def cargar_labels(path: Path) -> list[GTInterval]:
    gt: list[GTInterval] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = (row.get("video") or "").strip()
            inicio = (row.get("inicio") or "").strip()
            fin = (row.get("fin") or "").strip()
            if not video or not inicio or not fin:
                continue
            gt.append(GTInterval(video, float(inicio), float(fin)))
    return gt


def _reset_tracker(est: PoseEstimator) -> None:
    """Resetea el tracker entre videos para que los IDs no se arrastren."""
    try:
        for tr in est.model.predictor.trackers:
            tr.reset()
    except Exception:
        pass  # mejor esfuerzo; si la API cambia, no es critico


def cachear_pose(video_path: Path, est: PoseEstimator):
    """Devuelve (secuencia, fps, (w,h)). secuencia[i] = lista de (id, kp, conf)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    seq = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        people = est.track(frame)
        seq.append([(p.track_id, p.keypoints, p.conf) for p in people])
    cap.release()
    return seq, fps, (w, h)


def predicciones_para_umbral(cache: dict, cfg: AppConfig, umbral: float,
                             learned_det=None) -> list[Prediction]:
    """Corre el detector sobre la cache con un umbral dado.

    Si learned_det es un LearnedConcealmentDetector (ya cargado), se usa con
    `umbral` como umbral de probabilidad (reseteando su estado por video); si no,
    la heuristica con `umbral` como near_score_threshold.
    """
    cfg.concealment.near_score_threshold = umbral
    preds: list[Prediction] = []
    for video, (seq, fps, wh) in cache.items():
        if learned_det is not None:
            learned_det.threshold = umbral
            learned_det.reset()
            det = learned_det
        else:
            det = ConcealmentDetector(cfg.concealment, posture_cfg=cfg.posture,
                                      require_standing=cfg.require_standing,
                                      smoothing_cfg=cfg.smoothing, roi_cfg=cfg.roi)
        for idx, payload in enumerate(seq):
            for ev in det.update(idx, payload, frame_wh=wh):
                preds.append(Prediction(video, ev.frame_idx / fps))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="ShrinkGuard - calibracion precision/recall")
    ap.add_argument("--videos-dir", required=True, help="carpeta con los clips a evaluar")
    ap.add_argument("--labels", required=True, help="CSV con intervalos positivos")
    ap.add_argument("--sweep", nargs="+", type=float,
                    default=[0.30, 0.40, 0.45, 0.50, 0.55, 0.60],
                    help="umbrales near_score_threshold a probar")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="tolerancia en segundos al casar prediccion con intervalo")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default="cpu", help="cpu | 0 | mps")
    ap.add_argument("--consecutive", type=int, default=8)
    ap.add_argument("--no-posture-filter", action="store_true")
    ap.add_argument("--no-smoothing", action="store_true")
    ap.add_argument("--learned", default=None,
                    help="ruta a un modelo PoseLSTM: usa el clasificador aprendido "
                         "en vez de la heuristica (el sweep son umbrales de probabilidad)")
    ap.add_argument("--only", default=None,
                    help="fichero con nombres de video (uno por linea) a evaluar; "
                         "el resto se ignora (p. ej. el split de validacion)")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    # rglob: escanea tambien subcarpetas (p. ej. Shoplifting/ y Normal/).
    videos = sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if args.only:
        keep = {ln.strip() for ln in Path(args.only).read_text().splitlines() if ln.strip()}
        videos = [v for v in videos if v.name in keep]
    if not videos:
        raise SystemExit(f"No encontre videos en {videos_dir}")
    gt = cargar_labels(Path(args.labels))
    # Restringir la verdad-terreno a los videos realmente evaluados (si no, los
    # positivos de videos no evaluados contarian siempre como fallos -> recall mal).
    eval_names = {v.name for v in videos}
    gt = [g for g in gt if g.video in eval_names]

    cfg = AppConfig(model_name=args.model, device=args.device,
                    require_standing=not args.no_posture_filter)
    cfg.concealment.consecutive_frames = args.consecutive
    cfg.smoothing.enabled = not args.no_smoothing

    print(f"Cacheando pose de {len(videos)} videos (una pasada)...")
    est = PoseEstimator(cfg)
    cache: dict = {}
    for v in videos:
        _reset_tracker(est)
        seq, fps, wh = cachear_pose(v, est)
        cache[v.name] = (seq, fps, wh)
        print(f"  {v.name}: {len(seq)} frames @ {fps:.1f} fps")

    pos_intervalos = len(gt)
    print(f"\nVerdad-terreno: {pos_intervalos} intervalos positivos en "
          f"{len({g.video for g in gt})} videos.\n")

    print(f"{'umbral':>7} | {'precision':>9} | {'recall':>7} | {'F1':>5} | "
          f"{'TP':>3} {'FN':>3} {'FP':>3}")
    print("-" * 56)
    # El clasificador aprendido se carga UNA vez y se reutiliza (reset por video).
    learned_det = None
    if args.learned:
        from src.classifier import LearnedConcealmentDetector
        torch_dev = f"cuda:{args.device}" if args.device.isdigit() else args.device
        learned_det = LearnedConcealmentDetector(
            args.learned, cfg=cfg.concealment, threshold=args.sweep[0],
            device=torch_dev, smoothing_cfg=cfg.smoothing)

    best = None
    for u in args.sweep:
        preds = predicciones_para_umbral(cache, cfg, u, learned_det=learned_det)
        r = evaluate(gt, preds, tol=args.tol)
        print(f"{u:>7.2f} | {r.precision:>9.3f} | {r.recall:>7.3f} | "
              f"{r.f1:>5.3f} | {r.tp_intervals:>3} {r.fn_intervals:>3} {r.fp_preds:>3}")
        if best is None or r.f1 > best[1]:
            best = (u, r.f1)
    if best:
        print(f"\nMejor F1: umbral={best[0]:.2f} (F1={best[1]:.3f})")


if __name__ == "__main__":
    main()
