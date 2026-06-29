"""Construye el dataset de entrenamiento del clasificador (Fase 3).

Corre la pose sobre videos etiquetados, agrupa los keypoints POR PERSONA (track),
trocea cada track en ventanas temporales de pose normalizada y etiqueta cada
ventana (positiva si su tramo temporal cae en un intervalo de hurto del video).
Guarda X (N, window, 51) e y (N,) en un .npz.

Etiquetado: usa los mismos labels.csv (video,inicio,fin) que `evaluar.py`. Con
labels a nivel de clip (intervalo que cubre todo el video) todas las ventanas de
un clip positivo se marcan positivas: es ETIQUETADO DEBIL (hay ruido, porque no
todo el clip es el gesto), aceptable para una primera linea base; se puede afinar
con etiquetas por subclip (DCSASS) o por rango de frames (CCTV) mas adelante.

Uso (un .npz por dataset; el entrenador puede cargar varios):
    python tools/preparar_dataset.py --videos-dir "datos/clips/Shoplifting dataset" \
        --labels datos/labels.csv --out datos/ds_640.npz --device 0
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import AppConfig  # noqa: E402
from src.pose import PoseEstimator  # noqa: E402
from src.sequence import pose_features, sliding_windows  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def cargar_labels(path: Path) -> dict[str, list[tuple[float, float]]]:
    gt: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = (row.get("video") or "").strip()
            ini = (row.get("inicio") or "").strip()
            fin = (row.get("fin") or "").strip()
            if v and ini and fin:
                gt[v].append((float(ini), float(fin)))
    return gt


def _reset_tracker(est: PoseEstimator) -> None:
    try:
        for tr in est.model.predictor.trackers:
            tr.reset()
    except Exception:
        pass


def _ventana_positiva(frames_idx: list[int], s: int, window: int, fps: float,
                      intervalos: list[tuple[float, float]]) -> bool:
    """¿El tramo temporal de la ventana solapa algun intervalo positivo?"""
    if not intervalos:
        return False
    t0 = frames_idx[s] / fps
    t1 = frames_idx[s + window - 1] / fps
    return any(a <= t1 and t0 <= b for a, b in intervalos)


def procesar_video(path: Path, est: PoseEstimator, intervalos, window, stride,
                   min_conf):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  (no pude abrir {path.name}, lo salto)")
        return [], []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # por track: lista de (frame_idx, features)
    tracks: dict[int, list] = defaultdict(list)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for p in est.track(frame):
            tracks[p.track_id].append((idx, pose_features(p.keypoints, p.conf, min_conf)))
        idx += 1
    cap.release()

    X, y = [], []
    for _tid, seq in tracks.items():
        if len(seq) < window:
            continue
        frames_idx = [fi for fi, _ in seq]
        feats = np.stack([fe for _, fe in seq])
        for s, win in zip(range(0, len(feats) - window + 1, stride),
                          sliding_windows(feats, window, stride)):
            X.append(win)
            y.append(1 if _ventana_positiva(frames_idx, s, window, fps, intervalos) else 0)
    # grupo = nombre del video (para dividir train/val sin fuga de datos)
    return X, y, [path.name] * len(X)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepara dataset de ventanas de pose (Fase 3)")
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True, help="ruta del .npz de salida")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.videos_dir).rglob("*")
                    if p.suffix.lower() in VIDEO_EXT)
    if not videos:
        raise SystemExit(f"No hay videos en {args.videos_dir}")
    gt = cargar_labels(Path(args.labels))

    cfg = AppConfig(model_name=args.model, device=args.device)
    est = PoseEstimator(cfg)
    min_conf = cfg.concealment.min_keypoint_conf

    allX, allY, allG = [], [], []
    for v in videos:
        _reset_tracker(est)
        X, y, g = procesar_video(v, est, gt.get(v.name, []), args.window,
                                 args.stride, min_conf)
        allX += X
        allY += y
        allG += g
        pos = sum(y)
        print(f"  {v.name}: {len(y)} ventanas ({pos} pos / {len(y) - pos} neg)")

    if not allX:
        raise SystemExit("No se genero ninguna ventana (¿tracks mas cortos que la ventana?)")
    X = np.stack(allX).astype(np.float32)
    y = np.asarray(allY, dtype=np.int64)
    groups = np.asarray(allG)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, groups=groups)
    print(f"\nGuardado {out}: X={X.shape}, positivos={int(y.sum())}/{len(y)}")


if __name__ == "__main__":
    main()
