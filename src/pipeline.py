"""Pipeline de la Fase 1: video -> pose+tracking -> ocultacion -> revision.

Lee un video (archivo o webcam), estima pose multi-persona con tracking,
aplica la heuristica de ocultacion y, cuando una persona dispara una señal,
guarda un recorte con marca de tiempo en `salidas/revision/` para que una
persona lo valide. Nada se decide de forma automatica.
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.concealment import ConcealmentDetector
from src.config import AppConfig
from src.pose import PoseEstimator
from src.visualizer import (
    draw_alert_banner,
    draw_footer,
    draw_person,
    draw_roi,
)


def _open_source(source: str, cfg: AppConfig, dshow: bool = False) -> cv2.VideoCapture:
    """Abre la fuente de video.

    dshow=True usa el backend DirectShow de Windows (cv2.CAP_DSHOW), que suele
    ser mas estable que el MSMF por defecto en camaras USB en Windows.
    Solo se aplica cuando source es un indice numerico (webcam).
    Aplica MJPG/resolucion de cfg para reducir ancho de banda USB.
    """
    if source.isdigit():
        backend = cv2.CAP_DSHOW if dshow else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
        if cfg.use_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if cfg.cam_width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
        if cfg.cam_height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir la fuente de video: {source}")
    return cap


def run(source: str, cfg: AppConfig, dshow: bool = False) -> None:
    cfg.review_dir.mkdir(parents=True, exist_ok=True)
    # Etiqueta de camara: permite lanzar varios procesos guardando en la misma
    # carpeta. CSV separado por etiqueta para evitar que dos procesos escriban
    # a la vez en el mismo archivo.
    label = cfg.cam_label
    csv_name = f"eventos_{label}.csv" if label else "eventos.csv"
    log_path = cfg.review_dir / csv_name
    new_log = not log_path.exists()
    log = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["timestamp", "camara", "frame", "track_id",
                         "score", "recorte"])

    estimator = PoseEstimator(cfg)
    detector = ConcealmentDetector(
        cfg.concealment,
        posture_cfg=cfg.posture,
        require_standing=cfg.require_standing,
        smoothing_cfg=cfg.smoothing,
        roi_cfg=cfg.roi,
    )
    cap = _open_source(source, cfg, dshow=dshow)

    writer_video = None
    if cfg.save_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cfg.save_video.parent.mkdir(parents=True, exist_ok=True)
        writer_video = cv2.VideoWriter(str(cfg.save_video), fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            people = estimator.track(frame)
            payload = [(p.track_id, p.keypoints, p.conf) for p in people]
            h, w = frame.shape[:2]
            events = detector.update(frame_idx, payload, frame_wh=(w, h))

            if cfg.roi.enabled:
                draw_roi(frame, cfg.roi.polygon)
            for p in people:
                draw_person(frame, p, detector.current_score(p.track_id),
                            cfg.concealment.min_keypoint_conf)

            for ev in events:
                draw_alert_banner(frame, ev.track_id)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                prefix = f"sospecha_{label}_" if label else "sospecha_"
                crop_name = f"{prefix}{ev.track_id}_{stamp}.jpg"
                cv2.imwrite(str(cfg.review_dir / crop_name), frame)
                writer.writerow([datetime.now().isoformat(), label or "",
                                 frame_idx, ev.track_id, f"{ev.score:.3f}",
                                 crop_name])
                log.flush()
                print(f"[{label or 'cam'}][señal] frame={frame_idx} "
                      f"id={ev.track_id} score={ev.score:.2f} -> {crop_name}")

            draw_footer(frame)
            if writer_video is not None:
                writer_video.write(frame)
            if cfg.show_window:
                win = f"ShrinkGuard - {label}" if label else "ShrinkGuard (Fase 1)"
                cv2.imshow(win, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    finally:
        cap.release()
        if writer_video is not None:
            writer_video.release()
        cv2.destroyAllWindows()
        log.close()

    dt = time.time() - t0
    fps = frame_idx / dt if dt > 0 else 0.0
    print(f"\nProcesados {frame_idx} frames en {dt:.1f}s ({fps:.1f} FPS). "
          f"Evidencias en: {cfg.review_dir}")
