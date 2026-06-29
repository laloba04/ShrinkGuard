"""Procesamiento multicamara: un hilo por camara.

Cada camara tiene su PROPIA instancia de modelo y su PROPIO detector, de modo
que el tracking y el estado por persona son independientes (los IDs de la cam0
no se mezclan con los de la cam1).

Patron de hilos:
- Los workers (uno por camara) hacen captura + inferencia + deteccion + dibujo
  y guardan evidencias. Dejan el ultimo frame anotado en `latest` (con lock).
- El hilo PRINCIPAL es el unico que pinta ventanas (cv2.imshow / waitKey),
  porque el HighGUI de OpenCV no es seguro desde varios hilos (sobre todo en
  Windows).

Con una GPU potente (p. ej. RTX 4080) varias camaras van de sobra en tiempo real.
"""

from __future__ import annotations

import csv
import platform
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.config import AppConfig
from src.pipeline import build_detector
from src.pose import PoseEstimator
from src.visualizer import draw_alert_banner, draw_footer, draw_person, draw_roi


def open_capture(source: str, use_dshow: bool, width: int | None = None,
                 height: int | None = None, mjpg: bool = False) -> cv2.VideoCapture:
    """Abre una fuente (webcam '0','1'... o ruta/RTSP). En Windows, DSHOW
    suele abrir las webcams mas rapido y fiable que el backend por defecto.

    mjpg=True fuerza vídeo comprimido (MJPG): reduce muchisimo el ancho de banda
    USB y suele ser lo que permite abrir DOS webcams a la vez en el mismo bus.
    """
    is_cam = source.isdigit()
    target = int(source) if is_cam else source
    if use_dshow and is_cam and platform.system() == "Windows":
        cap = cv2.VideoCapture(target, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(target)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _safe_name(source: str, idx: int) -> str:
    base = re.sub(r"[^0-9A-Za-z]+", "_", source).strip("_") or "src"
    return f"cam{idx}_{base}"


class CameraWorker(threading.Thread):
    def __init__(self, idx: int, source: str, cfg: AppConfig,
                 stop_event: threading.Event, use_dshow: bool,
                 open_lock: threading.Lock,
                 name: str | None = None) -> None:
        super().__init__(daemon=True)
        self.idx = idx
        self.source = source
        self.cfg = cfg
        self.stop_event = stop_event
        self.use_dshow = use_dshow
        self.open_lock = open_lock
        self.name_tag = name if name else _safe_name(source, idx)
        self.window_title = f"ShrinkGuard - {self.name_tag}"
        self.review_dir = cfg.review_dir / self.name_tag

        self._latest = None
        self._lock = threading.Lock()
        self.frames = 0
        self.fps = 0.0
        self.error: str | None = None

    def get_latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # no queremos que un fallo tumbe el proceso
            self.error = str(exc)
            print(f"[{self.name_tag}] ERROR: {exc}")

    def _run(self) -> None:
        self.review_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.review_dir / "eventos.csv"
        new_log = not log_path.exists()
        log = open(log_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(log)
        if new_log:
            writer.writerow(["timestamp", "camara", "frame", "track_id",
                             "score", "recorte"])

        estimator = PoseEstimator(self.cfg)  # modelo propio -> tracking propio
        detector = build_detector(self.cfg)  # heuristica o clasificador aprendido
        # Apertura SERIALIZADA: DirectShow (y algunos backends) fallan si dos
        # camaras se inicializan a la vez. Solo un hilo abre cada vez; "calentar"
        # la camara dentro del lock fuerza a que el grafo termine de montarse
        # antes de soltar el turno. Los bucles de LECTURA si corren en paralelo.
        with self.open_lock:
            cap = None
            for intento in range(3):
                cap = open_capture(self.source, self.use_dshow,
                                   width=self.cfg.cam_width,
                                   height=self.cfg.cam_height,
                                   mjpg=self.cfg.use_mjpg)
                if cap.isOpened():
                    break
                cap.release()
                time.sleep(0.4)
            if cap is None or not cap.isOpened():
                self.error = f"no pude abrir la fuente {self.source}"
                print(f"[{self.name_tag}] {self.error}")
                log.close()
                return
            # calentamiento: unas lecturas para que DSHOW acabe de inicializar
            for _ in range(5):
                if cap.read()[0]:
                    break
                time.sleep(0.05)

        t0 = time.time()
        idx = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break  # fin del video o camara desconectada

                people = estimator.track(frame)
                payload = [(p.track_id, p.keypoints, p.conf) for p in people]
                h, w = frame.shape[:2]
                events = detector.update(idx, payload, frame_wh=(w, h))

                if self.cfg.roi.enabled:
                    draw_roi(frame, self.cfg.roi.polygon)
                for p in people:
                    draw_person(frame, p,
                                detector.current_score(p.track_id),
                                self.cfg.concealment.min_keypoint_conf)

                for ev in events:
                    draw_alert_banner(frame, ev.track_id)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    crop = f"sospecha_{ev.track_id}_{stamp}.jpg"
                    cv2.imwrite(str(self.review_dir / crop), frame)
                    writer.writerow([datetime.now().isoformat(), self.name_tag,
                                     idx, ev.track_id, f"{ev.score:.3f}", crop])
                    log.flush()
                    print(f"[{self.name_tag}][señal] frame={idx} "
                          f"id={ev.track_id} score={ev.score:.2f} -> {crop}")

                draw_footer(frame)
                with self._lock:
                    self._latest = frame
                idx += 1
                self.frames = idx
        finally:
            cap.release()
            log.close()
            dt = time.time() - t0
            self.fps = idx / dt if dt > 0 else 0.0
            print(f"[{self.name_tag}] fin: {idx} frames, {self.fps:.1f} FPS")


def run_multicam(sources: list[str], cfg: AppConfig, show: bool,
                 use_dshow: bool, names: list[str] | None = None) -> None:
    stop = threading.Event()
    open_lock = threading.Lock()  # serializa la apertura de camaras
    workers = [CameraWorker(i, s, cfg, stop, use_dshow, open_lock,
                            name=names[i] if names else None)
               for i, s in enumerate(sources)]
    for w in workers:
        w.start()

    print(f"Lanzadas {len(workers)} camaras. "
          f"{'Pulsa q en una ventana para parar.' if show else 'Ctrl+C para parar.'}")
    try:
        if show:
            placed: set[str] = set()
            while not stop.is_set():
                any_alive = False
                for k, w in enumerate(workers):
                    f = w.get_latest()
                    if f is not None:
                        cv2.imshow(w.window_title, f)
                        if w.window_title not in placed:
                            # separar las ventanas para que no se solapen
                            cv2.moveWindow(w.window_title, 30 + k * 680, 60)
                            placed.add(w.window_title)
                    if w.is_alive():
                        any_alive = True
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stop.set()
                if not any_alive:
                    break
            cv2.destroyAllWindows()
        else:
            while any(w.is_alive() for w in workers):
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nParando...")
        stop.set()
    finally:
        stop.set()
        for w in workers:
            w.join(timeout=3)

    print("\nResumen:")
    for w in workers:
        estado = w.error if w.error else f"{w.frames} frames, {w.fps:.1f} FPS"
        print(f"  {w.name_tag}: {estado}")
    print(f"Evidencias por camara en: {cfg.review_dir}/<camara>/")