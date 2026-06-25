"""Envoltorio fino sobre Ultralytics YOLO-pose con tracking integrado.

Aisla la dependencia de Ultralytics en un solo sitio: el resto del codigo solo
ve `TrackedPerson` (track_id + keypoints + confianzas), nunca la API de YOLO.
Asi, si mañana cambias de modelo (RTMPose, MoveNet...), solo tocas este archivo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import AppConfig


@dataclass
class TrackedPerson:
    track_id: int
    keypoints: np.ndarray  # (17, 2) -> (x, y) en pixeles
    conf: np.ndarray       # (17,)  -> confianza por keypoint
    box: tuple[float, float, float, float]  # (x1, y1, x2, y2)


class PoseEstimator:
    def __init__(self, cfg: AppConfig) -> None:
        # Import perezoso: Ultralytics solo se carga si de verdad se usa el
        # pipeline (los tests de la heuristica no lo necesitan).
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(cfg.model_name)

    def track(self, frame) -> list[TrackedPerson]:
        """Procesa un frame BGR (numpy) y devuelve las personas seguidas."""
        results = self.model.track(
            frame,
            persist=True,            # mantiene los IDs entre frames
            tracker=self.cfg.tracker,
            conf=self.cfg.person_conf,
            device=self.cfg.device,
            classes=[0],             # 0 = "person" en COCO
            verbose=False,
        )
        people: list[TrackedPerson] = []
        if not results:
            return people

        r = results[0]
        if r.keypoints is None or r.boxes is None or r.boxes.id is None:
            return people

        ids = r.boxes.id.int().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy()
        kp_xy = r.keypoints.xy.cpu().numpy()        # (N, 17, 2)
        kp_conf = r.keypoints.conf
        kp_conf = (kp_conf.cpu().numpy() if kp_conf is not None
                   else np.ones(kp_xy.shape[:2]))

        for i, tid in enumerate(ids):
            people.append(TrackedPerson(
                track_id=int(tid),
                keypoints=kp_xy[i],
                conf=kp_conf[i],
                box=tuple(xyxy[i].tolist()),
            ))
        return people
