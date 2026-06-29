"""Clasificador temporal aprendido (Fase 3).

Sustituye la heuristica geometrica por un modelo que APRENDE el gesto a partir
de secuencias de pose normalizada. Este archivo es el unico (junto con pose.py)
que depende de una libreria pesada: aqui, PyTorch. Asi, igual que pose.py aisla
Ultralytics, si se cambia de framework solo se toca este modulo.

Contiene:
  - PoseLSTM: el modelo (LSTM sobre ventanas de features de pose).
  - save_model / load_model: persistencia (pesos + hiperparametros).
  - LearnedConcealmentDetector: inferencia en streaming con la MISMA interfaz
    que ConcealmentDetector (update(frame_idx, people, frame_wh) -> eventos),
    de modo que es intercambiable en el pipeline sin tocar nada mas. Reutiliza
    TrackState (contador temporal + cooldown) y KeypointSmoother de la Fase 1/2.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn

from src.concealment import ConcealmentConfig, ConcealmentEvent, TrackState
from src.sequence import FEATURES_PER_FRAME, pose_features
from src.smoothing import KeypointSmoother, SmoothingConfig


class PoseLSTM(nn.Module):
    """LSTM sobre ventanas (B, T, FEATURES_PER_FRAME) -> 2 clases (normal/oculta)."""

    def __init__(self, n_features: int = FEATURES_PER_FRAME, hidden: int = 64,
                 layers: int = 1, n_classes: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.hparams = dict(n_features=n_features, hidden=hidden, layers=layers,
                            n_classes=n_classes, dropout=dropout)
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)            # (B, T, H)
        return self.head(out[:, -1])     # ultimo paso temporal -> (B, n_classes)


def save_model(model: PoseLSTM, path: str, window: int) -> None:
    torch.save({"hparams": model.hparams, "window": window,
                "state_dict": model.state_dict()}, path)


def load_model(path: str, device: str = "cpu") -> tuple[PoseLSTM, int]:
    ckpt = torch.load(path, map_location=device)
    model = PoseLSTM(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, int(ckpt["window"])


class LearnedConcealmentDetector:
    """Detector aprendido con la misma interfaz que ConcealmentDetector.

    Mantiene por persona un buffer deslizante de features de pose; cuando hay
    `window` frames, el LSTM puntua la ventana y la probabilidad de "ocultacion"
    pasa por el mismo contador temporal (consecutive/gap/cooldown) que la
    heuristica, para no spamear alertas.
    """

    def __init__(self, model_path: str, cfg: ConcealmentConfig | None = None,
                 threshold: float = 0.5, device: str = "cpu",
                 smoothing_cfg: SmoothingConfig | None = None) -> None:
        self.cfg = cfg or ConcealmentConfig()
        self.threshold = threshold
        self.device = device
        self.model, self.window = load_model(model_path, device)
        self._smoothing_cfg = smoothing_cfg
        self._smoother = KeypointSmoother(smoothing_cfg)
        self._buffers: dict[int, deque] = {}
        self._states: dict[int, TrackState] = {}

    def reset(self) -> None:
        """Limpia el estado por persona (para reutilizar el detector, ya cargado,
        entre videos distintos sin recargar el modelo)."""
        self._buffers.clear()
        self._states.clear()
        self._smoother = KeypointSmoother(self._smoothing_cfg)

    @torch.no_grad()
    def _prob(self, window_feats: np.ndarray) -> float:
        x = torch.from_numpy(window_feats[None]).float().to(self.device)
        logits = self.model(x)
        return float(torch.softmax(logits, dim=1)[0, 1])

    def update(self, frame_idx: int,
               people: list[tuple[int, np.ndarray, np.ndarray]],
               frame_wh: tuple[int, int] | None = None) -> list[ConcealmentEvent]:
        seen = set()
        events: list[ConcealmentEvent] = []
        for track_id, kp, conf in people:
            seen.add(track_id)
            kp, conf = self._smoother.apply(track_id, kp, conf)
            buf = self._buffers.setdefault(track_id, deque(maxlen=self.window))
            buf.append(pose_features(kp, conf, self.cfg.min_keypoint_conf))
            state = self._states.setdefault(track_id, TrackState())

            score = 0.0
            if len(buf) == self.window:
                score = self._prob(np.stack(buf))
            near = score >= self.threshold
            if state.update(near, score, self.cfg):
                events.append(ConcealmentEvent(track_id, frame_idx, score))

        for gone in set(self._states) - seen:
            del self._states[gone]
            self._buffers.pop(gone, None)
            self._smoother.drop(gone)
        return events

    def current_score(self, track_id: int) -> float:
        st = self._states.get(track_id)
        return st.history[-1] if st and st.history else 0.0
