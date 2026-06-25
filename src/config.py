"""Configuracion central del pipeline ShrinkGuard (Fase 1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.concealment import ConcealmentConfig
from src.posture import PostureConfig


@dataclass
class AppConfig:
    # Modelo YOLO-pose. "yolo11n-pose.pt" es el nano (rapido, CPU-friendly).
    # Para mas precision: yolo11s-pose.pt / yolo11m-pose.pt.
    model_name: str = "yolo11n-pose.pt"
    # "cpu", "0" (primera GPU), "mps" (Apple Silicon).
    device: str = "cpu"
    # Confianza minima de deteccion de persona.
    person_conf: float = 0.35
    # Tracker de Ultralytics. "bytetrack.yaml" o "botsort.yaml".
    tracker: str = "bytetrack.yaml"
    # Carpeta donde se guardan los recortes para revision humana.
    review_dir: Path = field(default_factory=lambda: Path("salidas/revision"))
    # Mostrar ventana en vivo (False util en servidores sin display).
    show_window: bool = True
    # Guardar video anotado de salida (ruta o None).
    save_video: Path | None = None
    # Parametros del detector de ocultacion.
    concealment: ConcealmentConfig = field(default_factory=ConcealmentConfig)
    # Parametros del filtro de postura (piernas visibles y erguidas).
    posture: PostureConfig = field(default_factory=PostureConfig)
    # Si True, solo analiza personas que parecen estar de pie.
    require_standing: bool = True
    # Resolucion de captura (None = dejar el defecto de la camara).
    cam_width: int | None = None
    cam_height: int | None = None
    # Si True, fuerza codec MJPG en webcams: reduce ancho de banda USB y
    # permite abrir dos camaras en el mismo bus sin que se bloqueen.
    use_mjpg: bool = False
    # Etiqueta de camara: se añade al nombre del recorte y al CSV para poder
    # lanzar VARIOS procesos (uno por camara) que guarden en la MISMA carpeta
    # diferenciando solo por esta etiqueta. None = sin etiqueta (un proceso).
    cam_label: str | None = None
