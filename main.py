"""Entrada por linea de comandos de ShrinkGuard (Fase 1).

Ejemplos:
    python main.py --source data/tienda.mp4
    python main.py --source 0 --device mps          # webcam en Mac
    python main.py --source data/tienda.mp4 --save salidas/anotado.mp4 --no-window
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig
from src.pipeline import run


def main() -> None:
    ap = argparse.ArgumentParser(description="ShrinkGuard - deteccion de ocultacion por pose")
    ap.add_argument("--source", required=True,
                    help="ruta a un video o '0' para webcam")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="modelo YOLO-pose (n/s/m)")
    ap.add_argument("--device", default="cpu", help="cpu | 0 | mps")
    ap.add_argument("--save", default=None, help="ruta del video anotado de salida")
    ap.add_argument("--no-window", action="store_true",
                    help="no abrir ventana (util en servidores)")
    ap.add_argument("--consecutive", type=int, default=8,
                    help="frames consecutivos para disparar una señal")
    ap.add_argument("--no-posture-filter", action="store_true",
                    help="desactiva el filtro de postura (analiza tambien a personas sentadas)")
    ap.add_argument("--no-smoothing", action="store_true",
                    help="desactiva el suavizado temporal y el manejo de oclusiones de keypoints")
    ap.add_argument("--dshow", action="store_true",
                    help="usa backend DirectShow (Windows, mas estable en camaras USB)")
    ap.add_argument("--label", default=None,
                    help="etiqueta de camara (p. ej. cam0): permite lanzar varios "
                         "procesos guardando en la misma carpeta sin pisarse")
    ap.add_argument("--mjpg", action="store_true",
                    help="forzar MJPG (vídeo comprimido): menos ancho de banda USB")
    ap.add_argument("--width", type=int, default=None, help="ancho de captura, p. ej. 640")
    ap.add_argument("--height", type=int, default=None, help="alto de captura, p. ej. 480")
    args = ap.parse_args()

    cfg = AppConfig(
        model_name=args.model,
        device=args.device,
        save_video=Path(args.save) if args.save else None,
        show_window=not args.no_window,
        require_standing=not args.no_posture_filter,
        cam_width=args.width,
        cam_height=args.height,
        use_mjpg=args.mjpg,
        cam_label=args.label,
    )
    cfg.concealment.consecutive_frames = args.consecutive
    cfg.smoothing.enabled = not args.no_smoothing
    run(args.source, cfg, dshow=args.dshow)


if __name__ == "__main__":
    main()
