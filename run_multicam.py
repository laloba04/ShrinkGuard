"""Entrada multicamara de ShrinkGuard.

Ejemplos:
    # dos webcams
    python run_multicam.py --sources 0 1

    # webcam + camara IP, sin ventanas (solo evidencias)
    python run_multicam.py --sources 0 rtsp://user:pass@192.168.1.50/stream --no-window

    # dos videos grabados con la 4080 y modelo medio
    python run_multicam.py --sources data/cam_a.mp4 data/cam_b.mp4 --device 0 --model yolo11m-pose.pt
"""

from __future__ import annotations

import argparse

from src.config import AppConfig
from src.multicam import run_multicam


def main() -> None:
    ap = argparse.ArgumentParser(description="ShrinkGuard multicamara (un hilo por camara)")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="lista de fuentes: '0 1' (webcams) o rutas/RTSP")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="modelo YOLO-pose")
    ap.add_argument("--device", default="cpu", help="cpu | 0 | mps")
    ap.add_argument("--no-window", action="store_true",
                    help="no abrir ventanas (solo guardar evidencias)")
    ap.add_argument("--no-dshow", action="store_true",
                    help="no usar el backend DSHOW en Windows")
    ap.add_argument("--mjpg", action="store_true",
                    help="forzar MJPG (vídeo comprimido): permite 2 webcams USB a la vez")
    ap.add_argument("--width", type=int, default=None, help="ancho de captura, p. ej. 640")
    ap.add_argument("--height", type=int, default=None, help="alto de captura, p. ej. 480")
    ap.add_argument("--consecutive", type=int, default=8,
                    help="frames consecutivos para disparar una señal")
    ap.add_argument("--no-posture-filter", action="store_true",
                    help="desactiva el filtro de 'solo personas de pie'")
    ap.add_argument("--names", nargs="+", default=None,
                    help="nombre para cada camara, en el mismo orden que --sources "
                         "(p. ej. Entrada Caja Pasillo)")
    args = ap.parse_args()

    if args.names and len(args.names) != len(args.sources):
        ap.error(f"--names necesita el mismo numero de valores que --sources "
                 f"({len(args.sources)} fuentes, {len(args.names)} nombres)")

    cfg = AppConfig(
        model_name=args.model,
        device=args.device,
        show_window=not args.no_window,
        require_standing=not args.no_posture_filter,
        cam_width=args.width,
        cam_height=args.height,
        use_mjpg=args.mjpg,
    )
    cfg.concealment.consecutive_frames = args.consecutive

    run_multicam(args.sources, cfg, show=not args.no_window,
                 use_dshow=not args.no_dshow, names=args.names)


if __name__ == "__main__":
    main()