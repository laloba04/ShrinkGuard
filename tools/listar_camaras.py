"""Lista las camaras disponibles con su NOMBRE e indice de OpenCV.

Multiplataforma: obtiene los nombres segun el sistema operativo.
  - Windows: pygrabber (DirectShow)        ->  pip install pygrabber
  - Linux:   v4l2-ctl --list-devices       ->  sudo apt install v4l-utils
  - macOS:   system_profiler SPCameraDataType (viene de serie)

Sirve para distinguir tus camaras fisicas de las virtuales (NVIDIA Broadcast,
OBS Virtual Camera, etc.) y saber que numero pasar a run_multicam.py.

Uso:
    python tools/listar_camaras.py           # lista nombres + indices que abren
    python tools/listar_camaras.py --ver      # ademas muestra cada camara en pantalla
"""

from __future__ import annotations

import argparse
import platform
import re
import subprocess

import cv2

_SO = platform.system()


def _nombres_windows():
    try:
        from pygrabber.dshow_graph import FilterGraph
        return {i: n for i, n in enumerate(FilterGraph().get_input_devices())}
    except Exception:
        return None


def _nombres_linux():
    """Mapea /dev/videoN -> nombre a partir de `v4l2-ctl --list-devices`."""
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    nombres: dict[int, str] = {}
    actual = None
    for linea in out.splitlines():
        if not linea.strip():
            continue
        if not linea.startswith((" ", "\t")):
            actual = linea.strip().rstrip(":")
        else:
            m = re.search(r"/dev/video(\d+)", linea)
            if m and actual:
                nombres.setdefault(int(m.group(1)), actual)
    return nombres or None


def _nombres_macos():
    """Nombres en orden via system_profiler (el indice = posicion, aprox.)."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType"],
                             capture_output=True, text=True, timeout=8).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    nombres = {}
    idx = 0
    for linea in out.splitlines():
        s = linea.strip()
        # Nombres de camara: lineas indentadas que acaban en ':' y no son campos.
        if (s.endswith(":") and s != "Camera:"
                and not any(c in s for c in ("Model", "Unique", "ID"))):
            nombres[idx] = s.rstrip(":")
            idx += 1
    return nombres or None


def nombres_dispositivos():
    if _SO == "Windows":
        return _nombres_windows()
    if _SO == "Linux":
        return _nombres_linux()
    if _SO == "Darwin":
        return _nombres_macos()
    return None


def _pista_instalacion() -> str:
    if _SO == "Windows":
        return "instala pygrabber:  pip install pygrabber"
    if _SO == "Linux":
        return "instala v4l-utils:  sudo apt install v4l-utils"
    if _SO == "Darwin":
        return "system_profiler deberia venir de serie en macOS"
    return "sistema no reconocido"


def _abrir(i: int):
    if _SO == "Windows":
        return cv2.VideoCapture(i, cv2.CAP_DSHOW)
    return cv2.VideoCapture(i)


def indices_que_abren(maxn: int):
    disponibles = []
    for i in range(maxn):
        cap = _abrir(i)
        ok = cap.isOpened()
        if ok:
            ok, _ = cap.read()
        if ok:
            disponibles.append(i)
        cap.release()
    return disponibles


def _parece_virtual(nombre: str) -> bool:
    return any(k in nombre.lower()
               for k in ("virtual", "broadcast", "obs", "nvidia", "droidcam"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", action="store_true",
                    help="muestra cada camara en una ventana")
    ap.add_argument("--max", type=int, default=10,
                    help="cuantos indices probar (0..max-1)")
    args = ap.parse_args()

    print(f"Sistema detectado: {_SO}\n")
    nombres = nombres_dispositivos()

    print("=== Camaras por nombre ===")
    if nombres:
        for i in sorted(nombres):
            pista = "  <-- probablemente VIRTUAL" if _parece_virtual(nombres[i]) else ""
            print(f"  index {i}: {nombres[i]}{pista}")
    else:
        print(f"  (no pude leer nombres) {_pista_instalacion()}")

    print("\n=== Indices que abren en OpenCV ===")
    disp = indices_que_abren(args.max)
    print(f"  {disp}")

    if args.ver:
        print("\nMostrando cada camara (pulsa una tecla para la siguiente)...")
        for i in disp:
            cap = _abrir(i)
            ok, frame = cap.read()
            if ok:
                etiqueta = f"index {i}"
                if nombres and i in nombres:
                    etiqueta += f" - {nombres[i]}"
                cv2.putText(frame, etiqueta, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("Identifica tu camara - pulsa una tecla", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            cap.release()

    print("\nElige los indices de tus DOS camaras fisicas y lanza:")
    print("  python run_multicam.py --sources IDX1 IDX2 --device 0")


if __name__ == "__main__":
    main()