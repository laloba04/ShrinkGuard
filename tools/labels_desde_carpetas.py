"""Genera labels.csv a partir de una estructura de carpetas por clase.

Para datasets que ya separan los clips en carpetas (p. ej. Shoplifting/ y
Normal/), cada clip bajo la carpeta POSITIVA se marca como un intervalo positivo
que cubre todo el clip (etiquetado a nivel de clip). Los clips bajo cualquier
otra carpeta quedan sin filas -> negativos.

Etiquetado a nivel de clip: medimos si el detector dispara en algun momento de
un clip de hurto (recall) y si dispara en clips normales (falsas alarmas ->
precision). No marca el instante exacto del gesto; es una linea base honesta
para datos debilmente etiquetados.

Uso:
    python tools/labels_desde_carpetas.py \
        --root "datos/clips/Shoplifting dataset" --positivo Shoplifting \
        --salida datos/labels.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
FIN_CLIP = 1_000_000.0  # fin "infinito": cualquier disparo en el clip cuenta


def main() -> None:
    ap = argparse.ArgumentParser(description="Genera labels.csv desde carpetas por clase")
    ap.add_argument("--root", required=True, help="carpeta raiz del dataset")
    ap.add_argument("--positivo", required=True,
                    help="nombre de la carpeta de la clase positiva (p. ej. Shoplifting)")
    ap.add_argument("--salida", default="datos/labels.csv")
    args = ap.parse_args()

    root = Path(args.root)
    pos = args.positivo.lower()
    filas = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT:
            continue
        # ¿algun componente de la ruta (relativa a root) es la carpeta positiva?
        partes = [s.lower() for s in p.relative_to(root).parts[:-1]]
        if pos in partes:
            filas.append((p.name, 0, FIN_CLIP))

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video", "inicio", "fin"])
        w.writerows(filas)
    print(f"{len(filas)} clips positivos escritos en {salida}")


if __name__ == "__main__":
    main()
