"""Entrena el clasificador temporal de la Fase 3 (PoseLSTM).

Carga uno o varios .npz (de preparar_dataset.py), divide train/val POR VIDEO
(usando 'groups', para que ventanas del mismo clip no caigan en ambos lados y
las metricas no se inflen) y entrena un LSTM. Reporta precision/recall/F1 a
nivel de ventana en validacion y guarda el mejor modelo por F1.

Uso:
    python tools/entrenar.py --data datos/ds_640.npz datos/ds_cctv.npz \
        datos/ds_dcsass.npz --out modelos/poselstm.pt --epochs 30 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classifier import PoseLSTM, save_model  # noqa: E402


def cargar(datasets: list[str]):
    Xs, ys, gs = [], [], []
    for d in datasets:
        z = np.load(d, allow_pickle=True)
        Xs.append(z["X"])
        ys.append(z["y"])
        # prefijo con el dataset para que los nombres de video sean unicos
        gs.append(np.array([f"{Path(d).stem}/{g}" for g in z["groups"]]))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def split_por_grupo(groups: np.ndarray, val_frac: float, seed: int = 0):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac))
    val_groups = set(uniq[:n_val].tolist())
    val_mask = np.array([g in val_groups for g in groups])
    return ~val_mask, val_mask


def metricas(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Entrena PoseLSTM (Fase 3)")
    ap.add_argument("--data", nargs="+", required=True, help="ficheros .npz")
    ap.add_argument("--out", default="modelos/poselstm.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # torch espera "cuda:0", no "0" (a diferencia de Ultralytics).
    dev = f"cuda:{args.device}" if args.device.isdigit() else args.device
    X, y, g = cargar(args.data)
    tr, va = split_por_grupo(g, args.val_frac, args.seed)
    print(f"Total ventanas={len(y)} (pos={int(y.sum())}). "
          f"Train={int(tr.sum())} Val={int(va.sum())} "
          f"({len(np.unique(g[va]))} videos en val)")

    Xtr = torch.tensor(X[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.long)
    Xva = torch.tensor(X[va], dtype=torch.float32).to(dev)
    yva = y[va]

    # peso de clase inverso a la frecuencia (desbalance)
    counts = np.bincount(y[tr], minlength=2).astype(float)
    w = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))),
                     dtype=torch.float32).to(dev)
    print(f"Pesos de clase: normal={w[0]:.2f} oculta={w[1]:.2f}")

    window = X.shape[1]
    model = PoseLSTM(n_features=X.shape[2], hidden=args.hidden,
                     layers=args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    n = len(Xtr)
    best_f1 = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            xb = Xtr[idx].to(dev)
            yb = ytr[idx].to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1).cpu().numpy()
        m = metricas(pred, yva)
        flag = ""
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            save_model(model, args.out, window)
            flag = "  <- guardado (mejor F1)"
        print(f"ep {ep:>3} | loss {total / n:.4f} | val P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f}{flag}")

    print(f"\nMejor F1 (val, nivel ventana) = {best_f1:.3f}. Modelo en {args.out}")


if __name__ == "__main__":
    main()
