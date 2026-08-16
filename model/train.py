"""Trains CongestionUNet to predict DRC-violation-hotspot heatmaps from
pre-route layout rasters (cell density, pin density, RUDY).

Usage:
  python model/train.py --dataset-dir dataset --out-dir outputs --epochs 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from unet import build_model  # noqa: E402


def load_split(dataset_dir, name):
    d = np.load(Path(dataset_dir) / f"{name}.npz")
    x = torch.from_numpy(d["x"]).float()
    y = torch.from_numpy(d["y"]).float().unsqueeze(1)  # (N, 1, H, W), raw violation counts
    return x, y


def normalize(x, mean, std):
    return (x - mean) / std


def dice_loss(logits, target_binary, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target_binary).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target_binary.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def bin_metrics(probs, target_binary, threshold=0.5):
    pred = (probs > threshold).float()
    tp = (pred * target_binary).sum().item()
    fp = (pred * (1 - target_binary)).sum().item()
    fn = ((1 - pred) * target_binary).sum().item()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--arch", default="attention_unet", choices=["unet", "attention_unet"])
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None,
                     help="path to a checkpoint (from a prior train.py run) to initialize weights "
                          "from before training -- for transfer learning. Requires the same --arch "
                          "and --base-ch, and in_channels matching the checkpoint's.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_x, train_y = load_split(args.dataset_dir, "train")
    val_x, val_y = load_split(args.dataset_dir, "val")

    mean = train_x.mean(dim=(0, 2, 3), keepdim=True)
    std = train_x.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    train_x = normalize(train_x, mean, std)
    val_x = normalize(val_x, mean, std)

    train_y_bin = (train_y > 0).float()
    val_y_bin = (val_y > 0).float()

    n_pos = train_y_bin.sum().item()
    n_neg = train_y_bin.numel() - n_pos
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1.0), 200.0)], device=device)
    print(f"train bins: {int(n_pos)} positive / {int(n_neg)} negative -> pos_weight={pos_weight.item():.1f}")

    train_loader = DataLoader(TensorDataset(train_x, train_y_bin), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y_bin), batch_size=args.batch_size, shuffle=False)

    model = build_model(args.arch, in_channels=train_x.shape[1], base_ch=args.base_ch).to(device)
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"init-from {args.init_from}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb) + args.dice_weight * dice_loss(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = bce(logits, yb) + args.dice_weight * dice_loss(logits, yb)
                val_loss += loss.item() * xb.size(0)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_targets.append(yb.cpu())
        val_loss /= len(val_loader.dataset)
        probs = torch.cat(all_probs)
        targets = torch.cat(all_targets)
        precision, recall, f1 = bin_metrics(probs, targets, threshold=0.5)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_precision": precision, "val_recall": recall, "val_f1": f1})
        print(f"epoch {epoch:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_P={precision:.3f}  val_R={recall:.3f}  val_F1={f1:.3f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "in_channels": train_x.shape[1],
                "base_ch": args.base_ch,
                "feat_mean": mean, "feat_std": std,
                "epoch": epoch, "val_f1": f1,
            }, out_dir / "best_model.pt")

    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val F1 = {best_val_f1:.3f}; checkpoint -> {out_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
