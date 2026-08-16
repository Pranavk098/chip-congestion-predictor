"""Evaluates a trained CongestionUNet checkpoint on the held-out test design(s)
and produces precision/recall/F1 curves plus predicted-vs-actual heatmap
overlays for the README.

Usage:
  python model/evaluate.py --dataset-dir dataset --checkpoint outputs/best_model.pt --out-dir outputs
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from unet import build_model  # noqa: E402


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "unet")  # older checkpoints predate --arch and are plain U-Nets
    model = build_model(arch, in_channels=ckpt["in_channels"], base_ch=ckpt["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["feat_mean"].to(device), ckpt["feat_std"].to(device)


def precision_recall_at_thresholds(probs, targets, thresholds):
    rows = []
    for t in thresholds:
        pred = (probs > t).float()
        tp = (pred * targets).sum().item()
        fp = (pred * (1 - targets)).sum().item()
        fn = ((1 - pred) * targets).sum().item()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows.append({"threshold": t, "precision": precision, "recall": recall, "f1": f1})
    return rows


def recall_at_top_k_percent(probs, targets, k_percent):
    """Of the top-k% highest-scored bins by predicted probability, what
    fraction of actual violation bins are captured? This is the headline
    'recall on violation regions' metric — matches how a designer would
    actually use the tool (triage the top-K riskiest regions pre-route)."""
    flat_p = probs.flatten().numpy()
    flat_t = targets.flatten().numpy()
    n_top = max(1, int(len(flat_p) * k_percent / 100))
    top_idx = np.argpartition(-flat_p, n_top - 1)[:n_top]
    n_pos_total = flat_t.sum()
    if n_pos_total == 0:
        return None
    n_pos_captured = flat_t[top_idx].sum()
    return float(n_pos_captured / n_pos_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="dataset")
    ap.add_argument("--checkpoint", default="outputs/best_model.pt")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = load_checkpoint(args.checkpoint, device)

    d = np.load(Path(args.dataset_dir) / f"{args.split}.npz")
    x = torch.from_numpy(d["x"]).float()
    y = torch.from_numpy(d["y"]).float()
    targets = (y > 0).float()

    x_norm = (x.to(device) - mean) / std
    with torch.no_grad():
        logits = model(x_norm)
        probs = torch.sigmoid(logits).squeeze(1).cpu()

    thresholds = [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    pr_curve = precision_recall_at_thresholds(probs, targets, thresholds)
    best = max(pr_curve, key=lambda r: r["f1"])

    results = {
        "split": args.split,
        "n_patches": int(x.shape[0]),
        "n_violation_bins": int(targets.sum().item()),
        "n_total_bins": int(targets.numel()),
        "best_threshold": best["threshold"],
        "best_precision": best["precision"],
        "best_recall": best["recall"],
        "best_f1": best["f1"],
        "recall_at_top_1pct": recall_at_top_k_percent(probs, targets, 1),
        "recall_at_top_5pct": recall_at_top_k_percent(probs, targets, 5),
        "recall_at_top_10pct": recall_at_top_k_percent(probs, targets, 10),
        "pr_curve": pr_curve,
    }
    with open(out_dir / f"eval_{args.split}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "pr_curve"}, indent=2))

    # PR curve plot
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r["recall"] for r in pr_curve], [r["precision"] for r in pr_curve], marker="o", ms=3)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"DRC hotspot precision/recall ({args.split})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pr_curve_{args.split}.png", dpi=150)
    plt.close(fig)

    # Predicted vs actual overlay for a handful of patches with real violations
    pos_idx = [i for i in range(len(targets)) if targets[i].sum() > 0]
    sample_idx = (pos_idx[:4] if pos_idx else list(range(min(4, len(targets)))))
    if sample_idx:
        fig, axes = plt.subplots(len(sample_idx), 3, figsize=(9, 3 * len(sample_idx)))
        axes = np.atleast_2d(axes)
        for row, i in enumerate(sample_idx):
            axes[row, 0].imshow(x[i, 0], cmap="viridis")
            axes[row, 0].set_title("cell density" if row == 0 else "")
            axes[row, 1].imshow(probs[i], cmap="hot", vmin=0, vmax=1)
            axes[row, 1].set_title("predicted hotspot prob." if row == 0 else "")
            axes[row, 2].imshow(targets[i], cmap="hot", vmin=0, vmax=1)
            axes[row, 2].set_title("actual DRC violations" if row == 0 else "")
            for c in range(3):
                axes[row, c].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"predictions_{args.split}.png", dpi=150)
        plt.close(fig)

    print(f"wrote eval_{args.split}.json, pr_curve_{args.split}.png, predictions_{args.split}.png -> {out_dir}")


if __name__ == "__main__":
    main()
