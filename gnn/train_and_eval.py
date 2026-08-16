"""
Leave-one-design-out cross-validation for the GNN baseline, mirroring
scripts/cross_validate.py's protocol exactly so the numbers are directly
comparable to the CNN: same held-out designs, same recall@top-K% metric,
computed on the same 2um grid (reusing dataset/raw/<design>_labels.npz for
grid alignment and ground truth, so "top-K%" means the same thing in both).

Usage:
  python gnn/train_and_eval.py --designs gcd aes ibex jpeg riscv32i
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import BipartiteGNN

PROJECT_DIR = Path(__file__).parent.parent


def load_design(name, raw_dir):
    d = np.load(Path(raw_dir) / f"{name}_graph.npz")
    return {k: d[k] for k in d.keys()}


def combine(designs_data):
    """Block-diagonal combine of several designs' graphs into one batch."""
    node_feat, label, edge_cell, edge_net = [], [], [], []
    cell_offset, net_offset = 0, 0
    for d in designs_data:
        node_feat.append(d["node_feat"])
        label.append(d["label"])
        edge_cell.append(d["edge_cell"] + cell_offset)
        edge_net.append(d["edge_net"] + net_offset)
        cell_offset += int(d["n_cells"])
        net_offset += int(d["n_nets"])
    return (np.concatenate(node_feat), np.concatenate(label),
            np.concatenate(edge_cell) if edge_cell else np.array([], dtype=np.int64),
            np.concatenate(edge_net) if edge_net else np.array([], dtype=np.int64),
            cell_offset, net_offset)


def recall_at_topk_grid(cell_scores, bbox, grid_meta, grid_label, pct):
    """Project per-cell scores onto the design's raster grid (max score per
    bin over cells whose center falls in it), then compute recall@top-pct%
    against the SAME ground-truth raster the CNN was scored against."""
    llx, lly, nx, ny, bs = (grid_meta["llx_um"], grid_meta["lly_um"],
                             int(grid_meta["nx"]), int(grid_meta["ny"]), grid_meta["bin_size_um"])
    cx = (bbox[:, 0] + bbox[:, 2]) / 2
    cy = (bbox[:, 1] + bbox[:, 3]) / 2
    ix = np.clip(((cx - llx) / bs).astype(int), 0, nx - 1)
    iy = np.clip(((cy - lly) / bs).astype(int), 0, ny - 1)

    grid_score = np.full((ny, nx), -1e9, dtype=np.float32)
    np.maximum.at(grid_score, (iy, ix), cell_scores)

    flat_score = grid_score.flatten()
    flat_label = (grid_label > 0).astype(np.float32).flatten()
    n_pos = int(flat_label.sum())
    if n_pos == 0:
        return None
    k = max(1, int(len(flat_score) * pct / 100))
    top_idx = np.argsort(-flat_score)[:k]
    return float(flat_label[top_idx].sum() / n_pos)


def train_one_fold(train_designs_data, in_dim, device, epochs=150, hidden=64, rounds=3, lr=2e-3):
    node_feat, label, edge_cell, edge_net, n_cells, n_nets = combine(train_designs_data)
    feat_mean, feat_std = node_feat.mean(0, keepdims=True), node_feat.std(0, keepdims=True).clip(1e-6)
    node_feat = (node_feat - feat_mean) / feat_std

    x = torch.tensor(node_feat, dtype=torch.float32, device=device)
    y = torch.tensor(label, dtype=torch.float32, device=device)
    ec = torch.tensor(edge_cell, dtype=torch.long, device=device)
    en = torch.tensor(edge_net, dtype=torch.long, device=device)

    n_pos, n_neg = y.sum().item(), y.numel() - y.sum().item()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    print(f"  train: {n_cells} cells, {n_pos:.0f} positive (pos_weight={pos_weight.item():.1f})")

    model = BipartiteGNN(cell_in_dim=in_dim, hidden=hidden, rounds=rounds).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, ec, en, n_nets)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        if (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

    return model, feat_mean, feat_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--raw-dir", default="gnn/raw")
    ap.add_argument("--grid-raw-dir", default="dataset/raw", help="for grid alignment + ground truth")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", default="gnn/cv_summary.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_data = {d: load_design(d, args.raw_dir) for d in args.designs}
    testable = [d for d in args.designs if all_data[d]["label"].sum() > 0]
    print(f"designs usable as test fold (have positive cells): {testable}")

    in_dim = next(iter(all_data.values()))["node_feat"].shape[1]
    fold_results = []
    for test_design in testable:
        print(f"\n=== fold: test={test_design} ===")
        train_designs = [d for d in args.designs if d != test_design]
        model, feat_mean, feat_std = train_one_fold(
            [all_data[d] for d in train_designs], in_dim, device, epochs=args.epochs)

        model.eval()
        test_d = all_data[test_design]
        x_test = torch.tensor((test_d["node_feat"] - feat_mean) / feat_std, dtype=torch.float32, device=device)
        ec_test = torch.tensor(test_d["edge_cell"], dtype=torch.long, device=device)
        en_test = torch.tensor(test_d["edge_net"], dtype=torch.long, device=device)
        with torch.no_grad():
            scores = torch.sigmoid(model(x_test, ec_test, en_test, int(test_d["n_nets"]))).cpu().numpy()

        grid_label_npz = np.load(Path(args.grid_raw_dir) / f"{test_design}_labels.npz")
        grid_meta = {k: grid_label_npz[k] for k in ["llx_um", "lly_um", "nx", "ny", "bin_size_um"]}
        grid_label = grid_label_npz["label"]

        result = {"test_design": test_design, "train_designs": train_designs,
                  "n_positive_cells": int(test_d["label"].sum())}
        for pct in [1, 5, 10]:
            r = recall_at_topk_grid(scores, test_d["bbox"], grid_meta, grid_label, pct)
            result[f"recall_at_top_{pct}pct"] = r
            print(f"  recall@top-{pct}%: {r}")
        fold_results.append(result)

    summary = {"folds": fold_results}
    for k in ["recall_at_top_1pct", "recall_at_top_5pct", "recall_at_top_10pct"]:
        vals = [f[k] for f in fold_results if f[k] is not None]
        weights = [f["n_positive_cells"] for f in fold_results if f[k] is not None]
        summary[f"naive_mean_{k}"] = float(np.mean(vals)) if vals else None
        summary[f"weighted_mean_{k}"] = float(np.average(vals, weights=weights)) if vals else None

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
