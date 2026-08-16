"""Splits the preprocessed CircuitNet .npz (all samples) into train/val/test
.npz files in the same format model/train.py and model/evaluate.py expect,
so those scripts can be reused unmodified as a benchmark on real external
data. No patching: each 256x256 resized sample is used whole, matching
CircuitNet's own training convention (their DRC baseline batch_size=8 over
full resized images, not patches)."""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path")
    ap.add_argument("--out-dir", default="external_data/circuitnet_dataset")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.npz_path)
    x, y = d["x"], d["y"]
    n = len(x)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(n)

    n_test = max(1, int(n * args.test_fraction))
    n_val = max(1, int(n * args.val_fraction))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "train.npz", x=x[train_idx], y=y[train_idx])
    np.savez_compressed(out_dir / "val.npz", x=x[val_idx], y=y[val_idx])
    np.savez_compressed(out_dir / "test.npz", x=x[test_idx], y=y[test_idx])
    print(f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} -> {out_dir}")


if __name__ == "__main__":
    main()
