"""
Combines per-design feature/label rasters (produced by feature_extraction/)
into a patch dataset for training the U-Net.

Layout expected on disk (see scripts/run_pipeline.sh):
  dataset/raw/<design>_features.npz  (key "features", shape (3, ny, nx))
  dataset/raw/<design>_labels.npz    (key "label",    shape (ny, nx))

Patches are extracted with a sliding window (stride < patch_size for overlap),
zero-padded if a design's raster is smaller than one patch. Split is by
DESIGN, not by patch, so validation/test measure generalization to an unseen
design rather than leaking neighboring patches of the same layout across
splits.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_design(raw_dir, design):
    feat_path = Path(raw_dir) / f"{design}_features.npz"
    label_path = Path(raw_dir) / f"{design}_labels.npz"
    feats = np.load(feat_path)["features"]  # (3, ny, nx)
    label = np.load(label_path)["label"]    # (ny, nx)
    assert feats.shape[1:] == label.shape, f"{design}: feature/label grid mismatch {feats.shape} vs {label.shape}"
    return feats, label


def pad_to_at_least(arr, min_h, min_w):
    c_or_none = arr.shape[0] if arr.ndim == 3 else None
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 3:
        return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")


def extract_patches(feats, label, patch_size, stride):
    feats = pad_to_at_least(feats, patch_size, patch_size)
    label = pad_to_at_least(label, patch_size, patch_size)
    _, h, w = feats.shape

    patches_x, patches_y = [], []
    ys = list(range(0, h - patch_size + 1, stride)) or [0]
    xs = list(range(0, w - patch_size + 1, stride)) or [0]
    if ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    for y in ys:
        for x in xs:
            fp = feats[:, y:y + patch_size, x:x + patch_size]
            lp = label[y:y + patch_size, x:x + patch_size]
            patches_x.append(fp)
            patches_y.append(lp)
    return np.stack(patches_x), np.stack(patches_y)


def augment(patches_x, patches_y):
    """Adds horizontal- and vertical-flipped copies (rows stay horizontal, so
    flips are physically realistic augmentations for standard-cell layouts)."""
    xs = [patches_x, patches_x[:, :, :, ::-1], patches_x[:, :, ::-1, :]]
    ys = [patches_y, patches_y[:, :, ::-1], patches_y[:, ::-1, :]]
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="dataset/raw")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--train-designs", nargs="+", required=True)
    ap.add_argument("--test-designs", nargs="+", required=True)
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--val-fraction", type=float, default=0.15,
                     help="fraction of TRAIN-design patches held out for validation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def collect(designs):
        xs, ys, design_ids = [], [], []
        for d in designs:
            feats, label = load_design(args.raw_dir, d)
            px, py = extract_patches(feats, label, args.patch_size, args.stride)
            xs.append(px)
            ys.append(py)
            design_ids += [d] * len(px)
            print(f"  {d}: raster {feats.shape[1:]} -> {len(px)} patches "
                  f"({(py > 0).any(axis=(1, 2)).sum()} contain a violation)")
        return np.concatenate(xs), np.concatenate(ys), design_ids

    print("Train/val designs:", args.train_designs)
    train_x, train_y, train_ids = collect(args.train_designs)
    print("Test designs:", args.test_designs)
    test_x, test_y, test_ids = collect(args.test_designs)

    n = len(train_x)
    idx = rng.permutation(n)
    n_val = max(1, int(n * args.val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    val_x, val_y = train_x[val_idx], train_y[val_idx]
    tr_x, tr_y = train_x[tr_idx], train_y[tr_idx]

    if not args.no_augment:
        tr_x, tr_y = augment(tr_x, tr_y)

    np.savez_compressed(out_dir / "train.npz", x=tr_x, y=tr_y)
    np.savez_compressed(out_dir / "val.npz", x=val_x, y=val_y)
    np.savez_compressed(out_dir / "test.npz", x=test_x, y=test_y)

    manifest = {
        "train_designs": args.train_designs,
        "test_designs": args.test_designs,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "n_train": len(tr_x),
        "n_val": len(val_x),
        "n_test": len(test_x),
        "train_violation_patch_fraction": float((tr_y > 0).any(axis=(1, 2)).mean()),
        "val_violation_patch_fraction": float((val_y > 0).any(axis=(1, 2)).mean()),
        "test_violation_patch_fraction": float((test_y > 0).any(axis=(1, 2)).mean()),
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Wrote train/val/test .npz + manifest.json to {out_dir}")


if __name__ == "__main__":
    main()
