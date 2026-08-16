"""
Preprocesses a downloaded CircuitNet routability_features tarball into the
same (9-channel feature, 1-channel label) patch format used elsewhere in this
repo, following CircuitNet's OWN official preprocessing exactly (resize to a
common 256x256 grid, per-channel min-max normalize) -- see
circuitnet_repo/routability_ir_drop_prediction/preprocess_scripts/
generate_training_set.py, task == 'DRC' branch, which this reimplements
1:1 (same feature_list, same label clip/scale, same resize functions).

Real data, not our own OpenROAD extraction: this is Chai et al.'s CircuitNet
benchmark (https://circuitnet.github.io) -- real placed/routed samples across
several open-source RISC-V designs (here: 96 Vortex-small variants), used to
benchmark our architecture against a large, independently-produced dataset,
and as a base for transfer learning experiments.

Usage:
  python external_data/build_circuitnet_dataset.py \
      --data-root external_data/circuitnet/Vortex-small \
      --out-npz external_data/circuitnet_vortex_small.npz
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

FEATURE_LIST = [
    "macro_region",
    "cell_density",
    "RUDY/RUDY_long",
    "RUDY/RUDY_short",
    "RUDY/RUDY_pin_long",
    "congestion/congestion_early_global_routing/overflow_based/congestion_eGR_horizontal_overflow",
    "congestion/congestion_early_global_routing/overflow_based/congestion_eGR_vertical_overflow",
    "congestion/congestion_global_routing/overflow_based/congestion_GR_horizontal_overflow",
    "congestion/congestion_global_routing/overflow_based/congestion_GR_vertical_overflow",
]
LABEL = "DRC/DRC_all"
GRID = 256


def zoom_resize(arr):
    """CircuitNet's `resize`: cubic-spline zoom to a fixed 256x256 grid."""
    h, w = arr.shape
    return ndimage.zoom(arr, (GRID / h, GRID / w), order=3)


def cv2_resize(arr):
    """CircuitNet's `resize_cv2`: used for the DRC label specifically."""
    return cv2.resize(arr.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA)


def min_max_norm(arr):
    """CircuitNet's `std`: per-channel min-max normalization to [0, 1]."""
    if arr.max() == 0:
        return arr
    return (arr - arr.min()) / (arr.max() - arr.min())


def load_npz_array(path):
    return np.load(path)["data"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="e.g. external_data/circuitnet/Vortex-small")
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    label_dir = data_root / LABEL
    sample_names = sorted(p.name for p in label_dir.glob("*.npz"))
    print(f"found {len(sample_names)} samples in {label_dir}")

    all_features, all_labels, kept_names = [], [], []
    for name in sample_names:
        try:
            channels = []
            for feat in FEATURE_LIST:
                arr = load_npz_array(data_root / feat / name)
                channels.append(min_max_norm(zoom_resize(arr)).astype(np.float32))
            feature = np.stack(channels, axis=0)  # (9, 256, 256)

            label = load_npz_array(label_dir / name)
            label = np.clip(label, 0, 200)
            label = (cv2_resize(label) / 200).astype(np.float32)  # (256, 256)
        except FileNotFoundError as e:
            print(f"  skipping {name}: missing {e.filename}")
            continue

        all_features.append(feature)
        all_labels.append(label)
        kept_names.append(name)

    x = np.stack(all_features)  # (N, 9, 256, 256)
    y = np.stack(all_labels)    # (N, 256, 256)
    print(f"built {x.shape[0]} samples, feature shape {x.shape[1:]}, label shape {y.shape[1:]}")
    print(f"label: mean={y.mean():.4f} max={y.max():.4f} nonzero-pixel-frac={(y > 0).mean():.4f}")

    np.savez_compressed(args.out_npz, x=x, y=y, names=np.array(kept_names), channel_names=np.array(FEATURE_LIST))
    print(f"wrote {args.out_npz}")


if __name__ == "__main__":
    main()
