"""
Re-runs inference on a couple of real held-out test patches and dumps the
raw arrays (input feature, predicted probability, ground truth) as compact
JSON, for embedding directly in the interactive demo artifact -- real
numbers from the actual trained checkpoints, not illustrative fake data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from unet import build_model  # noqa: E402
from train import normalize  # noqa: E402


def load_split(dataset_dir, split):
    d = np.load(Path(dataset_dir) / f"{split}.npz")
    return torch.from_numpy(d["x"]).float(), torch.from_numpy(d["y"]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature-channel", type=int, default=1, help="which input channel to export for display")
    ap.add_argument("--max-size", type=int, default=None,
                     help="downsample (block-max-pool) each array to at most this many bins per side, for a smaller embed")
    args = ap.parse_args()

    device = torch.device("cpu")
    x, y = load_split(args.dataset_dir, args.split)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    x_norm = normalize(x, ckpt["feat_mean"], ckpt["feat_std"])

    model = build_model(ckpt["arch"], in_channels=x.shape[1], base_ch=ckpt.get("base_ch", 32))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    targets = (y > 0).float()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_norm)).squeeze(1)

    pos_idx = [i for i in range(len(targets)) if targets[i].sum() > 0]
    pos_idx.sort(key=lambda i: -targets[i].sum().item())
    chosen = pos_idx[:args.n_samples] if pos_idx else list(range(min(args.n_samples, len(targets))))

    def block_max_pool(arr, max_size):
        h, w = arr.shape
        if max_size is None or (h <= max_size and w <= max_size):
            return arr
        fh, fw = max(1, h // max_size), max(1, w // max_size)
        nh, nw = h // fh, w // fw
        cropped = arr[:nh * fh, :nw * fw]
        return cropped.reshape(nh, fh, nw, fw).max(axis=(1, 3))

    samples = []
    for i in chosen:
        feat = block_max_pool(x[i, args.feature_channel].numpy(), args.max_size)
        prob = block_max_pool(probs[i].numpy(), args.max_size)
        targ = block_max_pool(targets[i].numpy(), args.max_size)
        samples.append({
            "feature": np.round(feat, 4).tolist(),
            "prob": np.round(prob, 4).tolist(),
            "target": targ.astype(int).tolist(),
            "n_violations": int(targets[i].sum().item()),
        })

    Path(args.out).write_text(json.dumps({"samples": samples}))
    print(f"wrote {len(samples)} samples -> {args.out}")


if __name__ == "__main__":
    main()
