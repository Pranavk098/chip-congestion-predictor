"""
Leave-one-design-out cross-validation: for every design that actually has
DRC violations (there's no meaningful "recall" to measure on an all-clean
design), train on all other extracted designs and test on the held-out one.
Reports per-fold metrics and the average -- the honest measure of whether
the model generalizes to a layout it has never seen, as opposed to the
val/train metrics from a single split, which only measure fit to designs
already seen during training.

Usage:
  python scripts/cross_validate.py --designs gcd aes ibex jpeg riscv32i
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).parent.parent
PY = str(PROJECT_DIR / ".venv" / "Scripts" / "python.exe")


def has_violations(raw_dir, design):
    label = np.load(Path(raw_dir) / f"{design}_labels.npz")["label"]
    return bool((label > 0).any())


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_DIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", required=True)
    ap.add_argument("--raw-dir", default="dataset/raw")
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--arch", default="attention_unet", choices=["unet", "attention_unet"])
    ap.add_argument("--work-dir", default="cv_work")
    ap.add_argument("--init-from", default=None,
                     help="pretrained checkpoint to fine-tune from in every fold (transfer learning)")
    args = ap.parse_args()

    testable = [d for d in args.designs if has_violations(args.raw_dir, d)]
    print(f"designs with real DRC violations (usable as test fold): {testable}")
    print(f"all designs (usable as train-only, e.g. clean designs): {args.designs}")

    work_dir = PROJECT_DIR / args.work_dir
    work_dir.mkdir(exist_ok=True)

    fold_results = []
    for test_design in testable:
        train_designs = [d for d in args.designs if d != test_design]
        fold_dir = work_dir / f"test_{test_design}"
        dataset_dir = fold_dir / "dataset"
        out_dir = fold_dir / "outputs"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        run([PY, "dataset/build_dataset.py",
             "--train-designs", *train_designs,
             "--test-designs", test_design,
             "--patch-size", str(args.patch_size), "--stride", str(args.stride),
             "--out-dir", str(dataset_dir), "--raw-dir", args.raw_dir])

        train_cmd = [PY, "model/train.py",
                     "--dataset-dir", str(dataset_dir), "--out-dir", str(out_dir),
                     "--epochs", str(args.epochs), "--arch", args.arch]
        if args.init_from:
            train_cmd += ["--init-from", args.init_from]
        run(train_cmd)

        run([PY, "model/evaluate.py",
             "--dataset-dir", str(dataset_dir), "--checkpoint", str(out_dir / "best_model.pt"),
             "--out-dir", str(out_dir), "--split", "test"])

        with open(out_dir / "eval_test.json") as f:
            result = json.load(f)
        result["test_design"] = test_design
        result["train_designs"] = train_designs
        fold_results.append(result)

    summary = {
        "folds": [{"test_design": r["test_design"], "train_designs": r["train_designs"],
                   "n_violation_bins": r["n_violation_bins"],
                   "best_f1": r["best_f1"], "best_precision": r["best_precision"], "best_recall": r["best_recall"],
                   "recall_at_top_1pct": r["recall_at_top_1pct"],
                   "recall_at_top_5pct": r["recall_at_top_5pct"],
                   "recall_at_top_10pct": r["recall_at_top_10pct"]}
                  for r in fold_results],
    }
    for k in ["best_f1", "recall_at_top_1pct", "recall_at_top_5pct", "recall_at_top_10pct"]:
        vals = [f[k] for f in summary["folds"] if f[k] is not None]
        summary[f"mean_{k}"] = float(np.mean(vals)) if vals else None

    with open(work_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
