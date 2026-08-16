"""
Builds a feature raster with the EXACT same channel composition and order as
CircuitNet's official DRC-prediction recipe (external_data/build_circuitnet_
dataset.py FEATURE_LIST), so weights pretrained on real CircuitNet data can
be loaded directly onto our own sky130 designs for fine-tuning:

  0: macro_region
  1: cell_density
  2: RUDY_long
  3: RUDY_short
  4: pin_RUDY_long
  5: congestion_eGR_horizontal  (fast, 1-iteration global-route pass)
  6: congestion_eGR_vertical
  7: congestion_GR_horizontal   (fully-converged global-route pass)
  8: congestion_GR_vertical

Each channel is min-max normalized to [0, 1] per design, matching
CircuitNet's own `std()` preprocessing step, so the input distribution seen
by a CircuitNet-pretrained model isn't shifted by scale alone.

Known domain-shift caveat (documented, not hidden): CircuitNet resizes every
sample to a fixed 256x256 grid; we keep each sky130 design's native
micron-binned grid instead (resizing a 39x39 gcd grid up to 256x256, or a
437x438 jpeg grid down to 256x256, would distort real physical density far
more than it would help). This means one "pixel" does not represent exactly
the same physical area across the two domains -- a real limitation of this
transfer-learning experiment, not swept under the rug.
"""

import argparse

import numpy as np

from rasterize import (load_dump, make_grid, cell_density_raster,
                        rudy_family_rasters, macro_region_raster)
from extract_congestion import build_congestion_rasters

CHANNEL_NAMES = [
    "macro_region", "cell_density", "RUDY_long", "RUDY_short", "pin_RUDY_long",
    "congestion_eGR_horizontal", "congestion_eGR_vertical",
    "congestion_GR_horizontal", "congestion_GR_vertical",
]


def min_max_norm(arr):
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def build(dump_path, egr_rpt_path, gr_rpt_path, bin_size_um=2.0):
    d = load_dump(dump_path)
    dbu = d["dbu_per_micron"]
    llx, lly, nx, ny = make_grid(d["core"], dbu, bin_size_um)

    macro = macro_region_raster(d["instances"], llx, lly, nx, ny, bin_size_um, dbu)
    cell = cell_density_raster(d["instances"], llx, lly, nx, ny, bin_size_um, dbu)
    _, rudy_long, rudy_short, _, pin_rudy_long = rudy_family_rasters(
        d["nets"], llx, lly, nx, ny, bin_size_um, dbu)
    egr_h, egr_v = build_congestion_rasters(egr_rpt_path, llx, lly, nx, ny, bin_size_um)
    gr_h, gr_v = build_congestion_rasters(gr_rpt_path, llx, lly, nx, ny, bin_size_um)

    channels = [macro, cell, rudy_long, rudy_short, pin_rudy_long, egr_h, egr_v, gr_h, gr_v]
    stacked = np.stack([min_max_norm(c) for c in channels], axis=0)
    meta = {"llx_um": llx, "lly_um": lly, "nx": nx, "ny": ny,
            "bin_size_um": bin_size_um, "dbu_per_micron": dbu}
    return stacked, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_json")
    ap.add_argument("out_npz")
    ap.add_argument("--egr-rpt", default=None)
    ap.add_argument("--gr-rpt", default=None)
    ap.add_argument("--bin-size-um", type=float, default=2.0)
    args = ap.parse_args()

    feats, meta = build(args.dump_json, args.egr_rpt, args.gr_rpt, args.bin_size_um)
    np.savez_compressed(args.out_npz, features=feats, **meta)
    print(f"features shape {feats.shape} -> {args.out_npz}")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  [{i}] {name:<28} mean={feats[i].mean():.4f} max={feats[i].max():.4f}")
