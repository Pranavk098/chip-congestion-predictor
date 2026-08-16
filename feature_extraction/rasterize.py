"""
Turns the JSON layout dump (feature_extraction/dump_layout.tcl output) into
multi-channel numpy rasters. Channel set and formulas follow the CircuitNet
benchmark's published DRC-prediction recipe (Chai et al., "CircuitNet: An
Open-Source Dataset for ML in EDA", 2022 -- see feature_extraction/src/read.py
in circuitnet_repo for the reference implementation this was checked against):

  0: cell_density   -- exact placed-cell area fraction per bin
  1: pin_density    -- pin count per bin
  2: RUDY           -- Rectangular Uniform wire Density: per net, weight
                       1/w + 1/h spread uniformly over the net's pin bbox
                       (verified identical to CircuitNet's compute_RUDY)
  3: RUDY_long      -- RUDY contribution from nets spanning >1 bin (non-local)
  4: RUDY_short     -- RUDY contribution from nets contained in a single bin
  5: pin_RUDY       -- pin density weighted by its net's RUDY weight
                       (local pin clustering scaled by how demanding the net is)
  6: macro_region   -- binary mask of macro/blockage instances (all-zero for
                       the pure-standard-cell sky130hd test designs used here;
                       kept for architecture/channel-count compatibility with
                       designs that do have macros)
  7: GR_congestion_horizontal -- real OpenROAD global-router congestion
                       (horizontal), from actual post-global-route violation
                       reports, not a placement-only proxy
  8: GR_congestion_vertical   -- same, vertical

Grid is defined in real microns (bin_size_um) so density is physically
comparable across designs of different sizes.
"""

import argparse
import json

import numpy as np


def load_dump(path):
    with open(path) as f:
        return json.load(f)


def make_grid(core, dbu_per_micron, bin_size_um):
    llx, lly = core["llx"] / dbu_per_micron, core["lly"] / dbu_per_micron
    urx, ury = core["urx"] / dbu_per_micron, core["ury"] / dbu_per_micron
    width_um = urx - llx
    height_um = ury - lly
    nx = max(1, int(np.ceil(width_um / bin_size_um)))
    ny = max(1, int(np.ceil(height_um / bin_size_um)))
    return llx, lly, nx, ny


def cell_density_raster(instances, llx, lly, nx, ny, bin_size_um, dbu_per_micron):
    """Exact axis-aligned overlap area of each placed-cell bbox with each grid bin."""
    raster = np.zeros((ny, nx), dtype=np.float32)
    for inst in instances:
        if not inst["placed"]:
            continue
        x0 = inst["llx"] / dbu_per_micron - llx
        x1 = inst["urx"] / dbu_per_micron - llx
        y0 = inst["lly"] / dbu_per_micron - lly
        y1 = inst["ury"] / dbu_per_micron - lly
        if x1 <= x0 or y1 <= y0:
            continue

        ix0 = max(0, int(np.floor(x0 / bin_size_um)))
        ix1 = min(nx - 1, int(np.floor((x1 - 1e-9) / bin_size_um)))
        iy0 = max(0, int(np.floor(y0 / bin_size_um)))
        iy1 = min(ny - 1, int(np.floor((y1 - 1e-9) / bin_size_um)))
        if ix1 < ix0 or iy1 < iy0:
            continue

        xs = np.arange(ix0, ix1 + 2) * bin_size_um
        ys = np.arange(iy0, iy1 + 2) * bin_size_um
        overlap_x = np.clip(np.minimum(xs[1:], x1) - np.maximum(xs[:-1], x0), 0, None)
        overlap_y = np.clip(np.minimum(ys[1:], y1) - np.maximum(ys[:-1], y0), 0, None)
        area = overlap_y[:, None] * overlap_x[None, :]
        raster[iy0:iy1 + 1, ix0:ix1 + 1] += area

    bin_area = bin_size_um * bin_size_um
    return raster / bin_area  # fraction of bin covered by cells


def pin_density_raster(nets, llx, lly, nx, ny, bin_size_um, dbu_per_micron):
    xs, ys = [], []
    for pins in nets:
        for (px, py) in pins:
            xs.append(px / dbu_per_micron - llx)
            ys.append(py / dbu_per_micron - lly)
    raster = np.zeros((ny, nx), dtype=np.float32)
    if not xs:
        return raster
    xs = np.clip(np.array(xs) / bin_size_um, 0, nx - 1e-6).astype(int)
    ys = np.clip(np.array(ys) / bin_size_um, 0, ny - 1e-6).astype(int)
    np.add.at(raster, (ys, xs), 1.0)
    return raster


def rudy_family_rasters(nets, llx, lly, nx, ny, bin_size_um, dbu_per_micron):
    """Computes RUDY, RUDY_long, RUDY_short, pin_RUDY, pin_RUDY_long in one
    pass over nets.

    RUDY(net) = (bbox_w + bbox_h) / (bbox_w * bbox_h) = 1/w + 1/h, spread
    uniformly (area-weighted) over the net's pin bounding box -- this is
    CircuitNet's compute_RUDY weight formula, verified against their source.

    long/short split follows CircuitNet's definition exactly: a net is
    "short" if its bbox is contained within a single grid bin in both x and
    y (i.e. purely local), "long" otherwise.

    pin_RUDY accumulates each net's RUDY weight at its individual PIN
    locations (not spread over the bbox) -- a local pin-clustering signal
    scaled by how routing-demanding the owning net is.
    """
    rudy = np.zeros((ny, nx), dtype=np.float32)
    rudy_long = np.zeros((ny, nx), dtype=np.float32)
    rudy_short = np.zeros((ny, nx), dtype=np.float32)
    pin_rudy = np.zeros((ny, nx), dtype=np.float32)
    pin_rudy_long = np.zeros((ny, nx), dtype=np.float32)
    min_extent = bin_size_um * 0.5  # avoid div-by-zero for degenerate (0-area) nets

    for pins in nets:
        if len(pins) < 2:
            continue
        xs_um = [p[0] / dbu_per_micron - llx for p in pins]
        ys_um = [p[1] / dbu_per_micron - lly for p in pins]
        x0, x1 = min(xs_um), max(xs_um)
        y0, y1 = min(ys_um), max(ys_um)
        w = max(x1 - x0, min_extent)
        h = max(y1 - y0, min_extent)
        weight = 1.0 / w + 1.0 / h

        ix0 = max(0, int(np.floor(x0 / bin_size_um)))
        ix1 = min(nx - 1, int(np.floor(x1 / bin_size_um)))
        iy0 = max(0, int(np.floor(y0 / bin_size_um)))
        iy1 = min(ny - 1, int(np.floor(y1 / bin_size_um)))
        if ix1 < ix0:
            ix1 = ix0
        if iy1 < iy0:
            iy1 = iy0
        is_long = (ix1 > ix0) or (iy1 > iy0)

        # weight by fractional overlap of each covered bin with [x0,x1]x[y0,y1]
        xs = np.arange(ix0, ix1 + 2) * bin_size_um
        ys = np.arange(iy0, iy1 + 2) * bin_size_um
        overlap_x = np.clip(np.minimum(xs[1:], x1) - np.maximum(xs[:-1], x0), 0, None)
        overlap_y = np.clip(np.minimum(ys[1:], y1) - np.maximum(ys[:-1], y0), 0, None)
        bbox_area = max(w * h, 1e-9)
        area_weight = (overlap_y[:, None] * overlap_x[None, :]) / bbox_area
        contribution = weight * area_weight

        rudy[iy0:iy1 + 1, ix0:ix1 + 1] += contribution
        if is_long:
            rudy_long[iy0:iy1 + 1, ix0:ix1 + 1] += contribution
        else:
            rudy_short[iy0:iy1 + 1, ix0:ix1 + 1] += contribution

        for (px, py) in pins:
            pin_ix = int(np.clip((px / dbu_per_micron - llx) / bin_size_um, 0, nx - 1e-6))
            pin_iy = int(np.clip((py / dbu_per_micron - lly) / bin_size_um, 0, ny - 1e-6))
            pin_rudy[pin_iy, pin_ix] += weight
            if is_long:
                pin_rudy_long[pin_iy, pin_ix] += weight

    return rudy, rudy_long, rudy_short, pin_rudy, pin_rudy_long


def macro_region_raster(instances, llx, lly, nx, ny, bin_size_um, dbu_per_micron):
    """Binary mask: 1 where a macro/blockage instance overlaps the bin."""
    raster = np.zeros((ny, nx), dtype=np.float32)
    for inst in instances:
        if not inst.get("is_macro"):
            continue
        x0 = inst["llx"] / dbu_per_micron - llx
        x1 = inst["urx"] / dbu_per_micron - llx
        y0 = inst["lly"] / dbu_per_micron - lly
        y1 = inst["ury"] / dbu_per_micron - lly
        if x1 <= x0 or y1 <= y0:
            continue
        ix0 = max(0, int(np.floor(x0 / bin_size_um)))
        ix1 = min(nx - 1, int(np.floor((x1 - 1e-9) / bin_size_um)))
        iy0 = max(0, int(np.floor(y0 / bin_size_um)))
        iy1 = min(ny - 1, int(np.floor((y1 - 1e-9) / bin_size_um)))
        if ix1 < ix0 or iy1 < iy0:
            continue
        raster[iy0:iy1 + 1, ix0:ix1 + 1] = 1.0
    return raster


CHANNEL_NAMES = [
    "cell_density", "pin_density", "RUDY", "RUDY_long", "RUDY_short",
    "pin_RUDY", "macro_region", "GR_congestion_horizontal", "GR_congestion_vertical",
]


def build_features(dump_path, bin_size_um=2.0, congestion_rpt_path=None):
    d = load_dump(dump_path)
    dbu = d["dbu_per_micron"]
    llx, lly, nx, ny = make_grid(d["core"], dbu, bin_size_um)

    cell = cell_density_raster(d["instances"], llx, lly, nx, ny, bin_size_um, dbu)
    pin = pin_density_raster(d["nets"], llx, lly, nx, ny, bin_size_um, dbu)
    rudy, rudy_long, rudy_short, pin_rudy, pin_rudy_long = rudy_family_rasters(
        d["nets"], llx, lly, nx, ny, bin_size_um, dbu)
    macro = macro_region_raster(d["instances"], llx, lly, nx, ny, bin_size_um, dbu)

    if congestion_rpt_path is not None:
        # imported lazily to avoid a hard dependency when congestion data isn't available
        from extract_congestion import build_congestion_rasters
        gr_h, gr_v = build_congestion_rasters(congestion_rpt_path, llx, lly, nx, ny, bin_size_um)
    else:
        gr_h = np.zeros((ny, nx), dtype=np.float32)
        gr_v = np.zeros((ny, nx), dtype=np.float32)

    stacked = np.stack([cell, pin, rudy, rudy_long, rudy_short, pin_rudy, macro, gr_h, gr_v], axis=0)
    meta = {
        "llx_um": llx, "lly_um": lly, "nx": nx, "ny": ny,
        "bin_size_um": bin_size_um, "dbu_per_micron": dbu,
        "core": d["core"],
    }
    return stacked, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_json")
    ap.add_argument("out_npz")
    ap.add_argument("--bin-size-um", type=float, default=2.0)
    ap.add_argument("--congestion-rpt", default=None,
                     help="path to OpenROAD global-route congestion report (e.g. congestion.rpt); "
                          "adds real GR_congestion_horizontal/vertical channels if given")
    args = ap.parse_args()

    feats, meta = build_features(args.dump_json, args.bin_size_um, args.congestion_rpt)
    np.savez_compressed(args.out_npz, features=feats, **{k: v for k, v in meta.items() if k != "core"})
    print(f"features shape {feats.shape} -> {args.out_npz}")
    for i, name in enumerate(CHANNEL_NAMES):
        print(f"  [{i}] {name:<26} mean={feats[i].mean():.4f} max={feats[i].max():.4f}")
