"""
Parses OpenROAD's real global-router congestion report (written during the
`grt` stage, before any detailed routing happens -- see flow/scripts/
global_route.tcl's `-congestion_report_file`) and rasterizes it into
horizontal/vertical congestion-overflow channels on the same grid as
rasterize.py's other features.

This is real router-computed congestion (capacity vs usage on actual GRT
edges), not a placement-only proxy like RUDY -- the CircuitNet benchmark
(Chai et al. 2022) and RouteNet (Xie et al. 2018) both find global-route
congestion substantially stronger than placement-only features for DRC
prediction, while still being available well before detailed routing (global
route is orders of magnitude cheaper to run than detailed route).

Report format (same style as the DRC report, TritonRoute/GRT house style):
  violation type: Vertical congestion
      srcs: net:... net:...
      comment: capacity:14 usage:15 congestion:1
      bbox = (455.4000, 427.8000) - (462.3000, 434.7000) on Layer -
"""

import argparse
import re

import numpy as np

from rasterize import load_dump, make_grid

BBOX_RE = re.compile(
    r"bbox\s*=?\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*-\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)",
    re.IGNORECASE,
)
TYPE_RE = re.compile(r"violation type:\s*(Horizontal|Vertical) congestion", re.IGNORECASE)
COMMENT_RE = re.compile(r"capacity:\s*(-?[\d.]+)\s+usage:\s*(-?[\d.]+)\s+congestion:\s*(-?[\d.]+)")


def parse_congestion_report(path):
    """Returns list of dicts: {direction, congestion, x0,y0,x1,y1} in microns."""
    if path is None:
        return []
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return []
    if not text.strip():
        return []

    entries = []
    direction, magnitude = None, 1.0
    for line in text.splitlines():
        m = TYPE_RE.search(line)
        if m:
            direction = m.group(1).lower()
            magnitude = 1.0
            continue
        m = COMMENT_RE.search(line)
        if m:
            magnitude = float(m.group(3))
            continue
        m = BBOX_RE.search(line)
        if m and direction is not None:
            x0, y0, x1, y1 = map(float, m.groups())
            entries.append({
                "direction": direction, "magnitude": magnitude,
                "x0": min(x0, x1), "x1": max(x0, x1),
                "y0": min(y0, y1), "y1": max(y0, y1),
            })
    return entries


def _rasterize_entries(entries, llx, lly, nx, ny, bin_size_um):
    raster = np.zeros((ny, nx), dtype=np.float32)
    for e in entries:
        x0, x1 = e["x0"] - llx, e["x1"] - llx
        y0, y1 = e["y0"] - lly, e["y1"] - lly
        ix0 = max(0, min(nx - 1, int(np.floor(x0 / bin_size_um))))
        ix1 = max(0, min(nx - 1, int(np.floor(max(x1 - 1e-9, x0) / bin_size_um))))
        iy0 = max(0, min(ny - 1, int(np.floor(y0 / bin_size_um))))
        iy1 = max(0, min(ny - 1, int(np.floor(max(y1 - 1e-9, y0) / bin_size_um))))
        raster[iy0:iy1 + 1, ix0:ix1 + 1] += e["magnitude"]
    return raster


def build_congestion_rasters(congestion_rpt_path, llx, lly, nx, ny, bin_size_um):
    entries = parse_congestion_report(congestion_rpt_path)
    horizontal = _rasterize_entries([e for e in entries if e["direction"] == "horizontal"],
                                     llx, lly, nx, ny, bin_size_um)
    vertical = _rasterize_entries([e for e in entries if e["direction"] == "vertical"],
                                   llx, lly, nx, ny, bin_size_um)
    return horizontal, vertical


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("congestion_rpt")
    ap.add_argument("dump_json", help="same JSON used for feature extraction, for grid alignment")
    ap.add_argument("out_npz")
    ap.add_argument("--bin-size-um", type=float, default=2.0)
    args = ap.parse_args()

    d = load_dump(args.dump_json)
    dbu = d["dbu_per_micron"]
    llx, lly, nx, ny = make_grid(d["core"], dbu, args.bin_size_um)

    h, v = build_congestion_rasters(args.congestion_rpt, llx, lly, nx, ny, args.bin_size_um)
    np.savez_compressed(args.out_npz, gr_congestion_horizontal=h, gr_congestion_vertical=v)
    entries = parse_congestion_report(args.congestion_rpt)
    print(f"parsed {len(entries)} GR congestion violations -> "
          f"H:{int((h > 0).sum())} V:{int((v > 0).sum())} hotspot bins -> {args.out_npz}")
