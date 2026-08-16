"""
Parses TritonRoute's DRC violation report (results/.../reports/5_route_drc.rpt)
and rasterizes violation locations onto the same bin grid used for features
(feature_extraction/rasterize.py), producing a ground-truth label raster:

  label[y, x] = number of DRC violations whose bbox overlaps bin (y, x)

Coordinates in the .rpt file are in microns (TritonRoute reports them in the
design's real units), so no dbu conversion is needed here.
"""

import argparse
import re

import numpy as np

from rasterize import load_dump, make_grid

# Matches "bbox = ( x1, y1 ) - ( x2, y2 )" (tolerant of spacing/units-layer suffix)
BBOX_RE = re.compile(
    r"bbox\s*=?\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)\s*-\s*\(\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)",
    re.IGNORECASE,
)
VIOLATION_TYPE_RE = re.compile(r"violation type:\s*(.+)")


def parse_drc_report(path):
    """Returns list of dicts: {type, x0, y0, x1, y1} in microns."""
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return []

    if not text.strip():
        return []

    violations = []
    current_type = "unknown"
    # Walk line by line so each bbox picks up the most recent "violation type:" line
    for line in text.splitlines():
        m = VIOLATION_TYPE_RE.search(line)
        if m:
            current_type = m.group(1).strip()
            continue
        m = BBOX_RE.search(line)
        if m:
            x0, y0, x1, y1 = map(float, m.groups())
            violations.append({
                "type": current_type,
                "x0": min(x0, x1), "x1": max(x0, x1),
                "y0": min(y0, y1), "y1": max(y0, y1),
            })
    return violations


def rasterize_violations(violations, llx, lly, nx, ny, bin_size_um):
    raster = np.zeros((ny, nx), dtype=np.float32)
    for v in violations:
        x0, x1 = v["x0"] - llx, v["x1"] - llx
        y0, y1 = v["y0"] - lly, v["y1"] - lly
        ix0 = max(0, min(nx - 1, int(np.floor(x0 / bin_size_um))))
        ix1 = max(0, min(nx - 1, int(np.floor(max(x1 - 1e-9, x0) / bin_size_um))))
        iy0 = max(0, min(ny - 1, int(np.floor(y0 / bin_size_um))))
        iy1 = max(0, min(ny - 1, int(np.floor(max(y1 - 1e-9, y0) / bin_size_um))))
        raster[iy0:iy1 + 1, ix0:ix1 + 1] += 1.0
    return raster


def build_labels(drc_report_path, dump_json_path, bin_size_um=2.0):
    d = load_dump(dump_json_path)
    dbu = d["dbu_per_micron"]
    llx, lly, nx, ny = make_grid(d["core"], dbu, bin_size_um)

    violations = parse_drc_report(drc_report_path)
    raster = rasterize_violations(violations, llx, lly, nx, ny, bin_size_um)
    return raster, violations, {"llx_um": llx, "lly_um": lly, "nx": nx, "ny": ny, "bin_size_um": bin_size_um}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("drc_report")
    ap.add_argument("dump_json", help="same JSON used for feature extraction, for grid alignment")
    ap.add_argument("out_npz")
    ap.add_argument("--bin-size-um", type=float, default=2.0)
    args = ap.parse_args()

    raster, violations, meta = build_labels(args.drc_report, args.dump_json, args.bin_size_um)
    np.savez_compressed(args.out_npz, label=raster, **meta)
    n_hot_bins = int((raster > 0).sum())
    print(f"parsed {len(violations)} DRC violations -> {n_hot_bins}/{raster.size} hotspot bins -> {args.out_npz}")
    if violations:
        types = {}
        for v in violations:
            types[v["type"]] = types.get(v["type"], 0) + 1
        print("violation types:", types)
