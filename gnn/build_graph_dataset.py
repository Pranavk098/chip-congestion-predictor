"""
Builds a per-design netlist graph dataset for GNN-based hotspot prediction,
an alternative signal source to the raster/CNN pipeline: instead of learning
from a rasterized density image, this learns directly from netlist topology
(which cells are connected to which, via which nets) -- the structurally
different signal that LHNN/PGNN-style papers report beats CNN/U-Net by 35%+
F1 on congestion prediction (see README's "Research this builds on").

Bipartite star-expansion graph: cell nodes and net-hub nodes, edges only
between a cell and the nets it participates in (no cell-cell clique, which
would blow up combinatorially for high-fanout nets). Global power/ground
nets (degree above --max-net-degree) are excluded from message passing --
standard practice, since a net touching most of the design carries no
localized topology signal and would dominate every node's neighborhood.

Label: a cell is positive if its bbox overlaps a real DRC violation bbox
(same parser as feature_extraction/extract_labels.py, same source reports).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "feature_extraction"))
from extract_labels import parse_drc_report  # noqa: E402


def build(graph_json_path, drc_report_path, out_npz, max_net_degree=200):
    d = json.load(open(graph_json_path))
    dbu = d["dbu_per_micron"]
    die = d["die"]
    die_w = (die["urx"] - die["llx"]) / dbu
    die_h = (die["ury"] - die["lly"]) / dbu

    insts = d["instances"]
    n_cells = len(insts)

    cx = np.array([(i["llx"] + i["urx"]) / 2 / dbu for i in insts], dtype=np.float32)
    cy = np.array([(i["lly"] + i["ury"]) / 2 / dbu for i in insts], dtype=np.float32)
    w = np.array([(i["urx"] - i["llx"]) / dbu for i in insts], dtype=np.float32)
    h = np.array([(i["ury"] - i["lly"]) / dbu for i in insts], dtype=np.float32)
    is_macro = np.array([1.0 if i["is_macro"] else 0.0 for i in insts], dtype=np.float32)
    bbox_x0 = np.array([i["llx"] / dbu for i in insts], dtype=np.float32)
    bbox_x1 = np.array([i["urx"] / dbu for i in insts], dtype=np.float32)
    bbox_y0 = np.array([i["lly"] / dbu for i in insts], dtype=np.float32)
    bbox_y1 = np.array([i["ury"] / dbu for i in insts], dtype=np.float32)

    # --- filter mega-fanout (power/ground-like) nets, build bipartite edges ---
    nets = d["nets"]
    edge_cell, edge_net = [], []
    net_degree = []
    kept_net_idx = 0
    pin_count = np.zeros(n_cells, dtype=np.float32)
    for net in nets:
        member_insts = sorted(set(net["insts"]))
        deg = len(member_insts)
        for c in net["insts"]:
            if 0 <= c < n_cells:
                pin_count[c] += 1
        if deg == 0 or deg > max_net_degree:
            continue
        for c in member_insts:
            edge_cell.append(c)
            edge_net.append(kept_net_idx)
        net_degree.append(deg)
        kept_net_idx += 1
    n_nets = kept_net_idx

    net_degree = np.array(net_degree, dtype=np.float32) if n_nets else np.zeros(0, dtype=np.float32)

    # --- labels: cell overlaps a real DRC violation bbox ---
    violations = parse_drc_report(drc_report_path)
    label = np.zeros(n_cells, dtype=np.float32)
    if violations:
        vx0 = np.array([v["x0"] for v in violations])
        vx1 = np.array([v["x1"] for v in violations])
        vy0 = np.array([v["y0"] for v in violations])
        vy1 = np.array([v["y1"] for v in violations])
        for i in range(n_cells):
            overlap = (bbox_x0[i] <= vx1) & (bbox_x1[i] >= vx0) & (bbox_y0[i] <= vy1) & (bbox_y1[i] >= vy0)
            if overlap.any():
                label[i] = 1.0

    node_feat = np.stack([
        cx / max(die_w, 1e-6),
        cy / max(die_h, 1e-6),
        np.log1p(w),
        np.log1p(h),
        np.log1p(w * h),
        is_macro,
        np.log1p(pin_count),
    ], axis=1).astype(np.float32)

    np.savez_compressed(
        out_npz,
        node_feat=node_feat,
        label=label,
        bbox=np.stack([bbox_x0, bbox_y0, bbox_x1, bbox_y1], axis=1),
        edge_cell=np.array(edge_cell, dtype=np.int64),
        edge_net=np.array(edge_net, dtype=np.int64),
        net_degree=net_degree,
        n_cells=n_cells, n_nets=n_nets,
    )
    print(f"{graph_json_path}: {n_cells} cells, {n_nets}/{len(nets)} nets kept "
          f"(dropped {len(nets) - n_nets} with degree>{max_net_degree} or 0), "
          f"{int(label.sum())} positive cells, {len(edge_cell)} edges -> {out_npz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json")
    ap.add_argument("drc_report")
    ap.add_argument("out_npz")
    ap.add_argument("--max-net-degree", type=int, default=200)
    args = ap.parse_args()
    build(args.graph_json, args.drc_report, args.out_npz, args.max_net_degree)
