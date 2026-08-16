"""
Bipartite message-passing GNN over the netlist star-expansion graph (cell
nodes <-> net hub nodes; see gnn/build_graph_dataset.py). Hand-rolled with
plain PyTorch scatter ops rather than torch_geometric: this graph has one
edge type in one fixed direction pattern (cell<->net), so a library adds
dependency risk without buying real functionality here.

Message passing round:
  net_embed  = mean over connected cells of cell_embed          (cell -> net)
  cell_embed = MLP([cell_embed, mean over connected nets of net_embed])  (net -> cell)

K rounds gives each cell a K-hop netlist neighborhood in its embedding --
the structurally different signal (real topology, not spatial adjacency)
that a raster CNN cannot see.
"""

import torch
import torch.nn as nn


def scatter_mean(src, index, dim_size):
    """Mean-aggregate src[i] into buckets index[i], for dim_size buckets."""
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


class BipartiteGNN(nn.Module):
    def __init__(self, cell_in_dim, hidden=64, rounds=3, dropout=0.1):
        super().__init__()
        self.rounds = rounds
        self.cell_in = nn.Sequential(nn.Linear(cell_in_dim, hidden), nn.ReLU())
        self.net_in = nn.Linear(1, hidden)  # net nodes start from a learned bias (no raw features needed)

        self.cell_update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
            for _ in range(rounds)
        ])
        self.net_update = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))
            for _ in range(rounds)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cell_feat, edge_cell, edge_net, n_nets):
        n_cells = cell_feat.shape[0]
        cell_h = self.cell_in(cell_feat)
        net_h = self.net_in(torch.ones(max(n_nets, 1), 1, device=cell_feat.device))

        for r in range(self.rounds):
            if edge_cell.numel() > 0:
                msg_to_net = scatter_mean(cell_h[edge_cell], edge_net, n_nets)
                net_h = self.net_update[r](msg_to_net + net_h * 0.0) if n_nets > 0 else net_h

                msg_to_cell = scatter_mean(net_h[edge_net], edge_cell, n_cells)
            else:
                msg_to_cell = torch.zeros_like(cell_h)
            cell_h = self.cell_update[r](torch.cat([cell_h, msg_to_cell], dim=1)) + cell_h

        return self.head(cell_h).squeeze(-1)  # per-cell logit
