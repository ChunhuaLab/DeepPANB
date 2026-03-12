# model.py
import torch
from torch import nn
from torch_scatter import scatter

class PainnResidueMessage(nn.Module):
    def __init__(self, node_size, edge_size, cutoff, dropout_rate=0.1):
        super().__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.cutoff = cutoff

        # g_ij = phi_H([s_j, e_ij]) -> produces 3*H channels for split
        self.scalar_mlp = nn.Sequential(
            nn.Linear(node_size + edge_size, node_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(node_size, node_size * 3)
        )

        # F_ij = phi_F(rho(r_ij)) * c(r_ij) -> also 3*H to match split
        self.filter_net = nn.Sequential(
            nn.Linear(20 + edge_size, node_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(node_size, node_size * 3)
        )

    def forward(self, s, vec, edge_index, edge_diff, edge_dist, edge_attr):
        # s:   [N, H]
        # vec: [N, 3, H]
        # edge_attr: [E, H]
        row, col = edge_index  # [E], [E]

        # ---- radial basis (20 dim) + cosine cutoff ----
        # radial: [E, 20]
        radial = torch.sin(
            edge_dist.unsqueeze(-1)
            * torch.arange(1, 21, device=edge_dist.device, dtype=edge_dist.dtype)
            * torch.pi / self.cutoff
        )
        # cutoff: [E]
        cutoff = 0.5 * (torch.cos(edge_dist * torch.pi / self.cutoff) + 1.0)

        # ---- filter ----
        # concat radial and learned edge_attr (already in H-dim)
        # filter_input: [E, 20 + H] -> filter_weight: [E, 3H]
        filter_input = torch.cat([radial, edge_attr], dim=-1)
        filter_weight = self.filter_net(filter_input) * cutoff.unsqueeze(-1)

        # ---- gates from neighbor scalars + edge embedding ----
        # s[col]: [E, H]; edge_attr: [E, H] -> [E, 2H] -> [E, 3H]
        s_j = self.scalar_mlp(torch.cat([s[col], edge_attr], dim=-1))

        # combine and split into (vec gate, dir gate, scalar msg), each [E, H]
        gate_vec, gate_edge, msg_s = torch.split(
            s_j * filter_weight,
            self.node_size,
            dim=-1
        )

        # unit edge direction \hat{r}_{ij}: [E, 3]
        edge_dir = edge_diff / (edge_dist.unsqueeze(-1) + 1e-7)

        # ---- vector message ----
        # vec[col]: [E, 3, H]
        # gate_vec: [E, H] -> [E, 1, H]
        # edge_dir: [E, 3] -> [E, 3, 1]; gate_edge: [E, H] -> [E, 1, H]
        msg_vec = vec[col] * gate_vec.unsqueeze(1) + edge_dir.unsqueeze(-1) * gate_edge.unsqueeze(1)

        # ---- aggregate to target node i = row ----
        # msg_s:   [E, H]   -> [N, H]
        # msg_vec: [E, 3,H] -> [N, 3, H]
        s_out = scatter(msg_s, row, dim=0, dim_size=s.size(0))
        vec_out = scatter(msg_vec, row, dim=0, dim_size=vec.size(0))

        # residual add
        return s + s_out, vec + vec_out


class PainnResidueUpdate(nn.Module):
    def __init__(self, node_size, dropout_rate=0.1):
        super().__init__()
        self.node_size = node_size

        # linear maps U, V applied on feature dim H (keep 3D axis intact)
        self.vec_transform_U = nn.Linear(node_size, node_size)
        self.vec_transform_V = nn.Linear(node_size, node_size)

        # [s, ||Vv||] -> [a_vv, a_ss, a_sv]   (original Eq.(9): two scalar gates + one "dot product coefficient")
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(node_size, node_size * 3)
        )

    def forward(self, s, vec):
        # s:   [N, H]
        # vec: [N, 3, H]

        # Uv, Vv  (linear transform on H dim only; keep 3D axis)
        Uv = self.vec_transform_U(vec)   # [N, 3, H]
        Vv = self.vec_transform_V(vec)   # [N, 3, H]

        # ||Vv||_2 over 3D axis -> [N, H]
        v_norm = torch.linalg.norm(Vv, dim=1)

        # a_vv, a_ss, a_sv  <-  a(s, ||Vv||)
        a_vv, a_ss, a_sv = torch.split(
            self.update_mlp(torch.cat([v_norm, s], dim=-1)),
            self.node_size, dim=-1
        )

        # <Uv, Vv> over 3D axis -> [N, H]    (dot product term from original Eq.(9))
        dot_uv = (Uv * Vv).sum(dim=1)

        # delta_s = a_ss + a_sv * <Uv, Vv>        (Eq.9)
        delta_s = a_ss + a_sv * dot_uv

        # delta_v = a_vv * (Uv)                   (Eq.10)
        delta_vec = a_vv.unsqueeze(1) * Uv

        return s + delta_s, vec + delta_vec



class PAINN(nn.Module):
    def __init__(self, input_dim=1583, edge_dim=1, hidden_dim=128, num_layers=4, cutoff=15.0, dropout_rate=0.1):
        super().__init__()
        self.cutoff = cutoff
        self.hidden_dim = hidden_dim

        # scalar embedding
        self.s_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )

        # learnable edge encoder: [E, edge_dim] -> [E, H]
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.message_layers = nn.ModuleList()
        self.update_layers = nn.ModuleList()

        for _ in range(num_layers):
            self.message_layers.append(
                PainnResidueMessage(
                    node_size=hidden_dim,
                    edge_size=hidden_dim,
                    cutoff=cutoff
                )
            )
            self.update_layers.append(
                PainnResidueUpdate(hidden_dim, dropout_rate=dropout_rate)
            )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data):
        # s: [N, H]
        s = self.s_embed(data.x)

        # v^{(0)} = 0, keep 3D axis explicit: [N, 3, H]
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device, dtype=s.dtype)

        # edges
        edge_index = data.edge_index
        row, col = edge_index
        edge_diff = data.pos[row] - data.pos[col]              # [E, 3]
        edge_dist = torch.norm(edge_diff, dim=-1)              # [E]

        # edge_attr: [E, edge_dim] -> [E, H]
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            # fallback if no edge_attr provided
            edge_attr = torch.zeros(edge_index.size(1), self.hidden_dim, device=s.device, dtype=s.dtype)

        # message passing stacks
        for msg_layer, upd_layer in zip(self.message_layers, self.update_layers):
            s, vec = msg_layer(s, vec, edge_index, edge_diff, edge_dist, edge_attr)
            s, vec = upd_layer(s, vec)

        # sigmoid outside if you keep BCELoss; here we keep your original BCELoss setup
        return torch.sigmoid(self.output(s))
