import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class UncertaintyGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_types, RE_c, alpha=1.0, detach_p=True):
        super().__init__()
        # self vs neighbor 两路
        self.lin_self1 = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GCNConv(in_dim, hidden_dim)

        self.lin_self2 = nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, n_types)

        # 存储 VAE 得到的 type-level 不确定性
        self.RE_c = torch.tensor(RE_c, dtype=torch.float32)
        self.alpha = alpha
        self.detach_p = detach_p

    def forward(self, x, edge_index):
        """
        x: (N, G)  spot 表达
        edge_index: 图边
        """
        # 第一层：分别算 self 和 neighbor
        h_self1 = F.relu(self.lin_self1(x))          # (N,H)
        h_neigh1 = F.relu(self.conv1(x, edge_index)) # (N,H)

        # 用邻居信息粗预测 p_tmp
        p_tmp = F.softmax(self.decoder(h_neigh1), dim=-1)  # (N,C)

        # ---- 计算不确定性 u_j ----
        RE_c = self.RE_c.to(x.device)
        RE_z = (RE_c - RE_c.mean()) / (RE_c.std() + 1e-8)
        RE_scaled = torch.sigmoid(self.alpha * RE_z).unsqueeze(0)  # (1,C)

        p_used = p_tmp.detach() if self.detach_p else p_tmp
        u_jc = RE_scaled * (1.0 - p_used)   # (N,C)
        u_j = u_jc.mean(dim=1, keepdim=True)  # (N,1)
        u_j = u_j.clamp(0.0, 1.0)

        # ---- self vs neighbor 插值 ----
        h1 = (1.0 - u_j) * h_self1 + u_j * h_neigh1
        h1 = F.relu(h1)

        # 第二层：普通 GCN
        h2 = F.relu(self.conv2(h1, edge_index))
        logits = self.decoder(h2)
        p = F.softmax(logits, dim=-1)

        return p, u_j


# def compute_loss(p, u, Y, E, edge_index, alpha=0.1):
#     """
#     Y: (N,G) ST 基因表达
#     E: (C,G) scRNA pseudo-bulk
#     p: (N,C) 预测的细胞类型比例
#     u: (N,1) spot 不确定性
#     """
#     Y_hat = p @ torch.tensor(E, dtype=torch.float32, device=Y.device)
#     loss_recon = F.mse_loss(Y_hat, Y)

#     # 邻居平滑 loss
#     row, col = edge_index
#     diff = (p[row] - p[col]).pow(2).sum(1)
#     loss_smooth = (u[row].squeeze() * diff).mean()

#     return loss_recon + alpha * loss_smooth, loss_recon, loss_smooth

def compute_loss(p, u, Y, E, edge_index,
                 alpha=0.1, beta=0.01, prior_type="entropy", prior_target=None,
                 type_uncertainty=None, eps=1e-8):
    """
    GCN/空间去卷积用复合损失（重建 + 不确定性加权平滑 + 先验）。
    若 u 为 None 且提供 type_uncertainty 则自动计算 spot-level u。
    """
    if u is None:
        if type_uncertainty is None:
            raise ValueError("Either u or type_uncertainty must be provided.")
        t_unc = torch.tensor(type_uncertainty, dtype=torch.float32, device=p.device)
        t_unc = (t_unc - t_unc.min()) / (t_unc.max() - t_unc.min() + eps)
        u = ((1.0 - p) * t_unc.unsqueeze(0)).mean(dim=1, keepdim=True)

    # ensure E is tensor on same device as Y
    E_t = torch.tensor(E, dtype=torch.float32, device=Y.device) if not torch.is_tensor(E) else E.to(Y.device).float()

    # 1) reconstruction in gene space
    Y_hat = p @ E_t
    loss_recon = F.mse_loss(Y_hat, Y, reduction="mean")

    # 2) graph smoothness (symmetric edge weights)
    if isinstance(edge_index, (tuple, list)):
        row, col = edge_index
    else:
        row, col = edge_index[0], edge_index[1]
    diff = (p[row] - p[col]).pow(2).sum(dim=1)
    edge_w = 0.5 * (u[row].squeeze() + u[col].squeeze())
    loss_smooth = (edge_w * diff).mean()

    # 3) prior on p
    if prior_type == "entropy":
        loss_prior = -(p * (p + 1e-8).log()).sum(dim=1).mean()
    elif prior_type == "l1":
        loss_prior = p.abs().sum(dim=1).mean()
    elif prior_type == "kl":
        if prior_target is None:
            raise ValueError("KL prior requires prior_target")
        q = torch.tensor(prior_target, dtype=torch.float32, device=p.device)
        q = q / q.sum()
        p_mean = p.mean(0)
        loss_prior = F.kl_div((p_mean + 1e-8).log(), q, reduction="batchmean")
    elif prior_type is None or prior_type == "none":
        # 明确允许不使用 prior，返回 0 张量以便后续计算/返回
        loss_prior = torch.tensor(0.0, device=p.device, dtype=p.dtype)
    else:
        raise ValueError("Unknown prior_type")

    loss = loss_recon + alpha * loss_smooth + beta * loss_prior
    return loss, loss_recon, loss_smooth, loss_prior