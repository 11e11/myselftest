import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ImprovedUncertaintyGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_types, type_uncertainty, dropout=0.2):
        super().__init__()
        # 更复杂的GCN架构
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # MLP解码器
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_types)
        )
        
        # 不确定性嵌入
        self.type_uncertainty = nn.Parameter(torch.tensor(type_uncertainty, dtype=torch.float32), 
                                            requires_grad=False)
        self.uncertainty_embed = nn.Linear(n_types, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # 基本GCN处理
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)
        
        h2 = F.relu(self.conv2(h1, edge_index))
        h2 = self.dropout(h2)
        
        h3 = F.relu(self.conv3(h2, edge_index))
        
        # 粗略预测初始细胞类型比例
        logits_initial = self.mlp(h3)
        p_initial = F.softmax(logits_initial, dim=-1)
        
        # 基于预测比例和不确定性生成注意力权重
        uncertainty_weights = p_initial * self.type_uncertainty.unsqueeze(0)
        uncertainty_signal = self.uncertainty_embed(uncertainty_weights)
        
        # 自注意力机制融合空间上下文信息
        h3_reshaped = h3.unsqueeze(0)  # 适应注意力层输入格式
        attn_output, _ = self.attention(
            h3_reshaped + uncertainty_signal.unsqueeze(0),
            h3_reshaped,
            h3_reshaped
        )
        
        # 最终解码
        h_final = h3 + attn_output.squeeze(0)
        logits = self.mlp(h_final)
        p = F.softmax(logits, dim=-1)
        
        # 计算整体不确定性
        total_uncertainty = (p * self.type_uncertainty.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        return p, total_uncertainty
    
def improved_loss(p, y, E, edge_index, type_uncertainty, alpha=0.1, beta=0.01, gamma=0.05):
    """
    综合损失函数:
    1. 重建损失: MSE或Poisson NLL
    2. 不确定性加权图平滑
    3. 稀疏性正则化
    4. 基于不确定性的自适应KL惩罚
    """
    # 设备转换
    device = y.device
    E_tensor = torch.tensor(E, dtype=torch.float32, device=device)
    
    # 1. 基因空间重建损失
    y_hat = p @ E_tensor
    # 可以选用泊松损失代替MSE
    # recon_loss = F.poisson_nll_loss(y_hat, y, log_input=False, reduction="mean")
    recon_loss = F.mse_loss(y_hat, y)
    
    # 2. 不确定性加权图平滑
    row, col = edge_index
    # 计算不确定性相似度
    unc_weights = torch.tensor(type_uncertainty, dtype=torch.float32, device=device)
    p_diff = (p[row] - p[col]).pow(2)
    # 基于不确定性动态调整邻居权重
    edge_weights = torch.exp(-torch.sum(p_diff * unc_weights.unsqueeze(0), dim=1))
    smooth_loss = (1 - edge_weights).mean()
    
    # 3. 稀疏性正则化: 鼓励每个spot的细胞类型组成简单
    sparsity_loss = -((p + 1e-8) * torch.log(p + 1e-8)).sum(dim=1).mean()
    
    # 4. 基于不确定性的自适应KL惩罚
    # 高不确定性类型应该有更宽松的分布约束
    unc_scaled = F.softmax(-torch.tensor(type_uncertainty, device=device), dim=0)
    prior = unc_scaled.repeat(p.shape[0], 1)  # 展开到batch维度
    kl_loss = F.kl_div((p + 1e-8).log(), prior, reduction="batchmean")
    
    # 总损失
    total_loss = recon_loss + alpha * smooth_loss + beta * sparsity_loss + gamma * kl_loss
    
    return total_loss, recon_loss, smooth_loss, sparsity_loss, kl_loss