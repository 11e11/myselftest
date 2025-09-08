import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

class ImprovedUncertaintyGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_types, type_uncertainty, dropout=0.2, num_heads=4, temp=0.7):
        super().__init__()
        # GCN 基础层
        self.conv1 = GCNConv(in_dim, hidden_dim)
        
        # 用 GATConv 替换全局 MultiheadAttention ——> 只在邻居 edge_index 上做注意力
        self.attn = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout, concat=True)

        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # 解码器
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
        self.temp = temp  # softmax 温度系数

    def forward(self, x, edge_index):
        # 基本 GCN
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = self.dropout(h1)

        # 局部注意力：邻居范围内做 attention
        h2 = F.relu(self.attn(h1, edge_index))
        h2 = self.dropout(h2)

        # 再一层 GCN
        h3 = F.relu(self.conv3(h2, edge_index))

        # 初步预测比例
        logits_initial = self.mlp(h3)
        p_initial = F.softmax(logits_initial / self.temp, dim=-1)

        # 不确定性信号
        uncertainty_weights = p_initial * self.type_uncertainty.unsqueeze(0)
        uncertainty_signal = self.uncertainty_embed(uncertainty_weights)

        # 融合不确定性信号（残差加法）
        h_final = h3 + uncertainty_signal

        # 最终预测
        logits = self.mlp(h_final)
        p = F.softmax(logits / self.temp, dim=-1)

        # 计算整体不确定性（可用于 regularization / 可视化）
        total_uncertainty = (p * self.type_uncertainty.unsqueeze(0)).sum(dim=1, keepdim=True)

        return p, total_uncertainty



# class ImprovedUncertaintyGCN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_types, type_uncertainty, dropout=0.2):
#         super().__init__()
#         # 更复杂的GCN架构
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
#         # 自注意力机制
#         self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
#         # MLP解码器
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, n_types)
#         )
        
#         # 不确定性嵌入
#         self.type_uncertainty = nn.Parameter(torch.tensor(type_uncertainty, dtype=torch.float32), 
#                                             requires_grad=False)
#         self.uncertainty_embed = nn.Linear(n_types, hidden_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x, edge_index):
#         # 基本GCN处理
#         h1 = F.relu(self.conv1(x, edge_index))
#         h1 = self.dropout(h1)
        
#         h2 = F.relu(self.conv2(h1, edge_index))
#         h2 = self.dropout(h2)
        
#         h3 = F.relu(self.conv3(h2, edge_index))
        
#         # 粗略预测初始细胞类型比例
#         logits_initial = self.mlp(h3)
#         p_initial = F.softmax(logits_initial, dim=-1)
        
#         # 基于预测比例和不确定性生成注意力权重
#         uncertainty_weights = p_initial * self.type_uncertainty.unsqueeze(0)
#         uncertainty_signal = self.uncertainty_embed(uncertainty_weights)
        
#         # 自注意力机制融合空间上下文信息
#         h3_reshaped = h3.unsqueeze(0)  # 适应注意力层输入格式
#         attn_output, _ = self.attention(
#             h3_reshaped + uncertainty_signal.unsqueeze(0),
#             h3_reshaped,
#             h3_reshaped
#         )
        
#         # 最终解码
#         h_final = h3 + attn_output.squeeze(0)
#         logits = self.mlp(h_final)
#         p = F.softmax(logits, dim=-1)
        
#         # 计算整体不确定性
#         total_uncertainty = (p * self.type_uncertainty.unsqueeze(0)).sum(dim=1, keepdim=True)
        
#         return p, total_uncertainty

    # def forward(self, x, edge_index):
    #         # --- 第1步：GCN骨干网络提取基础特征 (这部分保持不变) ---
    #         h1 = F.relu(self.conv1(x, edge_index))
    #         h1 = self.dropout(h1)
            
    #         h2 = F.relu(self.conv2(h1, edge_index))
    #         h2 = self.dropout(h2)
            
    #         h3 = F.relu(self.conv3(h2, edge_index))
            
    #         # --- 第2步：直接用GCN提取的特征进行预测 (最关键的修改) ---
    #         # 基于GCN的直接输出进行解码，得到细胞类型比例
    #         logits_initial = self.mlp(h3)
    #         p_initial = F.softmax(logits_initial, dim=-1)
            
    #         # 创建一个假的uncertainty输出，以确保函数返回两个值，避免在notebook中报错
    #         dummy_uncertainty = torch.zeros_like(p_initial[:, :1])
            
    #         # 直接返回这个最基础、最稳健的预测结果
    #         return p_initial, dummy_uncertainty

    #         # -------------------------------------------------------------------------
    #         # | 以下您设计的、导致了问题的复杂注意力机制，已全部被暂时绕过。        |
    #         # | 您原来的代码保留在下方，仅供参考，它们在当前版本中不会被执行。        |
    #         # -------------------------------------------------------------------------
            
    #         # # 粗略预测初始细胞类型比例
    #         # logits_initial = self.mlp(h3)
    #         # p_initial = F.softmax(logits_initial, dim=-1)
    #         #     h3_reshaped,
    #         #     h3_reshaped
    #         # )
    #         # 
    #         # # 最终解码
    #         # h_final = h3 + attn_output.squeeze(0)
    #         # logits = self.mlp(h_final)
    #         # p = F.softmax(logits, dim=-1)
    #         # 
    #         # # 计算整体不确定性
    #         # total_uncertainty = (p * self.type_uncertainty.unsqueeze(0)).sum(dim=1, keepdim=True)
    #         # 
    #         # return p, total_uncertainty 
    #         # # 基于预测比例和不确定性生成注意力权重
    #         # uncertainty_weights = p_initial * self.type_uncertainty.unsqueeze(0)
    #         # uncertainty_signal = self.uncertainty_embed(uncertainty_weights)
    #         # 
    #         # # 自注意力机制融合空间上下文信息
    #         # h3_reshaped = h3.unsqueeze(0)  # 适应注意力层输入格式
    #         # attn_output, _ = self.attention(
    #         #     h3_reshaped + uncertainty_signal.unsqueeze(0),
    #         #


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
    # sparsity_loss = ((p + 1e-8) * torch.log(p + 1e-8)).sum(dim=1).mean()
    
    # 4. 基于不确定性的自适应KL惩罚
    # 高不确定性类型应该有更宽松的分布约束
    unc_scaled = F.softmax(-torch.tensor(type_uncertainty, device=device), dim=0)
    prior = unc_scaled.repeat(p.shape[0], 1)  # 展开到batch维度
    kl_loss = F.kl_div((p + 1e-8).log(), prior, reduction="batchmean")
    
    # 总损失
    total_loss = recon_loss + alpha * smooth_loss + beta * sparsity_loss + gamma * kl_loss
    
    return total_loss, recon_loss, smooth_loss, sparsity_loss, kl_loss