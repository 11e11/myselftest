import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, latent_dim=16):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc2(z))
        return F.softplus(self.fc3(h))  # 用 ReLU 保证非负

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# def vae_loss(x, x_hat, mu, logvar, beta=0.001):
#     # MSE 重构误差
#     recon_loss = F.mse_loss(x_hat, x, reduction="mean")
#     # KL 散度
#     kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return recon_loss + beta*kl_loss, recon_loss, kl_loss

def vae_loss(x, x_hat, mu, logvar, beta=0.005):
    # 替换MSE为负二项分布或泊松分布损失
    # 方案1: 使用负二项分布
    from torch.distributions import NegativeBinomial
    theta = torch.ones_like(x_hat)  # 可学习参数
    nb_dist = NegativeBinomial(total_count=theta, probs=x_hat/(x_hat+theta))
    recon_loss = -nb_dist.log_prob(x).mean()
    
    # 方案2: 泊松损失(更简单)
    # recon_loss = F.poisson_nll_loss(x_hat, x, log_input=False, reduction="mean")
    
    # KL散度权重调大一些
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta*kl_loss, recon_loss, kl_loss

# def compute_uncertainty(model, data_loader, device="cpu"):
#     """
#     给定训练好的 VAE 和 scRNA 单细胞数据，计算每个细胞的重构误差
#     返回: RE (n_cells,), RE_c (n_types,)
#     """
#     model.eval()
#     RE = []
#     y_true = []
#     with torch.no_grad():
#         for x_batch, labels in data_loader:
#             x_batch = x_batch.to(device)
#             x_hat, _, _ = model(x_batch)
#             re = torch.mean((x_batch - x_hat) ** 2, dim=1).cpu().numpy()
#             RE.extend(re)
#             y_true.extend(labels.numpy())

#     RE = np.array(RE)
#     y_true = np.array(y_true)
#     RE_c = np.array([np.median(RE[y_true == c]) for c in np.unique(y_true)])
#     return RE, RE_c
# def compute_uncertainty(model, data_loader, device="cpu"):
#     """
#     增强版不确定性计算：结合重构误差、KL散度和潜变量方差
    
#     Args:
#         model: 训练好的VAE模型
#         data_loader: 包含(x, labels)的数据加载器
#         device: 计算设备
        
#     Returns:
#         RE: 每个细胞的重构误差
#         KL: 每个细胞的KL散度
#         VAR: 每个细胞的潜变量方差
#         RE_c: 每种细胞类型的重构误差中位数
#         KL_c: 每种细胞类型的KL散度中位数
#         VAR_c: 每种细胞类型的潜变量方差中位数
#     """
#     model.eval()
#     RE, KL, VAR = [], [], []
#     y_true = []
    
#     with torch.no_grad():
#         for x_batch, labels in data_loader:
#             x_batch = x_batch.to(device)
#             x_hat, mu, logvar = model(x_batch)
            
#             # 重构误差 (每个细胞)
#             re = torch.mean((x_batch - x_hat) ** 2, dim=1).cpu().numpy()
            
#             # KL散度 (每个细胞)
#             kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).cpu().numpy()
            
#             # 潜变量方差 (每个细胞)
#             var = torch.exp(logvar).mean(dim=1).cpu().numpy()
            
#             RE.extend(re)
#             KL.extend(kl)
#             VAR.extend(var)
#             y_true.extend(labels.numpy())

#     # 转换为numpy数组
#     RE = np.array(RE)
#     KL = np.array(KL)
#     VAR = np.array(VAR)
#     y_true = np.array(y_true)
    
#     # 计算每种细胞类型的中位数
#     unique_types = np.unique(y_true)
#     RE_c = np.array([np.median(RE[y_true == c]) for c in unique_types])
#     KL_c = np.array([np.median(KL[y_true == c]) for c in unique_types])
#     VAR_c = np.array([np.median(VAR[y_true == c]) for c in unique_types])
    
#     # 组合不确定性指标 (可选)
#     # combined_c = (RE_c - RE_c.min()) / (RE_c.max() - RE_c.min()) + 
#     #              (KL_c - KL_c.min()) / (KL_c.max() - KL_c.min())
    
#     return RE, KL, VAR, RE_c, KL_c, VAR_c

def compute_uncertainty(model, data_loader, device="cpu", n_types=None, fill_value=0.0):
    """
    增强版不确定性计算，返回按 type 索引 0..n_types-1 的 per-type 指标（保证长度为 n_types，且无 NaN/inf）。
    参数:
      n_types: 总的细胞类型数（若 None，则从 labels 中推断最大 label+1）
      fill_value: 若某类型无样本，用该值填充（默认 0.0）
    返回:
      RE, KL, VAR, RE_c, KL_c, VAR_c  (numpy arrays, dtype=float32)
    """
    model.eval()
    RE_list, KL_list, VAR_list = [], [], []
    y_true = []

    with torch.no_grad():
        for x_batch, labels in data_loader:
            x_batch = x_batch.to(device)
            x_hat, mu, logvar = model(x_batch)

            re = torch.mean((x_batch - x_hat) ** 2, dim=1).cpu().numpy()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).cpu().numpy()
            var = torch.exp(logvar).mean(dim=1).cpu().numpy()

            RE_list.extend(re)
            KL_list.extend(kl)
            VAR_list.extend(var)
            y_true.extend(labels.numpy())

    RE = np.asarray(RE_list, dtype=np.float32)
    KL = np.asarray(KL_list, dtype=np.float32)
    VAR = np.asarray(VAR_list, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)

    if n_types is None:
        n_types = int(y_true.max()) + 1 if y_true.size > 0 else 0

    RE_c = np.full(n_types, np.nan, dtype=np.float32)
    KL_c = np.full(n_types, np.nan, dtype=np.float32)
    VAR_c = np.full(n_types, np.nan, dtype=np.float32)

    for c in range(n_types):
        mask = (y_true == c)
        if mask.sum() > 0:
            RE_c[c] = np.median(RE[mask])
            KL_c[c] = np.median(KL[mask])
            VAR_c[c] = np.median(VAR[mask])

    # 若全是 NaN，使用 fill_value；否则用有效元素的中位数/0 填充 NaN、并把 inf 转为有限值
    def sanitize(arr):
        if np.all(np.isnan(arr)):
            arr[:] = fill_value
        else:
            med = np.nanmedian(arr)
            arr = np.nan_to_num(arr, nan=med, posinf=np.finfo(np.float32).max/100.0, neginf=-np.finfo(np.float32).max/100.0)
        return arr.astype(np.float32)

    RE_c = sanitize(RE_c)
    KL_c = sanitize(KL_c)
    VAR_c = sanitize(VAR_c)

    return RE, KL, VAR, RE_c, KL_c, VAR_c