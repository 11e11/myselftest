import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], latent_dim=32, n_conditions=None):
        super().__init__()
        self.n_conditions = n_conditions
        
        # 编码器
        encoder_layers = []
        # 如果有条件信息，输入维度需要加上条件的one-hot编码维度
        encoder_input_dim = input_dim + (n_conditions if n_conditions else 0)
        last_dim = encoder_input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(nn.LeakyReLU())
            last_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_layers = []
        # 解码器输入包括潜变量和条件信息
        decoder_input_dim = latent_dim + (n_conditions if n_conditions else 0)
        last_dim = decoder_input_dim
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.LayerNorm(h_dim))
            decoder_layers.append(nn.LeakyReLU())
            last_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder_mu = nn.Linear(hidden_dims[0], input_dim)
        self.decoder_theta = nn.Linear(hidden_dims[0], input_dim)
        self.decoder_pi = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x, conditions=None):
        if self.n_conditions and conditions is not None:
            # 将条件转换为one-hot编码
            if conditions.dim() == 1:
                conditions_onehot = F.one_hot(conditions, num_classes=self.n_conditions).float()
            else:
                conditions_onehot = conditions.float()
            # 将基因表达和条件信息拼接
            x = torch.cat([x, conditions_onehot], dim=1)
        
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, conditions=None):
        if self.n_conditions and conditions is not None:
            # 将条件转换为one-hot编码
            if conditions.dim() == 1:
                conditions_onehot = F.one_hot(conditions, num_classes=self.n_conditions).float()
            else:
                conditions_onehot = conditions.float()
            # 将潜变量和条件信息拼接
            z = torch.cat([z, conditions_onehot], dim=1)
        
        h = self.decoder(z)
        dec_mu = torch.exp(self.decoder_mu(h))           # 均值参数，正数
        dec_theta = F.softplus(self.decoder_theta(h)) + 1e-4  # 过分散参数，正数
        dec_pi = torch.sigmoid(self.decoder_pi(h))       # dropout概率
        return dec_mu, dec_theta, dec_pi

    def forward(self, x, conditions=None):
        mu, logvar = self.encode(x, conditions)
        z = self.reparameterize(mu, logvar)
        dec_mu, dec_theta, dec_pi = self.decode(z, conditions)
        return dec_mu, dec_theta, dec_pi, mu, logvar

def zinb_loss(x, mu, theta, pi, eps=1e-8):
    # x, mu, theta, pi: [batch, genes]
    t1 = torch.lgamma(theta + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + theta + eps)
    t2 = (theta + x) * torch.log(1.0 + (mu / (theta + eps))) + (x * (torch.log(theta + eps) - torch.log(mu + eps)))
    nb_case = t1 + t2
    nb_case = torch.exp(torch.log(1.0 - pi + eps) - nb_case)
    zero_case = torch.exp(torch.log(pi + ((1.0 - pi) * torch.pow(theta / (theta + mu + eps), theta))) + eps)
    result = torch.where(x < 1e-8, -torch.log(zero_case + eps), -torch.log(nb_case + eps))
    return result.mean()

def vae_loss(x, dec_mu, dec_theta, dec_pi, mu, logvar, beta=0.005):
    zinb = zinb_loss(x, dec_mu, dec_theta, dec_pi)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return zinb + beta * kl, zinb, kl

def compute_uncertainty(model, data_loader, device="cpu", n_types=None, fill_value=0.0):
    """
    计算每个细胞类型的重构误差、KL散度和方差
    """
    model.eval()
    RE, KL, VAR = [], [], []
    labels_all = []
    with torch.no_grad():
        for x_batch, labels in data_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            # 传入条件信息
            dec_mu, dec_theta, dec_pi, mu, logvar = model(x_batch, labels)
            # 用dec_mu作为重构（或可用ZINB的均值/采样）
            re = torch.mean((x_batch - dec_mu) ** 2, dim=1).cpu().numpy()
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).cpu().numpy()
            var = torch.var(dec_mu, dim=1).cpu().numpy()
            RE.append(re)
            KL.append(kl)
            VAR.append(var)
            labels_all.append(labels.cpu().numpy())
    RE = np.concatenate(RE)
    KL = np.concatenate(KL)
    VAR = np.concatenate(VAR)
    labels_all = np.concatenate(labels_all)
    # 按细胞类型统计
    if n_types is not None:
        RE_c = np.full(n_types, fill_value, dtype=np.float32)
        KL_c = np.full(n_types, fill_value, dtype=np.float32)
        VAR_c = np.full(n_types, fill_value, dtype=np.float32)
        for i in range(n_types):
            mask = labels_all == i
            if np.any(mask):
                RE_c[i] = np.median(RE[mask])
                KL_c[i] = np.median(KL[mask])
                VAR_c[i] = np.median(VAR[mask])
        return RE, KL, VAR, RE_c, KL_c, VAR_c
    else:
        return RE, KL, VAR

def compute_enhanced_uncertainty(model, data_loader, device="cpu", n_types=None):
    """
    增强的不确定性计算: 融合多种指标并使用稳健归一化
    """
    model.eval()
    RE_list, KL_list, VAR_list = [], [], []
    y_true = []
    
    with torch.no_grad():
        for x_batch, labels in data_loader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            # 传入条件信息
            dec_mu, dec_theta, dec_pi, mu, logvar = model(x_batch, labels)
            # 这里假设模型输出格式，如果不匹配请调整
            x_hat = dec_mu  # 使用解码均值作为重构
            
            # 计算重建误差
            re = torch.mean((x_batch - x_hat) ** 2, dim=1).cpu().numpy()
            
            # KL散度
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).cpu().numpy()
            
            # 潜变量方差
            var = torch.exp(logvar).mean(dim=1).cpu().numpy()
            
            RE_list.extend(re)
            KL_list.extend(kl)
            VAR_list.extend(var)
            y_true.extend(labels.cpu().numpy())
    
    RE = np.asarray(RE_list, dtype=np.float32)
    KL = np.asarray(KL_list, dtype=np.float32)
    VAR = np.asarray(VAR_list, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)
    
    if n_types is None:
        n_types = int(y_true.max()) + 1 if y_true.size > 0 else 0
    
    # 计算每种细胞类型的中位数作为代表值(更稳健)
    RE_c = np.zeros(n_types, dtype=np.float32)
    KL_c = np.zeros(n_types, dtype=np.float32)
    VAR_c = np.zeros(n_types, dtype=np.float32)
    
    for c in range(n_types):
        mask = (y_true == c)
        if mask.sum() > 0:
            RE_c[c] = np.median(RE[mask])
            KL_c[c] = np.median(KL[mask])
            VAR_c[c] = np.median(VAR[mask])
    
    # 鲁棒归一化: 使用分位数而非min/max
    def robust_normalize(x):
        q_low, q_high = np.percentile(x, [5, 95])
        return np.clip((x - q_low) / (q_high - q_low + 1e-8), 0, 1)
    
    RE_c_norm = robust_normalize(RE_c)
    KL_c_norm = robust_normalize(KL_c)
    VAR_c_norm = robust_normalize(VAR_c)
    
    # 融合不确定性指标 (加权平均)
    combined_unc = 0.5 * RE_c_norm + 0.3 * KL_c_norm + 0.2 * VAR_c_norm
    
    return RE, KL, VAR, RE_c, KL_c, VAR_c, combined_unc