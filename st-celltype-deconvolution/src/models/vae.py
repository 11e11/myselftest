from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Assuming input is normalized between 0 and 1
        )

    def encode(self, x):
        encoded = self.encoder(x)
        mean, log_var = encoded.chunk(2, dim=-1)
        return mean, log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z), mean, log_var

    def loss_function(self, recon_x, x, mean, log_var):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return BCE + KLD

    def train_model(self, data_loader, optimizer, num_epochs):
        self.train()
        for epoch in range(num_epochs):
            for batch in data_loader:
                x = batch[0]  # 取出表达数据
                optimizer.zero_grad()
                recon_x, mean, log_var = self.forward(x)
                loss = self.loss_function(recon_x, x, mean, log_var)
                loss.backward()
                optimizer.step()
            print(f'epoch {epoch}, loss {loss.item()}')