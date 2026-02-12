import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3_mu = nn.Linear(32, latent_dim)
        self.fc3_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.fc4 = nn.Linear(latent_dim, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3_mu(h2), self.fc3_logvar(h2)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc4(z))
        h4 = F.relu(self.fc5(h3))
        return torch.sigmoid(self.fc6(h4)) # Input is normalized to [0, 1]
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum') # MSE for continuous data
    # KLD = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_model(model, train_data, epochs=20, batch_size=64, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Convert sparse matrix to dense if necessary (OneHotEncoder might produce sparse)
    if isinstance(train_data, np.ndarray):
        tensor_x = torch.Tensor(train_data)
    else:
        tensor_x = torch.Tensor(train_data.toarray())
        
    dataset = TensorDataset(tensor_x)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data,) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch: {epoch+1}, Average loss: {train_loss / len(dataloader.dataset):.4f}')
    
    return model
