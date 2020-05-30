import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # mx1x50x50
        self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 0),
                nn.ReLU(),
                # mx16x24x24
                nn.Conv2d(16, 4, 4, 2, 0)
                # mx4x11x11
            )

        self.mean = nn.Linear(4*11*11, 800)
        self.std = nn.Linear(4*11*11, 800)

        self.convertor = nn.Linear(800, 4*11*11)

        self.decoder = nn.Sequential(
                # mx4x11x11
                nn.ConvTranspose2d(4, 16, 4, 2, 0),
                nn.ReLU(),
                # mx16x24x24
                nn.ConvTranspose2d(16, 1, 4, 2, 0)
                # mx1x50x50
            )

    def encode(self, data):
        data = F.relu(self.encoder(data)).view(-1, 4*11*11)
        return self.mean(data), self.std(data)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        z = torch.randn_like(std)
        return mean + z*std

    def decode(self, data):
        data = F.relu(self.convertor(data)).view(-1, 4, 11, 11)
        return torch.sigmoid(self.decoder(data))
        
    def forward(self, x):
        mean, logvar = self.encode(x.float())
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar
