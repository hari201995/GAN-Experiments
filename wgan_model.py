import torch.nn as nn


class WGenerator(nn.Module):
    def __init__(self, latent_dim=128, image_size=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, image_size),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class WCritic(nn.Module):
    """Critic (no Sigmoid — outputs raw scores for Wasserstein distance)."""
    def __init__(self, image_size=784):
        super().__init__()
        h1 = max(256, image_size // 4)
        self.net = nn.Sequential(
            nn.Linear(image_size, h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(h1, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.zeros_(m.bias)
