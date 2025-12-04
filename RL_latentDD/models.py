# models.py

import torch
import torch.nn as nn


class ConvAEClassifier(nn.Module):
    """
    Convolutional autoencoder with a classifier head.
    """
    def __init__(self, latent_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 4x4
            nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(128 * 4 * 4, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32x32
            nn.Sigmoid(),  # [0,1]
        )

        # Classifier from latent
        self.classifier_head = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)
        h = h.view(h.size(0), 128, 4, 4)
        x_recon = self.decoder_cnn(h)
        return x_recon

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        logits = self.classify(z)
        return x_recon, logits, z


class LatentLinearClassifier(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z: torch.Tensor):
        return self.fc(z)


class LatentPrototypeModel(nn.Module):
    """
    Synthetic latent prototypes + linear classifier.

    - protos: [num_classes, m_per_class, latent_dim]
    - classifier: linear layer mapping latent_dim -> num_classes
    """
    def __init__(self, num_classes: int, latent_dim: int, m_per_class: int):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.m_per_class = m_per_class

        protos = torch.zeros(num_classes, m_per_class, latent_dim)
        self.protos = nn.Parameter(protos)

        self.classifier = nn.Linear(latent_dim, num_classes)

    def get_synthetic_dataset(self):
        z = self.protos.view(-1, self.latent_dim)
        labels = torch.arange(self.num_classes, device=z.device).repeat_interleave(self.m_per_class)
        return z, labels

    def forward(self, z: torch.Tensor):
        return self.classifier(z)
