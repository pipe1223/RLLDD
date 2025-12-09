# models.py

import torch
import torch.nn as nn
import torchvision


#class ConvAEClassifier(nn.Module):
#    """
#    Convolutional autoencoder with a classifier head.
#    """
#    def __init__(self, latent_dim: int = 128, num_classes: int = 10):
#        super().__init__()
#        self.latent_dim = latent_dim
#        self.num_classes = num_classes
#
#        # Encoder
#        self.encoder_cnn = nn.Sequential(
#            nn.Conv2d(3, 32, 4, 2, 1),  # 16x16
#            nn.ReLU(inplace=True),
#            nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
#            nn.ReLU(inplace=True),
#            nn.Conv2d(64, 128, 4, 2, 1),  # 4x4
#            nn.ReLU(inplace=True),
#        )
#        self.enc_fc = nn.Linear(128 * 4 * 4, latent_dim)
#
#        # Decoder
#        self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
#        self.decoder_cnn = nn.Sequential(
#            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
#            nn.ReLU(inplace=True),
#            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16
#            nn.ReLU(inplace=True),
#            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32x32
#            nn.Sigmoid(),  # [0,1]
#        )
#
#        # Classifier from latent
#        self.classifier_head = nn.Linear(latent_dim, num_classes)
#
#    def encode(self, x: torch.Tensor) -> torch.Tensor:
#        h = self.encoder_cnn(x)
#        h = h.view(h.size(0), -1)
#        z = self.enc_fc(h)
#        return z
#
#    def decode(self, z: torch.Tensor) -> torch.Tensor:
#        h = self.dec_fc(z)
#        h = h.view(h.size(0), 128, 4, 4)
#        x_recon = self.decoder_cnn(h)
#        return x_recon
#
#    def classify(self, z: torch.Tensor) -> torch.Tensor:
#        return self.classifier_head(z)
#
#    def forward(self, x: torch.Tensor):
#        z = self.encode(x)
#        x_recon = self.decode(z)
#        logits = self.classify(z)
#        return x_recon, logits, z

class ConvAEClassifier(nn.Module):

    def __init__(self, latent_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # -------- Encoder: 3 blocks exactly as you specified --------
        self.encoder_cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),  # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),  # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),  # 8x8 -> 4x4
        )

        # Fully-connected layer to get latent representation for AE
        self.enc_fc = nn.Linear(128 * 4 * 4, latent_dim)

        # -------- Decoder (same structure as before) --------
        self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 4x4 -> 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),   # 16x16 -> 32x32
            nn.Sigmoid(),  # outputs in [0,1]
        )

        # -------- Classifier head: matches your spec --------
        # Flatten (128 * 4 * 4) ? num_classes
        self.classifier_head = nn.Linear(128 * 4 * 4, num_classes)

    # Optional: keep a helper to get feature maps before flattening
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the conv feature maps of shape (B, 128, 4, 4).
        """
        h = self.encoder_cnn(x)
        return h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns latent vector z of shape (B, latent_dim).
        This keeps the same API you had before.
        """
        h = self.encoder_cnn(x)
        h_flat = h.view(h.size(0), -1)     # (B, 128*4*4)
        z = self.enc_fc(h_flat)            # (B, latent_dim)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z)                 # (B, 128*4*4)
        h = h.view(h.size(0), 128, 4, 4)   # (B, 128, 4, 4)
        x_recon = self.decoder_cnn(h)      # (B, 3, 32, 32)
        return x_recon

    def classify_from_features(self, h: torch.Tensor) -> torch.Tensor:
        """
        Classify directly from conv features (B, 128, 4, 4).
        """
        h_flat = h.view(h.size(0), -1)     # (B, 128*4*4)
        logits = self.classifier_head(h_flat)
        return logits

    def forward(self, x: torch.Tensor):
        # 1) Encoder conv blocks
        h = self.encoder_cnn(x)            # (B, 128, 4, 4)
        h_flat = h.view(h.size(0), -1)     # (B, 128*4*4)

        # 2) Latent for autoencoder
        z = self.enc_fc(h_flat)            # (B, latent_dim)

        # 3) Reconstruction from latent
        x_recon = self.decode(z)

        # 4) Classification from flattened conv features
        logits = self.classifier_head(h_flat)

        return x_recon, logits, z


class ResNetAEClassifier(nn.Module):
    """Autoencoder that leverages a ResNet18 encoder with a lightweight decoder."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 10, img_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        resnet = torchvision.models.resnet18(weights=None)
        self.encoder_cnn = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool + fc
        self.enc_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.enc_fc = nn.Linear(512, latent_dim)

        start = max(4, img_size // 8)  # works for 32 or 224
        self.dec_fc = nn.Linear(latent_dim, 512 * start * start)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.classifier_head = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_cnn(x)
        h = self.enc_pool(h)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        start = max(4, self.img_size // 8)
        h = self.dec_fc(z)
        h = h.view(h.size(0), 512, start, start)
        return self.decoder(h)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        logits = self.classify(z)
        return x_recon, logits, z


class ViTAEClassifier(nn.Module):
    """Vision Transformer autoencoder using the CLS token for the latent."""

    def __init__(self, latent_dim: int = 256, num_classes: int = 10, img_size: int = 224):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        vit = torchvision.models.vit_b_16(weights=None, image_size=img_size)
        self.conv_proj = vit.conv_proj
        self.encoder = vit.encoder
        self.class_token = nn.Parameter(vit.class_token.detach().clone())
        self.pos_embedding = nn.Parameter(vit.encoder.pos_embedding.detach().clone())
        self.enc_fc = nn.Linear(vit.hidden_dim, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 3 * img_size * img_size)
        self.classifier_head = nn.Linear(latent_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.class_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        cls = x[:, 0]
        return self.enc_fc(cls)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        recon = self.dec_fc(z)
        recon = recon.view(z.size(0), 3, self.img_size, self.img_size)
        return recon.sigmoid()

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
