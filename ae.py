from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 6, hidden: int = 16, latent_ch: int = 8, act_fn: Optional[nn.Module] = None):
        super().__init__()
        if act_fn is None:
            act_fn = nn.ReLU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            act_fn,
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            act_fn,
            nn.Conv1d(hidden, latent_ch, kernel_size=3, padding=1, stride=2),  # 20 -> 10
            act_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 6, hidden: int = 16, latent_ch: int = 8, act_fn: Optional[nn.Module] = None):
        super().__init__()
        if act_fn is None:
            act_fn = nn.ReLU()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(latent_ch, hidden, kernel_size=3, padding=1, stride=2, output_padding=1),  # 10 -> 20
            act_fn,
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            act_fn,
            nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


@dataclass
class AETrainHistory:
    training_loss_per_epoch: list[float]
    validation_loss_per_epoch: list[float]


class ConvolutionalAutoencoder:
    def __init__(self, autoencoder: Autoencoder, device: torch.device, lr: float = 1e-3):
        self.network = autoencoder.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def train_model(
        self,
        loss_function: nn.Module,
        epochs: int,
        batch_size: int,
        training_set: TensorDataset,
        validation_set: Optional[TensorDataset] = None,
    ) -> AETrainHistory:
        self.network.apply(self._init_weights)

        train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        val_loader = None
        if validation_set is not None:
            val_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

        train_hist: list[float] = []
        val_hist: list[float] = []

        for epoch in range(epochs):
            self.network.train()
            train_losses = []
            for (xb,) in train_loader:
                xb = xb.to(self.device)
                self.optimizer.zero_grad()
                x_hat, _ = self.network(xb)
                loss = loss_function(x_hat, xb)
                loss.backward()
                self.optimizer.step()
                train_losses.append(float(loss.item()))

            mean_train = float(np.mean(train_losses)) if train_losses else 0.0
            train_hist.append(mean_train)

            if val_loader is not None:
                self.network.eval()
                val_losses = []
                with torch.no_grad():
                    for (xb,) in val_loader:
                        xb = xb.to(self.device)
                        x_hat, _ = self.network(xb)
                        val_losses.append(float(loss_function(x_hat, xb).item()))
                mean_val = float(np.mean(val_losses)) if val_losses else 0.0
                val_hist.append(mean_val)
                print(f"AE Epoch {epoch+1}/{epochs} | train={mean_train:.4f} | val={mean_val:.4f}")
            else:
                print(f"AE Epoch {epoch+1}/{epochs} | train={mean_train:.4f}")

        return AETrainHistory(train_hist, val_hist)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            return self.network.encoder(x.to(self.device))

    def autoencode(self, x: torch.Tensor) -> torch.Tensor:
        self.network.eval()
        with torch.no_grad():
            x_hat, _ = self.network(x.to(self.device))
        return x_hat
