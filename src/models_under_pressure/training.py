# %%
#!%load_ext autoreload
#!%autoreload 2

from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models_under_pressure.config import TrainingConfig


class MLPAutoencoder(nn.Module):
    def __init__(self, config: TrainingConfig) -> None:
        super(MLPAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, config.input_dim),
            nn.Sigmoid(),  # Since MNIST pixels are in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(config: TrainingConfig) -> Tuple[MLPAutoencoder, List[float]]:
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        config=config.__dict__,
    )

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root=str(config.data_dir), train=True, transform=transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MLPAutoencoder(config)
    wandb.watch(model)  # Log model gradients and parameters

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    train_losses = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        data_BWH = None
        for batch_idx, batch in enumerate(train_loader):
            # Forward pass
            data_BWH = batch[0]
            output_BL = model(data_BWH)
            loss = criterion(output_BL, data_BWH.view(data_BWH.shape[0], -1))

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % config.print_every == 0:
                print(
                    f"Epoch: {epoch + 1}/{config.epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}"
                )
                # Log batch metrics
                wandb.log({"batch_loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}/{config.epochs}, Average Loss: {avg_loss:.6f}")

        # Log sample reconstructions periodically
        if (epoch + 1) % 2 == 0:  # Every 2 epochs
            assert data_BWH is not None
            log_reconstruction_images(model, data_BWH[:8], epoch)

    wandb.finish()
    return model, train_losses


def log_reconstruction_images(
    model: MLPAutoencoder, data: torch.Tensor, epoch: int
) -> None:
    """Log original and reconstructed images to wandb."""
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)

    # Create pairs of original and reconstructed images
    images = []
    for i in range(min(4, len(data))):  # Show 4 pairs
        # Combine original and reconstruction into one image
        combined = torch.cat(
            [data[i].cpu().squeeze(), reconstructions[i].cpu().view(28, 28)],
            dim=1,  # Concatenate horizontally
        )
        images.append(
            wandb.Image(
                combined.numpy(),
                caption=f"Pair {i}: Original (left) vs Reconstructed (right)",
            )
        )

    wandb.log({"reconstructions": images})
    model.train()


def visualize_reconstruction(model: MLPAutoencoder, config: TrainingConfig) -> None:
    # Load a batch of test data
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root=str(config.data_dir), train=False, transform=transform, download=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.vis_batch_size, shuffle=True
    )
    data, _ = next(iter(test_loader))

    # Get reconstructions
    model.eval()
    with torch.no_grad():
        reconstructions = model(data)

    # Plot original and reconstructed images side by side
    fig, axes = plt.subplots(4, 2, figsize=(8, 12))
    plt.suptitle("Original vs Reconstructed Images")

    for i in range(4):  # Show 4 pairs of images
        # Original image on the left
        axes[i, 0].imshow(data[i].cpu().squeeze(), cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Original")

        # Reconstructed image on the right
        axes[i, 1].imshow(reconstructions[i].cpu().view(28, 28), cmap="gray")
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Reconstructed")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Create config
    config = TrainingConfig(
        epochs=6,
        wandb_project="lasr-demo",
    )

    # Train the model
    model, losses = train_autoencoder(config)
    # %%

    # Visualize some reconstructions
    visualize_reconstruction(model, config)
    visualize_reconstruction(model, config)
