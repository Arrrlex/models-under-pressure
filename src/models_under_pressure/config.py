from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    # Model parameters
    input_dim: int = 784  # 28x28 MNIST images
    hidden_dim: int = 128

    # Training parameters
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3

    # Data parameters
    data_dir: Path = Path("./.data")

    # Visualization parameters
    vis_batch_size: int = 8
    print_every: int = 100  # Print loss every N batches

    # Weights & Biases settings
    wandb_project: str = "lasr-demo"
    wandb_entity: str = "josephmiller101"  # Your username or organization
