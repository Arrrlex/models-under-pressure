import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def visualize_activations_pca(
    pickle_file_path: Path,
    coefs_path: Path,
    dataset_name: str,
    use_mean_activations: bool = True,
):
    """
    Visualize activations using PCA and plot probe direction.

    Args:
        pickle_file_path: Path to joblib pickle file containing activations
        use_mean_activations: If True, use mean_activations, else use masked_activations
    """
    # Load data from pickle file
    data = joblib.load(pickle_file_path)

    data_entry = [entry for entry in data if entry["dataset_name"] == dataset_name][0]

    # Get activations based on flag
    if use_mean_activations:
        activations = data_entry["mean_activations"]
    else:
        activations = data_entry["masked_activations"]

    # Convert activations to numpy array and flatten if needed
    X = np.array([act[0] for act in activations])
    n_samples = len(X) // 2  # Assuming equal split between classes

    # Step 2: PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Load probe weights from json file
    with open(coefs_path, "r") as f:
        probe_weights = np.array(json.load(f)["coefs"])

    print(probe_weights)

    # Project probe weights to PCA space
    probe_weights_pca = pca.transform(probe_weights.reshape(1, -1))[0]

    # Step 4: Visualization
    plt.figure(figsize=(10, 7))
    plt.scatter(
        X_pca[:n_samples, 0], X_pca[:n_samples, 1], color="orange", label="Green Inputs"
    )
    plt.scatter(
        X_pca[n_samples:, 0], X_pca[n_samples:, 1], color="blue", label="Blue Inputs"
    )

    # Plot probe vector direction (scaled for visualization)
    origin = np.mean(X_pca, axis=0)
    plt.quiver(
        *origin,
        probe_weights_pca[0],
        probe_weights_pca[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color="red",
        width=0.005,
        label="Probe Direction",
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA of Activations with Probe Direction")
    plt.legend()
    plt.grid()
    plt.savefig(
        Path("data/results/evaluate_probes/")
        / f"{Path(pickle_file_path).stem.replace('_activations', '')}.png"
    )


if __name__ == "__main__":
    path = Path("data/results/evaluate_probes/results_urja_activations.pkl")
    coefs_path = Path("data/results/evaluate_probes/results_urja_coefs.json")
    visualize_activations_pca(path, coefs_path, "manual", use_mean_activations=True)
