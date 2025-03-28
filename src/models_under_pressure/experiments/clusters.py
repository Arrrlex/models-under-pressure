import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from sklearn.decomposition import PCA


def visualize_activations_pca(
    pickle_file_paths: dict[str, Path],
    data_paths: dict[str, Path],
    coefs_path: Path,
    labels_path: Path,
):
    labels = []
    activations = []
    inputs = []
    for dataset_name, pickle_file_path in pickle_file_paths.items():
        activations_data = h5py.File(pickle_file_path, "r")
        activations_ = activations_data[dataset_name]["mean_activations"][:]  # type: ignore
        activations_data.close()

        with open(labels_path, "r") as f:
            data = [json.loads(line) for line in f]
        data_entry = [entry for entry in data if entry["dataset_name"] == dataset_name][
            0
        ]
        labels_ = data_entry["ground_truth_labels"]

        with open(data_paths[dataset_name], "r") as f:
            data = [json.loads(line) for line in f]
        inputs.extend(
            [
                d["inputs"]
                for i, d in enumerate(data)
                #  if d["language"] == "English"
            ]
        )
        labels.extend(
            [
                label
                for i, label in enumerate(labels_)
                # if (data[i]["language"] == "English")
            ]
        )
        # activations.extend(activations_data[dataset_name]["mean_activations"][:])  # type: ignore

        activations.extend(
            [
                activation
                for i, activation in enumerate(activations_)  # type: ignore
                # if (data[i]["language"] == "English")
            ]
        )
        # data_from_dataset = pd.read_json(EVAL_DATASETS[dataset_name])
        # data_from_dataset = data_from_dataset[
        #     data_from_dataset["language"].isin(input_ids)
        # ]

        # labels_from_dataset = [
        #     (d["ids"], 1 if d["labels"] == "high-stakes" else 0)
        #     for d in data_from_dataset
        #     if d["ids"] in input_ids
        # ]

        # assert sorted(input_ids) == [
        #     d[0] for d in sorted(labels_from_dataset, key=lambda x: x[0])
        # ]
        # sort labels_ list as per input_ids
        # labels_dataset = [
        #     label for _, label in sorted(labels_from_dataset, key=lambda x: x[0])
        # ]
        # labels_here = [
        #     label for label, _ in sorted(zip(labels_, input_ids), key=lambda x: x[1])
        # ]
        # assert labels_here == labels_dataset
        # Extract labels in same order as input_ids to verify consistency

        #

    # row_mean = np.array(activations).mean(axis=1, keepdims=True)
    # row_std = np.array(activations).std(axis=1, keepdims=True)
    # activations_scaled = (np.array(activations) - row_mean) / row_std

    pca = PCA(n_components=4)
    activations_pca = pca.fit_transform(activations)  # type: ignore

    # Calculate variance explained by each component
    explained_variance_ratio = pca.explained_variance_ratio_

    with open(coefs_path, "r") as f:
        probe_weights = np.array(json.load(f)["coefs"])
    # probe_weights_scaled = (probe_weights - np.mean(probe_weights)) / np.std(
    #     probe_weights
    # )

    probe_weights_pca = pca.components_ @ probe_weights

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create a figure with 3D, 2D, and variance plots
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}], [{"colspan": 2}, None]],
        row_heights=[0.7, 0.3],
    )
    pc_x = 1
    pc_y = 2
    pc_z = 3
    # 3D scatter plot (PC1, PC2, PC3)
    for label in np.unique(labels):
        mask = np.array(labels) == label
        fig.add_trace(
            go.Scatter3d(
                x=activations_pca[mask, pc_x],
                y=activations_pca[mask, pc_y],
                z=activations_pca[mask, pc_z],
                mode="markers",
                name=f"Label {label}",
                text=[f"Input: {inp}" for inp in np.array(inputs)[mask]],
                hoverinfo="text+name",
                marker=dict(size=5),
                legendgroup=f"Label {label}",
            ),
            row=1,
            col=1,
        )

    # Add probe vector in 3D
    # origin = np.mean(activations_pca, axis=0)
    # # Scale the vector for better visualization
    # scale_factor = 2.0
    # fig.add_trace(
    #     go.Scatter3d(
    #         x=[origin[pc_x], origin[pc_x] + scale_factor * probe_weights_pca[pc_x]],
    #         y=[origin[pc_y], origin[pc_y] + scale_factor * probe_weights_pca[pc_y]],
    #         z=[origin[pc_z], origin[pc_z] + scale_factor * probe_weights_pca[pc_z]],
    #         mode="lines+markers",
    #         name="Probe Direction (3D)",
    #         line=dict(color="red", width=6),
    #         marker=dict(size=5),
    #     ),
    #     row=1,
    #     col=1,
    # )

    # Keep the original 2D plot (PC2 vs PC3)
    pc2_x = 1
    pc2_y = 2

    # 2D scatter plot
    for label in np.unique(labels):
        mask = np.array(labels) == label
        fig.add_trace(
            go.Scatter(
                x=activations_pca[mask, pc2_x],
                y=activations_pca[mask, pc2_y],
                mode="markers",
                name=f"Label {label}",
                text=[f"Input: {inp}" for inp in np.array(inputs)[mask]],
                hoverinfo="text+name",
                marker=dict(size=8),
                legendgroup=f"Label {label}",
                showlegend=False,  # Already shown in 3D plot
            ),
            row=1,
            col=2,
        )

    # Add probe vector in 2D
    # fig.add_trace(
    #     go.Scatter(
    #         x=[origin[pc_x], origin[pc_x] + probe_weights_pca[pc_x]],
    #         y=[origin[pc_y], origin[pc_y] + probe_weights_pca[pc_y]],
    #         mode="lines+markers",
    #         name="Probe Direction (2D)",
    #         line=dict(color="red", width=3),
    #         marker=dict(size=8),
    #         showlegend=False,
    #     ),
    #     row=1,
    #     col=2,
    # )

    # Update 3D layout
    fig.update_scenes(
        aspectmode="cube",
        xaxis_title=f"PC{pc_x + 1}",
        yaxis_title=f"PC{pc_y + 1}",
        zaxis_title=f"PC{pc_z + 1}",
    )

    # Update 2D layout
    fig.update_xaxes(title_text=f"PC{pc2_x + 1}", row=1, col=2)
    fig.update_yaxes(title_text=f"PC{pc2_y + 1}", row=1, col=2)

    # Add variance explained plot
    # Run PCA with more components to show variance distribution
    pca_variance = PCA(n_components=min(10, len(activations[0])))
    pca_variance.fit(activations)  # type: ignore

    # Create cumulative variance plot
    cumulative_variance = np.cumsum(pca_variance.explained_variance_ratio_)

    fig.add_trace(
        go.Bar(
            x=list(range(1, len(pca_variance.explained_variance_ratio_) + 1)),
            y=pca_variance.explained_variance_ratio_,
            name="Individual Variance",
            marker_color="blue",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(cumulative_variance) + 1)),
            y=cumulative_variance,
            mode="lines+markers",
            name="Cumulative Variance",
            marker=dict(color="red"),
            line=dict(color="red"),
        ),
        row=2,
        col=1,
    )

    # Add a horizontal line at 95% variance explained
    fig.add_shape(
        type="line",
        x0=0.5,
        x1=len(cumulative_variance) + 0.5,
        y0=0.95,
        y1=0.95,
        line=dict(color="green", width=2, dash="dash"),
        row=2,
        col=1,
    )

    # Add annotation for 95% line
    fig.add_annotation(
        x=len(cumulative_variance),
        y=0.95,
        text="95% Variance",
        showarrow=False,
        yshift=10,
        font=dict(color="green"),
        row=2,
        col=1,
    )

    # Update variance plot layout
    fig.update_xaxes(
        title_text="Principal Component",
        tickmode="linear",
        tick0=1,
        dtick=1,
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Variance Explained", range=[0, 1], row=2, col=1)

    # Update overall layout
    fig.update_layout(
        height=1000,  # Increased height to accommodate the new plot
        width=1400,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add annotation showing variance for all components
    variance_text = "<br>".join(
        [
            f"PC{i + 1}: {var:.2%} variance"
            for i, var in enumerate(explained_variance_ratio)
        ]
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text=variance_text,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12),
    )

    # Save to HTML file
    fig.write_html(
        Path("data/results/evaluate_probes/") / "all_datasets_pca_3d_visualization.html"
    )


def visualize_activations_pca_seaborn_3d(
    activations_pca: np.ndarray,
    labels: np.ndarray,
    inputs: Optional[list] = None,
    pc_indices: tuple = (0, 1, 2),
    title: str = "3D PCA Visualization",
    title_size: int = 20,
    axis_label_size: int = 20,
    tick_label_size: int = 15,
    legend_title: str = "Labels",
    legend_title_size: int = 20,
    legend_text_size: int = 18,
    marker_size: int = 100,
    alpha: float = 1,
    rotation: tuple = (20, 30),  # (elevation, azimuth)
    output_path: Optional[Path] = None,
    show_plot: bool = True,
):
    """
    Create a 3D PCA visualization using seaborn and matplotlib.

    Parameters:
    -----------
    activations_pca : numpy.ndarray
        PCA-transformed activations
    labels : list or numpy.ndarray
        Labels for each data point
    inputs : list or None
        Optional text inputs for hover annotations
    pc_indices : tuple
        Indices of principal components to plot (default: first three)
    title : str
        Plot title
    title_size : int
        Font size for title
    axis_label_size : int
        Font size for axis labels
    tick_label_size : int
        Font size for tick labels
    legend_title : str
        Title for the legend
    legend_title_size : int
        Font size for legend title
    legend_text_size : int
        Font size for legend text
    marker_size : int
        Size of scatter plot markers
    alpha : float
        Transparency of markers (0-1)
    rotation : tuple
        (elevation, azimuth) for 3D plot rotation
    output_path : Path or str or None
        Path to save the figure, if None, figure is not saved
    show_plot : bool
        Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the style
    sns.set_style("whitegrid")

    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Extract the principal components to plot
    pc_x, pc_y, pc_z = pc_indices

    # Convert labels to categorical if they're not already
    unique_labels = np.unique(labels)

    # Create a color palette with enough colors
    palette = sns.color_palette("husl", len(unique_labels))

    # Plot each label group
    for i, label in enumerate(unique_labels):
        mask = np.array(labels) == label
        ax.scatter(
            activations_pca[mask, pc_x],
            activations_pca[mask, pc_y],
            activations_pca[mask, pc_z],
            s=marker_size,
            alpha=alpha,
            c=[palette[i]],
            label="High-stakes" if label == 1 else "Low-stakes",
        )

    # Set labels and title
    ax.set_xlabel(f"PC{pc_x + 1}", fontsize=axis_label_size)
    ax.set_ylabel(f"PC{pc_y + 1}", fontsize=axis_label_size)
    ax.set_zlabel(f"PC{pc_z + 1}", fontsize=axis_label_size)

    # Set tick label size
    ax.tick_params(axis="x", labelsize=tick_label_size)
    ax.tick_params(axis="y", labelsize=tick_label_size)
    ax.tick_params(axis="z", labelsize=tick_label_size)

    # Set the viewing angle
    ax.view_init(elev=rotation[0], azim=rotation[1])

    # Add legend
    legend = ax.legend(
        title=legend_title,
        fontsize=legend_text_size,
        markerscale=0.7,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        loc="upper right",
    )
    legend.get_title().set_fontsize(legend_title_size)

    # Adjust layout
    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.3)

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig, ax


def visualize_activations_pca_seaborn_2d(
    activations_pca: np.ndarray,
    labels: np.ndarray,
    inputs: Optional[list] = None,
    pc_indices: tuple = (0, 1),
    title: str = "2D PCA Visualization",
    title_size: int = 20,
    axis_label_size: int = 20,
    tick_label_size: int = 15,
    legend_title: str = "Labels",
    legend_title_size: int = 20,
    legend_text_size: int = 18,
    marker_size: int = 100,
    alpha: float = 1,
    output_path: Optional[Path] = None,
    show_plot: bool = True,
):
    """
    Create a 2D PCA visualization using seaborn.

    Parameters:
    -----------
    activations_pca : numpy.ndarray
        PCA-transformed activations
    labels : list or numpy.ndarray
        Labels for each data point
    inputs : list or None
        Optional text inputs for hover annotations
    pc_indices : tuple
        Indices of principal components to plot (default: first two)
    title : str
        Plot title
    title_size : int
        Font size for title
    axis_label_size : int
        Font size for axis labels
    tick_label_size : int
        Font size for tick labels
    legend_title : str
        Title for the legend
    legend_title_size : int
        Font size for legend title
    legend_text_size : int
        Font size for legend text
    marker_size : int
        Size of scatter plot markers
    alpha : float
        Transparency of markers (0-1)
    output_path : Path or str or None
        Path to save the figure, if None, figure is not saved
    show_plot : bool
        Whether to display the plot
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    # Set the style
    sns.set_style("darkgrid")

    # Create figure
    plt.figure(figsize=(10, 9))

    # Extract the principal components to plot
    pc_x, pc_y = pc_indices

    # Create a DataFrame for seaborn
    df = pd.DataFrame(
        {
            f"PC{pc_x + 1}": activations_pca[:, pc_x],
            f"PC{pc_y + 1}": activations_pca[:, pc_y],
            "Label": [
                "High-stakes" if label == 1 else "Low-stakes" for label in labels
            ],
        }
    )

    # Create the scatter plot
    ax = sns.scatterplot(
        data=df,
        x=f"PC{pc_x + 1}",
        y=f"PC{pc_y + 1}",
        hue="Label",
        palette="husl",
        s=marker_size,
        alpha=alpha,
    )

    # Set labels and title
    plt.xlabel(f"PC{pc_x + 1}", fontsize=axis_label_size)
    plt.ylabel(f"PC{pc_y + 1}", fontsize=axis_label_size)

    # Set tick label size
    plt.xticks(fontsize=tick_label_size)
    plt.yticks(fontsize=tick_label_size)

    # Customize legend
    legend = ax.legend(
        title="Labels",
        fontsize=legend_text_size,
        markerscale=0.7,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        loc="best",
    )
    legend.get_title().set_fontsize(legend_title_size)

    # Adjust layout
    plt.tight_layout()

    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

    return plt.gcf(), ax


def prepare_data_for_pca(
    pickle_file_path: Path,
    dataset_name: str,
    data_paths: Path,
    labels_path: Path,
    inputs: Optional[list] = None,
):
    inputs = []
    labels_ = []
    activations = []
    activations_data = h5py.File(pickle_file_path, "r")
    activations_ = activations_data[dataset_name]["mean_activations"][:]  # type: ignore
    activations_data.close()

    with open(labels_path, "r") as f:
        data = [json.loads(line) for line in f]
    data_entry = [entry for entry in data if entry["dataset_name"] == dataset_name][0]
    labels = data_entry["ground_truth_labels"]

    with open(data_paths, "r") as f:
        data = [json.loads(line) for line in f]
    inputs.extend(
        [d["inputs"] for i, d in enumerate(data) if d["language"] == "English"]
    )
    labels_.extend(
        [label for i, label in enumerate(labels) if (data[i]["language"] == "English")]
    )
    # activations.extend(activations_data[dataset_name]["mean_activations"][:])  # type: ignore

    activations.extend(
        [
            activation
            for i, activation in enumerate(activations_)  # type: ignore
            if (data[i]["language"] == "English")
        ]
    )
    return activations, labels_, inputs


if __name__ == "__main__":
    paths = {
        "mts": Path(
            "data/results/evaluate_probes/cluster_on_training/evaluate_probes_mts_activations.h5"
        ),
        "anthropic": Path(
            "data/results/evaluate_probes/cluster_on_training/evaluate_probes_anthropic_activations.h5"
        ),
        "mt": Path(
            "data/results/evaluate_probes/cluster_on_training/evaluate_probes_mt_activations.h5"
        ),
        "toolace": Path(
            "data/results/evaluate_probes/cluster_on_training/evaluate_probes_toolace_activations.h5"
        ),
        "manual": Path(
            "data/results/evaluate_probes/cluster_updated/evaluate_probes_manual_activations.h5"
        ),
        "training": Path(
            "data/results/evaluate_probes/cluster_on_training/evaluate_probes_training_activations.h5"
        ),
    }
    data_paths = {
        "mts": Path("data/results/evaluate_probes/cluster_on_training/mts.jsonl"),
        "anthropic": Path(
            "data/results/evaluate_probes/cluster_on_training/anthropic.jsonl"
        ),
        "mt": Path("data/results/evaluate_probes/cluster_on_training/mt.jsonl"),
        "toolace": Path(
            "data/results/evaluate_probes/cluster_on_training/toolace.jsonl"
        ),
        "manual": Path("data/results/evaluate_probes/cluster_on_training/manual.jsonl"),
        "training": Path(
            "data/results/evaluate_probes/cluster_on_training/training.jsonl"
        ),
    }
    coefs_path = Path(
        "data/results/evaluate_probes/cluster_on_training/results_balanced_cluster_coefs.json"
    )
    labels_path = Path(
        "data/results/evaluate_probes/cluster_on_training/results_balanced_cluster.jsonl"
    )
    keyy = "mts"
    # visualize_activations_pca(
    #     {keyy: paths[keyy]}, {keyy: data_paths[keyy]}, coefs_path, labels_path
    # )

    # for keyy in paths.keys():
    #     activations = h5py.File(paths[keyy], "r")[keyy]["mean_activations"][:]  # type: ignore
    #     labels = [json.loads(line)["ground_truth_labels"] for line in open(labels_path)]
    #     inputs = [json.loads(line)["inputs"] for line in open(data_paths[keyy])]
    #     activations_, labels_, inputs_ = prepare_data_for_pca(
    #         pickle_file_path=paths[keyy],
    #         dataset_name=keyy,
    #         data_paths=data_paths[keyy],
    #         labels_path=labels_path,
    #         inputs=inputs,
    #     )
    #     pca = PCA(n_components=4)
    #     activations_pca = pca.fit_transform(activations_)
    #     rotation = (60, 120)
    # visualize_activations_pca_seaborn_3d(
    #     activations_pca,  # type: ignore
    #     labels_,  # type: ignore
    #     inputs_,
    #     (1, 2, 3),
    #     "3D PCA Visualization",
    #     output_path=Path(
    #         f"data/results/evaluate_probes/cluster_on_training/rotation_{rotation[0]}_{rotation[1]}/3d_pca_{keyy}.png"
    #     ),
    #     rotation=rotation,
    #     show_plot=True,
    # )
    # visualize_activations_pca_seaborn_2d(
    #     activations_pca,  # type: ignore
    #     labels_,  # type: ignore
    #     inputs_,
    #     (1, 2),
    #     "2D PCA Visualization",
    #     output_path=Path(
    #         f"data/results/evaluate_probes/cluster_on_training/2d_pca_{keyy}.png"
    #     ),
    #     show_plot=True,
    # )
    keyy = "training"
    activations = h5py.File(paths[keyy], "r")[keyy]["mean_activations"][:]  # type: ignore
    labels = [json.loads(line)["ground_truth_labels"] for line in open(labels_path)]
    inputs = [json.loads(line)["inputs"] for line in open(data_paths[keyy])]
    activations_, labels_, inputs_ = prepare_data_for_pca(
        pickle_file_path=paths[keyy],
        dataset_name=keyy,
        data_paths=data_paths[keyy],
        labels_path=labels_path,
        inputs=inputs,
    )
    pca = PCA(n_components=4)
    activations_pca = pca.fit_transform(activations_)
    rotation = (60, 120)
    visualize_activations_pca_seaborn_3d(
        activations_pca,  # type: ignore
        labels_,  # type: ignore
        inputs_,
        (1, 2, 3),
        "3D PCA Visualization",
        output_path=Path(
            f"data/results/evaluate_probes/cluster_on_training/3d_pca_{keyy}.png"
        ),
        rotation=rotation,
        show_plot=True,
    )
    visualize_activations_pca_seaborn_2d(
        activations_pca,  # type: ignore
        labels_,  # type: ignore
        inputs_,
        (1, 2),
        "2D PCA Visualization",
        output_path=Path(
            f"data/results/evaluate_probes/cluster_on_training/2d_pca_{keyy}.png"
        ),
        show_plot=True,
    )
