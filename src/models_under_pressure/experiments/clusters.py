import json
from pathlib import Path

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
    ids = []
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
        subplot_titles=(
            f"3D PCA Visualization of {dataset_name}",
            "2D PCA Visualization",
            "Variance Explained by Principal Components",
        ),
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
    origin = np.mean(activations_pca, axis=0)
    # Scale the vector for better visualization
    scale_factor = 2.0
    fig.add_trace(
        go.Scatter3d(
            x=[origin[pc_x], origin[pc_x] + scale_factor * probe_weights_pca[pc_x]],
            y=[origin[pc_y], origin[pc_y] + scale_factor * probe_weights_pca[pc_y]],
            z=[origin[pc_z], origin[pc_z] + scale_factor * probe_weights_pca[pc_z]],
            mode="lines+markers",
            name="Probe Direction (3D)",
            line=dict(color="red", width=6),
            marker=dict(size=5),
        ),
        row=1,
        col=1,
    )

    # Keep the original 2D plot (PC2 vs PC3)
    pc_x = 2
    pc_y = 3

    # 2D scatter plot
    for label in np.unique(labels):
        mask = np.array(labels) == label
        fig.add_trace(
            go.Scatter(
                x=activations_pca[mask, pc_x],
                y=activations_pca[mask, pc_y],
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
    fig.add_trace(
        go.Scatter(
            x=[origin[pc_x], origin[pc_x] + probe_weights_pca[pc_x]],
            y=[origin[pc_y], origin[pc_y] + probe_weights_pca[pc_y]],
            mode="lines+markers",
            name="Probe Direction (2D)",
            line=dict(color="red", width=3),
            marker=dict(size=8),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Update 3D layout
    fig.update_scenes(
        aspectmode="cube",
        xaxis_title=f"PC1 ({explained_variance_ratio[0]:.2%})",
        yaxis_title=f"PC2 ({explained_variance_ratio[1]:.2%})",
        zaxis_title=f"PC3 ({explained_variance_ratio[2]:.2%})",
    )

    # Update 2D layout
    fig.update_xaxes(
        title_text=f"PC{pc_x + 1} ({explained_variance_ratio[pc_x]:.2%})", row=1, col=2
    )
    fig.update_yaxes(
        title_text=f"PC{pc_y + 1} ({explained_variance_ratio[pc_y]:.2%})", row=1, col=2
    )

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
        title="PCA of Activations with Probe Direction",
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


if __name__ == "__main__":
    paths = {
        # "mts": Path(
        #     "data/results/evaluate_probes/cluster_on_training/evaluate_probes_mts_activations.h5"
        # ),
        # "anthropic": Path(
        #     "data/results/evaluate_probes/cluster_on_training/evaluate_probes_anthropic_activations.h5"
        # ),
        "mt": Path(
            "data/results/evaluate_probes/cluster_on_training/evaluate_probes_mt_activations.h5"
        ),
        # "toolace": Path(
        #     "data/results/evaluate_probes/cluster_on_training/evaluate_probes_toolace_activations.h5"
        # ),
        # "manual": Path(
        #     "data/results/evaluate_probes/cluster_updated/evaluate_probes_manual_activations.h5"
        # ),
        # "training": Path(
        #     "data/results/evaluate_probes/cluster_on_training/evaluate_probes_training_activations.h5"
        # ),
    }
    data_paths = {
        # "mts": Path("data/results/evaluate_probes/cluster_on_training/mts.jsonl"),
        # "anthropic": Path(
        #     "data/results/evaluate_probes/cluster_on_training/anthropic.jsonl"
        # ),
        "mt": Path("data/results/evaluate_probes/cluster_on_training/mt.jsonl"),
        # "toolace": Path(
        #     "data/results/evaluate_probes/cluster_on_training/toolace.jsonl"
        # ),
        # "manual": Path("data/results/evaluate_probes/cluster_on_training/manual.jsonl"),
        # "training": Path(
        #     "data/results/evaluate_probes/cluster_on_training/training.jsonl"
        # ),
    }
    coefs_path = Path(
        "data/results/evaluate_probes/cluster_on_training/results_balanced_cluster_coefs.json"
    )
    labels_path = Path(
        "data/results/evaluate_probes/cluster_on_training/results_balanced_cluster.jsonl"
    )
    visualize_activations_pca(paths, data_paths, coefs_path, labels_path)
