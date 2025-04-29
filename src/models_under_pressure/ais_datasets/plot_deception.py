import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from models_under_pressure.config import DATA_DIR, PLOTS_DIR
from models_under_pressure.interfaces.dataset import LabelledDataset


def line_plot_deception_by_label(datasets: dict[str, LabelledDataset]):
    """
    Creates a line plot showing the percentage of deceptive inputs across different scale labels (0-9)
    for each dataset. Each line represents a different dataset.

    Args:
        datasets: List of LabelledDataset instances to plot

    Returns:
        Figure: The matplotlib figure containing the plot
    """
    plot_data = []
    for name, dataset in datasets.items():
        deception_scores = np.array(dataset.other_fields["deception_scores"])
        scale_labels = np.array(dataset.other_fields["scale_labels"])

        unique_labels = sorted(set(scale_labels.tolist()))

        mean_scores = np.array(
            [deception_scores[scale_labels == scale].mean() for scale in unique_labels]
        )

        count = np.array([np.sum(scale_labels == scale) for scale in unique_labels])
        print()
        print(name)

        for scale, mean_score, count in zip(unique_labels, mean_scores, count):
            plot_data.append(
                {
                    "dataset": name,
                    "stakes": scale,
                    "deception_score": mean_score,
                    "count": count,
                }
            )
            print(f"{scale=} {mean_score=} {count=}")

    plot_data = pd.DataFrame(plot_data)

    # Create the line plot using seaborn
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=plot_data, x="stakes", y="deception_score", hue="dataset", marker="o"
    )

    # Customize the plot
    plt.xlabel("Stakes Level (1-10)")
    plt.ylabel("Deception Score")
    plt.title("Deception Score by Stakes Level Across Datasets")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")
    # Save the figure
    plt.savefig(PLOTS_DIR / "deception_by_scale_line.png", bbox_inches="tight", dpi=300)


def plot_deception_by_label(
    dataset_name: str, dataset: LabelledDataset, plot_type: str = "stripplot"
):
    """
    Creates a plot showing deception scores against scale labels with a line of best fit.
    Supports different types of plots for better visualization of overlapping points.

    Args:
        dataset_name: Name of the dataset
        dataset: LabelledDataset instance to plot
        plot_type: Type of plot to generate. Options: "stripplot", "swarmplot", "violinplot"
    """
    deception_scores = np.array(dataset.other_fields["deception_scores"])
    scale_labels = np.array(
        [int(label) for label in dataset.other_fields["scale_labels"]]
    )

    # Create a DataFrame for seaborn
    df = pd.DataFrame(
        {"Stakes Level": scale_labels, "Deception Score": deception_scores}
    )

    # Create the plot with the regression line first
    g = sns.lmplot(
        data=df,
        x="Stakes Level",
        y="Deception Score",
        robust=True,
        scatter=False,
        color="red",
        line_kws={"linestyle": "--"},
        height=6,
        aspect=1.67,  # This gives us roughly 10x6 figure
    )

    # Add the specified plot type on top
    if plot_type == "stripplot":
        sns.stripplot(
            data=df,
            x="Stakes Level",
            y="Deception Score",
            jitter=0.2,
            alpha=0.5,
            size=5,
            ax=g.ax,
        )
    elif plot_type == "swarmplot":
        sns.swarmplot(
            data=df,
            x="Stakes Level",
            y="Deception Score",
            size=3,
            alpha=0.7,
            ax=g.ax,
        )
    elif plot_type == "violinplot":
        sns.violinplot(data=df, x="Stakes Level", y="Deception Score", ax=g.ax)
        sns.stripplot(
            data=df,
            x="Stakes Level",
            y="Deception Score",
            jitter=0.2,
            alpha=0.3,
            size=4,
            color="black",
            ax=g.ax,
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    # Customize the plot
    g.ax.set_xlabel("Stakes Level (1-10)")
    g.ax.set_ylabel("Deception Score")
    g.ax.set_title(f"Deception Score by Stakes Level - {dataset_name} ({plot_type})")
    g.ax.grid(True, alpha=0.3)
    g.ax.legend(["Line of best fit", "Data points"])

    # Save the figure and close it
    plt.savefig(
        PLOTS_DIR / f"deception_by_scale_{plot_type}_{dataset_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(g.fig)  # Close the figure to free memory


def load_ais_datasets() -> dict[str, LabelledDataset]:
    """
    Loads all labelled datasets from data/evals/ais/*.jsonl files.

    Returns:
        List[LabelledDataset]: List of loaded labelled datasets
    """
    ais_dir = DATA_DIR / "evals" / "ais"
    return {
        file_path.stem: LabelledDataset.load_from(file_path)
        for file_path in ais_dir.glob("*.jsonl")
    }


if __name__ == "__main__":
    # Set matplotlib to not show warnings about too many figures
    plt.rcParams["figure.max_open_warning"] = 0

    datasets = load_ais_datasets()
    line_plot_deception_by_label(datasets)

    # Generate all plot variants for each dataset
    plot_types = ["stripplot", "swarmplot", "violinplot"]
    for name, dataset in datasets.items():
        for plot_type in plot_types:
            plot_deception_by_label(name, dataset, plot_type)
