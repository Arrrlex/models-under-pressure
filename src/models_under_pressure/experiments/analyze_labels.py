import matplotlib.pyplot as plt
import numpy as np

from models_under_pressure.config import EVALS_DIR, PLOTS_DIR
from models_under_pressure.interfaces.dataset import Dataset

dataset_files = [f for f in EVALS_DIR.glob("*_scale.jsonl")]

datasets = {f.stem: Dataset.load_from(f) for f in dataset_files}


def analyze_scale_labels(
    dataset: Dataset, dataset_name: str, confidence_threshold: float = 0.0
):
    # Dictionary to store scale labels for each label type
    label_types = ["high-stakes", "ambiguous", "low-stakes"]
    scale_labels_by_type = {label_type: [] for label_type in label_types}

    # Collect scale labels for all label types
    for record in dataset.to_records():
        label_type = record.other_fields.get("labels")
        if label_type in label_types:
            scale_label = record.other_fields.get("scale_labels")
            confidence = record.other_fields.get("scale_label_confidence", 10)

            # Only include if confidence meets threshold
            if scale_label is not None and confidence >= confidence_threshold:
                scale_labels_by_type[label_type].append(scale_label)

    # Create grouped bar chart
    plt.figure(figsize=(12, 7))

    # Set up bar positions
    bar_width = 0.25
    bins = range(1, 11)

    # Calculate histograms for each label type
    histograms = {}
    max_count = 0

    for i, label_type in enumerate(label_types):
        if scale_labels_by_type[label_type]:
            hist, _ = np.histogram(scale_labels_by_type[label_type], bins=range(1, 12))
            histograms[label_type] = hist
            max_count = max(max_count, max(hist))
        else:
            histograms[label_type] = np.zeros(10)

    # Plot bars for each label type
    x = np.arange(len(bins))
    for i, label_type in enumerate(label_types):
        plt.bar(
            x + (i - 1) * bar_width,
            histograms[label_type],
            width=bar_width,
            label=label_type.title(),
        )

    # Add labels and title
    confidence_info = (
        f" (Confidence ≥ {confidence_threshold})" if confidence_threshold > 0 else ""
    )
    plt.title(f"Distribution of Scale Labels for {dataset_name}{confidence_info}")
    plt.xlabel("Scale Label (1-10)")
    plt.ylabel("Count")
    plt.xticks(x, bins)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure
    threshold_str = f"_conf{confidence_threshold}" if confidence_threshold > 0 else ""
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"label_distribution_{dataset_name}_{threshold_str}.png")
    # plt.show()

    # Print statistics
    for label_type in label_types:
        labels = scale_labels_by_type[label_type]
        print(f"Total {label_type} samples: {len(labels)}")
        if labels:
            print(f"Average scale label for {label_type}: {np.mean(labels):.2f}")
        else:
            print(f"No {label_type} samples found")
    print("-" * 40)


if __name__ == "__main__":
    # Define confidence thresholds to analyze
    confidence_thresholds = [0, 6, 9]

    for dataset_name, dataset in datasets.items():
        print(f"Dataset: {dataset_name}")

        # Print confidence score distribution
        confidence_scores = dataset.other_fields["scale_label_confidence"]
        print("\nConfidence score distribution:")
        hist, bins = np.histogram(confidence_scores, bins=range(1, 12))
        for i, count in enumerate(hist, start=1):
            print(f"Confidence {i}: {count}")
        print()

        for threshold in confidence_thresholds:
            print(f"Analyzing with confidence threshold: {threshold}")
            analyze_scale_labels(dataset, dataset_name, confidence_threshold=threshold)
