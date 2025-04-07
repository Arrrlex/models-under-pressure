import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models_under_pressure.config import (
    AIS_DATASETS,
    EVAL_DATASETS_BALANCED,
    EVAL_DATASETS_RAW,
    PLOTS_DIR,
)
from models_under_pressure.interfaces.dataset import Input, Label, LabelledDataset


def show_label_distribution_with_field(
    dataset_key: str, field_name: str = "is_sandbagging"
):
    """
    Show label distribution for a dataset, breaking down by a boolean field (sandbagging or deception).
    Creates a visualization of the distribution.

    Args:
        dataset_key: Key in AIS_DATASETS dictionary
        field_name: Name of the boolean field to analyze (default: "is_sandbagging")
    """
    print(f"\nAnalyzing {dataset_key} with field {field_name}\n")
    dataset = LabelledDataset.load_from(AIS_DATASETS[dataset_key]["file_path_or_name"])

    # Get the labels (high-stakes or low-stakes)
    labels = dataset.labels
    # Get the field information (sandbagging or deception)
    field_values = dataset.other_fields.get(field_name, [])

    if not field_values:
        print(f"Field '{field_name}' not found in dataset '{dataset_key}'")
        return

    # Count high-stakes and low-stakes examples
    high_stakes_count = labels.count(Label.HIGH_STAKES)
    low_stakes_count = labels.count(Label.LOW_STAKES)
    ambiguous_count = labels.count(Label.AMBIGUOUS)

    print(f"Dataset: {dataset_key}")
    print(f"Total examples: {len(labels)}")
    print(
        f"High-stakes examples: {high_stakes_count} ({high_stakes_count / len(labels) * 100:.2f}%)"
    )
    print(
        f"Low-stakes examples: {low_stakes_count} ({low_stakes_count / len(labels) * 100:.2f}%)"
    )
    print(
        f"Ambiguous examples: {ambiguous_count} ({ambiguous_count / len(labels) * 100:.2f}%)"
    )
    print("\n")

    # Analyze field values (True vs False)
    true_examples = [i for i, val in enumerate(field_values) if val]
    false_examples = [i for i, val in enumerate(field_values) if not val]

    print(
        f"{field_name.replace('is_', '').title()} examples: {len(true_examples)} ({len(true_examples) / len(labels) * 100:.2f}%)"
    )
    print(
        f"Non-{field_name.replace('is_', '').lower()} examples: {len(false_examples)} ({len(false_examples) / len(labels) * 100:.2f}%)"
    )
    print("\n")

    # Count high-stakes and low-stakes for True examples
    true_high_stakes = sum(1 for i in true_examples if labels[i] == Label.HIGH_STAKES)
    true_low_stakes = sum(1 for i in true_examples if labels[i] == Label.LOW_STAKES)
    true_ambiguous = sum(1 for i in true_examples if labels[i] == Label.AMBIGUOUS)

    print(f"For {field_name.replace('is_', '').lower()} examples:")
    if true_examples:
        print(
            f"High-stakes: {true_high_stakes} ({true_high_stakes / len(true_examples) * 100:.2f}%)"
        )
        print(
            f"Low-stakes: {true_low_stakes} ({true_low_stakes / len(true_examples) * 100:.2f}%)"
        )
        print(
            f"Ambiguous: {true_ambiguous} ({true_ambiguous / len(true_examples) * 100:.2f}%)"
        )
    else:
        print("No examples found.")
    print("\n")

    # Count high-stakes and low-stakes for False examples
    false_high_stakes = sum(1 for i in false_examples if labels[i] == Label.HIGH_STAKES)
    false_low_stakes = sum(1 for i in false_examples if labels[i] == Label.LOW_STAKES)
    false_ambiguous = sum(1 for i in false_examples if labels[i] == Label.AMBIGUOUS)

    print(f"For non-{field_name.replace('is_', '').lower()} examples:")
    if false_examples:
        print(
            f"High-stakes: {false_high_stakes} ({false_high_stakes / len(false_examples) * 100:.2f}%)"
        )
        print(
            f"Low-stakes: {false_low_stakes} ({false_low_stakes / len(false_examples) * 100:.2f}%)"
        )
        print(
            f"Ambiguous: {false_ambiguous} ({false_ambiguous / len(false_examples) * 100:.2f}%)"
        )
    else:
        print("No examples found.")

    # Calculate precision and recall metrics for high-stakes as a predictor
    # Let's consider "high-stakes" as the positive prediction for the field value

    # True positives: high-stakes examples that are also True for the field
    true_positives = true_high_stakes

    # False positives: high-stakes examples that are False for the field
    false_positives = false_high_stakes

    # False negatives: non-high-stakes examples that are True for the field
    false_negatives = true_low_stakes + true_ambiguous

    # True negatives: non-high-stakes examples that are False for the field
    true_negatives = false_low_stakes + false_ambiguous

    # Calculate precision: TP / (TP + FP)
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )

    # Calculate recall: TP / (TP + FN)
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    accuracy = (true_positives + true_negatives) / len(labels) if len(labels) > 0 else 0

    # Print metrics
    print(
        "\nUsing high-stakes label to predict",
        field_name.replace("is_", "").lower(),
        "status:",
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # Create a bar chart for precision and recall
    plt.figure(figsize=(10, 6))

    metrics = ["Precision", "Recall", "F1 Score", "Accuracy"]
    values = [precision, recall, f1_score, accuracy]

    # Create bars with different colors
    bars = plt.bar(metrics, values, color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"])

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    # Add a horizontal line at y=0.5 for reference
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)

    # Add a horizontal line at y=1.0 for reference
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7)

    plt.ylim(0, 1.1)  # Set y-axis limits
    plt.title(
        f"Performance Metrics: High-Stakes Label as Predictor for {field_name.replace('is_', '').title()}"
    )
    plt.ylabel("Score")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a confusion matrix as text annotation
    plt.figtext(
        0.15,
        0.15,
        f"Confusion Matrix:\n\n"
        f"TP: {true_positives}\n"
        f"FP: {false_positives}\n"
        f"FN: {false_negatives}\n"
        f"TN: {true_negatives}",
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.5"),
    )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{dataset_key}_{field_name}_prediction_metrics.png")
    plt.show()


# For backward compatibility
def show_sandbagging_label_distribution(dataset_key: str = "mmlu_sandbagging"):
    """Backward compatibility wrapper for show_label_distribution_with_field"""
    show_label_distribution_with_field(dataset_key, "is_sandbagging")


def get_text_stats(text_input: Input) -> tuple[int, int]:
    """
    Get the length and number of turns for a text input.

    Args:
        text_input: Either a string or a Dialogue object

    Returns:
        tuple: (total_length, num_turns)
    """
    if isinstance(text_input, str):
        return len(text_input), 1
    return sum(len(message.content) for message in text_input), len(text_input)


def calculate_dataset_stats(
    dataset: LabelledDataset,
) -> tuple[float, str, float, float, str, float]:
    """
    Calculate length and turn statistics for a dataset.

    Args:
        dataset: LabelledDataset object

    Returns:
        tuple: (mean_length, length_range, length_std, mean_turns, turns_range, turns_std)
    """
    lengths, turns = zip(*[get_text_stats(text) for text in dataset.inputs])

    return (
        np.mean(lengths),
        f"{min(lengths)}-{max(lengths)}",
        np.std(lengths),
        np.mean(turns),
        f"{min(turns)}-{max(turns)}",
        np.std(turns),
    )


def print_dataset_length_comparison():
    """
    Creates and prints a pandas DataFrame comparing character lengths and turn counts
    between raw and balanced datasets. Includes mean, range, and standard deviation.
    """
    # Define metrics and create data dictionary
    metrics = [
        "Mean Length",
        "Length Range",
        # "Length Std",
        "Mean Turns",
        "Turns Range",
        # "Turns Std",
    ]
    data = {
        f"{prefix}{metric}": []
        for prefix in ["Raw ", "Balanced "]
        for metric in metrics
    }
    dataset_names = []

    # Process each dataset pair
    for name in EVAL_DATASETS_RAW.keys():
        dataset_names.append(name)

        # Process both raw and balanced datasets
        for prefix, datasets_dict in [
            ("Raw ", EVAL_DATASETS_RAW),
            ("Balanced ", EVAL_DATASETS_BALANCED),
        ]:
            if datasets_dict[name].exists():
                dataset = LabelledDataset.load_from(datasets_dict[name])
                stats = calculate_dataset_stats(dataset)

                # Unpack all statistics
                mean_len, range_len, std_len, mean_turns, range_turns, std_turns = stats

                data[f"{prefix}Mean Length"].append(mean_len)
                data[f"{prefix}Length Range"].append(range_len)
                # data[f"{prefix}Length Std"].append(std_len)
                data[f"{prefix}Mean Turns"].append(mean_turns)
                data[f"{prefix}Turns Range"].append(range_turns)
                # data[f"{prefix}Turns Std"].append(std_turns)
            else:
                for metric in metrics:
                    data[f"{prefix}{metric}"].append(None)

    # Create DataFrame
    df = pd.DataFrame(data, index=dataset_names)

    # Format numeric columns to 2 decimal places
    numeric_columns = [col for col in df.columns if "Range" not in col]
    for col in numeric_columns:
        df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else None)

    print("\nDataset Length and Turn Count Comparison:")
    print(df.to_string())
    return df


def main():
    # Print distributions for raw datasets
    print("\nRAW DATASETS:")
    for name, path in EVAL_DATASETS_RAW.items():
        if path.exists():
            print(f"\n{name.upper()}")
            LabelledDataset.load_from(path).print_label_distribution()

    # Print distributions for balanced datasets
    print("\nBALANCED DATASETS:")
    for name, path in EVAL_DATASETS_BALANCED.items():
        if path.exists():
            print(f"\n{name.upper()}")
            LabelledDataset.load_from(path).print_label_distribution()

    # Add the length comparison
    print()
    print_dataset_length_comparison()


if __name__ == "__main__":
    main()
    # print_dataset_length_comparison()

    # show_label_distribution_with_field("mmlu_sandbagging", "is_sandbagging")
    # show_label_distribution_with_field("deception", "is_deceptive")
