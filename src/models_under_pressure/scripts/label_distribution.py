from models_under_pressure.config import (
    AIS_DATASETS,
    EVAL_DATASETS_BALANCED,
    EVAL_DATASETS_RAW,
)
from models_under_pressure.interfaces.dataset import Label, LabelledDataset


def show_label_distribution_with_field(
    dataset_key: str, field_name: str = "is_sandbagging"
):
    """
    Show label distribution for a dataset, breaking down by a boolean field (sandbagging or deception).

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


# For backward compatibility
def show_sandbagging_label_distribution(dataset_key: str = "mmlu_sandbagging"):
    """Backward compatibility wrapper for show_label_distribution_with_field"""
    show_label_distribution_with_field(dataset_key, "is_sandbagging")


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


if __name__ == "__main__":
    main()
    # show_label_distribution_with_field("mmlu_sandbagging", "is_sandbagging")
    # show_label_distribution_with_field("deception", "is_deceptive")
