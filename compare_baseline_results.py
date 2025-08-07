#!/usr/bin/env python3
"""
Script to compare generation baseline results with other model results.

Creates an Excel file with dataset tabs and comparison data including:
- ID, full input, ground truth
- Scores and labels from both models
- Binary correctness flags
- Disagreement flag

Usage:
    Configure the variables below and call main() with appropriate parameters.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import LOCAL_MODELS, RESULTS_DIR, TEST_DATASETS
from models_under_pressure.interfaces.dataset import LabelledDataset

# Configuration variables
GENERATION_BASELINE_DIR = RESULTS_DIR / "baselines" / "generation"
OTHER_RESULTS_DIR = RESULTS_DIR / "monitoring_cascade_neurips"


def load_jsonl_results(file_path: Path) -> List[Dict[str, Any]]:
    """Load results from a JSONL file."""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_generation_baseline_results(file_path: Path) -> Dict[str, Any]:
    """Load results from a generation baseline JSONL file, taking only the last line."""
    last_result = None
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                last_result = json.loads(line)

    if last_result is None:
        raise ValueError(f"No valid JSON lines found in {file_path}")

    return last_result


def find_generation_baseline_files(model_name: str) -> List[Path]:
    """Find all generation baseline files for a given model."""

    # Try different patterns to find files
    # First try exact model name, then partial matches
    patterns = [
        f"{model_name}_*_generation_baseline.jsonl",
        f"{model_name.lower()}_*_generation_baseline.jsonl",
        f"*{model_name}*_generation_baseline.jsonl",
        f"*{model_name.lower()}*_generation_baseline.jsonl",
    ]

    # If model is in LOCAL_MODELS, also try with the actual model path name
    if model_name in LOCAL_MODELS:
        model_path = LOCAL_MODELS[model_name]
        model_short_name = model_path.split("/")[-1]
        patterns.extend(
            [
                f"{model_short_name}_*_generation_baseline.jsonl",
                f"{model_short_name.lower()}_*_generation_baseline.jsonl",
                f"*{model_short_name}*_generation_baseline.jsonl",
            ]
        )

    files = []
    for pattern in patterns:
        new_files = list(GENERATION_BASELINE_DIR.glob(pattern))
        files.extend(new_files)
        if files:  # If we found files with this pattern, stop trying other patterns
            break

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file in files:
        if file not in seen:
            seen.add(file)
            unique_files.append(file)

    if not unique_files:
        # Show available files to help user
        available_files = list(
            GENERATION_BASELINE_DIR.glob("*_generation_baseline.jsonl")
        )
        available_models = []
        for file in available_files:
            # Extract model name from filename
            name_part = file.stem.replace("_generation_baseline", "")
            model_part = name_part.split("_")[0]  # Take first part before underscore
            if model_part not in available_models:
                available_models.append(model_part)

        raise FileNotFoundError(
            f"No generation baseline files found for model '{model_name}' in {GENERATION_BASELINE_DIR}.\n"
            f"Tried patterns: {patterns}\n"
            f"Available model patterns from files: {sorted(available_models)}"
        )

    return sorted(unique_files)


def get_dataset_conversation(
    dataset_cache: Dict[str, Any], dataset_name: str, sample_id: str
) -> str:
    """Get the conversation content for a specific sample ID from a cached dataset."""
    if dataset_name not in dataset_cache:
        return ""

    try:
        dataset = dataset_cache[dataset_name]
        # Find the sample with matching ID
        for i, id in enumerate(dataset.ids):
            if id == sample_id:
                # Get the conversation content from inputs
                input_data = dataset.inputs[i]
                if isinstance(input_data, str):
                    return input_data
                else:
                    # It's a Dialogue (Sequence[Message])
                    try:
                        # Convert messages to readable string
                        messages = []
                        for msg in input_data:
                            if hasattr(msg, "role") and hasattr(msg, "content"):
                                messages.append(f"{msg.role}: {msg.content}")
                            else:
                                messages.append(str(msg))
                        return "\n".join(messages)
                    except (TypeError, AttributeError):
                        return str(input_data)
        return ""
    except Exception as e:
        print(f"Warning: Error accessing dataset {dataset_name}: {e}")
        return ""


def load_dataset_cache(dataset_names: List[str]) -> Dict[str, Any]:
    """Load and cache datasets to avoid repeated loading."""
    dataset_cache = {}

    for dataset_name in dataset_names:
        if dataset_name in TEST_DATASETS:
            try:
                print(f"Loading dataset: {dataset_name}")
                dataset_cache[dataset_name] = LabelledDataset.load_from(
                    TEST_DATASETS[dataset_name]
                )
            except Exception as e:
                print(f"Warning: Could not load dataset {dataset_name}: {e}")

    return dataset_cache


def validate_id_alignment(
    gen_result: Dict[str, Any], other_result: Dict[str, Any]
) -> None:
    """Validate that item IDs are aligned between both predictions."""
    gen_ids = gen_result["ids"]
    other_ids = other_result["ids"]

    if len(gen_ids) != len(other_ids):
        raise ValueError(
            f"Length mismatch: Generation baseline has {len(gen_ids)} items, "
            f"other model has {len(other_ids)} items for dataset {gen_result['dataset_name']}"
        )

    for i, (gen_id, other_id) in enumerate(zip(gen_ids, other_ids)):
        if gen_id != other_id:
            raise ValueError(
                f"ID mismatch at position {i}: Generation baseline has '{gen_id}', "
                f"other model has '{other_id}' for dataset {gen_result['dataset_name']}"
            )


def create_comparison_dataframe(
    gen_result: Dict[str, Any],
    other_result: Dict[str, Any],
    dataset_cache: Dict[str, Any],
) -> pd.DataFrame:
    """Create a comparison DataFrame from two result dictionaries."""

    # Validate ID alignment before processing
    validate_id_alignment(gen_result, other_result)

    # Extract basic data from both results
    ids = gen_result["ids"]
    ground_truth = gen_result["ground_truth"]
    dataset_name = gen_result["dataset_name"]

    # Get scores and labels from both models
    gen_scores = gen_result["scores"]
    gen_labels = gen_result["labels"]

    # For the other model, use high_stakes_scores if available, otherwise scores
    other_scores = other_result.get(
        "high_stakes_scores", other_result.get("scores", [])
    )
    other_labels = other_result["labels"]

    # Get full responses and valid responses if available
    full_responses = gen_result.get("full_response", [""] * len(ids))
    valid_responses = gen_result.get("valid_response", [""] * len(ids))

    data = []
    for i, sample_id in enumerate(ids):
        # Get the full input (conversation)
        full_input = get_dataset_conversation(dataset_cache, dataset_name, sample_id)

        # Calculate correctness for both models
        gen_correct = int(gen_labels[i] == ground_truth[i])
        prompted_correct = int(other_labels[i] == ground_truth[i])

        # Calculate disagreement
        disagreement = int(gen_labels[i] != other_labels[i])

        data.append(
            {
                "ID": sample_id,
                "Input": full_input,
                "Ground Truth": ground_truth[i],
                "CoT_Score": gen_scores[i] if i < len(gen_scores) else 0.0,
                "CoT_Label": gen_labels[i],
                "CoT_Correct": gen_correct,
                "CoT_Valid_Response": valid_responses[i]
                if i < len(valid_responses)
                else "",
                "CoT_Full_Response": full_responses[i]
                if i < len(full_responses)
                else "",
                "Prompted_Score": other_scores[i] if i < len(other_scores) else 0.0,
                "Prompted_Label": other_labels[i],
                "Prompted_Correct": prompted_correct,
                "Disagreement": disagreement,
            }
        )

    return pd.DataFrame(data)


def find_matching_results(
    gen_results: List[Dict], other_results: List[Dict]
) -> List[tuple]:
    """Find matching results between generation and other model results."""
    matches = []

    for gen_result in gen_results:
        gen_dataset = gen_result["dataset_name"]

        # Look for matching dataset in other results
        for other_result in other_results:
            other_dataset = other_result["dataset_name"]

            if gen_dataset == other_dataset:
                matches.append((gen_result, other_result))
                break

    return matches


def main(model_name: str, other_filename: str, output_file: Path):
    """
    Compare generation baseline results with other model results.

    Args:
        model_name: Name of the model for generation baseline (e.g., 'gemma-12b')
        other_filename: Filename of the other model results (e.g., 'baseline_llama-70b.jsonl')
        output_file: Path to output Excel file
    """

    # Find generation baseline files for the model
    print(f"Finding generation baseline files for model: {model_name}")
    try:
        gen_files = find_generation_baseline_files(model_name)
        print(f"Found {len(gen_files)} generation baseline files:")
        for file in gen_files:
            print(f"  - {file.name}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Find other model results file
    other_file = OTHER_RESULTS_DIR / other_filename
    if not other_file.exists():
        print(f"Error: Other model file {other_file} does not exist")
        print(f"Available files in {OTHER_RESULTS_DIR}:")
        for file in OTHER_RESULTS_DIR.glob("*.jsonl"):
            print(f"  - {file.name}")
        sys.exit(1)

    # Load generation results from all files (only last line from each)
    print("Loading generation baseline results...")
    gen_results = []
    for gen_file in gen_files:
        gen_results.append(load_generation_baseline_results(gen_file))

    print("Loading other model results...")
    other_results = load_jsonl_results(other_file)

    print(
        f"Found {len(gen_results)} generation results and {len(other_results)} other model results"
    )

    # Find matching results
    matches = find_matching_results(gen_results, other_results)
    print(f"Found {len(matches)} matching datasets")

    if not matches:
        print("Error: No matching datasets found between the two result files")
        sys.exit(1)

    # Load datasets once and cache them for efficiency
    dataset_names = list(set(gen_result["dataset_name"] for gen_result, _ in matches))
    print(f"Loading {len(dataset_names)} unique datasets...")
    dataset_cache = load_dataset_cache(dataset_names)

    # Create Excel file with multiple sheets
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        summary_data = []

        for gen_result, other_result in matches:
            dataset_name = gen_result["dataset_name"]
            print(f"Processing dataset: {dataset_name}")

            # Create comparison DataFrame
            df = create_comparison_dataframe(gen_result, other_result, dataset_cache)

            # Write to Excel sheet
            sheet_name = dataset_name[:31]  # Excel sheet names limited to 31 chars
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Calculate summary statistics
            total_samples = len(df)
            cot_accuracy = df["CoT_Correct"].mean()
            prompted_accuracy = df["Prompted_Correct"].mean()
            disagreement_rate = df["Disagreement"].mean()

            # Calculate fraction of invalid responses for CoT
            # Assuming valid_response is boolean or 1/0, where True/1 means valid
            cot_invalid_fraction = (
                1 - df["CoT_Valid_Response"].astype(bool).mean()
                if not df["CoT_Valid_Response"].empty
                else 0.0
            )

            # Calculate AUROC if we have valid scores and binary ground truth
            try:
                # Check if we have binary classification (0/1 ground truth)
                ground_truth_values = df["Ground Truth"].unique()
                if (
                    set(ground_truth_values).issubset({0, 1})
                    and len(ground_truth_values) > 1
                ):
                    cot_auroc = roc_auc_score(df["Ground Truth"], df["CoT_Score"])
                    prompted_auroc = roc_auc_score(
                        df["Ground Truth"], df["Prompted_Score"]
                    )
                else:
                    cot_auroc = float(
                        "nan"
                    )  # Not applicable for non-binary classification
                    prompted_auroc = float("nan")
            except (ValueError, TypeError):
                cot_auroc = float("nan")
                prompted_auroc = float("nan")

            summary_data.append(
                {
                    "Dataset": dataset_name,
                    "Total_Samples": total_samples,
                    "CoT_Accuracy": cot_accuracy,
                    "CoT_AUROC": cot_auroc,
                    "CoT_Invalid_Fraction": cot_invalid_fraction,
                    "Prompted_Accuracy": prompted_accuracy,
                    "Prompted_AUROC": prompted_auroc,
                    "Disagreement_Rate": disagreement_rate,
                    "CoT_Model": gen_result.get("model_name", "Unknown"),
                    "Prompted_Model": other_result.get("model_name", "Unknown"),
                }
            )

        # Create summary sheet
        summary_df = pd.DataFrame(summary_data)

        # Calculate overall averages for the summary row
        avg_total_samples = summary_df["Total_Samples"].sum()
        avg_cot_accuracy = summary_df["CoT_Accuracy"].mean()
        avg_prompted_accuracy = summary_df["Prompted_Accuracy"].mean()
        avg_disagreement_rate = summary_df["Disagreement_Rate"].mean()
        avg_cot_invalid_fraction = summary_df["CoT_Invalid_Fraction"].mean()

        # Calculate average AUROC (excluding NaN values)
        valid_cot_auroc = summary_df["CoT_AUROC"].dropna()
        valid_prompted_auroc = summary_df["Prompted_AUROC"].dropna()
        avg_cot_auroc = (
            valid_cot_auroc.mean() if len(valid_cot_auroc) > 0 else float("nan")
        )
        avg_prompted_auroc = (
            valid_prompted_auroc.mean()
            if len(valid_prompted_auroc) > 0
            else float("nan")
        )

        # Add average row to summary
        average_row = {
            "Dataset": "AVERAGE",
            "Total_Samples": avg_total_samples,
            "CoT_Accuracy": avg_cot_accuracy,
            "CoT_AUROC": avg_cot_auroc,
            "CoT_Invalid_Fraction": avg_cot_invalid_fraction,
            "Prompted_Accuracy": avg_prompted_accuracy,
            "Prompted_AUROC": avg_prompted_auroc,
            "Disagreement_Rate": avg_disagreement_rate,
            "CoT_Model": "Average across all",
            "Prompted_Model": "Average across all",
        }

        # Append the average row to the summary DataFrame
        summary_df = pd.concat(
            [summary_df, pd.DataFrame([average_row])], ignore_index=True
        )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Comparison Excel file saved to: {output_file}")
    print("\nDataset Summary:")
    for _, row in summary_df.iterrows():
        # Format AUROC values, showing "N/A" for NaN values
        cot_auroc_str = (
            f"{row['CoT_AUROC']:.3f}" if not pd.isna(row["CoT_AUROC"]) else "N/A"
        )
        prompted_auroc_str = (
            f"{row['Prompted_AUROC']:.3f}"
            if not pd.isna(row["Prompted_AUROC"])
            else "N/A"
        )

        print(
            f"  {row['Dataset']}: "
            f"CoT_Acc={row['CoT_Accuracy']:.3f}, CoT_AUROC={cot_auroc_str}, CoT_Invalid={row['CoT_Invalid_Fraction']:.3f}, "
            f"Prompted_Acc={row['Prompted_Accuracy']:.3f}, Prompted_AUROC={prompted_auroc_str}, "
            f"Disagree={row['Disagreement_Rate']:.3f}"
        )

    # Calculate and display overall averages (excluding the AVERAGE row we just added)
    print("\nOverall Averages:")
    dataset_rows = summary_df[summary_df["Dataset"] != "AVERAGE"]
    avg_cot_accuracy = dataset_rows["CoT_Accuracy"].mean()
    avg_prompted_accuracy = dataset_rows["Prompted_Accuracy"].mean()
    avg_disagreement = dataset_rows["Disagreement_Rate"].mean()
    avg_cot_invalid_fraction = dataset_rows["CoT_Invalid_Fraction"].mean()

    # Calculate average AUROC (excluding NaN values)
    valid_cot_auroc = dataset_rows["CoT_AUROC"].dropna()
    valid_prompted_auroc = dataset_rows["Prompted_AUROC"].dropna()
    avg_cot_auroc = valid_cot_auroc.mean() if len(valid_cot_auroc) > 0 else float("nan")
    avg_prompted_auroc = (
        valid_prompted_auroc.mean() if len(valid_prompted_auroc) > 0 else float("nan")
    )

    # Format averages
    avg_cot_auroc_str = f"{avg_cot_auroc:.3f}" if not pd.isna(avg_cot_auroc) else "N/A"
    avg_prompted_auroc_str = (
        f"{avg_prompted_auroc:.3f}" if not pd.isna(avg_prompted_auroc) else "N/A"
    )

    print(
        f"  CoT_Acc={avg_cot_accuracy:.3f}, CoT_AUROC={avg_cot_auroc_str}, CoT_Invalid={avg_cot_invalid_fraction:.3f}, "
        f"Prompted_Acc={avg_prompted_accuracy:.3f}, Prompted_AUROC={avg_prompted_auroc_str}, "
        f"Disagree={avg_disagreement:.3f}"
    )


if __name__ == "__main__":
    # Example usage:
    # main(
    #     model_name="deepseek-chat-v3",
    #     other_filename="baseline_llama-70b.jsonl",
    #     output_file=Path("comparison_deepseek_vs_llama70b.xlsx")
    # )

    # You can also test with available models:
    print("Available models:")
    for model in LOCAL_MODELS.keys():
        print(f"  - {model}")

    print(f"\nGeneration baseline directory: {GENERATION_BASELINE_DIR}")
    print(f"Other results directory: {OTHER_RESULTS_DIR}")

    print("\nExample usage:")
    print(
        'main("deepseek-chat-v3", "baseline_llama-70b.jsonl", Path("comparison.xlsx"))'
    )

    # Uncomment below to run a test:
    main(
        "claude-sonnet-4-20250514",
        # "deepseek-chat-v3",
        "baseline_llama-70b.jsonl",
        Path("sonnet4_vs_llama70b.xlsx"),
    )
