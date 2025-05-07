# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from models_under_pressure.config import DATA_DIR, PROJECT_ROOT, TEST_DATASETS_BALANCED
from models_under_pressure.figures.plot_auroc_by_token_length import (
    get_dataset_name,
    get_probe_results_with_token_lengths,
)
from models_under_pressure.interfaces.dataset import LabelledDataset, to_input_str

# %%

all_data = get_probe_results_with_token_lengths(
    list(DATA_DIR.glob("results/evaluate_probes/*test*.jsonl")),
    "meta-llama/Llama-3.2-1B-Instruct",
)

# Only attention probe with token length between 60 and 70
df = all_data[
    (all_data["token_length"] > 60)
    & (all_data["token_length"] < 70)
    & (all_data["probe_name"] == "Attention")
].copy()

# %%

print(f"Mean AUROC for attention probe: {roc_auc_score(df['label'], df['pred']):.3f}")

print("Distribution of datasets:")
for dataset in df["dataset_name"].unique():
    df_dataset = df[df["dataset_name"] == dataset]
    print(f"{dataset}:", end=" ")
    print(f"{df_dataset['label'].value_counts().to_dict()}", end=" ")
    print(f"AUROC: {roc_auc_score(df_dataset['label'], df_dataset['pred']):.3f}")

# %%

# Plot score distribution for each dataset
plt.figure(figsize=(12, 6))

for i, dataset in enumerate(df.dataset_name.unique()):
    dataset_data = df[df.dataset_name == dataset]
    plt.hist(
        dataset_data.pred,
        bins=30,
        alpha=0.7,
        density=True,
        label=dataset,
        histtype="step",
        linewidth=2,
    )

plt.xlabel("Prediction Score")
plt.ylabel("Density")
plt.title("Score Distribution by Dataset")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %%

# Calculate proportion of positive labels for each token length
token_length_stats = (
    df.groupby("token_length")["label"].agg(["mean", "count"]).reset_index()
)

plt.figure(figsize=(10, 6))
plt.plot(token_length_stats["token_length"], token_length_stats["mean"], marker="o")

plt.xlabel("Token Length")
plt.ylabel("Proportion of High Stakes Labels")
plt.title("Proportion of High Stakes Labels vs Token Length")
plt.grid(True, alpha=0.3)

# Add count annotations
for _, row in token_length_stats.iterrows():
    plt.annotate(
        f"n={int(row['count'])}",
        xy=(row["token_length"], row["mean"]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )

plt.tight_layout()
plt.show()


# %%

# Load all datasets to get input texts
datasets = {
    get_dataset_name(path): LabelledDataset.load_from(PROJECT_ROOT / path)
    for path in TEST_DATASETS_BALANCED.values()
}

texts = pd.DataFrame(
    [
        {
            "dataset_name": dataset_name,
            "text": to_input_str(record.input),
            "id": record.id,
        }
        for dataset_name, dataset in datasets.items()
        for record in dataset.to_records()
    ]
)
if "text" not in df.columns:
    df = df.merge(texts, how="left", on=["id", "dataset_name"])

# %%

# Sort by absolute difference between prediction and label
df["pred_label_diff"] = abs(df["pred"] - df["label"])
sorted_texts = df.sort_values("pred_label_diff", ascending=False)

# Display top examples with largest prediction-label difference
print("\nExamples with largest prediction-label difference:")
print("-" * 80)
for _, row in sorted_texts.head().iterrows():
    print(f"Dataset: {row['dataset_name']}")
    print(f"Prediction: {row['pred']:.3f}, Label: {row['label']}")
    print(f"Absolute Difference: {row['pred_label_diff']:.3f}")
    print(f"Text:\n{row['text']}\n")
    print("-" * 80)

# %%

# Print distribution statistics for predictions and labels
print("\nDistribution statistics for predictions and labels:")
print("-" * 80)
print("\nPredictions:")
print(df["pred"].describe())
print("\nLabels:")
print(df["label"].describe())

# Print value counts for labels since they're likely categorical
print("\nLabel value counts:")
print(df["label"].value_counts())


# %%
