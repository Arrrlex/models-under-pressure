# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from models_under_pressure.config import DATA_DIR
from models_under_pressure.interfaces.dataset import LabelledDataset

# %%

path = DATA_DIR / "evals/test/mask_samples_raw.jsonl"
df = LabelledDataset.load_from(path).to_pandas()


# Create crosstab
ct = pd.crosstab(df["pressure"], df["labels"])

# Create heatmap
# Reorder the columns and index
ct = ct.reindex(columns=["low-stakes", "ambiguous", "high-stakes"])
ct = ct.reindex(index=["low", "high"])

plt.figure(figsize=(10, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Distribution of Pressure vs Labels")
plt.xlabel("gpt-4o labels")
plt.tight_layout()
plt.savefig("mask_benchmark_heatmap.pdf")
plt.savefig("mask_benchmark_heatmap.png")
plt.show()

# %%

df.head()

# %%

# Calculate percentage of high pressure responses for each scale label
pressure_by_scale = (
    df.groupby("scale_labels")["pressure"]
    .apply(lambda x: (x == "high").mean() * 100)
    .reset_index()
)

# Create line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=pressure_by_scale, x="scale_labels", y="pressure", marker="o")

plt.title("Percentage High-Pressure Responses by Scale Label")
plt.xlabel("Scale Labels")
plt.ylabel("Percentage High-Pressure Responses")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("pressure_by_scale.pdf")
plt.savefig("pressure_by_scale.png")
plt.show()

# %%

df["pressure"].value_counts()
# %%
