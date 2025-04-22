# %%
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.config import EVAL_DATASETS_RAW
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %%

df = LabelledDataset.load_from(EVAL_DATASETS_RAW["mask"]).to_pandas()


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
