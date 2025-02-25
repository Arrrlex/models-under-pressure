import os

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

file_path = os.path.join(os.path.dirname(__file__), "dataset3.csv")
df = pd.read_csv(file_path)


# Add a new column for prompt length
df["prompt_length"] = df["prompt_text"].apply(len)


# Function to generate and display word cloud
def generate_word_cloud(data: pd.DataFrame, title: str):
    text = " ".join(data["prompt_text"])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        text
    )

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=14)
    plt.show()


# Generate word clouds for high-stakes and low-stakes prompts
high_stakes_data = df[df["high_stakes"] == 1]
low_stakes_data = df[df["high_stakes"] == 0]

generate_word_cloud(high_stakes_data, "Word Cloud for High-Stakes Prompts")
generate_word_cloud(low_stakes_data, "Word Cloud for Low-Stakes Prompts")

# Plot the distribution of prompt length for high-stakes vs low-stakes scenarios
plt.figure(figsize=(12, 5))
plt.hist(
    high_stakes_data["prompt_length"],
    bins=20,
    alpha=0.6,
    label="High-Stakes",
    color="red",
    edgecolor="black",
)
plt.hist(
    low_stakes_data["prompt_length"],
    bins=20,
    alpha=0.6,
    label="Low-Stakes",
    color="blue",
    edgecolor="black",
)
plt.xlabel("Prompt Length")
plt.ylabel("Frequency")
plt.title("Distribution of Prompt Length for High-Stakes vs. Low-Stakes Prompts")
plt.legend()
plt.show()

# Box plot to analyze prompt length distribution
plt.figure(figsize=(8, 5))
df.boxplot(
    column="prompt_length",
    by="high_stakes",
    grid=False,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="red"),
)
plt.xlabel("High Stakes (0 = No, 1 = Yes)")
plt.ylabel("Prompt Length")
plt.title("Box Plot of Prompt Length by High-Stakes Category")
plt.suptitle("")  # Remove default title
plt.show()
