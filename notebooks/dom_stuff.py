# %%
# Checking if DoM is working

from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from models_under_pressure.config import EVAL_DATASETS, LOCAL_MODELS
from models_under_pressure.interfaces.activations import Activation
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.model import LLMModel
from models_under_pressure.probes.pytorch_classifiers import (
    PytorchDifferenceOfMeansClassifier,
)

# %%


# Load dataset and get dialogues
dataset_path = EVAL_DATASETS["manual"]
dialogues = []

dataset = LabelledDataset.load_from(dataset_path)

# %%

# Load model
model = LLMModel.load(
    LOCAL_MODELS["llama-1b"],
)
layer = 7

# Get activations for layer 7
activations = model.get_batched_activations(dataset, layer=layer, batch_size=4)

print("Activation shape:", activations.get_activations().shape)
print("Attention mask shape:", activations.get_attention_mask().shape)
print("Input IDs shape:", activations.get_input_ids().shape)

# %%

out_dir = Path("../data/temp/")

np.save(out_dir / "activations.npy", activations.get_activations().astype(np.float16))
np.save(out_dir / "attention_mask.npy", activations.get_attention_mask())
np.save(out_dir / "input_ids.npy", activations.get_input_ids())

# %%

out_dir = Path("../data/temp/")
# Load the saved files
loaded_activations = np.load(out_dir / "activations.npy")
loaded_attention_mask = np.load(out_dir / "attention_mask.npy")
loaded_input_ids = np.load(out_dir / "input_ids.npy")

activations = Activation(
    _activations=loaded_activations,
    _attention_mask=loaded_attention_mask,
    _input_ids=loaded_input_ids,
)

# %%

# Compute mean activations where attention mask is 1
mask_expanded = activations.get_attention_mask()[
    :, :, None
]  # Expand dims to match activations
masked_activations = (
    activations.get_activations() * mask_expanded
)  # Zero out padded positions

# Sum and divide by number of non-padded tokens per sequence
sequence_lengths = activations.get_attention_mask().sum(axis=1, keepdims=True)
sample_activations = masked_activations.sum(axis=1) / sequence_lengths

print(sample_activations.shape)

# %%

high_stakes_indices = np.where(dataset.labels_numpy() == 1)[0]
low_stakes_indices = np.where(dataset.labels_numpy() == 0)[0]

high_stakes_activations = sample_activations[high_stakes_indices]
low_stakes_activations = sample_activations[low_stakes_indices]

print(high_stakes_activations.shape)
print(low_stakes_activations.shape)

overall_mean = np.mean(sample_activations, axis=0)
mean_high_stakes = np.mean(high_stakes_activations, axis=0)
mean_low_stakes = np.mean(low_stakes_activations, axis=0)

difference_of_means = mean_high_stakes - mean_low_stakes
difference_of_means = difference_of_means / np.linalg.norm(difference_of_means)

print(difference_of_means.shape)

# %%

token_level = True
sigmoid_first = True

# b = np.dot(overall_mean, difference_of_means)
b = 0

if token_level is False:
    # Compute dot product between difference of means and each sample
    # Compute scores for mean vectors to verify they give expected results
    mean_high_score = np.dot(mean_high_stakes, difference_of_means) - b
    mean_low_score = np.dot(mean_low_stakes, difference_of_means) - b
    print(f"Score for mean high stakes: {mean_high_score:.3f}")
    print(f"Score for mean low stakes: {mean_low_score:.3f}")

    # Compute scores for all samples
    scores = 1 / (1 + np.exp(-(np.dot(sample_activations, difference_of_means) - b)))

else:
    # Get token-level scores by multiplying with difference_of_means
    token_scores = np.dot(activations.get_activations(), difference_of_means)

    # Apply sigmoid to get probabilities
    if sigmoid_first is True:
        token_scores = 1 / (1 + np.exp(-(token_scores - b)))

    # Mask out padding tokens
    mask = activations.get_attention_mask()
    masked_scores = token_scores * mask

    # Average over sequence positions (excluding padding)
    sequence_lengths = mask.sum(axis=1)  # Remove keepdims=True
    scores = masked_scores.sum(axis=1) / sequence_lengths

    if sigmoid_first is False:
        scores = 1 / (1 + np.exp(-(scores - b)))

print(scores.shape)

# Classify based on whether the score is positive or negative
predictions = (scores > 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions == dataset.labels_numpy())
print(f"Classification accuracy: {accuracy:.3f}")

# Optional: Print confusion matrix

cm = confusion_matrix(dataset.labels_numpy(), predictions)
print("\nConfusion Matrix:")
print(cm)

accuracy = np.sum(predictions == dataset.labels_numpy()) / len(predictions)
print(f"Classification accuracy: {accuracy:.3f}")

# Compute AUROC

auroc = roc_auc_score(dataset.labels_numpy(), scores)
print(f"\nAUROC: {auroc:.3f}")


# %%

# Create and train the classifier

classifier = PytorchDifferenceOfMeansClassifier(training_args={"device": "cpu"})
classifier.train(activations, dataset.labels_numpy())

# Get predictions
print(f"Activations shape: {activations.shape}")
probs = classifier.predict_proba(activations)
print(f"Probs shape: {probs.shape}")
print(probs[:20])
predictions = classifier.predict(activations)

# Calculate accuracy
accuracy = np.mean(predictions == dataset.labels_numpy())
print(f"\nClassification accuracy: {accuracy:.3f}")

# Print confusion matrix
cm = confusion_matrix(dataset.labels_numpy(), predictions)
print("\nConfusion Matrix:")
print(cm)

# Compute AUROC
probs = classifier.predict_proba(activations)
auroc = roc_auc_score(dataset.labels_numpy(), probs)
print(f"\nAUROC: {auroc:.3f}")


# %%
# Normalize the vectors
normalized_diff = difference_of_means / np.linalg.norm(difference_of_means)
normalized_weight = classifier.model.weight.data.numpy().flatten()
normalized_weight = normalized_weight / np.linalg.norm(normalized_weight)

print("Normalized difference of means vector (first 10):")
print(normalized_diff[:10])
print("\nNormalized classifier weights (first 10):")
print(normalized_weight[:10])

# Calculate similarity metrics
cosine_sim = np.dot(normalized_diff, normalized_weight)
l2_dist = np.linalg.norm(normalized_diff - normalized_weight)

print(f"\nCosine similarity: {cosine_sim:.3f}")
print(f"L2 distance: {l2_dist:.3f}")

# %%
