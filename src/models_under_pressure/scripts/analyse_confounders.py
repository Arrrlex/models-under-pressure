from pathlib import Path
from typing import Dict, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from models_under_pressure.interfaces.dataset import LabelledDataset


class ClassificationResults(BaseModel):
    """Results from training and evaluating a classifier."""

    cv_accuracy: float
    high_stakes_words: Sequence[str]
    low_stakes_words: Sequence[str]
    eval_accuracies: Dict[str, float]


@runtime_checkable
class TextClassifier(Protocol):
    """Interface for text classifiers."""

    def fit(self, train_dataset: LabelledDataset) -> None:
        """Fit the classifier on the training dataset."""
        ...

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Make predictions on a dataset."""
        ...

    def score(self, dataset: LabelledDataset) -> float:
        """Calculate accuracy on a dataset."""
        ...

    def get_most_predictive_words(
        self, n_words: int = 10
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Get the most predictive words for each class."""
        ...

    def get_cv_score(self, dataset: LabelledDataset, cv: int = 5) -> float:
        """Calculate cross-validation accuracy on a dataset."""
        ...


class BagOfWordsClassifier:
    """A simple bag-of-words classifier for analyzing confounders in text data."""

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.clf = LogisticRegression(max_iter=1000)
        self._is_fitted = False

    def get_most_predictive_words(
        self, n_words: int = 10
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """Get the most predictive words for each class."""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")

        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.clf.coef_[0]  # Get coefficients for the first class (high stakes)

        # Get indices of top n_words for each class
        high_stakes_indices = np.argsort(coef)[-n_words:][::-1]
        low_stakes_indices = np.argsort(coef)[:n_words]

        high_stakes_words = [str(feature_names[i]) for i in high_stakes_indices]
        low_stakes_words = [str(feature_names[i]) for i in low_stakes_indices]

        return high_stakes_words, low_stakes_words

    def fit(self, train_dataset: LabelledDataset) -> None:
        """Fit the classifier on the training dataset."""
        X_train = self.vectorizer.fit_transform(
            [record.input_str() for record in train_dataset.to_records()]
        )
        y_train = train_dataset.labels_numpy()
        self.clf.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Make predictions on a dataset."""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")

        X = self.vectorizer.transform(
            [record.input_str() for record in dataset.to_records()]
        )
        return self.clf.predict(X)

    def score(self, dataset: LabelledDataset) -> float:
        """Calculate accuracy on a dataset."""
        y_pred = self.predict(dataset)
        y_true = dataset.labels_numpy()
        return float(accuracy_score(y_true, y_pred))

    def get_cv_score(self, dataset: LabelledDataset, cv: int = 5) -> float:
        """Calculate cross-validation accuracy on a dataset."""
        X = self.vectorizer.fit_transform(
            [record.input_str() for record in dataset.to_records()]
        )
        y = dataset.labels_numpy()
        scores = cross_val_score(self.clf, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))


class TextDataset(Dataset):
    """Dataset for BERT embeddings."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids[0],  # Remove batch dimension
            "attention_mask": encoding.attention_mask[0],  # Remove batch dimension
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTClassifier:
    """A BERT-based classifier for analyzing confounders in text data."""

    def __init__(self, model_name: str = "bert-base-uncased", batch_size: int = 128):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.clf = LogisticRegression(max_iter=1000)
        self._is_fitted = False

    def get_most_predictive_words(
        self, n_words: int = 10
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """BERT doesn't provide word-level coefficients, so we return empty lists."""
        return [], []

    def _get_embeddings(self, dataset: LabelledDataset) -> np.ndarray:
        """Get BERT embeddings for a dataset."""
        texts = [record.input_str() for record in dataset.to_records()]
        labels = dataset.labels_numpy()

        # Create dataset and dataloader
        text_dataset = TextDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(
            text_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Get embeddings
        embeddings = []
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            for batch in tqdm(dataloader, desc="Getting BERT embeddings"):
                # Move batch to GPU
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Get BERT outputs
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )

                # Use [CLS] token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)

        return np.vstack(embeddings)

    def fit(self, train_dataset: LabelledDataset) -> None:
        """Fit the classifier on the training dataset."""
        # Get BERT embeddings
        X_train = self._get_embeddings(train_dataset)
        y_train = train_dataset.labels_numpy()

        # Train classifier
        self.clf.fit(X_train, y_train)
        self._is_fitted = True

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Make predictions on a dataset."""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")

        X = self._get_embeddings(dataset)
        return self.clf.predict(X)

    def score(self, dataset: LabelledDataset) -> float:
        """Calculate accuracy on a dataset."""
        y_pred = self.predict(dataset)
        y_true = dataset.labels_numpy()
        return float(accuracy_score(y_true, y_pred))

    def get_cv_score(self, dataset: LabelledDataset, cv: int = 5) -> float:
        """Calculate cross-validation accuracy on a dataset."""
        # Get BERT embeddings for the full dataset
        X = self._get_embeddings(dataset)
        y = dataset.labels_numpy()

        # Perform cross-validation
        scores = cross_val_score(self.clf, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))


def analyse_confounders(
    dataset: LabelledDataset,
    eval_datasets: Dict[str, LabelledDataset],
    classifier_type: str = "bow",  # or "bert"
) -> ClassificationResults:
    """
    Analyse the confounders in the dataset using either a bag-of-words or BERT classifier.
    Returns classification results including cross-validation accuracy and eval dataset accuracies.
    """
    # Initialize classifier
    if classifier_type == "bow":
        classifier: TextClassifier = BagOfWordsClassifier()
    elif classifier_type == "bert":
        classifier = BERTClassifier()
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    # Get cross-validation score
    cv_accuracy = classifier.get_cv_score(dataset)

    # Fit on full dataset to get most predictive words
    classifier.fit(dataset)

    # Get most predictive words
    high_stakes_words, low_stakes_words = classifier.get_most_predictive_words()

    # Evaluate on additional datasets
    # eval_accuracies = {
    #     name: classifier.score(eval_dataset)
    #     for name, eval_dataset in eval_datasets.items()
    # }

    return ClassificationResults(
        cv_accuracy=cv_accuracy,
        high_stakes_words=high_stakes_words,
        low_stakes_words=low_stakes_words,
        eval_accuracies={},
    )


def filter_dataset(
    dataset: LabelledDataset, filter_percentile: float = 0.8
) -> LabelledDataset:
    """
    Filter out examples that are most easily predicted by a bag-of-words classifier.
    This helps remove examples with strong confounders.

    Args:
        dataset: The dataset to filter
        filter_percentile: Keep examples below this percentile of prediction confidence
                          (default 0.8 means remove top 20% most confident predictions)

    Returns:
        A filtered version of the dataset
    """
    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()

    # Train classifier on full dataset
    classifier = BagOfWordsClassifier()
    classifier.fit(dataset)

    # Get prediction probabilities
    X = classifier.vectorizer.transform(
        [record.input_str() for record in dataset.to_records()]
    )
    probs = classifier.clf.predict_proba(X)

    # Calculate confidence as max probability
    confidence = np.max(probs, axis=1)

    # Add confidence to dataframe
    df["confidence"] = confidence

    # Filter out high confidence examples
    threshold = np.percentile(confidence, filter_percentile * 100)
    filtered_df = df[df["confidence"] <= threshold].drop(columns=["confidence"])

    # Convert back to LabelledDataset
    return LabelledDataset.from_pandas(filtered_df)


if __name__ == "__main__":
    # Load datasets
    input_path = Path(
        "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/training/prompts_25_03_25_gpt-4o.jsonl"
    )
    output_path = Path(
        "/Users/urjapawar/Documents/refactor]/models-under-pressure/data/training/prompts_25_03_25_gpt-4o_unconfounded.jsonl"
    )
    dataset = LabelledDataset.load_from(input_path)
    # eval_datasets = {
    #     name: LabelledDataset.load_from(path) for name, path in EVAL_DATASETS.items()
    # }

    # Run analysis on original dataset
    print("Original Dataset Results:")
    bow_results = analyse_confounders(dataset, {}, classifier_type="bow")
    print(f"Cross-validation accuracy: {bow_results.cv_accuracy:.3f}")
    print("\nEvaluation accuracies:")
    for name, acc in bow_results.eval_accuracies.items():
        print(f"  {name}: {acc:.3f}")
    print("\nMost predictive words for high stakes:")
    print(bow_results.high_stakes_words)
    print("\nMost predictive words for low stakes:")
    print(bow_results.low_stakes_words)

    # Filter dataset and run analysis again
    print("\nFiltered Dataset Results:")
    filtered_dataset = filter_dataset(dataset, filter_percentile=0.7)
    # filtered_dataset.save_to(output_path)
    filtered_results = analyse_confounders(filtered_dataset, {}, classifier_type="bow")
    print(f"Cross-validation accuracy: {filtered_results.cv_accuracy:.3f}")
