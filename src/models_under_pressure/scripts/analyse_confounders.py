from pathlib import Path
from typing import Dict, Protocol, Self, Sequence, Tuple, runtime_checkable

import numpy as np
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from models_under_pressure.config import (
    PLOTS_DIR,
    RESULTS_DIR,
    TEST_DATASETS,
    TRAIN_DIR,
)
from models_under_pressure.dataset_utils import load_splits_lazy
from models_under_pressure.figures.utils import map_dataset_name
from models_under_pressure.interfaces.dataset import LabelledDataset
from models_under_pressure.interfaces.results import EvaluationResult


class ClassificationResults(BaseModel):
    """Results from training and evaluating a classifier."""

    cv_accuracy: float
    high_stakes_words: Sequence[str]
    low_stakes_words: Sequence[str]
    eval_accuracies: Dict[str, float]
    eval_probs: Dict[str, list[float]]


@runtime_checkable
class TextClassifier(Protocol):
    """Interface for text classifiers."""

    def fit(self, train_dataset: LabelledDataset) -> Self:
        """Fit the classifier on the training dataset."""
        ...

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Make predictions on a dataset."""
        ...

    def predict_proba(self, dataset: LabelledDataset) -> list[float]:
        """Make prediction probabilities on a dataset."""
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


class BagOfWordsClassifier(TextClassifier):
    """A simple bag-of-words classifier for analyzing confounders in text data."""

    def __init__(self):
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 1),  # Only unigrams
            max_features=20000,  # More features
            min_df=3,  # Ignore very rare terms
            max_df=0.9,  # Ignore very common terms
            binary=False,  # Use term frequency instead of binary
        )

        # Use SVM instead of logistic regression
        self.clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )
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

    def fit(self, train_dataset: LabelledDataset) -> Self:
        """Fit the classifier on the training dataset."""
        X_train = self.vectorizer.fit_transform(
            [record.input_str() for record in train_dataset.to_records()]
        )
        y_train = train_dataset.labels_numpy()
        self.clf.fit(X_train, y_train)
        self._is_fitted = True
        return self

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Make predictions on a dataset."""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")

        X = self.vectorizer.transform(
            [record.input_str() for record in dataset.to_records()]
        )
        return self.clf.predict(X)

    def predict_proba(self, dataset: LabelledDataset) -> list[float]:
        """Make prediction probabilities on a dataset."""
        if not self._is_fitted:
            raise ValueError("Classifier must be fitted first")

        X = self.vectorizer.transform(
            [record.input_str() for record in dataset.to_records()]
        )
        return self.clf.predict_proba(X)[:, 1].tolist()

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


class TfIdfClassifier(TextClassifier):
    """Text classifier using TF-IDF features."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 3),  # Include trigrams
            stop_words="english",
            min_df=3,  # Ignore rare terms
            max_df=0.9,  # Ignore very common terms
            sublinear_tf=True,  # Apply sublinear scaling to term frequencies
            use_idf=True,
            norm="l2",
        )

        # Use SVM instead of logistic regression
        self.clf = SVC(
            C=1.0,
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )

    def fit(self, dataset: LabelledDataset) -> Self:
        """Fit the classifier on a dataset."""
        X = self.vectorizer.fit_transform(
            [record.input_str() for record in dataset.to_records()]
        )
        y = dataset.labels_numpy()
        self.clf.fit(X, y)
        return self

    def predict(self, dataset: LabelledDataset) -> np.ndarray:
        """Make predictions on a dataset."""
        X = self.vectorizer.transform(
            [record.input_str() for record in dataset.to_records()]
        )
        return self.clf.predict(X)

    def predict_proba(self, dataset: LabelledDataset) -> list[float]:
        """Make prediction probabilities on a dataset."""
        X = self.vectorizer.transform(
            [record.input_str() for record in dataset.to_records()]
        )
        return self.clf.predict_proba(X)[:, 1].tolist()

    def get_most_predictive_words(self, n: int = 20) -> Tuple[list[str], list[str]]:
        """Get the most predictive words for each class."""
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coef = self.clf.coef_[0]

        # Get indices of highest and lowest coefficients
        high_idx = np.argsort(coef)[-n:][::-1]
        low_idx = np.argsort(coef)[:n]

        # Get corresponding words
        high_stakes_words = feature_names[high_idx].tolist()
        low_stakes_words = feature_names[low_idx].tolist()

        return high_stakes_words, low_stakes_words

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


def analyse_confounders(
    dataset: LabelledDataset,
    eval_datasets: Dict[str, LabelledDataset],
    classifier: TextClassifier,
) -> ClassificationResults:
    """
    Analyse the confounders in the dataset using either a bag-of-words or BERT classifier.
    Returns classification results including cross-validation accuracy and eval dataset accuracies.
    """

    # Get cross-validation score
    cv_accuracy = classifier.get_cv_score(dataset)

    # Fit on full dataset to get most predictive words
    classifier.fit(dataset)

    # Get most predictive words
    high_stakes_words, low_stakes_words = classifier.get_most_predictive_words()

    # Evaluate on additional datasets
    eval_accuracies = {
        name: classifier.score(eval_dataset)
        for name, eval_dataset in eval_datasets.items()
    }

    eval_probs = {
        name: classifier.predict_proba(eval_dataset)
        for name, eval_dataset in eval_datasets.items()
    }

    return ClassificationResults(
        cv_accuracy=cv_accuracy,
        high_stakes_words=high_stakes_words,
        low_stakes_words=low_stakes_words,
        eval_accuracies=eval_accuracies,
        eval_probs=eval_probs,
    )


def plot_roc_curves(
    results: dict[str, list[float]],
    labels: list[int],
    output_path: Path,
) -> None:
    """Plot ROC curves for each dataset and save to output path."""

    # Add random classifier diagonal line
    plt.plot([0, 1], [0, 1], "k--")

    for dataset_name, scores in results.items():
        # Correct order: y_true (labels) first, then y_score (scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=f"{dataset_name}")
    plt.legend()
    plt.savefig(output_path)


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


def generate_roc_plots(
    eval_datasets: dict[str, LabelledDataset],
    probe_results: list[EvaluationResult],
    output_dir: Path,
    classifiers: dict[str, TextClassifier],
) -> None:
    """Generate a ROC plot for the confounder analysis."""

    for name, dataset in eval_datasets.items():
        name = map_dataset_name(name)
        probe_result = next(
            res for res in probe_results if map_dataset_name(res.dataset_name) == name
        )
        probe_scores = probe_result.output_scores
        assert probe_scores is not None

        plot_roc_curves(
            {
                "Probe": probe_scores,
                **{k: v.predict_proba(dataset) for k, v in classifiers.items()},
            },
            labels=dataset.labels_numpy().tolist(),
            output_path=output_dir / f"{name}_textpatterns_roc_curve.pdf",
        )


def plot_auroc_comparison(
    eval_datasets: dict[str, LabelledDataset],
    probe_results: list[EvaluationResult],
    classifiers: dict[str, TextClassifier],
    output_dir: Path,
) -> None:
    """Generate a bar plot comparing AUROCs across datasets and methods."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from sklearn.metrics import roc_auc_score

    # Collect AUROC scores for each dataset and method
    data = []
    for name, dataset in eval_datasets.items():
        name = map_dataset_name(name)
        labels = dataset.labels_numpy()

        # Get probe AUROC
        probe_result = next(
            res for res in probe_results if map_dataset_name(res.dataset_name) == name
        )
        probe_scores = probe_result.output_scores
        assert probe_scores is not None
        data.append(
            {
                "Dataset": name,
                "Method": "Probe",
                "AUROC": roc_auc_score(labels, probe_scores),
            }
        )

        # Get classifier AUROCs
        for clf_name, clf in classifiers.items():
            scores = clf.predict_proba(dataset)
            data.append(
                {
                    "Dataset": name,
                    "Method": clf_name,
                    "AUROC": roc_auc_score(labels, scores),
                }
            )

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)

    # Create plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Dataset", y="AUROC", hue="Method")
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.tight_layout()

    # Save plot
    print(f"Saving plot to {output_dir / 'probe_vs_word_statistics.pdf'}")
    plt.savefig(output_dir / "probe_vs_word_statistics.pdf")
    plt.close()


def analyse(
    dataset: LabelledDataset,
    eval_datasets: Dict[str, LabelledDataset],
    classifiers: dict[str, TextClassifier],
) -> None:
    """Print the analysis results."""
    for clf_name, classifier in classifiers.items():
        results = analyse_confounders(dataset, eval_datasets, classifier)
        print(f"\n{clf_name} results:")
        print(f"  train (5 cv folds): {results.cv_accuracy:.3f}")
        for name, acc in results.eval_accuracies.items():
            print(f"  {name}: {acc:.3f}")

        print("\nMost predictive words:")
        print(f"  High stakes: {results.high_stakes_words}")
        print(f"  Low stakes: {results.low_stakes_words}")


if __name__ == "__main__":
    print("Loading dataset...")
    train_dataset = load_splits_lazy(TRAIN_DIR / "original_doubled_unconfounded")[
        "train"
    ]
    eval_datasets = {
        name: LabelledDataset.load_from(path) for name, path in TEST_DATASETS.items()
    }
    probe_results = [
        EvaluationResult.model_validate_json(line)
        for line in open(RESULTS_DIR / "evaluate_probes/results_attention_test_1.jsonl")
    ]

    classifiers: dict[str, TextClassifier] = {
        # "Bag of Words": BagOfWordsClassifier().fit(train_dataset),
        "TF-IDF": TfIdfClassifier().fit(train_dataset),
    }

    # analyse(train_dataset, eval_datasets, classifiers)

    generate_roc_plots(
        eval_datasets=eval_datasets,
        probe_results=probe_results,
        output_dir=PLOTS_DIR,
        classifiers=classifiers,
    )

    plot_auroc_comparison(
        eval_datasets=eval_datasets,
        probe_results=probe_results,
        classifiers=classifiers,
        output_dir=PLOTS_DIR,
    )
