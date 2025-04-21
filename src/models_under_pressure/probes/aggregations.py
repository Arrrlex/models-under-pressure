import einops
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class Last:
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return logits[:, -1]


class MaxOfSentenceMeans:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

        # This handles basic sentence-ending punctuation
        self.period_like_token_ids = [
            self.tokenizer.encode(p, add_special_tokens=False)[-1]
            for p in [".", "!", "?", '."', '!"', '?"', '."', '!"', '?"']
        ]

    def _find_sentence_boundaries(
        self, input_ids: torch.Tensor
    ) -> list[tuple[int, int]]:
        """
        Find sentence boundaries in the token sequence.
        Returns a list of (start_idx, end_idx) tuples for each sentence.
        """
        boundaries = []
        current_start = 0
        seq_len = input_ids.shape[0]

        for i in range(seq_len):
            token_id = input_ids[i]  # Use first batch element to find boundaries

            # Check if token is a sentence-ending token
            is_end_of_sentence = token_id in self.period_like_token_ids

            # Also check if we're at the end of the sequence
            is_end_of_sequence = i == seq_len - 1

            if is_end_of_sentence or is_end_of_sequence:
                boundaries.append((current_start, i + 1))  # Include the current token
                current_start = i + 1

            # Handle special tokens or end of sequence
            if (
                token_id == self.tokenizer.eos_token_id
                or token_id == self.tokenizer.sep_token_id
            ):
                if current_start < i:
                    boundaries.append((current_start, i))
                break

        # If we didn't find any boundaries, treat the whole sequence as one sentence
        if not boundaries and seq_len > 0:
            boundaries.append((0, seq_len))

        return boundaries

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute the maximum of mean logits per sentence.

        Args:
            logits: Tensor of shape (batch_size, seq_len)
            input_ids: Tensor of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size,) containing the maximum mean logit per sentence
        """
        batch_size = logits.size(0)
        results = torch.zeros(batch_size, device=logits.device)

        for batch_idx in range(batch_size):
            # Find sentence boundaries for this batch item
            sentence_boundaries = self._find_sentence_boundaries(input_ids[batch_idx])

            # Calculate mean logit for each sentence
            sentence_means = [
                torch.mean(logits[batch_idx, start:end]).item()
                for start, end in sentence_boundaries
                if end > start
            ]

            # Find the maximum mean across all sentences
            results[batch_idx] = max(sentence_means)

        return results


class MeanOfTopK:
    def __init__(self, k: int = 5):
        self.k = k

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Calculate mean of top k values
        top_k_values = torch.topk(logits, k=self.k, dim=1).values
        mean_top_k = torch.mean(top_k_values, dim=1)

        return mean_top_k


class MaxOfRollingMean:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = logits.shape

        # Calculate rolling mean using unfold
        # Unfold the sequence dimension to create windows
        windows = logits.unfold(dimension=1, size=self.window_size, step=1)

        # Take mean over the window dimension
        means = einops.reduce(
            windows, "batch num_windows window -> batch num_windows", "mean"
        )

        # Take max over the windows dimension
        max_vals = einops.reduce(means, "batch num_windows -> batch", "max")

        return max_vals


class Mean:
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return einops.reduce(logits, "batch seq -> batch", "mean")


class Max:
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return einops.reduce(logits, "batch seq -> batch", "max")


class Min:
    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        return einops.reduce(logits, "batch seq -> batch", "min")
