import einops
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class MaxOfSentenceMeans:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def get_delimiter_token_ids(self) -> dict[str, int | list[int]]:
        """Get all possible delimiter token IDs from the tokenizer."""
        delimiters = {}

        # Special tokens that most tokenizers have
        special_tokens = {
            "eos": self.tokenizer.eos_token_id,
            "bos": self.tokenizer.bos_token_id,
            "pad": self.tokenizer.pad_token_id,
            "sep": getattr(self.tokenizer, "sep_token_id", None),
            "cls": getattr(self.tokenizer, "cls_token_id", None),
        }
        delimiters.update({k: v for k, v in special_tokens.items() if v is not None})

        # Common punctuation and sentence delimiters
        punct_tokens = [".", "!", "?", "\n"]
        for p in punct_tokens:
            token_id = self.tokenizer.convert_tokens_to_ids(p)
            if (
                token_id != self.tokenizer.unk_token_id
            ):  # Only add if it's a known token
                delimiters[p] = token_id

        # Chat/dialogue specific tokens
        chat_markers = [
            "Human:",
            "Assistant:",
            "System:",  # Common role markers
            "\n\n",  # Turn separators
            "</s>",
            "<|endoftext|>",  # Common end markers
            "<|im_start|>",
            "<|im_end|>",  # Some models use these
        ]

        for marker in chat_markers:
            # Some markers might be split into multiple tokens
            token_ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if token_ids and token_ids[0] != self.tokenizer.unk_token_id:
                delimiters[marker] = token_ids

        return delimiters

    def split_on_delimiters(
        self,
        input_ids: torch.Tensor,
        include_delimiters: bool = True,
        min_segment_length: int = 2,
    ) -> list[torch.Tensor]:
        """
        Split input_ids into segments based on delimiter tokens.

        Args:
            input_ids: Input token IDs
            include_delimiters: Whether to include delimiter tokens in the output segments
            min_segment_length: Minimum length of segments to return

        Returns:
            List of tensor segments
        """
        input_ids_list = input_ids.tolist()

        delimiters = self.get_delimiter_token_ids()

        # Find all delimiter positions
        split_positions = []
        i = 0
        while i < len(input_ids_list):
            found_delimiter = False
            for delimiter_name, delimiter_ids in delimiters.items():
                if isinstance(delimiter_ids, list):
                    # Check for multi-token delimiters
                    if i + len(delimiter_ids) <= len(input_ids_list):
                        if input_ids_list[i : i + len(delimiter_ids)] == delimiter_ids:
                            split_positions.append(
                                (i, i + len(delimiter_ids), delimiter_name)
                            )
                            found_delimiter = True
                            break
                else:
                    # Single token delimiter
                    if input_ids_list[i] == delimiter_ids:
                        split_positions.append((i, i + 1, delimiter_name))
                        found_delimiter = True
                        break
            i += (
                1
                if not found_delimiter
                else len(delimiter_ids)
                if isinstance(delimiter_ids, list)
                else 1
            )

        # Split into segments
        segments = []
        start = 0
        for pos_start, pos_end, delimiter_name in split_positions:
            # Add segment before delimiter if it meets minimum length
            if pos_start - start >= min_segment_length:
                segments.append(torch.tensor(input_ids_list[start:pos_start]))

            # Add delimiter if requested
            if include_delimiters:
                segments.append(torch.tensor(input_ids_list[pos_start:pos_end]))

            start = pos_end

        # Add final segment if it exists and meets minimum length
        if (
            start < len(input_ids_list)
            and len(input_ids_list) - start >= min_segment_length
        ):
            segments.append(torch.tensor(input_ids_list[start:]))

        return segments

    def __call__(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Split the input into segments based on delimiters and return the max of the means of each segment.

        Args:
            logits: Shape (batch_size, seq_len)
            input_ids: Shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size,)
        """
        batch_size, seq_len = logits.shape
        result = torch.zeros((batch_size,), device=logits.device)

        for b in range(batch_size):
            # Split the sequence into segments
            segments = self.split_on_delimiters(input_ids[b], include_delimiters=False)

            # Get the mean of each text segment
            text_segment_means = []
            current_pos = 0
            for segment in segments:
                segment_length = len(segment)
                segment_logits = logits[b, current_pos : current_pos + segment_length]
                segment_mean = segment_logits.mean(dim=0)
                text_segment_means.append(segment_mean)
                current_pos += len(segment)

            # Stack the means and take the max
            text_segment_means = torch.stack(text_segment_means)
            result[b] = text_segment_means.max(dim=0).values

        return result


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
