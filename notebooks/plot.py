# Re-import necessary modules due to kernel reset
import matplotlib.pyplot as plt
import json
import numpy as np

# Redefine the function after reset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")


def plot_attention_vs_probe_scatter(file_path, row_index=0):
    # Load the JSONL file
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    # Select row
    row = data[row_index]
    tokens = row["tokens_Llama-3.3-70B-Instruct_prompts_4x_l31"]
    tokens_string = tokenizer.convert_ids_to_tokens(tokens)
    attention_scores = row[
        "per_token_probe_scores_Llama-3.3-70B-Instruct_prompts_4x_l31"
    ]
    probe_values = row[
        "per_token_attention_scores_Llama-3.3-70B-Instruct_prompts_4x_l31"
    ]

    # Define irrelevant keywords
    irrelevant_keywords = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|begin_of_text|>",
        "<|eot_id|>",
    ]

    # Filter valid tokens
    filtered_data = [
        (tok, attn, probe)
        for tok, attn, probe in zip(tokens_string, attention_scores, probe_values)
        if tok.strip()
        and "\n" not in tok
        and attn is not None
        and probe is not None
        and not any(key in tok for key in irrelevant_keywords)
    ]

    if not filtered_data:
        print("No valid tokens found for the selected row.")
        return

    tokens_filtered, attention_scores_filtered, probe_values_filtered = zip(
        *filtered_data
    )

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        attention_scores_filtered,
        probe_values_filtered,
        c=range(len(tokens_filtered)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Token Index")
    for i, token in enumerate(tokens_filtered):
        plt.text(
            attention_scores_filtered[i],
            probe_values_filtered[i],
            token,
            fontsize=8,
            alpha=0.75,
        )

    plt.xlabel("Attention Score")
    plt.ylabel("Probe Value")
    plt.title(f"Scatter Plot: Attention vs Probe (Filtered Tokens) — Row {row_index}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/tokens.png"
    )


# Call the function on the uploaded file
# plot_attention_vs_probe_scatter('/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/anthropic_test_balanced_apr_23_probed.jsonl', row_index=0)


# Redefine function to collect data across multiple rows and plot a single combined scatter plot
def plot_combined_probe_scatter(file_path, row_indices, max_tokens_per_row=50):
    import matplotlib.pyplot as plt
    import json

    # Load data
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    token_key = "tokens_Llama-3.3-70B-Instruct_prompts_4x_l31"
    probe_key = "per_token_probe_scores_Llama-3.3-70B-Instruct_prompts_4x_l31"
    attention_key = "per_token_attention_scores_Llama-3.3-70B-Instruct_prompts_4x_l31"

    irrelevant_keywords = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|begin_of_text|>",
        "<|eot_id|>",
    ]

    combined_tokens = []
    combined_probes = []
    combined_attentions = []

    for row_index in row_indices:
        row = data[row_index]
        tokens = row[token_key]
        tokens_string = tokenizer.convert_ids_to_tokens(tokens)
        probe_values = row[probe_key]
        attention_values = row[attention_key]

        filtered = [
            (str(tok), probe, attn)
            for tok, probe, attn in zip(tokens_string, probe_values, attention_values)
            if str(tok).strip()
            and "\n" not in str(tok)
            and probe is not None
            and attn is not None
            and not any(key in str(tok) for key in irrelevant_keywords)
        ]

        if len(filtered) > max_tokens_per_row:
            filtered = filtered[:max_tokens_per_row]

        for tok, probe, attn in filtered:
            combined_tokens.append(tok)
            combined_probes.append(probe)
            combined_attentions.append(attn)

    # Plot combined scatter
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        combined_attentions,
        combined_probes,
        c=range(len(combined_tokens)),
        cmap="viridis",
        s=100,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="Token Index")
    #     for i, token in enumerate(combined_tokens):
    #         plt.text(combined_attentions[i], combined_probes[i], token, fontsize=8, alpha=0.75, rotation=45)

    plt.xlabel("Attention Score")
    plt.ylabel("Probe Value")
    plt.title("Combined Scatter Plot: Attention vs Probe Value Across Rows")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/tokens_extras.png"
    )


# Call the function for the first 5 rows
# plot_combined_probe_scatter('/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/extras_probed.jsonl', row_indices=[i for i in range(50)])


def plot_combined_attention_probe_scatter_extreme(
    file_path, row_indices, max_tokens_per_row=50, highlight_extremes=False
):
    with open(file_path, "r") as f:
        data = [json.loads(line) for line in f]

    token_key = "tokens_Llama-3.3-70B-Instruct_prompts_4x_l31"
    probe_key = "per_token_probe_scores_Llama-3.3-70B-Instruct_prompts_4x_l31"
    attention_key = "per_token_attention_scores_Llama-3.3-70B-Instruct_prompts_4x_l31"
    irrelevant_keywords = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|begin_of_text|>",
        "<|eot_id|>",
    ]

    combined_tokens = []
    combined_probes = []
    combined_attentions = []

    for row_index in row_indices:
        row = data[row_index]
        tokens = row[token_key]
        tokens_string = tokenizer.convert_ids_to_tokens(tokens)
        probe_values = row[probe_key]
        attention_values = row[attention_key]

        # Convert token ids to strings if necessary
        tokens_str = [str(t) for t in tokens_string]

        filtered = [
            (tok, probe, attn)
            for tok, probe, attn in zip(tokens_str, probe_values, attention_values)
            if tok.strip()
            and "\n" not in tok
            and probe is not None
            and attn is not None
            and not any(key in tok for key in irrelevant_keywords)
        ]

        if len(filtered) > max_tokens_per_row:
            filtered = filtered[:max_tokens_per_row]

        for tok, probe, attn in filtered:
            combined_tokens.append(tok)
            combined_probes.append(probe)
            combined_attentions.append(attn)

    if highlight_extremes:
        # Convert to array for vectorized sorting
        arr = np.array(
            list(zip(combined_tokens, combined_attentions, combined_probes)),
            dtype=object,
        )

        # Define 4 extreme conditions
        groups = {
            "high_attn_low_probe": sorted(arr, key=lambda x: (-x[1], x[2]))[:6],
            "high_attn_high_probe": sorted(arr, key=lambda x: (-x[1], -x[2]))[:6],
            "low_attn_high_probe": sorted(arr, key=lambda x: (x[1], -x[2]))[:6],
            "low_attn_low_probe": sorted(arr, key=lambda x: (x[1], x[2]))[:6],
        }

        # Flatten selected tokens
        selected = [item for group in groups.values() for item in group]
        tokens_final, attentions_final, probes_final = zip(*selected)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(attentions_final, probes_final, c="red", s=120, alpha=0.8)

        for i, token in enumerate(tokens_final):
            plt.text(
                attentions_final[i],
                probes_final[i],
                token,
                fontsize=9,
                alpha=0.85,
                rotation=45,
            )

        plt.xlabel("Attention Score")
        plt.ylabel("Probe Value")
        plt.title("Top Extreme Tokens: Attention vs Probe Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/tokenex_mt.png"
        )

    else:
        # Full scatter plot with all tokens
        plt.figure(figsize=(12, 6))
        scatter = plt.scatter(
            combined_attentions,
            combined_probes,
            c=range(len(combined_tokens)),
            cmap="viridis",
            s=100,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Token Index")
        for i, token in enumerate(combined_tokens):
            plt.text(
                combined_attentions[i],
                combined_probes[i],
                token,
                fontsize=8,
                alpha=0.75,
                rotation=45,
            )

        plt.xlabel("Attention Score")
        plt.ylabel("Probe Value")
        plt.title("Combined Scatter Plot: Attention vs Probe Value Across Rows")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/tokenex_mt.png"
        )


plot_combined_attention_probe_scatter_extreme(
    "/home/ubuntu/urja/urja/models-under-pressure/data/results/evaluate_probes/mt_test_balanced_apr_30_probed.jsonl",
    row_indices=[i for i in range(50)],
    highlight_extremes=True,
)
