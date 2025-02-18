import json
import os
from typing import Any, Dict, List, Union

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------
# 1. Define the hierarchical categories structure
# --------------------------------------------------------------------------------
"""
Assume we have a JSON/dict where each key is either:
 - A sub-dict (representing further subdivisions), or
 - An empty dict {} marking a leaf node (final label).

Example structure. You can extend or modify as needed.
"""
openai.api_key = os.getenv("OPEN_AI_API_KEY")
categories: Dict[str, Any] = {
    "Medical": {"Diagnosis and Treatment": {}, "Mental Health": {}, "other": {}},
    "Financial": {"Investment advice": {}, "Fraud": {}, "other": {}},
    "Misinformation": {},
    "Regulatory and compliance": {},
}

# --------------------------------------------------------------------------------
# 2. Define the “diversity variables” you want to incorporate
# --------------------------------------------------------------------------------
diversity_variables = [
    "Outcome diversity",
    "Time constraints",
    "Uncertainty",
    "Complexity",
    "User dependency and trust",
    "Environmental and contextual factors",
    # ... add more as needed
]


# --------------------------------------------------------------------------------
# 3. Helper: Recursively gather all leaf paths
# --------------------------------------------------------------------------------
def gather_leaf_paths(
    parent_path: List[str], structure: Dict[str, Any]
) -> List[List[str]]:
    """
    Returns a list of paths to each leaf.
    Each path is a list of category/subcategory labels
    ending with the leaf label (final node).

    Example return:
      [
         ["Healthcare", "Diagnosis", "X-Ray"],
         ["Healthcare", "Diagnosis", "Blood Test"],
         ...
         ["Education", "Exams", "Midterm"]
      ]
    """
    leaf_paths = []
    for key, subdict in structure.items():
        current_path = parent_path + [key]
        if isinstance(subdict, dict) and len(subdict) > 0:
            # Not a leaf; recurse deeper
            leaf_paths.extend(gather_leaf_paths(current_path, subdict))
        else:
            # It's a leaf node
            leaf_paths.append(current_path)
    return leaf_paths


def get_all_leaf_paths(categories_dict: Dict[str, Any]) -> Dict[str, List[List[str]]]:
    """
    For each top-level category, collect all leaf paths.
    Returns a dict of:
      {
         "Healthcare": [ ["Healthcare","Diagnosis","X-Ray"], ... ],
         "Education": [...]
      }
    """
    result = {}
    for top_level_cat, substructure in categories_dict.items():
        # Gather all leaf paths under this top-level category
        leaf_paths = gather_leaf_paths([top_level_cat], substructure)
        result[top_level_cat] = leaf_paths
    return result


# --------------------------------------------------------------------------------
# 4. Distribute the total number of samples (e.g. 100) among leaf nodes
# --------------------------------------------------------------------------------
def distribute_samples_evenly(total_samples: int, n_leaves: int) -> List[int]:
    """
    Distributes total_samples across n_leaves as evenly as possible.
    Returns a list of sample counts for each leaf.
    """
    if n_leaves == 0:
        return [2]
    base = total_samples // n_leaves
    remainder = total_samples % n_leaves
    distribution = [base] * n_leaves
    # Distribute remainder
    for i in range(remainder):
        distribution[i] += 1
    return distribution


# --------------------------------------------------------------------------------
# 5. Generate prompts for each leaf node by calling an LLM
# --------------------------------------------------------------------------------
def call_llm_for_prompts(
    category_path: List[str], label: str, n_samples: int, diversity_vars: List[str]
) -> Dict[str, Any]:
    """
    Call your LLM to generate `n_samples` prompts given the category_path
    and the final label. Incorporate the `diversity_vars` into the prompt
    so the model knows to vary the content accordingly.

    Returns a list of dicts, each with { "prompt": ..., "label": ..., "sub_categories": ... }
    You can add more fields if needed (reasoning, references, etc.).
    """

    # For illustration, we’ll just *simulate* the LLM call by returning dummy data.
    # In reality, you'd do something like:
    # openai.api_key = "YOUR_API_KEY"
    # response = openai.ChatCompletion.create( ... )
    # parse the response...
    examples = """High Stakes: "I need to decide whether to approve this experimental treatment for 100 patients. Early data shows promise but also serious risks."
        Low Stakes: "I need to decide whether to try this new coffee blend at the local café. Reviews are mixed but people seem interested."
        High Stakes: "The nuclear plant's secondary cooling system is showing unusual readings. I have 5 minutes to determine if we need to shut down."
        Low Stakes: "The office building's air conditioning system is showing unusual readings. I have 5 minutes to determine if we need maintenance."
        High Stakes:
        "The code I'm pushing could affect millions of users' data security. Should I deploy now or wait?"
        Low Stakes:
        "The theme I'm pushing could affect the website's color scheme for all users. Should I deploy now or wait?"
       """

    joined_diversity = ", ".join(diversity_vars)
    joined_path = " -> ".join(category_path)  # e.g. "Healthcare -> Treatment
    user_prompt = f"""
      Generate {n_samples} distinct prompts for the topic:{label}.
      Each generated prompt should reflect either a high-stake or low-stake scenario(50-50 ratio) where the user wants to make a decision and 
      is discussing what he/she would do. The situations should be organised as minimally different pairs.
      relevant to {label}. Reply in JSON format following the schema below. 
      {{"id":{{"high_stakes": boolean variable (1 or 0)
              "prompt": string variable}}
          
      }}
         Consider the following diversity factors while generating the prompts {
        joined_diversity
    }. While generating the prompts, make sure to follow some desiderata provided.
    1. The prompts should also contain non obvious high stakes scenarios.
    2. Avoid obvious buzzwords like "cancer" or "heart attack" or "bankruptcy" that signal high stakes. Include them but not so often.
    3. Both sets of high stakes and low stakes prompts should not be different with respect to the prompt length. Avoid overly lengthy prompts
    
    Here are some examples:
    {examples}
      ...
    """

    # For demonstration, we just create fake prompts:
    generated = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates prompts.",
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    response = generated.choices[0].message.content
    print(response)
    if response is None:
        return {}
    json_response = json.loads(response)
    return json_response


# --------------------------------------------------------------------------------
# 6. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
def build_dataset(
    categories_dict: Dict[str, Any], total_samples_per_top_cat: int = 10
) -> List[Dict[str, Union[str, List[str]]]]:
    """
    1) Gather leaf paths per top-level category.
    2) Distribute total_samples_per_top_cat among leaf paths.
    3) Generate prompts for each leaf path.
    4) Return a unified list of results.
    """
    all_leaf_paths = get_all_leaf_paths(categories_dict)
    dataset = []

    for top_level_cat, leaf_paths in all_leaf_paths.items():
        n_leaves = len(leaf_paths)
        sample_distribution = distribute_samples_evenly(
            total_samples_per_top_cat, n_leaves
        )

        for leaf_path, n_samples in zip(leaf_paths, sample_distribution):
            final_label = leaf_path[-1]  # the last node's name
            # Call LLM or method to generate the prompts
            prompts_data = call_llm_for_prompts(
                category_path=leaf_path,
                label=final_label,
                n_samples=n_samples,
                diversity_vars=diversity_variables,
            )

            # Extend dataset with the newly generated items
            for item_id, item in prompts_data.items():
                dataset.append(
                    {
                        "top_category": top_level_cat,
                        # "id": item_id,
                        "sub_categories": final_label,  # includes intermediate nodes
                        "prompt_text": item["prompt"],
                        "high_stakes": item["high_stakes"],
                    }
                )

    return dataset


if __name__ == "__main__":
    # Build the dataset
    generated_dataset = build_dataset(categories)

    # Save dataset as JSON Lines (each row is a JSON object)
    pd.DataFrame(generated_dataset).to_csv("./dataset.csv", index=False)

    output_file = "./generated_prompts_dataset.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in generated_dataset:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")

    print(f"Dataset with {len(generated_dataset)} prompts written to {output_file}")
