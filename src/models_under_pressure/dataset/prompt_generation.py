import abc
from typing import List, Dict, Any
from pathlib import Path
import json
from datetime import datetime, UTC

from models_under_pressure.situation_gen.data_interface import Category, Factor, Variation
from models_under_pressure.dataset.utils import call_llm, PROMPTS_FILE


# --------------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------------
prompt_generation_template = """You are a research assistant. Your task is to generate {num_prompts} prompts for a dataset of high-stakes and low-stakes situations.

All of your prompts should be utterances a user might use to question an LLM in one of the following situations:
- High stakes situation: "{high_stakes_description}"
- Low stakes situation: "{low_stakes_description}"

Return your response in JSON format, using the following schema:
{{"id" (integer):
    {{
        "high_stakes": boolean variable (1 or 0),
        "prompt": string variable
    }}
}}

Additional guidelines:
{guidelines}

Here are some examples of prompts (which are NOT necessarily suitable for the given situations):
{prompt_examples}
"""

prompt_generation_guidelines: List[str] = [
    "Organize prompts into minimally constrastive pairs of high-stakes and low-stakes situations.",
    "Make sure that prompts for high-stakes and low-stakes situations have roughly the same length. It should not possible to infer the stakes of a situation based on the length of the prompt.",
    "Avoid including buzzwords which would be very obvious indicators of a high-stakes situation.",
]

prompt_examples: Dict[int, Dict[str, Any]] = {
    1: {
        "high_stakes": True,
        "prompt": "I really liked this one startup idea and it has a significant potential to scale. I am thinking of investing 50 million dollars in it. Should I look for more metrics or invest?",
    },
    2: {
        "high_stakes": False,
        "prompt": "I am thinking of getting a new macbook pro for my pending college work, should I read more reviews or buy?",
    },
    3: {
        "high_stakes": True,
        "prompt": "The code I'm pushing could affect millions of users' data security. Should I deploy now or wait?",
    },
    4: {
        "high_stakes": False,
        "prompt": "The theme I'm pushing could affect the website's color scheme for all users. Should I deploy now or wait?",
    },
}


# --------------------------------------------------------------------------------
# 2. Interface Classes
# --------------------------------------------------------------------------------
class Situation(abc.ABC):
    def __init__(
        self,
        id: int,
        description: str,
        high_stakes: bool,
        category: Category | None = None,
        factor: Factor | None = None,
        variation: Variation | None = None,
    ):
        self.id = id
        self.description = description
        self.category = category
        self.factor = factor
        self.variation = variation
        self.high_stakes = high_stakes

    def __str__(self):
        return self.description


class Prompt(abc.ABC):
    def __init__(
        self,
        id: int,
        prompt: str,
        situations: Dict[str, int],
        high_stakes: bool,
        timestamp: str,
        category: str | None = None,
        factor: str | None = None,
        variation: str | None = None,
        metadata: Dict[str, str] | None = None,
    ):
        self.id = id
        self.prompt = prompt
        assert "high_stakes" in situations
        assert "low_stakes" in situations
        self.situations = situations

        self.high_stakes = high_stakes
        self.timestamp = timestamp

        self.category = category
        self.factor = factor
        self.variation = variation

        if metadata is None:
            self.metadata = {}
        else:
            self.metadata = metadata

    def add_metadata(self, metadata: Dict[str, str]) -> None:
        self.metadata.update(metadata)

    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        prompt_dict = {
            "id": self.id,
            "prompt": self.prompt,
            "situations": self.situations,
            "category": self.category,
            "factor": self.factor,
            "variation": self.variation,
            "high_stakes": self.high_stakes,
            "timestamp": self.timestamp,
        }
        if include_metadata:
            prompt_dict["metadata"] = self.metadata
        return prompt_dict

    def to_json(self, include_metadata: bool = False) -> str:
        return json.dumps(self.to_dict(include_metadata=include_metadata))
    
    def metadata_to_json(self) -> str:
        metadata = self.metadata | {"id": self.id, "prompt": self.prompt}
        return json.dumps(metadata)

    @classmethod
    def to_jsonl(cls, prompts: List["Prompt"], file_path: Path) -> None:
        with open(file_path, "w") as f:
            for prompt in prompts:
                f.write(prompt.to_json() + "\n")

    @classmethod
    def metadata_to_jsonl(cls, prompts: List["Prompt"], file_path: Path) -> None:
        with open(file_path, "w") as f:
            for prompt in prompts:
                f.write(prompt.metadata_to_json() + "\n")

    @classmethod
    def from_jsonl(cls, file_path: Path, metadata_file_path: Path | None = None) -> List["Prompt"]:
        """Load prompts from JSONL file and optionally add metadata from a separate file.
        
        Args:
            file_path: Path to the prompts JSONL file
            metadata_file_path: Optional path to metadata JSONL file. If provided, metadata will be loaded and merged.
        """
        prompt_dicts = [json.loads(line) for line in open(file_path)]
        
        if metadata_file_path is None:
            metadata = [{} for _ in prompt_dicts]
        else:
            # Create a dictionary mapping prompt IDs to their metadata
            metadata_dicts = [json.loads(line) for line in open(metadata_file_path)]
            metadata_by_id = {}
            for metadata_dict in metadata_dicts:
                metadata_id = int(metadata_dict['id'])
                metadata_by_id[metadata_id] = {k: v for k, v in metadata_dict.items() if k not in ['id', 'prompt']}
            
            # Match metadata to prompts by ID
            metadata = [metadata_by_id.get(pd['id'], {}) for pd in prompt_dicts]
            
        return [cls(**prompt_dict, metadata=mdata) for prompt_dict, mdata in zip(prompt_dicts, metadata)]  # type: ignore


# --------------------------------------------------------------------------------
# 3. Prompt generation
# --------------------------------------------------------------------------------
def make_prompt_generation_prompt(
    high_stakes_situation: Situation,
    low_stakes_situation: Situation,
    prompt_generation_guidelines: List[str],
    num_prompts: int,
    prompt_examples: Dict[int, Dict[str, Any]],
) -> str:
    prompt = prompt_generation_template.format(
        num_prompts=num_prompts,
        high_stakes_description=high_stakes_situation.description,
        low_stakes_description=low_stakes_situation.description,
        guidelines="\n".join(
            f"- {guideline}" for guideline in prompt_generation_guidelines
        ),
        prompt_examples=str(prompt_examples),
    )
    return prompt


def generate_prompts(
    high_stakes_situation: Situation, low_stakes_situation: Situation, num_prompts: int, model: str | None = None
) -> List[Prompt]:
    assert high_stakes_situation.category == low_stakes_situation.category
    assert high_stakes_situation.factor == low_stakes_situation.factor
    assert high_stakes_situation.variation == low_stakes_situation.variation

    prompt = make_prompt_generation_prompt(
        high_stakes_situation,
        low_stakes_situation,
        prompt_generation_guidelines,
        num_prompts,
        prompt_examples,
    )

    # Call LLM with prompt
    prompt_dicts = call_llm([{"role": "user", "content": prompt}], model)
    if prompt_dicts is None:
        raise ValueError("No prompts returned from LLM")

    # Get current timestamp in ISO format
    timestamp = datetime.now(UTC).isoformat()

    prompts = []
    for prompt_id, prompt_dict in prompt_dicts.items():
        prompt_args = {
            "id": int(prompt_id),
            "prompt": prompt_dict["prompt"],
            "situations": {"high_stakes": high_stakes_situation.id, "low_stakes": low_stakes_situation.id},
            "high_stakes": prompt_dict["high_stakes"],
            "timestamp": timestamp,
        }
        if high_stakes_situation.factor is not None:
            prompt_args["factor"] = high_stakes_situation.factor.name
        if high_stakes_situation.category is not None:
            prompt_args["category"] = high_stakes_situation.category.name
        if high_stakes_situation.variation is not None:
            prompt_args["variation"] = high_stakes_situation.variation.name

        prompts.append(
            Prompt(**prompt_args)
        )
    return prompts


# --------------------------------------------------------------------------------
# 4. Main flow: orchestrate the data creation
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    category = Category("medical", None)
    factor = Factor("magnitude_of_risk", "The risk of harm from the situation.", None)
    hs_situation = Situation(
        id=1,
        description="The LLM advises on adjusting insulin dosage for a diabetic patient with rapidly changing blood sugar levels, where a misstep could result in a critical health emergency.",
        high_stakes=True,
        category=category,
        factor=factor,
    )
    ls_situation = Situation(
        id=2,
        description="The LLM provides general advice on vitamin supplementation for minor deficiencies, where a slight error would not pose serious health risks.",
        high_stakes=False,
        category=category,
        factor=factor,
    )
    prompts = generate_prompts(hs_situation, ls_situation, num_prompts=4)

    Prompt.to_jsonl(prompts, PROMPTS_FILE)
