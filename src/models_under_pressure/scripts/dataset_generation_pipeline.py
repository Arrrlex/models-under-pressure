from models_under_pressure.config import RunConfig
from models_under_pressure.dataset.combine_situations import (
    generate_combined_situations,
)
from models_under_pressure.dataset.medadata_generation import generate_metadata_file
from models_under_pressure.dataset.prompt_generation import generate_prompts_file
from models_under_pressure.dataset.situation_generation import generate_situations_file


def main(run_config: RunConfig):
    # Commend out where needed:
    generate_combined_situations(run_config)
    generate_situations_file(run_config)
    generate_prompts_file(run_config)
    generate_metadata_file(run_config)


if __name__ == "__main__":
    main(RunConfig(run_id="test"))
