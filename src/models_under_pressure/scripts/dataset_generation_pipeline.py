from models_under_pressure.config import RunConfig
from models_under_pressure.dataset_generation.combine_situations import (
    generate_combined_situations,
)
from models_under_pressure.dataset_generation.medadata_generation import (
    generate_metadata_file,
)
from models_under_pressure.dataset_generation.prompt_generation import (
    generate_prompts_file,
)
from models_under_pressure.dataset_generation.situation_generation import (
    generate_situations_file,
)


def main(run_config: RunConfig):
    # Commend out where needed:
    generate_combined_situations(run_config)
    generate_situations_file(run_config)
    generate_prompts_file(run_config)
    generate_metadata_file(run_config)


if __name__ == "__main__":
    main(RunConfig(run_id="test"))
