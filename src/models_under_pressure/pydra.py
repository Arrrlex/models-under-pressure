"""
This is a thin wrapper around hydra.main that allows us to use pydantic models as config.
"""

import functools
from typing import Callable, TypeVar, get_type_hints
import hydra
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def _dictconfig_to_dict(config: DictConfig) -> dict:
    """Recursively converts DictConfig to a regular dictionary."""
    return OmegaConf.to_container(config, resolve=True, enum_to_str=True)  # type: ignore


def main(config_path: str, version_base: str | None = None) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @hydra.main(config_path=config_path, version_base=version_base)
        @functools.wraps(func)
        def wrapper(dict_config: DictConfig) -> T:
            # Get the type annotation for the config parameter
            type_hints = get_type_hints(func)
            config_type = type_hints["config"]

            # Convert DictConfig to a regular dictionary, handling nested DictConfigs
            config_dict = _dictconfig_to_dict(dict_config)

            # Create the typed config
            typed_config = config_type.model_validate(config_dict)

            return func(typed_config)

        return wrapper

    return decorator
