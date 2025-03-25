"""
This is a thin wrapper around hydra.main that allows us to use pydantic models as config.
"""

import functools
from typing import Any, Callable, TypeVar, get_type_hints
import hydra
from omegaconf import DictConfig, OmegaConf

T = TypeVar("T")


def main(*args: Any, **kwargs: Any) -> Callable:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @hydra.main(*args, **kwargs)
        @functools.wraps(func)
        def wrapper(dict_config: DictConfig) -> T:
            # Get the type annotation for the config parameter
            type_hints = get_type_hints(func)
            config_type = type_hints["config"]

            # Convert DictConfig to a regular dictionary, handling nested DictConfigs
            config_dict = OmegaConf.to_container(
                dict_config, resolve=True, enum_to_str=True
            )

            # Create the typed config
            typed_config = config_type.model_validate(config_dict)

            return func(typed_config)

        return wrapper

    return decorator
