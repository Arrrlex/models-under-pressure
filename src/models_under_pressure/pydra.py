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
        def wrapper(config: DictConfig) -> T:
            config_type = get_type_hints(func)["config"]
            config_dict = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
            config_model = config_type.model_validate(config_dict)
            return func(config_model)

        return wrapper

    return decorator
