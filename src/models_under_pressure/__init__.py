import torch as t

default_tensor_repr = t.Tensor.__repr__


def custom_tensor_repr(self: t.Tensor) -> str:
    return f"{list(self.shape)}, {default_tensor_repr(self)}"


t.Tensor.__repr__ = custom_tensor_repr  # type: ignore
