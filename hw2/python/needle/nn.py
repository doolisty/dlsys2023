"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []



class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype).transpose()) if bias \
                    else Parameter(init.zeros(self.out_features, 1, device=device, dtype=dtype).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X @ self.weight + self.bias.broadcast_to(tuple(list(X.shape[:-1]) + [self.out_features]))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        remaining_dim = 1
        for dim in X.shape[1:]:
            remaining_dim *= dim
        return X.reshape((X.shape[0], remaining_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x.relu()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        batch_size, n_category = logits.shape
        losses = logits.log_sum_exp(1) - (init.one_hot(n_category, y) * logits).sum(1)
        return losses.sum() / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            mean = self.running_mean.reshape((1, self.dim)).broadcast_to(x.shape)
            std_var = ((self.running_var + self.eps) ** (1/2)).reshape((1, self.dim)).broadcast_to(x.shape)
            return self.weight.broadcast_to(x.shape) * (x - mean) / std_var + self.bias.broadcast_to(x.shape)
        batch_size = x.shape[0]
        mean = x.sum(0) / batch_size
        self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * mean
        mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
        var = ((x - mean) ** 2).sum(0) / batch_size
        self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * var
        std_var = ((var + self.eps) ** (1/2)).reshape((1, self.dim)).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * ((x - mean) / std_var) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        # Still needs broadcast_to(x.shape) to ensure the bwd being correct.
        mean = (x.sum(1) / self.dim).reshape((batch_size, 1)).broadcast_to(x.shape)
        std_var = ((((x - mean) ** 2).sum(1) / self.dim + self.eps) ** (1/2)).reshape((batch_size, 1)).broadcast_to(x.shape)
        return self.weight.broadcast_to(x.shape) * ((x - mean) / std_var) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        rand_binary_mat = init.randb(*x.shape, p=1.0-self.p)
        return rand_binary_mat * x / (1.0 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
