"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar * out_grad * node.inputs[0] ** (self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * -lhs / rhs ** 2
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def axes2perm(self, size):
        perm = [i for i in range(size)]
        if self.axes is None:
            perm[-1], perm[-2] = perm[-2], perm[-1]
        elif self.axes is not None:
            if len(self.axes) == size:
                perm = self.axes
            else:
                perm = [i for i in range(size)]
                perm[self.axes[0]], perm[self.axes[1]] = self.axes[1], self.axes[0]
        return perm

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.transpose(a, axes=self.axes2perm(len(a.shape)))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes2perm(len(node.inputs[0].shape)))
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        sum_axes = []

        # Since summation will eliminate a dim, we need to recover it by reshape
        need_reshape = False

        # record axes for additional dims
        offset = len(self.shape) - len(input_shape)  # number of additional dims
        for axis in range(offset):
            sum_axes.append(axis)

        # record axes who conducted (1 => shape) in fwd
        for axis in range(len(input_shape)):
            if input_shape[axis] < self.shape[offset + axis]:
                need_reshape = True
                sum_axes.append(offset + axis)
        out_grad = out_grad.sum(axes=tuple(sum_axes))  # no need to divide

        # padding 1s for dim alignment
        if need_reshape:
            out_grad = out_grad.reshape(input_shape)

        return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = tuple([axes]) if isinstance(axes, int) else axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        if self.axes is None:
            self.axes = [i for i in range(len(input_shape))]
        reshape_target = [input_shape[i] if i not in self.axes else 1 for i in range(len(input_shape))]
        return out_grad.reshape(tuple(reshape_target)).broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        # Y = XW
        # X' = Y'(W^T)  (lhs)
        # W' = (X^T)Y'  (rhs)
        lhs, rhs = node.inputs
        lhs_grad, rhs_grad = out_grad @ rhs.transpose(), lhs.transpose() @ out_grad

        # sum over batches
        if len(lhs.shape) < len(lhs_grad.shape):
            lhs_grad = lhs_grad.sum(axes=tuple([i for i in range(len(lhs_grad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rhs_grad.shape):
            rhs_grad = rhs_grad.sum(axes=tuple([i for i in range(len(rhs_grad.shape) - len(rhs.shape))]))

        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node.inputs[0] ** -1
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node.inputs[0].exp()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, array_api.zeros_like(a))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_ndarray = node.realize_cached_data()
        out_ndarray[out_ndarray > 0] = 1
        return out_grad * Tensor(out_ndarray)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = tuple([axes]) if isinstance(axes, int) else axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.maxvals = array_api.max(Z, axis=self.axes)
        # self.maxvals_keepdim_bc = array_api.broadcast_to(array_api.max(Z, axis=self.axes, keepdims=True), Z.shape)
        self.maxvals_keepdim_bc = array_api.max(Z, axis=self.axes, keepdims=True)  # 不做broadcast_to也可以计算
        return array_api.log(array_api.sum(array_api.exp(Z - self.maxvals_keepdim_bc), axis=self.axes)) + self.maxvals
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION

        # 从函数外层到内层逐层传播，逐层应用 outgrad * grad(input)
        input_tensor = node.inputs[0]
        maxval_tensor = Tensor(node.inputs[0].realize_cached_data().max(axis=self.axes, keepdims=True))
        exp_Z = (input_tensor - maxval_tensor).exp()  # the same as its derivative
        log_func_derivative = exp_Z.sum(self.axes) ** -1
        log_prop_grad = out_grad * log_func_derivative
        if self.axes is None:
            self.axes = [i for i in range(len(input_tensor.shape))]
        reshape_target = [input_tensor.shape[i] if i not in self.axes else 1 for i in range(len(input_tensor.shape))]
        exp_prop_grad = exp_Z * log_prop_grad.reshape(tuple(reshape_target)).broadcast_to(input_tensor.shape)
        return exp_prop_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
