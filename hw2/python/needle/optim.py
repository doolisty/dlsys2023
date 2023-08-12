"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            grad = p.grad.data
            grad += self.weight_decay * p.data
            if p not in self.u.keys():
                self.u[p] = (1 - self.momentum) * grad
            else:
                self.u[p] = self.momentum * self.u[p] + (1 - self.momentum) * grad
            p.data = p - self.lr * self.u[p]
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            grad = p.grad.data
            grad += self.weight_decay * p.data
            if p not in self.m.keys():
                self.m[p] = (1 - self.beta1) * grad
            else:
                self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grad
            if p not in self.v.keys():
                self.v[p] = (1 - self.beta2) * grad ** 2
            else:
                self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * grad ** 2
            # bc = bias correction
            m_bc, v_bc = self.m[p] / (1 - self.beta1 ** self.t), self.v[p] / (1 - self.beta2 ** self.t)
            p.data = p - self.lr * m_bc / (v_bc ** (1/2) + self.eps)
        ### END YOUR SOLUTION
