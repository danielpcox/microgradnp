import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data)
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be a scalar"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__neg__().__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), '/')

        def _backward():
            self.grad += out.grad / other.data
            other.grad -= self.data * out.grad / (other.data ** 2)

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return self.__pow__(-1).__mul__(other)

    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')

        def _backward():
            self.grad -= out.grad

        out._backward = _backward
        return out

    def mean(self):
        out_data = np.mean(self.data)
        out = Value(out_data, (self,), 'mean')

        def _backward():
            grad_divisor = self.data.size
            self.grad += out.grad / grad_divisor

        out._backward = _backward
        return out

    def relu(self):
        out = Value(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self, grad=None):
        if grad is None:
            if self.grad is None:
                grad = np.ones_like(self.data)
            else:
                return
        else:
            grad = np.array(grad)
        self.grad = grad

        for v in self._prev:
            v.backward()

        self._backward()

    def __repr__(self):
        return f"Value(_op={self._op}, grad={self.grad}, data={self.data})"