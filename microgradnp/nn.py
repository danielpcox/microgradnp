import numpy as np
from microgradnp.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Value(np.random.randn(in_features, out_features) * 0.01, _op=f'linW:{in_features}->{out_features}')
        self.b = Value(np.zeros((1, out_features)), _op=f'linb:{in_features}->{out_features}')

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

