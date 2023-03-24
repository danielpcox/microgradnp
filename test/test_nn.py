import numpy as np
from microgradnp.engine import Value
from microgradnp.nn import Linear, ReLU, Sequential, Module

def test_module():
    m = Module()
    assert m.parameters() == []
    m.zero_grad()  # should not raise any exception

def test_linear():
    in_features = 3
    out_features = 2
    linear = Linear(in_features, out_features)

    x = Value(np.random.randn(1, in_features))
    y = linear.forward(x)

    assert y.data.shape == (1, out_features)
    assert len(linear.parameters()) == 2

    linear.zero_grad()
    assert np.all(linear.W.grad == 0)
    assert np.all(linear.b.grad == 0)

def test_relu():
    relu = ReLU()
    x = Value(np.array([[-1, 0], [1, 2]]))
    y = relu.forward(x)

    assert np.all(y.data == np.maximum(x.data, 0))

def test_sequential():
    in_features = 3
    out_features = 2
    seq = Sequential(
        Linear(in_features, 4),
        ReLU(),
        Linear(4, out_features)
    )

    x = Value(np.random.randn(1, in_features))
    y = seq.forward(x)

    assert y.data.shape == (1, out_features)
    assert len(seq.parameters()) == 4

    seq.zero_grad()
    for param in seq.parameters():
        assert np.all(param.grad == 0)

def test_call_method():
    in_features = 3
    out_features = 2
    seq = Sequential(
        Linear(in_features, 4),
        ReLU(),
        Linear(4, out_features)
    )

    x = Value(np.random.randn(1, in_features))
    y = seq(x)

    assert y.data.shape == (1, out_features)
