import pytest
import numpy as np
from engine import Value

# Helper function for comparing gradients
def assert_almost_equal(actual, expected, rtol=1e-6, atol=1e-6):
    assert np.allclose(actual, expected, rtol, atol), f"Expected {expected}, but got {actual}"

@pytest.fixture
def x():
    return Value(np.array([1.0, 2.0, 3.0]))

@pytest.fixture
def y():
    return Value(np.array([4.0, 5.0, 6.0]))

def test_add(x, y):
    z = x + y
    z.backward()
    assert_almost_equal(x.grad, np.ones_like(x.data))
    assert_almost_equal(y.grad, np.ones_like(y.data))

def test_sub(x, y):
    z = x - y
    z.backward()
    assert_almost_equal(x.grad, np.ones_like(x.data))
    assert_almost_equal(y.grad, -1 * np.ones_like(y.data))

def test_mul(x, y):
    z = x * y
    z.backward()
    assert_almost_equal(x.grad, y.data)
    assert_almost_equal(y.grad, x.data)

def test_div(x, y):
    z = x / y
    z.backward()
    assert_almost_equal(x.grad, 1 / y.data)
    assert_almost_equal(y.grad, -x.data / (y.data ** 2))

def test_pow(x):
    z = x ** 2
    z.backward()
    assert_almost_equal(x.grad, 2 * x.data)

def test_matmul():
    x = Value(np.array([[1, 2], [3, 4]]))
    y = Value(np.array([[5, 6], [7, 8]]))
    z = (x @ y).mean()
    z.backward()
    assert_almost_equal(x.grad, np.array([[2.7500, 3.7500], [2.7500, 3.7500]]))
    assert_almost_equal(y.grad, np.array([[1.0000, 1.0000], [1.5000, 1.5000]]))

def test_mean(x):
    z = x.mean()
    z.backward()
    assert_almost_equal(x.grad, np.full_like(x.data, 1 / x.data.size))

def test_relu(x):
    x = Value(np.array([1.0, -2.0, 3.0]))
    z = x.relu()
    z.backward()
    assert_almost_equal(x.grad, np.array([1.0, 0.0, 1.0]))
