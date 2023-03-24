A small autodiff and neural network library patterned on Karpathy's micrograd, but with support for matrics and vectors
expressed as numpy arrays.

Written for my own clarity, but I hope it helps you too.

- Install with `pip install -e .` The only runtime dependency is numpy.
- Test with `pytest`.
- `main.py` runs a little demo of an MLP doing regression against random data.
- `nbs/demo.ipynb` is Karpathy's original demo notebook, adapted to use my library