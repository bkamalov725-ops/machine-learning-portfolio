# Linear Regression: From NumPy to PyTorch

This project builds and trains the same linear-regression model in three different ways:

1. NumPy with analytical gradients derived from the MSE formula.
2. PyTorch with automatic differentiation.
3. PyTorch with `Dataset`, `DataLoader`, and mini-batch training.

The notebook is written in Russian while common machine-learning terms and API names are kept in English.

## Objective

Fit the model

$$
\hat{y} = wx + b
$$

to a noisy synthetic dataset generated from

$$
y = 1.75x - 0.40 + \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, 0.45^2).
$$

The experiment uses 128 training observations and 32 held-out validation observations.

## Implemented components

- Mean squared error from scratch
- Numerical gradients using central finite differences
- Analytical gradients for weight and bias
- Gradient verification against PyTorch autograd
- Full-batch gradient descent in NumPy
- Full-batch training in PyTorch
- Custom PyTorch `Dataset`
- Mini-batch training with `DataLoader`
- Validation tracking, loss curves, fitted-line comparison, and residual analysis

## Results

| Implementation | Gradient method | Data feeding | Weight | Bias | Validation MSE |
|---|---|---|---:|---:|---:|
| NumPy | Analytical | Full batch | 1.733 | -0.344 | 0.2236 |
| PyTorch | Autograd | Full batch | 1.733 | -0.344 | 0.2236 |
| PyTorch | Autograd | `DataLoader` mini-batches | 1.737 | -0.342 | 0.2241 |

All three implementations converge to nearly the same fitted line. The mini-batch path is noisier, but its final parameters and validation error remain close to the full-batch solutions.

## Key observations

- Training and validation losses decrease and stabilize at similar values.
- The learned parameters are close to the data-generating values, $w=1.75$ and $b=-0.40$.
- Validation residuals are distributed around zero without a strong systematic pattern.
- Numerical differentiation is useful for checking gradients but scales poorly because each parameter requires additional loss evaluations.

## Notebook

[Open the completed notebook](Week_01_Lab_Linear_regression.ipynb)

Run every cell from top to bottom to reproduce the saved results. The notebook contains assertions for all eight implementation tasks.
