# Machine Learning Portfolio

A growing collection of practical machine-learning projects implemented from first principles and with modern Python frameworks.

The projects focus on understanding the complete ML workflow: preparing data, defining a model and loss function, optimizing parameters, validating results, and interpreting model behavior.

## Projects

| Project | Tools | Topics | Result |
|---|---|---|---|
| [Linear Regression: NumPy to PyTorch](linear_regression/) | Python, NumPy, PyTorch, Matplotlib | MSE, numerical and analytical gradients, autograd, gradient descent, mini-batches, validation | Validation MSE ≈ 0.224 |

## Skills demonstrated

- Python for numerical computing and data analysis
- NumPy implementations of ML algorithms from first principles
- PyTorch models, autograd, optimizers, `Dataset`, and `DataLoader`
- Training and validation workflows
- Model evaluation with loss curves and residual analysis
- Reproducible experiments in Jupyter notebooks

## Running the notebooks

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
jupyter lab
```

Open a notebook and run all cells from top to bottom.

## Repository structure

```text
machine-learning-portfolio/
├── README.md
├── requirements.txt
└── linear_regression/
    ├── README.md
    └── Week_01_Lab_Linear_regression.ipynb
```

More projects will be added as the portfolio develops.
