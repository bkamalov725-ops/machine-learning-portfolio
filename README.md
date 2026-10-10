# Machine Learning Portfolio

A growing collection of practical machine-learning projects implemented from first principles and with modern Python frameworks.

The projects focus on understanding the complete ML workflow: preparing data, defining a model and loss function, optimizing parameters, validating results, and interpreting model behavior.

Latest project: [Satellite Image Classification](satellite_classification/). A ResNet152/Inception v3 ensemble with public Kaggle Accuracy **0.96660**, two-GPU training, and a measured final training time of **22 min 59 sec**. Includes the final notebook, source code, synthetic tests, and an aggregate experiment report; no competition data or trained weights.

New project: [Hidden Pairs - Decision Trees, Ensembles and SVM](decision_tree_classifier/). Includes the completed tree/ensemble lab and the team's selected competition SVM, with validation ROC-AUC **0.947921**, a reproducible prediction notebook, and an experiment summary.

## Projects

| Project | Tools | Topics | Result |
|---|---|---|---|
| [Satellite Image Classification](satellite_classification/) | Python, PyTorch, torchvision, Kaggle T4 ×2 | Transfer learning, AdamW, EMA, augmentations, D4 TTA, probability ensembles, time-bounded training | Public Accuracy **0.96660**; final training **22 min 59 sec**; private score unknown |
| [Linear Regression: NumPy to PyTorch](linear_regression/) | Python, NumPy, PyTorch, Matplotlib | MSE, numerical and analytical gradients, autograd, gradient descent, mini-batches, validation | Validation MSE ≈ 0.224 |
| [Logistic Regression: EDA to Nonlinear Boundaries](logistic_regression/) | Python, pandas, scikit-learn, PyTorch, Seaborn | BCE, classification metrics, standardization, PCA, polynomial features, data-leakage prevention | Work in progress |

## Skills demonstrated

- Python for numerical computing and data analysis
- NumPy implementations of ML algorithms from first principles
- PyTorch models, autograd, optimizers, `Dataset`, and `DataLoader`
- Exploratory data analysis with pandas, Matplotlib, and Seaborn
- Feature preprocessing and dimensionality reduction with scikit-learn
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
├── satellite_classification/
│   ├── README.md
│   ├── notebooks/solution_kaggle.ipynb
│   ├── src/
│   ├── tests/
│   └── results/metrics.json
├── linear_regression/
│   ├── README.md
│   ├── Week_01_Lab_Linear_regression.html
│   └── Week_01_Lab_Linear_regression.ipynb
└── logistic_regression/
    ├── README.md
    └── 02-logistic-regression-ru.ipynb
```

More projects will be added as the portfolio develops.
