# Logistic Regression: From EDA to Nonlinear Decision Boundaries

An end-to-end binary-classification project built with PyTorch and the Breast Cancer Wisconsin Diagnostic dataset.

The project studies how feature scaling, dimensionality reduction, and polynomial feature engineering affect optimization and classification quality. The notebook is written in Russian while common machine-learning terms and API names are kept in English.

## Objective

Predict whether a breast tumor is malignant or benign from 30 numerical features extracted from digitized fine-needle aspiration images.

For this project, the target is encoded as:

- `1` — malignant tumor, the positive class;
- `0` — benign tumor.

The logistic-regression model computes a logit

$$
z_i = x_i^T w + b
$$

and converts it into a probability with the sigmoid function:

$$
p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}.
$$

## Compared pipelines

| Pipeline | Representation | Main question |
|---|---|---|
| A | All 30 original features | How do strongly different feature scales affect SGD optimization? |
| B | All 30 standardized features | Does standardization improve optimization under the same training budget? |
| C | Two standardized PCA components | How much classification information is lost when reducing 30 dimensions to 2? |
| D | Standardized $u_1,u_2,u_1^2,u_1u_2,u_2^2$ | Can a quadratic representation improve the decision boundary in PCA space? |

Pipeline D remains linear in its trainable parameters, but the polynomial inputs allow it to produce a nonlinear boundary in the two-dimensional PCA plane.

## Implemented workflow

- Stratified train, validation, and test split
- Exploratory data analysis performed only on training data
- Feature-quality, class-balance, scale, distribution, and correlation checks
- Binary cross-entropy gradient derived and verified with autograd
- Custom PyTorch `Dataset` and `DataLoader`
- Logistic regression with `nn.Linear`
- Numerically stable training with `BCEWithLogitsLoss`
- Reusable training and evaluation loops
- Standardization and PCA fitted only on training data
- Polynomial feature engineering in PCA space
- Comparison of optimization curves and classification metrics
- Validation-based pipeline selection followed by a single final test evaluation

## Evaluation

The pipelines are compared using:

- binary cross-entropy loss;
- accuracy;
- recall for the malignant class;
- ROC AUC;
- training and validation learning curves;
- exact decision boundaries in PCA space.

Recall for the malignant class is especially important because it measures the proportion of malignant cases correctly identified by the model.

## Data-leakage prevention

All preprocessing objects are fitted exclusively on the training split:

```text
training data   → fit scaler and PCA
validation data → transform only and select a pipeline
test data       → transform only and evaluate once after selection
```

This keeps validation useful for model selection and preserves the test split as an unbiased final check.

## Key concepts

- Logistic regression, logits, sigmoid, and binary cross-entropy
- Analytical gradients and PyTorch autograd
- Tensor shapes and mini-batch training
- Standardization and optimization stability
- PCA and explained variance
- Polynomial features and nonlinear decision boundaries
- Classification metrics and honest model selection

## Notebook

[Open the notebook](02-logistic-regression-ru.ipynb)

The notebook contains eight implementation tasks and seven interpretation questions. Run all cells from top to bottom after completing the tasks to reproduce the experiment.

## Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch jupyterlab
```

## Project status

Work in progress. Final pipeline metrics and conclusions will be added after all implementation tasks, assertions, and interpretation questions are completed.
