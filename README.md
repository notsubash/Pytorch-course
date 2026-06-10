# PyTorch Course

This repository contains a hands-on PyTorch learning path, organized as small scripts that progress from tensor basics to transfer learning and model persistence.

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/)

## Setup

Install dependencies:

```bash
poetry install
```

Activate the Poetry shell (optional):

```bash
poetry shell
```

## Run Examples

Run any lesson script with:

```bash
poetry run python "<script_name>.py"
```

Example:

```bash
poetry run python "01_TensorBasics.py"
```

> Some filenames include spaces, so keep script names in quotes when running them.

## Course Script Order

1. `01_TensorBasics.py` - Tensor creation, shapes, slicing, NumPy interop, GPU basics
2. `02_AutoGrad.py` - Autograd and gradient tracking
3. `03_BackPropagation.py` - Backpropagation fundamentals
4. `04_step1_gradient_descent_with_autograd_and _backpropagation.py`
5. `04_step2_gradient_descent_with_autograd_and _backpropagation.py`
6. `04_step3_gradient_descent_with_autograd_and _backpropagation.py`
7. `04_step4_gradient_descent_with_autograd_and _backpropagation.py`
8. `05_LinearRegression.py` - Linear regression in PyTorch
9. `06_LogisticRegression.py` - Logistic regression in PyTorch
10. `07_Dataset_and_DataLoader.py` - Dataset and DataLoader usage
11. `08_transforms.py` - Transform pipelines
12. `09_Softmax_and_Cross_Entropy.py` - Classification losses and outputs
13. `10_Activation_Functions.py` - Common activations
14. `10_plot_activations.py` - Activation visualization
15. `11_Feed_Forward_Neural_Network.py` - MLP fundamentals
16. `12_Convolutional_Neural_Network.py` - CNN example
17. `13_Transfer_Learning.py` - Transfer learning workflow
18. `14_TensorBoard.py` - TensorBoard logging
19. `15_Saving_and_Loading_models.py` - Model checkpointing

## Additional Resources

- `resources/PyTorch_Transforms.md`
- `data/wine.csv` (used by regression/data-loading examples)

