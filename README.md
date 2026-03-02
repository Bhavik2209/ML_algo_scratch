# ML-algo Project

This repository contains scratch implementations of various Machine Learning algorithms.

## Installation

To use these modules without import errors, you should install the project in **editable mode**. 

1. Open your terminal.
2. Navigate to the root of this project (`ML_algo` folder).
3. Run the following command:

```bash
pip install -e .
```

## Why this is necessary?
Installing in editable mode (`-e`) allows Python to recognize the `supervised`, `metrics`, `preprocessing`, and other folders as a package. This eliminates the need for manual `sys.path` modifications and ensures that imports work correctly from any script or notebook within the project.

## Structure
- `supervised/`: Supervised learning models (Linear, Tree-based, etc.)
- `unsupervised/`: Unsupervised learning models (K-Means, DBSCAN, etc.)
- `metrics/`: Custom implementation of evaluation metrics.
- `preprocessing/`: Data preprocessing utilities.
