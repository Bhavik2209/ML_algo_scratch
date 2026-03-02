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

```text
ML_algo/
├── supervised/
│   ├── linear_models/
│   ├── tree_models/
│   ├── ensemble_methods/
│   ├── naive_bayes/
│   └── neighbors/
├── unsupervised/
│   ├── kmeans/
│   ├── DBSCAN/
│   └── Hierarchical/
├── metrics/
│   ├── classification/
│   ├── regression/
│   └── clustering/
├── dimensionality_reduction/
├── preprocessing/
├── setup.py
└── README.md
```

The project is organized into several modules, each focusing on a specific area of Machine Learning:

- **`supervised/`**: Implementation of supervised learning algorithms.
    - `linear_models/`: Contains `Linear Regression`, `Lasso`, `Ridge`, `Elastic Net`, and `Logistic Regression`.
    - `tree_models/`: Includes `Decision Tree`, `Random Forest`, and `Regression Tree`.
    - `ensemble_methods/`: `Stacking` and `Voting` classifiers/regressors.
    - `naive_bayes/`: `Naive Bayes Classifier` implementation.
    - `neighbors/`: `K-Nearest Neighbors (KNN)` algorithm.
- **`unsupervised/`**: Implementation of unsupervised learning algorithms.
    - `kmeans/`: `K-Means Clustering`.
    - `DBSCAN/`: `DBSCAN Clustering`.
    - `Hierarchical/`: `Hierarchical Clustering`.
- **`dimensionality_reduction/`**: `PCA` (Principal Component Analysis) implementation.
- **`preprocessing/`**: Data preprocessing utilities like `Standard Scaler`.
- **`metrics/`**: Custom implementation of evaluation metrics for various tasks.
    - `classification/`: `Accuracy`, `Confusion Matrix`, `F1-Score`, `Precision`, and `Recall`.
    - `regression/`: `MSE`, `MAE`, `RMSE`, `R2-Score`, and `Adjusted R2-Score`.
    - `clustering/`: `Silhouette Score`.
