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
│   │   ├── code/
│   │   │   ├── elastic_net.ipynb
│   │   │   ├── lasso.ipynb
│   │   │   ├── linear_reg.ipynb
│   │   │   ├── logistic_reg.ipynb
│   │   │   └── ridge.ipynb
│   │   ├── elastic_net.py
│   │   ├── lasso_regression.py
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   └── ridge_regression.py
│   ├── tree_models/
│   │   ├── code/
│   │   │   ├── decision_tree.ipynb
│   │   │   ├── regression_tree.ipynb
│   │   │   └── rf.ipynb
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── regression_tree.py
│   ├── ensemble_methods/
│   │   ├── code/
│   │   │   ├── stacking.ipynb
│   │   │   └── voting.ipynb
│   │   ├── stacking.py
│   │   └── voting.py
│   ├── naive_bayes/
│   │   ├── code.ipynb
│   │   └── naive_bayes_classifier.py
│   └── neighbors/
│       ├── code.ipynb
│       └── knn.py
├── unsupervised/
│   ├── kmeans/
│   │   ├── code.ipynb
│   │   └── kmeans.py
│   ├── DBSCAN/
│   │   ├── DBSCAN.ipynb
│   │   └── DBSCAN.py
│   └── Hierarchical/
│       ├── hierarchical.ipynb
│       └── hierarchical.py
├── metrics/
│   ├── classification/
│   │   ├── accuracy.py
│   │   ├── confusion_matrix.py
│   │   ├── f1_score.py
│   │   ├── precision.py
│   │   └── recall.py
│   ├── regression/
│   │   ├── adjusted_r2_score.py
│   │   ├── mae.py
│   │   ├── mse.py
│   │   ├── r2_score.py
│   │   └── rmse.py
│   └── clustering/
│       └── silhouette_score.py
├── dimensionality_reduction/
│   ├── code.ipynb
│   └── pca.py
├── preprocessing/
│   └── standard_scaler.py
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
    - `regression/`: `Mean Absolute Error (MAE)`, `Mean Squared Error (MSE)`, `Root Mean Squared Error (RMSE)`, `R2-Score`, and `Adjusted R2-Score`.
    - `clustering/`: `Silhouette Score`.

## Implementation & Dataset Application

All algorithms in this repository are implemented **from scratch** using only fundamental libraries like `numpy` and `pandas`. 

To see these algorithms in action, look for the `code/` directory or `.ipynb` files within each module. These notebooks demonstrate:
1.  **Data Loading & Preprocessing**: Loading standard datasets (e.g., Iris, Wine, California Housing).
2.  **Model Training**: Training the scratch-built models on these datasets.
3.  **Evaluation**: Using the custom `metrics` module to evaluate model performance and compare results.

For example, check [supervised/linear_models/code/linear_reg.ipynb](file:///d:/stats/ML_algo/supervised/linear_models/code/linear_reg.ipynb) to see Simple Linear Regression applied to a dataset.
