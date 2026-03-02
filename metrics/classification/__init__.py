from .accuracy import Accuracy
from .confusion_matrix import ConfusionMatrix
from .f1_score import F1Score
from .precision import Precision
from .recall import Recall

def accuracy_score(y_true, y_pred):
    return Accuracy().calculate(y_true, y_pred)

def confusion_matrix(y_true, y_pred):
    return ConfusionMatrix().calculate(y_true, y_pred)

def f1_score(y_true, y_pred):
    return F1Score().calculate(y_true, y_pred)

def precision_score(y_true, y_pred):
    return Precision().calculate(y_true, y_pred)

def recall_score(y_true, y_pred):
    return Recall().calculate(y_true, y_pred)

__all__ = [
    "Accuracy", "ConfusionMatrix", "F1Score", "Precision", "Recall",
    "accuracy_score", "confusion_matrix", "f1_score", "precision_score", "recall_score"
]
