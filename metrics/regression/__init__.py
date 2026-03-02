from .mae import MAE
from .mse import MSE
from .rmse import RMSE
from .r2_score import R2Score
from .adjusted_r2_score import AdjustedR2Score

def mae(y_true, y_pred):
    return MAE().calculate(y_true, y_pred)

def mse(y_true, y_pred):
    return MSE().calculate(y_true, y_pred)

def rmse(y_true, y_pred):
    return RMSE().calculate(y_true, y_pred)

def r2_score(y_true, y_pred):
    return R2Score().calculate(y_true, y_pred)

def adjusted_r2_score(y_true, y_pred, n_features):
    return AdjustedR2Score().calculate(y_true, y_pred, n_features)

__all__ = [
    "MAE", "MSE", "RMSE", "R2Score", "AdjustedR2Score",
    "mae", "mse", "rmse", "r2_score", "adjusted_r2_score"
]
