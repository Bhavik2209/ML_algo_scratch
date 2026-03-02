import numpy as np


class ConfusionMatrix:

    def calculate(self, y_true, y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                tn += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1

        return {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }