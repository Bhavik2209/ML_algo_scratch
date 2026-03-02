import numpy as np


class Accuracy:

    def calculate(self, y_true, y_pred):
        correct = 0
        total = len(y_true)

        for i in range(total):
            if y_true[i] == y_pred[i]:
                correct += 1

        return correct / total