class Precision:

    def calculate(self, y_true, y_pred):
        tp = 0
        fp = 0

        for i in range(len(y_true)):
            if y_pred[i] == 1:
                if y_true[i] == 1:
                    tp += 1
                else:
                    fp += 1

        if tp + fp == 0:
            return 0

        return tp / (tp + fp)