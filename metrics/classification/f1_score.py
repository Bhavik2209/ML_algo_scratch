class F1Score:

    def calculate(self, y_true, y_pred):
        tp = 0
        fp = 0
        fn = 0

        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true[i] == 1:
                tp += 1
            elif y_pred[i] == 1 and y_true[i] == 0:
                fp += 1
            elif y_pred[i] == 0 and y_true[i] == 1:
                fn += 1

        if tp == 0:
            return 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0

        return 2 * (precision * recall) / (precision + recall)