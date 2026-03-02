class Recall:

    def calculate(self, y_true, y_pred):
        tp = 0
        fn = 0

        for i in range(len(y_true)):
            if y_true[i] == 1:
                if y_pred[i] == 1:
                    tp += 1
                else:
                    fn += 1

        if tp + fn == 0:
            return 0

        return tp / (tp + fn)