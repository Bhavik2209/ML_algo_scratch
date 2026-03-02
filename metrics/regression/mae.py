class MAE:

    def calculate(self, y_true, y_pred):
        total_error = 0

        for i in range(len(y_true)):
            total_error += abs(y_true[i] - y_pred[i])

        return total_error / len(y_true)