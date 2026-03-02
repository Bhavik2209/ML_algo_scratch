class MSE:

    def calculate(self, y_true, y_pred):
        total_error = 0

        for i in range(len(y_true)):
            diff = y_true[i] - y_pred[i]
            total_error += diff * diff

        return total_error / len(y_true)