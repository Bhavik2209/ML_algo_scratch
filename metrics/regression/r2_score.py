class R2Score:

    def calculate(self, y_true, y_pred):
        mean_y = sum(y_true) / len(y_true)

        ss_total = 0
        ss_residual = 0

        for i in range(len(y_true)):
            ss_total += (y_true[i] - mean_y) ** 2
            ss_residual += (y_true[i] - y_pred[i]) ** 2

        if ss_total == 0:
            return 0

        return 1 - (ss_residual / ss_total)