class AdjustedR2Score:

    def calculate(self, y_true, y_pred, n_features):
        n = len(y_true)

        mean_y = sum(y_true) / n

        ss_total = 0
        ss_residual = 0

        for i in range(n):
            ss_total += (y_true[i] - mean_y) ** 2
            ss_residual += (y_true[i] - y_pred[i]) ** 2

        if ss_total == 0:
            return 0

        r2 = 1 - (ss_residual / ss_total)

        return 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))