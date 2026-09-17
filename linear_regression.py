import random

# -------------------------------------------------------------
# 1. Generate 100 synthetic data points: y = 2.5 * x + 5.0 + noise
# -------------------------------------------------------------
random.seed(42)
n_samples = 100

x_data = [random.uniform(0, 10) for _ in range(n_samples)]
true_m = 2.5
true_c = 5.0
noise = [random.gauss(0, 1.2) for _ in range(n_samples)]
y_data = [true_m * x + true_c + eps for x, eps in zip(x_data, noise)]


# -------------------------------------------------------------
# 2. Linear Regression Class (OLS & Gradient Descent)
# -------------------------------------------------------------
class LinearRegressionScratch:
    def __init__(self):
        self.slope = 0.0      # m (weight)
        self.intercept = 0.0  # c (bias)

    def fit_ols(self, x, y):
        """Analytical closed-form solution via Ordinary Least Squares."""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        self.slope = numerator / denominator
        self.intercept = mean_y - (self.slope * mean_x)

    def fit_gradient_descent(self, x, y, lr=0.01, epochs=2000):
        """Iterative parameter optimization using Batch Gradient Descent."""
        n = len(x)
        m = 0.0
        c = 0.0

        for _ in range(epochs):
            dm = 0.0
            dc = 0.0
            for i in range(n):
                y_pred = m * x[i] + c
                error = y_pred - y[i]
                dm += error * x[i]
                dc += error

            # Update weights using gradients
            m -= (2 / n) * lr * dm
            c -= (2 / n) * lr * dc

        self.slope = m
        self.intercept = c

    def predict(self, x):
        return [self.slope * xi + self.intercept for xi in x]

    def evaluate(self, y_true, y_pred):
        """Computes Mean Squared Error (MSE) and R-squared (R2)."""
        n = len(y_true)
        mean_y = sum(y_true) / n

        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        ss_tot = sum((y_true[i] - mean_y) ** 2 for i in range(n))
        ss_res = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n))
        r2 = 1 - (ss_res / ss_tot)

        return mse, r2


# -------------------------------------------------------------
# 3. Execution & Comparison
# -------------------------------------------------------------
if __name__ == "__main__":
    # --- Analytical OLS ---
    model_ols = LinearRegressionScratch()
    model_ols.fit_ols(x_data, y_data)
    preds_ols = model_ols.predict(x_data)
    mse_ols, r2_ols = model_ols.evaluate(y_data, preds_ols)

    # --- Batch Gradient Descent ---
    model_gd = LinearRegressionScratch()
    model_gd.fit_gradient_descent(x_data, y_data, lr=0.01, epochs=3000)
    preds_gd = model_gd.predict(x_data)
    mse_gd, r2_gd = model_gd.evaluate(y_data, preds_gd)

    print("Target True Parameters: slope = 2.50, intercept = 5.00\n")
    print(f"Ordinary Least Squares (OLS):")
    print(f"  slope: {model_ols.slope:.4f}, intercept: {model_ols.intercept:.4f}")
    print(f"  MSE:   {mse_ols:.4f}, R²: {r2_ols:.4f}\n")

    print(f"Gradient Descent (GD):")
    print(f"  slope: {model_gd.slope:.4f}, intercept: {model_gd.intercept:.4f}")
    print(f"  MSE:   {mse_gd:.4f}, R²: {r2_gd:.4f}")