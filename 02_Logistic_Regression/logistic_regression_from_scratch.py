import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_gradient(theta, X, y):
    m = y.size
    predictions = sigmoid(X @ theta)
    return (X.T @ (predictions - y)) / m

def gradient_descent(X, y, alpha=0.1, n_iterations=300, tolerance=1e-6):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])

    for i in range(n_iterations):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad
        if np.linalg.norm(grad) < tolerance:
            print(f"Converged early at epoch {i}")
            break

    return theta


def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)


def predict(X, theta, threshold=0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_hat = gradient_descent(
    X_train_scaled, y_train, alpha=0.1, n_iterations=300
)

train_acc = accuracy_score(y_train, predict(X_train_scaled, theta_hat)) * 100
test_acc = accuracy_score(y_test, predict(X_test_scaled, theta_hat)) * 100

print("=== Logistic Regression From Scratch (Mathematical) ===")
print(f"Learned Parameters (theta) shape: {theta_hat.shape}")
print(f"Training Accuracy: {train_acc:.2f}%")
print(f"Testing Accuracy : {test_acc:.2f}%")

plt.figure(figsize=(8, 4))
test_probabilities = predict_proba(X_test_scaled, theta_hat)
plt.hist(
    test_probabilities[y_test == 0],
    bins=20,
    alpha=0.7,
    color="red",
    label="Class 0 (Malignant)",
)
plt.hist(
    test_probabilities[y_test == 1],
    bins=20,
    alpha=0.7,
    color="blue",
    label="Class 1 (Benign)",
)
plt.axvline(0.5, color="black", linestyle="--", label="Threshold = 0.5")
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Logistic Regression From Scratch - Prediction Probabilities")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()