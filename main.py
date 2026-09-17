import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)  
n_points = 100

study_time = np.random.uniform(20, 80, n_points)

score = 1.3 * study_time + 15 + np.random.normal(0, 6, n_points)

data = pd.DataFrame(
    {"study_time": np.round(study_time, 2), "score": np.round(score, 2)}
)
data.to_csv("data.csv", index=False)
print(f"Generated {len(data)} data points and saved to 'data.csv'.")

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].study_time
        y = points.iloc[i].score

        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.0001
epochs = 300

print("\nStarting Training...")
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch {i} | Slope (m): {m:.4f} | Intercept (b): {b:.4f}")
    m, b = gradient_descent(m, b, data, L)

print(f"\nFinal Parameters after {epochs} epochs:")
print(f"Slope (m)    : {m:.4f}")
print(f"Intercept (b): {b:.4f}")

plt.figure(figsize=(9, 6))
plt.scatter(
    data.study_time,
    data.score,
    color="black",
    alpha=0.7,
    label=f"100 Student Records",
)
x_range = np.linspace(20, 80, 100)
plt.plot(
    x_range,
    [m * x + b for x in x_range],
    color="red",
    linewidth=2,
    label="Fitted Regression Line",
)

plt.xlabel("Study Time (hours)")
plt.ylabel("Exam Score")
plt.title("Linear Regression from Scratch (100 Data Points)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()