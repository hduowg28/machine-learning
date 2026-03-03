import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

DATA_FILE = Path(__file__).parent / "gd4Dataset.txt"

def load_data(path=DATA_FILE):
    try:
        A = np.loadtxt(path, delimiter=",")
    except OSError as e:
        raise RuntimeError(f"could not read data file {path}") from e
    return A[:, 0], A[:, 1]

def step_gradient(X, y, a, b, lr):
    """Return updated (a,b) after one gradient step."""
    n = len(y)
    residual = y - (a * X + b)
    da = (-2 * X * residual).mean()
    db = (-2 * residual).mean()
    return a - lr * da, b - lr * db

def loss(X, y, a, b):
    return np.mean((y - (a * X + b)) ** 2)

def train(X, y, a=0.0, b=0.0, lr=0.01, iterations=1000):
    history = []
    for _ in tqdm(range(iterations)):
        a, b = step_gradient(X, y, a, b, lr)
        history.append(loss(X, y, a, b))
    return a, b, history

def predict(X, a, b):
    return a * X + b

def main():
    X, y = load_data()
    a, b, hist = train(X, y, a=0.1, b=0.1, lr=0.01, iterations=5000)
    y_pred = predict(X, a, b)

    order = np.argsort(X)
    plt.scatter(X, y, color="red")
    plt.plot(X[order], y_pred[order], color="blue")
    plt.show()

if __name__ == "__main__":
    main()