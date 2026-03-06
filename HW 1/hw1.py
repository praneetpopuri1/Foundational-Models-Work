# hw1.py
import numpy as np
import pandas as pd
import struct
from typing import Optional

class Model:
    def __init__(self, input_d, hidden_d, output_d, loss_type):
        self.hidden_layer = np.random.uniform(-0.01, 0.01, (input_d + 1, hidden_d))
        self.output_layer = np.random.uniform(-0.01, 0.01, (hidden_d + 1, output_d))
        self.loss_type = loss_type

    def forward(self, data_matrix_x: np.ndarray):
        ones = np.ones((data_matrix_x.shape[0], 1))
        Xb = np.hstack((data_matrix_x, ones))  # (m, n+1)

        # Hidden pre-activation
        hidden_pre = Xb @ self.hidden_layer    # (m, h)
        self.h_1 = hidden_pre

        # ReLU
        hidden_act = hidden_pre.clip(0.0)

        # Add bias to hidden
        ones_h = np.ones((hidden_act.shape[0], 1))
        Hb = np.hstack((hidden_act, ones_h))   # (m, h+1)
        self.a_1 = Hb

        # Linear output logits
        out = Hb @ self.output_layer           # (m, o)
        self.h_2 = out

        if self.loss_type == "cross_entropy":
            # Softmax (numerically stable)
            max_value = np.max(out, axis=1, keepdims=True)
            exp_logits = np.exp(out - max_value)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            return probs

        if self.loss_type == "mse":
            # Linear output for regression
            return out

        raise ValueError("Unsupported loss_type")

    def train(self, data_matrix_x, y_labels, learning_rate=0.05, epochs=100):
        losses = []
        batch_size = 32

        for epoch in range(epochs):
            perm = np.random.permutation(data_matrix_x.shape[0])
            X_shuf = data_matrix_x[perm]
            Y_shuf = y_labels[perm]

            for start in range(0, X_shuf.shape[0], batch_size):
                X = X_shuf[start:start + batch_size]
                Y = Y_shuf[start:start + batch_size]

                y_hat = self.forward(X)

                if self.loss_type == "cross_entropy":
                    # y_hat: probs, Y: one-hot
                    loss_grad = (y_hat - Y) / X.shape[0]  # (B, o)

                elif self.loss_type == "mse":
                    # y_hat: linear outputs, Y: regression targets (B, o)
                    # MSE = mean((y_hat - Y)^2)
                    loss_grad = (2.0 * (y_hat - Y)) / X.shape[0]  # (B, o)

                else:
                    raise ValueError("Unsupported loss_type")

                # Backprop
                dW2 = self.a_1.T @ loss_grad  # (h+1, o)

                dA1 = loss_grad @ self.output_layer.T  # (B, h+1)
                h = self.hidden_layer.shape[1]
                dA1 = dA1[:, :h]  # drop bias grad part

                dZ1 = dA1 * (self.h_1 > 0)  # ReLU grad (B, h)

                Xb = np.hstack((X, np.ones((X.shape[0], 1))))
                dW1 = Xb.T @ dZ1  # (n+1, h)

                self.hidden_layer -= learning_rate * dW1
                self.output_layer -= learning_rate * dW2

            pred = self.forward(data_matrix_x)

            if self.loss_type == "cross_entropy":
                eps = 1e-12
                N = y_labels.shape[0]
                loss = -np.sum(y_labels * np.log(pred + eps)) / N

            elif self.loss_type == "mse":
                # mean squared error over all samples (and output dims)
                loss = np.mean((pred - y_labels) ** 2)

            else:
                raise ValueError("Unsupported loss_type")

            losses.append(loss)

        return losses


def getting_housing_data(csv_path="house.csv"):
    """
    Loads housing CSV, does 80/20 split with np.random.seed(0),
    standardizes X using TRAIN mean/std,
    standardizes y using TRAIN mean/std,
    returns:
      X_train_std, y_train_std, X_test_std, y_test_std, y_mean, y_std
    """
    df = pd.read_csv(csv_path)

    # Drop missing
    df = df.dropna(axis=0).reset_index(drop=True)

    # One-hot ocean_proximity if present
    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=False)

    y = df["median_house_value"].to_numpy(dtype=np.float64).reshape(-1, 1)
    X = df.drop(columns=["median_house_value"]).to_numpy(dtype=np.float64)

    # 80/20 split with seed 0
    np.random.seed(0)
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    train_idx = idx[:split]
    test_idx = idx[split:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    # Standardize X using train stats
    x_mu = X_train.mean(axis=0, keepdims=True)
    x_sigma = X_train.std(axis=0, keepdims=True)
    x_sigma[x_sigma == 0] = 1.0

    X_train_std = (X_train - x_mu) / x_sigma
    X_test_std = (X_test - x_mu) / x_sigma

    # Standardize y using train stats (THIS is what prevents overflow)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True)
    y_std[y_std == 0] = 1.0

    y_train_std = (y_train - y_mean) / y_std
    y_test_std = (y_test - y_mean) / y_std

    return X_train_std, y_train_std, X_test_std, y_test_std, y_mean, y_std

def load_mnist_images(filename: str) -> np.ndarray:

    with open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    images = data.reshape(num_images, rows, cols)
    return images

def load_mnist_labels(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    labels = labels.astype(int)
    out = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    out[np.arange(labels.shape[0]), labels] = 1.0
    return out

def getting_mnist_data(
    train_images_path: str,
    train_labels_path: str,
    test_images_path: str,
    test_labels_path: str,
    limit_train: Optional[int] = None,
):
    X_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    X_test  = load_mnist_images(test_images_path)
    y_test  = load_mnist_labels(test_labels_path)

    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize to [0,1]
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Optional subset for speed (assignment allows this if you note it)
    if limit_train is not None:
        X_train = X_train[:limit_train]
        y_train = y_train[:limit_train]

    # One-hot labels
    y_train_oh = one_hot(y_train, 10)
    y_test_oh = one_hot(y_test, 10)

    return X_train, y_train_oh, X_test, y_test_oh