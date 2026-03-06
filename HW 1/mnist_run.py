import numpy as np
import matplotlib.pyplot as plt

from hw1 import Model, getting_mnist_data

def cross_entropy_loss(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    eps = 1e-12
    N = y_onehot.shape[0]
    return float(-np.sum(y_onehot * np.log(probs + eps)) / N)

def accuracy(probs: np.ndarray, y_onehot: np.ndarray) -> float:
    pred = np.argmax(probs, axis=1)
    true = np.argmax(y_onehot, axis=1)
    return float(np.mean(pred == true))

def plot_lr_sweep(losses_by_lr, out_path):
    plt.figure()
    for lr, losses in losses_by_lr.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"LR={lr:g}")
    plt.title("MNIST: Train Cross-Entropy vs Epoch (Hidden=5)")
    plt.xlabel("Epoch")
    plt.ylabel("Train Cross-Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hidden_sweep(losses_by_h, out_path):
    plt.figure()
    for h, losses in losses_by_h.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"H={h}")
    plt.title("MNIST: Train Cross-Entropy vs Epoch (LR=1e-2)")
    plt.xlabel("Epoch")
    plt.ylabel("Train Cross-Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    # Update these paths to wherever you downloaded the MNIST idx files
    train_images = "train-images.idx3-ubyte"
    train_labels = "train-labels.idx1-ubyte"
    test_images  = "t10k-images.idx3-ubyte"
    test_labels  = "t10k-labels.idx1-ubyte"

    # If runtime is too slow, set limit_train=20000 (and note it in your writeup)
    X_train, y_train, X_test, y_test = getting_mnist_data(
        train_images, train_labels, test_images, test_labels,
        limit_train=None
    )

    input_d = X_train.shape[1]   # 784
    output_d = y_train.shape[1]  # 10
    epochs = 10

    hidden_d = 5
    lrs = [1, 1e-2, 1e-3, 1e-8]
    losses_by_lr = {}
    test_metrics_by_lr = {}

    for lr in lrs:
        np.random.seed(0)
        model = Model(input_d, hidden_d, output_d, loss_type="cross_entropy")

        train_losses = model.train(X_train, y_train, learning_rate=lr, epochs=epochs)
        losses_by_lr[lr] = train_losses

        # Evaluate (no more updates)
        probs_test = model.forward(X_test)
        test_loss = cross_entropy_loss(probs_test, y_test)
        test_acc = accuracy(probs_test, y_test)
        test_metrics_by_lr[lr] = (test_loss, test_acc)

    plot_lr_sweep(losses_by_lr, "mnist_part3a_lr_sweep.png")

    print("\n[MNIST Part 3(c)] Test metrics by LR (Hidden=5):")
    for lr in lrs:
        loss, acc = test_metrics_by_lr[lr]
        print(f"  LR={lr:g}  Test CE={loss:.6f}  Test Acc={acc:.4f}")

    lr = 1e-2
    hidden_sizes = [2, 8, 16, 32]
    losses_by_h = {}
    test_metrics_by_h = {}

    for h in hidden_sizes:
        np.random.seed(0)
        model = Model(input_d, h, output_d, loss_type="cross_entropy")

        train_losses = model.train(X_train, y_train, learning_rate=lr, epochs=epochs)
        losses_by_h[h] = train_losses

        probs_test = model.forward(X_test)
        test_loss = cross_entropy_loss(probs_test, y_test)
        test_acc = accuracy(probs_test, y_test)
        test_metrics_by_h[h] = (test_loss, test_acc)

    plot_hidden_sweep(losses_by_h, "mnist_part3d_hidden_sweep.png")

    print("\n[MNIST Part 3(e)] Test metrics by Hidden Size (LR=1e-2):")
    for h in hidden_sizes:
        loss, acc = test_metrics_by_h[h]
        print(f"  H={h:2d}  Test CE={loss:.6f}  Test Acc={acc:.4f}")

    print("\nSaved plots:")
    print("  - mnist_part3a_lr_sweep.png")
    print("  - mnist_part3d_hidden_sweep.png")

if __name__ == "__main__":
    main()