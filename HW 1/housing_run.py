import numpy as np
import matplotlib.pyplot as plt

from hw1 import getting_housing_data, Model

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def unscale_y(y_std, y_mean, y_stddev):
    return y_std * y_stddev + y_mean

def plot_lr_sweep(losses_by_lr, out_path):
    plt.figure()
    for lr, losses in losses_by_lr.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"LR={lr:g}")
    plt.title("California Housing: Train MSE vs Epoch (Hidden=5) [on standardized y]")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE (standardized y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_hidden_sweep(losses_by_h, out_path):
    plt.figure()
    for h, losses in losses_by_h.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f"H={h}")
    plt.title("California Housing: Train MSE vs Epoch (LR=1e-2) [on standardized y]")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE (standardized y)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def safe_train(model, X_train, y_train, lr, epochs):
    losses = model.train(X_train, y_train, learning_rate=lr, epochs=epochs)
    # If it diverged, losses may become nan; keep them but you can also early-stop if you want.
    return losses

def main():
    X_train, y_train, X_test, y_test, y_mean, y_std = getting_housing_data("house.csv")

    input_d = X_train.shape[1]
    output_d = 1
    epochs = 10

    hidden_d = 5
    lrs = [1, 1e-2, 1e-3, 1e-8]
    losses_by_lr = {}
    test_mse_by_lr = {}

    for lr in lrs:
        np.random.seed(0)
        model = Model(input_d, hidden_d, output_d, loss_type="mse")

        train_losses = safe_train(model, X_train, y_train, lr, epochs)
        losses_by_lr[lr] = train_losses

        # Predict (standardized), then unscale to dollars
        y_pred_test_std = model.forward(X_test)
        y_pred_test = unscale_y(y_pred_test_std, y_mean, y_std)

        y_test_dollars = unscale_y(y_test, y_mean, y_std)
        test_mse_by_lr[lr] = mse(y_pred_test, y_test_dollars)

    plot_lr_sweep(losses_by_lr, "housing_part2a_lr_sweep.png")

    print("\n[Problem 2(c)] Test MSE by LR (Hidden=5) [in original dollars^2]:")
    for lr in lrs:
        v = test_mse_by_lr[lr]
        print(f"  LR={lr:g}  Test MSE={v:.6f}" if np.isfinite(v) else f"  LR={lr:g}  Test MSE=nan/inf (diverged)")

    lr = 1e-2
    hidden_sizes = [2, 8, 16, 32]
    losses_by_h = {}
    test_mse_by_h = {}

    for h in hidden_sizes:
        np.random.seed(0)
        model = Model(input_d, h, output_d, loss_type="mse")

        train_losses = safe_train(model, X_train, y_train, lr, epochs)
        losses_by_h[h] = train_losses

        y_pred_test_std = model.forward(X_test)
        y_pred_test = unscale_y(y_pred_test_std, y_mean, y_std)
        y_test_dollars = unscale_y(y_test, y_mean, y_std)
        test_mse_by_h[h] = mse(y_pred_test, y_test_dollars)

    plot_hidden_sweep(losses_by_h, "housing_part2d_hidden_sweep.png")

    print("\n[Problem 2(e)] Test MSE by Hidden Size (LR=1e-2) [in original dollars^2]:")
    for h in hidden_sizes:
        v = test_mse_by_h[h]
        print(f"  H={h:2d}  Test MSE={v:.6f}" if np.isfinite(v) else f"  H={h:2d}  Test MSE=nan/inf (diverged)")

    print("\nSaved plots:")
    print("  - housing_part2a_lr_sweep.png")
    print("  - housing_part2d_hidden_sweep.png")

if __name__ == "__main__":
    main()