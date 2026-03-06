import matplotlib.pyplot as plt
import numpy as np
from hw1 import getting_iris_data, Model

X_train, y_train, X_test, y_test = getting_iris_data()

input_d = X_train.shape[1]
output_d = y_train.shape[1]
epochs = 10
learning_rate = 1e-2
hidden_sizes = [2, 8, 16, 32]

all_losses = {}
results = {}

# Fix seed so only hidden size changes
np.random.seed(0)

for h in hidden_sizes:
    model = Model(input_d, h, output_d, loss_type="cross_entropy")
    
   # Train
    model.train(
        data_matrix_x=X_train,
        y_labels=y_train,
        learning_rate=learning_rate,
        epochs=epochs
    )
    

    probs = model.forward(X_test)
    eps = 1e-12
    N = y_test.shape[0]
    test_loss = -np.sum(y_test * np.log(probs + eps)) / N
    

    pred_labels = np.argmax(probs, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(pred_labels == true_labels)
    
    results[h] = (test_loss, accuracy)


print("\nTest Set Performance (LR = 1e-2)")
for h in hidden_sizes:
    loss, acc = results[h]
    print(f"Hidden Units: {h:2d} | Test Loss: {loss:.6f} | Test Accuracy: {acc:.4f}")