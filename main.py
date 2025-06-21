import numpy as np
import pandas as pd
from gensim.downloader import load as gensim_load
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- HyperParameters ---
EMBEDDING_DIM = 50
EPOCHS = 20
LR = 0.05
HIDDEN_SIZE = 16

# --- GloVe Word Vectors ---
print("Loading GloVe vectors...")
glove_model = gensim_load("glove-wiki-gigaword-50")  # size: 50

# --- Upload the Dataset  ---
print("Reading dataset...")
df = pd.read_csv("IMDB Dataset.csv")
texts = df["review"].astype(str).tolist()
labels = df["sentiment"].astype(int).tolist()

# --- Vectorization ---
def text_to_vec(text):
    words = text.lower().split()
    vectors = [glove_model[word] for word in words if word in glove_model]
    if not vectors:
        return np.zeros(EMBEDDING_DIM)
    return np.mean(vectors, axis=0)

print("Vectorizing...")
X_vec = np.array([text_to_vec(t) for t in texts])
y_vec = np.array(labels).reshape(-1, 1)

# --- Training and Testing ---
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.2, random_state=42)

# --- Weights ---
np.random.seed(42)
W1 = np.random.randn(HIDDEN_SIZE, EMBEDDING_DIM) * 0.1
b1 = np.zeros((HIDDEN_SIZE, 1))
W2 = np.random.randn(1, HIDDEN_SIZE) * 0.1
b2 = np.zeros((1, 1))

# --- Activation Function (sigmoid and ReLU) ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

# --- Training ---
loss_history = []

print("Training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(X_train.shape[0]):
        x = X_train[i].reshape(-1, 1)
        y = y_train[i]

        # Forward pass
        z1 = W1 @ x + b1
        a1 = relu(z1)
        z2 = W2 @ a1 + b2
        a2 = sigmoid(z2)

        # Loss (binary cross-entropy)
        loss = - (y * np.log(a2 + 1e-9) + (1 - y) * np.log(1 - a2 + 1e-9))
        total_loss += loss.item()

        # Backward pass
        dz2 = a2 - y
        dW2 = dz2 @ a1.T
        db2 = dz2

        dz1 = (W2.T @ dz2) * relu_deriv(z1)
        dW1 = dz1 @ x.T
        db1 = dz1

        # Update
        W2 -= LR * dW2
        b2 -= LR * db2
        W1 -= LR * dW1
        b1 -= LR * db1

    loss_history.append(total_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# --- Testing ---
def predict(x):
    z1 = W1 @ x.reshape(-1, 1) + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)
    return 1 if a2 > 0.5 else 0

y_pred = np.array([predict(x) for x in X_test])
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")


# --- Loss Graph ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS+1), loss_history, marker='o')
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.grid(True)
plt.tight_layout()
plt.show()