# Sentiment Analysis with Feedforward Neural Network

## 1. Overview

This project demonstrates how to build a simple feedforward neural network (FNN) from scratch using NumPy for binary sentiment classification (positive vs. negative movie reviews). Instead of using engineered features, it leverages pretrained word embeddings to represent text inputs, mimicking modern NLP pipelines.

The model is trained using a manually labeled dataset stored in CSV format and uses the GloVe pretrained word vectors for semantically meaningful representations.

---

## 2. Architecture Summary

- **Input**: Movie review (text)
- **Embedding Layer**: GloVe word vectors (50-dimensional)
- **Mean Pooling**: Average of all word vectors in the review
- **Hidden Layer**: ReLU activation
- **Output Layer**: Sigmoid activation
- **Loss Function**: Binary cross-entropy
- **Training Method**: Manual implementation of forward and backward propagation

---

## 3. Dataset Description

You are expected to supply a CSV file named `dataset.csv` with the following structure:

```
review,sentiment
"I love this movie",1
"This movie is terrible",0
```

- `review`: A string containing the user review.
- `sentiment`: Label where 1 = positive, 0 = negative.

The dataset is split into training and testing sets using an 80/20 ratio.

---

## 4. Word Embeddings (GloVe)

The model uses pretrained GloVe vectors to convert each word into a fixed-length vector. These vectors capture semantic and syntactic relationships.

To load them, we use:

```python
from gensim.downloader import load
glove_model = load("glove-wiki-gigaword-50")
```

This loads 50-dimensional vectors for ~400,000 words trained on Wikipedia + Gigaword.

---

## 5. Network Architecture in Detail

### Feedforward Structure:

```
Input: x ∈ ℝ^50  (mean word embedding)
Hidden Layer: h = ReLU(W1·x + b1)  → ℝ^16
Output Layer: ŷ = sigmoid(W2·h + b2)  → ℝ
Loss: Binary Cross-Entropy
```

- All weights and biases are initialized using small random values.
- Gradients are manually calculated and weights updated using simple gradient descent.

---

## 6. Training Loop

For each epoch:
- Forward pass: compute predictions
- Loss calculation: binary cross-entropy
- Backward pass: compute gradients
- Parameters update: apply gradients

Epoch-wise total loss is logged and stored for visualization.

---

## 7. Training Loss Graph

After training, the model plots the total training loss across epochs using Matplotlib.

Example output:

![Training Loss Plot](assets/loss_plot_example.png)

---

## 8. Usage Instructions

### Dependencies:

Install required libraries:
```bash
pip install numpy pandas gensim scikit-learn matplotlib
```

### Run the model:
Ensure your CSV file is named `IMDB Dataset.csv` and placed in the same directory as the script.

Run the script:

```bash
python sentiment_nn_from_csv.py
```

This will:
- Load data and GloVe vectors
- Train the model
- Plot training loss
- Evaluate model on the test set

---

## 9. Sample Output

```
Epoch 1/20, Loss: 148.0927
Epoch 2/20, Loss: 132.4103
...
Test Accuracy: 0.8450
```

---

## 10. Sources & References

- GloVe Embeddings: https://nlp.stanford.edu/projects/glove/
- Movie Reviews Dataset: https://www.cs.cornell.edu/people/pabo/movie-review-data/
- NumPy Documentation: https://numpy.org/
- Matplotlib Documentation: https://matplotlib.org/
- Gensim Downloader: https://radimrehurek.com/gensim/

---

## Notes

This implementation avoids using high-level libraries like TensorFlow or PyTorch to better illustrate how forward and backward propagation work in neural networks.
