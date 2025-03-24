import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# Load data frequent words
freq_words = pd.read_csv(r"\frequent_words_clean_10k.csv")
frequent_words = set(freq_words['word'].astype(str).str.strip())

# load articles dataset
articles = pd.read_csv(r"\cleaned_995000_news.csv")
articles.columns = articles.columns.str.strip()
articles['content'] = articles['content'].astype(str)


# Filter only top 10000 words
def filter_article(text, allowed_words):
    tokens = text.split()
    return ' '.join([word for word in tokens if word in allowed_words])

articles['filtered_content'] = articles['content'].apply(lambda x: filter_article(x, frequent_words))
# drop empty 
articles = articles[articles['filtered_content'].str.strip().astype(bool)].reset_index(drop=True)


# TF-IDF vectorization  sparse, every article is a vector of the TF-IDF numerical values only non-zero
vectorizer = TfidfVectorizer(max_features=10000)
X_sparse = vectorizer.fit_transform(articles['filtered_content'])
# label binary values
y = articles['type'].astype(int).values


# Train/Val/Test split
X_train, X_vt, y_train, y_vt = train_test_split(X_sparse, y, test_size=0.2, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_vt, y_vt, test_size=0.5, random_state=0)


# Define neural network (inherited form the torch.nn.module class)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        # Setup in layers
        super().__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        # How the data flows
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

# Model setup
input_size = X_sparse.shape[1]
model = NeuralNetwork(input_size)
# Error function - binary cross entropy 
loss_fn = nn.BCELoss()
# Adam algorithm for stochastic gradient descent, lr = learning rate, weight_decay = penalty for large weights (prevent overfitting)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)


# Model training loop, manual batching, only converting to dense (tensor) when needed
# Batch = articles at a time
batch_size = 128
# Epochs = how many times the model will go though all training data (too many times = over fitting)
epochs = 10

for epoch in range(epochs):
    model.train()
    train_loss = 0
    # Get training data in random order across epochs
    permutation = np.random.permutation(X_train.shape[0])

    for i in range(0, X_train.shape[0], batch_size):
        batch_indices = permutation[i:i + batch_size]
        X_batch_sparse = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # Sparse to dense (just for batch)
        X_batch_dense = torch.tensor(X_batch_sparse.toarray(), dtype=torch.float32)
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

        # Forward pass
        outputs = model(X_batch_dense)
        loss = loss_fn(outputs, y_batch_tensor)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        val_outputs = model(X_val_tensor)
        val_loss = loss_fn(val_outputs, y_val_tensor).item()

    avg_train_loss = train_loss / X_train.shape[0]
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")


# Evaluate / test 
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    y_pred_probs = model(X_test_tensor)
    y_pred_labels = (y_pred_probs >= 0.5).float()

    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_labels.numpy())
    f1 = f1_score(y_test_tensor.numpy(), y_pred_labels.numpy())
    
    print(f'Test Accuracy: {accuracy:.4f} | f1 score: {f1:.4f}')