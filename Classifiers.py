# classifiers.py

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Naïve Bayes Classifier
def naive_bayes_classifier(X_train, y_train, X_test):
    """
    Trains a Naïve Bayes classifier on the provided data and returns predictions.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
    
    Returns:
        array: Predicted labels for test data.
    """
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


# Support Vector Machine (SVM) Classifier
def svm_classifier(X_train, y_train, X_test):
    """
    Trains a Kernelized SVM classifier on the provided data and returns predictions.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
    
    Returns:
        array: Predicted labels for test data.
    """
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


# Neural Network Classifier (MLP)
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Initializes the Neural Network model with given parameters.
        
        Args:
            input_size (int): Size of input features.
            hidden_sizes (list): List of sizes of hidden layers.
            output_size (int): Number of output classes (sentiment categories).
        """
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)


def train_neural_network(X_train, y_train, X_test, batch_size=64, epochs=10, learning_rate=0.001):
    """
    Trains a Neural Network classifier on the provided data and returns predictions.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        batch_size (int): Size of mini-batches for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for optimizer.
    
    Returns:
        array: Predicted labels for test data.
    """
    # Convert numpy arrays to torch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)

    # Create DataLoader for batching
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MLPClassifier(input_size=X_train.shape[1], hidden_sizes=[64, 32, 16], output_size=3)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    # Test the model
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = torch.argmax(y_pred, dim=1)
    
    return y_pred.numpy()


# Combine all classifiers and train them
def classify_text(X_train, y_train, X_test, method='svm'):
    """
    Classifies the text samples into sentiment categories using the specified method.
    
    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_test (array-like): Test features.
        method (str): Classification method. Choices: 'svm', 'naive_bayes', 'mlp'.
    
    Returns:
        array: Predicted labels for test data.
    """
    if method == 'naive_bayes':
        return naive_bayes_classifier(X_train, y_train, X_test)
    elif method == 'svm':
        return svm_classifier(X_train, y_train, X_test)
    elif method == 'mlp':
        return train_neural_network(X_train, y_train, X_test)
    else:
        raise ValueError("Invalid method. Choose from 'svm', 'naive_bayes', or 'mlp'.")


# Example usage
if __name__ == "__main__":
    # Example data (replace with actual feature and label data)
    X_train = np.random.rand(1000, 50)  # 1000 samples, 50 features
    y_train = np.random.randint(0, 3, size=1000)  # 3 classes (0, 1, 2)
    X_test = np.random.rand(500, 50)   # 500 samples, 50 features
    
    # Classify using SVM
    y_pred_svm = classify_text(X_train, y_train, X_test, method='svm')
    print("SVM Predictions:", y_pred_svm)

    # Classify using Naive Bayes
    y_pred_nb = classify_text(X_train, y_train, X_test, method='naive_bayes')
    print("Naive Bayes Predictions:", y_pred_nb)

    # Classify using MLP
    y_pred_mlp = classify_text(X_train, y_train, X_test, method='mlp')
    print("MLP Predictions:", y_pred_mlp)
