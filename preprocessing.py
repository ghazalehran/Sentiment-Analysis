import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Function to clean text data
def clean_text(text):
    """
    Cleans text by removing special characters, numbers, and unnecessary whitespaces.
    """
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = text.lower()
    return text.strip()

# Function to load and preprocess data
def preprocess_text(train_data_dir, train_labels_dir, test_data_dir, max_len, tensor=True):
    """
    Loads, cleans, and tokenizes the training and test data. Pads the sequences and encodes labels.
    """
    # Load datasets
    train_data = pd.read_csv(train_data_dir)
    train_labels = pd.read_csv(train_labels_dir)
    test_data = pd.read_csv(test_data_dir)
    
    # Clean the text data
    train_data['cleaned_text'] = train_data['text'].apply(clean_text)
    test_data['cleaned_text'] = test_data['text'].apply(clean_text)
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels['sentiment'])
    
    # Create a CountVectorizer to tokenize the text
    vectorizer = CountVectorizer(max_features=2000)
    x_train = vectorizer.fit_transform(train_data['cleaned_text']).toarray()
    x_test = vectorizer.transform(test_data['cleaned_text']).toarray()
    
    # Padding sequences to max_len
    x_train = pad_sequences(x_train, max_len)
    x_test = pad_sequences(x_test, max_len)
    
    # Convert to tensor if needed
    if tensor:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        x_test = torch.tensor(x_test, dtype=torch.float32)
    
    return x_train, y_train, x_test

# Function to pad sequences
def pad_sequences(sequences, max_len):
    """
    Pads sequences to ensure they all have the same length (max_len).
    """
    padded_sequences = np.zeros((len(sequences), max_len))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq[:max_len]
    return padded_sequences
