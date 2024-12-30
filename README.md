# Sentiment-Analysis

## Step 1: Data Preprocessing
1. Reading the dataset: Load the CSV files for training and test data.
2. Cleaning and tokenizing text: Remove special characters and tokenize the text data.
3. Building vocabularies: Generate vocabularies from the training and test sets.
4. Padding sequences: Pad tokenized sentences to the same length (max_len).
5. Label encoding: Convert sentiment labels into numerical values (negative -> 0, neutral -> 1, positive -> 2).

## Step 2: Feature Engineering 
#### 1. **Bag of Words (BoW)**
The `bag_of_words` function converts text data into a matrix of token counts. It uses the **CountVectorizer** from `sklearn` and creates a feature vector based on the frequency of words in the text.

#### Parameters:
- `text_data`: List of text documents.
- `max_features`: The maximum number of features (words) to consider (default is 2000).

#### 2. **TF-IDF (Term Frequency-Inverse Document Frequency)**

The `tfidf` function converts text data into a matrix of TF-IDF features using the **TfidfVectorizer** from `sklearn`.

#### Parameters:
- `text_data`: List of text documents.
- `max_features`: The maximum number of features (words) to consider (default is 2000).

#### 3. **Latent Semantic Analysis (LSA)**

The `apply_lsa` function reduces the dimensionality of the feature space using **Latent Semantic Analysis** with **TruncatedSVD** from `sklearn`.

#### Parameters:
- `X`: The input feature matrix (e.g., from TF-IDF).
- `n_components`: The number of components (dimensions) for LSA (default is 100).

### 4. **Word2Vec**

The `word2vec` function generates word embeddings using the **Word2Vec** algorithm from `gensim`, representing each document as an average of its word embeddings.

#### Parameters:
- `text_data`: List of text documents.
- `embedding_dim`: The dimensionality of the word vectors (default is 100).

### Notes
- Bag of Words (BoW) and TF-IDF are basic techniques used for transforming text into numerical features based on word frequency.
- LSA helps reduce the dimensionality of features by capturing latent structures in the text data.
- Word2Vec generates word embeddings, capturing semantic relationships between words.

## Step 3:Classifiers for Sentiment Analysis

We use three advanced machine learning models used for text classification into sentiment categories (e.g., negative, neutral, positive). These models include:

- **Naïve Bayes**
- **Support Vector Machine (SVM)**
- **Neural Network (MLP)**


### 1. **naive_bayes_classifier(X_train, y_train, X_test)**
   - Trains a Naïve Bayes classifier on the provided data and returns predictions.
   - **Parameters**: 
     - `X_train` (array-like): Training features.
     - `y_train` (array-like): Training labels.
     - `X_test` (array-like): Test features.
   - **Returns**: Predicted labels for the test data.

### 2. **svm_classifier(X_train, y_train, X_test)**
   - Trains a Kernelized SVM classifier on the provided data and returns predictions.
   - **Parameters**: 
     - `X_train` (array-like): Training features.
     - `y_train` (array-like): Training labels.
     - `X_test` (array-like): Test features.
   - **Returns**: Predicted labels for the test data.

### 3. **train_neural_network(X_train, y_train, X_test, ...)**
   - Trains a Neural Network (MLP) classifier on the provided data and returns predictions.
   - **Parameters**: 
     - `X_train` (array-like): Training features.
     - `y_train` (array-like): Training labels.
     - `X_test

## Step 4:  Explainability with LIME

`explainability.py` demonstrates how to use **LIME (Local Interpretable Model-agnostic Explanations)** to explain predictions made by machine learning models on text classification tasks (e.g., sentiment analysis). LIME helps make black-box models more transparent and interpretable.

### Functions

#### 1. **explain_prediction_with_lime(model, vectorizer, X_train, y_train, X_test, index=0, num_features=10)**
   - Explains the prediction for a given instance in the test set using LIME.
   - **Parameters**: 
     - `model` (object): Trained classification model (e.g., Naive Bayes, SVM).
     - `vectorizer` (object): The vectorizer used for transforming text data into numerical features.
     - `X_train` (array-like): Training data features.
     - `y_train` (array-like): Training data labels.
     - `X_test` (array-like): Test data features.
     - `index` (int): Index of the instance in the test set to explain.
     - `num_features` (int): Number of features to include in the explanation.
   - **Returns**: None, but the explanation is shown in the notebook and a bar plot is generated.

#### How to Use

1. **Prepare your data**: Use a text dataset with corresponding labels. (Step 1)
2. **Train a model**: Train a classifier (e.g., Naive Bayes, SVM, etc.) on your dataset. (Step 3)
3. **Explain a prediction**: Call the `explain_prediction_with_lime()` function and provide a test sample index to explain.
4. **Visualize the explanation**: The explanation will be displayed both in a notebook-friendly format and as a bar plot.

