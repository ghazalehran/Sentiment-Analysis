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
