import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to apply Bag of Words
def bag_of_words(text_data, max_features=2000):
    """
    Converts a collection of text documents to a matrix of token counts.
    """
    vectorizer = CountVectorizer(max_features=max_features, stop_words=stop_words)
    X = vectorizer.fit_transform(text_data).toarray()
    return X, vectorizer

# Function to apply TF-IDF
def tfidf(text_data, max_features=2000):
    """
    Converts a collection of text documents to a matrix of TF-IDF features.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X = vectorizer.fit_transform(text_data).toarray()
    return X, vectorizer

# Function to apply Latent Semantic Analysis (LSA) for dimensionality reduction
def apply_lsa(X, n_components=100):
    """
    Apply Latent Semantic Analysis (LSA) to reduce the dimensionality of the feature space.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_lsa = svd.fit_transform(X)
    return X_lsa, svd

# Function to generate advanced NLP features using Word2Vec
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

def word2vec(text_data, embedding_dim=100):
    """
    Converts text data to Word2Vec embeddings.
    """
    tokenized_data = [word_tokenize(text) for text in text_data]
    model = Word2Vec(tokenized_data, vector_size=embedding_dim, window=5, min_count=1, workers=4)
    
    word_vectors = []
    for text in text_data:
        vector = np.mean([model.wv[word] for word in word_tokenize(text) if word in model.wv], axis=0)
        if vector is None:
            vector = np.zeros(embedding_dim)
        word_vectors.append(vector)
    
    return np.array(word_vectors), model

