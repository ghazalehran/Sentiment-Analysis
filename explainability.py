# explainability.py

import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# LIME Explainer Function
def explain_prediction_with_lime(model, vectorizer, X_train, y_train, X_test, index=0, num_features=10):
    """
    Explains a prediction using LIME for a given instance.
    
    Args:
        model (object): Trained model (e.g., Naive Bayes, SVM, MLP).
        vectorizer (object): Text vectorizer (e.g., CountVectorizer, TfidfVectorizer).
        X_train (array-like): Training data.
        y_train (array-like): Labels for the training data.
        X_test (array-like): Test data to explain.
        index (int): Index of the instance in the test set to explain.
        num_features (int): Number of features to include in the explanation.
        
    Returns:
        None
    """
    # Create the LIME text explainer
    explainer = lime.lime_text.LimeTextExplainer(class_names=np.unique(y_train))
    
    # Create a prediction function that applies the vectorizer and then the model
    def predict_proba(texts):
        # Convert the text data using the vectorizer
        text_features = vectorizer.transform(texts)
        return model.predict_proba(text_features)
    
    # Explain the prediction for the selected instance
    explanation = explainer.explain_instance(X_test[index], predict_proba, num_features=num_features)
    
    # Show the explanation in a readable way
    explanation.show_in_notebook(text=True)
    explanation.as_pyplot_figure()

    # Plot the feature importance
    feature_importance = explanation.as_list()
    features, importances = zip(*feature_importance)
    
    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title('LIME Feature Importance for Text Classification')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()


# Example usage of LIME with different classifiers
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample dataset (replace with your actual dataset)
    text_data = ["I love this movie", "This was a terrible film", "Great plot, but bad acting", "Worst movie ever", "Amazing performance"]
    labels = [2, 0, 1, 0, 2]  # 0: negative, 1: neutral, 2: positive
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(text_data, labels, test_size=0.2, random_state=42)
    
    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train a classifier (for example, Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Explain a prediction for a test instance (example index 0)
    explain_prediction_with_lime(model, vectorizer, X_train, y_train, X_test, index=0, num_features=10)
