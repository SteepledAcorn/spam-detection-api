import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from spam_detection_api.classifiers.base_classifier import BaseClassifier

class NaiveBayesClassifier(BaseClassifier):
    def __init__(self, model_path=None, vectorizer_path=None):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = CountVectorizer() ## ?
        self.classifier = None ## remove
        if model_path and vectorizer_path:
            self.load_model(model_path, vectorizer_path)

    def train(self, X, y):
        self.classifier = MultinomialNB()
        X_vectorized = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vectorized, y)

    def predict(self, texts):
        if self.classifier is None:
            if self.model_path and self.vectorizer_path:
                self.load_model(self.model_path, self.vectorizer_path)
            else:
                raise Exception("Model and vectorizer paths are not set. Cannot load the model.")
        texts_vectorized = self.vectorizer.transform(texts)
        return self.classifier.predict(texts_vectorized)

    def save_model(self, model_path, vectorizer_path):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path, vectorizer_path):
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
