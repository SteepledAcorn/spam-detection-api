from abc import ABC, abstractmethod
import joblib

class TemplateClassifier(ABC):
    """
    Class to define the required structure of a spam prediction classifier.
    
    Every classifier may have its own unique vectorizer and train/predict function.

    Model save/load is universal.
    """
    def __init__(self, model_path=None, vectorizer_path=None):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.classifier = None
        self.vectorizer = None
    
    @abstractmethod
    def train(self, X, y):
        """
        Function which trains the classifier.
        """
        pass

    @abstractmethod
    def predict(self, texts: list):
        """
        Predict spam/ham for given texts samples.
        """
        pass

    def save_model(self, model_path, vectorizer_path):
        if (self.classifier is None) or (self.vectorizer is None):
            raise Exception("Model and vectorizer not set, cannot save.")
        else:
            joblib.dump(self.classifier, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path, vectorizer_path):
        self.classifier = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)