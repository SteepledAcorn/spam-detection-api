from abc import ABC, abstractmethod
import joblib

class BaseClassifier(ABC):
    """
    Class to define the required structure of a spam prediction classifier.
    
    Every classifier will have its own unique train/predict function.

    Model save/load is universal.
    """
    # def __init__(self):
    #     self.classifier = None
    #     self.vectorizer = None
    
    @abstractmethod
    def train(self, X, y):
        """
        Function which trains the classifier.
        """
        pass

    @abstractmethod
    def predict(self, texts: list):
        """
        Given predict spam/ham for given texts samples.
        """
        pass

    # def save_model(self, model_path, vectorizer_path):
    #     joblib.dump(self.classifier, model_path)
    #     joblib.dump(self.vectorizer, vectorizer_path)

    # def load_model(self, model_path, vectorizer_path):
    #     self.classifier = joblib.load(model_path)
    #     self.vectorizer = joblib.load(vectorizer_path)