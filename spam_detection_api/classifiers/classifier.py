from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from spam_detection_api.classifiers.template_classifier import TemplateClassifier
import logging

logger = logging.getLogger(__name__)


class NaiveBayesClassifier(TemplateClassifier):
    def __init__(self,  model_path=None, vectorizer_path=None):
        super().__init__(model_path, vectorizer_path)

        self.vectorizer = CountVectorizer() 
        if self.model_path and self.vectorizer_path:
            self.load_model(self.model_path, self.vectorizer_path)

    def train(self, X, y):
        self.classifier = MultinomialNB()
        X_vectorized = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vectorized, y)

    def predict(self, texts):
        # NB: this if statement is key for all classifiers
        if self.classifier is None:
            logger.warning("Classifier not loaded, attempting to load now.")
            if self.model_path and self.vectorizer_path:
                self.load_model(self.model_path, self.vectorizer_path)
            else:
                raise Exception("Model and vectorizer paths are not set. Cannot load the model.")
        texts_vectorized = self.vectorizer.transform(texts)
        return self.classifier.predict(texts_vectorized)
    


class XGBoostClassifier(TemplateClassifier):
    def __init__(self,  model_path=None, vectorizer_path=None):
        super().__init__(model_path, vectorizer_path)

        self.vectorizer = TfidfVectorizer() 
        if self.model_path and self.vectorizer_path:
            self.load_model(self.model_path, self.vectorizer_path)

    def train(self, X, y):
        self.classifier = XGBClassifier(max_depth=5, n_estimators=500)
        X_vectorized = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_vectorized, y)


    def predict(self, texts):
        # NB: this if statement is key for all classifiers
        if self.classifier is None:
            logger.warning("Classifier not loaded, attempting to load now.")
            if self.model_path and self.vectorizer_path:
                self.load_model(self.model_path, self.vectorizer_path)
            else:
                raise Exception("Model and vectorizer paths are not set. Cannot load the model.")
        texts_vectorized = self.vectorizer.transform(texts)
        return self.classifier.predict(texts_vectorized)
    


def get_valid_classifiers():
    """
    Helper function to store valid classifiers.
    """
    valid_classifiers = {
        'naive_bayes': NaiveBayesClassifier(),
        'xgboost': XGBoostClassifier()
    }

    return valid_classifiers