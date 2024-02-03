import logging
from spam_detection_api.classifiers.naivebayes_classifier import NaiveBayesClassifier
from spam_detection_api.preprocessing.data_loading import *
from spam_detection_api.preprocessing.preprocessor import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

data_path = './data/raw/dataset.csv'
model_directory = './saved_models/naive_bayes'


if __name__=='__main__':
    
    target_column='label'
    email_column='text'

    logger.info(f"Fetching data at {data_path}")
    data = DataLoader(data_path).get_data()

    logger.info("Preprocessing data")
    preprocessed_data = preprocess_dataset(data, target_column, email_column)

    X = preprocessed_data[f'processed_{email_column}']
    y = preprocessed_data[f'processed_{target_column}']

    # Split the Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Apply SMOTE to generate synthetic samples for the minority class in the training set
    # smote = SMOTE(random_state=42)
    # X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    logger.info("Training NB classifer")
    classifier = NaiveBayesClassifier()
    classifier.train(X_train, y_train)

    # Evaluate the model performance
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Classifier Accuracy: {accuracy}")

    report = classification_report(y_test, y_pred)
    logger.info(f"Classification Report:\n {report}")

    logger.info(f"Saving model and vectorizer to {model_directory}")
    
    classifier.save_model(
        model_path=model_directory+'/naive_bayes_model_v2.pkl',
        vectorizer_path=model_directory+'/naive_bayes_vectorizer_v2.pkl')
