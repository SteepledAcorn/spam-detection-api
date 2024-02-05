import os
import logging
import yaml
from spam_detection_api.classifiers.classifier import *
from spam_detection_api.preprocessing.data_loading import *
from spam_detection_api.preprocessing.preprocessor import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('training_pipeline')


def preprocess_and_split_data(raw_data_path, target_column, email_column):
    """
    Function to encapsulate all data preprocessing
    """
    logger.info(f"Fetching data at {raw_data_path}")
    data = DataLoader(raw_data_path).get_data()

    class_distribution = data[target_column].value_counts()
    if len(set(class_distribution)) > 1:
        logger.warning(f"Possible dataset imbalance. See target value counts:\n {class_distribution}")

    logger.info("Preprocessing data")
    preprocessed_data = preprocess_dataset(data, target_column, email_column)
    X = preprocessed_data[f'processed_{email_column}']
    y = preprocessed_data[f'processed_{target_column}']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test



def train_and_test_classifier(classifier, X_train, X_test, y_train, y_test):
    """
    Function which trains, tests and saves given classifier
    """
    classifier_type = type(classifier).__name__

    logger.info(f"Training {classifier_type} classifer")
    classifier.train(X_train, y_train)

    # Evaluate the model performance
    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    logger.info(f"{classifier_type} Classification Report:\n {report}")



def create_filename_with_versioning(folder_path, file_name):
    """
    Create a filename with versioning to avoid overwriting existing files.
    
    folder_path: The path to the folder where the file will be saved.
    file_name: The original name of the file (without versioning).
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)
            
    # Construct the full path for the file
    base_name, extension = os.path.splitext(file_name)
    full_path = os.path.join(folder_path, file_name)
    
    # Initialize the version number
    version = 1
    versioned_file_name = f"{base_name}_v{version}{extension}"
    full_path = os.path.join(folder_path, versioned_file_name)
    
    # Check if the file exists and update the name with a version number if it does
    while os.path.exists(full_path):
        versioned_file_name = f"{base_name}_v{version}{extension}"
        full_path = os.path.join(folder_path, versioned_file_name)
        version += 1
    
    return full_path





if __name__=='__main__':
    
    # Set config values
    with open('./config.yaml', "r") as file:
        config = yaml.safe_load(file)

    data_path = config['data_path']
    model_directory = config['model_directory']
    target_column = config['target_column']
    email_column = config['email_column']

    classifiers = get_valid_classifiers()
    
    X_train, X_test, y_train, y_test = preprocess_and_split_data(data_path, target_column, email_column)

    logger.info(f"Training classifer(s): {classifiers.keys()}")

    for classifier_type in classifiers.keys():
        classifier = classifiers[classifier_type]
        train_and_test_classifier(classifier, X_train, X_test, y_train, y_test)

        model_dir = os.path.join(model_directory, classifier_type)
        logger.info(f"Saving model and vectorizer to {model_dir}")

        model_path = create_filename_with_versioning(model_dir, 'model.pkl')
        vectorizer_path = create_filename_with_versioning(model_dir, 'vectorizer.pkl')
        classifier.save_model(model_path, vectorizer_path)
