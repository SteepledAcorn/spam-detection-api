from flask import Flask, request, jsonify
from spam_detection_api.classifiers.classifier import *
from spam_detection_api.preprocessing.preprocessor import preprocess_email
import logging
import os
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)



def load_model(
        model_directory: str,
        classifiers_dict: dict,
        model_name: str, 
        version: int = 1,
        ):
    """
    Function to dynamically load model
    """
    model_path = os.path.join(model_directory, model_name, f'model_v{version}.pkl')
    vectorizer_path = os.path.join(model_directory, model_name, f'vectorizer_v{version}.pkl')

    classifier = classifiers_dict[model_name]

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        classifier.load_model(model_path, vectorizer_path)
        return classifier
    else:
        raise Exception("Error loading model and vectorizer. Path not found")
    


@app.route('/predict', methods=['POST'])
def predict():
    """
    API to predict if a given email or emails are spam or not.

    Example API request body: {"message":["Free money!", "How are you?"], "model":"naive_bayes"}
    """

    # Data validation
    if not request.is_json:
        return jsonify({'Error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if 'message' not in data:
        return jsonify({'Error': 'Missing required attribute: message'}), 400

    # API accepts single message or list
    message = data['message']
    if isinstance(message, str):
        messages = [message]
    elif isinstance(message, list):
        messages = message
    else:
        return jsonify({'Error': 'Attribute "message" must be a string or a list of strings'}), 400
    
    # Preprocess message(s)
    preprocessed_messages = [preprocess_email(m) for m in messages]

    # Use default model if not specified
    model_type = data.get('model', default_model_type)
    if model_type not in valid_models:
        valid_models_str = ", ".join(valid_models)
        return jsonify({'Error': f'Model {model_type} not found. Valid model types are: {valid_models_str}'}), 400

    # Make predictions
    try:
        classifier = load_model(model_directory, classifiers, model_type)
        predictions = classifier.predict(preprocessed_messages)
        responses = ['Spam' if prediction == 1 else 'Not Spam' for prediction in predictions]
        
        return jsonify({'messages': messages, 'predictions': responses, 'model': model_type})
    
    except Exception as e:
        logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'Rrror': str(e)}), 500


if __name__ == '__main__':

    # Set config values
    with open('./config.yaml', "r") as file:
        config = yaml.safe_load(file)

    data_path = config['data_path']
    model_directory = config['model_directory']
    default_model_type = config['default_model_type']
    default_model_version = config['default_model_version']
    target_column = config['target_column']
    email_column = config['email_column']

    classifiers = get_valid_classifiers()
    valid_models = list(classifiers.keys())

    # Run app
    app.run(debug=True)
