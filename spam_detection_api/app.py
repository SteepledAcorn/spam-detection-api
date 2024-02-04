from flask import Flask, request, jsonify
from spam_detection_api.classifiers.classifier import *
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model_directory = './saved_models/naive_bayes'
classifier = NaiveBayesClassifier(model_directory+'/naive_bayes_model_v1.pkl', model_directory+'/naive_bayes_vectorizer_v1.pkl')

logger.info("Classifier loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        message = data['message']
        prediction = classifier.predict([message])
        response = 'Spam' if prediction[0] == 1 else 'Not Spam'
        return jsonify({'message': message, 'prediction': response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
