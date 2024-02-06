### Problem Statement

Develop a software solution that can effectively identify and classify SMS messages as either 'spam' or 'not spam'. This involves creating a machine learning model or algorithm that can analyze the content of an SMS message and make a prediction based on its understanding of what constitutes spam. The solution should be able to learn from a dataset of SMS messages that have been pre-classified as 'spam' or 'not spam', and apply this learning to new, unseen messages. The ultimate goal is to reduce the amount of spam messages that reach users' inboxes by accurately identifying and filtering out spam.

### The Data

The data is a set of email subjects and a binary spam classification. We will withhold some data for validation of your model.

Columns included are as follows:

- Label - `spam`/`ham`

- Text - Content of the email subject

### Setup
1. Create a python virtual environment with version 3.11 or higher. Activate the environment (pyenv or conda etc.)
2. Install dependencies `pip install requirements.txt`

### Running the API
There are already pre-trained models provided in this repository so there is no need to train. By default Flask runs on port 5000.

1. Navigate to the `spam_detection_api` folder.
2. Run `python app.py`

A successful running app log should look like this:

```
$ python app.py 
[nltk_data] Downloading package wordnet to /home/adser/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
 * Serving Flask app 'app'
 * Debug mode: on
INFO:werkzeug:WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
INFO:werkzeug:Press CTRL+C to quit
INFO:werkzeug: * Restarting with stat
[nltk_data] Downloading package wordnet to /home/adser/nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
WARNING:werkzeug: * Debugger is active!
INFO:werkzeug: * Debugger PIN: 964-575-855
INFO:werkzeug:127.0.0.1 - - [05/Feb/2024 17:44:47] "POST /predict HTTP/1.1" 200 -
INFO:werkzeug:127.0.0.1 - - [05/Feb/2024 17:44:51] "POST /predict HTTP/1.1" 200 -
```

Once the app is running, you can make queries to the API via curl commands in a seperate terminal. If no model is specified it uses the `default_model_type` set in the config.yaml file. Please see below for some examples.

```
$ curl -X POST -H "Content-Type: application/json" -d '{"message":["Free Bitcoin!", "Hey are you free for a meeting today?"]}' http://127.0.0.1:5000/predict

{
  "messages": [
    "Free Bitcoin!",
    "Hey are you free for a meeting today?"
  ],
  "model": "naive_bayes",
  "predictions": [
    "Spam",
    "Not Spam"
  ]
}
```

Note: the API accepts both a single string message or a list of messages. Note the specification of a different model in this call.

```
$ curl -X POST -H "Content-Type: application/json" -d '{"message": "Free Bitcoin!", "model":"xgboost"}' http://127.0.0.1:5000/predict

{
  "messages": [
    "Free Bitcoin!"
  ],
  "model": "xgboost",
  "predictions": [
    "Spam"
  ]
}
```

### Model Training (Optional)
1. Navigate to the `spam_detection_api` folder. All scripts need to be ran from this working directory.
2. To train the various spam classifiers run `python training_pipeline.py`. This will save both the model & vectorizers to the relative folders in `./saved_models`.

### Project layout

```
spam_detection_api
    ├
    ├── classifiers
    │   ├── __init__.py
    │   ├── classifier.py
    │   └── template_classifier.py
    ├── preprocessing
    │   ├── __init__.py
    │   ├── data_loading.py
    │   └── preprocessor.py
    ├── saved_models
    │   ├── naive_bayes
    │   │   ├── model_v1.pkl
    │   │   └── vectorizer_v1.pkl
    │   └── xgboost
    │       ├── model_v1.pkl
    │       └── vectorizer_v1.pkl
    ├── training_pipeline.py
    ├── app.py
```

### Adding classifiers
The `classifier` file contains the various models which can be used to predict spam. These are modular and classifiers can be added very easily by inheriting from the `TemplateClassifier` object. Once a new classifier has been created, simply add it to the dictionary in `classifiers.get_valid_classifiers()`
