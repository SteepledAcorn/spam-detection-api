
from sklearn.preprocessing import LabelBinarizer
import re
import pandas as pd
import os

import nltk

required_corpa = ['stopwords', 'wordnet']
for corpus in required_corpa:
    if corpus not in os.listdir(nltk.data.find('corpora')):
        nltk.download(corpus)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def preprocess_email(email: str) -> str:
    """
    Clean emails by removing punctuation, stopwords, lowercasing, lemmatizing
    """

    remove_stopwords = set(stopwords.words('english'))

    email = re.compile('[^a-zA-Z]').sub(' ', email)
    email = email.lower()
    email_words = email.split()

    lemmatizer = WordNetLemmatizer()
    email_words = [lemmatizer.lemmatize(word) for word in email_words if word not in remove_stopwords]
    processed_email = ' '.join(email_words)
    
    return processed_email


def preprocess_dataset(
        data: pd.DataFrame,
        target_column: str,
        email_column: str
        ) -> pd.DataFrame:
    """
    Drop missing data, encode target variable, clean email column
    """
    
    data = data.dropna()
    # data = encode_target_variable(data, target_column)

    binarizer = LabelBinarizer()
    data[f'processed_{target_column}'] = binarizer.fit_transform(data[target_column])

    if email_column not in data.columns:
        raise ValueError(f"DataFrame must contain a {email_column} column")
    data[f'processed_{email_column}'] = data[email_column].apply(preprocess_email)

    return data

