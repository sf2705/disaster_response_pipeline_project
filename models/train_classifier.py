# import libraries
import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

#from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from custom_transformer import StartingVerbExtractor

import sys


def load_data(database_filepath):
    '''
    Input: database_filepath - str - path to the database
    Output: X - numpy array - messages
            y - numpy array - dummies for categories
            category_names - list - categories
    '''

    # connect to database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)  

    # Get X and y for the model
    X = df['message'].values
    y = df.drop(columns = ['id', 'message', 'original', 'genre']).values
    category_names = df.drop(columns = ['id', 'message', 'original', 'genre']).columns
    return X, y, category_names


def tokenize(text):
    '''
    Input: text - str - message to process
    Output: clean_tokens - list - list of tokens after cleaned
    '''

    # regular expression for urls
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    
    # find all urls and replace with urlplaceholder
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # separate text to tokens and limmatize, change to lower case and get rid of spaces
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    '''
    Input: None
    Output: cv - grid search object
    '''

    # building pipline with Feature Union
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__max_df': (0.5, 0.75)
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Input: model - grid search object
           X_test - numpy array
           Y_test - numpy array
           category_name - list
    Output: None
    '''

    Y_pred = model.predict(X_test)
    Y_test_transpose = Y_test.T
    Y_pred_transpose = Y_pred.T
    for i in range(len(Y_test_transpose)):
        print(category_names[i], classification_report(Y_test[i], Y_pred[i]))
    return 
    


def save_model(model, model_filepath):
    '''
    Input: model - grid search object
           model_filepath - string
    Output: None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()