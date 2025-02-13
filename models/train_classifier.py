import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, hamming_loss, recall_score

def load_data(database_filepath):
    '''Loads database file
        
    Args:
        database_filepath (str)
    
    Returns:
        X (pandas.DataFrame)
        Y (pandas.DataFrame)
        category_names (Index)
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)    
    
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis='columns')
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''Returns cleaned tokens from text
        - Removes punctuation
        - Lemmatizes tokens
        - Converts to lowercase and removes extra whitespace
    Args: 
        text (str)
    Returns:
        clean_tokens (list): list of tokens
    '''
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # initalize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
        
    return clean_tokens


def build_model():
    '''Builds a parameter optimized model
    
    Returns:
        cv (GridSearchCV object)
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'vect__stop_words':['english'],
        'vect__ngram_range':[(1,2)],
        'vect__max_features':[40000],
        'clf__estimator__learning_rate':[0.75, 1.0],
        'clf__estimator__n_estimators':[50, 75]
    }
    
    cv = GridSearchCV(pipeline, parameters, scoring='recall_macro', cv=3)    
    return cv


def get_scores(scorer, y_true, y_pred, average):
    '''Runs the scorer on all the paired columns
    
    Args:
        scorer: The scoring function to use
        y_true (pandas.DataFrame)
        y_pred (pandas.DataFrame)
        average (str): the average parameter passed to scorer
    
    Returns:
        scores (list): list of individual scores or tuples depending on
            which average option was used
    '''
    scores = []
    for i in range(0, y_true.shape[1]):
        score = scorer(y_true.iloc[:,i], y_pred.iloc[:,i], average=average)
        scores.append(score)
    return scores


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates a multi-label classifier and prints the results
    
    Args:
        model: classifier model
        X_test: test data
        Y_test (pandas.DataFrame): test labels
        category_names: names for test label columns
    '''
    Y_pred = pd.DataFrame(model.predict(X_test))
    
    for i in range(Y_test.shape[1]):
        print('Column {}, {}'.format(i, category_names[i]))
        print(classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i]))
    
    average = None # reports only pos_label
    scorer = recall_score
    scores = get_scores(scorer, Y_test, pd.DataFrame(Y_pred), average)
    print(scorer.__name__, average, np.mean(scores))
    
    print('Hamming loss: ', hamming_loss(Y_test, Y_pred))
    return None


def save_model(model, model_filepath):
    '''Saves the model as a pickle file'''
    try:
        f = open(model_filepath, 'wb')
        pickle.dump(model, f)
        return True
    except IOError:
        return False


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
        if save_model(model, model_filepath):
            print('Trained model saved!')
        else:
            print('Trained model NOT saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()