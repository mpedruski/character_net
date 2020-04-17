import pandas as pd
import numpy as np
import nltk
import logging
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def cross_validation_suite(vectorizer, training_file):
    '''
    model, str -> None
    Takes in a prefit vector space and the path to the file that will be used
    for sentiment analysis traiingin. Fits a number of different
    models to the data, reporting CV scores for each model.
    '''
    training_sentences = pd.read_csv(training_file)
    training_features = training_sentences['Text']
    training_sentence_labels = training_sentences['Label']

    features = vectorizer.transform(training_features).toarray()
    np.random.seed(17)

    ### Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, training_sentence_labels, test_size=0.3, stratify = training_sentence_labels, random_state=0)

    ### Balance the classes in the training data
    rus = RandomUnderSampler(sampling_strategy = 'not minority',random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    logging.debug('Resampled dataset shape {}'.format(Counter(y_res)))
    ### For the moment balanced data yields markedly worse CV scores than
    ### complete but imbalanced data for all but Multinomial Bayes

    models = [RandomForestClassifier(n_estimators=200, random_state=0),
        MultinomialNB(),
        ComplementNB()
        ,GradientBoostingClassifier(n_estimators=200, random_state=0)]
    names = ['Random forest:', 'Multinobial Bayes:', 'Complement Bayes:'
        ,'Gradient Boosting:']
    for i in range(len(models)):
        clf = models[i]
        score = cross_val_score(clf,X_train,y_train)
        y_pred = cross_val_predict(clf, X_train, y_train)
        conf_mat = confusion_matrix(y_train, y_pred)
        clas_rep = classification_report(y_train, y_pred)
        print("{}, \n {}, \n {}, \n {}".format(names[i], score.mean(), conf_mat, clas_rep))

def text_vectorization(training_file, target_file):
    '''
    csv, csv -> model
    Accepts two csvs both including sentence tolkens. One csv represents sentences
    from the text used for sentiment analysis training, the other represents
    the text whose analysis is desired. Returns a fitted tfidf vector space
    based on both texts.
    '''
    training_sentences = pd.read_csv(training_file)
    training_features = training_sentences['Text']
    target_sentences = pd.read_csv(target_file)
    target_features = target_sentences['Text']
    complete_features = pd.concat([training_features,target_features])
    vectorizer = TfidfVectorizer(max_features=2500, stop_words=stopwords.words('french'))
    vectorizer.fit(complete_features)

    return vectorizer

def sentiment_analysis_training(vectorizer, training_file):
    '''
    model, str -> model
    Accepts a pre-fit vectorizer based both on the target and training text,
    as well as a file name representing the file that will be used to train
    the sentiment analysis model. Fits the training text to the vector space,
    carries out a train/test split, and trains a sentiment classifier,
    previously selected by CV. Returns the classifier model.
    '''
    training_sentences = pd.read_csv(training_file)
    training_features = training_sentences['Text']
    training_sentence_labels = training_sentences['Label']

    features = vectorizer.transform(training_features).toarray()
    np.random.seed(17)

    ### Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, training_sentence_labels, test_size=0.3, stratify = training_sentence_labels, random_state=0)

    ### Use the model with the highest cross_val_score
    clf = RandomForestClassifier(n_estimators=200, random_state=0)

    # Fit and test the model
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))

    return clf
