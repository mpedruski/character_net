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
from joblib import dump
from collections import Counter

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def cross_validation_suite(X_train, y_train):
    models = [RandomForestClassifier(n_estimators=200, random_state=0),
        MultinomialNB(),
        ComplementNB()]
        # ,GradientBoostingClassifier(n_estimators=200, random_state=0)]
    names = ['Random forest:', 'Multinobial Bayes:', 'Complement Bayes:']
        # ,'Gradient Boosting:']
    for i in range(len(models)):
        clf = models[i]
        score = cross_val_score(clf,X_train,y_train)
        y_pred = cross_val_predict(clf, X_train, y_train)
        conf_mat = confusion_matrix(y_train, y_pred)
        clas_rep = classification_report(y_train, y_pred)
        print("{}, \n {}, \n {}, \n {}".format(names[i], score.mean(), conf_mat, clas_rep))

### Main module
if __name__ == "__main__":

    np.random.seed(17)

    ### Load labelled dataset (labelling is only processing done)
    sentences = pd.read_csv('./data/auberge_coded_sentences_short.csv')
    ### Load dataset that will be used for predictions
    target_text = pd.read_csv('./data/general_raw_sentences.csv')

    ### Define features and data
    features = sentences['Text']
    labels = sentences['Label']

    ### Cobine training text and target text for purposes of creating vectorizer
    target_features = target_text['Text']
    complete_features = pd.concat([features,target_features])

    ### Vectorize text
    # vectorizer = CountVectorizer(max_features=2500, stop_words=stopwords.words('french'))
    # vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
    vectorizer = TfidfVectorizer(max_features=2500, stop_words=stopwords.words('french'))
    vectorizer.fit(complete_features)
    features = vectorizer.transform(features).toarray()

    ### Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify = labels, random_state=0)

    ### Balance the classes in the training data
    rus = RandomUnderSampler(sampling_strategy = 'not minority',random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    logging.debug('Resampled dataset shape {}'.format(Counter(y_res)))
    ### For the moment balanced data yields markedly worse CV scores than
    ### complete but imbalanced data for all but Multinomial Bayes

    ### Cross validation
    cross_validation_suite(X_train, y_train)

    ### Use the model with the highest cross_val_score
    clf = RandomForestClassifier(n_estimators=200, random_state=0)

    # Test the model
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print(classification_report(y_test,predictions))
    print(accuracy_score(y_test, predictions))

    # dump(clf, './data/sentiment_analysis_model.joblib')
    # dump(vectorizer, './data/vectorizer.joblib')
