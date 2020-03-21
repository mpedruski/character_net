import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

np.random.seed(17)

### Load labelled dataset
sentences = pd.read_csv('./data/auberge_coded_sentences_short.csv')

### Define features and data
features = sentences['Text']
labels = sentences['Label']

### Vectorize text
vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('french'))
features = vectorizer.fit_transform(features).toarray()

### Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

### Train the model
clf = RandomForestClassifier(n_estimators=200, random_state=0)
clf.fit(X_train, y_train)

### Test the model
predictions = clf.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test, predictions))

dump(clf, './data/sentiment_analysis_model.joblib')
