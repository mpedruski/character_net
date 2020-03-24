import nltk as nltk
import numpy as np
import itertools
import re
import csv
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from sentiment_analysis_training_set_generation import file_processor, text_preprocessing

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

def list_comparer(lists, threshold):
    '''[[int]] -> arr
    Accepts a list of lists of integers and identifies how many elements exist
    within all pairs of lists that are within a threshold distance.
    The number of elements within this threshold is returned in the form of a
    numpy array.
    '''
    comparisons = np.zeros((len(lists),len(lists)))
    indices = []
    for i in range(len(lists)):
        list_a = lists[i]
        char_indices = []
        for j in range(len(lists)):
            list_b = lists[j]
            if i!=j:
                comparisons[i,j] = (sum([abs(a - b) < threshold for (a, b) in itertools.product(list_a, list_b)]))
                char_indices.append([(a,b) for (a, b) in itertools.product(list_a, list_b) if abs(a-b)< threshold])
            else:
                char_indices.append([])
        indices.append(char_indices)
    return comparisons, indices

def centered_passages(indices, threshold):
    '''Accepts a list of indices where a character is mentioned and returns
    the text surrounding this mention to a distance of threshold//2'''
    bounds = threshold//2
    passages = [' '.join(text[i-bounds:i+bounds]) for i in indices]
    return passages

def passage_sentiment(passage):
    '''Accepts a list of text passages, converts them to a numpy array,
    fits them to a pre-set vector space, and then makes sentiment predictions
    for each passage based on a pre-trained model'''
    arr = np.asarray(passage)
    features = vectorizer.transform(arr).toarray()
    coded_predictions = clf.predict(features)
    predictions = []
    for ind in coded_predictions:
        # if ind == 'i':
        #     predictions.append(0)
        if ind == 'p':
            predictions.append(1)
        if ind == 'n':
            predictions.append(-1)
    return predictions

### Load text file, remove licence, and do text_preprocessing
processed = file_processor('./data/general.txt','''A ma petite-fille''')

### Load pretrained vectorizer and classifier
vectorizer = load('./data/vectorizer.joblib')
clf = load('./data/sentiment_analysis_model.joblib')

### Tokenize into words to find cooccurrences
tokens = nltk.word_tokenize(processed)
text = nltk.Text(tokens)

### A list of characters whose relationships should be analyzed
### and determine where in the text they occur
names = ['général','dabrovine','papofski','dérigny','natasha',
    'jacques', 'paul', 'romane', 'pajarski','jackson']
locations = []
for i in names:
    locations.append([j for j, item in enumerate(text) if item == i])

### Combine jackson, pajarski, and romane as synonyms
jackson = locations[-3]+locations[-2]+locations[-1]
locations = locations[:-3]
locations.append(jackson)
logging.debug('Length of locations = {}'.format(len(locations)))

### How close in the text should character tokens have to be to be counted?
threshold = 20

### Count cooccurrences and determine which of these are important links
a,b = list_comparer(locations, threshold)
print('Matrix of cooccurrences:\n')
print(a)
logging.debug('Dimensions of cooccurrence array: {}'.format(a.shape))
logging.debug('Length of passages list: {}'.format(len(b)))

### Define links to be where cooccurrences are greater than average cooccurrences
print('Connection matrix defined by mean cooccurrences:\n')
print(np.greater(a,np.mean(a)))

### How positive or negative on average are mentions of the characters?
char_scores = []
for i in range(len(locations)):
    passages = centered_passages(locations[i], threshold)
    scores = passage_sentiment(passages)
    char_scores.append(np.mean(scores))

print(char_scores)
