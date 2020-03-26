import nltk as nltk
import numpy as np
import itertools
import re
# import csv
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
from sentiment_analysis_training_set_generation import file_processor, text_preprocessing
from co_occurrence_data_generation import list_comparer, general_character_handling

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(message)s')

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

### Main module
if __name__ == "__main__":
    
    ### Load text file, remove licence, and do text_preprocessing
    processed = file_processor('./data/general.txt','''A ma petite-fille''')

    ### Load pretrained vectorizer and classifier
    vectorizer = load('./data/vectorizer.joblib')
    clf = load('./data/sentiment_analysis_model.joblib')

    ### Tokenize into words to find cooccurrences
    tokens = nltk.word_tokenize(processed)
    text = nltk.Text(tokens)

    ### Generate list of locations for each character
    locations = general_character_handling(text)

    ### How close in the text should character tokens have to be to be counted?
    threshold = 20

    ### Count cooccurrences and determine which of these are important links
    a,b = list_comparer(locations, threshold)
    print('Matrix of cooccurrences:\n')
    print(a)

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
