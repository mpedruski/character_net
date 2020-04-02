import nltk as nltk
import numpy as np
import itertools
import csv
import re

def text_preprocessing(text):
    '''Accepts a raw text string (after the licence has been removed)
    and does some basic preprocessing to remove special characters, singletons
    multiple whitespaces, leading b's, converting to lower.'''
    processed = re.sub(r'[«»\,\_\;\':-]', ' ', text)
    processed = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed)
    processed = re.sub(r'\^[a-zA-Z]\s+', ' ', processed)
    processed = re.sub(r'\s+', ' ', processed, flags=re.I)
    processed = re.sub(r'^b\s+', '', processed)
    processed = processed.lower()
    return processed

def file_processor(file_name, initializer_string):
    '''Accepts a valid file name and a string of text that indicates the
    end of the Gutenberg licence. It removes the licence and returns
    the raw text of the document'''
    fh = open(file_name,'r')
    raw = fh.read()
    fh.close()
    end_of_preample = raw.index(initializer_string)
    end_of_book = raw.index('''End of Project Gutenberg's''')
    licence_removed = raw[end_of_preample:end_of_book]
    return licence_removed

def generate_sentence_csv(text, file_name):
    '''Accepts a processed list of tokens and writes them to a csv with
    each token on a separate line using the supplied file_name'''
    tokens = nltk.sent_tokenize(text)
    with open(file_name, mode='w') as sentences:
        sentence_writer = csv.writer(sentences, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in tokens:
            sentence_writer.writerow([i])

### Main module
if __name__ == "__main__":

    ### Load training file, remove licence, and preprocess
    training_file = text_preprocessing(file_processor('./data/auberge.txt','''A mes petits-fils, LOUIS ET GASTON DE MALARET'''))
    target_file = text_preprocessing(file_processor('./data/general.txt','''A ma petite-fille'''))

    ### Generate output csvs from text for training and target text
    generate_sentence_csv(training_file,'./data/auberge_raw_sentences.csv')
    generate_sentence_csv(target_file,'./data/general_raw_sentences.csv')
