import nltk as nltk
import numpy as np
import itertools
import csv
import re
import logging
from character_identification import execute_char_id
from sentiment_analysis_training_set_generation import file_processor, text_preprocessing

logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')

def passage_aggregator(text, locations, indices_of_cooccurrences):
    '''
    [[(ind,ind)]] -> [[passages]]
    Accepts nltk text, a list of locations and a a nested list of cooccurrences
    where the first list refers to individual characters (focal), and the next
    level refers to all the other characters (secondary).
    Finally, nested in these secondary lists are tuples
    that give textual indices for beginning and end points of the passage
    bracketed by the names of the focal and secondary characters.

    The function returns a list of lists, where each inner list is of the
    text passages bracketed by the focal and secondary character's names,
    or indicates this list already exists in another cell, or that the characters
    are identical.'''

    passages = []
    for i in range(len(locations)):
        for j in range(len(locations)):
            sentences = []
            if i < j:
                for k in range(len(indices_of_cooccurrences[i][j])):
                    start = min(indices_of_cooccurrences[i][j][k])
                    end = max(indices_of_cooccurrences[i][j][k])
                    sentence = ' '.join(text[start:end+1])
                    sentences.append(sentence)
                passages.append(sentences)
            elif i > j:
                passages.append(['Covered by another cell'])
            else:
                passages.append(['Identical character'])
    return passages

def passage_deaggregator(passage_list, title):
    '''[[passages]] -> csv
    Accepts a list of lists, where each nested list includes all the passages
    bracketed by two character names (or an indication that such a list doesn't
    make sense or has already been processed) and returns a csv where the
    first column includes each separate passage in a separate row, and the second
    column includes the key indicating what relationship is referred to.'''

    with open('./data/{}_relationship_passages.csv'.format(title), mode='w') as sentences:
        sentence_writer = csv.writer(sentences, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(passage_list)):
            passages = passage_list[i]
            sentence_writer.writerow(['Passage','Code'])
            for j in passages:
                sentence_writer.writerow([j,i])

def general_character_handling(text, names):
    ### A list of characters whose relationships should be analyzed
    ### and determine where in the text they occur
    locations = []
    for i in names:
        locations.append([j for j, item in enumerate(text) if item == i])
    logging.debug('Length of locations = {}'.format(len(locations)))
    return locations

def list_comparer(lists, threshold):
    '''[[int]] -> arr, [(tuple)]
    Accepts a list of lists of integers and identifies how many elements exist
    within all pairs of lists that are within a threshold distance.
    The number of elements within this threshold is returned in the form of a
    numpy array. Also returns the indices of the boundaries of these co-occurences.
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

def co_occurrence_data_generation(text, locations, cooccurrence_boundaries, title):
    ### Takes indices of boundaries, and returns passages as a list per relationship
    list_of_passages_per_relationship = passage_aggregator(text, locations, cooccurrence_boundaries)
    ### Turns list of passages per relationship into a csv for editing
    passage_deaggregator(list_of_passages_per_relationship, title)


### Main module
if __name__ == "__main__":

    ### Load text file, remove licence, and do text_preprocessing
    processed = text_preprocessing(file_processor('./data/general.txt','''A ma petite-fille''','''End of Project Gutenberg's'''))

    ### Tokenize into words to find cooccurrences
    tokens = nltk.word_tokenize(processed)
    text = nltk.Text(tokens)

    ### The names of characters
    names = execute_char_id('./data/general.txt','''A ma petite-fille''','''End of Project Gutenberg's''')
    names = [i.lower() for i in names]

    ### Generate list of locations for each character
    locations = general_character_handling(text, names)

    ### How close in the text should character tokens have to be to be counted?
    threshold = 20

    ### Number of co-occurrences (a), and boundaries of these text passages (b)
    a,indices_of_cooccurrences = list_comparer(locations, threshold)
    ### Takes indices of boundaries, and returns passages as a list per relationship
    c = passage_aggregator(text, locations, indices_of_cooccurrences)
    ### Turns list of passages per relationship into a csv for editing
    passage_deaggregator(c)
