import nltk as nltk
import numpy as np
import itertools
import csv
import logging

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
    '''[[passages]], str -> csv
    Accepts a list of lists, where each nested list includes all the passages
    bracketed by two character names (or an indication that such a list doesn't
    make sense or has already been processed) as well as the work's title.
    Returns a csv where the first column includes each separate passage in a
    separate row, and the second column includes the key indicating which
    relationship is referred to.'''

    with open('./data/{}_relationship_passages.csv'.format(title), mode='w') as sentences:
        sentence_writer = csv.writer(sentences, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        sentence_writer.writerow(['Passage','Code'])
        for i in range(len(passage_list)):
            passages = passage_list[i]
            for j in passages:
                sentence_writer.writerow([j,i])

def general_character_handling(text, names):
    '''
    text, [str] -> [int]
    Accepts an nltk text as well as a list of names. Returns a list of indices
    for each name, indicating where in the text that name occurs.
    '''
    locations = []
    for i in names:
        locations.append([j for j, item in enumerate(text) if item == i])
    logging.debug('Length of locations = {}'.format(len(locations)))
    return locations

def list_comparer(lists, threshold):
    '''[[int]], int -> arr, [(tuple)]
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
    '''
    text, [int], [(tuple)], str -> csv
    Accepts an nltk text, a nested list of indices indicating each occurrence
    of a each, character in the text, a list of tuples indicating indices for
    the boundaries of passages considered to be co-occurrences, and the title of
    the text. Calls a function which returns a list of text passages for each
    relationship, as well as a function that adds each passage to a file
    along with a label indicating which relationship corresponds to the passage.
    '''
    ### Takes indices of boundaries, and returns passages as a list per relationship
    list_of_passages_per_relationship = passage_aggregator(text, locations, cooccurrence_boundaries)
    ### Turns list of passages per relationship into a csv for editing
    passage_deaggregator(list_of_passages_per_relationship, title)
