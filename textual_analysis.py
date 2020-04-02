import nltk as nltk
import numpy as np
import pandas as pd
import itertools
import re
import networkx as nx
import matplotlib.pyplot as plt
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
        if ind == 'i':
            predictions.append(0)
        if ind == 'p':
            predictions.append(1)
        if ind == 'n':
            predictions.append(-1)
    return predictions

def network_visualizer(edge_matrix, character_list, edge_sentiment, node_sentiment):
    edge_list = []
    for i in range(len(character_list)):
        for j in range(len(character_list)):
            if i < j:
                if edge_matrix[i,j] == True:
                    edge_list.append((characters[i],characters[j]))
    # print(edge_list)
    ### Characterize sentiment as positive vs negative
    edge_sentiment = [1 if ind >= 0.5 else 0 for ind in edge_sentiment]
    node_sentiment = [1 if ind >= 0.5 else 0 for ind in node_sentiment]
    ### Create RGB tuple vectors for edges and nodes
    edge_colours = []
    for i in edge_sentiment:
        edge_colours.append((1-i,0,1))
    node_colours = []
    for i in node_sentiment:
        node_colours.append((1-i,0,1))
    ### Plot network with appropriate colours for edges and nodes
    G = nx.Graph()
    G.add_nodes_from(characters)
    G.add_edges_from(edge_list)
    nx.draw(G, with_labels = True, edge_color = edge_colours, node_color = node_colours)
    # plt.savefig('./data/character_network.png')

### Main module
if __name__ == "__main__":

    ### Load text file, remove licence, and do text_preprocessing
    processed = text_preprocessing(file_processor('./data/general.txt','''A ma petite-fille'''))

    ### Load pretrained vectorizer and classifier
    vectorizer = load('./data/vectorizer.joblib')
    clf = load('./data/sentiment_analysis_model.joblib')

    ### Tokenize into words to find cooccurrences
    tokens = nltk.word_tokenize(processed)
    text = nltk.Text(tokens)

    ### The names of characters
    names = ['général','dabrovine','papofski','dérigny','natasha',
        'jacques', 'paul', 'romane', 'pajarski','jackson']
    # names = load('./data/character_names.joblib')

    characters = ['général','dabrovine','papofski','dérigny','natasha',
        'jacques', 'paul', 'pajarski']

    ### Generate list of locations for each character
    locations = general_character_handling(text, names)

    ### How close in the text should character tokens have to be to be counted?
    threshold = 20

    ### Count cooccurrences and determine which of these are important links
    a,b = list_comparer(locations, threshold)

    ### Define links to be where cooccurrences are greater than average cooccurrences
    edges = np.greater(a,np.mean(a))


    ### How positive or negative on average are mentions of the characters?
    char_scores = []
    for i in range(len(locations)):
        passages = centered_passages(locations[i], threshold)
        scores = passage_sentiment(passages)
        ### Remove scores of 0 (when characterizing good vs bad neutral
        ### scores only dilute the signal)
        scores = [x for x in scores if x != 0]
        # scores = list(filter((0).__ne__, scores))
        char_scores.append(np.mean(scores))

    ### Sentiment scores for co-occurrence passages
    df = pd.read_csv('./data/relationship_passages.csv')
    df['Sentiment'] = passage_sentiment(df['Passage'])
    df.to_csv('./data/Sentiment_scores.csv')
    vals = []
    for i in range(64):
        df_sub = df[df['Code']==i]
        x = df_sub['Sentiment']
        y = [i for i in x if i != 0]
        if len(y) > 0:
            vals.append(np.mean(y))
        else:
            vals.append('No positive or negative scores')
    valid_edges = np.concatenate(edges).tolist()
    valid_edge_scores = list(itertools.compress(vals, valid_edges))

    ### Mean sentiment score for each edge in the graph, in same order
    ### as edges extracted from the table of edges (i.e. order in network_visualizer)
    mean_passage_scores = [ind for ind in valid_edge_scores if isinstance(ind,float)]

    print(mean_passage_scores)

    ## Visualizing network
    network_visualizer(edges, characters, mean_passage_scores, char_scores)
