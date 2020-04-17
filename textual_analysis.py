import nltk as nltk
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import logging
from sklearn import preprocessing
from sentiment_analysis_training_set_generation import file_processor, text_preprocessing

logging.basicConfig(level=logging.CRITICAL,format='%(asctime)s - %(levelname)s - %(message)s')

def centered_passages(indices, threshold, text):
    '''Accepts a list of indices where a character is mentioned and returns
    the text surrounding this mention to a distance of threshold//2'''
    bounds = threshold//2
    passages = [' '.join(text[i-bounds:i+bounds]) for i in indices]
    return passages

def passage_sentiment(passage, vectorizer, clf):
    '''Accepts a list of text passages, converts them to a numpy array,
    fits them to a pre-set vector space (vct), and then makes sentiment predictions
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

def network_visualizer(edge_matrix, character_list, edge_sentiment, node_sentiment, title):
    edge_list = []
    for i in range(len(character_list)):
        for j in range(len(character_list)):
            if i < j:
                if edge_matrix[i,j] == True:
                    edge_list.append((character_list[i],character_list[j]))
    ### Create RGB tuple vectors for edges and nodes
    edge_colours = []
    for i in edge_sentiment:
        edge_colours.append((1-i,0,1))
    node_colours = []
    for i in node_sentiment:
        node_colours.append((1-i,0,1))
    ### Plot network with appropriate colours for edges and nodes
    G = nx.Graph()
    G.add_nodes_from(character_list)
    G.add_edges_from(edge_list)
    nx.draw(G, with_labels = True, edge_color = edge_colours, node_color = node_colours, pos = nx.kamada_kawai_layout(G))
    # plt.show()
    plt.savefig('./data/{}_character_network.svg'.format(title))

def nltk_text_creation(file, beginning_of_text, end_of_text):
    ### Load text file, remove licence, and do text_preprocessing
    processed = text_preprocessing(file_processor(file, beginning_of_text, end_of_text))

    ### Tokenize and create an nltk text
    tokens = nltk.word_tokenize(processed)
    text = nltk.Text(tokens)
    return text

def co_occurrence_passage_sentiment(vectorizer, clf, title, names, edges):
    ### Sentiment scores for co-occurrence passages
    df = pd.read_csv('./data/{}_relationship_passages.csv'.format(title))
    df['Sentiment'] = passage_sentiment(df['Passage'], vectorizer, clf)
    df.to_csv('./data/{}_Sentiment_scores.csv'.format(title))
    vals = []
    for i in range(len(names)*len(names)):
        df_sub = df[df['Code']==i]
        x = df_sub['Sentiment']
        y = [i for i in x if i != 0]
        if len(y) > 0:
            vals.append(np.mean(y))
        else:
            vals.append('No positive or negative scores')
    valid_edges = np.concatenate(edges).tolist()
    valid_edge_scores = list(itertools.compress(vals, valid_edges))
    return valid_edge_scores

def character_sentiment_scores(locations, threshold, text, vectorizer, clf):
    char_scores = []
    for i in range(len(locations)):
        passages = centered_passages(locations[i], threshold, text)
        scores = passage_sentiment(passages, vectorizer, clf)
        ### Remove scores of 0 (when characterizing good vs bad neutral
        ### scores only dilute the signal)
        scores = [x for x in scores if x != 0]
        if len(scores) > 0:
            char_scores.append(np.mean(scores))
        else:
            char_scores.append(0)
    char_scores_regularized = [(i+1)/2 for i in char_scores]
    return char_scores_regularized
