import re
import numpy as np
from character_identification import execute_char_id
from textual_analysis import nltk_text_creation, co_occurrence_passage_sentiment, character_sentiment_scores, network_visualizer
from sentiment_analysis_training_set_generation import target_text_tokenizer
from sentiment_analysis_training import text_vectorization, sentiment_analysis_training, cross_validation_suite
from co_occurrence_data_generation import co_occurrence_data_generation, list_comparer, general_character_handling

### Main module
if __name__ == "__main__":

    ### Basic parameters of the files to be used, the title of the target file, and threshold for co-occurrences
    raw_file, start, end = './data/general.txt', 'A ma petite-fille','''End of Project Gutenberg's'''
    processed_file = './data/general_raw_sentences.csv'
    sentiment_training_file = './data/auberge_coded_sentences_short.csv'
    title = re.search(r'/data/(.*)\.txt', raw_file).group(1)
    threshold = 20

    ### Generate list of characters and preprocess for later use
    list_of_characters = execute_char_id(raw_file, start, end)
    list_of_characters = [i.lower() for i in list_of_characters]

    ### Generate csv of words needed to run vectorizer
    target_text_tokenizer(raw_file, start, end)
    ### Make vector space that includes target text and training text
    vectorizer = text_vectorization(sentiment_training_file, processed_file)
    ### Cross validation of sentiment analysis classifier
    # cross_validation_suite(vectorizer, sentiment_training_file)
    ### Train sentiment-anlaysis model based only on training text
    clf = sentiment_analysis_training(vectorizer, sentiment_training_file)
    ### Preprocessing of raw text and conversion to NLTK text class
    text = nltk_text_creation(raw_file, start, end)
    ### Generate list of locations for each character
    locations = general_character_handling(text, list_of_characters)
    ### Count cooccurrences based on threshold and determine which of these are important links
    ### also identify indices of tokens that bracket cooccurrences
    number_of_cooccurrences, cooccurrence_boundaries = list_comparer(locations, threshold)
    ### Define edges to be where cooccurrences are greater than average cooccurrences
    edges = np.greater(number_of_cooccurrences,np.mean(number_of_cooccurrences))
    ### Generate csv of relationship passages for sentiment_analysis
    co_occurrence_data_generation(text, locations, cooccurrence_boundaries, title)
    ### Sentiment scores of edges that have a numeric score
    valid_edge_scores = co_occurrence_passage_sentiment(vectorizer, clf, title, list_of_characters, edges)
    ### Sentiment scores of characters (nodes)
    char_scores = character_sentiment_scores(locations, threshold, text, vectorizer, clf)
    ### Mean sentiment score for each edge in the graph, in same order
    ### as edges extracted from the table of edges (i.e. order in network_visualizer)
    mean_passage_scores = [ind for ind in valid_edge_scores if isinstance(ind,float)]
    passage_scores_regularized = [(i+1)/2 for i in mean_passage_scores]
    ### Visualizing network
    network_visualizer(edges, list_of_characters, passage_scores_regularized, char_scores, title)
