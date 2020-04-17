# Who's that again?

## Introduction

A frustration for many readers is the difficulty of remembering who all the
characters of a book are and how they relate to one another. While some books
offer family trees or lists of characters, many books don't, even when the
dramatis personae is extensive. The goal of this project is to automate the
generation of a list of characters, the determination of whether each of these
characters is a hero or a villain, the identification of which characters have
close relationships, and the discernment of the nature of their relationships
when the user supplies a text.

## Pipeline

The project pipeline can broadly be described as follows:

1. Generate a list of characters from the source text that will serve as nodes
in a character graph.
2. Create a vector space encompassing both the text of interest and the text being used to train a sentiment analysis model (currently *L'auberge de l'ange gardien*).
3. Train a sentiment analysis model to identify whether a character or a relationship is positive or negative.
4. Identify the edges linking nodes together. These edges should represent significant
relationships between characters. Currently this involves determining which characters co-occur frequently in the text.
5. Classify characters and relationships using the sentiment analysis model.
6. Visualize the network in the form of a graph.

## Development

The project was initially developed to explore the comtesse de Ségur's novel
for children *Le Général Dourakine*. Given that machine learning models perform
best when training data and test data are similar, the sentiment anlaysis training
was done using another of the comtesse de Ségur's novels, *L'auberge de l'ange gardien*.

Ultimately the goal is to have a pipeline generally applicable to French
literature. For now this primarily involves investigating model performance
on a broad range of classic French novels, and could eventually entail
a significant reconsideration of the approach to sentiment analysis or
edge identification. 

At this point all steps in the pipeline are functional, though iterative improvements
are planned. You can see some sample networks built by the model below (positive characters and interactions in blue, neutral in purple, and negative in pink):

!["Visualization of network for Le Général Dourakine"](https://github.com/mpedruski/character_net/blob/master/results/general_character_network.svg.png "Network in progress")

## Files

Files used in this project include:

* character_identification.py: This file includes functions for a rules-based identification of characters found in the text. Essentially these functions look for capitalized words that don't occur at the beginning of a sentence as well as names that would not be capitalized, but come before colons (often a marker of a speaker in the target text). They discard any words that don't occur more than some threshold and remove words using NLTKs French stopwords and two custom stopword lists (one of common words that might have meaning in other contexts and as such are not in NLTKs stopwords, and one of geographical entities.

* character_identification_spacy.py: This file provides an out-of-the-box spaCy implementation of character identificaiton. While it does provide some information absent in the rules based approach, the value of this extra information is fairly limited and the data are quite messy, suggesting a custom-trained NER model might be necessary before spaCy is able to supplant the rules-based approach.

* co_occurrence_data_generation.py: This file includes functions that identify where in the text characters occur, where co-occurrences happen, and what text is found in co-occurrences.

* sentiment_analysis_training_set_generation.py: This file includes functions for text pre-processing and generating sentence token csv's of input texts

* sentiment_analysis_training.py: This file includes functions for the creation of a textual vector space (tf-idf) using both a training text and a target text, CV of a sentiment analysis model and the training of a final sentiment analysis model.

* textual_analysis.py: This file includes functions that create an nltk text from the raw text, identify text around a character occurrence, classify the sentiment of these passages as well as the passages involving co-occurrences, and visualize the network graph (positive characters and relationships in blue, neutral characters and relationships in purple, negative ones in pink).
