# Who's that again?

## Introduction

A frustration for many readers is the difficulty of remembering who all the
characters of a book are and how they relate to one another. While some books
offer family trees or lists of characters, many books don't, even when the
dramatis personae is extensive. The goal of this project is to automate the
generation of a list of characters, whether each of these characters is a
hero or a villain, establish which characters have close
relationships, and determine the nature of their relationships when the user
supplies a text.

## Pipeline

The project pipeline is envisaged as follows:

1. Generate a list of characters from the source text that will serve as nodes
in a character graph.
2. Identify the edges linking nodes together. These edges should represent significant
relationships between characters.
3. Identify whether each character is a positive one or a negative one (i.e. hero or villain).
4. Characterize the nature of the edges linking nodes, that is, is the relationship represented by the node positive or negative?

## Current status

Currently the project is being developed for la comtesse de Ségur's novel
for children *Le Général Dourakine*. The sentiment anlaysis training is being done using
her novel *L'auberge de l'ange gardien*. Ultimately the goal is to be able to generalize
it to any text.

At this point all steps in the pipeline are functional, though iterative improvements
are planned. You can see a sample network built by the model below:

!["Visualization of network in progress"](https://github.com/mpedruski/character_net/blob/master/results/character_network.svg.png "Network in progress")

## Files

Files used in this project include:

* sentiment_analysis_training_set_generation.py: This file includes functionality for text pre-processing and generating sentence-by-sentence csv's of input texts

* sentiment_analysis_trainin.py: This file builds a textual vector space (tf-idf) using both a training text and a target text. It additionally trains a sentiment analysis model using this vector space and the training text.

* textual_analysis.py: This file analyzes the target text given a list of characters and a pre-trained sentiment analysis model to return which of the characters are strongly linked, the nature each of those characters, and the nature of the strong relationships between the characters. Currently provides output in the form of a graph showing characters and relationships (positive characters and relationships in blue, neutral characters and relationships in purple, negative ones in pink).

* co_occurrence_data_generation.py: This file accepts the target text, produces a list of indices forming the boundaries of co-occurrences, and then writes a csv file that contains this bounded text in the first column, and the label indicating which relationship is referred to in the second column.

* character_identification.py: This file implements a rules-based identification of characters found in the text. Essentially it looks for capitalized words that don't occur at the beginning of a sentence as well as names that would not be capitalized, but come before colons (often a marker of a speaker in the target text). It discards any words that don't occur more than some threshold. It then removes words using two custom stopword lists (one of common words that might have meaning in other contexts and as such are not in NLTKs stopwords, and one of geographical entities).

* character_identification_spacy.py: This file provides an out-of-the-box spaCy implementation of character identificaiton. While it does provide some information absent in the rules based approach, the value of this extra information is fairly limited and the data are quite messy, suggesting a custom-trained NER model might be necessary before spaCy is able to supplant the rules-based approach.
