# Who's that again?

## Introduction

A frustration for many readers is the difficulty of remembering who all the
characters of a book are and how they relate to one another. While some books
offer family trees or lists of characters, many books don't, even when the
dramatis personae is extensive. The goal of this project is to automate the
generation of a list of characters, whether each of these characters is a 
hero or an anti-hero, establish which characters have close
relationships, and determine the nature of their relationships when the user
supplies a text.

## Pipeline

The project pipeline is envisaged as follows:

1. Generate a list of characters from the source text that will serve as nodes
in a character graph.
2. Identify the edges linking nodes together. These edges should represent significant
relationships between characters.
3. Identify whether each character is a positive one or a negative one (i.e. hero or antihero).
4. Characterize the nature of the edges linking nodes. Currently this is expected
to take the form of a simple characterization of relationships as positive or negative.

## Current status

Currently the project is being developed for la comtesse de Ségur's novel
for children *Le Général Dourakine*. The sentiment anlaysis training is being done using
her novel *L'auberge de l'ange gardien*. Ultimately the goal is to be able to generalize
it to any text.

Currently the the identification of important relationships (step 2) and identification of 
character positiveity (step 3) are functional. This file will be updated as progress is made!

## Files 

Files used in this project include:

* sentiment_analysis_training_set_generation.py: This file includes functionality for text pre-processing and generating sentence-by-sentence csv's of input texts

* sentiment_analysis_trainin.py: This file builds a textual vector space (tf-idf) using both a training text and a target text. It additionally trains a sentiment analysis model using this vector space and the training text.

* textual_analysis.py: This file analyzes the target text given a list of characters to return which of the characters are strongly linked, and the positivity of each of those characters.