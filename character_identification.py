import nltk as nltk
import numpy as np
import re
from sentiment_analysis_training_set_generation import file_processor, text_preprocessing
from joblib import dump

def capitalized_not_sentence_beginner(text):
    ### Use Regex to find all words that begin with a capital letter but that
    ### don't begin a sentence. Note: still captures words at beginning of sentence
    ### that begin their paragraph.
    pattern = re.compile(r'[^.?!-:\'][^\'«[]([A-Z][a-zà-ÿ]+)')
    results = pattern.findall(text)

    ### Remove stopwords, which should be capitalized given we're only working
    ### with capitals
    custom_stopwords = ['oui', 'non', 'merci', 'quand', 'où', 'comme', 'oh',
        'très', 'comment', 'pardon', 'va', 'vont', 'allez', 'allons', 'vas',
        'vais', 'alors', 'bien', 'tout', 'oh', 'ah', 'bon', 'dieu', 'pardon',
        'après', 'pendant','mme','madame']
    stopwords = nltk.corpus.stopwords.words('french') + custom_stopwords
    stopwords = [i.title() for i in stopwords]
    results = [i for i in results if i not in stopwords]

    ### A list of geographic place names and adjectives to remove from results
    fh = open('./data/geographic_stop_words.txt','r')
    geographic_stop_words = fh.read()
    fh.close()
    results = [i for i in results if i not in geographic_stop_words.title()]

    ### Count occurrences of each key, with a goal of removing rare words
    count = {}
    for i in results:
        count.setdefault(i,0)
        count[i] = count[i]+1
    # print(count)

    results = {i for i in count.keys() if count[i] > 4}
    return results
    ### Results offers names of characters. Remaining issues include
    ### lack of mentions of the general, resolution of synonyms.

def uncapitalized_speakers(text):
    ### One of the challenges with capitalized names is that honorifics
    ### (e.g. docteur, sergent) and family names following honorifics
    ### (e.g. ispravnik) aren't capitalized.

    ### As the dialogue in the text is semi-structured
    ### we can use that to identify speakers whose names aren't capitalized:
    pattern = re.compile(r'\s([a-zà-ÿ]+): «')
    results = pattern.findall(raw)
    ### Count occurrences of each key, with a goal of removing rare words
    count = {}
    for i in results:
        count.setdefault(i,0)
        count[i] = count[i]+1
    results = {i for i in count.keys() if count[i] > 9}
    return results


### Main module
if __name__ == "__main__":

    ### Load text file, remove licence, and do text_preprocessing
    raw = file_processor('./data/general.txt','''A ma petite-fille''')
    processed = text_preprocessing(raw)
    ### Tokenize into words to find cooccurrences
    tokens = nltk.word_tokenize(processed)
    text = nltk.Text(tokens)

    ### Generate set of proper nouns expected to be people
    capitalized_names = capitalized_not_sentence_beginner(raw)
    uncapitalized_names = uncapitalized_speakers(raw)
    characters = capitalized_names|uncapitalized_names
    dump(characters,'./data/character_names.joblib')

    ### TO DO ###
    ### Resolve characters whose names are the same
