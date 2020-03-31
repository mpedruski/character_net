import nltk as nltk
import numpy as np
import re
from sentiment_analysis_training_set_generation import file_processor, text_preprocessing

### Main module
if __name__ == "__main__":

    ### Load text file, remove licence, and do text_preprocessing
    processed = file_processor('./data/general.txt','''A ma petite-fille''')

    ### Tokenize into words to find cooccurrences
    tokens = nltk.word_tokenize(processed)
    text = nltk.Text(tokens)

    ### Collocations approach does identify most characters, but not in a way
    ### that's useful for an automatic list
    print(text.collocations())
    ### Notably there are lots of names for the General I'm missing now
    ### (oncle, grand père) and Dérigny is omitted altogether because he's
    ### typically invoked with a mononym
