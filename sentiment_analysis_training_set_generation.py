import nltk as nltk
import csv
import re

def text_preprocessing(text):
    '''
    str -> str
    Accepts a raw text string (after the licence has been removed)
    and does some basic preprocessing to remove special characters, singletons
    multiple whitespaces, leading b's, converting to lower.'''
    processed = re.sub(r'[«»\,\_\;\':-]', ' ', text)
    processed = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed)
    processed = re.sub(r'\^[a-zA-Z]\s+', ' ', processed)
    processed = re.sub(r'\s+', ' ', processed, flags=re.I)
    processed = re.sub(r'^b\s+', '', processed)
    processed = processed.lower()
    return processed

def file_processor(file_name, initializer_string, terminator_string):
    '''
    str, str, str -> str
    Accepts a valid file path and strings of text that indicate the start and
    end of the document. It removes the licence (external to the document) and
    returns the raw text of the document'''
    fh = open(file_name,'r')
    raw = fh.read()
    fh.close()
    end_of_preample = raw.index(initializer_string)
    end_of_book = raw.index(terminator_string)
    licence_removed = raw[end_of_preample:end_of_book]
    return licence_removed

def generate_sentence_csv(text, file_name):
    '''
    [str], str -> csv
    Accepts a text, and converts it into sentence tokens which are written
    to a csv with each token on a separate line using the supplied file_name'''
    tokens = nltk.sent_tokenize(text)
    with open(file_name, mode='w') as sentences:
        sentence_writer = csv.writer(sentences, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        sentence_writer.writerow(['Text'])
        for i in tokens:
            sentence_writer.writerow([i])

def target_text_tokenizer(file, text_start, text_end):
    '''
    str, str, str
    Accepts a file path, and strings indicating the start and end of a document.
    Calls functions that remove text before/after the start/end, preprocess
    the text, tolkenize the text and then generates a csv of the tolkens.
    '''
    title = re.search(r'/data/(.*)\.txt', file).group(1)
    ### Load target file, remove licence and preprocess
    target_file = text_preprocessing(file_processor(file, text_start, text_end))
    generate_sentence_csv(target_file,'./data/{}_raw_sentences.csv'.format(title))

### Main module
if __name__ == "__main__":

    ### Load training file, remove licence, and preprocess
    training_file = text_preprocessing(file_processor('./data/auberge.txt','''A mes petits-fils, LOUIS ET GASTON DE MALARET''','''End of Project Gutenberg's'''))

    ### Generate output csvs from text for training and target text
    generate_sentence_csv(training_file,'./data/auberge_raw_sentences.csv')
