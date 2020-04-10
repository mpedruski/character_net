import spacy
import nltk
import re
from sentiment_analysis_training_set_generation import file_processor
from joblib import load

### Main module
if __name__ == "__main__":

    ### Load a pre-trained model
    nlp = spacy.load("fr_core_news_sm")

    ### Load text file and remove licence
    raw = file_processor('./data/general.txt','''A ma petite-fille''','''End of Project Gutenberg's''')

    ### Do the processing
    doc = nlp(raw)

    results = []
    ### Print the results
    for ent in doc.ents:
        if ent.label_ == 'PER':
            results.append(ent.text)

    custom_stopwords = ['oui', 'non', 'merci', 'quand', 'où', 'comme', 'oh',
        'très', 'comment', 'pardon', 'va', 'vont', 'allez', 'allons', 'vas',
        'vais', 'alors', 'bien', 'tout', 'oh', 'ah', 'bon', 'dieu', 'pardon',
        'après', 'pendant','mme','madame']
    stopwords = nltk.corpus.stopwords.words('french') + custom_stopwords
    stopwords = [i.title() for i in stopwords]
    results = [i for i in results if i not in stopwords]

    ### Uniformizing honorifics and newline characters
    results = [re.sub(r'Madame',r'Mme',i) for i in results]
    results = [re.sub(r'(\w+)\n$',r'\1',i) for i in results]
    results = [re.sub(r'(\w+)\n(\w+)$',r'\1 \2',i) for i in results]

    ### Comparing to rules based approach
    spacy_names = sorted(set(results))
    rules_based_names = sorted(load('./data/character_names.joblib'))
    print("spaCy names: {}".format(spacy_names))
    print(len(spacy_names))
    print("Rules based names: {}".format(rules_based_names))
    print(len(rules_based_names))
    unique_spacy_names = {i for i in spacy_names if i not in rules_based_names}
    print("Unique spaCy names: {}".format(sorted(unique_spacy_names)))

    ### Conclusion: Out of the box spaCy approach not worth it. It returns
    ### a number of names not found by rules based approach, but no major
    ### characters, and it returns a significant number of not-characters
    ### (including lots of imperatives).
