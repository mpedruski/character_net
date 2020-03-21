import nltk as nltk
import numpy as np
import itertools
import re

def list_comparer(lists, threshold):
    '''[[int]] -> arr
    Accepts a list of lists of integers and identifies how many elements exist
    within all pairs of lists that are within a threshold distance.
    The number of elements within this threshold is returned in the form of a
    numpy array.
    '''
    comparisons = np.zeros((len(lists),len(lists)))
    indices = []
    for i in range(len(lists)):
        list_a = lists[i]
        char_indices = []
        for j in range(len(lists)):
            list_b = lists[j]
            if i!=j:
                comparisons[i,j] = (sum([abs(a - b) < threshold for (a, b) in itertools.product(list_a, list_b)]))
                char_indices.append([(a,b) for (a, b) in itertools.product(list_a, list_b) if abs(a-b)< threshold])
        indices.append(char_indices)
    return comparisons, indices

### Load file and remove licence
fh = open('./data/general.txt','r')
raw = fh.read()
end_of_preample = raw.index('''A ma petite-fille''')
end_of_book = raw.index('''End of Project Gutenberg's''')
licence_removed = raw[end_of_preample:end_of_book]

### Clean raw text
processed = re.sub(r'[«»\,\_\;\':-]', ' ', licence_removed)
processed = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed)
processed = re.sub(r'\^[a-zA-Z]\s+', ' ', processed)
processed = re.sub(r'\s+', ' ', processed, flags=re.I)
processed = re.sub(r'^b\s+', '', processed)
processed = processed.lower()

# print(processed[0:1000])

### Tokenize into words to find cooccurrences
tokens = nltk.word_tokenize(processed)
text = nltk.Text(tokens)

### Include a list of characters whose relationships should be analyzed
### and determine where in the text they occur
names = ['général','dabrovine','papofski','dérigny','natasha',
    'jacques', 'paul', 'romane', 'pajarski','jackson']
locations = []
for i in names:
    locations.append([j for j, item in enumerate(text) if item == i])

### Combine jackson, pajarski, and romane as synonyms
jackson = locations[-3]+locations[-2]+locations[-1]
locations = locations[:-3]
locations.append(jackson)

### How close in the text should character tokens have to be to be counted?
threshold = 20

### Count cooccurrences and determine which of these are important links
a,b = list_comparer(locations, threshold)
print('Matrix of cooccurrences:\n')
print(a)

### Define links to be where cooccurrences are greater than average cooccurrences
print('Connection matrix defined by mean cooccurrences:\n')
print(np.greater(a,np.mean(a)))

### Quick sample of text involved in cooccurrences?
# for i in range(15):
#     print(b[0][0][i])
#     print(text[b[0][0][i][0]:b[0][0][i][1]+1])
