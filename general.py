import nltk as nltk
import numpy as np
import itertools

def list_comparer(lists, threshold):
    '''[[int]] -> arr
    Accepts a list of lists of integers and identifies how many elements exist
    within all pairs of lists that are within a threshold distance.
    The number of elements within this threshold is returned in the form of a
    numpy array.
    '''
    comparisons = np.zeros((len(lists),len(lists)))
    for i in range(len(lists)):
        list_a = lists[i]
        for j in range(len(lists)):
            list_b = lists[j]
            if i!=j:
                # print("i: {}, j:{}, val:{}".format(i,j,sum([a == b for (a, b) in itertools.product(list_a, list_b)])))
                comparisons[i,j] = (sum([abs(a - b) < threshold for (a, b) in itertools.product(list_a, list_b)]))
    return comparisons

### Load file and pre-process text
fh = open('./data/general.txt','r')
raw = fh.read().lower()
tokens = nltk.word_tokenize(raw)
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

### Count cooccurrences and then determine which of these are important links
a = list_comparer(locations, threshold)
print('Matrix of cooccurrences:\n')
print(a)

### Define links to be where cooccurrences are greater than average cooccurrences
print('Connection matrix defined by mean cooccurrences:\n')
print(np.greater(a,np.mean(a)))
