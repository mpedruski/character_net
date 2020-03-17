import nltk as nltk
import numpy as np
fh = open('./data/general.txt','r')
raw = fh.read().lower()
# raw = 'The brown cow jumped over the brown moon'
tokens = nltk.word_tokenize(raw)

### Stopwords on hand, but is their removal needed?
# final_stops = stopwords.words('french')

text = nltk.Text(tokens)

characters = ['général','dabrovine','papofski','jackson','dérigny','natasha']
locations = []

for i in characters:
    locations.append([j for j, item in enumerate(text) if item == i])
print('Length of character locations vector: {}'.format(len(locations)))

count = [0,0,0,0,0]
for i in range(1,len(characters)):
    print(i)
    character_0 = locations[0]
    character_1 = locations[i]
    for j in character_0:
        for k in character_1:
            if abs(j-k) < 20:
                count[i-1] += 1
print(count)
