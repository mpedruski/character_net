import nltk as nltk
import numpy as np
import itertools
import csv
import re

### Load file and tolkenize into sentences
fh = open('./data/auberge.txt','r')
raw = fh.read()
fh.close()

end_of_preample = raw.index('''A mes petits-fils, LOUIS ET GASTON DE MALARET''')
end_of_book = raw.index('''End of Project Gutenberg's''')
licence_removed = raw[end_of_preample:end_of_book]
# print(licence_removed[0:100])


### Clean raw text

### Special characters to remove
processed = re.sub(r'[«»\,\_\;\':-]', ' ', licence_removed)
processed = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed)
processed = re.sub(r'\^[a-zA-Z]\s+', ' ', processed)
processed = re.sub(r'\s+', ' ', processed, flags=re.I)
processed = re.sub(r'^b\s+', '', processed)
processed = processed.lower()

print(processed[0:1000])

tokens = nltk.sent_tokenize(processed)

with open('./data/auberge_raw_sentences.csv', mode='w') as sentences:
    sentence_writer = csv.writer(sentences, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for i in tokens:
        sentence_writer.writerow([i])
