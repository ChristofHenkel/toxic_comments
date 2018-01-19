from utils import *
from preprocess_utils import detect


import pandas as pd
from collections import Counter
a = ['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
letter_counts = Counter(a)
df = pd.DataFrame.from_dict(letter_counts, orient='index')
df.plot(kind='bar')

def show_hist(meta_data):
    meta_data.hist()


meta_data = read_metadata(raw_data_dir,train_fn)

a = [sum(item['label']) for item in corpus]
len([item for item in a if item == 0])
86061
len([item for item in a if item == 1])
3833
len([item for item in a if item == 2])
2107
len([item for item in a if item == 3])
2523
len([item for item in a if item == 4])
1076
len([item for item in a if item == 5])
231

data = read_data(raw_data_dir,train_fn)

####
###  languages
k=0
for j, d in enumerate(data):
    if j % 100 == 0:
        print(j)
    text = d['text'].lower()
    lang = detect(text)

    if lang != 'en':
        k += 1
        print(text)



####
## distribution of length

## by character
lengths = [len(d['text']) for d in data]
d = {k:val for k, val in enumerate(lengths)}

df = pd.DataFrame.from_dict(d, orient='index')
df.hist(bins=20)

## by words

lengths = [len(d['text'].split(' ')) for d in data]
d = {k:val for k, val in enumerate(lengths)}

df = pd.DataFrame.from_dict(d, orient='index')
df.hist(bins=50)
df.hist(cumulative=True, normed=1, bins=100)

for d in data[:20]:
    print(d['text'])
    print(d['text'].split(' '))