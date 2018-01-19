from utils import *
from preprocess_utils import preprocess
import pickle

class Config:
    save_path = 'assets/corpora/corpus1/'
    rm_punctuation = True
    rm_stopwords = True
    lowercase = True

cfg = Config()
data = read_data(raw_data_dir,train_fn)

####
for d in data:
    text = d['text']



def process(data):
    data2 = []
    for k,item in enumerate(data):
        if k % 1000 == 0:
            print(k)
        text = item['text']
        words = preprocess(text)
        data2.append({'id':data['id'],
                      'words':words,
                      'label':data['label']})
    return data2

with open(cfg.save_path + 'train.corpus','wb') as f:
    pickle.dump(preprocessed_data,f)
