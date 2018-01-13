from utils import *
from preprocess import rm_punctuation, rm_stopwords, tokenize, lower_case
import pickle

class Config:
    save_path = 'assets/corpora/corpus1/'


    rm_punctuation = True
    rm_stopwords = True
    lowercase = True

cfg = Config()
data = read_data(raw_data_dir,train_fn)

def preprocess_text(data):
    for k,item in enumerate(data):
        if k % 1000 == 0:
            print(k)
        words = tokenize(item['text'])
        if cfg.rm_punctuation:
            words = rm_punctuation(words)
        if cfg.lowercase:
            words = lower_case(words)
        if cfg.rm_stopwords:
            words = rm_stopwords(words)
        item['text'] = ' '.join(words)
    return data

preprocessed_data = preprocess_text(data)

with open(cfg.save_path + 'train.corpus','wb') as f:
    pickle.dump(preprocessed_data,f)
