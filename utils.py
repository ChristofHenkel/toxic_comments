import matplotlib
matplotlib.use('TkAgg')
import os
import pandas as pd

labels = ['identity_hate', 'insult', 'obscene', 'severe_toxic', 'threat', 'toxic']
label2id = {name:id for id,name in enumerate(labels)}
id2label = {id:name for id,name in enumerate(labels)}
raw_data_dir = 'assets/raw_data/'
train_fn = 'train.csv'
test_fn = 'test.csv'

def read_metadata(data_dir, filename):
    meta_filepath = os.path.join(data_dir, filename)
    meta_data = pd.read_csv(meta_filepath)
    return meta_data

def read_data(data_dir, filename):
    meta_data = read_metadata(data_dir, filename)
    data = meta_data.to_dict('index')
    data = list(data.values())
    data = [{'id': item['id'], 'text': item['comment_text'], 'label': [item[label] for label in labels]} for item in data]
    return data

def create_textcorpus(corpus,fp_out):
    textcorpus = [item['text'] for item in corpus]
    with open(fp_out,'w') as f:
        f.writelines(textcorpus)

def create_ft_txt(corpus,fp_out):
