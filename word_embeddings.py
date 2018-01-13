from gensim.models.fasttext import FastText
import logging

logging.basicConfig(level=logging.INFO)

lines = open('test.txt').readlines()
sentences = [item.split(' ') for item in lines]

model = FastText(sg=0,bucket = 100000)
model.build_vocab(sentences)
model.train(sentences)
model.save('test.model_untrained')

model2 = FastText.load('test.model_untrained')
model2.train(sentences, )