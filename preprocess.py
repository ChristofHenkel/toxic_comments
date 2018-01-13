import nltk.data
from nltk.corpus import stopwords
import re


sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-', u'–',
                      '+', '*', '--', '\'\'', '``', "'"]
punctuation = '?.!/;:()&+'
stop_words = stopwords.words('english')

def lower_case(words):
    words = [x.lower() for x in words]
    return words

def rm_punctuation(words):
    words = [x for x in words if x not in punctuation_tokens]
    words = [re.sub('[' + punctuation + ']', '', x) for x in words]
    return words

def rm_stopwords(words):
    words = [x for x in words if x not in stop_words]
    return words

def tokenize(text):
    words = nltk.word_tokenize(text)
    return words
