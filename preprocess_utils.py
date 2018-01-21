import nltk.data
from nltk.corpus import stopwords
import json
import re
import numpy as np
import collections
from langdetect import detect
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer

# text features
# words features

class CNNTransformer:

    def __init__(self):
        self.char2index = None
        self.index2char = None
        self.vocab_size = None
        self.max_len = 500
        self.unknown_char = 'ⓤ'
        self.pad_char = '℗'

        pass

    def create_vocabulary(self, texts, keep=0.9, min_count = 10):
        counter = collections.Counter()
        for k, text in enumerate(texts):
            counter.update(text)
        if keep:
            raw_counts = counter.most_common(int(keep*len(counter)))
        else:
            raw_counts = list(counter.items())

        vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > min_count]
        self.char2index = {char:ind for ind, char in enumerate(vocab)}
        self.char2index[self.unknown_char] = len(self.char2index)
        self.char2index[self.pad_char] = len(self.char2index)
        self.index2char = {ind:char for char, ind in self.char2index.items()}
        self.vocab_size = len(self.char2index)

    def convert_text_to_matrix(self,text):
        matrix = np.zeros((self.max_len,self.vocab_size))
        text = text[:self.max_len]
        for k,char in enumerate(text):
            try:
                ind = self.char2index[char]
                matrix[k, ind] = 1
            except KeyError:
                pass
        return matrix

    def convert_text_to_seq(self,text):
        seq = np.asarray(self.max_len * [self.char2index[self.pad_char]], dtype=np.int16)
        text = text[:self.max_len]
        for k,char in enumerate(text):
            try:
                ind = self.char2index[char]
            except KeyError:
                ind = self.char2index[self.unknown_char]
            seq[k] = ind
        return seq



class Tokenizer:

    def __init__(self,max_number_of_words = None,min_count_words=None,keep_words=0.9,min_count_chars=None,keep_chars = 0.9):
        self.tokenize_mode = 'nltk_twitter'
        if self.tokenize_mode == 'nltk_twitter':
            self.twitter_tokenizer = TweetTokenizer()
        self.sentence_detector = self._get_sentence_detector()
        self.punctuation = self._get_punctuation(level=0)
        self.stop_words = self._get_stopwords()
        self.contractions, self.contractions_re = self._get_contractions()
        self.word2index = None
        self.index2word = None
        self.word_counts = None
        self.char2index = None
        self.index2char = None
        self.char_counts = None
        self.max_number_of_words = max_number_of_words
        self.min_count_words = min_count_words
        self.min_count_chars = min_count_chars
        self.keep_words = keep_words
        self.keep_chars = keep_chars

    def _get_sentence_detector(self):
        detector = nltk.data.load('tokenizers/punkt/english.pickle')
        return detector

    def _get_punctuation(self, level=2):
        punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-',
                              u'–','+', '*', '--', '\'\'', '``', "'"]
        p_level1 = '?.!/;:()&+'
        p_level2 = ''.join(punctuation_tokens)
        p_level3 = '^\w ^_'

        punctuations = [p_level1, p_level2, p_level3]
        return punctuations[level]

    def _get_stopwords(self):
        stop_words = stopwords.words('english')
        return stop_words

    # might be improved using https://pypi.python.org/pypi/pycontractions/1.0.1

    def _get_contractions(self):
        contractions = {
            "ain't": "am not", "can't": "cannot", "aren't": "are not", "can't've": "cannot have", "'cause": "because",
            "could've": "could have", "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
            "don't": "do not", "doesn't": "does not",            "hadn't": "had not",            "hadn't've": "had not have",
            "hasn't": "has not",            "haven't": "have not",            "he'd": "he had",
            "he'd've": "he would have",            "he'll": "he shall",            "he'll've": "he shall have",            "he's": "he has",
            "how'd": "how did",            "how'd'y": "how do you",            "how'll": "how will",
            "how's": "how has",            "i'd": "I had",
            "i'd've": "I would have",
            "i'll": "I shall",
            "i'll've": "I shall have",
            "i'm": "I am",
            "i've": "I have",
            "isn't": "is not",
            "it'd": "it had",
            "it'd've": "it would have",
            "it'll": "it shall",
            "it'll've": "it shall have",
            "it's": "it has",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had",
            "she'd've": "she would have",
            "she'll": "she shall",
            "she'll've": "she shall have",
            "she's": "she has",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so is",
            "that'd": "that would",
            "that'd've": "that would have",
            "that's": "that is",
            "there'd": "there would",
            "there'd've": "there would have",
            "there's": "there is",
            "they'd": "they had",
            "they'd've": "they would have",
            "they'll": "they will",
            "they'll've": "they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what'll've": "what will have",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "when's": "when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where is",
            "where've": "where have",
            "who'll": "who will",
            "who'll've": "who will have",
            "who's": "who is",
            "who've": "who have",
            "why's": "why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you would",
            "you'd've": "you would have",
            "you'll": "you will",
            "you'll've": "you will have",
            "you're": "you are",
            "you've": "you have"
        }
        contractions_re = re.compile('(%s)' % '|'.join(contractions.keys()))
        return contractions, contractions_re

    def expand_contractions(self, text):
        def replace(match):
            return self.contractions[match.group(0)]

        return self.contractions_re.sub(replace, text)


    def rm_punctuation(self, text):
        text = re.sub('[' + self.punctuation + ']', '', text)
        return text

    @staticmethod
    def lower(text):
        text = text.lower()
        return text

    @staticmethod
    def rm_breaks(text):
        " ".join(text.split())
        return text

    @staticmethod
    def _get_language(text):
        lang = detect(text)
        return lang

    @staticmethod
    def rm_hyperlinks(words):
        words = [w for w in words if not (w.startswith('http') or w.startswith('www') or w.endswith('.com'))]
        return words

    @staticmethod
    def rm_links(text):
        text = re.sub("http://.*com","", text)
        text = re.sub("www.*com", "", text)
        return text

    @staticmethod
    def rm_user(text):
        text = re.sub("\[\[User(.*)\|","", text)
        return text

    @staticmethod
    def rm_ip(text):
        text = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","",text)
        return text

    @staticmethod
    def rm_article_id(text):
        text = re.sub("\d:\d\d\s{0,5}$","" ,text)
        return text

    def rm_leaky_features(self,text):
        text = self.rm_links(text)
        text = self.rm_article_id(text)
        text = self.rm_ip(text)
        text = self.rm_user(text)
        text = self.rm_breaks(text)
        return text

    def tokenize(self,text):
        if self.tokenize_mode == 'nltk_word':
            words = word_tokenize(text)
        elif self.tokenize_mode == 'nltk_twitter':
            words = self.twitter_tokenizer.tokenize(text)
        elif self.tokenize_mode == ' ':
            words = text.split(' ')
        else:
            print('wrong mode')
            words = None
            pass
        return words


    def sentenize(self,text):
        sentences = self.sentence_detector.tokenize(text)
        return sentences


    def rm_stopwords(self,words):
        words = [w for w in words if not w in self.stop_words]
        return words

    def tokens2seq(self):
        pass

    def fit_on_text(self,list_of_words):
        counter_chars = collections.Counter()
        counter_words = collections.Counter()
        for k, words in enumerate(list_of_words):
            for w in words:
                counter_words.update(w)
        # count words
        # create word2index
        # create index2word
        pass

    #def preprocess(text):
    #    text = rm_linebreak(text)
    #    lang = detect(text)
    #    if lang == 'en':
    #        text = expand_contractions(text)
    #        text = lower(text)
    #        text = rm_punctuation(text)
    #        words = tokenize(text, mode='nltk')
    #        words = rm_stopwords(words)
    #        words = rm_hyperlinks(words)
    #        return words

    def _get_word_index(self):
        pass

if __name__ == '__main__':

    raw_data_dir = 'review_based/assets/raw_data/'
    raw_data_fn = 'review.json'
    json_data = open(raw_data_dir + raw_data_fn, 'r').readlines()

    counter = collections.Counter()
    for k, line in enumerate(json_data[:2000]):
        print(k)
        text = json.loads(line)['text']
        words = preprocess(text)
        if words:
            for w in words:
                counter.update(w)
    print(counter)
    print('len %s' % len(counter))
