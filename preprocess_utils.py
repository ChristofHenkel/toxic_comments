import nltk.data
from nltk.corpus import stopwords
import json
import itertools
import re
import numpy as np
import collections
from langdetect import detect
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer
import logging
from utilities import load_bad_words
from textblob import TextBlob
import tqdm
from num2words import num2words
from global_variables import COMMENT

logging.basicConfig(level=logging.INFO)

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

    def create_char_vocabulary(self, texts, keep=0.9, min_count = 10):
        counter = collections.Counter()
        for k, text in enumerate(texts):
            counter.update(text)
        if keep:
            raw_counts = counter.most_common(int(keep*len(counter)))
        else:
            raw_counts = list(counter.items())

        vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > min_count]
        self.char2index = {char:(ind+1) for ind, char in enumerate(vocab)}
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


class Preprocessor:

    def __init__(self,max_number_of_words = None,min_count_words=5,keep_words=0.9,min_count_chars=20,keep_chars = 0.9):
        self.tokenize_mode = 'keras'
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
        self.unknown_char = 'ⓤ'
        self.pad_char = '℗'
        self.unknown_word = '<unk>'
        self.pad_word = '<pad>'
        self.char_vocab_size = None
        self.word_vocab_size = None
        #self.bad_words, self.bad_words_synonyms = load_bad_words()

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
            "how's": "how has",            "i'd": "i had",
            "i'd've": "i would have",
            "i'll": "i shall",
            "i'll've": "i shall have",
            "i'm": "i am",
            "i've": "i have",
            'marcolfuck':'marcol fuck',
            'wikiprojects':'wiki projects',
            'youbollocks':'you bull shit',
            'ancestryfuck':'ancestry fuck',
            'ricehappy':'rice happy',
            'aidsaids':'aids aids',
            'smileyrick':'smiley rick',
            'wikipediahappy':'wikipedia happy',
            'talkhappy':'talk happy',
            'talklol':'talk lol',
            'userhappy':'user happy',
            'mainpagebg':'mainpage background',
            '@ggot':'faggot',
            'smileyrecious':'smiley recious',
            'nooob':'noob',
            'urlsmiley':'url smiley',
            'ashol':'asshole',
            'smileyp':'smiley',
            'latinus':'latino',
            'userlol':'user lol',
            "god's":'gods',
            "pneis":'penis',
            "else's":'else his',
            'pennnis':'penis',
            'youfuck':'you fuck',
            'phuq':'fuck',
            'philippineslong':'philippines long',
            "women's":'womens',
            'wplol':'wikipedia lol',
            "editor's":'editors',
            'itsuck':'it suck',
            "offfuck":'off fuck',
            'tommytwo':'tommy two',
            "file's":'files',
            "other's":'others',
            "gayfrozen":'gay frozen',
            "mother's":'mothers',
            "gayfag":'gay faggot',
            "ip's":'ips',
            "men's":'mens',
            "today's":'todays',
            "mothjer":'mother mispelled',
            "isn't": "is not",
            "it'd": "it had",
            "anyone's":'anyones',
            "website's":'websites',
            "wiki's":'wikis',
            "page's":'page is',
            "aseven":'as even',
            "wikipedia's":'wikipedias',
            'npov':'neutral point of view',
            "world's":'worlds',
            "user's":'users',
            "securityfuck": 'security fuck',
            "one's":'ones',
            'néger':'nigger',
            "author's":'authors',
            'roflspam':'rofl spam',
            'niggors':'niggers',
            'helloz':'hello',
            'phck':'fuck',
            'bonergasm':'boner orgasm',
            'schäbig':'lame',
            'bitchbot':'bitch robot',
            'donkeysex':'donkey sex',
            'faggt':'faggot',
            'niggerjew':'nigger jew',
            'dixz':'dicks',
            'gayyour':'gay your',
            'smileyo':'smile yo',
            'backgrounhappy':'background happy',
            'vaginapenis':'vagina penis',
            'wphappy':'wp happy',
            'smileyist':'smiley ist',
            'radicalnigger':'radical nigger',
            'oldihappy':'oldi happy',
            'smileyx':'smiley x',
            'peenus':'penis',
            'motherfuckerdie':'motherfucker die',
            'homopetersymonds':'homo peter symonds',
            'honkhonk':'honk honk',
            'analanal':'anal anal',
            "sex'butt":"sex butt",
            "here's":'here is',
            "subject's":'subject is',
            'fucksex':'fuck sex',
            'smileyol':'smiley',
            'yourselfgo':'yourself go',
            "fggt":'faggot',
            "person's":'persons',
            "man's":"mans",
            "article's":'articles',
            "it'd've": "it would have",
            "it'll": "it shall",
            "it'll've": "it shall have",
            "it's": "it has",
            '#zero':'zero',
            'pagedelete':'page delete',
            'addressip':'address ip',
            "image's":'images',
            'imagehappy':'image happy',
            'imagelol':'image lol',
            'slimvirgin':'slim virgin',
            "let's": "let us",
            "ma'am": "madam",
            'b00ll00x':'bull shit',
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "people's":'peoples',
            'cuntfranks':'cunt franks',
            "3rr":'three revert rule',
            '#f5fffa': 'mint green',
            '`':' ',
            'roycy':'badass',
            '@hotmail':'email adress',
            'fvckers':'fuckers',
            'suckernguyen':'sucker nguyen',
            'turkeyfuck':'turkey fuck',
            'wpneutral':'wp neutral',
            'faggotgay':'faggot gay',
            'cuntliz':'cunt liz',
            'smileylease':'smiley lease',
            'sucksgeorge':'sucks george',
            'hornyhorny':'horny horny',
            'headsdick':'heads dick',
            'helloe':'hello',
            'kfuckity':'fuck city',
            'smileyi':'smiley',
            'ballsballs':'balls balls',
            'serbiafack':'serbia fuck',
            "must've": "must have",
            'wikipedialol':'wikipedia lol',
            'wikilove':'wiki love',
            'penispenis':'penis penis',
            'fagsgod':'fags god',
            'nigggers':'niggers',
            'bitchbitch':'bitch bitch',
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

    @staticmethod
    def strip_spaces(words):
        return [w.replace(' ', '') for w in words]

    def replace_badwords_with_syn(self,text):
        for word in self.bad_words_synonyms:
            text = text.replace(word, self.bad_words_synonyms[word])
        return text

    def rm_punctuation(self, text):
        text = re.sub('[' + self.punctuation + ']', '', text)
        return text

    @staticmethod
    def lower(text):
        text = text.lower()
        return text

    @staticmethod
    def rm_breaks(text):
        text = text.replace('\n', ' ')
        " ".join(text.split())
        return text

    @staticmethod
    def _get_language(text):
        lang = detect(text)
        return lang

    @staticmethod
    def rm_hyperlinks(words):
        words = [w if not (w.startswith('http') or
                           w.startswith('www') or
                           w.endswith('.com') or
                            w.startswith('en.wikipedia.org/')) else 'url' for w in words]
        return words


    @staticmethod
    def rm_links_text(text):
        text = re.sub("http?s://.* ","url", text)
        text = re.sub("www.* ", "url", text)
        return text

    @staticmethod
    def replace_numbers(text):

        years = re.findall('[1-2][0-9]{3}.', text)
        for n in years:
            try:
                text = text.replace(n[:-1],num2words(int(n[:-1]),to='year') + ' ')
            except:
                continue
        numbers = re.findall('\d{1,2}.[^\d{3,}]', text)
        for n in numbers:
            try:
                text = text.replace(n[:-1],num2words(int(n[:-1])) + ' ')
            except:
                continue
        return text

    @staticmethod
    def rm_links_words(words):
        words = [w for w in words if not (w.startswith('http') or w.startswith('www.') or w.endswith('.com'))]
        return words

    @staticmethod
    def rm_user(text):
        text = re.sub("\[\[User(.*)\|","user-id", text)
        return text

    @staticmethod
    def replace_ip(text):
        text = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}","ip-address",text)
        return text

    @staticmethod
    def rm_article_id(text):
        text = re.sub("\d:\d\d\s{0,5}$","article-id" ,text)
        return text

    @staticmethod
    def rm_bigrams(text):
        text = re.sub(r'[-–_]',' ',text)
        return text

    @staticmethod
    def isolate_punc(text):
        text = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', text)
        return text

    @staticmethod
    def replace_smileys(text):
        """
        adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

        """
        eyes = "[8:=;]"
        nose = "['`\-]?"
        # Different regex parts for smiley faces
        text = re.sub("<3", 'heart emoji', text)
        text = re.sub(eyes + nose + "[Dd)\]]", 'happy smiley', text)
        text = re.sub("[(d]" + nose + eyes, 'happy smiley', text)
        text = re.sub(eyes + nose + "p", 'lol smiley', text)
        text = re.sub(eyes + nose + "\(", 'sad smiley', text)
        text = re.sub("\)" + nose + eyes, 'sad smiley', text)
        text = re.sub(eyes + nose + "[/|l*]", 'neutral smiley', text)

        return text

    def rm_leaky_features(self,text):
        text = self.rm_links_text(text)
        text = self.rm_article_id(text)
        text = self.replace_ip(text)
        text = self.rm_user(text)
        text = self.rm_breaks(text)
        return text

    def text_to_word_sequence(self,text,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=' '):
        """Converts a text to a sequence of words (or tokens).

        Arguments:
            text: Input text (string).
            filters: Sequence of characters to filter out.
            lower: Whether to convert the input to lowercase.
            split: Sentence split marker (string).

        Returns:
            A list of words (or tokens).
        """
        if lower:
            text = text.lower()
        maketrans = str.maketrans
        translate_map = maketrans(filters, split * len(filters))

        text = text.translate(translate_map)
        seq = text.split(split)
        return [i for i in seq if i]

    def simple_tokenize(self,text):
        if self.tokenize_mode == 'nltk_word':
            words = word_tokenize(text)
        elif self.tokenize_mode == 'nltk_twitter':
            words = self.twitter_tokenizer.tokenize(text)
        elif self.tokenize_mode == ' ':
            words = text.split(' ')
        elif self.tokenize_mode == 'keras':
            words = self.text_to_word_sequence(text)
        else:
            print('wrong mode')
            words = None
            pass
        return words

    def word2seq(self,word,maxlenseq):
        word = word[:maxlenseq]
        seq = np.zeros(maxlenseq)
        if not word == self.pad_word:
            for k, c in enumerate(word):
                seq[k] = self.char2index[c]
        return seq

    def words2charseq(self,words,maxlenseq):
        matrix = np.zeros((len(words),maxlenseq))
        for k,word in enumerate(words):
            seq = self.word2seq(word, maxlenseq)
            matrix[k] = seq
        return matrix

    def sentenize(self,text):
        sentences = self.sentence_detector.tokenize(text)
        return sentences


    def rm_stopwords(self,words):
        words = [w for w in words if not w in self.stop_words]
        return words

    def tokens2seq(self,tokens):
        seq = np.asarray([self.word2index[token] for token in tokens])
        return seq

    def seq2gazetter(self,seq):
        known_bw = [w for w in self.bad_words if w in self.word2index]
        gazetter = np.zeros((len(known_bw)))
        for k, g in enumerate(gazetter):
            if self.word2index[known_bw[k]] in seq:
                gazetter[k] = 1
        return gazetter

    def texts_to_sequences(self, texts):
        logging.info('converting %s lines to sequences' %len(texts))
        sequences = [self.tokens2seq(self.tokenize(text)) for text in texts]
        return sequences

    def seq2words(self,seq):
        words = [self.index2word[index] if index != 0 else self.pad_word for index in seq]
        return words

    def seqs_to_char_sequences(self, seqs, maxlenseq=12):
        logging.info('converting %s lines to character sequences' %len(seqs))
        list_of_tokens = [self.seq2words(seq) for seq in seqs]
        char_sequences = [self.words2charseq(tokens,maxlenseq=maxlenseq) for tokens in list_of_tokens]
        return char_sequences

    def create_char_vocabulary(self, texts):
        counter = collections.Counter()
        for k, text in enumerate(texts):
            counter.update(text)
        if self.keep_chars:
            logging.info('keeping top %s percent characters' % (self.keep_chars * 100))
            raw_counts = counter.most_common(int(self.keep_chars*len(counter)))
        else:
            raw_counts = list(counter.items())
        logging.info('%s remaining characters' %len(counter))
        logging.info('keepin characters with count > %s' % self.min_count_chars)
        vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > self.min_count_chars]
        self.char2index = {char:(ind+1) for ind, char in enumerate(vocab)}
        self.char2index[self.unknown_char] = 0
        self.char2index[self.pad_char] = -1
        self.index2char = {ind:char for char, ind in self.char2index.items()}
        self.char_vocab_size = len(self.char2index)
        logging.info('%s remaining characters' % self.char_vocab_size)

    def initial_cleaning(self,text):
        text = self.lower(text)
        text = self.expand_contractions(text)
        text = self.rm_bigrams(text)
        text = self.replace_badwords_with_syn(text)
        text = self.rm_leaky_features(text)
        text = self.rm_breaks(text)
        return text


    def fit_on_texts(self,list_of_texts,rm_stopwords = True):
        texts = [self.initial_cleaning(text) for text in list_of_texts]
        logging.info('creating character vocab')
        self.create_char_vocabulary(texts)
        logging.info('Done - kept %s characters' %len(self.char2index))
        logging.info('Getting Words')
        list_of_words = [self.simple_tokenize(text) for text in texts]
        list_of_words = [self.rm_links_words(words) for words in list_of_words]
        if rm_stopwords: list_of_words = [self.rm_stopwords(words) for words in list_of_words]
        logging.info('creating word vocab')
        #replace unknown characters in words and count
        #r = re.compile(r'[^' + ''.join(self.char2index.keys()) + ']')
        counter_words = collections.Counter()
        for k, words in enumerate(list_of_words):
            for word in words:
                word = self.replace_unknowns(word)
                counter_words.update([word])

        if self.keep_words:
            raw_counts = counter_words.most_common(int(self.keep_words*len(counter_words)))
        else:
            raw_counts = list(counter_words.items())

        #word_vocab = [char_tuple[0] for char_tuple in raw_counts if char_tuple[1] > self.min_count_words]
        raw_counts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in raw_counts]
        sorted_voc = [w for w in sorted_voc if counter_words[w] > self.min_count_words]
        # note that index 0 is reserved, never assigned to an existing word
        self.word2index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))


        #self.word2index = {word:ind for ind, word in enumerate(word_vocab)}
        self.word2index[self.unknown_word] = len(self.word2index)+1
        self.word2index[self.pad_word] = len(self.word2index)+1
        self.index2word = {ind:word for word, ind in self.word2index.items()}
        self.word_vocab_size = len(self.word2index)
        logging.info('Done - kept %s words' %len(self.word2index))

    def replace_unknowns(self,word):
        new_word = ''.join(c if c in self.char2index else self.unknown_char for c in word)
        return new_word

    def get_gazetter_(self,tokens):
        pass


    def tokenize(self,text, rm_stopwords = True):
        text = self.initial_cleaning(text)
        words = self.simple_tokenize(text)
        words = self.rm_links_words(words)

        tokens = [self.replace_unknowns(word) for word in words]
        tokens = [w if w in self.word2index else self.unknown_word for w in tokens]
        if rm_stopwords: tokens = self.rm_stopwords(tokens)
        return tokens

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

    def char2seq(self, texts, maxlen):
        res = np.zeros((len(texts),maxlen))
        for k,text in tqdm.tqdm(enumerate(texts)):
            seq = np.zeros((len(text)))
            for l, char in enumerate(text):
                try:
                    id = self.char2index[char]
                    seq[l] = id
                except KeyError:
                    seq[l] = self.char2index[self.unknown_char]
            seq = seq[:maxlen]
            res[k][:len(seq)] = seq
        return res

    @staticmethod
    def glove_preprocess(text):
        """
        adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

        """
        # Different regex parts for smiley faces
        eyes = "[8:=;]"
        nose = "['`\-]?"
        text = re.sub("https?:* ", "<URL>", text)
        text = re.sub("www.* ", "<URL>", text)
        text = re.sub("\[\[User(.*)\|", '<USER>', text)
        text = re.sub("<3", '<HEART>', text)
        text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
        text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
        text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
        text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
        text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
        text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
        text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
        text = re.sub("/", " / ", text)
        text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
        text = re.sub("([!]){2,}", "! <REPEAT>", text)
        text = re.sub("([?]){2,}", "? <REPEAT>", text)
        text = re.sub("([.]){2,}", ". <REPEAT>", text)
        pattern = re.compile(r"(.)\1{2,}")
        text = pattern.sub(r"\1" + " <ELONG>", text)

        return text


def preprocess(data, add_polarity = False):

    print('preprocessing')
    p = Preprocessor()
    data[COMMENT] = data[COMMENT].map(lambda x: p.lower(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_breaks(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.expand_contractions(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.replace_smileys(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.replace_ip(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_links_text(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.replace_numbers(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.rm_bigrams(x))
    data[COMMENT] = data[COMMENT].map(lambda x: p.isolate_punc(x))

    if add_polarity:
        print('adding polarity')
        zpolarity = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight',
                     9: 'nine', 10: 'ten'}
        zsign = {-1: 'negative', 0.: 'neutral', 1: 'positive'}
        data['polarity'] = data[COMMENT].map(lambda x: int(TextBlob(x).sentiment.polarity * 10))
        data[COMMENT] = data.apply(lambda r: str(r[COMMENT]) + ' polarity' + zsign[np.sign(r['polarity'])] + zpolarity[np.abs(r['polarity'])],axis=1)

    #data[COMMENT] = data[COMMENT].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    return data



