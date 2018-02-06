import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import
df = pd.read_csv("assets/raw_data/train.csv")
df['total_length'] = df['comment_text'].apply(len)
df['capitals'] = df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
df['caps_vs_length'] = df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
df['num_exclamation_marks'] = df['comment_text'].apply(lambda comment: comment.count('!'))
df['num_question_marks'] = df['comment_text'].apply(lambda comment: comment.count('?'))
df['num_marks']= df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '!?'))
df['num_punctuation'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '.,;:'))
df['num_symbols'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in '*&$#%'))
df['num_words'] = df['comment_text'].apply(lambda comment: len(comment.split()))
df['num_unique_words'] = df['comment_text'].apply(lambda comment: len(set(w for w in comment.split())))
df['words_vs_unique'] = df['num_unique_words'] / df['num_words']
df['num_smilies'] = df['comment_text'].apply(lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))


# psycho features
df['sixltr'] = df['comment_text'].apply(lambda comment: len([w for w in comment.split() if len(w) > 6])) / df['num_words']
# wps
# count you, ya
# count third person
# count I, We

# I, my, me, we, our, us

def plot():
    features = ('total_length', 'capitals', 'caps_vs_length', 'num_exclamation_marks',
                'num_question_marks','num_marks', 'num_punctuation', 'num_words', 'num_unique_words',
                'words_vs_unique', 'num_smilies', 'num_symbols','sixltr')
    columns = ('toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate')

    rows = [{c:df[f].corr(df[c]) for c in columns} for f in features]
    df_correlations = pd.DataFrame(rows, index=features)
    ax = sns.heatmap(df_correlations, vmin=-0.2, vmax=0.2, center=0.0)


X_train = hstack((
    vectorizer.transform(df['comment_text']),
    csr_matrix(np.reshape(df['caps_vs_length'].values, (df.shape[0], 1))),
    csr_matrix(np.reshape(df['num_unique_words'].values, (df.shape[0], 1)))
))

import re, string
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

train = pd.read_csv('assets/raw_data/train.csv')
test = pd.read_csv('assets/raw_data/test.csv')
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()

COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = vec.fit_transform(train[COMMENT])
test_term_doc = vec.transform(test[COMMENT])

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)