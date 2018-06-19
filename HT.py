
# coding: utf-8

# ## Logic:
# 
# #### - compute attention scores
# #### - derive sentiment score lexicon
# #### - compute tf-idf
# #### - derive sentence-level sentiment scores
# #### - carry out t-tests
# 
# 

# ####  *References 
# 
# https://nlp.stanford.edu/projects/socialsent/  
# 
# https://nlp.stanford.edu/pubs/hamilton2016inducing.pdf

# ### Load files

# In[2]:


import cPickle
import csv
import operator
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.manifold import TSNE
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge as ridge
from sklearn.feature_extraction.text import TfidfVectorizer
import keras.preprocessing.text as tokenizer


# In[3]:


socialsent = pd.read_table('socialsent.tsv')[['word', 'score']]
token2id = cPickle.load(open('inputs_model.p', 'rb'))[1]
embedding = cPickle.load(open('embedding.p', 'rb'))


# In[4]:


ix_senti = [i for i in range(len(token2id)) if token2id.keys()[i] in list(socialsent['word'])]   # len = 1379
word_senti = [token2id.keys()[i] for i in ix_senti]                                              # len = 1379
emb_senti = embedding[ix_senti]                                                                  # shape = 1379 * 50
score_senti = [float(socialsent[socialsent['word']==token2id.keys()[i]]['score']) for i in ix_senti]

ix_nonSenti = [i for i in range(len(token2id)) if i not in ix_senti]   # len = 877
word_nonSenti = [token2id.keys()[i] for i in ix_nonSenti]              # len = 877
emb_nonSenti = embedding[ix_nonSenti]                                  # shape = 877 * 50


# In[27]:


## regressions

# X0 = emb_senti
# X = sm.add_constant(X0)
# y = score_senti
# est = sm.OLS(y, X).fit()
# print(est.summary())
# 
# X0_proj = TSNE(random_state=201806, n_components=3).fit_transform(X0)
# X_ = sm.add_constant(X0_proj)
# est = sm.OLS(y, X_).fit()
# print(est.summary())


# ### Attention Method

# In[96]:


corr = np.array([(np.matmul(emb_senti, emb_nonSenti[j]) / np.sqrt(sum(emb_nonSenti[j]*emb_nonSenti[j]))) /                  np.array([np.sqrt(sum(emb_senti[i]*emb_senti[i])) for i in range(len(emb_senti))])                  for j in range(len(emb_nonSenti))])
att = np.array([np.exp(corr[i])/sum(np.exp(corr[i])) for i in range(len(corr))])
score_nonSenti = [sum(att[i] * score_senti) for i in range(len(att))]


# In[134]:


word2_senti = {word_senti[i]:score_senti[i] for i in range(len(word_senti))}
word2_senti.update({word_nonSenti[i]:score_nonSenti[i] for i in range(len(word_nonSenti))})
word2_senti        # len = 2256


# ### tf-idf

# In[198]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def notes2corpus(notes):
    corpus = ''
    for note in notes:
        note = note.replace('\xc2', '')
        note = note.replace('\xa0', '')
        note = note.replace('\xe2', '')
        note = note.replace('\x80', '')
        note = note.replace('\x91', '')
        corpus += (note + ' ')
    return corpus


# In[200]:


pf_notes = pd.read_table('data/pratt_f.txt', sep='\n', header=None)[0]
pm_notes = pd.read_table('data/pratt_m.txt', sep='\n', header=None)[0]
tf_notes = pd.read_table('data/trinity_f.txt', sep='\n', header=None)[0]
tm_notes = pd.read_table('data/trinity_m.txt', sep='\n', header=None)[0]

pf_corpus = notes2corpus(pf_notes)
pm_corpus = notes2corpus(pm_notes)
tf_corpus = notes2corpus(tf_notes)
tm_corpus = notes2corpus(tm_notes)

corpus = []
corpus.append(pf_corpus)
corpus.append(pm_corpus)
corpus.append(tf_corpus)
corpus.append(tm_corpus)


# In[256]:


stopWords = set([u'jennifer', u'camilla', u'participant', u'jocelyn', u'Bridgette', u'laughs', u'inaudible', u'laughter', u'pause', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u"you're", u"you've", u"you'll", u"you'd", u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u"she's", u'her', u'hers', u'herself', u'it', u"it's", u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u"that'll", u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u"don't", u'should', u"should've", u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u"aren't", u'couldn', u"couldn't", u'didn', u"didn't", u'doesn', u"doesn't", u'hadn', u"hadn't", u'hasn', u"hasn't", u'haven', u"haven't", u'isn', u"isn't", u'ma', u'mightn', u"mightn't", u'mustn', u"mustn't", u'needn', u"needn't", u'shan', u"shan't", u'shouldn', u"shouldn't", u'wasn', u"wasn't", u'weren', u"weren't", u'won', u"won't", u'wouldn', u"wouldn't"])
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = stopWords)

tfidf_matrix =  tf.fit_transform(corpus)
tfidf_tokens = tf.get_feature_names() 
tfidf_scores = tfidf_matrix.todense()[0].tolist()[0]
for i in range(len(tfidf_scores)):
    if tfidf_scores[i] == 0:
        tfidf_scores[i] = np.random.uniform(0, 0.0015)
tfidf_scores_scaled = (np.array(tfidf_scores)-min(np.array(tfidf_scores))) / (max(np.array(tfidf_scores))-min(np.array(tfidf_scores)))

tfidf_dict = {tfidf_tokens[i]:tfidf_scores_scaled[i] for i in range(len(tfidf_tokens))}       # len = 2106


# In[257]:


sorted(tfidf_dict.items(), key=operator.itemgetter(1), reverse=True)


# ### Sentence Sentiment

# In[270]:


def doc2senti(notes):
    doc_senti_list = []
    for note in notes:
        note = note.replace('\xc2', '')
        note = note.replace('\xa0', '')
        note = note.replace('\xe2', '')
        note = note.replace('\x80', '')
        note = note.replace('\x91', '')
        note = tokenizer.text_to_word_sequence(note, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n', lower=True, split=' ')

        sent_score = 0
        for j in note:
            if (j in word2_senti.keys()) and (j in tfidf_tokens):
                token_score = word2_senti[j] * tfidf_dict[j]
            else:
                token_score = 0 
            sent_score += token_score
        
        doc_senti_list.append(sent_score)
    return doc_senti_list 


# In[272]:


pf_doc_sentiment = doc2senti(pf_notes)
pm_doc_sentiment = doc2senti(pm_notes)
tf_doc_sentiment = doc2senti(tf_notes)
tm_doc_sentiment = doc2senti(tm_notes)


# ### Hypothesis Tests

# In[285]:


## t-test for Pratt
stats.ttest_ind(pf_doc_sentiment, pm_doc_sentiment, equal_var=False)


# In[279]:


## t-test for Trinity
stats.ttest_ind(tf_doc_sentiment, tm_doc_sentiment, equal_var=False)

