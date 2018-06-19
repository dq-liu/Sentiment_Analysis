
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import cPickle
import keras.preprocessing.text as tokenizer
import gc
import string


# In[17]:


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def convert_word_to_ix(data):
    result = []
    i=1
    x=set(wordtoix.keys())
    for sent in data:
        temp = []
        for w in sent:
            if w in x:
                temp.append(wordtoix[w])
            else:
                temp.append(1)
        temp.append(0)
        result.append(temp)
    return result


# In[36]:


pf_notes = pd.read_table('data/pratt_f.txt', sep='\n', header=None)[0]
pm_notes = pd.read_table('data/pratt_m.txt', sep='\n', header=None)[0]
tf_notes = pd.read_table('data/trinity_f.txt', sep='\n', header=None)[0]
tm_notes = pd.read_table('data/trinity_m.txt', sep='\n', header=None)[0]

notes = pf_notes.append(pm_notes, ignore_index=True).append(tf_notes, ignore_index=True).append(tm_notes, ignore_index=True)


# In[45]:


# stop_words = set([u'laughs', u'inaudible', u'laughter', u'pause', u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u"you're", u"you've", u"you'll", u"you'd", u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u"she's", u'her', u'hers', u'herself', u'it', u"it's", u'its', u'itself', u'they', u'them', u'their', u'theirs', u'themselves', u'what', u'which', u'who', u'whom', u'this', u'that', u"that'll", u'these', u'those', u'am', u'is', u'are', u'was', u'were', u'be', u'been', u'being', u'have', u'has', u'had', u'having', u'do', u'does', u'did', u'doing', u'a', u'an', u'the', u'and', u'but', u'if', u'or', u'because', u'as', u'until', u'while', u'of', u'at', u'by', u'for', u'with', u'about', u'against', u'between', u'into', u'through', u'during', u'before', u'after', u'above', u'below', u'to', u'from', u'up', u'down', u'in', u'out', u'on', u'off', u'over', u'under', u'again', u'further', u'then', u'once', u'here', u'there', u'when', u'where', u'why', u'how', u'all', u'any', u'both', u'each', u'few', u'more', u'most', u'other', u'some', u'such', u'no', u'nor', u'not', u'only', u'own', u'same', u'so', u'than', u'too', u'very', u's', u't', u'can', u'will', u'just', u'don', u"don't", u'should', u"should've", u'now', u'd', u'll', u'm', u'o', u're', u've', u'y', u'ain', u'aren', u"aren't", u'couldn', u"couldn't", u'didn', u"didn't", u'doesn', u"doesn't", u'hadn', u"hadn't", u'hasn', u"hasn't", u'haven', u"haven't", u'isn', u"isn't", u'ma', u'mightn', u"mightn't", u'mustn', u"mustn't", u'needn', u"needn't", u'shan', u"shan't", u'shouldn', u"shouldn't", u'wasn', u"wasn't", u'weren', u"weren't", u'won', u"won't", u'wouldn', u"wouldn't"])

sent = []
vocab = {}
x = set(vocab.keys())
for note in notes:
    note = note.replace('\xc2', '')
    note = note.replace('\xa0', '')
    note = note.replace('\xe2', '')
    note = note.replace('\x80', '')
    note = note.replace('\x91', '')
    
    temp = tokenizer.text_to_word_sequence(note, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\n', lower=True, split=' ')
    temp = [j if not is_number(j) else '0' for j in temp]
    sent.append(temp)
    temp = set(temp)
    for word in temp:
        if word in x:
            vocab[word] += 1
        else:
            vocab[word] = 1

vocab = {x:y for x, y in vocab.items()}

ixtoword = {}
ixtoword[0] = 'END'
ixtoword[1] = 'UNK'
wordtoix = {}
wordtoix['END'] = 0
wordtoix['UNK'] = 1

ix = 2
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

train_x = convert_word_to_ix(sent)


# In[48]:


cPickle.dump([vocab, wordtoix, ixtoword, train_x, sent], open('inputs_model.p', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
