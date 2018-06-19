
# coding: utf-8

# In[1]:


import cPickle
import pandas as pd
import numpy as np
import gc
import gensim
from gensim.models import word2vec


# In[ ]:


def load_embedding_vectors_glove_w2v(vocabulary, filename):
    print("loading embedding")
    model = gensim.models.Word2Vec.load(filename)
    vector_size = model.vector_size
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    glove_vocab = set(model.wv.vocab.keys())
    count = 0
    mis_count = 0
    for word in vocabulary.keys():
        idx = vocabulary.get(word)
        if word in glove_vocab:
            embedding_vectors[idx] = model.wv[word]
            count += 1
        else:
            mis_count += 1
    print("num of vocab in glove: {}".format(count))
    print("num of vocab not in glove: {}".format(mis_count))
    return embedding_vectors


# In[32]:


sent = cPickle.load(open('inputs_model.p', 'rb'))[4]
token2id = cPickle.load(open('inputs_model.p', 'rb'))[1]
sent = np.array(sent)


# In[51]:


model = word2vec.Word2Vec(sent, size=50, window=30, min_count=1, workers=4)
model.save('model_1')


# In[52]:


embedding = load_embedding_vectors_glove_w2v(token2id, 'model_1')


# In[53]:


cPickle.dump(embedding, open('embedding.p', 'wb'))

