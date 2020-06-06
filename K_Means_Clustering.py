
# coding: utf-8

# In[1]:


from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import sys
from time import time

import numpy as np


# In[2]:


df = pd.read_csv('lyrics.csv')
df.head(10)


# In[3]:


df['lyrics'].replace('', np.nan, inplace=True)
df.dropna(subset=['lyrics'], inplace=True)
ind_drop = df[df['genre'].apply(lambda x: x.startswith('Other'))].index
df = df.drop(ind_drop)


# In[4]:


ind_drop = df[df['genre'].apply(lambda x: x.startswith('Not Available'))].index
df = df.drop(ind_drop)


# In[5]:


ind_drop = df[df['lyrics'].apply(lambda x: x.startswith('INSTRUMENTAL'))].index
df = df.drop(ind_drop)
df.drop(columns=['index'])

ind_drop = df[df['lyrics'].apply(lambda x: x.startswith('instrumental'))].index
df = df.drop(ind_drop)
df.drop(columns=['index'])


# In[6]:


genre=df['genre'].values
lyrics=df['lyrics'].values
true_k = len(np.unique(genre))
print(np.unique(genre), "The total number of genres are", true_k)


#shaping:
lyrics = np.array(lyrics)[:,None]
print(lyrics.shape)
genre = np.array(genre)[:,None]
print(genre.shape)


# In[7]:


data = np.append(lyrics,genre,axis=1)
data.shape
print(data)


# In[8]:


np.random.shuffle(data)

data_test = data[10001:20001,]
data = data[:10000,]


# In[9]:


data_lyrics=data[:,0]
data_genre=data[:,1]

data_lyrics_test = data_test[:,0]
data_genre_test = data_test[:,1]

# print(data_lyrics)
# print(data_genre.shape)


# In[10]:


vectorizer = TfidfVectorizer( 
                max_df=0.75, # max doc freq (as a fraction) of any word to include in the vocabulary
                min_df=0.3,   # min doc freq (as doc counts) of any word to include in the vocabulary
                max_features=10000,           # max number of words in the vocabulary
                stop_words='english',         # remove English stopwords
                use_idf=True ) 


# In[11]:


labels={'Country':1, 'Electronic':2, 'Folk':3, 'Hip-Hop':4, 'Indie':5, 'Jazz':6,
       'Metal':7, 'Pop':8, 'R&B':9, 'Rock':10}

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
vectorizer.fit(data_lyrics)

X = vectorizer.transform(data_lyrics)
Y = [labels[i] for i in data_genre]

X_test = vectorizer.transform(data_lyrics_test)
Y_test = [labels[i] for i in data_genre_test]

n_features = X.shape[1]
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)


# In[12]:


doc_ind = 1  # Index of an example document
xi = X[doc_ind,:].todense()
term_ind = xi.argsort()[:, ::-1]
xi_sort = xi[0,term_ind]
terms = vectorizer.get_feature_names()

for i in range(n_features):
    term = terms[term_ind[0,i]]
    tfidf = xi[0,term_ind[0,i]]
    print('{0:20s} {1:f} '.format(term, tfidf))


# In[13]:


km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
                verbose=True)


# In[14]:


print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()


# In[15]:


order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(true_k):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()


# In[16]:


labels={'Country':1, 'Electronic':2, 'Folk':3, 'Hip-Hop':4, 'Indie':5, 'Jazz':6,
       'Metal':7, 'Pop':8, 'R&B':9, 'Rock':10}
print(labels.values)
# genre_names
# data_genre
genre_labels=[]
#print(genre_labels.shape)
for j,i in enumerate(data_genre):
    x=labels[i]
    #print(x)
    np.append(genre_labels,x)
    genre_labels.append(x)
#print(genre_labels)


# In[17]:


print((Y_test == km.predict(X_test)).sum() / len(Y_test))


# In[18]:


labelkm = km.labels_
print(labelkm.shape)
print(type(labelkm))


# In[19]:


#print(data_genre)
labelkm = km.labels_
from sklearn.metrics import confusion_matrix
C = confusion_matrix(genre_labels,labelkm)

Csum = np.sum(C,axis=0)
Cnorm = C / Csum[None,:]
print(Cnorm)
print(np.array_str(C, precision=3, suppress_small=True))
plt.imshow(C, interpolation='none')
plt.colorbar()

