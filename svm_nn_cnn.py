
# coding: utf-8

# In[1]:


import tensorflow as tf
import keras
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import librosa 
audio_path = 'reggae.00000.wav' 
x , sr = librosa.load(audio_path)
librosa.load(audio_path, sr=44100)


# In[3]:



import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
plt.title('waveform of a song')
librosa.display.waveplot(x, sr=sr)


# In[4]:



X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()


# In[5]:



import tarfile 
myfile = tarfile.open('genres.tar.gz')

myfile.extractall()
cmap = plt.get_cmap('inferno')

plt.figure(figsize=(10,10))
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)     
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'img_data/{g}/{filename[:-3].replace(".", "")}.png')
        plt.clf()


# In[6]:



header = 'filename rmse chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


# In[7]:


# creating csv file, storing features 
file = open('data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        rmse = librosa.feature.rmse(y=y) 
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(rmse)} {np.mean(chroma_stft)}  {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())


# In[7]:



# reading csv file 
data = pd.read_csv('data.csv')

data.shape
# drop filename 
data = data.drop(['filename'],axis=1)
data.shape
data.head()


# In[8]:


genre_list = data.iloc[:, -1] # creating a list and indexing it 
# print(genre_list)
encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
# print(y)
print(len(y))
print(X.shape)


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# In[10]:


from sklearn import svm

# Create a classifier: a support vector classifier
# svc = svm.SVC(probability=False,  kernel="rbf", C=2.8, gamma=.0073,verbose=10)
svc = svm.SVC(probability=False,  kernel="linear", C=2.8, gamma=.0073,verbose=10)


# In[11]:


svc.fit(X_train,y_train)


# In[12]:


yhat_ts = svc.predict(X_test)


# In[13]:


acc = np.mean(yhat_ts == y_test)
print('Accuaracy = {0:f}'.format(acc))


# In[14]:


np.max(y_train)


# In[15]:


X_train.shape


# In[16]:


nin=X_train.shape[1]


# In[17]:


nin 


# In[18]:



from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers


# In[19]:


import tensorflow
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout,Dense


# In[45]:


nh = 150 # number of hidden units
nout = int(np.max(y_train)+1)    # number of outputs = 10 since there are 10 classes
model = Sequential()
model.add(Dense(units=nh, input_shape=(nin,), activation='sigmoid', name='hidden'))
model.add(Dense(units=nout, activation='softmax', name='output'))


# In[46]:


model.summary()


# In[47]:


from tensorflow.keras import optimizers

opt = optimizers.Adam(lr=0.01) # beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[1]:


hist = model.fit(X_train, y_train, epochs=30, batch_size=100, validation_data=(X_test,y_test))


# In[49]:


tr_accuracy = hist.history['acc']
val_accuracy = hist.history['val_acc']

plt.plot(tr_accuracy)
plt.plot(val_accuracy)
plt.grid()
plt.xlabel('epochs')
plt.ylabel('accuarcy')
plt.legend(['training accuracy', 'validation accuracy'])


# In[101]:



# from keras import backend as K
import tensorflow
from tensorflow.keras.models import Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout,Dense
K.clear_session() 

model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(10, activation='softmax'))

model.summary()

print(X_train.shape[1])


# In[16]:


from tensorflow.keras.optimizers import Adam

opt = Adam(lr=0.01)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[17]:


history = model.fit(X_train, y_train, epochs=100, verbose=0,batch_size=128,
                   validation_data=(X_test,y_test))


# In[18]:


plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.legend(['test', 'train'])


# In[19]:


test_loss, test_acc = model.evaluate(X_test,y_test)


# In[20]:


print(test_acc*100,'%')


# In[ ]:




