#!/usr/bin/env python
# coding: utf-8

# # Detection of PCL

# In[2]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
import torch

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, recall_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec

#for imbalanced data
from imblearn.over_sampling import SMOTE

#for neural networks
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.backend import clear_session

#for BERT
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
from transformers import Trainer, TrainingArguments


# In[3]:


#Check working directory and change to 'Downloads' where dataset exists
os.chdir('..\Downloads')
os.getcwd()


# In[4]:


f = "dontpatronizeme_pcl.tsv"

#Rename columns corresponding to paragraph id, article id, keyword,country, text, and pcl label
features = ['par','art','keyword','country','txt','lab' ]
df= pd.read_csv(f,sep='\t',skiprows=3,names=features)


# In[5]:


#Reassign labels 0 and 1 as 0 (no pcl) and 2,3,4 as 1 (presence of pcl)
old_labs = [0,1,2,3,4]
new_labs = [0,0,1,1,1]
df['lab'] = df['lab'].replace(old_labs,new_labs)


# # Exploratory Data Analysis

# In[7]:


#Explore class distribution
num_0s = len(df['lab'][df['lab']==0])
num_1s = len(df['lab'][df['lab']==1])
print("Proportion of data that is pcl:",round(num_1s/(num_1s+num_0s),3))

#Plot bar graph
pd.value_counts(df['lab']).plot.bar()
plt.title('PCL Class Histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
df['lab'].value_counts()


# In[8]:


#Check missing values
print("Before:",df.isna().sum(),'\n')

#Check missing text data and remove
print("Missing text row:", df[df['txt'].isna()],'\n')
df.drop(labels=8639,axis=0,inplace=True)
print("After deleting row 8638",'\n', df.isna().sum())


# # Check PCL Features

# In[54]:


pcl = df['txt'][df['lab']==1]
no_pcl = df['txt'][df['lab']==0]

#Look through individual entries
i = 18
pcl.iloc[3]


# In[10]:


#Check percent of paragraphs contain string
words = ['homeless','talented','disabled','home','color','money','donate']
def word_percent(word):
    percent_pcl = sum(pcl.str.count(word))/len(pcl)
    percent_no_pcl = sum(no_pcl.str.count(word))/len(no_pcl)
    print('% of pcl texts that include \'',word,'\':', percent_pcl)
    print('% of non-pcl texts that include \'',word,'\':', percent_no_pcl)
for word in words:
    word_percent(word)


# # Text Preprocessing

# In[11]:


#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text = text.strip()  
    text = re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text) 
    return text

 
# STOPWORD REMOVAL
def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)
#LEMMATIZATION
# Initialize the lemmatizer
wl = WordNetLemmatizer()
 
# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)


# In[12]:


#Final pre-processing and create clean text
def finalpreprocess(string):
    return lemmatizer(stopword(preprocess(string)))
df['clean_text'] = df['txt'].apply(lambda x: finalpreprocess(x))
df.head()


# In[67]:


#Check example of clean text
df['clean_text'][12]


# # Word Embedding

# In[13]:


#Word2Vec
# Word2Vec runs on tokenized sentences
clean_txt_tok= [nltk.word_tokenize(i) for i in df['clean_text']]  


# In[14]:


# create Word2vec model
#here words_f should be a list containing words from each document. say 1st row of the list is words from the 1st document/sentence
#length of words_f is number of documents/sentences in your dataset
df['clean_text_tok']=[nltk.word_tokenize(i) for i in df['clean_text']] #convert preprocessed sentence to tokenized sentence
model = Word2Vec(df['clean_text_tok'],min_count=1)  #min_count=1 means word should be present at least across all documents,
#if min_count=2 means if the word is present less than 2 times across all the documents then we shouldn't consider it


w2v = dict(zip(model.wv.index_to_key, model.wv.vectors))  #combination of word and its vector

#for converting sentence to vectors/numbers from word vectors result by Word2Vec
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# In[15]:


#Word2vec
# Fit and transform
modelw = MeanEmbeddingVectorizer(w2v)
txt_vectors_w2v = modelw.transform(df['clean_text_tok'])


# In[16]:


#Normalize data
txt_norm_w2v = StandardScaler().fit_transform(txt_vectors_w2v)


# In[17]:


#Split into 80% train set and 20% development set
txt_train_w2v, txt_test_w2v, lab_train_w2v, lab_test_w2v = train_test_split(txt_norm_w2v, df['lab'],test_size=0.2,random_state=7)

print("Number of paragraphs in txt_train dataset: ", txt_train_w2v.shape)
print("Number of labels in lab_train dataset: ", lab_train_w2v.shape)
print("Number of paragraphs in txt_test dataset: ", txt_test_w2v.shape)
print("Number of labels in lab_test dataset: ", lab_test_w2v.shape)


# In[18]:


#SMOTE to oversample from pcl data (label=1) and balance data
print("Before OverSampling, counts of label '1': {}".format(sum(lab_train_w2v==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(lab_train_w2v==0)))

sm = SMOTE(random_state=76)
txt_train_w2v_res, lab_train_w2v_res = sm.fit_resample(txt_train_w2v, lab_train_w2v.ravel())

print('After OverSampling, the shape of txt_train: {}'.format(txt_train_w2v_res.shape))
print('After OverSampling, the shape of lab_train: {} \n'.format(lab_train_w2v_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(lab_train_w2v_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(lab_train_w2v_res==0)))


# # Model 1: Logistic Regression (with word2vec)

# In[19]:


#5-fold Cross-validation for hyperparameter C for l2 penalty (Ridge regression)
parameters = {
    'C': np.linspace(1, 20, 20)
             }
lr = LogisticRegression(solver = 'liblinear', penalty = 'l2', max_iter=100)
clf_w2v = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf_w2v.fit(txt_train_w2v_res, lab_train_w2v_res.ravel())


# In[20]:


print('Best choice of C: \n',
      'Best parameter:',clf_w2v.best_params_, '\n',
      'Score: ',clf_w2v.best_score_)


# In[21]:


#FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
lr=LogisticRegression(solver = 'liblinear', C=clf_w2v.best_params_['C'], penalty = 'l2')
lr.fit(txt_train_w2v_res, lab_train_w2v_res)  #model
#Predict y value for test dataset
lab_predict = lr.predict(txt_test_w2v)
lab_prob = lr.predict_proba(txt_test_w2v)[:,1]
print(classification_report(lab_test_w2v,lab_predict))
print('Confusion Matrix:',confusion_matrix(lab_test_w2v, lab_predict))
 
fpr, tpr, thresholds = roc_curve(lab_test_w2v, lab_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)


# In[22]:


conf_mat = confusion_matrix(lab_test_w2v, lab_predict)
t_neg = conf_mat[0][0]
f_pos = conf_mat[0][1]
f_neg = conf_mat[1][0]
t_pos = conf_mat[1][1]
acc = (t_pos+t_neg)/len(lab_test_w2v)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)


# In[23]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[24]:


#Confusion Matrix for Train data
lab_train_pre = lr.predict(txt_train_w2v_res)

cnf_matrix_tra = confusion_matrix(lab_train_w2v_res, lab_train_pre)
t_neg = cnf_matrix_tra[0,0]
f_pos = cnf_matrix_tra[0,1]
f_neg = cnf_matrix_tra[1,0]
t_pos = cnf_matrix_tra[1,1]
acc = (t_pos+t_neg)/len(lab_train_w2v_res)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the train dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix for Train Data')
plt.show()


# In[25]:


#Confusion Matrix for Test Data
lab_test_pre = lr.predict(txt_test_w2v)

cnf_matrix_test = confusion_matrix(lab_test_w2v, lab_test_pre)
t_neg = cnf_matrix_test[0,0]
f_pos = cnf_matrix_test[0,1]
f_neg = cnf_matrix_test[1,0]
t_pos = cnf_matrix_test[1,1]
acc = (t_pos+t_neg)/len(lab_test_w2v)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the test dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_test , classes=class_names, title='Confusion matrix for Test Data')
plt.show()


# # TF-Idf (Bag of Words)

# In[26]:


#Create Tf-Idf data
#Tf-Idf
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
txt_vectors_tfidf = tfidf_vectorizer.fit_transform(df['clean_text']) 


# In[69]:


print(txt_vectors_tfidf.shape)
txt_vectors_tfidf[0]


# In[27]:


#Split into 80% train set and 20% development set
txt_train_tfidf, txt_test_tfidf, lab_train, lab_test = train_test_split(txt_vectors_tfidf,
                                                                        df['lab'],test_size=0.2,random_state=7)

print("Number of paragraphs in txt_train_tfidf dataset: ", txt_train_tfidf.shape)
print("Number of labels in lab_train dataset: ", lab_train.shape)
print("Number of paragraphs in txt_test_tfidf dataset: ", txt_test_tfidf.shape)
print("Number of labels in lab_test dataset: ", lab_test.shape)


# In[28]:


#SMOTE to oversample from pcl data (label=1) and balance data
print("Before OverSampling, counts of label '1': {}".format(sum(lab_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(lab_train==0)))

sm = SMOTE(random_state=8)
txt_train_tfidf_res, lab_train_res = sm.fit_resample(txt_train_tfidf, lab_train.ravel())

print('After OverSampling, the shape of txt_train: {}'.format(txt_train_tfidf_res.shape))
print('After OverSampling, the shape of lab_train: {} \n'.format(lab_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(lab_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(lab_train_res==0)))


# # Model 2: Logistic Regression with Tf-Idf

# In[72]:


#5-fold Cross-validation for hyperparameter C for l2 penalty (Ridge regression)
parameters = {
    'C': np.linspace(1, 30, 10)
             }
lr = LogisticRegression(solver = 'liblinear', penalty = 'l2', max_iter=100)
clf_tfidf = GridSearchCV(lr, parameters, cv=5, verbose=5, n_jobs=3)
clf_tfidf.fit(txt_train_tfidf_res, lab_train_res.ravel())


# In[73]:


print('Best choice of C: \n',
      'Best parameter:',clf_tfidf.best_params_, '\n',
      'Score: ',clf_tfidf.best_score_)


# In[74]:


#FITTING THE CLASSIFICATION MODEL using Logistic Regression (W2v)
lr=LogisticRegression(solver = 'liblinear', C=clf_tfidf.best_params_['C'], penalty = 'l2')
lr.fit(txt_train_tfidf_res, lab_train_res)  #model
#Predict y value for test dataset
lab_predict = lr.predict(txt_test_tfidf)
lab_prob = lr.predict_proba(txt_test_tfidf)[:,1]
print(classification_report(lab_test,lab_predict))
print('Confusion Matrix:',confusion_matrix(lab_test, lab_predict))
 
fpr, tpr, thresholds = roc_curve(lab_test, lab_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)


# In[75]:


#Confusion Matrix for Train data
lab_train_pre = lr.predict(txt_train_tfidf_res)

cnf_matrix_tra = confusion_matrix(lab_train_res, lab_train_pre)
t_neg = cnf_matrix_tra[0,0]
f_pos = cnf_matrix_tra[0,1]
f_neg = cnf_matrix_tra[1,0]
t_pos = cnf_matrix_tra[1,1]
acc = (t_pos+t_neg)/len(lab_train_res)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the train dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix for Train Data (Tf-Idf)')
plt.show()


# In[78]:


#Confusion Matrix for Test Data
lab_test_pre = lr.predict(txt_test_tfidf)

cnf_matrix_test = confusion_matrix(lab_test, lab_test_pre)
t_neg = cnf_matrix_test[0,0]
f_pos = cnf_matrix_test[0,1]
f_neg = cnf_matrix_test[1,0]
t_pos = cnf_matrix_test[1,1]
acc = (t_pos+t_neg)/len(lab_test)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the test dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_test , classes=class_names, title='Confusion matrix for Test Data (Tf-Idf)')
plt.show()


# # Model 3: Naive Bayes Classifier with Tf-Idf

# In[34]:


#FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
nb_tfidf = MultinomialNB()
nb_tfidf.fit(txt_train_tfidf_res, lab_train_res)  
#Predict label for test dataset
lab_predict = nb_tfidf.predict(txt_test_tfidf)
lab_prob = nb_tfidf.predict_proba(txt_test_tfidf)[:,1]
print(classification_report(lab_test,lab_predict))
print('Confusion Matrix:',confusion_matrix(lab_test, lab_predict))
 
fpr, tpr, thresholds = roc_curve(lab_test, lab_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)


# In[35]:


#Confusion Matrix for Train data
lab_train_pre = nb_tfidf.predict(txt_train_tfidf_res)

cnf_matrix_tra = confusion_matrix(lab_train_res, lab_train_pre)
t_neg = cnf_matrix_tra[0,0]
f_pos = cnf_matrix_tra[0,1]
f_neg = cnf_matrix_tra[1,0]
t_pos = cnf_matrix_tra[1,1]
acc = (t_pos+t_neg)/len(lab_train_res)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the train dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix for Train Data (Naive Bayes)')
plt.show()


# In[36]:


#Confusion Matrix for Test Data
lab_test_pre = nb_tfidf.predict(txt_test_tfidf)

cnf_matrix_test = confusion_matrix(lab_test, lab_test_pre)
t_neg = cnf_matrix_test[0,0]
f_pos = cnf_matrix_test[0,1]
f_neg = cnf_matrix_test[1,0]
t_pos = cnf_matrix_test[1,1]
acc = (t_pos+t_neg)/len(lab_test)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the test dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_test , classes=class_names, title='Confusion matrix for Test Data (Naive Bayes)')
plt.show()


# # Model 4: Neural Network

# In[37]:


#Sequential neural model
input_dim = txt_train_w2v_res.shape[1]  # Number of features

model = Sequential()
model.add(layers.Dense(7, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
model.summary()


# In[38]:


history = model.fit(txt_train_w2v_res, lab_train_w2v_res,
                     epochs=100,
                     verbose=False,
                     validation_data=(txt_test_w2v, lab_test_w2v),
                     batch_size=10)


# In[39]:


#Evaluate training accuracy and testing accuracy
loss, accuracy = model.evaluate(txt_train_w2v_res, lab_train_w2v_res, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(txt_test_w2v, lab_test_w2v, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))


# In[40]:


#Visualize loss and accuracy of training and testing data
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


# In[41]:


plot_history(history)


# In[42]:


clear_session()


# In[43]:


#Run 60 epochs
model.fit(txt_train_w2v_res, lab_train_w2v_res,
                     epochs=60,
                     verbose=False,
                     validation_data=(txt_test_w2v, lab_test_w2v),
                     batch_size=10)


# In[44]:


#Confusion Matrix for Train data
lab_train_prob = model.predict(txt_train_w2v_res)
lab_train_pre = [1 if x >= 0.5 else 0 for x in lab_train_prob]

cnf_matrix_tra = confusion_matrix(lab_train_w2v_res, lab_train_pre)
t_neg = cnf_matrix_tra[0,0]
f_pos = cnf_matrix_tra[0,1]
f_neg = cnf_matrix_tra[1,0]
t_pos = cnf_matrix_tra[1,1]
acc = (t_pos+t_neg)/len(lab_train_w2v_res)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the train dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix for Train Data (Neural Net)')
plt.show()


# In[45]:


#Confusion Matrix for Test data
lab_test_prob = model.predict(txt_test_w2v)
lab_test_pre = [1 if x >= 0.5 else 0 for x in lab_test_prob]

cnf_matrix_tra = confusion_matrix(lab_test_w2v, lab_test_pre)
t_neg = cnf_matrix_tra[0,0]
f_pos = cnf_matrix_tra[0,1]
f_neg = cnf_matrix_tra[1,0]
t_pos = cnf_matrix_tra[1,1]
acc = (t_pos+t_neg)/len(lab_test_w2v)
prec = t_pos/(f_pos+t_pos)
rec = t_pos/(t_pos+f_neg)
f1 = (2*prec*rec)/(prec+rec)

print("F1 metric in the test dataset: {}".format(f1))


class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix for Test Data (Neural Net)')
plt.show()

