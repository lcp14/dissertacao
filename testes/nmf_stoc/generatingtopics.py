import pandas as pd
import numpy as np
import unicodedata
import statistics
import operator

from fuzzywuzzy import fuzz 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


from sklearn.decomposition import NMF

from SToC import *

import os

from IPython.display import HTML, display
from tabulate import tabulate
from metrics import Evaluation as Eval

from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys
from nltk.stem.snowball import PorterStemmer


def print_top_words_display(H, feature_names, n_top_words):
    table_output = [['Topic','topwords']]
    topics = []
    for topic_idx, topic in enumerate(H):
        top_index = ["Topic # %d" % topic_idx]
        top_words = [" ' ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])]
        values = [value for value in np.sort(topic)[:-n_top_words - 1:-1]]
        table_output.append(top_index + top_words)
        topics.append(" ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
    df_out = pd.DataFrame(table_output, index=None)
    #display(HTML(tabulate(table_output, tablefmt='html')))
    return topics,df_out
def execute_tfidf(dataset,max_df=1,min_df=1,ngram=(0,2),stop_words=[]):
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,ngram_range=ngram,stop_words=stop_words)
    tfidf = tfidf_vectorizer.fit_transform(dataset)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    return tfidf,tfidf_feature_names
def execute_nmf(tfidf,n_topics = 10,n_components = 5):
    n_topics = n_topics
    n_components = n_components
    nmf = NMF(n_components=n_components,max_iter=400)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_.transpose() 
    return nmf,W,H
def execute_stoc(W,H,n_final,n_components):
    n_words = H.shape[0]

    topXtop_norm = getIrredutibleMatrix(W,H,n_components)
    assign, t2 = joinTopics(n_components, topXtop_norm)


    tops = print_join(t2, H.transpose(), W, n_components, None, n_words)

    limiar = print_estats(t2)

    W_new, H_new = see_join(t2, W, H, limiar,n_final)
    return W_new,H_new
def area_parents(area,areas_id,areas_parents,similar_area):
    li_id = getKeysByValue(areas_id,area)
    parents = [areas_parents[i] for i in li_id]
    for li in li_id:
        if li not in similar_area.keys():
            similar_area[li] = 1
        else:
            similar_area[li] += 1
    for parent in parents:
        for p in parent:
            if p not in similar_area.keys():
                similar_area[p] =1
            else:
                similar_area[p] +=1
                
        #print(area+"-->"+"-->".join([areas_id[i] for i in parent][::-1]))
    return similar_area
def topic_acm_area(topic,areas_id,areas_parents):
    similar_area = {}
    print(topic)
    m = 0
    for area in [areas_id[j] for j in y]:
        a = fuzz.partial_token_set_ratio(area,topic)
        if a > m:
            m = a
            s = area
        
             #area_parents(area,areas_id,areas_parents,similar_area)
    print(s)
    return similar_area
def getKeysByValue(dictOfElements, valueToFind):

    listOfKeys = list()
    listOfItems = dictOfElements.items()
    for item  in listOfItems:
        if item[1] == valueToFind:
            listOfKeys.append(item[0])
    return  listOfKeys
def areas_level(hierarchy):
    hierachy_acm = []
    for parents in areas_parents:
        if len(areas_parents[parents]) is hierarchy:
            hierachy_acm.append(areas_id[parents])
    return hierachy_acm
def stemming(words):
    new_words = []
    for word in words:
        new_words.append(stemmer.stem(word))
    return new_words
with open('../../datasets/'+ 'acm_hierachy' + '.pkl', 'rb') as f:
    acm = pickle.load(f)
with open('../../datasets/'+ 'acm_areas' + '.pkl', 'rb') as f:
    areas = pickle.load(f)
with open('../../datasets/'+ 'acm_areas_id' + '.pkl', 'rb') as f:
    areas_id = pickle.load(f)
with open('../../datasets/'+ 'acm_areas_parents' + '.pkl', 'rb') as f:
    areas_parents = pickle.load(f)
ano = int(sys.argv[1])
anos = range(2008,2019)
timeslice = [1168, 1246, 1509, 1748, 2004, 2204, 2275, 2485, 2597, 2374, 638]
timeslice_stem = [1170, 1244, 1510, 1751, 2008, 2204, 2275, 2484, 2590, 2372, 638]
idx = anos.index(ano)
inicio = sum(timeslice[:idx])
fim = inicio + timeslice[idx]

stem = sys.argv[2]
stemmer = PorterStemmer()

if stem == 'no':
    inicio = sum(timeslice[:idx])
    fim = inicio + timeslice[idx]
    artigos = (open('/home/leandror/Documents/dissertacao/new_corpus.txt','r').read().split('\n')[inicio:fim])

if stem == 'yes':
    inicio = sum(timeslice_stem[:idx])
    fim = inicio + timeslice_stem[idx]
    artigos = (open('/home/leandror/Documents/dissertacao/new_corpus_stem.txt','r').read().split('\n')[inicio:fim])    


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(artigos)
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
indices = np.arange(len(words_freq))

word = []
frequency = []
for i in range(len(words_freq)):
    word.append(words_freq[i][0])
    frequency.append(words_freq[i][1])

f = open('../acm_words.txt','r')

acm_words = f.read().replace('\n',' ').split()
if stem == 'yes':
    acm_words = stemming(acm_words)
stop_words_without_acm_words = list(set(word) - set(acm_words))

"""Parametros"""
# TF-IDF
# Quando númerico é a quantidade de documentos explicita!!
max_df = 0.85 # Só palavras que aparecem no máximo em 85% dos documentos
min_df = 0.01 # Só palavras que aparecem no minimo em 1% dos documentos
ngram= (1,1)
#NMF
#n_components = 15 # Numero de tópicos
n_topics = 10 # Quantidade de palavras para definir o tópico
for n_components in [10,15,20]:
    tfidf,tfidf_feature_names = execute_tfidf(artigos,max_df=max_df,min_df=min_df,ngram=ngram,stop_words = stop_words_without_acm_words)

    nmf,W,H = execute_nmf(n_components=n_components,n_topics=n_topics,tfidf=tfidf)

    topics_nmf,a = print_top_words_display(H.transpose(), tfidf_feature_names,n_topics)

    if stem == 'no':
        model = Word2Vec.load("/home/leandror/Documents/dissertacao/models/model_default.model")
    if stem == 'yes':
        model = Word2Vec.load("/home/leandror/Documents/dissertacao/models/model_default_stem.model")
    word_vectors = model.wv
    vectors = model.wv.vectors
    vector_size = model.wv.vector_size
    index2word = model.wv.index2word

    hierachy = int(sys.argv[3])
    hierachy_acm = areas_level(hierachy)
    how_many_levels = int(sys.argv[4])
    levels = [i+1 for i in range(hierachy,how_many_levels+hierachy)]

    acm_vectors = {}
    for acm_first in hierachy_acm:
        idx = getKeysByValue(areas_id,acm_first)
        y =[]
        for i in areas_parents:
            value = areas_parents[i]
            if idx[0] in value:
                if len(value) in levels: #hierarquia
                    y.append(i)            
        acmv = np.zeros(150)
        for idx1,word in enumerate(set(" ".join([areas_id[j] for j in y]).split())):
            word = stemmer.stem(word)
            if word in index2word:
                word_idx = index2word.index(word)
                acmv += vectors[word_idx]
            # else:
            #     print("{} não existe no vocabulario".format(word))
        acm_vectors[acm_first] = acmv

    table = [['id','topic','area']]
    for idx_topic,topic in enumerate(topics_nmf):
        tv = np.zeros(150)
        for idx,word in enumerate(topic.split()):
            if word in index2word:
                word_idx = index2word.index(word)
                tv += vectors[word_idx] 
                maxi = 0
                area = ''
                for acm in acm_vectors:
                    value = cosine_similarity(tv.reshape(1,150),acm_vectors[acm].reshape(1,150))
                    if value > maxi:
                        maxi = value
                        area = acm
        table.append([idx_topic,topic,area])
    if stem == 'yes':
        with open('stem/tabela_t'+str(n_components)+'.pickle', 'wb') as handle:
            pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if stem == 'no':
        with open('no_stem/tabela_t'+str(n_components)+'.pickle', 'wb') as handle:
            pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)    
    #display(HTML(tabulate(table, tablefmt='html')))