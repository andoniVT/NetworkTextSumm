import os
from configuration import corpus_dir
import codecs
import re
from morphological_analysis import lemma
import string
import nltk
from nltk import word_tokenize , sent_tokenize
from gensim import corpora, models, similarities , matutils

documents = os.listdir(corpus_dir['temario_v1'])

document_path = corpus_dir['temario_v1'] + '/' +documents[0]

print document_path

document = codecs.open(document_path, encoding="utf-8")
content = ""
for i in document:
    i = i.rstrip()
    content+= i + " "

content = content.lower()

sentences = sent_tokenize(content, language='portuguese')

psentences = []
for sentence in sentences:
    for c in string.punctuation:
        sentence = sentence.replace(c, '')
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    sentence = " ".join(sentence.split())
    valores = sentence.split(' ')
    psentence = []
    for i in valores:
        psentence.append(lemma(i))
    psentences.append(psentence)


dictionary = corpora.Dictionary(psentences)
theCorpus = [dictionary.doc2bow(text) for text in psentences]
tfidf = models.TfidfModel(theCorpus)

print tfidf









'''
for c in string.punctuation:
    content = content.replace(c, "")
content = ''.join([i for i in content if not i.isdigit()])

content = " ".join(content.split())
valores = content.split(' ')
#for i in valores:
#    print i , lemma(i)

'''
#stopwords = nltk.corpus.stopwords.words('portuguese')


