import os
from configuration import corpus_dir
import codecs
import re
from morphological_analysis import lemma
import string
import nltk
from nltk import word_tokenize , sent_tokenize
from gensim import corpora, models, similarities , matutils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import unicodedata

import igraph
from igraph import *

def tfidf():
	s1 = ['brazil', 'be', 'large', 'country', 'south', 'america']
	s2 = ['be', 'world', 'five', 'large', 'country', 'area', 'population']
	s3 = ['be', 'large', 'country', 'have', 'portuguese', 'official', 'language', 'one', 'america']
	s4 = ['bound', 'atlantic', 'ocean', 'east', 'brazil', 'have', 'costline', 'kilometer']
	s5 = ['border', 'south', 'american', 'country', 'ecuador', 'chile']
	s6 = ['brazil', 'economy', 'be', 'world', 'nine', 'large', 'nominal', 'gdp']

	allS = [s1,s2,s3,s4,s5,s6]


	dictionary = corpora.Dictionary(allS)
	theCorpus = [dictionary.doc2bow(text) for text in allS]
	c_tfidf = models.TfidfModel(theCorpus)
	corpus_tfidf = c_tfidf[theCorpus]

	v1 = corpus_tfidf[0]
	v2 = corpus_tfidf[1]
	v3 = corpus_tfidf[2]
	v4 = corpus_tfidf[3]
	v5 = corpus_tfidf[4]
	v6 = corpus_tfidf[5]

	completo = Graph.Full(len(allS))
	all_edges  = completo.get_edgelist()

	network = Graph()
	network.add_vertices(len(allS))

	edges = []
	weights = []

	for i in all_edges:
		in1 = i[0]
		in2 = i[1]
		sim = round(matutils.cossim(corpus_tfidf[in1], corpus_tfidf[in2]), 2)
		if sim > 0:
			edges.append((in1,in2))
			weights.append(sim)

	network.add_edges(edges)
	network.es['weight'] = weights
	#network.vs["color"] = ['#1AA1E4','cyan','magenta','#F11533','#52DF4B','#FC8405']
	network.vs["color"] = ['#8272C8','#8272C8','#8272C8','#8272C8','#8272C8','#8272C8']


	for index, edge in enumerate(edges):
		print (edge[0]+1 , edge[1]+1) ,  weights[index]

	return network




def draw_graph(graph):
	size = graph.vcount()
	ids =  [x+1 for x in range(size)]


	layout = graph.layout("kk")

	visual_style = {}
	visual_style["vertex_label"] = ids
	visual_style["vertex_size"] = 35
	visual_style["layout"] = layout
	visual_style["bbox"] = (800,600)
	visual_style["margin"] = 80
	plot(graph, **visual_style)


red = tfidf()
draw_graph(red)




'''
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


for i in psentences:
    for word in i:
        haber = word
        word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
        print (haber ,word) ,
    print ""



file = codecs.open('PRUEBA.txt',  'w', 'utf-8')
for i in sentences:
    file.write(i + '\n')



#file = open(location, 'w')
#    for i in summary_sentences:
#        file.write(i + "\n")



dictionary = corpora.Dictionary(psentences)
theCorpus = [dictionary.doc2bow(text) for text in psentences]
tfidf = models.TfidfModel(theCorpus)

#print tfidf



allSentences = []
for i , sent in enumerate(psentences):
    sent_name = str(i)
    allSentences.append((sent, sent_name))


labeled_sentences = []
for i in allSentences:
    sentence = i[0]
    label = i[1]
    labeled_sentences.append(LabeledSentence(sentence, [label]))


model = Doc2Vec(min_count=1, window=10, size=len(allSentences), sample=1e-4, negative=5, workers=8)
model.build_vocab(labeled_sentences)
print "training d2v ...."
model.train(labeled_sentences, total_examples=model.corpus_count, epochs=model.iter)





for c in string.punctuation:
    content = content.replace(c, "")
content = ''.join([i for i in content if not i.isdigit()])

content = " ".join(content.split())
valores = content.split(' ')
#for i in valores:
#    print i , lemma(i)

'''
#stopwords = nltk.corpus.stopwords.words('portuguese')


