import os
from configuration import corpus_dir , extras
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
from utils import read_document_english , write_data_to_disk , load_data_from_disk
from  nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


sentence = 'this is a foo bar sentences and i want to ngramize it'
n = 2
sixgrams = nltk.ngrams(sentence.split(), n)
print sixgrams
for grams in sixgrams:
  print grams



input_list = ['all', 'this', 'happened', 'more', 'or', 'less']


print find_ngrams(input_list, 1)
print find_ngrams(input_list, 2)
print find_ngrams(input_list, 3)
print find_ngrams(input_list, 4)
print ''

words = nltk.word_tokenize('hola yo me llamo jorge andoni valverde tohalino')
my_bigrams = nltk.bigrams(words)
my_trigrams = nltk.trigrams(words)
haber = nltk.ngrams(4 , words)


text = "this is a foo bar sentences and i want to ngramize it"
text2 = "hola yo me llamo jorge andoni valverde tohalino"
text3 = "hola yo me llamo"
vectorizer = CountVectorizer(ngram_range=(1,6))
analyzer = vectorizer.build_analyzer()
#for i in analyzer(text3):
#    print i











'''
def remover(text):
	for c in string.punctuation:
		text = text.replace(c, "")
	text = ''.join([i for i in text if not i.isdigit()])
	stopwords = nltk.corpus.stopwords.words('english')
	words = text.split()

	result = []

	for word in words:
		if not word in stopwords:
			result.append(word)

	tags = ['NN', 'NNP', 'NNPS', 'NNS']
	verbTags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	tagged = nltk.pos_tag(result)

	pSentence = []
	for i in tagged:
		if i[1] in verbTags:
			#pSentence.append(WordNetLemmatizer().lemmatize(i[0], 'v') + " ")
			pSentence.append(WordNetLemmatizer().lemmatize(i[0], 'v'))
		else:
			#pSentence.append(WordNetLemmatizer().lemmatize(i[0]) + " ")
			pSentence.append(WordNetLemmatizer().lemmatize(i[0]))


	return pSentence



print "duc2002 :)"
corpus_dictionary = dict()
path = corpus_dir['duc2002']

path2 = corpus_dir['duc2004']

clusters = os.listdir(path)
clusters2 = os.listdir(path2)

if '.DS_Store' in clusters:
	clusters.remove('.DS_Store')

if '.DS_Store' in clusters2:
	clusters2.remove('.DS_Store')



vocabulary = dict()

sentences = []

for i in clusters:
	sub_path = path + i + '/'
	documents = os.listdir(sub_path)
	for j in documents:
		document = sub_path + j
		document_sentences = read_document_english(document)
		for sentence in document_sentences:
			sentence = sentence.lower()
			sentence = remover(sentence)
			sentences.append(sentence)


for i in clusters2:
	sub_path = path2 + i + '/'
	documents = os.listdir(sub_path)
	for j in documents:
		document = sub_path + j
		document_sentences = read_document_english(document)
		for sentence in document_sentences:
			sentence = sentence.lower()
			sentence = remover(sentence)
			sentences.append(sentence)





for sentence in sentences:
	for word in sentence:
		if word in vocabulary:
			vocabulary[word]+=1
		else:
			vocabulary[word] = 1

if '' in vocabulary:
	del vocabulary['']



write_data_to_disk('vocabulary.pk' , vocabulary)

print len(vocabulary)
'''



'''
ratings = np.array([[1,4,5,8],
                    [2,5,3,4],
                    [4,10,20,1]], dtype=np.float)
a = np.array([1,4,5,8])
b = np.array([2,5,3,4])
c = np.array([4,10,20,1])
vector = [a]
#matrix = np.array([a , b , c])
#print matrix
#print ratings
movie_means = np.mean(vector, axis=0)
print movie_means
'''


'''
import random
from scipy import spatial
w2v_vocabulary = load_data_from_disk(extras['google_w2v'])
key = random.choice(w2v_vocabulary.keys())
key2 = random.choice(w2v_vocabulary.keys())

v1 = w2v_vocabulary[key]

v2 = np.repeat(999999999999999, 300)

v3 = w2v_vocabulary[key2]


print 1 - spatial.distance.cosine(v2, v3)
'''


'''
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



sentence = 'hola que tal?'

print sentence[:4]

'''


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


