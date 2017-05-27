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


file = codecs.open('PRUEBA.txt',  'w', 'utf-8')
for i in sentences:
    file.write(i + '\n')



#file = open(location, 'w')
#    for i in summary_sentences:
#        file.write(i + "\n")


'''
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



'''





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


