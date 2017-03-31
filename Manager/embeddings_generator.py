from gensim import corpora, models, similarities , matutils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from random import shuffle
from scipy import spatial

class Vectorization(object):

    def __init__(self, corpus, vectorization_type, use_inference=None, vector_size=None):
        self.corpus = corpus
        self.vectorization_type = vectorization_type
        self.use_inference = use_inference
        self.vector_size = vector_size

    def tf_idf_vectorization(self):
        obj = TfIdfModel(self.corpus)
        obj.train()
        return obj.get_matrix_tfidf()


    def d2v_vectorization(self):
        obj = Doc2VecModel(self.corpus, self.use_inference, self.vector_size)

    def calculate(self):
        if self.vectorization_type == 'tfidf':
            return self.tf_idf_vectorization()
        else:
            self.d2v_vectorization()
        return ['dictionary', 'key: nombre del documento o cluster', 'value: matrix con los vectores de cada sentence del documento']


class TfIdfModel(object):

    def __init__(self, corpus):
        print "vectorization tfidf!!"
        self.corpus = corpus

    def train(self):
        allSentences = []
        for i in self.corpus.items():
            allSentences.extend(i[1][1])
        self.dictionary = corpora.Dictionary(allSentences)
        theCorpus = [self.dictionary.doc2bow(text) for text in allSentences]
        self.tfidf = models.TfidfModel(theCorpus)

    def get_matrix_tfidf(self):
        corpus_matrix = dict()
        for i in self.corpus.items():
            doc_name = i[0]
            doc_sentences = i[1][1]
            doc_matrix = []
            for j in doc_sentences:
                vec_bow = self.dictionary.doc2bow(j)
                vec_tfidf = self.tfidf[vec_bow]
                doc_matrix.append(vec_tfidf)

            corpus_matrix[doc_name] = doc_matrix
        return corpus_matrix


class Doc2VecModel(object):

    def __init__(self, corpus, inference, size):
        print "vectorizacion doc2vec!!"
        self.corpus = corpus
        self.inference = inference
        self.size = size

    def train(self):
        print ""

    def get_matrix_doc2vec(self):
        print ""
