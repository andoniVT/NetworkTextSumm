from gensim import corpora, models, similarities , matutils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from random import shuffle
from scipy import spatial
from utils import permutate_data

class Vectorization(object):

    #def __init__(self, corpus, vectorization_type, use_inference=None, vector_size=None, auxiliar_corpus=None):
    def __init__(self, corpus, vectorization_type, vector_size=None, auxiliar_corpus=None):
        self.corpus = corpus
        self.vectorization_type = vectorization_type
        #self.use_inference = use_inference
        self.vector_size = vector_size
        self.auxiliar_corpus = auxiliar_corpus

    def tf_idf_vectorization(self):
        obj = TfIdfModel(self.corpus, self.auxiliar_corpus)
        obj.train()
        return obj.get_matrix_tfidf()


    def d2v_vectorization(self):
        #obj = Doc2VecModel(self.corpus, self.use_inference, self.vector_size, self.auxiliar_corpus)
        obj = Doc2VecModel(self.corpus, self.vector_size, self.auxiliar_corpus)
        obj.train()
        return obj.get_matrix_doc2vec()

    def calculate(self):
        if self.vectorization_type == 'tfidf':
            return self.tf_idf_vectorization()
        else:
            return self.d2v_vectorization()

        #return ['dictionary', 'key: nombre del documento o cluster', 'value: matrix con los vectores de cada sentence del documento']


class TfIdfModel(object):

    def __init__(self, corpus, auxiliar=None):
        print "vectorization tfidf!!"
        self.corpus = corpus
        self.auxiliar = auxiliar
    '''
            for i in processed_corpus.items():
            valores =  i[1]
            psentences = valores[1]
            for j in psentences:
                print j[0] , j[1]
    '''

    def train(self):
        allSentences = []
        for i in self.corpus.items():
            sentence_values = i[1]
            psentences = sentence_values[1]
            for j in psentences:
                allSentences.append(j[0])


        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                sentence_values = i[1]
                psentences = sentence_values[1]
                for j in psentences:
                    allSentences.append(j[0])

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
                vec_bow = self.dictionary.doc2bow(j[0])  # modificar aquii,, preprocesed (content  , id=null)
                vec_tfidf = self.tfidf[vec_bow]
                doc_matrix.append(vec_tfidf)

            corpus_matrix[doc_name] = doc_matrix
        return corpus_matrix


class Doc2VecModel(object):

    #def __init__(self, corpus, inference, size, auxiliar):
    def __init__(self, corpus, size, auxiliar):
        print "vectorizacion doc2vec!!"
        self.corpus = corpus
        #self.inference = inference
        self.size = size
        self.auxiliar = auxiliar

    def train(self):
        allSentences = []
        for i in self.corpus.items():
            doc_name = i[0]
            sentences = i[1][1]     # modificar aquiii!!
            for index, sent in enumerate(sentences):
                sent_name = doc_name + "_" + str(index)
                allSentences.append((sent[0], sent_name))

        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                doc_name = i[0]
                sentences = i[1][1]     # modificar aqui?
                for index, sent in enumerate(sentences):
                    sent_name = doc_name + "_" + str(index)
                    allSentences.append((sent[0], sent_name))

        labeled_sentences = []
        #if self.inference:
        #    print "aun falta implementarrr!"
        #    print "posible error aqui!"

        for i in allSentences:
            sentence = i[0]
            label = i[1]
            labeled_sentences.append(LabeledSentence(sentence, [label]))

        #self.model = Doc2Vec(min_count=1, window=10, size=self.size, sample=1e-4, negative=5, workers=8)
        #self.model = Doc2Vec(min_count=1, window=10, size=self.size, sample=1e-4, workers=8)
        self.model = Doc2Vec(min_count=1, window=10, size=self.size, workers=8)  # okk
        self.model.build_vocab(labeled_sentences)
        print "training d2v ...."
        #for epoch in range(10):
        #    self.model.train(permutate_data(labeled_sentences), total_examples=self.model.corpus_count)
        self.model.train(labeled_sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
        #self.model.train(labeled_sentences, total_examples=self.model.corpus_count, epochs=10)




    def get_matrix_doc2vec(self):
        print "obtaining matrix"
        corpus_matrix = dict()
        for i in self.corpus.items():
            doc_name = i[0]
            size = len(i[1][1])
            doc_matrix = []
            for i in range(size):
                key = doc_name + "_" + str(i)
                vec_d2v = self.model.docvecs[key]
                doc_matrix.append(vec_d2v)
            corpus_matrix[doc_name] = doc_matrix

        return corpus_matrix


