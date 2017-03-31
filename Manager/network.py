import igraph
from igraph import *
from utils import has_common_elements
class NetworkManager(object):

    def __init__(self, network_type, network_sub_type, corpus, vector_representation, distance, inter_edge, intra_edge, limiar_value):
        self.network_type = network_type
        self.network_sub_type = network_sub_type
        self.corpus = corpus
        self.vector_representation = vector_representation
        self.distance = distance
        self.inter_edge = inter_edge
        self.intra_edge = intra_edge
        self.limiar_value = limiar_value

    def create_networks(self):
        corpus_networks = dict()

        for i in self.corpus.items():
            doc_name = i[0]
            doc_sentences = i[1][1]
            doc_vector = None
            if self.vector_representation is not None:
                doc_vector = self.vector_representation[doc_name]
            document_data = [doc_sentences, doc_vector]

            obj = CNetwork(self.network_type, self.network_sub_type, document_data, self.distance,
                           self.inter_edge, self.intra_edge, self.limiar_value)

            networkData = obj.generate()

            corpus_networks[doc_name] = networkData

        return corpus_networks


class CNetwork(object):

    def __init__(self, network_type, network_sub_type, document_data, distance, inter_edge, intra_edge, limiar_value):
        self.network_type = network_type
        self.network_sub_type = network_sub_type
        self.document_data = document_data
        self.distance = distance
        self.inter_edge = inter_edge
        self.intra_edge = intra_edge
        self.limiar_value = limiar_value

    def noun_based_network(self):
        print "creando red de sustantivos"
        network_size = len(self.document_data[0])
        document_sentences = self.document_data[0]

        only_auxiliar = Graph.Full(network_size)
        all_edges  = only_auxiliar.get_edgelist()

        network = Graph()
        network.add_vertices(network_size)
        network_edges =[]
        weight_list = []
        for i in all_edges:
            index1 = i[0]
            index2 = i[1]
            common_elements = has_common_elements(document_sentences[index1] , document_sentences[index2])
            if common_elements>0:
                network_edges.append((index1,index2))
                weight_list.append(common_elements)

        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        return [network, None] #None es el valor de treshold para MDS, para NOUns debe calcularse en la misma etapa de generacion



    def tfidf_d2v_based_network(self):
        print "creando red de vectorres tfidf o doc2vec"
        network_size = len(self.document_data[0])
        document_sentences = self.document_data[0]

        only_auxiliar = Graph.Full(network_size)
        all_edges = only_auxiliar.get_edgelist()
        network = Graph()
        network.add_vertices(network_size)
        network_edges = []
        weight_list = []
        print self.network_type  # tf o doc 2 vecc para aplicar la medida correcta del coseno
        for i in all_edges:
            index1 = i[0]
            index2 = i[1]



        return ['d2vs-tfidfs' , 'threshold']

    def multilayer_based_network(self):
        print "creando red MLN !"
        return ['mln']


    def generate(self):
        if self.network_type == 'noun':
            return self.noun_based_network()
        if self.network_type == 'tfidf' or self.network_type == 'd2v':
            return self.tfidf_d2v_based_network()
        if self.network_type == 'mln':
            return self.multilayer_based_network()

        return ["dictionary", "key: nombre del documento o cluster",
                'value: red del documento con los limiares aplicados',
                'devolver tambien el threshold para antiredundancia MDS']






