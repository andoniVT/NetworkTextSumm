import igraph
from igraph import *
from utils import has_common_elements, cosineSimilarity, calculate_similarity, reverseSortList, sortList, average

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
        cosine_sim_list = []
        for i in all_edges:
            index1 = i[0]
            index2 = i[1]
            common_elements = has_common_elements(document_sentences[index1] , document_sentences[index2])
            if common_elements>0:
                network_edges.append((index1,index2))
                weight_list.append(common_elements)
                cosine = cosineSimilarity(document_sentences[index1], document_sentences[index2])
                cosine_sim_list.append(cosine)

        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        threshold = (max(cosine_sim_list) + min(cosine_sim_list))/2
        return [network, threshold] #None es el valor de treshold para MDS, para NOUns debe calcularse en la misma etapa de generacion



    def tfidf_d2v_based_network(self):
        print "creando red de vectorres tfidf o doc2vec"
        network_size = len(self.document_data[0])
        #document_sentences = self.document_data[0]
        document_vectors = self.document_data[1]

        only_auxiliar = Graph.Full(network_size)
        all_edges = only_auxiliar.get_edgelist()
        network = Graph()
        network.add_vertices(network_size)
        network_edges = []
        weight_list = []
        #print self.limiar_value #aun sin usar

        for i in all_edges:
            index1 = i[0]
            index2 = i[1]
            similarity = calculate_similarity(document_vectors[index1] , document_vectors[index2], self.network_type, self.distance)
            if similarity>0:
                network_edges.append((index1, index2))
                weight_list.append(similarity)

        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        threshold = (max(weight_list)+min(weight_list))/2

        if self.network_type=='d2v':
            network = self.remove_redundant_edges(network)

        print len(network_edges) , len(network.get_edgelist())
        return [network , threshold]

    def remove_redundant_edges(self, network):
        edgesList = network.get_edgelist()
        weight_list = network.es['weight']
        max_weight = max(weight_list)
        min_weight = min(weight_list)

        average = (max_weight + min_weight) / 2
        average2 = (max_weight + average) / 2
        average3 = (max_weight + average2) / 2
        average4 = (max_weight + average3) / 2
        limiar=-1
        if self.limiar_value==0:
            limiar = average
        elif self.limiar_value==1:
            limiar=average2
        elif self.limiar_value==2:
            limiar=average3
        elif self.limiar_value==3:
            limiar=average4


        new_weight_list = []
        for i , edge in enumerate(edgesList):
            weight = weight_list[i]
            if weight <= limiar:
                network.delete_edges([(edge[0], edge[1])])
            else:
                new_weight_list.append(weight)

        network.es['weight'] = new_weight_list
        return network


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




class CNMeasures(object):

    def __init__(self, network):
        self.network = network

    def degree(self, paremeters=None):
        print "measuring degree"
        graph_degree = self.network.degree()
        graph_stg = self.network.strength(weights=self.network.es['weight'])
        ranked_by_degree = reverseSortList(graph_degree)
        ranked_by_stg = reverseSortList(graph_stg)
        print ranked_by_degree
        print ranked_by_stg
        return [ranked_by_degree, ranked_by_stg]

    def shortest_path(self, paremeters=None):
        print "measuring sp" # falta basada en pesos, hay que modificar
        measure = []
        network_size = self.network.vcount()
        for i in range(network_size):
            lenghts = self.network.shortest_paths(i)[0]
            sp = average(lenghts)
            measure.append(sp)
        ranked_by_sp = sortList(measure)
        print ranked_by_sp


    def page_rank(self, paremeters=None):
        print "measuring pr"

    def betweenness(self, paremeters=None):
        print "measuring btw"

    def clustering_coefficient(self, paremeters=None):
        print "measuring cc"

    def absortion_time(self, paremeters=None):
        print "measuring at"

    '''
    type = backbone / merged
    order = greater / less
    '''
    #def symmetry(self, type, order, h):
    def symmetry(self, parameters):
        print "measuring symetry"
        if len(parameters)!=0:
            order = parameters[0]
            type = parameters[1]
            h = parameters[2]
            print "type: " , type
            print "order: " , order
            print "h:" , h
        else:
            print "todas las simetrias"

    def accessibility(self, h):
        print "measuring accesibility"
        print "h:" , h

    def generalized_accessibility(self):
        print "measuring generalized accesibility"

    #def concentrics(self, type, h):
    def concentrics(self, parameters):
        print "measuring concentrics"
        if len(parameters)!=0:
            type = parameters[0]
            h = parameters[1]
            print "type", type
            print "h", h
        else:
            print "todas las concentricas con todas las h, o solo un subconjunto de las mejores"


    def all_measures(self, parameters=None):
        print "measuring all"
        '''
        [self.degree, self.shortest_path, self.page_rank, self.betweenness, self.clustering_coefficient,
         self.concentrics, self.symmetry, self.accessibility, self.generalized_accessibility,
         self.absortion_time]
         '''

    def traditional_measures(self, parameters=None):
        print "measuring traditional measures"
        '''
        [self.degree, self.shortest_path, self.page_rank, self.betweenness, self.clustering_coefficient]
        '''



    def manage_measures(self):
        dictionary = dict()
        dictionary['dg'] = self.degree
        dictionary['sp'] = self.shortest_path
        dictionary['pr'] = self.page_rank
        dictionary['btw'] = self.betweenness
        dictionary['cc'] = self.clustering_coefficient
        dictionary['at'] = self.absortion_time
        dictionary['gaccs'] = self.generalized_accessibility
        dictionary['sym'] = self.symmetry
        dictionary['accs'] = self.accessibility
        dictionary['ccts'] = self.concentrics # con parametrossss

        dictionary['trad'] = self.traditional_measures
        dictionary['*'] = self.all_measures


        return dictionary



class NodeManager(object):

    def __init__(self, networks, measures):
        self.networks = networks
        self.measures = measures

    def ranking(self):
        actual_network = self.networks['op94ag07-a'][0]
        #print actual_network
        obj = CNMeasures(actual_network)
        dictionary = obj.manage_measures()
        #print self.measures
        #print dictionary

        for i in self.measures:
            measure_parameter =  i.split('_')
            measure = measure_parameter[0]
            parameters = measure_parameter[1:]

            dictionary[measure](parameters)



        '''
        *

        '''


