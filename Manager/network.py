import igraph
from igraph import *
from utils import has_common_elements, cosineSimilarity, calculate_similarity, reverseSortList, sortList, average
from utils import inverse_weights , find_term, sort_network, draw_graph , get_weights, vector_normalize, assign_mln_weight
import utils
import absorption
import hierarchical

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
            #print "problem:" , doc_name
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
        #print "creando red de sustantivos"
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
            #common_elements = has_common_elements(document_sentences[index1] , document_sentences[index2])
            common_elements = has_common_elements(document_sentences[index1][0], document_sentences[index2][0])
            if common_elements>0:
                network_edges.append((index1,index2))
                weight_list.append(common_elements)    # MLN     -------
                #cosine = cosineSimilarity(document_sentences[index1], document_sentences[index2])
                cosine = cosineSimilarity(document_sentences[index1][0], document_sentences[index2][0])
                cosine_sim_list.append(cosine)

        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        #print network.es['weight']
        #print cosine_sim_list  ###### PROBLEMAS PARA INGLES sds
        threshold = (max(cosine_sim_list) + min(cosine_sim_list))/2  #PROBLMAS PARA INGLES sds
        #print threshold ####################
        #threshold = 0
        diameter = network.diameter()
        #print diameter
        #draw_graph(network)
        #if diameter == 6:
        #    draw_graph(network)

        return [network, threshold] #None es el valor de treshold para MDS, para NOUns debe calcularse en la misma etapa de generacion



    def tfidf_d2v_based_network(self):
        #print "creando red de vectorres tfidf o doc2vec"
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

            #belong_same_document=  document_sentences[index1][1] == document_sentences[index2][1]   #True -> son del mismo documento  False->son de distintos documentos
            #print self.inter_edge , self.intra_edge , belong_same_document



            if similarity>0:
                network_edges.append((index1, index2))
                weight_list.append(similarity)

        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        threshold = (max(weight_list)+min(weight_list))/2

        if self.network_type=='d2v':
            #network = self.remove_redundant_edges(network)
            if self.limiar_value == 'knn':
                network = self.generate_knn_network(network)
            else:
                network = self.remove_redundant_edges_2(network)

        #print len(all_edges) ,  len(network_edges) , len(network.get_edgelist())
        #print len(all_edges) , len(network_edges), len(network.get_edgelist())
        #draw_graph(network)
        #print network.get_edgelist()
        #print network.es['weight']
        #diameter = network.diameter()
        #print diameter
        #draw_graph(network)
        #if diameter == 2 or diameter==1:
        #   draw_graph(network)

        return [network , threshold]

    def remove_redundant_edges(self, network):
        edgesList = network.get_edgelist()
        weight_list = network.es['weight']
        max_weight = max(weight_list)
        min_weight = min(weight_list)


        average = (max_weight + min_weight) / 2
        min_average = (average + min_weight)/2

        average2 = (max_weight + average) / 2
        average3 = (max_weight + average2) / 2
        average4 = (max_weight + average3) / 2
        average5 = (max_weight + average4) / 2
        average6 = (max_weight + average5) / 2
        limiar=-1
        if self.limiar_value==0:
            limiar = average
        elif self.limiar_value==1:
            limiar=average2
        elif self.limiar_value==2:
            limiar=average3
        elif self.limiar_value==3:
            limiar=average4
        elif self.limiar_value==4:
            limiar=average5
        elif self.limiar_value==5:
            limiar=average6
        elif self.limiar_value==-1:
            limiar = min_average

        new_weight_list = []
        for i , edge in enumerate(edgesList):
            weight = weight_list[i]
            if weight <= limiar:
                network.delete_edges([(edge[0], edge[1])])
            else:
                new_weight_list.append(weight)

        network.es['weight'] = new_weight_list
        return network


    def remove_redundant_edges_2(self, network):
        #print 'removing'
        #self.limiar_value=0.30 - 0.50 - 0.7
        network_size = network.vcount()
        edgesList = network.get_edgelist()
        weight_list = network.es['weight']

        limiar_per = self.limiar_value
        x = (len(edgesList) * limiar_per)
        new_size = int(len(edgesList) - x)
        sorted_values = sort_network(edgesList, weight_list)

        new_weights = []
        new_edges = []
        for i in range(new_size):
            values = sorted_values[i]
            edge = values[0].split('-')
            edge_pair = (int(edge[0]), int(edge[1]))
            new_edges.append(edge_pair)
            weight = values[1]
            new_weights.append(weight)

        new_network = Graph()
        new_network.add_vertices(network_size)
        new_network.add_edges(new_edges)
        new_network.es['weight'] = new_weights
        return new_network

    def generate_knn_network(self, network):
        k = 15
        print "knn red"
        network_size = network.vcount()
        edgesList = network.get_edgelist()
        weight_list = network.es['weight']
        dict_weights = get_weights(edgesList, weight_list)

        new_network = Graph()
        new_network.add_vertices(network_size)

        k_edges = []
        for i in range(network_size):
            edges_to_analize = dict()
            vertex = i
            vecinos = network.neighbors(vertex)
            #print vertex , vecinos
            for j in vecinos:
                if vertex < j:
                    key = str(vertex) + '-' + str(j)
                else:
                    key = str(j) + '-' + str(vertex)

                weight = dict_weights[key]
                edges_to_analize[key] = weight
            edges_to_analize_sorted = sorted(edges_to_analize.items(), key=operator.itemgetter(1), reverse=True)
            #edges_to_analize_sorted = sorted(edges_to_analize.items(), key=operator.itemgetter(1))
            number_vecinos = len(vecinos)
            index_remove = number_vecinos - k
            #print number_vecinos, k, index_remove
            k_best = edges_to_analize_sorted[0:k]
            removed = edges_to_analize_sorted[k:]
            #print k_best


            for j in k_best:
                key = j[0]
                aresta = key.split('-')
                aresta_i = int(aresta[0])
                aresta_f = int(aresta[1])
                edge_pair = (aresta_i, aresta_f)
                k_edges.append(edge_pair)

        #print k_edges
        new_network.add_edges(k_edges)


        #print len(edgesList) , len(new_network.get_edgelist())
        new_edge_list =  new_network.get_edgelist()
        k_weights = []

        for i in new_edge_list:
            key = str(i[0]) + '-' + str(i[1])
            weight = dict_weights[key]
            k_weights.append(weight)


        new_network.es['weight'] = k_weights
        #draw_graph(new_network)
        return new_network


    def multilayer_based_network(self):
        print "creando red MLN !"
        '''
        noun
        tfidf
        d2v
        '''
        if self.network_sub_type  == 'noun':
            return self.multilayer_noun_based_network()
        elif self.network_sub_type == 'tfidf' or self.network_sub_type == 'd2v':
            return self.multilayer_tfidf_d2v_based_network()
        #elif self.network_sub_type == 'd2v':
        #    print "-"

        return ['mln']

    def multilayer_noun_based_network(self):
        print 'MLN-Noun'
        network_size = len(self.document_data[0])
        document_sentences = self.document_data[0]
        only_auxiliar = Graph.Full(network_size)
        all_edges = only_auxiliar.get_edgelist()
        network = Graph()
        network.add_vertices(network_size)
        network_edges = []
        weight_list = []
        auxiliar_list = []
        flag_list = []
        cosines = []

        for i in all_edges:
            index1 = i[0]
            index2 = i[1]
            #common_elements = has_common_elements(document_sentences[index1][0], document_sentences[index2][0])
            similarity = cosineSimilarity(document_sentences[index1][0], document_sentences[index2][0])
            belong_same_document = document_sentences[index1][1] == document_sentences[index2][1]
            if similarity > 0:
                network_edges.append((index1, index2))
                auxiliar_list.append(similarity)

                if belong_same_document:
                    weight_list.append(similarity*self.intra_edge)
                else:
                    weight_list.append(similarity*self.inter_edge)
        '''
            belong_same_document = document_sentences[index1][1] == document_sentences[index2][1]
            similarity = cosineSimilarity(document_sentences[index1][0], document_sentences[index2][0])
            if common_elements > 0:
                network_edges.append((index1, index2))
                auxiliar_list.append(common_elements)
                cosines.append(similarity)
                if belong_same_document:
                    flag_list.append(True)
                else:
                    flag_list.append(False)

        normalized = vector_normalize(auxiliar_list)
        weight_list = assign_mln_weight(normalized, flag_list, self.inter_edge, self.intra_edge)
        '''


        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        threshold = (max(auxiliar_list) + min(auxiliar_list)) / 2
        #threshold = (max(cosines) + min(cosines)) / 2
        #print threshold

        auxiliar_network = self.remove_edges_for_mln(network, 0.4)
        #return [network, threshold]
        return [(network, auxiliar_network), threshold]







    def multilayer_tfidf_d2v_based_network(self):
        if self.network_sub_type=='tfidf':
            print 'MLN-TfIdf'
        else:
            print 'MLN-Doc2vec'

        network_size = len(self.document_data[0])
        document_sentences = self.document_data[0]
        document_vectors = self.document_data[1]

        only_auxiliar = Graph.Full(network_size)
        all_edges = only_auxiliar.get_edgelist()
        network = Graph()
        network.add_vertices(network_size)
        network_edges = []
        weight_list = []
        auxiliar_list = []

        for i in all_edges:
            index1 = i[0]
            index2 = i[1]
            similarity = calculate_similarity(document_vectors[index1], document_vectors[index2], self.network_sub_type, self.distance)
            belong_same_document=  document_sentences[index1][1] == document_sentences[index2][1]   #True -> son del mismo documento  False->son de distintos documentos
            #print  belong_same_document


            if similarity > 0:
                network_edges.append((index1, index2))
                auxiliar_list.append(similarity)

                if belong_same_document:
                    weight_list.append(similarity*self.intra_edge)
                else:
                    weight_list.append(similarity*self.inter_edge)


        network.add_edges(network_edges)
        network.es['weight'] = weight_list
        #print 'Intra-edge:' , self.intra_edge
        #print 'Inter-edge' , self.inter_edge

        #threshold = (max(weight_list) + min(weight_list)) / 2
        threshold = (max(auxiliar_list) + min(auxiliar_list)) / 2

        auxiliar_network = self.remove_edges_for_mln(network, 0.3)
        #return [network, threshold]
        return [(network,auxiliar_network), threshold]

    def remove_edges_for_mln(self, network, percentage):
        #self.limiar_value=0.30 - 0.50 - 0.7
        network_size = network.vcount()
        edgesList = network.get_edgelist()
        weight_list = network.es['weight']

        limiar_per = percentage
        x = (len(edgesList) * limiar_per)
        new_size = int(len(edgesList) - x)
        sorted_values = sort_network(edgesList, weight_list)

        new_weights = []
        new_edges = []
        for i in range(new_size):
            values = sorted_values[i]
            edge = values[0].split('-')
            edge_pair = (int(edge[0]), int(edge[1]))
            new_edges.append(edge_pair)
            weight = values[1]
            new_weights.append(weight)

        new_network = Graph()
        new_network.add_vertices(network_size)
        new_network.add_edges(new_edges)
        new_network.es['weight'] = new_weights
        return new_network


    def generate(self):
        if self.network_type == 'noun':
            #print "creando red de sustantivos"
            return self.noun_based_network()
        if self.network_type == 'tfidf' or self.network_type == 'd2v':
            return self.tfidf_d2v_based_network()
        if self.network_type == 'mln':
            return self.multilayer_based_network()

        return ["dictionary", "key: nombre del documento o cluster",
                'value: red del documento con los limiares aplicados',
                'devolver tambien el threshold para antiredundancia MDS']




class CNMeasures(object):

    def __init__(self, network, extra_network_mln=None):
        self.network = network
        self.extra_network = extra_network_mln
        if self.extra_network is None:
            self.extra_network = self.network
        self.node_rankings = dict()

    def get_node_rankings(self):
        return self.node_rankings

    def degree(self, paremeters=None):
        print "measuring degree"
        #graph_degree = self.network.degree()
        graph_degree = self.extra_network.degree()
        graph_stg = self.network.strength(weights=self.network.es['weight'])
        ranked_by_degree = reverseSortList(graph_degree)
        ranked_by_stg = reverseSortList(graph_stg)
        print ranked_by_degree
        print ranked_by_stg
        self.node_rankings['dg'] = ranked_by_degree
        self.node_rankings['stg'] = ranked_by_stg
        #return [ranked_by_degree, ranked_by_stg]

    def shortest_path(self, paremeters=None):
        print "measuring sp" # falta basada en pesos, hay que modificar
        measure = []
        measure2 = []
        measure3 = []
        network_size = self.network.vcount()
        new_weights = inverse_weights(self.network.es['weight'])
        weight = new_weights[0]
        weight2 = new_weights[1]

        for i in range(network_size):
            #lenghts = self.network.shortest_paths(i)[0]
            lenghts = self.extra_network.shortest_paths(i)[0]
            lenghts2 = self.network.shortest_paths(i, weights=weight)[0]
            lenghts3 = self.network.shortest_paths(i, weights=weight2)[0]
            sp = average(lenghts)
            sp2= average(lenghts2)
            sp3 = average(lenghts3)
            measure.append(sp)
            measure2.append(sp2)
            measure3.append(sp3)
        ranked_by_sp = sortList(measure)
        ranked_by_sp_w = sortList(measure2)
        ranked_by_sp_w2 = sortList(measure3)
        print ranked_by_sp
        print ranked_by_sp_w
        print ranked_by_sp_w2
        self.node_rankings['sp'] = ranked_by_sp
        self.node_rankings['sp_w'] = ranked_by_sp_w
        self.node_rankings['sp_w2'] = ranked_by_sp_w2
        #return [ranked_by_sp, ranked_by_sp_w, ranked_by_sp_w2]



    def page_rank(self, paremeters=None):
        print "measuring pr"
        #graph_pr = self.network.pagerank()
        graph_pr = self.extra_network.pagerank()
        graph_pr_w = self.network.pagerank(weights=self.network.es['weight'])
        ranked_by_pr = reverseSortList(graph_pr)
        ranked_by_pr_w = reverseSortList(graph_pr_w)
        print ranked_by_pr
        print ranked_by_pr_w
        self.node_rankings['pr'] = ranked_by_pr
        self.node_rankings['pr_w'] = ranked_by_pr_w
        #return [ranked_by_pr, ranked_by_pr_w]


    def betweenness(self, paremeters=None):
        print "measuring btw"
        #graph_btw = self.network.betweenness()
        graph_btw = self.extra_network.betweenness()
        graph_btw_w = self.network.betweenness(weights=self.network.es['weight'])
        ranked_by_btw = reverseSortList(graph_btw)
        ranked_by_btw_w = reverseSortList(graph_btw_w)
        print ranked_by_btw
        print ranked_by_btw_w
        self.node_rankings['btw'] = ranked_by_btw
        self.node_rankings['btw_w'] = ranked_by_btw_w
        #return [ranked_by_btw , ranked_by_btw_w]


    def clustering_coefficient(self, paremeters=None):
        print "measuring cc"
        #graph__cc = self.network.transitivity_local_undirected()
        graph__cc = self.extra_network.transitivity_local_undirected()
        graph__cc_w = self.network.transitivity_local_undirected(weights=self.network.es['weight'])
        ranked_by_cc = reverseSortList(graph__cc)
        ranked_by_cc_w = reverseSortList(graph__cc_w)
        print ranked_by_cc
        print ranked_by_cc_w
        self.node_rankings['cc'] = ranked_by_cc
        self.node_rankings['cc_w'] = ranked_by_cc_w
        #return [ranked_by_cc, ranked_by_cc_w]



    def absortion_time(self, paremeters=None):
        print "measuring at"
        #obj = absorption.AbsorptionTime(self.network)
        obj = absorption.AbsorptionTime(self.extra_network)
        absorption_time = obj.get_all_times()
        ranked_by_absorption = sortList(absorption_time)
        print ranked_by_absorption
        self.node_rankings['at'] = ranked_by_absorption


    '''
    type = backbone / merged
    order = greater / less
    '''
    #def symmetry(self, type, order, h):
    def symmetry(self, parameters):
        print "measuring symetry"
        #obj = hierarchical.Symmetry(self.network)
        obj = hierarchical.Symmetry(self.extra_network)
        results = []
        # order : h - l
        # type: b - m
        # h: 2-3

        if len(parameters)!=0:
            #order = parameters[0]
            #type = parameters[1]
            #h = parameters[2]
            #print "order: ", order
            #print "type: " , type
            #print "h:" , h
            print "algnunas measures" , parameters
            for i in range(0, len(parameters), 3):
                order = parameters[i]
                type = parameters[i+1]
                h = parameters[i+2][1]
                sorted_by_syms = obj.sort_by_symmetry(order, type, h)
                key = 'sym_' + order + '_' + type + '_' + parameters[i+2]
                self.node_rankings[key] = sorted_by_syms
        else:
            print "todas las simetrias"
            sorted_h_b_h2 = obj.sort_by_symmetry('h', 'b', '2')
            sorted_h_b_h3 = obj.sort_by_symmetry('h', 'b', '3')
            sorted_h_m_h2 = obj.sort_by_symmetry('h', 'm', '2')
            sorted_h_m_h3 = obj.sort_by_symmetry('h', 'm', '3')
            sorted_l_b_h2 = obj.sort_by_symmetry('l', 'b', '2')
            sorted_l_b_h3 = obj.sort_by_symmetry('l', 'b', '3')
            sorted_l_m_h2 = obj.sort_by_symmetry('l', 'm', '2')
            sorted_l_m_h3 = obj.sort_by_symmetry('l', 'm', '3')
            self.node_rankings['sym_h_b_h2'] = sorted_h_b_h2
            self.node_rankings['sym_h_b_h3'] = sorted_h_b_h3
            self.node_rankings['sym_h_m_h2'] = sorted_h_m_h2
            self.node_rankings['sym_h_m_h3'] = sorted_h_m_h3
            self.node_rankings['sym_l_b_h2'] = sorted_l_b_h2
            self.node_rankings['sym_l_b_h3'] = sorted_l_b_h3
            self.node_rankings['sym_l_m_h2'] = sorted_l_m_h2
            self.node_rankings['sym_l_m_h3'] = sorted_l_m_h3
            '''
            print sorted_h_b_h2
            print sorted_h_b_h3
            print sorted_h_m_h2
            print sorted_h_m_h3
            print sorted_l_b_h2
            print sorted_l_b_h3
            print sorted_l_m_h2
            print sorted_l_m_h3
            '''
            #results = [sorted_h_b_h2, sorted_h_b_h3, sorted_h_m_h2, sorted_h_m_h3, sorted_l_b_h2,
            #           sorted_l_b_h3, sorted_l_m_h2, sorted_l_m_h3]
        #return results

    def concentrics(self, parameters):
        print "measuring concentrics"
        results = []
        #obj = hierarchical.Concentric(self.network)
        obj = hierarchical.Concentric(self.extra_network)

        if len(parameters)!=0:
            print "algunas measures" , parameters
            for i in range(0, len(parameters), 2):
                type = int(parameters[i])-1
                h = int(parameters[i+1][1])
                sorted_by_ccts = obj.sort_by_concentric(type, h)
                key = 'ccts_' + str(type+1) + '_h' + str(h)
                self.node_rankings[key] = sorted_by_ccts
                #results.append(sorted_by_ccts)

        else:
            print "todas las concentricas con todas las h, o solo un subconjunto de las mejores, devuelve las 16"
            for h in range(2,4):
                for type in range(8):
                    sorted_by_ccts = obj.sort_by_concentric(type, h)
                    print sorted_by_ccts
                    key = 'ccts_' + str(type+1) + '_h' + str(h)
                    self.node_rankings[key] = sorted_by_ccts
                    #results.append(sorted_by_ccts)


        '''
        else:
            print "medidas concentricas con las 8 tipos de medidas, pero h adaptado al diametro de la red"
            diameter = self.network.diameter()
            if diameter<=4:
                h = 2
            else:
                h = 3

            for type in range(8):
                sorted_by_ccts = obj.sort_by_concentric(type, h)
                key = 'ccts_' + str(type + 1)
                print diameter, h, key, sorted_by_ccts
                self.node_rankings[key] = sorted_by_ccts
        '''

        # modificar, adaptar las concentricas para que h=2 o h=3 dependiendo del diametro de la red en cuestion



    def accessibility(self, h):
        print "measuring accesibility"
        results = []
        #obj = hierarchical.Accessibility(self.network)
        obj = hierarchical.Accessibility(self.extra_network)
        if len(h)==0:
            h2 = obj.sort_by_accessibility("2")
            h3 = obj.sort_by_accessibility("3")
            key = 'accs_h2'
            key2 = 'accs_h3'
            self.node_rankings[key] = h2
            self.node_rankings[key2] = h3
        else:
            parameter = h[0][1]
            sorted_by_accs = obj.sort_by_accessibility(parameter)
            key = 'accs_h' + parameter
            self.node_rankings[key] = sorted_by_accs
            #results = [acc]

        #return results


    def generalized_accessibility(self, parameters=None):
        print "measuring generalized accesibility"
        #obj = hierarchical.GeneralizedAccesibility(self.network)
        obj = hierarchical.GeneralizedAccesibility(self.extra_network)
        sorted_by_generalized = obj.sort_by_accesibility()
        self.node_rankings['gaccs'] = sorted_by_generalized
        print sorted_by_generalized
        #return sorted_by_generalized


    def all_measures(self, parameters=None):
        print "measuring all"
        self.degree()
        self.shortest_path()
        self.page_rank()
        self.betweenness()
        self.clustering_coefficient()
        self.generalized_accessibility()
        #self.absortion_time()
        #self.concentrics([])
        #self.symmetry([])
        #self.accessibility([])


    def traditional_measures(self, parameters=None):
        print "measuring traditional measures"
        self.degree()
        self.shortest_path()
        self.page_rank()
        self.betweenness()
        self.clustering_coefficient()
        #sorted_by_shortest_path
        #print sorted_by_degree
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
        allRankings = dict()
        if find_term(self.measures, 'ccts'):
            self.measures = utils.manage_vector(self.measures, 'ccts')
        if find_term(self.measures, 'sym'):
            self.measures = utils.manage_vector(self.measures, 'sym')
        print "obtained measures", self.measures

        index = 1
        for i in self.networks.items():
            document_name = i[0]
            actual_network = i[1][0] ######
            if type(actual_network) is not tuple: # para verificar si usar los dos tipos de grafos para MLN
                obj = CNMeasures(actual_network)
            else:
                normal_network = actual_network[0]  # network sin aristas removidas
                extra_network = actual_network[1]   # network con aristas removidas por limiares
                obj = CNMeasures(normal_network, extra_network)

            #obj = CNMeasures(actual_network)
            dictionary = obj.manage_measures()
            print index, document_name
            index+=1
            for i in self.measures:
                measure_parameter = i.split('_')
                measure = measure_parameter[0]
                parameters = measure_parameter[1:]
                dictionary[measure](parameters)
            document_rankings = obj.get_node_rankings()
            allRankings[document_name] = document_rankings
        return allRankings

        '''
        actual_network = self.networks['op94ag07-a'][0]
        obj = CNMeasures(actual_network)
        dictionary = obj.manage_measures()
        if find_term(self.measures, 'ccts'):
            self.measures = utils.manage_vector(self.measures, 'ccts')
        if find_term(self.measures, 'sym'):
            self.measures = utils.manage_vector(self.measures, 'sym')
        print "obtained measures" , self.measures

        for i in self.measures:
            measure_parameter =  i.split('_')
            measure = measure_parameter[0]
            parameters = measure_parameter[1:]
            dictionary[measure](parameters)

        print "Final rankings de todas las medidas!!!!!"
        the_rankings = obj.get_node_rankings()
        return the_rankings
        '''
