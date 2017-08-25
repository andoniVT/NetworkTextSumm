from corpus_loader import Loader
from utils import parameter_extractor, deleteFolders
from embeddings_generator import Vectorization
from text_conversion import CorpusConversion
from network import NetworkManager , NodeManager
from summarization import SummaryGenerator
from validation import Validation
from configuration import extras, final_results
from random import shuffle, choice


'''
Portuguese:
   SDS: Temario2006 Abstracts
   MDS: CSTNews Extracts
English:
   SDS: DUC2002 Abstracts   
   MDS: DUC2002 Extracts - DUC2004 Abstracts
'''



class Summarizer(object):
    def __init__(self, test, output):
        self.data = self.parse_file(test)
        self.output_excel = output

    def execute(self):
        data = self.data
        language = data['language']
        if data['type'][0] == 'SDS':
            type_summary = 0
        else:
            type_summary = 1
        anti_redundancy_method = data['type'][1]
        corpus_name = data['corpus']

        resumo_size_parameter = data['size']  # para definir el tamanio de los sumarios, en relacion a numero de palabras o sentencias, o fijo

        network = data['network']
        network_type = network[0]  # tipo de red: noun, tfidf, d2v , mln
        network_parameters = network[1]  # todos los parametros del tipo de red que se va a utilizar
        mln_type_flag = network_type == 'mln'  # para verificar en corpus loader si se tiene que cargar para una multilayer network

        extracted_net_parameters = parameter_extractor(network_type, network_parameters)
        mln_type = extracted_net_parameters['mln_type']
        sw_removal = extracted_net_parameters['sw_removal']
        limiar_value = extracted_net_parameters['limiar_value']
        limiar_type = extracted_net_parameters['limiar_type']
        #distance = extracted_net_parameters['distance']
        size_d2v = extracted_net_parameters['size_d2v']
        #inference_d2v = extracted_net_parameters['inference_d2v']
        inter_edge = extracted_net_parameters['inter_edge']
        #intra_edge = extracted_net_parameters['intra_edge']
        limiar_mln = extracted_net_parameters['limiar_mln']

        print extracted_net_parameters


        #anti_redundancy_threshold = None  # si los documentos no requieren vectorizacion , este es calculado en la generacion de los sumarios
        # basado en la distancia coseno de las palabras, sin necesidad de generar los vectores de cada documento
        # si los documentos requierein vectorizacion, entonces este valor sera atribuido a partir del valor calculado en la etapa de vectorizacion


        network_measures = data['measures']
        selection_method = data['selection']
        validation = data['validation']

        '''
        0 cargar el corpus indicado y dejarlo listo para ser pre-procesado    
        '''
        obj = Loader(language=language, type_summary=type_summary, corpus=corpus_name, size=resumo_size_parameter, mln=mln_type_flag)
        loaded_corpus = obj.load()  # diccionario que tiene como key el nombre del documento o nombre del grupo y como claves los documentos y sus sizes

        for i in loaded_corpus.items():
            print i



        #for i in loaded_corpus.items():
        #    grupos =  i[1]
        #    sentences = grupos[0]
        #    sizes = grupos[1]
        #    for j in sentences:
        #        print j
        #        print j[0] , j[1]




        top_sentences = dict()   # solo para MDS
        #if anti_redundancy_method is not None:
        #    for i in loaded_corpus.items():
        #        doc_name = i[0]
        #        tops = i[1][2]
        #        top_sentences[doc_name] = tops


        '''
        1. Pre-procesamiento de los corpus
        


        obj = CorpusConversion(loaded_corpus, language, network_type, mln_type, sw_removal)
        processed_corpus = obj.convert()

        

        #for i in processed_corpus.items():
        #    print len(i[1][1])
        '''



        '''
        2. Vectorizacion de los corpus (auxiliar - caso sea requerido)
        
        vectorized_corpus = None

        if network_type == 'noun' or mln_type == 'noun':
            pass
        else:

            if  network_type== 'mln':
                network_type_subtype = mln_type
            else:
                network_type_subtype = network_type


            #cargar corpus auxiliar para entrenamiento
            if language == 'eng':
                #obj = Vectorization(processed_corpus, network_type, inference_d2v, size_d2v)
                #obj = Vectorization(processed_corpus, network_type_subtype, inference_d2v, size_d2v)
                obj = Vectorization(processed_corpus, network_type_subtype, size_d2v)
                vectorized_corpus = obj.calculate()
            else:
                print "cargando nuevo"
                type_summary_inverted = 0
                if type_summary==0:
                    type_summary_inverted=1
                obj = Loader(language=language, type_summary=type_summary_inverted, corpus=corpus_name, size=resumo_size_parameter, mln=mln_type_flag)
                auxiliar_corpus = obj.load()
                obj = CorpusConversion(auxiliar_corpus, language, network_type, mln_type, sw_removal)
                processed_auxiliar = obj.convert()
                #obj = Vectorization(processed_corpus, network_type, inference_d2v, size_d2v, processed_auxiliar)
                #obj = Vectorization(processed_corpus, network_type_subtype, inference_d2v, size_d2v, processed_auxiliar)
                obj = Vectorization(processed_corpus, network_type_subtype, size_d2v, processed_auxiliar)
                vectorized_corpus = obj.calculate()
        '''


        '''
        3. Creacion de la red  y  4. Eliminacion de nodos, limiares
        

        #obj = NetworkManager(network_type, mln_type, processed_corpus, vectorized_corpus, distance, inter_edge, limiar_mln, limiar_value)
        obj = NetworkManager(network_type, mln_type, processed_corpus, vectorized_corpus, inter_edge, limiar_mln, limiar_value, limiar_type)
        complex_networks = obj.create_networks()
        '''


        #for i in complex_networks.items():
        #    print i



        '''
        5. Node weighting and node ranking
        

        obj = NodeManager(complex_networks, network_measures)
        all_documentRankings = obj.ranking()
        '''


        #for i in all_documentRankings.items():
        #    print i



        '''
        6. Summarization
        #corpus, rankings, sentence_selection, anti_redundancy
        
        print "Summarization!!!"
        obj = SummaryGenerator(processed_corpus, complex_networks, all_documentRankings, selection_method, anti_redundancy_method, top_sentences)
        obj.generate_summaries()
        '''




        '''
        7. Validation
        
        
        key = choice(all_documentRankings.keys())
        number_of_measures = len(all_documentRankings[key][0])
        print  limiar_mln
        parameters_to_show_table = []

        if limiar_mln is not None:
            first_value =  len(inter_edge)
            second_value = len(limiar_mln)
            third_value = number_of_measures
            parameters_to_show_table.append(inter_edge)
            parameters_to_show_table.append(limiar_mln)
            #third_value = len(inter_edge)
            #first_value = number_of_measures * len(limiar_mln)
        elif limiar_value is not None:
            first_value = 1
            second_value = len(limiar_value)
            third_value = number_of_measures
            parameters_to_show_table.append(None)
            parameters_to_show_table.append(limiar_value)
        else:
            first_value = 1
            second_value = 1
            third_value = number_of_measures

        #second_value = len(limiar_value)
        print first_value , second_value , third_value
        # validation language type_summary corpus_name
        obj = Validation(validation, language, type_summary, corpus_name, [first_value, second_value, third_value], self.output_excel, parameters_to_show_table)
        obj.validate('results.csv')
        deleteFolders(extras['Automatics'])
        '''



    def parse_file(self, file):
        intra = 0
        inter = 0
        dictionary = dict()
        dictionary['language'] = 'ptg'
        #dictionary['language'] = 'eng'
        #dictionary['type'] = ('SDS' , None)
        dictionary['type'] = ('MDS', 0)  #0->sin antiredundancia, 1->metodo de ribaldo 2->metodo de ngrams  3-> maximum marginal relevance
        dictionary['corpus'] = 0  #1  para DUC2004 en caso del ingles, solo para MDS
        dictionary['size'] = 'w'




        #dictionary['network'] = ('noun', [])
        #dictionary['network'] = ('tfidf', [True, -1, 'cos']) # remover todos los parametros, vacio como la red baseada en sustantivos
        #dictionary['network'] = ('tfidf', [])
        # todas las preuvas que iniclaes fueron con limiar=2
        # 5-4 no sirve, muy alto
        # 3(no)-2 si,  puede ser alto
        # 1 ok  normal
        #dictionary['network'] = ('d2v', [False, 2,  'cos', 300, False])
        #dictionary['network'] = ('d2v', [False, 0.1, 'cos', 200, False])  # ahora con porcentajes , nueva funcion de redundancia
        #dictionary['network'] = ('d2v', [False, 'knn', 'cos', 200, False])  # ahora red knn

        #dictionary['network'] = ('d2v' , [False, ('limiar',[0.1,0.15,0.20,0.25,0.3]), 'cos', 200, False ])  # eliminar el ultimo False , ya que no se usara la prediccion
        #dictionary['network'] = ('d2v', [False, ('knn', [3,5,7,9,11,13,15]), 'cos', 200, False])  # eliminar el ultimo False , ya que no se usara la prediccion, eliminar tambien medicion del coseno



        #dictionary['network'] = ('d2v', [False, ('limiar', [0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45,0.5]), 300])
        #dictionary['network'] = ('d2v', [False, ('knn', [3,5,7,11,13,15]), 200])

        #dictionary['network'] = ('gd2v', [('limiar', [0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45,0.5])])



        dictionary['network'] = ('mln', ['tfidf', [1.1, 1.3, 1.5, 1.7, 1.9], [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]])
        #dictionary['network'] = ('mln', ['noun', [1.1, 1.3, 1.5], [0.1, 0.15, 0.20]])
        #dictionary['network'] = ('mln', ['noun', [1.1, 1.3, 1.5, 1.7, 1.9], [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]])  # inter - limiar remocion

        #dictionary['network'] = ('mln', ['tfidf', [1.1, 1.3], [0.1, 0.15, 0.20]])  # inter - limiar remocion
        #dictionary['network'] = ('mln', ['tfidf', [1.5, 1.7, 1.9], [0.1, 0.15, 0.20]])


    
        '''
        dictionary['network'] = ('d2v', [False, ('limiar', [0.1, 0.15, 0.20, 0.25, 0.3]), 200])  # eliminar el ultimo False , ya que no se usara la prediccion
        dictionary['network'] = ('d2v', [False, ('knn', [3, 5, 7, 9, 11, 13, 15]), 200])  # eliminar el ultimo False , ya que no se usara la prediccion, eliminar tambien medicion del coseno

        #dictionary['network'] = ('mln', ['noun', 1.9, 0.5])  # inter - limiar remocion
        dictionary['network'] = ('mln', ['tfidf', True, -1, 'cos', 1.9, 0.5])  # inter - intra (pasado)    inter - limiar remocion (ahora) # remover los parametros de tfidf
        #dictionary['network'] = ('mln', ['d2v', False, 0.3, 'cos', 300, False, 1.5, 1.0])   #

        dictionary['network'] = ('mln', ['noun', [1.7, 1.9], [0.4, 0.45, 0.5]])  # inter - limiar remocion
        dictionary['network'] = ('mln', ['tfidf', [1.7, 1.9], [0.4, 0.45, 0.5]])  # inter - limiar remocion
        dictionary['network'] = ('mln', ['d2v', False, ('limiar',[0.3]), 300, [1.7,1.9], [0.4,0.45,0.5]])  #
        '''



        #dictionary['measures'] = ['dg', 'pr', 'accs_h2' , 'ccts_2_h2', 'ccts_4_h3' , 'sym_h_b_h3']  # postions pode ser
        #dictionary['measures'] = ['dg', 'gaccs', 'accs_h2', 'ccts', 'sym']  # postions pode ser
        #dictionary['measures'] = ['dg', 'ccts_2_h2', 'ccts_4_h3', 'ccts_7_h2']
        #dictionary['measures'] = ['dg', 'ccts']
        #dictionary['measures'] = ['sym_h_m_h2', 'sym_l_b_h3' , 'dg', 'sym_h_b_h3']
        #dictionary['measures'] = ['dg' , 'pr', 'btw']
        #dictionary['measures'] = ['ccts']
        #dictionary['measures'] = ['dg']
        #dictionary['measures'] = ['katz']
        #dictionary['measures'] = ['at']
        #dictionary['measures'] = ['trad']
        dictionary['measures'] = ['*']
        #dictionary['measures'] = ['accs_h2' , 'ccts_4_h3' , 'dg', 'sym_h_m_h2']
        #dictionary['measures'] = ['sp' , 'pr' , 'btw' , 'cc']
        #dictionary['measures'] = ['ccts_2_h2', 'ccts_4_h3']
        #dictionary['measures'] = ['accs' , 'sym' , 'ccts']

        dictionary['selection'] = 's'  # simple    s
        #dictionary['selection'] = 'v' # votacion  v
        # dictionary['selection'] = 'ml' # machine learning  ml


        dictionary['validation'] = '*'  # todos
        # dictionary['validation'] = 'R'  # rouge1
        # dictionary['validation'] = 'ST' # P-R-F
        return dictionary



if __name__ == '__main__':

    output = final_results['prueba2']

    #obj = Summarizer('input.txt' , 'output.csv')
    obj = Summarizer('input.txt', output)
    obj.execute()



