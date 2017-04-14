from corpus_loader import Loader
from utils import parameter_extractor, deleteFolders
from embeddings_generator import Vectorization
from text_conversion import CorpusConversion
from network import NetworkManager , NodeManager
from summarization import SummaryGenerator
from validation import Validation
from configuration import extras

class Summarizer(object):
    def __init__(self, test):
        self.data = self.parse_file(test)

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
        distance = extracted_net_parameters['distance']
        size_d2v = extracted_net_parameters['size_d2v']
        inference_d2v = extracted_net_parameters['inference_d2v']
        inter_edge = extracted_net_parameters['inter_edge']
        intra_edge = extracted_net_parameters['intra_edge']

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


        '''
        1. Pre-procesamiento de los corpus
        '''

        obj = CorpusConversion(loaded_corpus, language, network_type, mln_type, sw_removal)
        processed_corpus = obj.convert()

        #for i in processed_corpus.items():
        #    print i

        '''
        2. Vectorizacion de los corpus (auxiliar - caso sea requerido)
        '''


        vectorized_corpus = None

        if network_type == 'noun' or mln_type == 'noun':
            pass
        else:
            '''
            cargar corpus auxiliar para entrenamiento
            '''
            if language == 'eng':
                obj = Vectorization(processed_corpus, network_type, inference_d2v, size_d2v)
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
                obj = Vectorization(processed_corpus, network_type, inference_d2v, size_d2v, processed_auxiliar)
                vectorized_corpus = obj.calculate()

        '''
        for i in vectorized_corpus.items():
            print len(i[1][0])
        '''



        '''
        3. Creacion de la red  y  4. Eliminacion de nodos, limiares

        obj = CNetwork(network_type, mln_type, processed_corpus, vectorized_corpus, distance, inter_edge, intra_edge, limiar_value)
        networks = obj.generate_networks()
        print networks
        '''

        obj = NetworkManager(network_type, mln_type, processed_corpus, vectorized_corpus, distance, inter_edge, intra_edge, limiar_value)
        complex_networks = obj.create_networks()
        #print complex_networks

        '''
        for i in complex_networks.items():
            network = i[1][0]
            #print network.es['weight']
            print network
        '''

        '''
        5. Node weighting and node ranking
        '''

        obj = NodeManager(complex_networks, network_measures)
        all_documentRankings = obj.ranking()

        #for i in all_documentRankings.items():
        #    print i




        '''
        6. Summarization
        #corpus, rankings, sentence_selection, anti_redundancy
        '''

        print "Summarization!!!"
        obj = SummaryGenerator(processed_corpus, complex_networks, all_documentRankings, selection_method, anti_redundancy_method)
        obj.generate_summaries()


        '''
        7. Validation
        '''

        # validation language type_summary corpus_name
        obj = Validation(validation, language, type_summary, corpus_name)

        obj.validate('results.csv')


        deleteFolders(extras['Automatics'])










    def parse_file(self, file):
        intra = 0
        inter = 0
        dictionary = dict()
        #dictionary['language'] = 'ptg'
        dictionary['language'] = 'eng'
        dictionary['type'] = ('SDS' , None)
        #dictionary['type'] = ('MDS', 1)
        dictionary['corpus'] = 0
        dictionary['size'] = 'w'

        # dictionary['network'] = [('noun',[]) , ('tfidf', [True , 0, 'dist_cos' ]), ('d2v',[300, True, 1, 'dist_euc', False]),
        # ('mln', ['noun', intra, inter]), ('mln', ['tfidf', intra, inter, True, 0, 'dist_cos']), ('mln' , ['d2v', intra, inter, 300, True, 1, 'dist_euc', True])]


        #dictionary['network'] = ('noun', [])
        #dictionary['network'] = ('tfidf', [True, -1, 'cos'])
        dictionary['network'] = ('d2v', [False, 2, 'cos', 300, False])
        #dictionary['network'] = ('mln', ['noun', 0.5, 0.5])
        #dictionary['network'] = ('mln', ['tfidf', True, -1, 'cos', 0.5, 0.5])
        # dictionary['network'] = ('mln', ['d2v', True, 0, 'euc', 100, False, 0.5, 0.5])



        #dictionary['measures'] = ['dg', 'pr', 'accs_h2' , 'ccts_2_h2', 'ccts_4_h3' , 'sym_h_b_h3']  # postions pode ser
        #dictionary['measures'] = ['dg', 'gaccs', 'accs_h2', 'ccts', 'sym']  # postions pode ser
        #dictionary['measures'] = ['dg', 'ccts_2_h2', 'ccts_4_h3', 'ccts_7_h2']
        #dictionary['measures'] = ['dg', 'ccts']
        #dictionary['measures'] = ['sym_h_m_h2', 'sym_l_b_h3' , 'dg', 'sym_h_b_h3']
        #dictionary['measures'] = ['dg' , 'sp' ]
        dictionary['measures'] = ['*']
        #dictionary['measures'] = ['ccts']
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

    obj = Summarizer('test1.txt')
    obj.execute()

    '''
    1. Ingles problemas con SDS para  el momento de calcular therehold para red de nouns , vector de cosenos 0
    2. Ingles problemas con SDS para medidas con pesos, basadas en shortest paths, max weight min weight


    '''

