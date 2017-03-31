from corpus_loader import Loader
from utils import parameter_extractor
from embeddings_generator import Vectorization
from text_conversion import CorpusConversion

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

        anti_redundancy_threshold = None  # si los documentos no requieren vectorizacion , este es calculado en la generacion de los sumarios
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
            obj = Vectorization(processed_corpus, network_type, inference_d2v, size_d2v)
            vectorized_corpus = obj.calculate()


        print vectorized_corpus


        '''
        return matutils.cossim(vec_tfidf, vec_tfidf2)  gemsim
        '''








    def parse_file(self, file):
        intra = 0
        inter = 0
        dictionary = dict()
        dictionary['language'] = 'ptg'
        #dictionary['language'] = 'eng'
        dictionary['type'] = ('SDS' , 0)
        #dictionary['type'] = ('MDS', 1)
        dictionary['corpus'] = 0
        dictionary['size'] = 'w'

        # dictionary['network'] = [('noun',[]) , ('tfidf', [True , 0, 'dist_cos' ]), ('d2v',[300, True, 1, 'dist_euc', False]),
        # ('mln', ['noun', intra, inter]), ('mln', ['tfidf', intra, inter, True, 0, 'dist_cos']), ('mln' , ['d2v', intra, inter, 300, True, 1, 'dist_euc', True])]


        #dictionary['network'] = ('noun', [])
        #dictionary['network'] = ('tfidf', [True, -1, 'cos'])
        dictionary['network'] = ('d2v', [True, 0, 'euc', 100, False])
        #dictionary['network'] = ('mln', ['noun', 0.5, 0.5])
        #dictionary['network'] = ('mln', ['tfidf', True, -1, 'cos', 0.5, 0.5])
        # dictionary['network'] = ('mln', ['d2v', True, 0, 'euc', 100, False, 0.5, 0.5])



        dictionary['measures'] = ['dg', 'st', 'acc_h2']  # postions pode ser

        dictionary['selection'] = 1  # simple
        # dictionary['selection'] = 2 # votacion
        # dictionary['selection'] = 3 # machine learning


        dictionary['validation'] = '*'  # todos
        # dictionary['validation'] = 'R'  # rouge1
        # dictionary['validation'] = 'ST' # P-R-F
        return dictionary


if __name__ == '__main__':

    obj = Summarizer('test1.txt')
    obj.execute()

