from corpus_loader import Loader
from utils import parameter_extractor, deleteFolders
from embeddings_generator import Vectorization
from text_conversion import CorpusConversion
from network import NetworkManager , NodeManager
from summarization import SummaryGenerator
from validation import Validation
from configuration import extras, final_results
from random import shuffle, choice



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

        use_machine_learning = data['ml'][0]  ## VERY IMPORTANT NOW
        method, classifier, kFold, use_traditional_features  = None , None, None, None
        if use_machine_learning:
            method, classifier, kFold, use_traditional_features = data['ml'][1][0], data['ml'][1][1], data['ml'][1][2], data['ml'][1][3]


        network = data['network']
        network_type = network[0]  # tipo de red: noun, tfidf, d2v , mln
        network_parameters = network[1]  # todos los parametros del tipo de red que se va a utilizar
        mln_type_flag = network_type == 'mln'  # para verificar en corpus loader si se tiene que cargar para una multilayer network

        extracted_net_parameters = parameter_extractor(network_type, network_parameters)

        mln_type = extracted_net_parameters['mln_type']
        sw_removal = extracted_net_parameters['sw_removal']
        limiar_value = extracted_net_parameters['limiar_value']
        limiar_type = extracted_net_parameters['limiar_type']
        size_d2v = extracted_net_parameters['size_d2v']
        inter_edge_mln = extracted_net_parameters['inter_edge']
        limiar_mln = extracted_net_parameters['limiar_mln']

        network_measures = data['measures']
        selection_method = data['selection']

        print use_machine_learning
        print method, classifier, kFold, use_traditional_features


        # use_machine_learning and method   ---> muy importantes


        '''
        0 cargar el corpus indicado y dejarlo listo para ser pre-procesado    
        '''
        # 1. Corpus loader
        obj = Loader(language=language, type_summary=type_summary, corpus=corpus_name, size=resumo_size_parameter, mln=mln_type_flag, use_ml=use_machine_learning)
        loaded_corpus = obj.load()  # diccionario que tiene como key el nombre del documento o nombre del grupo y como claves los documentos y sus sizes


        #for i in loaded_corpus.items():
        #    print i














        # 2. Corpus processing

        # 3. Corpus vectorization

        # 4. Network creation

        # 5. Network prunning

        # 6. Node weighting

        # 7. Node ranking

        # 8. Summarization

        # 9. Validation





    def parse_file(self, doc):
        dictionary = dict()
        dictionary['language'] = 'ptg'
        #dictionary['language'] = 'eng'
        dictionary['type'] = ('SDS' , None)
        #dictionary['type'] = ('MDS', 0)  # 0->sin antiredundancia, 1->metodo de ribaldo 2->metodo de ngrams  3-> maximum marginal relevance
        dictionary['corpus'] = 0  # 1  para DUC2004 en caso del ingles, solo para MDS
        dictionary['size'] = 'w'
        dictionary['ml'] = (True, ['method1','naive_bayes' , 10, False])  # metodo ,classifier(naive_bayes/svm/decision_tree/logistic) , kfoldcrossvalidation (10), use traditional measures
        # method1 --> training and testing CN y ML
        # method2 --> primero hago ranking como estaba usando antes y luego para la seleccion final de sentencias aplico ml en el ranking final para verificar si la sentencia va o no va
        #dictionary['ml'] = (False, [])
        '''
        - forma de usar ml:  1. usar enfoque training and testing (cn y ml) o  2. todo como la metodoligia anterior y al final usar ml para verificar los rankings
        - clasificador: naive bayes , svm, dt , logistic regression , etc
        - k fold cross validation : 10 
        - features CN : las que son seleccionadas en cn measures
        - features tradicionales: seleccionar features tradicionales
        - CN + tradicionales      
        '''




        dictionary['network'] = ('noun', [])
        # dictionary['network'] = ('tfidf', [])
        #dictionary['network'] = ('mln', ['tfidf', [1.1, 1.3, 1.5, 1.7, 1.9], [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]]) # inter - limiar remocion
        # dictionary['network'] = ('mln', ['noun', [1.1, 1.3, 1.5], [0.1, 0.15, 0.20]])



        dictionary['measures'] = ['dg' , 'pr', 'btw']
        # dictionary['measures'] = ['at' , 'gaccs']
        # dictionary['measures'] = ['trad']
        #dictionary['measures'] = ['*']
        # dictionary['measures'] = ['sp' , 'pr' , 'btw' , 'cc']


        dictionary['selection'] = 's'  # simple    s
        #dictionary['selection'] = 'v' # votacion  v






        return dictionary



if __name__ == '__main__':

    output = final_results['prueba2']
    obj = Summarizer('input.txt', output)
    obj.execute()