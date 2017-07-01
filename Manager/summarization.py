
from utils import selectSentencesSingle , folder_creation, summary_creation, summary_random_top_creation, selectSentencesMulti, summary_random_top_creation_mds
import re

class SummaryGenerator(object):

    def __init__(self, corpus, complex_networks, node_rankings, selection_method, anti_redundancy_method, top_sentences_mds):
        self.corpus = corpus  # sentencias originales, preprocesadas , tamanio del sumario en palabras
        self.networks = complex_networks  # para obtener el threhold para MDS
        self.node_rankings = node_rankings    # dictionario , key:documento value: dictionarios con todos los rankings
        self.selection_method = selection_method   # si simple, votacion o ML
        self.anti_redundancy_method = anti_redundancy_method # tipo de antitrdundancia para MDS
        self.top_sentences_for_mds = top_sentences_mds
        folder_creation(self.node_rankings, self.anti_redundancy_method)




    def generate_summaries(self):
        if self.anti_redundancy_method is None:
            self.generate_for_SDS()
        else:
            '''
            anti_redundancy_method = 0 : sin antiredundancia aplicada
            anti_redundancy_method = 1 : metodo de ribaldo
            anti_redundancy_method = 2 : metodo de maximum marginal relevance
            '''
            self.generate_for_MDS()

    def generate_for_SDS(self):
        print "SDS!"
        sentence_selection_methods = [self.selection_simple, self.selection_voting, self.selection_ml]
        if self.selection_method == 's':
            selection = 0
        elif self.selection_method =='v':
            selection = 1
        else:
            selection =2

        for i in self.corpus.items():
            document_name = i[0]
            sentences = i[1][0]
            resumo_size = i[1][2]
            #threshold_mds = self.networks[document_name][1]  #not used
            document_rankings = self.node_rankings[document_name]
            #print document_name, sentences, resumo_size, threshold_mds
            for index, ranking in enumerate(document_rankings):
                print index+1, ranking
                selected_sentences = sentence_selection_methods[selection](sentences, ranking, resumo_size)  # llamar a los metodos de seleccion de sentencias
                resumo_name = document_name + '_system1.txt'
                summary_creation(resumo_name, selected_sentences, index+1)
                #print selected_sentences

            #selected_sentences = sentence_selection_methods[selection](sentences, document_rankings, resumo_size)  # llamar a los metodos de seleccion de sentencias okkk
            #resumo_name = document_name + '_system1.txt'
            #summary_creation(resumo_name ,selected_sentences)
            #summary_random_top_creation(resumo_name, sentences, resumo_size)

    def generate_for_MDS(self):
        print "MDS!"
        sentence_selection_methods = [self.selection_simple, self.selection_voting, self.selection_ml]
        if self.selection_method == 's':
            selection = 0
        elif self.selection_method == 'v':
            selection = 1
        else:
            selection = 2

        for i in self.corpus.items():
            document_name = i[0]
            sentences = i[1][0]
            pSentences = i[1][1]
            resumo_size = i[1][2]
            threshold_mds = self.networks[document_name][1]
            document_rankings = self.node_rankings[document_name]

            for index, ranking in enumerate(document_rankings):
                print index+1, ranking
                selected_sentences = sentence_selection_methods[selection](sentences, ranking, resumo_size, self.anti_redundancy_method, threshold_mds, pSentences)  # llamar a los metodos de seleccion de sentencias
                resumo_name = re.sub('_', '', document_name) + '_system1.txt'  ###### verificar si para el ingles no hay problemas
                summary_creation(resumo_name, selected_sentences, index+1)
                print resumo_name



            #a = input()

            #selected_sentences = sentence_selection_methods[selection](sentences, document_rankings, resumo_size,self.anti_redundancy_method, threshold_mds, pSentences)  # llamar a los metodos de seleccion de sentencias
            #resumo_name = re.sub('_', '', document_name) + '_system1.txt'  ###### verificar si para el ingles no hay problemas
            #resumo_name = document_name + '_system1.txt'
            #summary_creation(resumo_name, selected_sentences)
            #the_top =  self.top_sentences_for_mds[document_name]
            #summary_random_top_creation_mds(resumo_name, sentences, resumo_size, the_top)





    def selection_simple(self, sentences, document_rankings, resumo_size, anti_redundancy=None, threshold=None, pSentences=None):
        print "Sentence Selection: Simple"
        summary_sentences = dict()

        if anti_redundancy is None: # es single
            for ranking in document_rankings.items():
                selectedS = selectSentencesSingle(sentences, ranking, resumo_size)
                summary_sentences[selectedS[0]] = selectedS[1]
            return summary_sentences
        else: # es multi
            for ranking in document_rankings.items():
                selectedS = selectSentencesMulti(sentences, ranking, resumo_size, anti_redundancy, threshold, pSentences)
                summary_sentences[selectedS[0]] = selectedS[1]
            return summary_sentences


    def selection_voting(self, sentences, document_rankings, resumo_size,anti_redundancy=None, threshold=None, pSentences=None):
        print "Sentence Selection: Voting"

    def selection_ml(self, sentences, document_rankings, resumo_size, anti_redundancy, threshold, pSentences=None):
        print "Sentence Selection: Machine Learning"



