
from utils import selectSentencesSingle , folder_creation, summary_creation, summary_random_top_creation


class SummaryGenerator(object):

    def __init__(self, corpus, complex_networks, node_rankings, selection_method, anti_redundancy_method):
        self.corpus = corpus  # sentencias originales, preprocesadas , tamanio del sumario en palabras
        self.networks = complex_networks  # para obtener el threhold para MDS
        self.node_rankings = node_rankings    # dictionario , key:documento value: dictionarios con todos los rankings
        self.selection_method = selection_method   # si simple, votacion o ML
        self.anti_redundancy_method = anti_redundancy_method # tipo de antitrdundancia para MDS
        folder_creation(self.node_rankings)





    def generate_summaries(self):
        if self.anti_redundancy_method is None:
            self.generate_for_SDS()
        else:
            self.generate_for_MDS()


        #print self.selection_method


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
            #threshold_mds = self.networks[document_name][1]  not used
            document_rankings = self.node_rankings[document_name]
            #print document_name, sentences, resumo_size, threshold_mds
            #print document_rankings
            print document_name

            selected_sentences = sentence_selection_methods[selection](sentences, document_rankings, resumo_size)  # llamar a los metodos de seleccion de sentencias
            resumo_name = document_name + '_system1.txt'
            summary_creation(resumo_name ,selected_sentences)

            summary_random_top_creation(resumo_name, sentences, resumo_size)







    def generate_for_MDS(self):
        print "MDS!"


    def selection_simple(self, sentences, document_rankings, resumo_size):
        print "Sentence Selection: Simple"
        summary_sentences = dict()

        for ranking in document_rankings.items():
            selectedS = selectSentencesSingle(sentences, ranking, resumo_size)
            summary_sentences[selectedS[0]] = selectedS[1]

        return summary_sentences

    def selection_voting(self, sentences, document_rankings, resumo_size):
        print "Sentence Selection: Voting"

    def selection_ml(self, sentences, document_rankings, resumo_size):
        print "Sentence Selection: Machine Learning"



