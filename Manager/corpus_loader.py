from utils import read_document, count_words, read_document_english
from configuration import corpus_dir, summaries_dir
import os

class Loader(object):

    def __init__(self, language, type_summary, corpus, size, mln):
        self.dictionary = dict()
        self.language = language
        self.type_summary = type_summary
        self.corpus = corpus
        self.size = size
        self.mln = mln
        # self.dictionary['ptg'] = [['temario_v1' , 'temario_v2'] , ['cstnews_v1' , 'cstnews_v2']]
        self.dictionary['ptg'] = [['temario_v1', 'temario_v2'], ['cstnews_v1', 'cstnews_v2']]
        # self.dictionary['eng'] = [['duc2002' , 'duc2003'] , ['duc2002' , 'duc2003']]
        self.dictionary['eng'] = ['duc2002', 'duc2003']

        self.dictionary_corpus = dict()
        self.dictionary_corpus['temario_v1'] = self.load_temario
        self.dictionary_corpus['temario_v2'] = self.load_temario
        self.dictionary_corpus['cstnews_v1'] = self.load_cst_news
        self.dictionary_corpus['cstnews_v2'] = self.load_cst_news
        self.dictionary_corpus['duc2002'] = self.load_duc2002
        self.dictionary_corpus['duc2003'] = self.load_duc2003

    def load(self):
        if self.language == 'ptg':
            selected = self.dictionary[self.language][self.type_summary][self.corpus]
            # print selected
            return self.dictionary_corpus[selected](selected)
        else:
            selected = self.dictionary[self.language][self.corpus]
            # print selected
            return self.dictionary_corpus[selected](selected, self.type_summary)

    def load_temario(self, version):
        '''
        TEMARIO: NUmero de palabras o sentencias del resumo final
        CSTNews: 70% en numero de palabras del documento con mayor peso
        '''
        print "temario :)"
        corpus_dictionary = dict()

        if version == 'temario_v1':
            path = corpus_dir[version]
            path_sumarios = summaries_dir[version]
            documents = os.listdir(path)
            sumarios = os.listdir(path_sumarios)

            for i in documents:
                docPath = path + '/' + i
                # print docPath
                document_name = i[3:]
                document_name = document_name[:-4]

                document_sentences = read_document(docPath, self.language)

                corpus_dictionary[document_name] = [document_sentences]

            for i in sumarios:
                summPath = path_sumarios + i
                # print summPath
                summary_name = i[4:]
                summary_name = summary_name[:-4]
                size_summary = count_words(summPath, self.language)

                value = corpus_dictionary[summary_name]  # size_summary
                value.append(size_summary)
                corpus_dictionary[summary_name] = value

        else:
            print 'version 2'

        # corpus = ['diccionario con nombres y los datos' ,'loaded corpus sin procesar' , 'vectores de sizes de sumarios']
        # return corpus
        return corpus_dictionary

    def load_cst_news(self, version):
        print "cst news :)"
        corpus_dictionary = dict()
        if version == 'cstnews_v1':
            path = corpus_dir[version]
            clusters = os.listdir(path)
            for i in clusters:
                sub_path = path + i + '/' + corpus_dir['textosFonte']
                documents = os.listdir(sub_path)
                allSentences = []
                document_lenghts = []

                for j in documents:
                    document = sub_path + j
                    document_sentences = read_document(document, self.language)
                    document_size = count_words(document, self.language)
                    document_lenghts.append(document_size)
                    allSentences.extend(document_sentences)

                size_cluster = max(document_lenghts)
                size_summary = (30 * size_cluster) / 100
                corpus_dictionary[i] = [allSentences, size_summary]

        else:
            print 'version 2'

        # corpus = ['diccionario con nombres y los datos' ,'loaded corpus sin procesar' , 'vectores de sizes de sumarios']
        return corpus_dictionary

    def load_duc2002(self, version, summary_type):
        print "duc2002 :)"
        corpus_dictionary = dict()
        path = corpus_dir[version]
        clusters = os.listdir(path)

        if summary_type == 0:
            print "SDS"
            for i in clusters:
                sub_path = path + i + '/'
                documents = os.listdir(sub_path)
                for j in documents:
                    document = sub_path + j
                    document_sentences = read_document_english(document)
                    # print document_sentences

                    name = i + "_" + j
                    name = name[:name.find('_') - 1] + '-' + name[name.find('_') + 1:]
                    corpus_dictionary[name] = [document_sentences, 100]

        else:
            print "MDS"
            for i in clusters:
                allSentences = []
                sub_path = path + i + '/'
                documents = os.listdir(sub_path)
                for j in documents:
                    document = sub_path + j
                    document_sentences = read_document_english(document)
                    allSentences.extend(document_sentences)

                name = i[:len(i) - 1]
                corpus_dictionary[name] = [allSentences, 200]

            print len(corpus_dictionary)

        return corpus_dictionary

    def load_duc2003(self, version, summary_type):
        print "duc2003 :)"


if __name__ == '__main__':

    print "haberr"