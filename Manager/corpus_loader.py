from utils import read_document, count_words, read_document_english, tag_sentence , naive_tag
from configuration import corpus_dir, summaries_dir , extras
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
        self.dictionary['eng'] = ['duc2002', 'duc2004']

        self.dictionary_corpus = dict()
        self.dictionary_corpus['temario_v1'] = self.load_temario
        self.dictionary_corpus['temario_v2'] = self.load_temario
        self.dictionary_corpus['cstnews_v1'] = self.load_cst_news
        self.dictionary_corpus['cstnews_v2'] = self.load_cst_news
        self.dictionary_corpus['duc2002'] = self.load_duc2002
        self.dictionary_corpus['duc2004'] = self.load_duc2004

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
                naive_tagged_sentences = naive_tag(document_sentences)

                #corpus_dictionary[document_name] = [document_sentences]
                corpus_dictionary[document_name] = [naive_tagged_sentences]

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
            special = '.DS_Store'
            if special in clusters: clusters.remove(special)
            for i in clusters:
                sub_path = path + i + '/' + corpus_dir['textosFonte']
                documents = os.listdir(sub_path)
                if special in documents: documents.remove(special)
                #print len(documents) ,
                allSentences = []
                document_lenghts = []
                top_sentences = []
                index = 1

                for j in documents:
                    document = sub_path + j
                    document_sentences = read_document(document, self.language)
                    for k in  range(3):
                        top_sentences.append(document_sentences[k])


                    document_size = count_words(document, self.language)
                    document_lenghts.append(document_size)

                    #print document_sentences , index
                    taggedSentences = tag_sentence(document_sentences, index)
                    #print taggedSentences
                    index+=1
                    #allSentences.extend(document_sentences)
                    allSentences.extend(taggedSentences)

                size_cluster = max(document_lenghts)
                size_summary = (30 * size_cluster) / 100
                corpus_dictionary[i] = [allSentences, size_summary, top_sentences]

        else:
            print 'version 2'

        # corpus = ['diccionario con nombres y los datos' ,'loaded corpus sin procesar' , 'vectores de sizes de sumarios']
        return corpus_dictionary

    def load_duc2002(self, version, summary_type):
        print "duc2002 :)"
        corpus_dictionary = dict()
        path = corpus_dir[version]
        clusters = os.listdir(path)
        if '.DS_Store' in clusters:
            clusters.remove('.DS_Store')

        if summary_type == 0:
            print "SDS"
            for i in clusters:
                sub_path = path + i + '/'
                documents = os.listdir(sub_path)
                for j in documents:
                    document = sub_path + j
                    document_sentences = read_document_english(document)
                    # print document_sentences

                    tagged_sentences = naive_tag(document_sentences)
                    #print tagged_sentences

                    name = i + "_" + j
                    name = name[:name.find('_') - 1] + '-' + name[name.find('_') + 1:]
                    #corpus_dictionary[name] = [document_sentences, 100]
                    corpus_dictionary[name] = [tagged_sentences, 100]

        else:
            print "MDS"
            for i in clusters:
                allSentences = []
                top_sentences = []
                sub_path = path + i + '/'
                documents = os.listdir(sub_path)
                index = 1
                if len(documents)>6:
                    top_n = 1
                else:
                    top_n = 2

                for j in documents:
                    document = sub_path + j
                    document_sentences = read_document_english(document)
                    for k in range(top_n):
                        top_sentences.append(document_sentences[k])

                    tagged_sentences = tag_sentence(document_sentences, index)
                    #allSentences.extend(document_sentences)
                    allSentences.extend(tagged_sentences)
                    index+=1

                name = i[:len(i) - 1]
                corpus_dictionary[name] = [allSentences, 200, top_sentences]

            #print len(corpus_dictionary)

        return corpus_dictionary

    def load_duc2004(self, version, summary_type):
        print "duc2004 :)"
        corpus_dictionary = dict()
        path = corpus_dir[version]
        clusters = os.listdir(path)
        if '.DS_Store' in clusters:
            clusters.remove('.DS_Store')

        print "MDS"
        for i in clusters:
            allSentences = []
            sub_path = path + i + '/'
            documents = os.listdir(sub_path)
            index = 1
            sumatoria = 0
            for j in documents:
                document = sub_path + j
                document_sentences = read_document_english(document)
                tagged_sentences = tag_sentence(document_sentences, index)
                allSentences.extend(tagged_sentences)
                index += 1

            name = i[:len(i) - 1]
            corpus_dictionary[name] = [allSentences, 665]  #### tener cuidado que es cantidad de caracteres, no de palabras
        return corpus_dictionary


if __name__ == '__main__':

    print "haberr"