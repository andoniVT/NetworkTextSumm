import string
import unicodedata
import nltk
from morphological_analysis import lemma
from configuration import extras
from utils import load_data_from_disk

noun_list = []

class CorpusConversion(object):

    def __init__(self, corpus, language, network_type, mln_type, remove_sw):
        self.corpus = corpus
        self.language = language
        self.network_type = network_type
        self.mln_type = mln_type
        self.remove_sw = remove_sw
        self.noun_list = load_data_from_disk(extras['NounsList'])

    def convert(self):
        processed_corpus = dict()
        if self.language == 'ptg':
            print 'ptg'
            proccesing = PortugueseProcessing
        else:
            print 'eng'
            proccesing = EnglishProcessing

        only_nouns = self.network_type=='noun' or self.mln_type=='noun'

        for i in self.corpus.items():
            doc_name = i[0]
            sentences = i[1][0]
            valid_sentences = []
            processed_sentences = []
            for j in sentences:
                tp = proccesing(j, only_nouns, self.remove_sw, self.noun_list)
                value = tp.process()
                print value


        return ""


class PortugueseProcessing(object):

    def __init__(self, text, remove_nouns, remove_sw, noun_list=None):
        self.text = text
        self.remove_nouns = remove_nouns
        self.remove_sw = remove_sw
        self.noun_list = noun_list

    def remove_stop_words(self, text):
        text = text.lower()
        for c in string.punctuation:
            text = text.replace(c, "")
        text = ''.join([i for i in text if not i.isdigit()])
        stopwords = nltk.corpus.stopwords.words('portuguese')
        words = text.split()
        result = []
        stopsP = []
        for i in stopwords:
            i = unicodedata.normalize('NFKD', i).encode('ascii', 'ignore')
            stopsP.append(i)

        for word in words:
            if not word in stopsP:
                result.append(word)

        return result

    def lemas(self, words):
        result = []
        for i in words:
            result.append(lemma(i))
        return result

    def process(self):
        pSentence = []
        if self.remove_nouns:
            # print "removerrrr nouns"

            procesed = self.remove_stop_words(self.text)
            procesed = self.lemas(procesed)
            for i in procesed:
                if self.noun_list.has_key(i):
                    pSentence.append(i)

        else:
            if self.remove_sw:
                print "con stopwords removidos"
            else:
                print "incluyendo los stopwrods en el procesamiento"
        return pSentence
        #procesed = self.remove_stop_words(self.text)
        #print self.lemas(procesed)
        #return "procesado(ptg): " + self.text



class EnglishProcessing(object):

    def __init__(self, text, remove_nouns, remove_sw):
        self.text = text
        self.remove_nouns = remove_nouns
        self.remove_sw = remove_sw

    def process(self):
        return "procesado(eng): " + self.text


if __name__ == '__main__':

    noun_list = load_data_from_disk(extras['NounsList'])
    for i in noun_list:
        print i



