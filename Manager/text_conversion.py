import string
import unicodedata
import nltk
from morphological_analysis import lemma
from configuration import extras
from utils import load_data_from_disk
from nltk import word_tokenize
from  nltk.stem.wordnet import WordNetLemmatizer
noun_list = []

class CorpusConversion(object):

    def __init__(self, corpus, language, network_type, mln_type, remove_sw):
        self.corpus = corpus
        self.language = language
        self.network_type = network_type
        self.mln_type = mln_type
        self.remove_sw = remove_sw
        self.noun_list = load_data_from_disk(extras['NounsList'])
        #self.not_noun_list = load_data_from_disk(extras['NotNounsList'])
        self.not_noun_list = load_data_from_disk(extras['NotNounsList_v2'])


        #print len(self.not_noun_list)
        #a = input()

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
                #tp = proccesing(j, only_nouns, self.remove_sw, self.noun_list) #########

                sentence = j[0]  ##### modificadaaaaaa para MLN aun no seee
                group_sent_id = j[1]
                tp = proccesing(sentence, only_nouns, self.remove_sw, self.not_noun_list) ## modificadaa

                value = tp.process()
                if len(value)!=0:
                    processed_sentences.append((value, group_sent_id))  # adicionar id de documento , o valor nulo , dependiendo
                    #valid_sentences.append(j)
                    valid_sentences.append(sentence)

            corpus_data = self.corpus[doc_name]
            sizes = corpus_data[1] #### ahi difiere el corpus para ptg y el corpus para ingles (el corpus de ingles son 100 o 200)
            processed_corpus[doc_name] = [valid_sentences, processed_sentences, sizes]


        return processed_corpus


class PortugueseProcessing(object):

    def __init__(self, text, remove_nouns, remove_sw, noun_list=None):
        self.text = text
        self.remove_nouns = remove_nouns
        self.remove_sw = remove_sw
        #self.noun_list = noun_list
        self.not_noun_list = noun_list

    def remove_unused(self, text): # aqui se decide si se remove o no los stopwords
        text = text.lower()
        for c in string.punctuation:
            text = text.replace(c, "")
        text = ''.join([i for i in text if not i.isdigit()])

        if self.remove_sw == False:
            return word_tokenize(text)

        stopwords = nltk.corpus.stopwords.words('portuguese')
        words = text.split()
        result = []
        #stopsP = []
        #for i in stopwords:
        #    #i = unicodedata.normalize('NFKD', i).encode('ascii', 'ignore')
        #    stopsP.append(i)





        for word in words:
            #if not word in stopsP:
            if not word in stopwords:
                result.append(word)

        return result

    def lemas(self, words):
        result = []
        for i in words:
            result.append(lemma(i))
        return result

    def process(self):
        pSentence = []
        procesed = self.remove_unused(self.text)
        procesed = self.lemas(procesed)
        if self.remove_nouns: # print "removerrrr nouns"
            for i in procesed:
                #if self.noun_list.has_key(i):
                if i not in self.not_noun_list:
                    pSentence.append(i)
        else:
            pSentence = procesed
            '''
            if self.remove_sw:
                #print "con stopwords removidos"
                procesed = self.remove_stop_words(self.text)
                procesed = self.lemas(procesed)
                pSentence = procesed

            else:
                #print "incluyendo los stopwrods en el procesamiento"
                procesed = self.special_removal(self.text)
                procesed = self.lemas(procesed)
                pSentence = procesed
            '''
        return pSentence


class EnglishProcessing(object):

    def __init__(self, text, remove_nouns, remove_sw, noun_list=None):
        self.text = text
        self.remove_nouns = remove_nouns
        self.remove_sw = remove_sw


    def remove_unused(self, text):
        text = text.lower()
        for c in string.punctuation:
            text = text.replace(c, '')

        text = ''.join([i for i in text if not i.isdigit()])

        if self.remove_sw == False:
            return word_tokenize(text)

        stopwords = nltk.corpus.stopwords.words('english')
        words = text.split()

        result = []

        for word in words:
            if not word in stopwords:
                result.append(word)
        return result

    def process(self):
        pSentence = []
        tags = ['NN', 'NNP', 'NNPS', 'NNS']
        verbTags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        #tagged = nltk.pos_tag(text)
        procesed = self.remove_unused(self.text)
        tagged = nltk.pos_tag(procesed)

        if self.remove_nouns:
            for i in tagged:
                if i[1] in tags:
                    pSentence.append(WordNetLemmatizer().lemmatize(i[0], 'n') + " ")
        else:
            for i in tagged:
                if i[1] in verbTags:
                    pSentence.append(WordNetLemmatizer().lemmatize(i[0], 'v') + " ")
                else:
                    pSentence.append(WordNetLemmatizer().lemmatize(i[0]) + " ")

        return pSentence
        #return "procesado(eng): " + self.text



if __name__ == '__main__':

    noun_list = load_data_from_disk(extras['NounsList'])
    for i in noun_list:
        print i



