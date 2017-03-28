
class Loader(object):

    def __init__(self, language, type_summary, corpus, size, mln):
        self.dictionary = dict()
        self.language = language
        self.type_summary = type_summary
        self.corpus = corpus
        self.size = size
        self.mln = mln
        self.dictionary['ptg'] = [['temario_v1', 'temario_v2'], ['cstnews_v1', 'cstnews_v2']]
        self.dictionary['eng'] = [['duc2002', 'duc2003'], ['duc2002', 'duc2003']]

    def load(self):
        # print self.dictionary[self.language][self.type_summary][self.corpus]
        corpus = ['diccionario con nombres y los datos', 'loaded corpus sin procesar', 'vectores de sizes de sumarios']
        return corpus



if __name__ == '__main__':

    print "haberr"