import codecs
import unicodedata
from nltk import word_tokenize , sent_tokenize
import string
import xml.etree.ElementTree as ET
import re
import cPickle
from random import shuffle
from collections import Counter
from gensim import matutils
from scipy import spatial

def write_data_to_disk(file, data):
    with open(file, 'wb') as fid:
        cPickle.dump(data, fid)

def load_data_from_disk(file):
    with open(file, 'rb') as fid:
        data = cPickle.load(fid)
    return data

def parameter_extractor(network_type, data):
    parameters = dict()
    size_parameter = len(data)

    mln_type = None
    sw_removal = None
    limiar_value = None
    distance = None
    size_d2v = None
    inference_d2v = None
    inter_edge = None
    intra_edge = None

    if network_type == 'mln':
        mln_type = data[0]
        if size_parameter > 3:
            sw_removal = data[1]
            limiar_value = data[2]
            distance = data[3]
            if size_parameter == 6:
                inter_edge = data[4]
                intra_edge = data[5]
            else:
                size_d2v = data[4]
                inference_d2v = data[5]
                inter_edge = data[6]
                intra_edge = data[7]
        else:
            inter_edge = data[1]
            intra_edge = data[2]
    else:
        if size_parameter != 0:
            sw_removal = data[0]
            limiar_value = data[1]
            distance = data[2]
            if size_parameter == 5:
                size_d2v = data[3]
                inference_d2v = data[4]

    parameters['mln_type'] = mln_type
    parameters['sw_removal'] = sw_removal
    parameters['limiar_value'] = limiar_value
    parameters['distance'] = distance
    parameters['size_d2v'] = size_d2v
    parameters['inference_d2v'] = inference_d2v
    parameters['inter_edge'] = inter_edge
    parameters['intra_edge'] = intra_edge

    return parameters

def read_document(file, language='ptg'):
	document = codecs.open(file, encoding="utf-8", errors='ignore')
	content = ""
	for i in document:
		i = i.rstrip()
		i = unicodedata.normalize('NFKD', i).encode('ascii', 'ignore')
		content+=i + " "

	if language == 'ptg':
		sentences = sent_tokenize(content, language='portuguese')
	else:
		sentences = sent_tokenize(content, language='english')
	return sentences


def wordCountString(source):
    for c in string.punctuation:
        source =source.replace(c, "")
    return len(word_tokenize(source))


def count_words(file, language):
	sentences = read_document(file, language)
	words=0
	for i in sentences:
		words+= wordCountString(i)
	return words


def clean_sentences(sentences):
    result = []
    for i in sentences:
        i = i.replace('\n', ' ')
        result.append(i)
    return result

def read_document_english(document):
    data = ""
    tree = ET.parse(document)
    root = tree.getroot()
    for i in root.iter('TEXT'):
        data+= i.text + " "
    data = re.sub("\s\s+", " ", data)

    sentences = sent_tokenize(data)
    sentences = clean_sentences(sentences)
    return sentences

def permutate_data(data):
    shuffle(data)
    return data

def has_common_elements(vec, vec2):
    value = 0
    for i in vec:
        if i in vec2:
            value+=1
    return value


def cosineSimilarity(sentence1, sentence2):
   a_vals = Counter(sentence1)
   b_vals = Counter(sentence2)
   words = list(set(a_vals) | set(b_vals))
   a_vect = [a_vals.get(word, 0) for word in words]
   b_vect = [b_vals.get(word, 0) for word in words]
   len_a = sum(av * av for av in a_vect) ** 0.5
   len_b = sum(bv * bv for bv in b_vect) ** 0.5
   dot = sum(av * bv for av, bv in zip(a_vect, b_vect))
   cosine = dot / (len_a * len_b)
   return cosine

'''
        return matutils.cossim(vec_tfidf, vec_tfidf2)  gemsim
        from scipy import spatial
        return 1 - spatial.distance.cosine(vec_sentence1, vec_sentence2)  doc2vec
        '''

def calculate_similarity(vec_sentence1 , vec_sentence2, network_type, distance_method):
    if distance_method=='euc':
        return ['falta implementar']
    if network_type=='tfidf':
        return matutils.cossim(vec_sentence1, vec_sentence2)
    if network_type=='d2v':
        return 1 - spatial.distance.cosine(vec_sentence1, vec_sentence2)


if __name__ == '__main__':

    pass