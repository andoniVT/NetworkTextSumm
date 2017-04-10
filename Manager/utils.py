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
from configuration import extras
from igraph import *
from subprocess import call


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


def sortList(vector):
    return [i[0] for i in sorted(enumerate(vector), key=lambda x:x[1])]


def specialSortList(vector):
    return vector[::-1]

def reverseSortList(vector):
    return [i[0] for i in sorted(enumerate(vector), key=lambda x:x[1], reverse=True)]


def average(lenghts):
    result = 0.0
    N = len(lenghts)
    for i in lenghts:
        if i == float('inf'):
            N-=1
        else:
            result+=i
    if result == 0:
        return 99999
    else:
        return result/N

'''
['dg', 'gaccs', 'accs_h2', 'ccts', 'sym']
['dg', 'pr', 'accs_h2', 'ccts_2_h2', 'sym_h_b_h3']
'''


def find_term(measures, parameter):
    for i in measures:
        if i.find(parameter)!=-1:
            return True
    return False



def manage_vector(measures, parameter):
    print "managing vector measures"
    #parameter = 'ccts'
    allConcentrics = parameter in measures
    if allConcentrics:
        print "todasssss"
        return measures

    others = []
    concentrics = ""

    for i in measures:
        if i.find(parameter)!=-1:
            i = i[i.find('_')+1:]
            concentrics+=i + "_"
        else:
            others.append(i)
    concentrics = concentrics[:-1]
    concentrics = parameter + '_' + concentrics
    others.append(concentrics)
    return others

def save_file(data, file_name):
	with codecs.open(file_name , "w" , "utf-8" , errors='replace') as temp:
		temp.write(data)

def generate_net(graph):
    location = extras['NetAux']
    graph.write_pajek(location)
    #print location


def generate_xnet(graph):
    result = "#vertices " + str(graph.vcount()) + " nonweighted \n"
    result = result + "#edges nonweighted undirected \n"
    lista = graph.get_edgelist()
    for i in lista:
        edge = str(i[0]) + " " + str(i[1]) + "\n"
        result = result + edge
    save_file(result, extras['XNetAux'])



def execute_concentric(command):
    # tener cuidado,, single ok ,, multi noseeeee
    print command
    sub_espace = command[command.find('..'):]
    sub_espace = sub_espace[:sub_espace.rfind('..') - 1]
    first_part = command[:command.find('..') - 1]
    second_part = command[command.rfind('..'):]
    values = first_part.split(' ')
    values.append(sub_espace)
    values.extend(second_part.split(' '))
    call(values)

def execute_symmetry(command):
    # os.system(command)  # cuando es singleeeeeeeeeeeeeeeee

    sub_espace = command[command.find('..'):]
    sub_espace = sub_espace[:sub_espace.rfind('..')-1]
    first_part = command[:command.find('..')-1]
    second_part = command[command.rfind('..'):]
    values = first_part.split(' ')
    values.append(sub_espace)
    values.append(second_part)
    call(values)


def  read_dat_file(file):
    h2 = []
    h3 = []
    file = open(file)
    for i in file:
        i = i.rstrip("\n")
        values = i.split(' ')
        values = values[:len(values)-1]
        if 2 < len(values):
            h2.append(float(values[2]))
        else:
            h2.append(0.0)
        if 3 < len(values):
            h3.append(float(values[3]))
        else:
            h3.append(0.0)
    return [h2, h3]

def read_dat_files():
    base = extras['FolderAux']
    result = []
    for i in range(8):
        file =  base + 'OutNet_hier' + str(i+1) + '.dat'
        result.append(read_dat_file(file))
    return result


def read_csv_file():
    base = extras['CSVAux']
    file = open(base, 'r')
    aux=0
    backbone_h2 = []
    merged_h2 = []
    backbone_h3 = []
    merged_h3 = []
    for i in file:
        i = i.rstrip("\n")
        i = " ".join(i.split())
        if aux!=0:
            values = i.split(" ")
            backbone_h2.append(float(values[0]))
            merged_h2.append(float(values[1]))
            backbone_h3.append(float(values[2]))
            merged_h3.append(float(values[3]))
        aux+=1
    return [[backbone_h2, backbone_h3] , [merged_h2,  merged_h3]]


def inverse_weights(weights):
    nuevo = []
    nuevo2 = []
    maximo = max(weights)
    for i in weights:
        if i==0:
            nuevo.append(0)
            nuevo2.append(0)
        else:
            nuevo.append(maximo-i+1)
            nuevo2.append(1/float(i))

    return [nuevo, nuevo2]





if __name__ == '__main__':

    haber = [0, 5, 8, 15, 19, 21, 23, 11, 7, 1, 12, 24, 4, 10, 9, 17, 6, 14, 22, 16, 18, 3, 20, 2, 13]
    #haber2 = [13, 2, 20, 3, 18, 16, 22, 14, 6, 17, 9, 10, 4, 24, 12, 1, 7, 11, 0, 5, 8, 15, 19, 21, 23]
    print specialSortList(haber)





