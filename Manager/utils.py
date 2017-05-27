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
import subprocess
import random
import os
import shutil

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
    #document = codecs.open(file, encoding="utf-8", errors='ignore')
    document = codecs.open(file, encoding="utf-8")
    content = ""
    for i in document:
        i = i.rstrip()
        #i = unicodedata.normalize('NFKD', i).encode('ascii', 'ignore')
        content += i + " "

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
    #print command
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

def get_terminal_values(command):
    #values = command.split(' ')  ### cuando es single !!!!!!!!!!
    sub_space = command[command.rfind('/')+1:]
    sub_normal = command[:command.rfind('/')+1]
    values = sub_normal.split(' ')
    values[3] = values[3] + sub_space
    output = subprocess.Popen(values, stdout=subprocess.PIPE).communicate()[0]
    return output


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


def remove_punctuation(sentence):
    for c in string.punctuation:
        sentence = sentence.replace(c,"")
    return sentence


def selectSentencesSingle(sentences, measures, resumo_size):
    limit = 0
    result = []
    name_measure = measures[0]
    ranked = measures[1]
    for index, sents in enumerate(sentences):
        index_selected = ranked[index]
        selected = sentences[index_selected]
        result.append(selected)
        selected = remove_punctuation(selected)
        #result.append(selected)
        limit+=len(word_tokenize(selected))
        if limit > resumo_size:
            break
    return (name_measure,result)

def isRedundant(index, psentences, selected, limit):
    actual_sentence = psentences[index]
    for i in selected:
        sentence = psentences[i]
        similarity = cosineSimilarity(actual_sentence, sentence)  ##### verificar si aplicando los vectores ya calculado faz diferencia
        if similarity > limit:
            return True
    return False

def extract_only_sentences(sentences):
    result = []
    for i in sentences:
        result.append(i[0])
    return result


def selectSentencesMulti_ribaldo(sentences, ranking, resumo_size, threshold, pSentences):
    selected = []
    name_measure = ranking[0]
    ranked = ranking[1]
    initial_index = ranked[0]
    selected.append(initial_index)
    sentences_sin_punct = remove_punctuation(sentences[initial_index])
    pSentences = extract_only_sentences(pSentences)

    size_sentence = len(word_tokenize(sentences_sin_punct))
    for i in range(1, len(ranked)):
        index = ranked[i]
        if not isRedundant(index, pSentences, selected, threshold):
            selected.append(index)
            #selected = remove_punctuation(selected)
            #auxi = sentences[index]
            sentences_sin_punct = remove_punctuation(sentences[index])
            #size_sentence+= len(word_tokenize(sentences[index]))
            size_sentence += len(word_tokenize(sentences_sin_punct))

        if size_sentence > resumo_size:
            break

    selected_sentences = []
    for i in range(len(selected)):
        index = selected[i]
        sentence = sentences[index]
        selected_sentences.append(sentence)

    return (name_measure, selected_sentences)


def selectSentencesMulti(sentences, ranking, resumo_size, anti_redundancy, threshold, pSentences):
    #print ranking
    if anti_redundancy==0:
        #print "Seleccion sin anti-redundancia"
        return selectSentencesSingle(sentences, ranking, resumo_size)
    elif anti_redundancy==1:
        #print "Seleccion Rivaldo"
        return selectSentencesMulti_ribaldo(sentences, ranking, resumo_size, threshold, pSentences)

    elif anti_redundancy==2:
        print "Seleccion MMR"




def folder_creation(dictionary_rankings, type):
    key = random.choice(dictionary_rankings.keys())
    dict_measures = dictionary_rankings[key]
    measures = []
    for i in dict_measures.items():
        measures.append(i[0])

    measures.append('random')
    measures.append('top')

    #if type is None:
    #    measures.append('top')

    for i in measures:
        #path = "Automatic/" + i
        path = extras['Automatics'] + i
        if not os.path.exists(path):
            os.makedirs(path)


def saveSummary(location, summary_sentences):
    file = open(location, 'w')
    for i in summary_sentences:
        file.write(i + "\n")


def summary_creation(resumo_name, selected_sentences):
    print "Generacion de sumarios en file"
    print resumo_name
    location = extras['Automatics']
    for i in selected_sentences.items():
        measure = i[0]
        sentences = i[1]
        path = location + measure + '/' + resumo_name
        saveSummary(path, sentences)



def summary_random_top_creation(resumo_name, sentences, resumo_size):
    print "Creando random y top baseline for SDS"
    ranking_top = [x for x in range(len(sentences))]
    ranking_random = ranking_top[:]
    random.shuffle(ranking_random)

    #path = "Automatic/top/" + resumo_name
    path = extras['Automatics'] + 'top/' + resumo_name
    path2 = extras['Automatics'] + 'random/' + resumo_name

    measures = ('top' , ranking_top)
    sentences_top = selectSentencesSingle(sentences, measures, resumo_size)[1]

    measures2 = ('random', ranking_random)
    sentences_random = selectSentencesSingle(sentences, measures2, resumo_size)[1]

    saveSummary(path, sentences_top)
    saveSummary(path2, sentences_random)



def summary_random_top_creation_mds(resumo_name, sentences, resumo_size, top_sentences):
    print "Creando random y top baseline for MDS"
    ranking_top = [x for x in range(len(sentences))]
    ranking_random = ranking_top[:]
    random.shuffle(ranking_random)
    path = extras['Automatics'] + 'random/' + resumo_name
    measures = ('random', ranking_random)
    sentences_random = selectSentencesSingle(sentences, measures, resumo_size)[1]
    saveSummary(path, sentences_random)

    ranking_top2 = [x for x in range(len(top_sentences))]
    ranking_random2 = ranking_top2[:]
    random.shuffle(ranking_random2)
    path2 = extras['Automatics'] + 'top/' + resumo_name
    measures2 = ('top', ranking_random2)
    sentences_top = selectSentencesSingle(top_sentences, measures2, resumo_size)[1]
    saveSummary(path2, sentences_top)




def deleteFiles(type):
    files = os.listdir(type)
    for f in files:
        os.remove(type +f)

def deleteFolders(location):
    files = os.listdir(location)
    for f in files:
        shutil.rmtree(location + f)




def get_csv_values(file):
    avg_precision = 0
    avg_recall = 0
    avg_fmeasure = 0
    doc = open(file, 'r')
    index = 0
    for i in doc:
        if index != 0:
            fila = i.split(',')
            recall = float('0.' + fila[4])
            precision = float('0.' + fila[6])
            fmeasure = float('0.' + fila[8])
            avg_recall += recall
            avg_precision += precision
            avg_fmeasure += fmeasure
        index += 1
    print index
    return round(avg_precision / (index - 1), 4), round(avg_recall / (index - 1), 4), round(avg_fmeasure / (index - 1),
                                                                                            4)


def sort_results(matrix):
    dictionary = dict()
    dictionary_positions = dict()
    pos = 0
    for i in matrix:
        dictionary[i[0]] = i[2]
        dictionary_positions[i[0]] = pos
        pos+=1
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    ordered_matrix = []
    titles = ['MEASURE', 'P', 'R', 'F']
    ordered_matrix.append(titles)
    for i in sorted_x:
        key = i[0]
        position = dictionary_positions[key]
        ordered_matrix.append(matrix[position])

    return ordered_matrix

def sort_network(edges, weights):
	dictionary = dict()
	for index,  edge in  enumerate(edges):
		key = str(edge[0]) + '-' +str(edge[1])
		dictionary[key] = weights[index]
	sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
	return sorted_x


def draw_graph(g):
    size  = g.vcount()
    vec = [x for x in range(size)]
    layout = g.layout("kk")
    visual_style = {}
    visual_style["vertex_label"] = vec
    visual_style["vertex_size"] = 15
    visual_style["layout"] = layout
    visual_style["bbox"] = (800, 600)
    visual_style["margin"] = 70
    plot(g, **visual_style)


def get_weights(edgesList, weight_list):
    dictionary = dict()
    for index in range(len(edgesList)):
        edge = edgesList[index]
        weight = weight_list[index]
        key = str(edge[0]) + '-' + str(edge[1])
        dictionary[key] = weight
    return dictionary


def tag_sentence(document_sentences, index):
    tagged = []
    for i in document_sentences:
        tagged.append((i , index))
    return tagged

def naive_tag(document_sentences):
    tagged = []
    for i in document_sentences:
        tagged.append((i, None))
    return tagged




if __name__ == '__main__':

    matrix = [['sp_w2', '0.434', '0.473', '0.4523'], ['top', '0.4421', '0.4757', '0.458'],
              ['random', '0.4261', '0.4594', '0.4417'], ['stg', '0.4346', '0.4734', '0.4528'],
              ['sp_w', '0.4346', '0.4729', '0.4525'], ['sp', '0.4394', '0.475', '0.4561'],
              ['dg', '0.4382', '0.4748', '0.4553']]


    for i in  matrix:
        print i

    print ""
    sort_results(matrix)


















