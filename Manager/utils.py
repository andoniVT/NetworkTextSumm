import codecs
import unicodedata
from nltk import word_tokenize , sent_tokenize
import string
import xml.etree.ElementTree as ET
import re
import cPickle
from random import shuffle, choice
from collections import Counter
from gensim import matutils
from scipy import spatial
from configuration import extras
from igraph import *
from subprocess import call
import subprocess
#import random
#random.choice
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from pylab import *
import csv

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
    limiar_type = None
    #distance = None
    size_d2v = None
    #inference_d2v = None
    inter_edge = None
    intra_edge = None

    '''
    dictionary['network'] = ('mln', ['noun', [1.7, 1.9], [0.4, 0.45, 0.5]]) # size:3
    dictionary['network'] = ('mln', ['tfidf', [1.7, 1.9], [0.4, 0.45, 0.5]])  # inter - limiar remocion  # size:3
    dictionary['network'] = ('mln', ['d2v', False, ('limiar', [0.3]), 300, [1.7, 1.9], [0.4, 0.45, 0.5]])  # size:6
    '''

    if network_type == 'mln':
        mln_type = data[0]
        if size_parameter == 6:
            sw_removal = data[1]
            limiar_type = data[2][0]
            limiar_value = data[2][1]
            size_d2v = data[3]
            inter_edge = data[4]
            intra_edge = data[5]
        else:
            inter_edge = data[1]
            intra_edge = data[2]
    else:
        # [False, ('limiar', [0.15]), 200]
        if size_parameter != 0:
            sw_removal = data[0] #ok
            limiar_type = data[1][0]
            limiar_value = data[1][1]
            size_d2v = data[2]
            #distance = data[2]
            #if size_parameter == 5:
            #    size_d2v = data[3]
            #    inference_d2v = data[4]

    parameters['mln_type'] = mln_type
    parameters['sw_removal'] = sw_removal
    parameters['limiar_type'] = limiar_type
    parameters['limiar_value'] = limiar_value
    #parameters['distance'] = distance
    parameters['size_d2v'] = size_d2v
    #parameters['inference_d2v'] = inference_d2v
    parameters['inter_edge'] = inter_edge
    #parameters['intra_edge'] = intra_edge
    parameters['limiar_mln'] = intra_edge

    return parameters

def read_document(file, language='ptg'):
    document = codecs.open(file, encoding="utf-8", errors='ignore')
    #document = codecs.open(file, encoding="utf-8")
    content = ""
    for i in document:
        i = i.rstrip()
        i = unicodedata.normalize('NFKD', i).encode('ascii', 'ignore')
        content += i + " "

    if language == 'ptg':
        sentences = sent_tokenize(content, language='portuguese')
    else:
        sentences = sent_tokenize(content, language='english')
    return sentences

def remove_portuguese_caracteres(sentence):
    news = []
    for word in sentence:
        news.append(unicodedata.normalize('NFKD', word).encode('ascii', 'ignore'))
    return news


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
    signos = '`"\''
    for i in sentences:
        i = i.replace('\n', ' ')
        for c in signos:
            i = i.replace(c, "")
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

#def calculate_similarity(vec_sentence1 , vec_sentence2, network_type, distance_method):
def calculate_similarity(vec_sentence1 , vec_sentence2, network_type):
    #if distance_method=='euc':
    #    return ['falta implementar']
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
    #random.choice
    print dictionary_rankings.keys()
    #key = random.choice(dictionary_rankings.keys())
    key = choice(dictionary_rankings.keys())

    dict_measures = dictionary_rankings[key]

    measures = []
    for i in dict_measures[0].items(): #### modificacion aqui ,
        measures.append(i[0])

    #measures.append('random')
    #measures.append('top')

    #if type is None:
    #    measures.append('top')
    index = 1

    for i in range(len(dict_measures)):
        for j in measures:
            path = extras['Automatics'] + str(index) + '/' + j
            if not os.path.exists(path):
                os.makedirs(path)
            #print path
        index+=1


    ''' 
    for i in measures:
        #path = "Automatic/" + i
        path = extras['Automatics'] + i
        print path
        #if not os.path.exists(path):
        #    os.makedirs(path)
    '''


'''
file = codecs.open('PRUEBA.txt',  'w', 'utf-8')
for i in sentences:
    file.write(i + '\n')
'''


def saveSummary(location, summary_sentences):
    file = open(location, 'w')
    #file = codecs.open(location, 'w', 'utf-8')
    for i in summary_sentences:
        file.write(i + "\n")


def summary_creation(resumo_name, selected_sentences, index):
    print "Generacion de sumarios en file"
    print resumo_name
    location = extras['Automatics'] + str(index) + '/'
    for i in selected_sentences.items():
        measure = i[0]
        sentences = i[1]
        path = location + measure + '/' + resumo_name
        print path
        saveSummary(path, sentences)



def summary_random_top_creation(resumo_name, sentences, resumo_size):
    print "Creando random y top baseline for SDS"
    ranking_top = [x for x in range(len(sentences))]
    ranking_random = ranking_top[:]
    shuffle(ranking_random)

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

def delete_dsStore(vector):
    special = '.DS_Store'
    if special in vector: vector.remove(special)
    return vector


def deleteFolders(location):
    files = os.listdir(location)
    files = delete_dsStore(files)
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

def vector_normalize(lista):
    normalized = []
    maximo = max(lista)
    for i in lista:
        value = i/float(maximo)
        normalized.append(value)
    return normalized

def assign_mln_weight(normalized, flag_list, inter, intra):
    weights = []
    for i in range(len(normalized)):
        if flag_list[i]:
            weights.append(normalized[i]*intra)
        else:
            weights.append(normalized[i]*inter)
    return weights

def generate_comparative_graphic(matrix, x):
    plt.plot(x, matrix[0], color='blue', linewidth=3.0)  ##4.0
    plt.plot(x, matrix[1], color='red', linewidth=3.0)
    plt.plot(x, matrix[2], color='darkgreen', linewidth=3.0)  ## 2.0
    plt.plot(x, matrix[3], color='black', linewidth=3.2)
    plt.plot(x, matrix[4], color='yellow', linewidth=3.3)  ## 6.0
    plt.plot(x, matrix[5], color='brown', linewidth=3.0)
    plt.plot(x, matrix[6], color='cyan', linewidth=3.0)  ## 8.0
    plt.plot(x, matrix[7], color='m', linewidth=3.3)
    plt.plot(x, matrix[8], color='y', linewidth=3.0)
    plt.plot(x, matrix[9], color='deeppink', linewidth=3.0)

    plt.ylim(0.450, 0.480)
    plt.legend(['dg', 'stg', 'pr', 'pr_w', 'sp', 'sp_w', 'gaccs', 'at', 'kats', 'btw'], loc='upper right')
    plt.xlabel('Number of removed edges (%)')
    # plt.xlabel('K (3-19)')
    # plt.xlabel('K (3-21)')
    plt.ylabel('Rouge Recall')
    plt.title('Portuguese SDS - Limiares')
    # plt.title('Portuguese SDS - Knn network')
    # plt.title('Portuguese MDS - Limiares')
    # plt.title('Portuguese MDS - Knn network')
    plt.show()


'''
def sort_results(matrix):
    pos = 0
    dictionary = dict()
    dictionary_positions = dict()
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
'''

def sort_recall_results(results):
    dictionary = dict()
    for i in results:
        element = i[0]
        dictionary[element[0]] = element[1]
    sorted_x = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_x



def generate_excel_simple(excel_name, results):
    print 'Generating Excel Simple Version'
    print excel_name
    results_sorted = sort_recall_results(results)
    first_row = ['Measurement' , 'Recall']
    myfile = open(excel_name, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(first_row)

    for i in results_sorted:
        to_write = [i[0], i[1]]
        wr.writerow(to_write)



def generate_excel_d2v_mln(excel_name, results, parameters_table):
    print 'Generating Excel Limiars and Inter-edges Version'
    print excel_name
    print results
    print parameters_table
    pesos_inter_edge_mln = parameters_table[0]
    limiares = parameters_table[1]

    first_row = ['Inter-edge Weight MLN' , 'Measurement']
    for i in limiares:
        first_row.append(str(i))

    pesos_mln_table = []


    if pesos_inter_edge_mln is not None:
        divisions = len(results) / len(pesos_inter_edge_mln)
        for i in pesos_inter_edge_mln:
            peso = i
            for j in range(divisions):
                pesos_mln_table.append(peso)
    else:
        pesos_mln_table = ['None' for x in results]

    print first_row
    myfile = open(excel_name, 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(first_row)

    for index, actual_result in enumerate(results):
        measure = actual_result[0][0]
        recalls = []
        for j in actual_result:
            recalls.append(j[1])

        write_line = [pesos_mln_table[index], measure]
        write_line.extend(recalls)
        wr.writerow(write_line)
        print write_line



if __name__ == '__main__':

    x = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    degree = [0.4728, 0.4686, 0.4643, 0.4679, 0.4665, 0.4655, 0.4682, 0.4659, 0.4618, 0.4605, 0.4621]
    strengt = [0.4528, 0.46, 0.4588, 0.4586, 0.4522, 0.4555, 0.4533, 0.4587, 0.4542, 0.4534, 0.4539]
    page_rank = [0.4572, 0.4615, 0.4594, 0.455, 0.461, 0.4655, 0.4586, 0.4621, 0.4618, 0.4558, 0.4622]
    page_rank_w = [0.4522, 0.4613, 0.4599, 0.4584, 0.4511, 0.4552, 0.4534, 0.4584, 0.4549, 0.4533, 0.4548]
    shortest_paths = [0.4739, 0.4702, 0.4663, 0.4679, 0.4666, 0.4661, 0.468, 0.4652, 0.462, 0.4606, 0.4621]
    shortest_paths_w = [0.4534, 0.4628, 0.4596, 0.4577, 0.451, 0.4561, 0.4532, 0.4584, 0.4542, 0.4525, 0.4542]
    btw = [0.4727, 0.4689, 0.4663, 0.4678, 0.469, 0.4667, 0.4677, 0.4642, 0.4617, 0.4602, 0.461]
    kats = [0.4592, 0.4602, 0.4609, 0.4589, 0.4539, 0.4633, 0.4545, 0.4549, 0.4618, 0.4589, 0.4643]
    generalized = [0.466, 0.47, 0.4689, 0.4717, 0.47, 0.468, 0.4773, 0.4729, 0.4725, 0.4752, 0.4767]
    at = [0.4562, 0.4629, 0.456, 0.456, 0.4561, 0.4543, 0.4588, 0.4593, 0.4576, 0.4529, 0.4584]


    matrix = [degree, strengt, page_rank, page_rank_w, shortest_paths, shortest_paths_w, btw, kats, generalized, at]

    generate_comparative_graphic(matrix, x)




















