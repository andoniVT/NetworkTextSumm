
from configuration import references_dir , corpus_dir , extras
import os
from utils import read_document , cosineSimilarity , delete_dsStore, read_document_extract_cst
import numpy as np
import operator
'''
Creating dictionary of class labels for Ptg SDS with Temario for Machine Learning
'''

'''
path_referencias =  references_dir['temario_v1']
path_documents = corpus_dir['temario_v1']


dictionary_references = dict()
referencias = os.listdir(path_referencias)
for i in referencias:
    referencia = path_referencias + i
    clave = i[:i.rfind('_')]
    value = read_document(referencia, 'ptg')
    dictionary_references[clave] = value




documents = os.listdir(path_documents)
dict_temario = dict()
for i in documents:
    document = path_documents + '/' + i
    document_name = i[3:]
    document_name = document_name[:-4]
    print document_name
    document_sentences = read_document(document, 'ptg')
    reference_sentences = dictionary_references[document_name]
    dict_values = dict()

    for j in document_sentences:
        actual_sentence = j.split()
        similarities = []

        for k in reference_sentences:
            reference = k.split()
            #similarities[j] = cosineSimilarity(actual_sentence, reference)
            similarities.append(cosineSimilarity(actual_sentence, reference))

        dict_values[j] = max(similarities)
        #print round(max(similarities) , 2)

    sorted_dict = sorted(dict_values.items(), key=operator.itemgetter(1), reverse=True)
    size = len(reference_sentences)
    selected = dict()
    for j in range(size):
        selected[sorted_dict[j][0]] = sorted_dict[j][1]

    #for j in selected.items():
    #    print j


    dict_temario[document_name] = selected

#print cosineSimilarity(['hola' , 'que' , 'tal'] , ['holaa' , 'aque' , 'taal'])
for i in dict_temario.items():
    print i 
'''




'''
Creating dictionary of class labels for Ptg MDS with CSTNews for Machine Learning
Verificar algunos documentos no captura todas las sentencas
'''

'''
path_documents_references = corpus_dir['cstnews_v1']

cluster = os.listdir(path_documents_references)
cluster = delete_dsStore(cluster)
dictionary_cst = dict()

for i in cluster:
    path_documents = path_documents_references + i + '/' + corpus_dir['textosFonte']
    path_reference = path_documents_references + i + '/'  + corpus_dir['cst_extrato']

    documents = os.listdir(path_documents)
    cluster_sentences = []
    print i
    prefix = i[:i.find('_')]
    path_summary = path_reference + prefix + corpus_dir['cst_extrato_name']
    reference_sentences = read_document_extract_cst(path_summary , 'ptg')

    for j in documents:
        sentences = read_document(path_documents + j , 'ptg')
        cluster_sentences.extend(sentences)


    auxi = 0
    dict_sentences = dict()
    for j in cluster_sentences:
        if j in reference_sentences:
            dict_sentences[j] = 0


    print path_summary
    dictionary_cst[i] = dict_sentences


for i in dictionary_cst.items():
    print i

'''



'''
Creating dictionary of class labels for Eng SDS with DUC2002 for Machine Learning
'''

print 'working!'