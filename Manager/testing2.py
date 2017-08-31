
from configuration import references_dir , corpus_dir , extras
import os
from utils import read_document , cosineSimilarity , delete_dsStore, read_document_extract_cst , read_document_english, load_data_from_disk, write_data_to_disk
import numpy as np
import operator

def get_max_similarity(sentence, extractos):
    similarities = []
    for i in extractos:
        similarities.append(cosineSimilarity(sentence.split(), i.split()))
    return max(similarities)

'''
Creating dictionary of class labels for Ptg SDS with Temario for Machine Learning
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
    #print document_name
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

#write_data_to_disk(extras['PtgSDS_labels'], dict_temario)




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
Abstracts : 
'''

'''


print 'working!'

path_documents = corpus_dir['duc2002']
path_references = references_dir['duc2002_s']

clusters = os.listdir(path_documents)
clusters = delete_dsStore(clusters)

references = os.listdir(path_references)

anterior = 'null'

dictionary_references = dict()
for i in references:
    doc_sentences = read_document(path_references + i, 'eng')

    doc_name = i[:i.rfind('_')]
    actual = doc_name


    if anterior!=actual:
        dictionary_references[actual] = [doc_sentences]
    else:
        value = dictionary_references[actual]
        value.append(doc_sentences)
        dictionary_references[actual] = value

    anterior = actual

documents_sds_duc2002 = dict()

for i in clusters:
    cluster_name = i[:-1]
    #print cluster_name
    documents = os.listdir(path_documents + i)

    for j in documents:
        path_doc = path_documents + i + '/' + j
        document_sentences = read_document_english(path_doc) # vector de sentencias de un documento
        key = cluster_name + '-' + j
        #print key
        the_references = dictionary_references[key] # 1 o 2   vectores de sentencias de los extractos disponibles del documento
        dict1 = dict()
        dict2 = dict()
        vector_dicts = [dict1 , dict2]

        for k in document_sentences:
            sent = k
            index = 0
            for l in the_references:
                max_sim = get_max_similarity(sent, l)
                vector_dicts[index][sent] = max_sim
                index+=1


        dict_extract1 =  vector_dicts[0]
        dict_extract2 = vector_dicts[1]

        sorted_dict_ext1 = sorted(dict_extract1.items(), key=operator.itemgetter(1), reverse=True)
        if len(dict_extract2)!=0 :
            sorted_dict_ext2 = sorted(dict_extract2.items(), key=operator.itemgetter(1), reverse=True)
            size_extract2 = len(the_references[1])
        else:
            sorted_dict_ext2 = []
            size_extract2 = 0

        size_extract1 = len(the_references[0])


        selected1 = dict()
        for k in range(size_extract1):
            selected1[sorted_dict_ext1[k][0]] = sorted_dict_ext1[k][1]
        documents_sds_duc2002[key] = [selected1]


        if len(dict_extract2)!=0 :
            selected2 = dict()
            for k in range(size_extract2):
                selected2[sorted_dict_ext2[k][0]] = sorted_dict_ext2[k][1]
            value = documents_sds_duc2002[key]
            value.append(selected2)
            documents_sds_duc2002[key] = value

# cluster terminado
for i in documents_sds_duc2002.items():
    print len(i[1])
'''



'''
Creating dictionary of class labels for Eng MDS with DUC2002 for Machine Learning
Extracts : 
'''

'''
path_documents = corpus_dir['duc2002']
path_references_200 = corpus_dir['duc2002_extract_mds_200']
path_references_400 = corpus_dir['duc2002_extract_mds_400']

clusters = os.listdir(path_documents)
clusters = delete_dsStore(clusters)

references_200 = load_data_from_disk(path_references_200)
references_400 = load_data_from_disk(path_references_400)

documents_mds_duc2002 = dict()

for i in clusters:
    cluster_name = i[:-1]
    #print cluster_name
    documents = os.listdir(path_documents + i)
    allSentences = []
    for j in documents:
        path_doc = path_documents + i + '/' + j
        document_sentences = read_document_english(path_doc)  # vector de sentencias de un documento
        allSentences.extend(document_sentences)

    cluster_references_200 = references_200[cluster_name]  # vector de extracts1 y extracts2 de 200 words
    cluster_references_400 = references_400[cluster_name]  # vector de extracts1 y extracts2 de 400 words

    reference_sizes = []
    all_cluster_references = []
    all_cluster_references.append(cluster_references_200[0])
    reference_sizes.append(len(cluster_references_200[0]))
    if len(cluster_references_200)==2:
        all_cluster_references.append(cluster_references_200[1])
        reference_sizes.append(len(cluster_references_200[1]))
    else:
        all_cluster_references.append([])
        reference_sizes.append(0)
    all_cluster_references.append(cluster_references_400[0])
    reference_sizes.append(len(cluster_references_400[0]))
    if len(cluster_references_400)==2:
        all_cluster_references.append(cluster_references_400[1])
        reference_sizes.append(len(cluster_references_400[1]))
    else:
        all_cluster_references.append([])
        reference_sizes.append(0)


    vector_dicts = [dict(), dict(), dict(), dict()]

    for j in allSentences:
        sent = j
        index = 0
        for k in all_cluster_references:
            if len(k)!=0:
                max_sim = get_max_similarity(sent, k)
                vector_dicts[index][sent] = max_sim
            index+=1

    sorted_dictionaries = []
    index = 0
    for j in vector_dicts:
        if len(j)!=0:
            sorted_dict_ext = sorted(j.items(), key=operator.itemgetter(1), reverse=True)
            sorted_dictionaries.append(sorted_dict_ext)
        else:
            sorted_dictionaries.append([])
        index+=1

    index = 0
    final_selected = []
    for j in sorted_dictionaries:
        selected = dict()
        sort_dict = j
        for k in range(reference_sizes[index]):
            selected[sort_dict[k][0]] = sort_dict[k][1]
        final_selected.append(selected)
        index+=1


    documents_mds_duc2002[cluster_name] = final_selected


for i in documents_mds_duc2002.items():
    print i

'''



'''
Creating dictionary of class labels for Eng MDS with DUC2004 for Machine Learning
Abstracts : 
'''

'''
documents_mds_duc2004 = dict()

path_documents = corpus_dir['duc2004']
path_references = references_dir['duc2004_m']

references = os.listdir(path_references)
references = delete_dsStore(references)

dictionary_references = dict()
anterior = 'null'
for i in references:
    cluster_name = i[:i.find('_')]
    document_sentences = read_document(path_references + i, 'eng')
    actual = cluster_name
    if anterior!=actual:
        dictionary_references[actual] = [document_sentences]
    else:
        value = dictionary_references[actual]
        value.append(document_sentences)
        dictionary_references[actual] = value
    anterior = actual



cluster = os.listdir(path_documents)
cluster = delete_dsStore(cluster)
for i in cluster:
    cluster_name = i[:-1]
    print cluster_name
    documents = os.listdir(path_documents + i)
    allSentences = []
    for j in documents:
        document_sentences = read_document_english(path_documents + i + '/' + j)
        allSentences.extend(document_sentences)

    cluster_references = dictionary_references[cluster_name] # vector de 4  vectores de referencias
    cluster_sizes = []
    for i in cluster_references:
        cluster_sizes.append(len(i))


    vector_dicts = [dict(), dict(), dict(), dict()]

    
    for j in allSentences:
        sent = j
        index = 0
        for k in cluster_references:
            max_sim = get_max_similarity(sent, k)
            vector_dicts[index][sent] = max_sim
            index+=1


    sorted_dictionaries = []
    index = 0

    for j in vector_dicts:
        sorted_dict_ext = sorted(j.items(), key=operator.itemgetter(1), reverse=True)
        sorted_dictionaries.append(sorted_dict_ext)
        index+=1


    index = 0
    final_selected = []
    for j in sorted_dictionaries:
        selected = dict()
        sort_dict = j
        for k in range(cluster_sizes[index]):
            selected[sort_dict[k][0]] = sort_dict[k][1]
        final_selected.append(selected)
        index+=1


    documents_mds_duc2004[cluster_name] = final_selected


for i in documents_mds_duc2004.items():
    print i

'''









