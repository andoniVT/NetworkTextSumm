import matplotlib.pyplot as plt
import numpy as np
from pylab import *



def create_free_vectors(size):
	vector = []
	for i in range(size):
		vector.append([])
	return vector

def read_file(file):
	sin_pesos = ['dg', 'pr', 'sp', 'accs_h2' , 'gaccs', 'at', 'sym_l_b_h2', 'sym_h_b_h2']
	con_pesos = ['stg' , 'pr_w', 'sp_w']

	dictionary = dict()
	document = open(file, 'r')
	for i in document:
		i = i.rstrip('\n')
		i = i.rstrip('\r')
		datos =  i.split(',') 
		datos = datos[1:]
		measure = datos[0]
		results = datos[1:]
		results = [float(i) for i in results]
		if measure in con_pesos:
			if measure in dictionary:
				vector = dictionary[measure]
				vector[0].append(results[0])
				dictionary[measure] = vector
			else:
				vector = [[]]
				vector[0].append(results[0])
				dictionary[measure] = vector
		elif measure in sin_pesos:
			if measure in dictionary:
				vector = dictionary[measure]
				for j in range(len(results)):
					vector[j].append(results[j]) 
				dictionary[measure] = vector
			else:
				vector = create_free_vectors(5)
				for j in range(len(results)):
					vector[j].append(results[j]) 
				dictionary[measure] = vector
		else:
			pass

	return dictionary

def merge(set1, set2):
	merged = []
	for i , j in zip(set1, set2):
		if i > j:
			merged.append(i)
		else:
			merged.append(j)
	return merged


def merge_results(set_ribaldo, set_ngrams):
	dictionary = dict()
	for ribaldo, ngram  in zip(set_ribaldo.items(), set_ngrams.items()):
		measure = ribaldo[0]
		datos_ribaldo = ribaldo[1]
		datos_ngrams = ngram[1]
		datos_merged = []
		for d_r , d_n in zip(datos_ribaldo, datos_ngrams):
			merged = merge(d_r, d_n)
			datos_merged.append(merged)
		dictionary[measure] = datos_merged
	return dictionary 


def get_max_min(dictionary):
	maximos = []
	minimos = []

	for i in dictionary.items():
		coordenadas = i[1]
		for j in coordenadas:
			maximos.append(max(j))
			minimos.append(min(j))   
	
	return [max(maximos), min(minimos)] 


def analyze_non_weighted(measures, set_ribaldo, set_ngrams, language='ptg'): # measures maximo tamanio 4
	datos_merged = merge_results(set_ribaldo, set_ngrams)
	if language!='ptg':
		measures.remove('at')


	max_min_parameters = get_max_min(datos_merged)
	max_limit = max_min_parameters[0]
	min_limit = max_min_parameters[1]

	print max_limit , min_limit










if __name__ == '__main__':
	
	path_ribaldo_cst = 'CSTNews/cst_news_ribaldo.csv'
	path_ngrams_cst = 'CSTNews/cst_news_ngrams.csv'

	path_ribaldo_duc2002 = 'DUC2002/duc2002_ribaldo.csv'
	path_ngrams_duc2002 = 'DUC2002/duc2002_ngrams.csv'

	path_ribaldo_duc2004 = 'DUC2004/duc2004_ribaldo.csv'
	path_ngrams_duc2004 = 'DUC2004/duc2004_ngrams.csv'



	datos_ribaldo = read_file(path_ribaldo_cst)
	datos_ngrams = read_file(path_ngrams_cst)

	#datos_ribaldo = read_file(path_ribaldo_duc2002)
	#datos_ngrams = read_file(path_ngrams_duc2002)

	#datos_ribaldo = read_file(path_ribaldo_duc2004)
	#datos_ngrams = read_file(path_ngrams_duc2004)



	'''
	'dg', 'pr', 'sp', 'accs_h2' , 'gaccs', 'at', 'sym_l_b_h2', 'sym_h_b_h2'
	'stg' , 'pr_w', 'sp_w'
	'''

	measures = ['at', 'pr', 'sp', 'gaccs']


	analyze_non_weighted(measures, datos_ribaldo, datos_ngrams) # adicionar lenguahe ingles
	#analyze_non_weighted(measures, datos_ribaldo, datos_ngrams, language='en') # adicionar lenguahe ingles









