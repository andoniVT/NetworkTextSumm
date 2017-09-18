import matplotlib.pyplot as plt
import numpy as np
from pylab import *
 


def create_free_vectors(size):
	vector = []
	for i in range(size):
		vector.append([])
	return vector


def read_file(file):
	sin_pesos = ['at' , 'dg', 'gaccs', 'pr', 'sp']
	con_pesos = ['pr_w', 'sp_w', 'stg']

	symmetry = ['sym_l_b_h2', 'sym_l_m_h2', 'sym_h_b_h3', 'sym_h_m_h3',
	'sym_h_m_h2', 'sym_l_b_h3', 'sym_l_m_h3', 'sym_h_b_h2']

	accessibility = ['accs_h2' , 'accs_h3']

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
		elif measure in sin_pesos or measure in symmetry or measure in accessibility:
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



def draw(measure, datos):
	coordenadas = datos[measure]
	dictionary = dict()
	dictionary['pr'] = ('Page Rank' , True)
	dictionary['pr_w'] = ('Page Rank Weights' , False )
	dictionary['dg'] = ('Degree' , True)
	dictionary['stg'] = ('Strenght', False)
	dictionary['sp'] = ('Shortest Path', True)
	dictionary['sp_w'] = ('Shortest Path Weights', False)
	dictionary['at'] = ('Absorption Time', True)
	dictionary['gaccs'] = ('Generalized Accessibility', True) 


	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]

	legends = ['10%', '20%', '30%', '40%', '50%']

	colors = ['blue', 'red', 'darkgreen', 'yellow', 'cyan']


	for index, coordenada in enumerate(coordenadas):
		plt.plot(x, coordenada, color=colors[index], linewidth=3.0)


	#plt.plot(x, prueba, color='blue', linewidth=3.0)



	if dictionary[measure][1]:
		plt.legend(legends, loc='upper right')



	plt.xlabel('Inter-edge weight')
	plt.ylabel('Rouge Recall')

	plt.title('MLN-TfIdf ' + dictionary[measure][0])

	
	plt.show()


def draw_weighted(datos, datos2):
	measures = ['pr_w', 'stg', 'sp_w'] 
	coordenadas = []
	coordenadas2 = []
	for i in measures:
		coordenadas.append(datos[i][0])
		coordenadas2.append(datos2[i][0])

	maximos = []
	minimos = []
	for i, j in zip(coordenadas, coordenadas2):
		maximos.append(max(i))
		maximos.append(max(j)) 
		minimos.append(min(i))
		minimos.append(min(j))

	max_limit = max(maximos) 
	min_limit = min(minimos) 
	
	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]
	#legends = measures
	legends = ['Page Rank W.' , 'Strength' , 'Shortest Path W.']
	colors = ['blue', 'red', 'darkgreen']

	f, axarr = plt.subplots(1, 2)


	for index, coordenada in enumerate(coordenadas):
		axarr[0].plot(x, coordenada, color=colors[index], linewidth=3.0)
	axarr[0].set_title('MLN-TfIdf ARD-Ribaldo')
	axarr[0].legend(legends, loc='upper right')
	axarr[0].set_xlabel('Inter-edge weight')
	axarr[0].set_ylabel('Rouge Recall')
	axarr[0].set_ylim(min_limit, max_limit)
	


	for index, coordenada in enumerate(coordenadas2):
		axarr[1].plot(x, coordenada, color=colors[index], linewidth=3.0)
	axarr[1].set_title('MLN-TfIdf ARD-Ngrams')
	axarr[1].legend(legends, loc='upper right')
	axarr[1].set_xlabel('Inter-edge weight')
	axarr[1].set_ylabel('Rouge Recall')
	axarr[1].set_ylim(min_limit, max_limit)
	
	'''
	for index, coordenada in enumerate(coordenadas):
		plt.plot(x, coordenada, color=colors[index], linewidth=3.0)
	'''	
	#plt.xlabel('Inter-edge weight')
	#plt.ylabel('Rouge Recall')
	
	plt.show()


def draw_four_non_weighted(exclude, datos, datos2):
	measures = ['pr' ,'dg' ,'sp', 'at' ,'gaccs']
	measures.remove(exclude) ####### tener cuidadooooooo
	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]
	colors = ['blue', 'red', 'darkgreen', 'yellow', 'cyan']
	dictionary = dict()
	dictionary['pr'] = 'Page Rank' 
	dictionary['dg'] = 'Degree' 
	dictionary['sp'] = 'Shortest Path'
	dictionary['at'] = 'Absorption Time'
	dictionary['gaccs'] = 'Generalized Accessibility'


	maximos = []
	minimos = []

	for index, measure in enumerate(measures):
		coordenadas = datos[measure]
		coordenadas2 = datos[measure]
		for i , j in zip(coordenadas, coordenadas2):
			maximos.append(max(i))
			minimos.append(min(i)) 

			maximos.append(max(j))
			minimos.append(min(j)) 


	max_limit = max(maximos) 
	min_limit = min(minimos) 


	#measures.remove(exclude) ######## tener cuidadooooo


	legends = ['10%', '20%', '30%', '40%', '50%']


	f, axarr = plt.subplots(2, 2)
	axes = [axarr[0,0] , axarr[0,1], axarr[1,0], axarr[1,1]]
	for index, measure in enumerate(measures):
		coordenadas = datos2[measure] #############
		for index2, coordenada in enumerate(coordenadas):
			axes[index].plot(x, coordenada, color=colors[index2], linewidth=3.0)
		axes[index].set_title(dictionary[measure])
		axes[index].set_ylim(min_limit, max_limit)
		axes[index].legend(legends, loc='upper right')
		axes[index].set_ylabel('Rouge Recall')
	
	plt.show()
	


def symmetry_analysis(type_measure, datos):  # mg bb h2 h3 l h 
	symmetry = ['sym_L_b_h2', 'sym_L_m_h2', 'sym_H_b_h3', 'sym_H_m_h3', 'sym_H_m_h2', 
	'sym_L_b_h3', 'sym_L_m_h3', 'sym_H_b_h2']


	maximos = []
	minimos = []

	for index, measure in enumerate(symmetry):
		coordenadas = datos[measure.lower()]
		for i in coordenadas:
			maximos.append(max(i))
			minimos.append(min(i))

	max_limit = max(maximos) 
	min_limit = min(minimos)


	parameter = '_' + type_measure
	measures = []

	for i in symmetry:
		if i.find(parameter)!=-1:
			measures.append(i.lower())


	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]  

	colors = ['blue', 'red', 'darkgreen', 'yellow', 'cyan']
	
	legends = ['10%', '20%', '30%', '40%', '50%']

	f, axarr = plt.subplots(2, 2)
	axes = [axarr[0,0] , axarr[0,1], axarr[1,0], axarr[1,1]]

	for index, measure in enumerate(measures):
		coordenadas = datos[measure]
		for index2 , coordenada in enumerate(coordenadas):
			axes[index].plot(x, coordenada, color=colors[index2], linewidth=3.0)
		axes[index].set_title(measure)
		axes[index].set_ylim(min_limit, max_limit)
		axes[index].legend(legends, loc='upper right')
		axes[index].set_ylabel('Rouge Recall')

	plt.show()


def symmetry_second_analysis(datos):
	symmetry = ['sym_l_b_h2', 'sym_l_m_h2', 'sym_l_b_h3', 'sym_l_m_h3']

	maximos = []
	minimos = []

	for index, measure in enumerate(symmetry):
		coordenadas = datos[measure.lower()]
		for i in coordenadas:
			maximos.append(max(i))
			minimos.append(min(i))

	max_limit = max(maximos) 
	min_limit = min(minimos)
	
	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]  
	colors = ['blue', 'red', 'darkgreen', 'yellow', 'cyan']
	legends = ['10%', '20%', '30%', '40%', '50%']

	f, axarr = plt.subplots(2, 2)
	axes = [axarr[0,0] , axarr[0,1], axarr[1,0], axarr[1,1]]

	for index, measure in enumerate(symmetry):
		coordenadas = datos[measure]
		for index2 , coordenada in enumerate(coordenadas):
			axes[index].plot(x, coordenada, color=colors[index2], linewidth=3.0)
		axes[index].set_title(measure)
		axes[index].set_ylim(min_limit, max_limit)
		axes[index].legend(legends, loc='upper right')
		axes[index].set_ylabel('Rouge Recall')

	plt.show()








def accessibility_analysis(datos):
	measures = ['accs_h2' , 'accs_h3'] 
	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]
	colors = ['blue', 'red', 'darkgreen', 'yellow', 'cyan']

	f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

	axes = [ax1, ax2]

	for index, measure in enumerate(measures):
		coordenadas = datos[measure]
		title = measure
		for index2 , coordenada in enumerate(coordenadas):
			axes[index].plot(x, coordenada, color=colors[index2], linewidth=3.0)
		axes[index].set_title(title)

	legends = ['10%', '20%', '30%', '40%', '50%']
	plt.legend(legends, loc='upper right')


	plt.show()









def draw_five_non_weighted(datos):
	measures = ['pr' ,'dg' ,'sp', 'at' ,'gaccs']	
	x = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.3, 1.5, 1.7, 1.9]
	colors = ['blue', 'red', 'darkgreen', 'yellow', 'cyan']
	dictionary = dict()
	dictionary['pr'] = 'Page Rank' 
	dictionary['dg'] = 'Degree' 
	dictionary['sp'] = 'Shortest Path'
	dictionary['at'] = 'Absorption Time'
	dictionary['gaccs'] = 'Generalized Accessibility'


	f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True)

	axes = [ax1, ax2, ax3, ax4, ax5]

	for index, measure in enumerate(measures):
		coordenadas = datos[measure]
		title = dictionary[measure]
		for index2 , coordenada in enumerate(coordenadas):
			axes[index].plot(x, coordenada, color=colors[index2], linewidth=3.0)
		axes[index].set_title(title)

	legends = ['10%', '20%', '30%', '40%', '50%']
	plt.legend(legends, loc='upper right')

	plt.show()

def paper():
	dg = [0.5469, 0.5482, 0.5400, 0.5528]
	stg = [0.5453, 0.5390, 0.5433, 0.5552]
	sp = [0.5441, 0.5438, 0.5432, 0.5509]
	sp_w = [0.5346, 0.5478, 0.5454, 0.5636]
	sp_w2 = [0.5417, 0.5314, 0.5545, 0.5515]
	btw = [0.5298, 0.5404, 0.5341, 0.5452]
	btw_w = [0.4763, 0.4745, 0.4901, 0.4790]
	pr = [0.5501, 0.5367, 0.5426, 0.5435]
	pr_w = [0.5458, 0.5460, 0.5471, 0.5605]
	cc = [0.4151, 0.4266, 0.4270, 0.4424]
	cc_w = [0.4180, 0.4326, 0.4337, 0.4532]
	conc1 = [0.3999, 0.3957, 0.4171, 0.4083]
	conc2 = [0.3943, 0.3895, 0.4157, 0.4057]
	conc3 = [0.4035, 0.4095, 0.4246, 0.4187]
	conc4 = [0.3919, 0.3858, 0.4115, 0.4068]
	conc5 = [0.4204, 0.4214, 0.4376, 0.4324]
	conc6 = [0.4077, 0.4259, 0.4235, 0.4393]
	conc7 = [0.3989, 0.3730, 0.4116, 0.3934]
	conc8 = [0.4179, 0.4276, 0.4283, 0.4432]
	access_h2 = [0.4925, 0.5093, 0.5032, 0.5102]
	access_h3 = [0.4484, 0.4302, 0.4540, 0.4369]
	gaccs = [0.5489, 0.5478, 0.5395, 0.5494]
	hsymbb_h2 = [0.4183, 0.4202, 0.4242, 0.4228]
	hsymbb_h3 = [0.4010, 0.4307, 0.4200, 0.4438]
	hsymmg_h2 = [0.4745, 0.4856, 0.4829, 0.4906]
	hsymmg_h3 = [0.4525, 0.4621, 0.4591, 0.4744]
	lsymbb_h2 = [0.5207, 0.5302, 0.5288, 0.5461]
	lsymbb_h3 = [0.4829, 0.4716, 0.4918, 0.4732]
	lsymmg_h2 = [0.4576, 0.4731, 0.4712, 0.4763]
	lsymmg_h3 = [0.4780, 0.4664, 0.4896, 0.4725]
	abst = [0.5435, 0.5449, 0.5441, 0.5534]


	measures = [dg, stg, sp, sp_w, sp_w2, btw, btw_w, pr, pr_w, cc, cc_w, conc1, conc2, conc3, conc4,
	conc5, conc6, conc7, conc8, access_h2, access_h3, gaccs, hsymbb_h2, hsymbb_h3, hsymmg_h2, hsymmg_h3,
	lsymbb_h2, lsymbb_h3, lsymmg_h2, lsymmg_h3, abst]

	red_noun = []
	red_tfidf = []
	red_noun_ard = []
	red_tfidf_ard = []

	redes = [red_noun, red_tfidf, red_noun_ard, red_tfidf_ard]

	for i in measures:
		for index , j in enumerate(i):
			redes[index].append(j)  


	x = [a+1 for a in range(len(measures))]


	legends = ['Red Noun', 'Red TfIdf', 'Red Noun + ARD', 'Red TfIDf + ARD']

	legend1 = ['Red Noun' , 'Red Noun + ARD']
	legend2 = ['Red TfIdf', 'Red TfIDf + ARD']

	colors = ['blue', 'red', 'darkgreen', 'yellow']


	

	redes_test = [redes[0] , redes[2]]
	redes_test2 = [redes[1], redes[3]]


	for index, red in enumerate(redes):
		plt.plot(x, red, color=colors[index], linewidth=3.0)


	plt.legend(legends, loc='upper right')
	#plt.xlabel('Inter-edge weight')
	#plt.ylabel('Rouge Recall')

	#plt.title('MLN-TfIdf ' + dictionary[measure][0])

	
	plt.show()





		



if __name__ == '__main__':

	#paper()

	
	
	#test = 'CSTNews/1_2_best_rivaldo_AB.csv'
	#test2 = 'CSTNews/3_4_best_ngram_AB.csv'

	#test = 'DUC2002/1_2_best_rivaldo_AB.csv'
	#test2 = 'DUC2002/3_4_best_ngram_AB.csv'

	#test = 'DUC2004/1_2_best_rivaldo_AB.csv'
	#test2 = 'DUC2004/3_4_best_ngram_AB.csv'

	#test = 'CSTNews/symmetry_ribaldo.csv'
	#test = 'DUC2002/symmetry_ribaldo_accs.csv'
	test = 'DUC2004/symmetry_ribaldo_accs.csv'


	#test = 'CSTNews/accessibility_ribaldo.csv'




	
	datos_rivaldo = read_file(test)
	#datos_ngrams = read_file(test2)

	
	#'pr' 'dg' 'sp' 'at' 'gaccs'
	#'pr_w' 'stg' 'sp_w'

	#draw('pr' , datos_rivaldo)

	#draw_weighted(datos_rivaldo, datos_ngrams)

	#draw_four_non_weighted('at', datos_rivaldo , datos_ngrams)

	#draw_five_non_weighted(datos)

	symmetry_analysis('h3', datos_rivaldo) # m b h2 h3 L H 
	#symmetry_second_analysis(datos_rivaldo) # m b h2 h3 L H 

	##accessibility_analysis(datos_rivaldo)
	








