import numpy as np
import igraph
from igraph import *
import itertools

class AbsorptionTime(object):
	
	def __init__(self, network):
		self.network = network
		self.components = self.network.components()
		self.matrix_data = self.calculate_transitive()
		self.transitive_matrix = self.matrix_data[0]
		self.big_component = self.matrix_data[1]
		self.all_times = self.calculate_all_times()

	def calculate_transitive(self):
		mayor = -9999
		index_mayor = 0
		for index, component in enumerate(self.components):
			size = len(component) 
			if size > mayor:
				mayor = size
				index_mayor = index
		big_component = self.components[index_mayor]
		network_weights = self.network.es['weight']
		size = self.network.vcount()
		new_size = len(big_component)
		matrix = np.zeros((new_size, new_size), dtype=float)
		adjacency_matrix = self.network.get_adjacency()
		strenghts = self.network.strength(weights=network_weights)
		index_i = 0
		index_j = 0

		for i in range(size):
			if i in big_component:
				vector = []
				for j in range(size):
					if j in big_component:
						value = adjacency_matrix[i][j]
						if value == 1:
							id_edge = self.network.get_eid(i,j)
							weight = network_weights[id_edge]
							stg = strenghts[i]
							value = weight/float(stg) 	
						matrix[index_i][index_j] = value
						index_j+=1
				index_j=0
				index_i+=1
		return [matrix, big_component] 

	def calculate_time(self, node_initial, node_final):
		Q = np.delete(self.transitive_matrix, (node_initial), axis=0)
		Q = np.delete(Q, (node_final-1), axis=0)
		Q = np.delete(Q, (node_initial), axis=1)
		Q = np.delete(Q, (node_final-1), axis=1)
		I = np.identity(Q.shape[0])
		N = np.linalg.inv(I-Q)
		o = np.ones(Q.shape[0])
		t = np.dot(N,o)
		time = sum(t)
		return time

	def calculate_time_v2(self, node_initial, node_final):
		Q = np.delete(self.transitive_matrix, (node_initial), axis=0)
		Q = np.delete(Q, (node_final - 1), axis=0)
		Q = np.delete(Q, (node_initial), axis=1)
		Q = np.delete(Q, (node_final - 1), axis=1)
		I = np.identity(Q.shape[0])
		o = np.ones(Q.shape[0])
		t = np.linalg.solve(I - Q, o)
		time = sum(t)
		return time

	def calculate_all_times(self):
		indexes = [x for x in range(len(self.big_component))]
		node_combinations = itertools.combinations(indexes,2)
		times = dict()

		for i in node_combinations:
			key = str(i[0]) + '-' + str(i[1])
			time = self.calculate_time_v2(i[0], i[1])
			times[key] = time
		return times

	def mean_absorption_time_v2(self, node_id):
		mean = 0
		for i in range(len(self.big_component)):
			if node_id!=i:
				if node_id < i:
					key = str(node_id) + '-' + str(i)
				else:
					key = str(i) + '-' + str(node_id)

				time = self.all_times[key]
				mean+=time
		denominador = float(len(self.big_component)-1)
		if denominador!=0:
			return mean/denominador
		else:
			return 9999999


	def mean_absorption_time(self, node_id):
		mean = 0
		for i in range(len(self.big_component)):
			if node_id!=i:
				time = self.calculate_time(node_id, i)
				mean+=time 
		denominador = float(len(self.big_component)-1)
		if denominador!=0:
			return mean/denominador
		else:
			return 9999999

	def get_all_times(self):
		result = []
		index = 0
		network_size = self.network.vcount()
		for i in range(network_size):
			if i in self.big_component:
				value = self.mean_absorption_time_v2(index)
				result.append(value)
				index+=1
			else:
				result.append(9999999)
		return result

'''
network = Graph()
network.add_vertices(9)

aristas = [(0,1,), (1,2), (1,6),(2,6),(5,6),(3,6),(4,7)]
network.add_edges(aristas)

weights = [5,2,3,6,4,1,5]
network.es['weight'] = weights


obj = AbsorptionTime(network)
times = obj.get_all_times()
for index, values in enumerate(times):
	print index, values
'''
