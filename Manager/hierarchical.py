import numpy as np
import utils
from configuration import extras

class Concentric(object):

    def __init__(self, graph):
        self.graph = graph
        utils.generate_net(self.graph) # no paso la locacion porque por defecto se guardara en extras
        self.location = extras['NetAux']
        self.generate_measures()

    def generate_measures(self):
        command = "java -jar Concentric.jar 10 0 0 200 1 OutNet 0 0 0 0 0 "
        output = extras['FolderAux']
        command2 = self.location + " " + output + " false false"
        execute_command = command + command2
        print execute_command



class Symmetry(object):

    def __init__(self):
        pass


class Accessibility(object):

    def __init__(self):
        pass


class GeneralizedAccesibility(object):

    def __init__(self, graph):
        self.__graph = graph

    def calculate_accesibility(self):
        G = self.__graph.as_undirected()
        M = G.get_adjacency()
        N = np.shape(M)[0]
        P1 = np.zeros([N, N], dtype=np.float)
        for i in range(N):
            P1[i, :] = np.array(M[i, :]) / np.float(np.sum(M[i, :]));
        P = np.exp(P1) / np.e
        va = np.zeros(N, dtype=np.float);
        for i in range(N):
            va[i] = np.exp(-np.sum(P[i, :] * np.log(P[i, :])))
        return va

    def sort_by_accesibility(self):
        measures = self.calculate_accesibility()
        measures = measures.tolist()
        return utils.reverseSortList(measures)