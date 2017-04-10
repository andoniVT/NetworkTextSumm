import numpy as np
import utils
from configuration import extras
from utils import execute_concentric , read_dat_files, reverseSortList, execute_symmetry, read_csv_file, sortList, specialSortList

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
        execute_concentric(execute_command)

    def sort_by_concentric(self, type, h):
        results = read_dat_files()
        measure_type = results[type]
        if h == 2:
            measure = measure_type[0]
        else:
            measure = measure_type[1]
        return reverseSortList(measure)




class Symmetry(object):

    def __init__(self, graph):
        self.graph = graph
        utils.generate_xnet(self.graph)  # no paso la locacion porque por defecto se guardara en extras
        self.location = extras['XNetAux']
        self.generate_measures()

    def generate_measures(self):
        command = "./CVSymmetry.exe -c -M -l 3 " + self.location + " " + extras['CSVAux']
        print command
        execute_symmetry(command)

    def sort_by_symmetry(self, order, type, h):
        #order : h - l
        #type: b - m
        #h: 2-3
        if type == 'b':
            nType = 0
        else:
            nType = 1
        if h=='2':
            nH = 0
        else:
            nH = 1

        symmetries = read_csv_file()
        #[[backbone_h2, backbone_h3] , [merged_h2,  merged_h3]]
        measure = symmetries[nType][nH]

        the_high = reverseSortList(measure)

        if order == 'h':
            return the_high
        else:
            return specialSortList(the_high)
            #return sortList(measure)



class Accessibility(object):

    def __init__(self, graph):
        self.graph = graph
        utils.generate_xnet(self.graph)  # no paso la locacion porque por defecto se guardara en extras
        self.location = extras['XNetAux']

    def sort_by_accessibility(self, h):
        command = "./CVAccessibility -l " + h + " " + self.graph




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