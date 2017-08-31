import os
from shutil import copyfile, rmtree
import glob

from configuration import references_dir, extras
from utils import deleteFiles , get_csv_values, sort_results, generate_excel_simple, generate_excel_d2v_mln
import csv

import random
class Validation(object):

    #def __init__(self, validation, language, type_summary, corpus_name, table_parameters, excel_name=None, parameters_table=None):
    def __init__(self, language, type_summary, corpus_name, table_parameters, excel_name=None, parameters_table=None):
        #self.validation = validation
        self.language = language
        self.type_summary = type_summary
        self.corpus_name = corpus_name

        self.dictionary = dict()
        self.dictionary['ptg'] = [['temario_v1', 'temario_v2'], ['cstnews_v1', 'cstnews_v2']]
        self.dictionary['eng'] = [['duc2002_s' , 'duc2004_s'] , ['duc2002_m' , 'duc2004_m']]

        reference_summaries_key = self.dictionary[self.language][self.type_summary][self.corpus_name]
        self.path_references = references_dir[reference_summaries_key]

        self.first_value_table = table_parameters[0]
        self.second_value_table = table_parameters[1]
        self.third_value_table = table_parameters[2]

        self.excel_name = excel_name
        self.parameters_table = parameters_table


    def validate(self, output):
        print "Saving results in: " +  output
        print "testttt", self.path_references
        results = []
        for i in range(self.first_value_table*self.third_value_table):
            results.append([])

        for i in results:
            print i

        #folders = os.listdir('Automatic/')
        #folders = os.listdir(extras['Automatics']+ '1/')
        folders = os.listdir(extras['Automatics'])
        number_tests = len(folders)
        '''
        for i in range(number_tests): # carpeta 1 (0.25) , carpeta 2(0.3) , ... carpeta n(0.35)
            path = extras['Automatics'] + str(i+1)
            sub_folders = os.listdir(path)
            for index,  j in enumerate(sub_folders): # medidas : dg - pr
                path_systems = extras['Automatics'] + str(i+1) + '/' + j   # automactics + 'dg'
                rouge_values = self.evaluate(path_systems)
                #values = [j, str(rouge_values[0]), str(rouge_values[1]), str(rouge_values[2])]
                rouge_recall = (j, str(rouge_values[1]))
                print rouge_recall
                results[index].append(rouge_recall)
        '''

        print 'first' , self.first_value_table
        print 'second' , self.second_value_table
        print 'third' , self.third_value_table

        #testes = 2
        testes =  self.third_value_table
        #limiares = 3
        limiares =  self.second_value_table
        counter = 1
        index = 0
        #for i in range(testes):
        for i in range(self.first_value_table):
            #for j in range(limiares):
            for j in range(self.second_value_table):
                path = extras['Automatics'] + str(counter)
                sub_folders = os.listdir(path)
                print counter , path, sub_folders
                for num , k in enumerate(sub_folders):
                    path_systems = extras['Automatics'] + str(counter) + '/' + k
                    rouge_values = self.evaluate(path_systems)
                    rouge_recall = (k, str(rouge_values[1]))

                    print path_systems
                    #value = random.random()
                    #print value
                    #rouge_recall = (k, random.random())
                    #print rouge_recall
                    results[num+index].append((rouge_recall))
                    #results[num + index].append(((k , value)))
                counter += 1
            #index +=testes
            index += self.third_value_table

            print ''

        for i in results:
            print i



        if self.first_value_table == 1 and self.second_value_table==1:
            generate_excel_simple(self.excel_name, results)
        else:
            generate_excel_d2v_mln(self.excel_name, results, self.parameters_table)



        #print folders
        #for i in  results:
        #    print i
        #a = input()

    '''
        for i in folders:
            path_systems = extras['Automatics'] + i
            rouge_values = self.evaluate(path_systems)
            values = [i , str(rouge_values[0]), str(rouge_values[1]), str(rouge_values[2])]
            results.append(values)

        print "FINAL RESULTS"
        results = sort_results(results)
        myfile = open(output, 'wb')
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for i in results:
            #print i
            print i[0] , i[2]
            wr.writerow(i)
    '''





    def evaluate(self,  path_systems):
        print self.path_references , path_systems
        deleteFiles(references_dir['rougeReferences'])
        deleteFiles(references_dir['rougeSystems'])
        references = os.listdir(self.path_references)
        print references
        for i in references:
            source = self.path_references + i
            destination = references_dir['rougeReferences'] + i
            copyfile(source, destination)

        systems = os.listdir(path_systems)
        print systems

        for i in systems:
            source = path_systems + '/' + i
            destination = references_dir['rougeSystems'] + i
            copyfile(source, destination)

        os.system("java -jar rouge2.0.jar")
        precision, recall, fmeasure = get_csv_values("results.csv")
        print precision, recall, fmeasure
        return [precision, recall, fmeasure]








