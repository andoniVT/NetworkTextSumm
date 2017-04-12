import os
from shutil import copyfile, rmtree
import glob

from configuration import references_dir, extras
from utils import deleteFiles , get_csv_values, sort_results
import csv

class Validation(object):

    def __init__(self, validation, language, type_summary, corpus_name):
        self.validation = validation
        self.language = language
        self.type_summary = type_summary
        self.corpus_name = corpus_name

        self.dictionary = dict()
        self.dictionary['ptg'] = [['temario_v1', 'temario_v2'], ['cstnews_v1', 'cstnews_v2']]
        self.dictionary['eng'] = [['duc2002_s' , 'duc2003_s'] , ['duc2002_m' , 'duc2003_m']]

        reference_summaries_key = self.dictionary[self.language][self.type_summary][self.corpus_name]
        self.path_references = references_dir[reference_summaries_key]


    def validate(self, output):
        print "Saving results in: " +  output
        print self.path_references
        results = []

        #folders = os.listdir('Automatic/')
        folders = os.listdir(extras['Automatics'])
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
            print i
            wr.writerow(i)





    def evaluate(self,  path_systems):
        print self.path_references , path_systems
        deleteFiles(references_dir['rougeReferences'])
        deleteFiles(references_dir['rougeSystems'])
        references = os.listdir(self.path_references)
        for i in references:
            source = self.path_references + i
            destination = references_dir['rougeReferences'] + i
            copyfile(source, destination)

        systems = os.listdir(path_systems)

        for i in systems:
            source = path_systems + '/' + i
            destination = references_dir['rougeSystems'] + i
            copyfile(source, destination)

        os.system("java -jar rouge2.0.jar")
        precision, recall, fmeasure = get_csv_values("results.csv")
        print precision, recall, fmeasure
        return [precision, recall, fmeasure]








