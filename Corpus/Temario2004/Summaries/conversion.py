
import codecs
import os 
from shutil import copyfile

input_files = 'Summarios/'

output_files = 'temario2003_b/'


documents = os.listdir(input_files)

for i in documents:
	path = input_files + i 
	parte =  i[i.find('-')+1:]
	new_path = output_files + parte[:parte.rfind('.')] + '_reference1.txt'
	print path , new_path
	copyfile(path, new_path)





#ce94ab10-a_reference1.txt


#document = codecs.open(input_file, encoding="utf-8")
#content = ""
#for i in document:
#	print i 
    #i = i.rstrip()
    #content+= i + " "



