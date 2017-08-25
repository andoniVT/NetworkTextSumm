

corpus_dir = dict()

corpus_dir['temario_v1'] = '../Corpus/Temario2004/Textos-fonte'
corpus_dir['cstnews_v1'] = '../Corpus/CSTNews/'
corpus_dir['duc2002'] = '../Corpus/DUC2002/docs/'
corpus_dir['duc2004'] = '../Corpus/DUC2004/docs/'

corpus_dir['textosFonte'] = "Textos-fonte segmentados/"
corpus_dir['cst_extratos'] = "Sumarios/Processed/Extratos/"
corpus_dir['cst_extrato'] = "Sumarios/"
corpus_dir['cst_extrato_name'] = "_extrato_humano.txt"


summaries_dir = dict()

summaries_dir['temario_v1'] = '../Corpus/Temario2004/Summaries/Summarios/'
summaries_dir['cstnews_v1'] = '../Corpus/CSTNews/'
summaries_dir['duc2002'] = '../Corpus/DUC2002/'





references_dir = dict()
references_dir['temario_v1'] = 'References/Portuguese/temario2003/'
#references_dir['temario_v1'] = 'References/Portuguese/temario2003_b/'
references_dir['cstnews_v1'] = 'References/Portuguese/cstnews/'
references_dir['duc2002_s'] = 'References/English/duc2002-single/'
references_dir['duc2002_m'] = 'References/English/duc2002-multi/'
references_dir['duc2004_m'] = 'References/English/duc2004-multi/'

references_dir['rougeReferences'] = "test-summarization/reference/"
references_dir['rougeSystems'] = "test-summarization/system/"





extras = dict()

extras['MarisaTree'] = '../Extras/lexicon/DELAF_PT.marisa'
extras['NounsList'] = '../Extras/substantivosDict.pk'
extras['NotNounsList'] = '../Extras/not_nouns.pk'
extras['NotNounsList_v2'] = '../Extras/not_nouns_2.pk'
extras['NetAux'] = '../Extras/auxiliar.NET'
extras['XNetAux'] = '../Extras/auxiliarX.xnet'
extras['FolderAux'] = '../Extras/concentrics/'
extras['CSVAux'] = '../Extras/auxiliar_symmetry.csv'
extras['Automatics'] = '../Extras/automatics/'
extras['google_w2v'] = '../Extras/duc02_04_w2v_vectors.pk'

extras['PtgSDS_labels'] = '../Extras/dictionary_temario_class_labels.pk'
extras['PtgMDS_labels'] = '../Extras/dictionary_cstnews_class_labels.pk'
extras['EngSDS_labels'] = '../Extras/dictionary_duc2002_sds_class_labels.pk'
extras['EngMDS_labels_1'] = '../Extras/dictionary_duc2002_mds_class_labels.pk'
extras['EngMDS_labels_2'] = '../Extras/dictionary_duc2004_mds_class_labels.pk'





final_results = dict()
final_results['prueba'] = '../Results/test1.csv'
final_results['prueba2'] = '../Results/test2.csv'