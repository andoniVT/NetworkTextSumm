
class Vectorization(object):

	def __init__(self, corpus, vectorization_type, use_inference=None, vector_size=None):
		self.corpus = corpus
		self.vectorization_type = vectorization_type
		self.use_inference = use_inference
		self.vector_size = vector_size


	def tf_idf_vectorization(self):
		print "vectorization tfidf"

	def d2v_vectorization(self):
		print "vectorization d2v"


	def calculate(self):
		if self.vectorization_type == 'tfidf':
			self.tf_idf_vectorization()
		else:
			self.d2v_vectorization()

		return ['dictionary' , 'key: nombre del documento o cluster', 'value: matrix con los vectores de cada sentence del documento']


