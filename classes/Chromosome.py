

class Chromosome(object):
	#MAKE TYPE SAFE LATER
	#PASS IN LIST OF MAXIMUMS AND LIST OF INCREMENTS
	def __init__(self, algos):

		#These variables refer to values in the string 
		#Note, all the number variables start from 1 if the increment
		#is 1. If the increment is less than 1, starts from 0
		self.possibleAlgos = algos #A String of all possible algorithms
		self.max_beta = 10000 #The maximum value for beta
		self.incr_beta = 1	#The increment used in beta
		self.max_tol = 1	#The maximum value for tolerance
		self.incr_tol = 0.001 #The increment for the tolerance function
		self.max_scale = 10000 #Maximum possible value for scale
		self.incr_scale = 1 #How much scale increments by
		self.max_sigma = 10 #The max possible value of sigm
		self.incr_sigma = 0.01 #The increment for sigma
		self.weight_min_sigma = 0 #Sigma is usually between 0-1, so we 
								  #we want to weight these values more
		self.weight_max_sigma = 1 #The end of the weighted sigma values
		self.max_min_size = 10000 #Maximum value for min_size variable
		self.incr_min_size = 1 #Increment for min_size
		self.max_n_segments = 10000 #Maximum for n_segments variable
		self.incr_n_segments = 

		self.chromosome = []


