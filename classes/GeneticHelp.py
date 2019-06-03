import random

class GeneticHelp(object):
	
	#Returns a number from a list with certain values being weighted
	def weighted_choice(seq, minVal, maxVal, weight):
		#seq is the sequence to be weighted
		#minVal is the minimum value to be weighted higher
		#maxVal is the maximum value to be weighted higher
		#weight is what the values from minVal to maxVal should be weighted
		weights = []
		#Here we compute the number of values between minVal and maxVal
		counter = 0
		for i in seq:
			if minVal <= i <= maxVal:
				counter += 1

		for i in range(0, len(seq)):
			if (i < counter):
				weights.append(weight/counter)
			else:
				weights.append((1-weight)/(len(seq) - counter))
		x = random.random()
		for i, elem in enumerate(seq):
			if x <= weights[i]:
				return elem
			x -= weights[i]


	#The fitness function. Need to change a lot
	def evalOneMax(individual):
		return sum(individual)

	#Executes a crossover between two numpy arrays of the same length
	def twoPointCopy(np1, np2):
		assert(len(np1) == len(np2))
		size = len(np1)
		point1 = random.randint(1, size)
		point2 = random.randint(1, size-1)
		if (point2 >= point1):
			point2 +=1
		else: #Swap the two points
			point1, point2 = point2, point1
		np1[point1:point2], np2[point1:point2] = np2[point1:point2].copy(), np1[point1:point2].copy()
		return np1, np2

	#Takes in two ndarrays of image and compares them pixel by pixel
	#Returns the overall different number of pixels
	def FitnessFunction(img1, img2):
		if np.array_equal(img1, img2):
			#The segmentation was perfect
			return 0
		if np.allclose(img1, img2):
			#The segmentation is very close
			return 5
		assert(img1.shape == img2.shape)

		#Comparing the Structual Similarity Index (SSIM) of two images
		ssim = skimage.measure.compare_ssim(img1, img2)
		#Comparing the Mean Squared Error
		mse = skimage.measure.compare_mse(img1, img2)
		return ssim + mse

	#Runs an imaging algorithm given the parameters from the population
	def runAlgo(img, individual):
		'''params = AlgorithmParams.AlgorithmParams(img, individual[0],
			individual[1], individual[2], individual[3], individual[3],
			individual[4], individual[5], individual[6], individual[7],
			individual[8], )
'''
		try:
			pass
		except Exception as e:
			raise
		else:
			pass
		finally:
			pass