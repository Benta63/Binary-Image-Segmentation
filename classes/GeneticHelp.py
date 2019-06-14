import random
import numpy as np
import skimage.measure
from . import AlgorithmParams
from . import AlgorithmSpace
import cv2
import copy
class GeneticHelp(object):
	
	#Returns a number from a list with certain values being weighted.
	#Variables:
	#seq is the sequence to be weighted
	#minVal is the minimum value to be weighted higher
	#maxVal is the maximum value to be weighted higher
	#weight is what the values from minVal to maxVal should be weighted
	def weighted_choice(seq, minVal, maxVal, weight):
		
		weights = []
		#Here we compute the number of values between minVal and maxVal
		counter = 0
		for i in seq:
			if minVal <= i <= maxVal:
				counter += 1

		for i in range(0, len(seq)):
			'''Populates the weights list. 
			Example: If weight is 0.5 and there are 5 values between
			minVal and maxVal, there is a 0.1 chance of each of those
			values
			'''
			if (i < counter):
				weights.append(weight/counter)
			else:
				weights.append((1-weight)/(len(seq) - counter))
		totals = []
		cur_total = 0
		
		for w in weights:
			cur_total += w
			totals.append(cur_total)
		#The randomization
		rand = random.random() * cur_total
		weightIndex = 0
		#And to select the correct index
		for i, total in enumerate(totals):
			if rand < total:
				weightIndex = i
				break
		#So we have the index
		return seq[weightIndex]

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

	'''Executes a crossover between two arrays (np1 and np2) picking a 
	random amount of indexes to change between the two.
	'''
	def skimageCrossRandom(np1, np2):
		assert(len(np1) == len(np2))
		#The number of places that we'll cross
		crosses = random.randrange(len(np1))
		#We pick that many crossing points
		indexes = random.sample(range(0, len(np1)), crosses)
		#And at those crossing points, we switch the parameters
		
		for i in indexes:
			np1[i], np2[i] = np2[i], np1[i]

		return np1, np2

	#This is wrong. I have to debug
	def mutate(child, posVals, flipProb = 0.3):
		#Just because we chose to mutate a value doesn't mean we mutate
		#Every aspect of the value	
		for index in range(0, len(child)):
			randVal = random.random()
			if randVal < flipProb:
				#Then we mutate said value

				child[index] = random.choice(posVals[index])
				

	'''Takes in two ImageData obects and compares them according to
	skimage's Structual Similarity Index and the mean squared error
	Variables:
	img1 is an image array segmented by the algorithm. 
	img2 is the validation image
	imgDim is the number of dimensions of the image.
	'''
	def FitnessFunction(img1, img2, imgDim):	
		assert(len(img1.shape) == len(img2.shape) == imgDim)

		#The channel deterimines if this is a RGB or grayscale image
		channel = False
		if imgDim > 2: channel = True
		#Comparing the Structual Similarity Index (SSIM) of two images
		#ssim = skimage.measure.compare_ssim(img1, img2, win_size=3, 
			#multichannel=channel)
		#Comparing the Mean Squared Error of the two image
		mse = skimage.measure.compare_nrmse(img1, img2)
		
		return [mse,]

	'''Runs an imaging algorithm given the parameters from the population
	Variables:
	copyImg is an ImageData object of the image
	valImg is an ImageData object of the validation image
	individual is the parameter that we chose
	'''
	def runAlgo(copyImg, valImg, individual):
		img = copy.deepcopy(copyImg)
		#Making an AlorithmParams object
		params = AlgorithmParams.AlgorithmParams(img, individual[0],
			individual[1], individual[2], individual[3], individual[4],
			individual[5], individual[6], individual[7], individual[8],
			individual[9], individual[10], individual[11], individual[12]
			, individual[13], individual[14], individual[15][0],
			individual[15][1], individual[16], individual[17], individual[18]
			, individual[19], 'auto', individual[20], individual[21])


		Algo = AlgorithmSpace.AlgorithmSpace(params)

		#Python's version of a switch-case
		#Listing all the algorithms. For fun?
		AllAlgos = {
			'RW': Algo.runRandomWalker,
			'FB': Algo.runFelzenszwalb,
			'SC': Algo.runSlic,
			'QS': Algo.runQuickShift,
			'WS': Algo.runWaterShed,
			'CV': Algo.runChanVese,
			'MCV': Algo.runMorphChanVese,
			'AC': Algo.runMorphGeodesicActiveContour,
			'FD': Algo.runFlood,
			'FF': Algo.runFloodFill
		}
		#Some algorithms cannot be used on grayscale images
		GrayAlgos = {
			'RW': Algo.runRandomWalker,
			'WS': Algo.runWaterShed,
			'CV': Algo.runChanVese,
			'MCV': Algo.runMorphChanVese,
			'AC': Algo.runMorphGeodesicActiveContour,
			'FD': Algo.runFlood,			
			'FF': Algo.runFloodFill
		}
		#Some algorithms are only good for colored images
		RGBAlgos = {
			'RW': Algo.runRandomWalker,
			'FB': Algo.runFelzenszwalb,
			'SC': Algo.runSlic, 
			'QS': Algo.runQuickShift,
			'WS': Algo.runWaterShed,
			#HAVING SOME TROUBLE WITH AC. NEED TO RETUL
			#'AC': Algo.runMorphGeodesicActiveContour,
			'FF': Algo.runFloodFill
		}
		#Some algorithms return masks as opposed to the full images
		Masks = ['FB', 'SC', 'QS']
		#Some algorithms return boolean arrays
		BoolArrs = ['CV','FD']
		#The functions in Masks and BoolArrs will need to pass through
		#More functions before they are ready for the fitness function
		switcher = GrayAlgos
		if (img.getDim() > 2): switcher = RGBAlgos
		#If the algorithm is not right for the image, return large number
		if (params.getAlgo() not in switcher): return [1000,]
		#Running the algorithm and parameters on the image
		func = switcher.get(params.getAlgo(), "Invalid Code")	
		img = func()

		runAlg = AlgorithmSpace.AlgorithmSpace(params)
		img = runAlg.runAlgo()

		#The algorithms in Masks and BoolArrs need to be applied to the img
		#using runMarkBoundaries
		if  params.getAlgo() in Masks or params.getAlgo() in BoolArrs:
			img = runAlg.runMarkBoundaries(img)

		#Running the fitness function
		evaluate = GeneticHelp.FitnessFunction(img, 
			valImg.getImage(), len(img.shape))
		
		return (evaluate)
