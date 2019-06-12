import random
import numpy as np
import skimage.measure
from . import AlgorithmParams
from . import AlgorithmSpace
import cv2
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
			#Populates the weights list. 
			#Example: If weight is 0.5 and there are 5 values between
			#minVal and maxVal, there is a 0.1 chance of each of those
			#values
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

	'''Takes in two ImageData obects and compares them according to
	skimage's Structual Similarity Index and the mean squared error
	'''
	def FitnessFunction(img1, img2, imgDim):
		
		#assert(len(img1.shape) == len(img2.shape))
		#assert(type(img1) == type(img2))
		channel = False
		if imgDim > 2: channel = True
		#Comparing the Structual Similarity Index (SSIM) of two images
		#ssim = skimage.measure.compare_ssim(img1, img2, multichannel=channel)
		#Comparing the Mean Squared Error
		
		mse = skimage.measure.compare_nrmse(img1, img2)
		
		
		return [mse,]

	#Runs an imaging algorithm given the parameters from the population
	def runAlgo(img, valImg, individual):
		params = AlgorithmParams.AlgorithmParams(img, individual[0],
			individual[1], individual[2], individual[3], individual[4],
			individual[5], individual[6], individual[7], individual[8],
			individual[9], individual[10], individual[11], individual[12]
			, individual[13], individual[14], individual[15][0],
			individual[15][1], individual[16], individual[17], individual[18]
			, individual[19], 'auto', individual[20], individual[21])

		Algo = AlgorithmSpace.AlgorithmSpace(params)
		print(params.getAlgo())

		#Python's version of a switch-case
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
		#Some algorithms can't be used on grayscale images
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
		#If the algorithm is not right for the image, return an
		#very large number
		if (params.getAlgo() not in switcher): return [1000,]
		print("in switcher")
		func = switcher.get(params.getAlgo(), "Invalid Code")
		newImg = func()
		print("Before algo")
		runAlg = AlgorithmSpace.AlgorithmSpace(params)
		img = runAlg.runAlgo()

		print("after algo")

		if  params.getAlgo() in Masks or params.getAlgo() in BoolArrs:
			img = runAlg.runMarkBoundaries(img)

		#print ("Shapes", len(img.shape), valImg.getDim())
		# if params.getAlgo() == 'AC':
		# 	print ("Meep")
		# 	print(img.shape, valImg.getImage().shape)
		#cv2.imwrite("test.png", img)
		#print ("End genetic")
		evaluate = GeneticHelp.FitnessFunction(img, 
			valImg.getImage(), len(img.shape))
		print("evaluated")
		
		return (evaluate)
