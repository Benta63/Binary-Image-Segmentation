

import random
import numpy as np
import skimage.measure
import cv2
import copy
from PIL import Image

from . import AlgorithmParams
from . import AlgorithmSpace
from . import AlgorithmHelper
from .AlgorithmHelper import AlgoHelp


class GeneticHelp(object):

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
		#TODO: Only change values associated with algorithm
		assert(len(np1) == len(np2))
		#The number of places that we'll cross
		crosses = random.randrange(len(np1))
		#We pick that many crossing points
		indexes = random.sample(range(0, len(np1)), crosses)
		#And at those crossing points, we switch the parameters
		
		for i in indexes:
			np1[i], np2[i] = np2[i], np1[i]

		return np1, np2

	''' Changes a few of the parameters of the weighting a random
		number against the flipProb.
		Variables:
		copyChild is the individual to mutate
		posVals is a list of lists where each list are the possible
			values for that particular parameter
		flipProb is how likely, it is that we will mutate each value.
			It is computed seperately for each value. 
	'''
	def mutate(copyChild, posVals, flipProb = 0.5):

		#Just because we chose to mutate a value doesn't mean we mutate
		#Every aspect of the value	
		child = copy.deepcopy(copyChild)

		#Not every algorithm is associated with every value
		#Let's first see if we change the algorithm
		randVal = random.random()
		if randVal < flipProb:
			#Let's mutate
			child[0] = random.choice(posVals[0])
		#Now let's get the indexes (parameters) related to that value
		switcher = AlgoHelp().algoIndexes()
		indexes = switcher.get(child[0])

		for index in indexes:
			randVal = random.random()
			if randVal < flipProb:
				#Then we mutate said value
				if index == 22:
					#Do some special
					X = random.choice(posVals[22])
					Y = random.choice(posVals[23])
					Z = random.choice(posVals[24])
					child[index] = (X, Y, Z)
					continue

				child[index] = random.choice(posVals[index])
		return child		

	'''Takes in two ImageData obects and compares them according to
	skimage's Structual Similarity Index and the mean squared error
	Variables:
	img1 is an image array segmented by the algorithm. 
	img2 is the validation image
	imgDim is the number of dimensions of the image.
	'''
	def __FitnessFunction(img1, img2, imgDim):	
		assert(len(img1.shape) == len(img2.shape) == imgDim)

		#The channel deterimines if this is a RGB or grayscale image
		channel = False
		if imgDim > 2: channel = True
		#print(img1.dtype, img2.dtype)
		img1 = np.uint8(img1)
		#print(img1.dtype, img2.dtype)
		assert(img1.dtype == img2.dtype)
		#TODO: Change to MSE
		#Comparing the Structual Similarity Index (SSIM) of two images
		ssim = skimage.measure.compare_ssim(img1, img2, win_size=3, 
			multichannel=channel, gaussian_weights=True)
		#Comparing the Mean Squared Error of the two image
		#print("About to compare")
		#print(img1.shape, img2.shape, imgDim)
		#mse = skimage.measure.compare_mse(img1, img2)
		#Deleting the references to the objects and freeing memory
		del img1
		del img2
		#print("eror above?")
		return [abs(ssim),]

	'''Runs an imaging algorithm given the parameters from the 
		population
	Variables:
	copyImg is an ImageData object of the image
	valImg is an ImageData object of the validation image
	individual is the parameter that we chose
	'''
	def runAlgo(copyImg, valImg, individual):

		img = copy.deepcopy(copyImg)
		#Making an AlorithmParams object
		params = AlgorithmParams.AlgorithmParams(img, individual)



		Algo = AlgorithmSpace.AlgorithmSpace(params)

		#Python's version of a switch-case
		#Listing all the algorithms. For fun?
		AllAlgos = [
			'RW',#: Algo.runRandomWalker,
			'FB',#: Algo.runFelzenszwalb,
			'SC',#: Algo.runSlic,
			'QS',#: Algo.runQuickShift,
			'WS',#: Algo.runWaterShed,
			'CV',#: Algo.runChanVese,
			'MCV',#: Algo.runMorphChanVese,
			'AC',#: Algo.runMorphGeodesicActiveContour,
			'FD',#: Algo.runFlood,
			'FF',#: Algo.runFloodFill
		]
		#Some algorithms return masks as opposed to the full images
		#The functions in Masks and BoolArrs will need to pass through
		#More functions before they are ready for the fitness function
		switcher = AlgoHelp().channelAlgos(img)


		#If the algorithm is not right for the image, returna large 
		#	number
		if (params.getAlgo() not in switcher): return [100,]
		
		#Running the algorithm and parameters on the image
	
		runAlg = AlgorithmSpace.AlgorithmSpace(params)
		img = runAlg.runAlgo()
				#The algorithms in Masks and BoolArrs need to be applied to the
		#	img
	
		#Running the fitness function

		evaluate = GeneticHelp.__FitnessFunction(np.array(img), 
			valImg.getImage(), len(np.array(img).shape))
		#Explicitely freeing memory
		del img
		del valImg
		
		return (evaluate)
