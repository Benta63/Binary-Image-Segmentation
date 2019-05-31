import numpy as np
import os
from PIL import Image
import skimage
<<<<<<< HEAD
import random

#https://github.com/DEAP/deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
=======
from skimage import segmentation
import cv2
>>>>>>> master

from classes import ImageData
from classes import AlgorithmSpace
from classes.AlgorithmSpace import AlgorithmSpace
from classes import AlgorithmParams

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
	np1[point1:point2], np2[point1:point2] = np2[point1:point2].copy(),
		np1[point1:point2].copy()
	return np1, np2

def writeData(img, newImg, imgName, txtName):

	file = open(txtName, 'w+')
	for line in img.getImage():
		for number in line:
			file.write(str(number) + " ")
		file.write('\n')
	file.write(str(img.getImage()))
	file.close()

	#Saving the image
	cv2.imwrite(imgName, newImg)	

def check_dir(path):
	directory = os.path.dirname(path)
	if not os.path.exists(path):
		return False
	return True

if __name__ == '__main__':
	#Will later have user input to find where the images are
	ImagePath = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'
	if (check_dir(ImagePath) == False):
		print ('ERROR: Directory %s Does not exist'%ImagePath)


	#Making an ImageData object for all of the images
	AllImages = [ImageData.ImageData(os.path.join(root, name)) for 
		root, dirs, files in os.walk(ImagePath) for name in files]

	#Now to make labels (an empty array of the same shape)
<<<<<<< HEAD

	#AllLabels = [np.zeros(img.getShape()) for img in AllImages]

	#Let's get all possible values in lists
	Algos = ['FB','SC','QS','WS','CV','MCV','AC'] #Need to add floods
	betas = [i for i in range(0,10000)]
	tolerance = [float(i)/1000 for i in range(0,1000,1)]
	scale = [i for i in range(0,10000)]
	sigma = [float(i)/100 for i in range(0,1000,1)]
	#Later weight sigmas from 0-1 higher
	min_size = [i for i in range(0,10000)]
	n_segments = [i for i in range(2,10000)]
	iterations = [1000]
	ratio = [float(i)/100 for i in range(0,100)]
	kernel = [i for i in range(0,10000)]
	max_dists = [i for i in range(0,10000)]
	random_seed = 134
	compactness = [0.0001,0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
	mu = [float(i)/100 for i in range(0,100)]
	#The values for Lambda1 and Lambda2 respectively
	Lambdas = [[1,1], [1,2], [2,1]]
	dt = [float(i)/10 for i in range(0,100)]
	init_level_set_chan = ['checkerboard', 'disk', 'small disk']
	init_level_set_morph = ['checkerboard', 'circle']
	#Should weight 1-4 higher
	smoothing = [i for i in range(1,100)]
	alphas = [i for i in range(0,10000)]
	#Should weight values -1, 0 and 1 higher
	balloon = [i for i in range(-1000,1000)]
	#For flooding, which I will add later
	#seed_point = 
	#new_value = 

	#Using the DEAP genetic algorithm to make One Max
	#Creator factory builds new classes
	creator.create("FitnessMax", base.Fitness, weight=(1.0,))
	creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)
	#
	toolbox = base.Toolbox()
	#Attribute generator
	toolbox.register("attr_bool", random.randint, 0, 1)
	#Structural initializers
	toolbox.register("mate", twoPointCopy)
	toolbox.register("evaluate", evalOneMax)
	toolbox.register("mutate", tools.mutFlibBit, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=3)

	hof = tools.HallOfFame(1, similar=np.allclose)
	pop = toolbox.population(n=100)
	stats = tools.Statistics(lambda img: img.fitness.value)
=======
	print (AllImages[0].getShape())
	AllLabels = [np.ones(img.getShape()) for img in AllImages]

	#Some debugging statements
	assert (len(AllLabels) == len(AllImages))
	assert (AllLabels[0].shape == AllImages[0].getShape())
	assert (len(AllImages[0].getImage()) == len(AllLabels[0]))
	
	#Testing variables
	betas = [80, 90, 100, 110, 120, 130, 140, 150]
	imgTest = AllImages[0]
	labelTest = AllLabels[0]

	#For Felzenszwalb
	scale = [i for i in range(0, 10)]
	sigmas = [0.8, 0.9, 0.95, 1]
	min_sizes = [i for i in range(5, 10)]
	#For slic
	n_segments = [2, 3, 4, 5, 6, 10, 100]
	compacts = [0.01, 0.1, 1, 10, 100]
	max_iters = [10]

	#For quickshift
	ratios = [0.1, 0.5, 1]
	kernels = [5,]
	max_dists = [5, 10, 15]
	seeds = [134, 34, 63, 6]

	#Random Walker
	'''np_img = skimage.segmentation.random_walker(AllImages[0].getImage(), AllLabels[0], mode='cg', copy=False)
	#Random Walker from AlgorithmSpace
	npImgs = 1*AlgorithmSpace.runRandomWalker(AllImages[0], AllLabels[0], betas)
	'''
	#Active Countour
	#How to do snakes??	
	#Felzenszwalb
	'''np_img = skimage.segmentation.felzenszwalb(AllImages[0].getImage(), scale =3, sigma=0.9, min_size=10)

	npImgs = AlgorithmSpace.runFelzenszwalb(AllImages[0], scale, sigmas, min_sizes)
	'''
	#slic
	'''np_img = skimage.segmentation.slic(AllImages[0].getImage(), n_segments=2, compactness=10, max_iter=100, sigma=0.9)
	npImgs = 1*AlgorithmSpace.runSlic(AllImages[0], n_segments, compacts, max_iters, sigmas)
	'''

	#quickshift
	'''np_img = skimage.segmentation.quickshift(AllImages[0].getImage())
	npImgs = 1*AlgorithmSpace.runQuickShift(AllImages[0], ratios, kernels, max_dists, sigmas, seeds)
	'''
	npImg = skimage.segmentation.watershed(AllImages[0].getImage())
	writeData(AllImages[0], npImg, 'testImg.png', 'testData.txt')
>>>>>>> master
