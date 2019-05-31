import numpy as np
import os
from PIL import Image
import skimage
import random

#https://github.com/DEAP/deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

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
