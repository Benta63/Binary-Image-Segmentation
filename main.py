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
from skimage import segmentation
import cv2


from classes import ImageData
from classes import AlgorithmSpace
from classes.AlgorithmSpace import AlgorithmSpace
from classes import AlgorithmParams

from classes import FileClass
from classes.FileClass import FileClass
from classes import GeneticHelp
from classes.GeneticHelp import GeneticHelp as GA


if __name__ == '__main__':
	#Will later have user input to find where the images are
	ImagePath = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'
	if (FileClass.check_dir(ImagePath) == False):
		print ('ERROR: Directory %s does not exist'%ImagePath)

	ValidationPath = 'Image_data\\Coco_2017_unlabeled\\rgbd_label'
	if(FileClass.check_dir(ImagePath) == False):
		print("ERROR: Directory %s does not exist"%ValidationPath)

	#Making an ImageData object for all of the regular images
	AllImages = [ImageData.ImageData(os.path.join(root, name)) for 
		root, dirs, files in os.walk(ImagePath) for name in files]

	#Making an ImageData object for all of the labeled images
	ValImages = [ImageData.ImageData(os.path.join(root, name)) for
		root, dirs, files in os.walk(ValidationPath) for name in
		files]

	#Let's get all possible values in lists
	Algos = ['FB','SC','QS','WS','CV','MCV','AC'] #Need to add floods
	betas = [i for i in range(0,10000)]
	tolerance = [float(i)/1000 for i in range(0,1000,1)]
	scale = [i for i in range(0,10000)]
	sigma = [float(i)/100 for i in range(0,10,1)]
	#Sigma should be weighted more from 0-1
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
	smoothing = [i for i in range(1, 10)]
	alphas = [i for i in range(0,10000)]
	#Should weight values -1, 0 and 1 higher
	balloon = [i for i in range(-50,50)]
	connectivity = [i for i in range(0, 9)]
	#For flooding, which I will add later
	#seed_point = 
	#new_value = 

	#Using the DEAP genetic algorithm to make One Max
	#https://deap.readthedocs.io/en/master/api/tools.html
	#Creator factory builds new classes

	#random.seed(34)

	#Need to read up on base.Fitness function
	####### 
	creator.create("FitnessMax", base.Fitness, weights=(100,))
	######

	creator.create("Individual", list, fitness=creator.FitnessMax)
	
	#

	toolbox = base.Toolbox()
	#Attribute generator
	toolbox.register("attr_bool", random.randint, 0, 1000)
	#Structural initializers
	
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("evaluate", GA.runAlgo)
	toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
	toolbox.register("select", tools.selTournament, tournsize=3)
	
	#deap.tools.init_Cycle(container, seq_func, n)
	#Container: data type
	#seq_func: List of function objects to be called in order to fill container
	#n: number of times to iterate through list of functions
	#Returns: An instance of the container filled with data returned from functions
	#Here we register all the parameters to the toolbox
	SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT = 0, 1, 0.5	
	ITER = 10
	SEED = 134
	SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT = 1, 4, 0.5
	BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT = -1, 1, 0.9

	toolbox.register("attr_Algo", random.choice, Algos)
	toolbox.register("attr_Beta", random.choice, betas)
	toolbox.register("attr_Tol", random.choice, tolerance)
	toolbox.register("attr_Scale", random.choice, scale)
	toolbox.register("attr_Sigma", GA.weighted_choice, sigma, SIGMA_MIN, 
		SIGMA_MAX, SIGMA_WEIGHT)
	toolbox.register("attr_minSize", random.choice, min_size)
	toolbox.register("attr_nSegment", random.choice, n_segments)
	toolbox.register("attr_iterations", int, ITER)
	toolbox.register("attr_ratio", random.choice, ratio)
	toolbox.register("attr_kernel", random.choice, kernel)
	toolbox.register("attr_maxDist", random.choice, max_dists)
	toolbox.register("attr_seed", int, SEED)
	toolbox.register("attr_connect", random.choice, connectivity)
	toolbox.register("attr_compact", random.choice, compactness)
	toolbox.register("attr_mu", random.choice, mu)
	toolbox.register("attr_lambda", random.choice, Lambdas)
	toolbox.register("attr_dt", random.choice, dt)
	toolbox.register("attr_init_chan", random.choice, init_level_set_chan)
	toolbox.register("attr_init_morph", random.choice, init_level_set_morph)
	toolbox.register("attr_smooth", GA.weighted_choice, smoothing, SMOOTH_MIN
		, SMOOTH_MAX, SMOOTH_WEIGHT)
	toolbox.register("attr_alphas", random.choice, alphas)
	toolbox.register("attr_balloon", GA.weighted_choice, balloon, BALLOON_MIN
		, BALLOON_MAX, BALLOON_WEIGHT)

	#toolbox.register()

	#toolbox.register("attr_Sigma", random.)

	func_seq = [toolbox.attr_Algo, toolbox.attr_Beta, toolbox.attr_Tol,
		toolbox.attr_Scale, toolbox.attr_Sigma, toolbox.attr_minSize,
		toolbox.attr_nSegment, toolbox.attr_compact, toolbox.attr_iterations, 
		toolbox.attr_ratio,
		toolbox.attr_kernel, toolbox.attr_maxDist, toolbox.attr_seed, 
		toolbox.attr_connect, toolbox.attr_mu, 
		toolbox.attr_lambda, toolbox.attr_dt, toolbox.attr_init_chan,
		toolbox.attr_init_morph, toolbox.attr_smooth, toolbox.attr_alphas,
		toolbox.attr_balloon]
	#Here we populate our individual with all of the 
	toolbox.register("individual", tools.initCycle, creator.Individual
		, func_seq, n=1)

	toolbox.register("population", tools.initRepeat, list, 
		toolbox.individual, 300)

	pop = toolbox.individual()
	#pop = toolbox.population()
	print (pop.fitness.valid)
	pop.fitness.values = GA.runAlgo(AllImages[0], ValImages[0], pop)
	AlgoParams = AlgorithmParams.AlgorithmParams(pop)
	print (type(AlgoParams).__name__)
	Algo = AlgorithmSpace(AlgoParams)


	print (pop.fitness.valid)
	print (pop.fitness)
	#fitness = GA.runAlgo(AllImages[0], ValImages[0], pop[0])
	hof = tools.HallOfFame(1)
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean)

	#cxpb = probability of two individuals mating
	#mutpb = probability of mutation
	#ngen = Number of generations

	cxpb, mutpb, ngen = 0.2, 0.5, 2
	#print(algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40,
#		stats=stats, halloffame=hof))
	#toolbox.population()
	#hof = tools.HallOfFame(1, similar=np.allclose)
 	
	#stats = tools.Statistics(lambda img: img.fitness.value)