import numpy as np
import os
from PIL import Image
import skimage
import random
from operator import attrgetter
import sys

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


IMAGE_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'
VALIDATION_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_label'
SEED = 134
POPULATION = 10
GENERATIONS = 10


if __name__ == '__main__':

	#To determine the seed for debugging purposes
	seed = random.randrange(sys.maxsize)
	rng = random.Random(seed)
	print("Seed was:", seed)
	#seed = SEED

	#Will later have user input to find where the images are

	#Checking the directories
	if (FileClass.check_dir(IMAGE_PATH) == False):
		print ('ERROR: Directory \"%s\" does not exist'%IMAGE_PATH)
		sys.exit(1)

	if(FileClass.check_dir(VALIDATION_PATH) == False):
		print("ERROR: Directory \"%s\" does not exist"%VALIDATION_PATH)
		sys.exit(1)

	#Making an ImageData object for all of the regular images
	AllImages = [ImageData.ImageData(os.path.join(root, name)) for 
		root, dirs, files in os.walk(IMAGE_PATH) for name in files]

	#Making an ImageData object for all of the labeled images
	ValImages = [ImageData.ImageData(os.path.join(root, name)) for
		root, dirs, files in os.walk(VALIDATION_PATH) for name in
		files]

	#Let's get all possible values in lists
	Algos = ['FB','SC','WS','CV','MCV','AC'] #Need to add floods
	#Quickshift(QS) takes a long time, so I'm taking it out for now.
	betas = [i for i in range(0,10000)]
	tolerance = [float(i)/1000 for i in range(0,1000,1)]
	scale = [i for i in range(0,10000)]
	sigma = [float(i)/100 for i in range(0,10,1)]
	#Sigma should be weighted more from 0-1
	min_size = [i for i in range(0,10000)]
	n_segments = [i for i in range(2,10000)]
	iterations = [10, 10]
	ratio = [float(i)/100 for i in range(0,100)]
	kernel = [i for i in range(0,10000)]
	max_dists = [i for i in range(0,10000)]
	random_seed = [134]
	connectivity = [i for i in range(0, 9)] #How much a turtle likes
	#its neighbors
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
	#For flood and flood_fill, which I will add later
	seed_point = [] #x,y,z coordinate
	new_value = ""
	AllVals = [Algos, betas, tolerance, scale, sigma, min_size,
			  n_segments, compactness, iterations, ratio, kernel, 
			  max_dists, random_seed, connectivity, mu, Lambdas, dt,
			  init_level_set_chan, init_level_set_morph, smoothing,
			  alphas, balloon]

	#Using the DEAP genetic algorithm to make One Max
	#https://deap.readthedocs.io/en/master/api/tools.html
	#Creator factory builds new classes

	#random.seed(34)

	#Minimizing fitness function
	creator.create("FitnessMin", base.Fitness, weights=(-0.001,))

	creator.create("Individual", list, fitness=creator.FitnessMin)
	
	#The functions that the GA knows
	toolbox = base.Toolbox()
	#Attribute generator
	toolbox.register("attr_bool", random.randint, 0, 1000)
	
	#Genetic functions
	toolbox.register("mate", GA.skimageCrossRandom) #crossover
	toolbox.register("evaluate", GA.runAlgo) #Fitness
	toolbox.register("mutate", GA.mutate) #Mutation
	toolbox.register("select", tools.selTournament, tournsize=3) 
	#Selection

	#May want to later do a different selection process

	
	#Here we register all the parameters to the toolbox
	SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT = 0, 1, 0.5	
	ITER = 10
	SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT = 1, 4, 0.5
	BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT = -1, 1, 0.9

	#We choose the parameters, for the most part, random
	toolbox.register("attr_Algo", random.choice, Algos)
	toolbox.register("attr_Beta", random.choice, betas)
	toolbox.register("attr_Tol", random.choice, tolerance)
	toolbox.register("attr_Scale", random.choice, scale)
	#While sigma can be any positive value, it should be small (0-1). 
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
	toolbox.register("attr_init_chan", random.choice, 
		init_level_set_chan)
	toolbox.register("attr_init_morph", random.choice, 
		init_level_set_morph)
	#smoothing should be 1-4, but can be any positive number
	toolbox.register("attr_smooth", GA.weighted_choice, smoothing, 
		SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT)
	toolbox.register("attr_alphas", random.choice, alphas)
	#Should be from -1 to 1, but can be any value
	toolbox.register("attr_balloon", GA.weighted_choice, balloon, 
		BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT)
	#Need to register a random seed_point and a correct new_value
	
	#tools.initCycle
	#Container: data type
	#func_seq: List of function objects to be called in order to fill 
	#container
	#n: number of times to iterate through list of functions
	#Returns: An instance of the container filled with data returned 
	#from functions
	func_seq = [toolbox.attr_Algo, toolbox.attr_Beta, toolbox.attr_Tol,
		toolbox.attr_Scale, toolbox.attr_Sigma, toolbox.attr_minSize,
		toolbox.attr_nSegment, toolbox.attr_compact, 
		toolbox.attr_iterations, toolbox.attr_ratio,
		toolbox.attr_kernel, toolbox.attr_maxDist, toolbox.attr_seed, 
		toolbox.attr_connect, toolbox.attr_mu, 
		toolbox.attr_lambda, toolbox.attr_dt, toolbox.attr_init_chan,
		toolbox.attr_init_morph, toolbox.attr_smooth, 
		toolbox.attr_alphas, toolbox.attr_balloon]
	
	#Here we populate our individual with all of the parameters
	toolbox.register("individual", tools.initCycle, creator.Individual
		, func_seq, n=1)

	#And we make our population
	toolbox.register("population", tools.initRepeat, list, 
		toolbox.individual, n=POPULATION)

	pop = toolbox.population()
	
	
	Images = [AllImages[0] for i in range(0, len(pop))]
	ValImages = [ValImages[0] for i in range(0, len(pop))]

	fitnesses = list(map(toolbox.evaluate, Images, ValImages, pop))
	
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	#Algo = AlgorithmSpace(AlgoParams)
	extractFits = [ind.fitness.values[0] for ind in pop]

	#print (pop.fitness.valid)
	#print (pop.fitness)
	#fitness = GA.runAlgo(AllImages[0], ValImages[0], pop[0])
	#hof = tools.HallOfFame(1)
	#stats = tools.Statistics(lambda ind: ind.fitness.values)
	#stats.register("avg", np.mean)


	#cxpb = probability of two individuals mating
	#mutpb = probability of mutation
	#ngen = Number of generations

	cxpb, mutpb, ngen = 0.2, 0.5, 50
	gen = 0

	leng = len(pop)
	mean = sum(extractFits) / leng
	sum1 = sum(i*i for i in extractFits)
	stdev = abs(sum1 / leng - mean **2) ** 0.5
	print(" Min: ", min(extractFits))
	print(" Max: ", max(extractFits))
	print(" Avg: ", mean)
	print(" Std ", stdev)
	#Beginning evolution
	while min(extractFits) > 0 and gen < ngen:
		gen += 1
		print ("Generation: ", gen)
		offspring = toolbox.select(pop, len(pop))
		offspring = list(map(toolbox.clone, offspring))

		#crossover
		#Two point crossover won't work as not all the values are the 
		#same
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			#Do we crossover?
			if random.random() < cxpb:
				toolbox.mate(child1, child2)
				#The parents may be okay values so we should keep them
				#in the set
				#del child1.fitness.values
				#del child2.fitness.values
		
		#mutation
		#Right now we don't have a working mutation function
		for mutant in offspring:
			if random.random() < mutpb:
				flipProb = 0.05
				toolbox.mutate(mutant, AllVals, flipProb)
				del mutant.fitness.values

		#Let's just evaluate the mutated and crossover individuals
		invalInd = [ind for ind in offspring if not ind.fitness.valid]
		NewImage = [AllImages[0] for i in range(0, len(invalInd))]
		NewVal = [ValImages[0] for i in range(0, len(invalInd))]
		fitnesses = map(toolbox.evaluate, NewImage, NewVal, invalInd)
		
		for ind, fit in zip(invalInd, fitnesses):
			ind.fitness.values = fit
		print("Got the fitnesses")
		#Replacing the old population
		pop[:] = offspring

		extractFits = [ind.fitness.values[0] for ind in pop]
		#Evaluating the new population
		leng = len(pop)
		mean = sum(extractFits) / leng
		sum1 = sum(i*i for i in extractFits)
		stdev = abs(sum1 / leng - mean **2) ** 0.5
		print(" Min: ", min(extractFits))
		print(" Max: ", max(extractFits))
		print(" Avg: ", mean)
		print(" Std ", stdev)

		#Can use tools.Statistics for this stuff maybe?


	#We ran the population 'n' times. Let's see how we did:

	best = min(pop, key=attrgetter("fitness"))
	#And let's run the algorithm to get an image
	Space = AlgorithmSpace(AlgorithmParams.AlgorithmParams(AllImages[0], 
		best[0], best[1], best[2], best[3], best[4], best[5], best[6], 
		best[7], best[8], best[9], best[10], best[11], best[12], 
		best[13], best[14], best[15][0], best[15][1], best[16], 
		best[17], best[18], best[19], 'auto', best[20], best[21])
	)

	img = Space.runAlgo()
	cv2.imwrite("dummy.png", img)


