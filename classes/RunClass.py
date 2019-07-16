import numpy as np
import os
from PIL import Image
import skimage
from skimage import segmentation
import random
from operator import attrgetter
import sys
import cv2

#https://github.com/DEAP/deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
'''
from . import AlgorithmSpace
from . import AlgorithmParams
from . import GeneticHelp as GA
from . import ImageData
from . import FileClass
'''
from classes import FileClass
from classes.FileClass import FileClass
from classes import GeneticHelp
from classes.GeneticHelp import GeneticHelp as GA
from classes import ImageData
from classes.ImageData import ImageData

IMAGE_PATH = '..\\Image_data\\Coco_2017_unlabeled\\rgbd_plant'
VALIDATION_PATH = '..\\Image_data\\Coco_2017_unlabeled\\rgbd_label'
#Image should be an input
SEED = 134
POPULATION = 10
GENERATIONS = 10

class RunClass(object):
	def __init__(self, imgPath, initImage, initValid):
		#We should do error checking in main, but let's do it again
		if (FileClass.check_dir(imgPath) == False):
			print ('ERROR: Directory \"%s\" does not exist'%IMAGE_PATH)
			sys.exit(1)
		#Getting all of the images in a manageable way	
		AllImages = [ImageData.ImageData(os.path.join(root, name)) for 
			root, dirs, files in os.walk(IMAGE_PATH) for name in files]
		self.ImgPath = AllImages
		self.Images = [ImageData.ImageData(initImage)]
		self.Validation = [ImageData.ImageData(initValid)]


	#Finds the best algorithm for the specified image
	def RunGA(self):

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


		#Here we register all the parameters to the toolbox
		SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT = 0, 1, 0.5	
		ITER = 10
		SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT = 1, 4, 0.5
		BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT = -1, 1, 0.9

		#Minimizing fitness function
		creator.create("FitnessMin", base.Fitness, weights=(minFit,))

		creator.create("Individual", list, fitness=creator.FitnessMin)

		#Functions that the GA knows
		toolbox = base.Toolbox()
		#Attribute generator
		toolbox.register("attr_bool", random.randint, 0, 1000)

		#Genetic Functions
		toolbox.register("mate", GA.skimageCrossRandom) #crossover
		toolbox.register("evaluate", GA.runAlgo) #Fitness
		toolbox.register("mutate", GA.mutate) #Mutation
		toolbox.register("select", tools.selTournament, tournsize=3) 
			#I may want to play with this selection

		#We choose the parameters, for the most part, random
		toolbox.register("attr_Algo", random.choice, AllParams[0])
		toolbox.register("attr_Beta", random.choice, AllParams[1])
		toolbox.register("attr_Tol", random.choice, AllParams[2])
		toolbox.register("attr_Scale", random.choice, AllParams[3])
		#While sigma can be any positive value, it should be small (0-1). 
		toolbox.register("attr_Sigma", RandHelp.weighted_choice, AllParams[4], SIGMA_MIN, 
			SIGMA_MAX, SIGMA_WEIGHT)
		toolbox.register("attr_minSize", random.choice, AllParams[5])
		toolbox.register("attr_nSegment", random.choice, AllParams[6])
		toolbox.register("attr_compact", random.choice, AllParams[7])

		toolbox.register("attr_iterations", int, ITER)
		toolbox.register("attr_ratio", random.choice, AllParams[8])
		toolbox.register("attr_kernel", random.choice, AllParams[9])
		toolbox.register("attr_maxDist", random.choice, AllParams[10])
		toolbox.register("attr_seed", int, SEED)
		toolbox.register("attr_connect", random.choice, AllParams[11])
		toolbox.register("attr_mu", random.choice, AllParams[12])
		toolbox.register("attr_lambda", random.choice, AllParams[13])
		toolbox.register("attr_dt", random.choice, AllParams[14])
		toolbox.register("attr_init_chan", random.choice, 
			AllParams[15])
		toolbox.register("attr_init_morph", random.choice, 
			AllParams[16])
		#smoothing should be 1-4, but can be any positive number
		toolbox.register("attr_smooth", RandHelp.weighted_choice, 
			AllParams[17], SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT)
		toolbox.register("attr_alphas", random.choice, AllParams[18])
		#Should be from -1 to 1, but can be any value
		toolbox.register("attr_balloon", RandHelp.weighted_choice, AllParams[19], 
			BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT)
		#Need to register a random seed_point and a correct new_value

		#Need to add floods (seed point)

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
		
		#We're only looking at one image for now	
		Images = [self.Images[0] for i in range(0, len(pop))]
		ValImages = [self.Validation[0] for i in range(0, len(pop))]

		#Evaluating the initial fitnesses
		fitnesses = list(map(toolbox.evaluate, Images, ValImages, pop))
	
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit
		#Algo = AlgorithmSpace(AlgoParams)
		#Let's store the best indovidual 
		hof = tools.HallOfFame(1)

		extractFits = [ind.fitness.values[0] for ind in pop]
		hof.update(pop)

		#cxpb = probability of two individuals mating
		#mutpb = probability of mutation
		#ngen = Number of generations

		cxpb, mutpb, ngen = 0.2, 0.5, 50

		gen = 0 #The initial generation

		#Let's print some statistics
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

			#Selection process
			offspring = toolbox.select(pop, len(pop))
			offspring = list(map(toolbox.clone, offspring))

			#crossover
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				#Do we crossover?
				if random.random() < cxpb:
					toolbox.mate(child1, child2)
					#The parents may be okay values so we should keep them
					#in the set
					#del child1.fitness.values
					#del child2.fitness.values
			
			#mutation

			for mutant in offspring:
				if random.random() < mutpb:
					flipProb = 0.05
					toolbox.mutate(mutant, AllVals, flipProb)
					#del mutant.fitness.values

			#Let's just evaluate the mutated and crossover individuals
			invalInd = [ind for ind in offspring if not ind.fitness.valid]
			NewImage = [AllImages[0] for i in range(0, len(invalInd))]
			NewVal = [ValImages[0] for i in range(0, len(invalInd))]
			fitnesses = map(toolbox.evaluate, NewImage, NewVal, invalInd)
			
			for ind, fit in zip(invalInd, fitnesses):
				ind.fitness.values = fit

			#Replacing the old 
			hof.update(pop)
			pop[:] = offspring

			#Evaluating the new population
			extractFits = [ind.fitness.values[0] for ind in pop]
			#And now more statistics
			leng = len(pop)
			mean = sum(extractFits) / leng
			sum1 = sum(i*i for i in extractFits)
			stdev = abs(sum1 / leng - mean **2) ** 0.5
			print(" Min: ", min(extractFits))
			print(" Max: ", max(extractFits))
			print(" Avg: ", mean)
			print(" Std ", stdev)

			if hof[0].fitness.values[0] < 0.0001:
				#This implementations should be good enough
				return(best, True)

			hof.update(extractFits)

		#Can use tools.Statistics for this stuff maybe?

		#Let's now find the 'best algorithm
		#best = self.FindBest(pop)
		best = hof[0]
		if (best.fitness.values[0] >= 0.0001):
			return(best, False)
		#And now let's get an image
		Space = AlgorithmSpace(AlgorithmParams.AlgorithmParams(imgFile, 
			best[0], best[1], best[2], best[3], best[4], best[5], 
			best[6], best[7], best[8], best[9], best[10], best[11],
			best[12], best[13], best[14], best[15][0], best[15][1],
			best[16], best[17], best[18], best[19], 'auto', best[20],
			best[21]))
		
		img = Space.runAlgo()
		
		cv2.imwrite(imgName, img)
		return(best, True)

	#Tests an algorithm on an image based on the SSIM
	#algoParams gives all the algorithm and all the parameters for said
		#algorithm
	def runAlgoOnImage(self, algoParams, img, valImg):
		Space = AlgorithmSpace(AlgorithmParams.AlgorithmParams(img,
			algoParams[0], algoParams[1], algoParams[2], algoParams[3],
			algoParams[4], algoParams[5], algoParams[6], algoParams[7],
			algoParams[8], algoParams[9], algoParams[10],
			algoParams[12], algoParams[13], algoParams[14],
			algoParams[15][0], algoParams[15][1], algoParams[16],
			algoParams[17], algoParams[18], algoParams[19], 'auto',
			algoParams[20], algoParams[21]))
		newImg = Space.runAlgo()

