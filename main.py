#TODO: MAJOR: Seed search space
#EMPHASIZE: We knew that it would not converge. good first step. future work is cool. What are bad numbers
#Takeaways: built framework, validated frame, verified hypothesis (it didn't work), attempted scaling.
#Comment in readme about pillow using 3.5.3. Can change pillow to matplotlib.
#Add to Tasks?? about changeing pillow to matplotlib
#Talk about finding baseline. "Dr. Colbry says that this is a complete success"

import numpy as np
import os
from PIL import Image
import skimage
import random
from operator import attrgetter
import sys
import pickle

#https://github.com/DEAP/deap
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from skimage import segmentation
import scoop
from scoop import futures
import cv2
import time

from GAHelpers import ImageData
from GAHelpers import AlgorithmSpace
from GAHelpers.AlgorithmSpace import AlgorithmSpace
from GAHelpers import AlgorithmParams

from GAHelpers import FileClass
from GAHelpers.FileClass import FileClass
from GAHelpers import AlgorithmHelper
from GAHelpers.AlgorithmHelper import AlgoHelp
from GAHelpers import GeneticHelp
from GAHelpers.GeneticHelp import GeneticHelp as GA
from GAHelpers import RandomHelp
from GAHelpers.RandomHelp import RandomHelp as RandHelp



#TODO: Make input params changeable by input arguments
IMAGE_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'

#TODO: Change validation to ground_truth
GROUNDTRUTH_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_new_label'
#Quickshift relies on a C long. As this is platform dependent, I will change
#this later. 
SEED = 134
POPULATION = 1000
GENERATIONS = 100
MUTATION = 0
FLIPPROB = 0
CROSSOVER = 0


if __name__ == '__main__':
	initTime = time.time()
	#To determine the seed for debugging purposes
	seed = random.randrange(sys.maxsize)
	random.seed(seed)

	print("Seed was:", seed)

	#Will later have user input to find where the images are

	#Checking the directories
	if (FileClass.check_dir(IMAGE_PATH) == False):
		print ('ERROR: Directory \"%s\" does not exist'%IMAGE_PATH)
		sys.exit(1)

	if(FileClass.check_dir(GROUNDTRUTH_PATH) == False):
		print("ERROR: Directory \"%s\" does not exist"%VALIDATION_PATH)
		sys.exit(1)

	#TODO: Take these out and change to getting one image

	#Making an ImageData object for all of the regular images
	AllImages = [ImageData.ImageData(os.path.join(root, name)) for 
		root, dirs, files in os.walk(IMAGE_PATH) for name in files]

	#Making an ImageData object for all of the labeled images
	GroundImages = [ImageData.ImageData(os.path.join(root, name)) for
		root, dirs, files in os.walk(VALIDATION_PATH) for name in
		files]

	#Let's get all possible values in lists

	#TODO: Make seed point input parameter
	#Getting the seedpoint for floodfill
	#Dimensions of the image
	x = AllImages[0].getShape()[0]
	y = AllImages[0].getShape()[1]

	#Multichannel?
	z = 0
	if (AllImages[0].getDim() > 2):
		z = AllImages[0].getShape()[2] -1

	seedX = [ix for ix in range(0, x)]
	seedY = [iy for iy in range(0, y)]
	seedZ = [z]


	#ADD VALUES FOR NEW PARAMETERS HERE
	#Used in mutate
	AllVals = AlgoHelp().allVals()

	if len(AllVals) == 22:
		#We can just put the seed point at the end
		AllVals.append(seedX)
		AllVals.append(seedY)
		AllVals.append(seedZ)
	else:
		AllVals.insert(22, seedX)
		AllVals.insert(23, seedY)
		AllVals.insert(24, seedZ)

	'''[Algos, betas, tolerance, scale, sigma, min_size,
			  n_segments, compactness, iterations, ratio, kernel, 
			  max_dists, random_seed, connectivity, mu, Lambdas, dt,
			  init_level_set_chan, init_level_set_morph, smoothing,
			  alphas, balloon, seedX, seedY, seedZ]
	'''
	#Using the DEAP genetic algorithm to make One Max
	#https://deap.readthedocs.io/en/master/api/tools.html
	#Creator factory builds new classes


	
	toolbox = AlgoHelp().makeToolbox(POPULATION, seedX, seedY, seedZ)

	#Here we check if we have a saved state
	#From: https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html
	pop = None

	#Keeps track of the best individual from any population
	hof = None
	start_gen = 0

	#TODO: Use copy better
	Images = [AllImages[0] for i in range(0, POPULATION)]
	GroundImages = [GroundImages[0] for i in range(0, POPULATION)]
	### TODO: Implement a save-state function:
	# https://deap.readthedocs.io/en/master/tutorials/advanced/checkpoint.html

	'''try:
		#A file name was given, so we load it
		with open(sys.argv[1], "r") as cp_file:
			cp = pickle.load(cp_file)
		pop = cp["population"]
		fitnesses = list(map(toolbox.evaluate, Images, GroundImages, pop))
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit

		start_gen = cp["generation"]
		hof = cp["halloffame"]
		random.setstate(cp["rndstate"])
	except IndexError:
		pop = toolbox.population()
		fitnesses = list(map(toolbox.evaluate, Images, GroundImages, pop))
	
		for ind, fit in zip(pop, fitnesses):
			ind.fitness.values = fit
		hof = tools.HallOfFame(1)
	'''
	pop = toolbox.population()
	fitnesses = list(map(toolbox.evaluate, Images, GroundImages, pop))

	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	hof = tools.HallOfFame(1)
	#Algo = AlgorithmSpace(AlgoParams)
	extractFits = [ind.fitness.values[0] for ind in pop]
	hof.update(pop)

	#stats = tools.Statistics(lambda ind: ind.fitness.values)
	#stats.register("avg", np.mean)

	#cxpb = probability of two individuals mating
	#mutpb = probability of mutation
	#ngen = Number of generations

	cxpb, mutpb, ngen = CROSSOVER, MUTATION, GENERATIONS
	gen = 0

	leng = len(pop)
	mean = sum(extractFits) / leng
	sum1 = sum(i*i for i in extractFits)
	stdev = abs(sum1 / leng - mean **2) ** 0.5
	print(" Min: ", min(extractFits))
	print(" Max: ", max(extractFits))
	print(" Avg: ", mean)
	print(" Std: ", stdev)
	print(" Size: ", leng)
	print(" Time: ", time.time() - initTime)

	#Beginning evolution
	pastPop = pop
	pastMean = mean
	pastMin = min(extractFits)

	BestAvgs = []

	#while min(extractFits) > 0 and gen < ngen:
	#TODO: Think about changing algorithm to:
	#Calc fitness
	#Update population
	while gen < ngen:

		gen += 1
		print ("Generation: ", gen)
		offspring = toolbox.select(pop, len(pop))
		offspring = list(map(toolbox.clone, offspring))

		#crossover
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			#Do we crossover?
			if random.random() < cxpb:
				toolbox.mate(child1, child2)
				#The parents may be okay values so we should keep them
				#in the set
				del child1.fitness.values
				del child2.fitness.values
		
		#mutation
		for mutant in offspring:
			if random.random() < mutpb:
				flipProb = FLIPPROB
				toolbox.mutate(mutant, AllVals, flipProb)
				del mutant.fitness.values

		#Let's just evaluate the mutated and crossover individuals
		invalInd = [ind for ind in offspring if not ind.fitness.valid]
		NewImage = [AllImages[0] for i in range(0, len(invalInd))]
		NewVal = [GroundImages[0] for i in range(0, len(invalInd))]
		fitnesses = map(toolbox.evaluate, NewImage, NewVal, invalInd)
		
		for ind, fit in zip(invalInd, fitnesses):
			ind.fitness.values = fit

		#Replacing the old population
		pop[:] = offspring
		hof.update(pop)
		extractFits = [ind.fitness.values[0] for ind in pop]
		#Evaluating the new population
		leng = len(pop)
		mean = sum(extractFits) / leng
		BestAvgs.append(mean)
		sum1 = sum(i*i for i in extractFits)
		stdev = abs(sum1 / leng - mean **2) ** 0.5
		print(" Min: ", min(extractFits))
		print(" Max: ", max(extractFits))
		print(" Avg: ", mean)
		print(" Std: ", stdev)
		print(" Size: ", leng)
		print(" Time: ", time.time() - initTime)
		#Did we improve the population?
		pastPop = pop
		pastMin = min(extractFits)
		if (mean >= pastMean):
			#This population is worse than the one we had before

			if hof[0].fitness.values[0] <= 0.0001:
				#The best fitness function is pretty good
				break
			else:
				continue
		pastMean = mean

		#TODO: use tools.Statistics for this stuff
	
	#We ran the population 'ngen' times. Let's see how we did:
	#Now let's checkpoint
	cp = dict(population=pop, generation=gen, halloffame=hof, rndstate=random.getstate())
	best = hof[0]
	
	with open("checkpoint_name.pkl", "wb") as cp_file:
		pickle.dump(cp, cp_file)
	print("Best Fitness: ", hof[0].fitness.values)
	print(hof[0])

	finalTime = time.time()
	diffTime = finalTime - initTime
	print("Final time: %.5f seconds"%diffTime)

	#And let's run the algorithm to get an image
	Space = AlgorithmSpace(AlgorithmParams.AlgorithmParams(AllImages[0], best))
	img = Space.runAlgo()
	cv2.imwrite("dummy.png", img)

	#Let's put the best algos into a file. Can later graph with matplotlib.
	file = open("newfile.txt", "a+")
	for i in BestAvgs:
		file.write(str(i) + "\n")
	file.close()
	