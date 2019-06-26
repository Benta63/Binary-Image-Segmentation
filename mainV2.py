from classes import ImageData
from classes import AlgorithmSpace
from classes.AlgorithmSpace import AlgorithmSpace
from classes import AlgorithmParams

from classes import FileClass
from classes.FileClass import FileClass
from classes import GeneticHelp
from classes.GeneticHelp import GeneticHelp as GA

from classes import RunClass
from classes.RunClass import RunClass as RC
from classes import RandomHelp
from classes.RandomHelp import RandomHelp as RandHelp

import threading
import multiprocessing
from multiprocessing import Process, Queue
import logging
import time
import sys
from sys import stderr, stdin, stdout
import os
import random
import copy
import pickle

IMAGE_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'
VALIDATION_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_new_label'

def thread_func(name):
	logging.ingo("Thread %s is starting", name)
	time.sleep(2)
	logging.info("Thread %s is finishing", name)

def testStuff():
	print("Made a thread!!")
	return

#Need to incorporate threading
def background():
	while True:
		time.sleep(3)

def save_state():
	print("State saved")

if __name__=='__main__':
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


	#Here we register all the parameters to the toolbox
	SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT = 0, 1, 0.5	
	ITER = 10
	SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT = 1, 4, 0.5
	BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT = -1, 1, 0.9

	imageCounter = 0
	goodAlgoCounter = 0
	goodAlgo = 0

	goodEnough = False

	#Let's do some threading
	format = "%(asctime)s: %(message)s"
	#logging.basicConfig(fomat=format, level=logging.INFO, 
		#datefmt="%H:%M:%S")

	numFiles = input("Input the total number of files in your dataset ")
	print(numFiles)
	processes = multiprocessing.Queue(int(numFiles))
	p = multiprocessing.Process(target=time.sleep, args=(1000,))
	print(multiprocessing.current_process())
	#print(p, p.is_alive())

	#Do some kind of locking??
	for i in range(0,int(numFiles)):
		p = multiprocessing.Process(target=testStuff)
		processes.put(p,False)
		p.start()
		print ("Size of queue: ",processes.qsize())

	print(multiprocessing.active_children())
	cursor = 0
	print ("Possible commands are:\n\thelp -- Recreates this menu\n\t"
		   + "")
	while cursor < len(AllImages):
		#First we get the correct file
		isImg, isVal, imgFile, valFile = False, False, "", ""
		while(isImg == False and isVal == False):
			imgFile = input("Give the path to a image file in your dataset. ")
			valFile = input("Give the path to the segmented image of the last file. ")
		
			isImg = FileClass.check_dir(imgFile)
			isVal = FileClass.check_dir(valFile)
			if (isImg == False):
				print("Image file %s does not exist.\n"%imgFile)
			if (isVal == False):
				print("Segmentented image file %s does not exist.\n"%valFile)

		print("Good files")


		cursor += 1
		#If it's empty, we ignore it
		
		#Let's parse the line



		if command == 'image':
			#ImageData objects
			imageObj = ImageData.ImageData(imageFile)
			valObj = ImageData.ImageData(imageVal)
			#Let's write to the same image type that was read in
			imgType = findImageType(imageFile)
			imageName = "data\\newImage" + str(imageCounter) + imgType
			p = multiprocessing.Process(target=RC.RunGA, args=[AllVals, 
							SIGMA_MIN, SIGMA_MAX, SIGMA_WEIGHT, ITER,
							SMOOTH_MIN, SMOOTH_MAX, SMOOTH_WEIGHT, 
				  			BALLOON_MIN, BALLOON_MAX, BALLOON_WEIGHT,
				  			imgObj,valObj, imageName])
			processes.put(p, False())
			p.start()
			'''Algo, didWork = RunGA(AllVals, SIGMA_MIN, SIGMA_MAX, 
							SIGMA_WEIGHT, ITER, SMOOTH_MIN, 
							SMOOTH_MAX, SMOOTH_WEIGHT, 
				  			BALLOON_MIN, BALLOON_MAX, 
				  			BALLOON_WEIGHT, imgObj,valObj, 
				  			imageName)
			'''
			if (didWork == False):
				print("Did not converge for the algorithm")
				if goodAlgoCounter != 0:
					goodAlgoCounter -= 1
				continue
			else:
				#We found a good algorithm, We should test it on
				#another image as well
				goodAlgoCounter += 1
				goodAlgo = Algo

		break

		'''elif command == 'quit':
			#Should also get best image
			sys.exit()

		elif command == 'help':
			continue

		#if goodAlgoCounter >= 2:
		'''

	#Let's do some threading