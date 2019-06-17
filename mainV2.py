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

import threading
import time
import sys
import os
import random
import copy


IMAGE_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_plant'
VALIDATION_PATH = 'Image_data\\Coco_2017_unlabeled\\rgbd_label'

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

	print("Before Loop")


	thread1 = threading.Thread(target=background)
	thread1.daemon = True
	thread1.start()

	while True:

		if input() == 'quit':
			save_state()

			#Need to add other stuff as well.

			sys.exit()
		else:

			print("continue")

