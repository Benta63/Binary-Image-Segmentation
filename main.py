import numpy as np
import os
from PIL import Image
import skimage
from skimage import segmentation
import cv2

from classes import ImageData
from classes import AlgorithmSpace
from classes.AlgorithmSpace import AlgorithmSpace

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
