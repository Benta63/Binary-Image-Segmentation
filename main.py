import numpy as np
import os
from PIL import Image
import skimage
from skimage import segmentation

from classes import ImageData
from classes import AlgorithmSpace
from classes.AlgorithmSpace import AlgorithmSpace

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

	AllLabels = [np.zeros(img.getShape()) for img in AllImages]

	#Some debugging statements
	assert (len(AllLabels) == len(AllImages))
	assert (AllLabels[0].shape == AllImages[0].getShape())
	assert (len(AllImages[0].getImage()) == len(AllLabels[0]))
	
	#Testing RandomWalker
	#betas = [80, 90, 100, 110, 120, 130, 140, 150]

	#So, not even the original random_walker is working, so we're manipulating the labels
	#for i in range(0, len(AllLabels[0])):
	#	if i < len(AllLabels[0])/2 :
	#		AllLabels[0][i] = 1
		#else:
		#	AllLabels[0][i] = 2

	file = open("writeData.txt", 'a+')
	for line in AllImages[0].getImage():
		for number in line:
			file.write(str(number) + " ")
		file.write('\n')
	file.write(str(AllImages[0].getImage()))
	#print(AllImages[0].getImage())
	#print(AllLabels[0])
	print(skimage.segmentation.random_walker(AllImages[0].getImage(), AllLabels[0], beta=10, mode='cg', copy=False))
	file.close()

	#img = Image.fromarray(newLabels)
	#img.save('my.png')

