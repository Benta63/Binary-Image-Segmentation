import os
import numpy as np
import cv2
import copy

class ImageData(object):

	def __init__(self, imagePath):
		self.imageName = imagePath
		self.image = cv2.imread(imagePath, 1)
		self.type = len(self.image.shape)

	#Accessors
	def getImage(self):
		return copy.deepcopy(self.image)

	def getDim(self):
		return self.type

	def getShape(self):
		return self.image.shape

	def getName(self):
		return self.imageName