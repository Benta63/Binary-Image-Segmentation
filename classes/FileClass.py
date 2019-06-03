import cv2
import os
#from . import ImageData

class FileClass(object):
	#Writes an image to a file
	#img is an ImageData object, fileName is the name of the file
	def writeImage(img, fileName):
		cv2.imwrite(fileName, img.getImage())

	#Writes the text representation of an image to a file
	#img is an ImageData object, txtName is the name of the file
	def writeData(img, txtName):
		file = open(txtName, 'w+')
		for line in img.getImage():
			for number in line:
				file.write(str(number) + " ")
			file.write('\n')
		file.write(str(img.getImage()))
		file.close()

	#Checks if a directory exists
	def check_dir(path):
		directory = os.path.dirname(path)
		if not os.path.exists(path):
			return False
		return True

	#Checks if a directory exists. If it doesn't creates it
	def check_and_create(path):
		if (check_dir(path) == False):
			os.makedirs(path)
