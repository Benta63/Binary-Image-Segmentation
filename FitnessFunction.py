import numpy as np
import skimage.measure
import argparse
import imutils
#https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.compare_ssim

#Takes in two ndarrays of image and compares them pixel by pixel
#Returns the overall different number of pixels
def FitnessFunction(img1, img2):
	if np.array_equal(img1, img2):
		#The segmentation was perfect
		return 0
	if np.allclose(img1, img2):
		#The segmentation is very close
		return 5
	assert(img1.shape == img2.shape)

	#Comparing the Structual Similarity Index (SSIM) of two images
	ssim = skimage.measure.compare_ssim(img1, img2)
	#Comparing the Mean Squared Error
	mse = skimage.measure.compare_mse(img1, img2)
	return ssim + mse