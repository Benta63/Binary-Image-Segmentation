#Strongly coupled with AlgorithmParams class
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

import numpy as np
import pandas as pd
import copy
import skimage
from skimage import segmentation
from itertools import combinations

#from . import ImageData
#from . import AlgorithmParams

'''
This class will run through all the algorithms in skimage.segmentation
and change the parameters
'''

#Perhaps inherit from AlgorithmParams?
class AlgorithmSpace(object):
	def __init__(self, parameters):
		#parameters is a AlgorithmParams object

		self.params = parameters
		#Whether this is a multichannel array or grayscale
		self.channel = False
		if (self.params.getImage().getDim() > 2):
			#This is at least a 3D array, so multichannel
			self.channel = True
	

	#Algorithms
	'''
	#Random walker algorithm: 
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html
	Segments an image from set markers. Works
	on grayscale and multichannel images,

	#Returns a labeled image (ndarray)

	#Variables:
	data -> Image (ndarray)
	labels -> Same shape as data, ndarray
	beta -> float, penalization coefficient High beta = more difficult
	diffusion. as different coefficients
	may be better for different images
	mode -> string. mode is normally bf, but should use cg_mg for images smaller than 512X512
	tol is the tolerance to achieve when solving a linear system
	copy should be True. Whether or not to overwrite label array
	multichannel -- False is gray image, True if multichannel image
	return_full_prob -> should be false
	spacing -> spacing in between pixels, Going to leave this at 1 for now

	'''
	#Code for algorithms == RW
	#Not using RandomWalker because labels complicates the searchspace
		
	def runRandomWalker(self):
		#Let's deterime what mode to use
		mode = ""
		if len(self.parameters.getImage.getImage()) < 512 :
			mode = "cg_mg"
		else:
			mode = "bf"
		
		#If data is 2D, then this is a grayscale, so multichannel is 
		#false
		
		output = skimage.segmentation.random_walker(self.params.getImage().getImage()
			, self.params.getLabel(), beta=self.params.getBeta(), mode=mode,
			tol=self.params.getTolerance(), copy=True, multichannel=self.channel,
			return_full_prob=False) 
		return output

	'''
	#felzenszwalb
	#ONLY WORKS FOR RGB
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	The felzenszwalb algorithms computes a graph based on the segmentation
	Produces an oversegmentation of the multichannel using min-span tree.
	Returns an integer mask indicating the segment labels
	scale: float, higher meanse larger clusters
	sigma: float, std. dev of Gaussian kernel for preprocessing
	min_size: int, minimum component size. For postprocessing
	mulitchannel: bool, Whether the image is 2D or 3D. 2D images
		are not supported at all

	#Needs testing to find correct values'

	'''
	#Code for algorithm == FB

	def runFelzenszwalb(self):
		output = skimage.segmentation.felzenszwalb(
			self.params.getImage().getImage(), self.params.getScale(),
			self.params.getSigma(), self.params.getMinSize(),
			multichannel=self.channel)
		
		return output

	#Needs testing
	'''
	#slic
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	segments k-means clustering in Color space (x, y, z)
	#Returns a 2D or 3D array of labels

	#Variables
	image -- ndarray, input image
	n_segments -- int,  number of labels in segmented output image (approx)
		Should find a way to compute n_segments
	compactness -- float, Balances color proximity and space proximity.
		Higher values mean more weight to space proximity (superpixels
		become more square/cubic) #Recommended log scale values (0.01, 
		0.1, 1, 10, 100, etc)
	max_iter -- int, max number of iterations of k-means
	sigma -- float or (3,) shape array of floats,  width of Guassian
		smoothing kernel. For pre-processing for each dimesion of the
		image. Zero means no smoothing
	spacing -- (3,) shape float array : voxel spacing along each image
		dimension. Defalt is uniform spacing
	multichannel -- bool,  multichannel (True) vs grayscale (False)
	#Needs testing to find correct values
	
	#Code for algorithm == SC
	'''
	def runSlic(self):
		output = skimage.segmentation.slic(self.params.getImage(),
			n_segments=self.params.getSegments(), compactness=
			self.params.getCompact(), max_iter=self.params.getIters(),
			sigma=self.params.getSigma(), multichannel=self.channel)

		return output

	'''
	#quickshift
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	Segments images with quickshift clustering in Color (x,y) space
	#Returns ndarray segmentation mask of the labels
	#Variables
	image -- ndarray, input image
	ratio -- float, balances color-space proximity & image-space
		proximity. Higher vals give more weight to color-space
	kernel_size: float, Width of Guassian kernel using smoothing.
		Higher means fewer clusters
	max_dist -- float: Cut-off point for data distances. Higher 
		means fewer clusters
	return_tree -- bool: Whether to return the full segmentation
		hierachy tree and distances. Set as False
	sigma -- float: Width of Guassian smoothing as preprocessing.
		Zero means no smoothing
	convert2lab -- bool: leave alone
	random_seed -- int, Random seed used for breacking ties. 
	'''
	#Code for algorithm == QS

	def runQuickShift(self):
		output = skimage.segmentation.quickshift(
			self.params.getImage().getImage(), ratio=self.params.getRatio()
			, kernel_size=self.params.getKernel(), max_dist=
			self.params.getMaxDist(), sigma=self.params.getSigma(),
			random_seed=self.params.getSeed())
		
		return output


	'''
	#Watershed
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
	Uses user-markers. treats markers as basins and 'floods' them.
	Especially good if overlapping objects. 
	#Returns a labeled image ndarray
	#Variables
	image -> ndarray, input array
	markers -> int, or int ndarray same shape as image: markers indicating
		'basins'
	connectivity -> ndarray, indicates neighbors for connection
	offset -> array, same shape as image: offset of the connectivity
	mask -> ndarray of bools (or 0s and 1s): 
	compactness -> float, compactness of the basins Higher values make more
		regularly-shaped basin
	'''
	#Not using connectivity or markers params as arrays as they would
	#expand the search space too much.
	#Code for algorithm = WS

	def runWaterShed(self):
		output = skimage.segmentation.watershed(
			self.params.getImage().getImage(), 
			compactness=self.params.getCompact())
		return output
	
	'''
	#chan_vese
	#ONLY GRAYSCALE
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_chan_vese.html
	Segments objects without clear boundaries
	#Returns: segmentation array of algorithm. Optional: When the algorithm converges
	#Variables
	image -> ndarray grayscale image to be segmented
	mu -> float, 'edge length' weight parameter. Higher mu vals make a 'round edge'
		closer to zero will detect smaller objects
	lambda1 -> float 'diff from average' weight param to determine if output region 
		is True. If lower than lambda1, the region has a larger range of values than the other
	lambda2 -> float 'diff from average' weight param to determine if output region
		is False. If lower than lambda1, the region will have a larger range of values
	Note: Typical values for mu are from 0-1. 
	Note: Typical values for lambda1 & lambda2 are 1. If the background is
		'very' different from the segmented values, in terms of distribution,
		then the lambdas should be different from eachother
	tol: positive float, typically (0-1), very low level set variation 
		tolerance between iterations.
	max_iter: uint,  max number of iterations before algorithms stops
	dt: float, Multiplication factor applied at the calculations step
	init_level_set: str/ndarray, defines starting level set used by
		algorithm. Accepted values are:
		'checkerboard': fast convergence, hard to find implicit edges
		'disk': Somewhat slower convergence, more likely to find
			implicit edges
		'small disk': Slowest convergence, more likely to find implicit
			edges
		can also be ndarray same shape as image
	extended_output: bool, If true, adds more returns 
		(Final level set & energies)
	'''
	#Code for Algorithm = CV
	
	def runChanVese(self):
		output = skimage.segmentation.chan_vese(
			self.params.getImage().getImage(), mu=self.params.getMu,
				lambda1=self.params.getLambdaOne(), lambda2=
				self.params.getLambdaTwo(), tol=self.params.getTolerance()
				, max_iter=self.params.getIters(), dt=self.params.getDT()
				, init_level_set=self.params.getInitLvlSetChan())

		return output
	'''

	#morphological_chan_vese
	#ONLY WORKS ON GRAYSCALE
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_chan_vese
	Active contours without edges. Can be used to segment images/volumes without good borders. Required that the 
		inside of the object looks different than outside (color, shade, darker).
	#Returns Final segmention
	#Variables:
	image -> ndarray of grayscale image
	iterations -> uint, number of iterations to run
	init_level_set: str, or array same shape as image. Accepted string
		values are:
		'checkerboard': Uses checkerboard_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set
		returns a binary level set of a checkerboard
		'circle': Uses circle_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.circle_level_set
		Creates a binary level set of a circle, given a radius and a center

	smoothing: uint, number of times the smoothing operator is applied
		per iteration. Usually around 1-4. Larger values make stuf smoother
	lambda1: Weight param for outer region. If larger than lambda2, outer
		region will give larger range of values than inner value
	lambda2: Weight param for inner region. If larger thant lambda1, inner
		region will have a larger range of values than outer region
	'''
	#Code for algorithm = MCV

	def runMorphChanVese(self):
		output = skimage.segmentation.morphological_chan_vese(
			self.params.getImage().getImage(), init_level_set=
			self.params.getInitLvlSetMorph(), smoothing=
			self.params.getSmoothing(), lambda1=self.params.getLambdaOne()
			, lambda2=self.params.getLambdaTwo())
		return output

	'''
	#morphological_geodesic_active_contour
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.morphological_geodesic_active_contour
	Uses an image from inverse_gaussian_gradient in order to segment
		object with visible, but noisy/broken borders
	#inverse_gaussian_gradient
	https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.inverse_gaussian_gradient
	Compute the magnitude of the gradients in an image. returns a
		preprocessed image suitable for above function
	#Returns ndarray of segmented image
	#Variables
	gimage: array, preprocessed image to be segmented
	iterations: uint, number of iterations to run
	init_level_set: str, array same shape as gimage. If string, possible
		values are:
		'checkerboard': Uses checkerboard_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.checkerboard_level_set
		returns a binary level set of a checkerboard
		'circle': Uses circle_level_set
		https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.circle_level_set
		Creates a binary level set of a circle, given a radius and a center
	smoothing: uint, number of times the smoothing operator is applied per
		iteration. Usually 1-4, larger values have smoother segmentation
	threshold: Areas of image with a smaller value than the threshold
		are borders
	balloon: float, guides contour of low-information parts of image, 	
	'''
	#Code for algorithm = AC

	def runMorphGeodesicActiveContour(self):
		#We run the inverse_gaussian_gradient to get the image to use
		gimage = skimage.segmentation.inverse_gaussian_gradient(
			self.params.getImage().getImage(), self.params.getAlpha(), 
			self.params.getSigma())
		print("gimage")
		output = skimage.segmentation.morphological_geodesic_active_contour(
			gimage, self.params.getIters(), self.params.getInitLvlSetMorph(),
			smoothing= self.params.getSmoothing(), 
			threshold=self.params.getThresh(), balloon= 
			self.params.getBalloon())
		print("outputting")

		return output
	'''
	#flood
	#DOES NOT SUPPORT MULTICHANNEL IMAGES
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
	Uses a seed point and to fill all connected points within/equal to
		a tolerance around the seed point
	#Returns a boolean array with 'flooded' areas being true
	#Variables
	image: ndarray, input image
	seed_point: tuple/int, x,y,z referring to starting point for flood fill
	selem: ndarray of 1's and 0's, Used to determine neighborhood of
		each pixel
	connectivity: int, Used to find neighborhood of each pixel. Can use this
		or selem.
	tolerance: float or int, If none, adjacent values must be equal to 
		seed_point. Otherwise, how likely adjacent values are flooded.
	'''
	#Code for algorithm = FD

	def runFlood(self):
		output = skimage.segmentation.flood(self.params.getImage().getImage()
			, self.params.getSeedPoint(), selem=self.params.getSelem(),
			connectivity=self.params.getConnect(), tolerance=
			self.params.getTolerance())
		return output

	'''
	#flood_fill
	https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_floodfill.html
	Like a paint-bucket tool in paint. Like flood, but changes the color equal to
		new_type
	#Returns A filled array of same shape as the image
	#Variables
	image: ndarray, input image
	seed_point: tuple or int, starting point for filling (x,y,z)
	new_value: new value to set the fill to (e.g. color). Must agree
		with image type
	selem: ndarray, Used to find neighborhood of filling
	connectivity: Also used to find neighborhood of filling if selem is
		None
	tolerance: float or int, If none, adjacent values must be equal to 
		seed_point. Otherwise, how likely adjacent values are flooded.
	inplace: bool, If true, the flood filling is applied to the image,
		if False, the image is not modified. Default False, don't change
	'''
	#Code for algorithm == FF

	def runFloodFill(self):
		output = skimage.segmentation.flood_fill(
			self.params.getImage().getImage(), self.params.getSeedPoint()
			, self.params.getNewVal(), selem=self.params.getSelem(),
			connectivity=self.params.getConnect(), tolerance=
			self.params.getTolerance())

		return output