<<<<<<< HEAD
#Strongly coupled with AlgorithmParams class
=======
#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

>>>>>>> master
import numpy as np
import pandas as pd
import copy
import skimage
from skimage import segmentation
from itertools import combinations

from classes import ImageData
from classes import AlgorithmParams

'''
This class will run through all the algorithms in skimage.segmentation
and change the parameters
'''

#Perhaps inherit from AlgorithmParams?
class AlgorithmSpace(object):
<<<<<<< HEAD
	def __init__(self, parameters):
		#parameters is a AlgorithmParams object

		self.params = parameters
		#Whether this is a multichannel array or grayscale
		self.channel = False
		if (self.params.getImage().getType() > 2):
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
	diffusion. beta will most likely be a list, as different coefficients
	may be better for different images
	mode -> string. mode is normally bf, but should use cg_mg for images smaller than 512X512
	tol is the tolerance to achieve when solving a linear system
	copy should be True. Whether or not to overwrite label array
	multichannel -- False is gray image, True if multichannel image
	return_full_prob -> should be false
	spacing -> spacing in between pixels, Going to leave this at 1 fr now

	'''
	#Code for algorithms == RW
	#Not using RandomWalker because labels complicates the searchspace
	def runRandomWalker(self):
		#Let's deterime what mode to use
		mode = ""
		if len(self.params.getdata.getImage().getImage()) < 512 :
=======
	def __init__(algoParam, self):
		#image is a ImageData object
		self.parameters = algoParam
		
	

	#Algorithms
	#Comments are: Variable -- Type: explanation
	#Need a better way to find labels

	#Random walker algorithm: For gray-level images, 
	#Variables:
	#data -> Image (ndarray),
	#labels --> Same shape as data, ndarray
	#beta --> penalization coefficient?? High beta = more difficult diffusion
	#beta will most likely be a list, as different coefficients may be better for different images
	#mode is normally bf, but should use cg_mg for images smaller than 512X512
	#tol is the ????
	#copy should be True
	#multichannel -- False is gray image, True if multichannel
	#return_full_prob -> should be false
	#spacing -> Start at none

	#Returns a list of labels for each beta value in the beta list
	def runRandomWalker(self):
		#Let's deterime what mode to use
		mode = ""
		if len(self.parameters.getImage.getImage()) < 512 :
>>>>>>> master
			mode = "cg_mg"
		else:
			mode = "bf"
		
		#If data is 2D, then this is a grayscale, so multichannel is 
		#false
<<<<<<< HEAD
		
		output = skimage.segmentation.random_walker(self.params.getImage()
			, self.params.getLabel(), beta=self.params.getBeta(), mode=mode,
			tol=self.params.getTolerance, copy=True, multichannel=self.channel,
			return_full_prob=False) 
		return output

	'''
	#felzenszwalb
	#ONLY WORKS FOR RGB
	https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_segmentations.html
	Computes a graph based on the segmentation
=======
		channel = False
		if (self.parameters.getImage.getDim() > 2):
			#This is at least a 3D array, so multichannel
			channel = True
		return skimage.segmentation.random_walker()
		#for b in beta:
			#for t in tol:
			#tol is after mode and before multichannel
			
		#	new_labels.append(scimage.segmentation.random_walker(data.getImage(), labels, b, mode, copy=False, multichannel=channel, return_full_prob=True))

		return new_labels

	#Needs testing. Needs understanding
	#Active contour model algorithm. 
	#Variables:
	#image: ndarray of input image
	#snake: ndarray, start point. Should vary
	#alpha: float, snake length/shape param. Higher vals make snake contract faster
	#beta: float snake smoothness?? Higher value makes smoother snakes
	#w_line: float controls attraction to brightness. Use negative values to attract towards darker regions
	#w_edge: float controls attraction to edges. Negative = repels snake from edges
	#gamma: float time-step
	'''bc: Boundary conditions for worm. ‘periodic’ attaches the two ends of the snake,
	 ‘fixed’ holds the end-points in place, and ‘free’ allows free movement of the ends. 
	 ‘fixed’ and ‘free’ can be combined by parsing ‘fixed-free’, ‘free-fixed’. Parsing 
	 ‘fixed-fixed’ or ‘free-free’ yields same behaviour as ‘fixed’ and ‘free’, respectively.
	'''
	#make_px_move: float max pixel distance move per iterations
	#max_iterations: int
	#convergence: float convergence criteria

	#Returns ndarray, optimized snake same shape as input snake
	#Should make lists of alpha, beta, w_line, w_edge, gamma, make_px_move, max_iterations, convergence vals to try
	def runActiveContour(image, snake, alpha, beta, w_line, w_edge, gamma, 
		max_px_move, convergence):
		new_snakes = []
		#active_contour has a bc variable with these five options
		BC = ['periodic', 'fixed', 'free', 'fixed-free', 'free-fixed']
		#Shuffling all the variables and finding the snake for each
		new_snakes = [skimage.segmentation.active_contour(image.getImage(),
			snake=snake, alpha=a, beta=b, w_line=wL, w_edge=wE, gamma=g, bc=bc,
			max_px=maxPx, max_iterations=ITERATIONS, convergence=conv) for a in
			alpha for b in beta for wL in w_Line for wE in w_edge for
			g in gamma for maxPx in max_px_move for conv in convergence]

		return new_snakes


	#Needs testing to find correct values
	'''The felzenszwalb algorithms computes a graph based on the segmentation
>>>>>>> master
	Produces an oversegmentation of the multichannel using min-span tree.
	Returns an integer mask indicating the segment labels
	#Variables
	image -- ndarray Input image
	scale -- float,  Higher means larger clusters
	sigma -- float, st. dev of width of a Guassian kernel by preprocessing
	min_size -- int,  min component size using postprocessing
	#multichannel -- True = multichannel, False = Grayscale
		Don't need to mess with this. If False, image is grayscale
	pass in lists: scale, sigma, min_size

	'''
<<<<<<< HEAD
	#Code for algorithm == FB

	def runFelzenszwalb(self):
=======
	def runFelzenszwalb(image, scale, sigma, min_size):
		graphs = [skimage.segmentation.felzenszwalb(image.getImage(),
			scale=sc, sigma=si, min_size=m_s) for sc in scale for si
			in sigma for m_s in min_size]
		return graphs
>>>>>>> master

		output = skimage.segmentation.felzenszwalb(
			self.params.getImage().getImage(), self.params.getScale(),
			self.params.getSigma(), self.params.getMinSize(),
			multichannel=self.channel)
		
		return output

<<<<<<< HEAD
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
=======
	#Needs testing to find correct values
	'''slic algorithms segments k-means clustering in Color space (x, y, z)
	Returns a 2D or 3D array of labels
	#Variables
	image -- ndarray
	n_segments -- int = number of labels in segmented output image (approx)
		Should usually be 2? 
	compactness -- Balances color proximity and space proximity. Higher values 
		mean more weight to space proximity (superpixels become more square/cubic)
		#Recommended log scale values (0.01, 0.1, 1, 10, 100, etc)
	max_iter -- int max number of iterations of k-means
	sigma -- float or (3,) shape array of floats: wdth of Guassian smoothing kernel
		for pre-processing for each dimesion of the image. Zero means no smoothing
	spacing -- (3,) shape float array : voxel spacing along each image dimension.
		defalt is uniform spacing
	multichannel -- bool: multichannel vs grayscale
>>>>>>> master
	convert2lab -- bool: Whether the image-space should be converted to Lab colorspace
		before segmentation. Input image must be RGB to be true. If multichannel is true,
		this is also true.
	enforce_connectivity -- bool: whether the generated segments are connected or not
		don't need to mess with this
	min_size_factor -- float: proportion of minimum segmentation size to be removed with
		respect to the supposed segment size. 'depth*width*height/n_segments'
	max_size factor -- proportion of max size connected segment size
	slic_zero -- bool: run SLIC-zero, the zero parameter mode of SLIC
	'''
	#Code for algorith == SC

	def runSlic(self):
		output = skimage.segmentation.slic(self.params.getImage(),
			n_segments=self.params.getSegments(), compactness=
			self.params.getCompact(), max_iter=self.params.getIters(),
			sigma=self.params.getSigma, multichannel=self.channel)

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
<<<<<<< HEAD
	#Code for algorithm == QS

	def runQuickShift(self):
		output = skimage.segmentation.quickshift(
			self.params.getImage().getImage(), ratio=self.params.getRatio()
			, kernel_size=self.params.getKernel(), max_dist=
			self.params.getMaxDist(), sigma=self.params.getSigma(),
			random_seed=self.params.getSeed())
		
		return output
=======
	def runSlic(image, n_segments, compactness, sigma):

		channel = False
		if (image.getShape()[0] > 2):
			channel = True
		labels = [skimage.segmentation.slic(image.getImage(), n_segments=seg, 
			compactness=comp, max_iter=ITERATIONS, sigma=s, slic_zero=False) 
			for seg in n_segments for comp in compactness for s in sigma]
		[labels.append(skimage.segmentation.slic(image.getImage(), n_segments=seg,
			compactness=comp, max_iter=iters, sigma=s, slic_zero=True)) for seg 
			in n_segments for comp in compactness for s in sigma]
>>>>>>> master

	#Write comments for below later

<<<<<<< HEAD
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
=======
	#Needs to test values so that I only have 2 diff colors
	'''quickshift algorithms segments images with quickshift clustering in Color (x,y) space
	Returns ndarray segmentation mask of the labels
	#Variables
	image -- ndarray: input image
	ratio -- float between zero and onebalances color-space proximity & image-space proximity.
		Higher vals give more weight to color-space
	kernel_size: Width of Guassian kernel using smoothing. Higher means fewer clusters
	max_dist -- float: Cut-off point for data distances. Higher means fewer clusters
	return_tree -- bool: Whether to return the full segmentation hierachy tree and distances
		Set as False
	sigma -- float: Width of Guassian smoothin as preprocessing. Zero means no smoothing
	conver2lab -- bool: leave alone
	random_seed -- Random seed used for breacking ties. May have a list of random seeds to use
>>>>>>> master

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
<<<<<<< HEAD
=======
	def runQuickShift(image, ratio, kernel_size, max_dist, sigma, random_seed):
>>>>>>> master

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
<<<<<<< HEAD
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

	def runMorphGeodesicActiveCountour(self):
		#We run the inverse_gaussian_gradient to get the image to use
		gimage = skimage.segmentation.inverse_gaussian_gradient(
			self.params.getImage().getImage(), self.params.getAlpha(), 
			self.params.getSigma())

		output = morphological_geodesic_active_contour(gimage,
			self.params.getIters(), self.params.getInitLvlSetMorph(), smoothing=
			self.params.getSmoothing(), threshold=self.params.getThresh(),
			balloon= self.params.getBalloon())

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
=======
	find_boundaries
	Returns a bool aray where bondaries between labeled regions are True
	#Variables
	label_img -- array of int or bool which labels different regions
	connectivity -- int between {1, ..., label_img.ndim}
		A pixel is a boundary pixel if any of it's neighbors has a different
		label. Connectivity controls which pixels are considered neighbors. If
		1 (defalt, pixels sharing an edge will be neighbors). If label = label_img.ndim
		pixels sharing a corner are neighbors.
	mode -- string
		thick: any pixel not completely surrounded by pixels of the same label is marked
			as a boundary. So, boundaries are 2 pixels thick
		inner: outline the pixels $just inside$ of objects. background pixels are untouched
		outer: outline pixels in the background boundaries. When two objects touch, their
			boundary is also marked
		subpixel: return a doubled image with pixels $between$ the original pixels and the new
			boundary

		Mode should be inner or outer
	background -- int: If inner/outer, need to define background. Do more research 
	'''
	#Returns: array of bools same shape as label_img

	#Working: need label_img -- Already segment image befor passing into find_boundaries
	#This is most likely useless
	#def find_boundaries():
		#NEED TO TUL MORE


	#Needs testing
	'''
	mark_boundaries
	Returns image with boundaries between labeled images are highlighted
	image -- ndarray (M, N,[,3]): the image
	label_img -- int array (M,N): array where regions are marked by 
		different int vals
	color -- RGB color sequence == boundary for output image 
		(e.g. black and white). May want to mess around with this so that
		we only have two colors
	outline_color -- Same as color, but for boundaries in image
	mode: string {thick, inner, outer, subpixel} of boundary line
	background_label -- int: label to consider background. Only good for
		modes 'inner' and 'outer'

	Need (image, label_img), no additional lists
	'''
	def runMarkBoundaries(image, label_img):
		#May want to mess with color
		return skimage.segmentation.mark_boundaries(image, label_img) 

	#clear_border is most likely useless
	#join_segmentation is most likely useless
	#relabel_sequential is most likely useless
	
	#Needs testing
	'''
	watershed: seperates diff objects in an image based on user-defined
		markers. Usefull for overlapping objects (e.g. two overlapping circles)
	variables
	image -- ndarray: the image array
	(optional)
	labels -- labels: ndarray of int, same shape as image
	connectivity -- ndarray: An array with same shape as image. non-zero
		elements are neighbors for connection.
	offset -- array of image shape: offset of the connectivity?? should 
		leave alone
	mask -- bool (or 0 vs 1) ndarray same shape of image: Only points at 
		mask == True are labeled
	compactness -- float: higher values give more regular (circle) shaped
		basins
	watershed_line -- bool: If true, 1-pixel wide line. Line label = 0 
	
	Should get different connectivity, offset, mask, compactness. 
	'''

	def runWaterShed(image, labels, connects, offsets, mask, compact):
		return([skimage.segmentation.watershed(image.getImage(),
			markers=labels, connectivity=con, offset=off, mask=m, 
			compactness=comp) for con in connects for off in offsets
			for m in mask for comp in compact])

	#Needs testing
	'''
	chan_vese: Active countouring by evolving level set. Good for segmenting
		items without clearly defined boundaries. Iterative. 
		Only good with grayscale
	#Returns segmentation (ndarray of image), phi (not important), energies 
		(list of floats for each step of algorithm. Can analyze to find
		convergence, or not)
	#variables
	image -- ndarray: grayscale image
	mu -- float: 'edge length' mu is between 0-1, but higher values(<1) for 
		shapes with bad contours. High mu makes round edge, zero mu to detect
		smaller objects
	lambda1 -- float: difference from average. weight param for output region
		If lower than lambda2, this region will have a larger range of values
		than the other
	lambda2 -- float: Similar to lambda1
	lambda1 & lambda2 are usually 1. If 'background' is extremely diff from
		eachother, should be different
	tol -- float, positive: tolerance between iterations (e.g. 0.0001)
	max_iter -- uint: max number of iterations allowed
	dt -- float: Multiplication factor applied at calculations for each step.
		Can accelerate algorithm, higher values speed up, but may not converge
	init_level_set -- string or ndarray: starting level used by algorithm
		strings are:
			'checkerboard': Fastest convergence, can fail implicit edges
			'disk': opposite distance from center of image minus half of
			 	minimum value between image width and image height. Slower, 
			 	but more likely to get
			'small_disk': starting level set. Opposite of the distance from
				the center of the image minus a quarter of the min value
				between image width and image height
			Or, make your own array
	extended_output -- bool: True = The return value will be a tuple of
		'segmentation', 'phi': Final level and 'energies': list of floats,s
		evolution of the energy for each step of the algorithm. Can check
		convergence
		False = Just segmentation
		This should usually be True to check for convergence

	Need lists of mu, lambda1, lambda2, tol, max_iter, dt
	'''
	def runChanVese(image, mus, lambda1, lambda2, tols, dts):
		outputs_checker = [skimage.segmentation.chan_vese(image, mu=m, 
			lambda1=l1, lambda2=l2, tol=t, dt=d, 
			init_level_set='checkerboard', extended_output=True) for 
			m in mus for l1 in lambda1 for l2 in lambda2 for t in tols
			for d in dts]

		outputs_disk = [skimage.segmentation.chan_vese(image, mu=m, 
			lambda1=l1, lambda2=l2, tol=t, dt=d, 
			init_level_set='disk', extended_output=True) for 
			m in mus for l1 in lambda1 for l2 in lambda2 for t in tols
			for d in dts]	

		outputs_small = [skimage.segmentation.chan_vese(image, mu=m, 
			lambda1=l1, lambda2=l2, tol=t, dt=d, 
			init_level_set='small_disk', extended_output=True) for 
			m in mus for l1 in lambda1 for l2 in lambda2 for t in tols
			for d in dts]

		outputs_all = outputs_checkers + outputs_disk + outputs_small
		segments = [segs for segs, phi, energs in outputs_all]
		energies = [energs for segs, phi, energs in outputs_all]

		return (segments, energies)


	#Needs testing
	'''
	morphological_geodesic_active_contour: segments objects with lots
		of noise on borders
	#Returns 2D or 3D array, the final segmentation
	#Variables
	gimage -- 2d or 3d array: preprocessed image to be segmented. 
		enhanced/highlighted borders to segment
	iterations -- uint: Number of iterations to run
	init_level_set -- string or array. 'checkerboard' or'circle'.
		Not sure what these mean
	smoothing -- uint: Number of times the smoothing operator is applied
		per iteration. 
	threshold -- float: Areas of the image with a value smaller than this
		threshold are borders
	balloon -- float: Negative will shrink contour, positive expands. Zero
		disables
	iter_callback -- function: calls function once per iteration. IGNORE or
		Use for Debugging
	
	#Should pass in lists for: init_level_set, 
	




	#morphological_chan_vese
	#inverse_guassian_gradient
	#checkerboard_level_set
	#flood
	#flood_fill
	'''



>>>>>>> master






		