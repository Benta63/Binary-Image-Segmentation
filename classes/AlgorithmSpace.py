#https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.active_contour

import numpy as np
import pandas as pd
import copy
import skimage
from skimage import segmentation
from itertools import combinations

'''
This class will run through all the algorithms in skimage.segmentation
and change the parameters
'''
class AlgorithmSpace(object):
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
			mode = "cg_mg"
		else:
			mode = "bf"
		
		#If data is 2D, then this is a grayscale, so multichannel is 
		#false
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
	Produces an oversegmentation of the multichannel using min-span tree.
	Returns an integer mask indicating the segment labels
	#Variables
	image -- ndarray Input image
	scale -- float Higher means larger clusters
	sigma -- st. dev of width of a Guassian kernel by preprocessing
	min_size -- min component size using postprocessing
	#miltichannel -- optional (true) -- Whether daya has multiple channels.
		Don't need to mess with this. If False, image is grayscale
	pass in lists: scale, sigma, min_size

	'''
	def runFelzenszwalb(image, scale, sigma, min_size):
		graphs = [skimage.segmentation.felzenszwalb(image.getImage(),
			scale=sc, sigma=si, min_size=m_s) for sc in scale for si
			in sigma for m_s in min_size]
		return graphs


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
	convert2lab -- bool: Whether the image-space should be converted to Lab colorspace
		before segmentation. Input image must be RGB to be true. If multichannel is true,
		this is also true.
	enforce_connectivity -- bool: whether the generated segments are connected or not
		don't need to mess with this
	min_size_factor -- float: proportion of minimum segmentation size to be removed with
		respect to the supposed segment size. 'depth*width*height/n_segments'
	max_size factor -- proportion of max size connected segment size
	slic_zero -- bool: run SLIC-zero, the zero parameter mode of SLIC

	Should get an list of compactness, max_iter, sigma 
	'''
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

		return labels

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


	Pass in arrays of ratio, kernal_size, max_dist, sigma, random_seed
	'''
	def runQuickShift(image, ratio, kernel_size, max_dist, sigma, random_seed):

		labels = [skimage.segmentation.quickshift(image.getImage(), r, ker, m_d,
			sigma=s, random_seed=r_d) for r in ratio for ker in kernel_size
			for m_d in max_dist for s in sigma for r_d in random_seed]
		return labels	


	#Needs testing
	'''
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









		