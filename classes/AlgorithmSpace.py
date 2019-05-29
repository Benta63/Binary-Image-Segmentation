
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
	def __init__(self):
		#image is a ImageData object
		self.stuf = 1
		
	

	#Algorithms

	#Needs testing
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
	def runRandomWalker(data, labels, beta):
		#Let's deterime what mode to use
		mode = ""
		if len(data.getImage()) < 512 :
			mode = "cg_mg"
		else:
			mode = "bf"
		
		#If data is 2D, then this is a grayscale, so multichannel is 
		#false
		channel = False
		if (len(data.getShape()) > 2):
			#This is at least a 3D array, so multichannel
			channel = True
		print(channel)
		new_labels = [skimage.segmentation.random_walker(data.getImage(), labels, b, mode, copy=False, multichannel=channel, return_full_prob=True) for b in beta]
		#for b in beta:
			#for t in tol:
			#tol is after mode and before multichannel
			
		#	new_labels.append(scimage.segmentation.random_walker(data.getImage(), labels, b, mode, copy=False, multichannel=channel, return_full_prob=True))

		return new_labels

	#Needs testing
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
		max_px_move, max_iteration, convergence, self):
		new_snakes = []
		#active_contour has a bc variable with these five options
		BC = ['periodic', 'fixed', 'free', 'fixed-free', 'free-fixed']
		#Shuffling all the variables and finding the snake for each
		new_snakes = [skimage.segmentation.active_contour(image.getImage(),
			snake, a, b, wL, wE, g, bc, maxPx, iters, conv) for a in
			alpha for b in beta for wL in w_Line for wE in w_edge for
			g in gamma for maxPx in max_px_move for iters in max_iterations
			for conv in convergence]

		return new_snakes

	#Needs testing
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
	def ruFelzenszwalb(image, scale, sigma, min_size, self):
		graphs = [skimage.segmentation.felzenszwalb(image.getImage(), sc, si, m_s)
			for sc in scale for si in sigma for m_s in min_size]
		return graphs


	#Needs testing
	'''slic algorithms segments k-means clustering in Color space (x, y, z)
	Returns a 2D or 3D array of labels
	#Variables
	image -- ndarray
	n_segments -- int = number of labels in segmented output image (approx)
		Should find a way to compute n_segments
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
	def runSlic(image, compactness, max_iter, sigma, self):

		channel = False
		if (image.getShape()[0] > 2):
			channel = True
		labels = [skimage.segmentation.slic(image.getImage(), compactness=comp,
		 max_iter=iters, sigma=s, slic_zero=False) for comp in compactness
		 for iters in max_iter for s in sigma]
		[labels.append(skimage.segmentation.slic(image.getImage(), compactness=comp,
		 max_iter=iters, sigma=s, slic_zero=True)) for comp in compactness for
		 iters in max_iter for s in sigma]

		return labels

	#Needs Testing
	'''quickshift algorithms segments images with quickshift clustering in Color (x,y) space
	Returns ndarray segmentation mask of the labels
	#Variables
	image -- ndarray: input image
	ratio -- balances color-space proximity & image-space proximity. Higher vals give more
		weight to color-space
	kernel_size: Width of Guassian kernel using smoothing. Higher means fewer clusters
	max_dist -- float: Cut-off point for data distances. Higher means fewer clusters
	return_tree -- bool: Whether to return the full segmentation hierachy tree and distances
		Set as False
	sigma -- float: Width of Guassian smoothin as preprocessing. Zero means no smoothing
	conver2lab -- bool: leave alone
	random_seed -- Random seed used for breacking ties. May have a list of random seeds to use


	Pass in arrays of ratio, kernal_size, max_dist, sigma, random_seed
	'''
	def runQuickShift(image, ratio, kernel_size, max_dist, sigma, random_seed, self):

		labels = [skimage.segmentation.quickshift(image.getImage(), r, ker, m_d,
			sigma=s, random_seed=r_d) for r in ratio for ker in kernel_size
			for m_d in max_dist for s in sigma for r_d in random_seed]
		return labels	


	#Needs testing
	'''
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
'''
	def find_boundaries(self):
		#NEED TO TUL MORE
		return 1

	#Needs testing
	def mark_moundaries(self): 

'''





		