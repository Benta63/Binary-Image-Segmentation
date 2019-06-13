#Lightly Coupled with ImageData class
#A class with accessors and modifiers for all the parameters used in
#the skimage algorithms
#from . import ImageData

import random
class AlgorithmParams(object):
	
	def __init__(self, img, algo="", beta=0.0, tol=0.0, scale=0.0,
		sigma=0.1, min_size=0, n_segments=2, compactness=0.001, iters=10,
		ratio=0.0, kernel=1.0, max_dist=1, random_seed=134, connectivity=1,
		mu=0.0, lambda1=1.0, lambda2=1.0, dt=0.0, init_level_set_chan=None,
		init_level_set_morph=None,smoothing=1, threshold='auto', alpha=0.0,
		balloon=0.0, seed_point=[], new_value=""):

		#img is an ImageData object
		self.Image = img
		self.algorithm = algo
		self.beta = beta
		self.tolerance = tol
		self.scale = scale
		self.sigma = sigma
		self.min_size = min_size
		self.n_segments = n_segments
		self.compactness = compactness
		self.iterations = iters
		self.ratio = ratio
		self.kernel_size = kernel
		self.max_dist = max_dist
		self.seed = random_seed
		self.connectivity = connectivity
		self.mu = mu
		self.lambda1 = lambda1
		self.lambda2 = lambda2
		self.dt = dt
		#May want to make seperate level sets for different functions
			#e.g. Morph_chan_vese vs morph_geo_active_contour
		self.init_level_set_chan = init_level_set_chan
		self.init_level_set_morph = init_level_set_morph
		self.smoothing = smoothing
		self.threshold=threshold
		self.smoothing = smoothing
		self.alpha = alpha
		self.balloon = balloon
		self.seed_point = seed_point
		self.new_value = new_value


	#Accessors
	def getImage(self): return self.Image
	def getAlgo(self): return self.algorithm
	def getBeta(self): return self.beta
	def getTolerance(self): return self.tolerance
	def getScale(self): return self.scale
	def getSigma(self): return self.sigma
	def getMinSize(self): return self.min_size
	def getSegments(self): return self.n_segments
	def getCompact(self): return self.compactness
	def getIters(self): return self.iterations
	def getRatio(self): return self.ratio
	def getKernel(self): return self.kernel_size
	def getMaxDist(self): return self.max_dist
	def getSeed(self): return self.seed
	def getConnect(self): return self.connectivity
	def getMu(self): return self.mu
	def getLambdaOne(self): return self.lambda1
	def getLambdaTwo(self): return self.lambda2
	def getDT(self): return self.dt
	def getInitLvlSetChan(self): return self.init_level_set_chan
	def getInitLvlSetMorph(self): return self.init_level_set_morph
	def getSmoothing(self): return self.smoothing
	def getThresh(self): return self.threshold
	def getAlpha(self): return self.alpha
	def getSmoothing(self): return self.smoothing
	def getBalloon(self): return self.balloon
	def getSeedPoint(self): return self.seed_point
	def getNewVal(self): return self.new_value

	#Modifiers
	def changeImage(self, newImg):
		self.Image = newImg
		#If the dimension is different, we should also reset the seed shape
		if (len(newImg.getShape()) != len(self.seed_point)):
			self.seed_point = [random.randrange(0, dim) for dim in newImg.getShape()]
	def changeAlgo(self, algo):
		self.algorithm = algo
	def changeBeta(self, beta):
		self.beta = beta
	def changeTolerance(self, tol):
		self.tolerance = tol
	def changeScale(self, scale):
		self.scale = scale
	def changeSigma(self, sigma):
		self.sigma = sigma
	def changeMinSize(self, minSize):
		self.min_size = minSize
	def changeSegments(self, segments):
		self.n_segments = segments
	def changeCompact(self, compactness):
		self.compactness = compactness
	def changeIter(self, iters):
		self.iterations = iters
	def changeRatio(self, ratio):
		self.ratio = ratio
	def changeKernel(self, kernel):
		self.kernel_size = kernel
	def changeMaxDist(self, maxDist):
		self.max_dist = maxDist
	def changeSeed(self, random_seed):
		self.seed = random_seed
	def changeConnect(self, connects):
		self.connectivity = connects
	def changeMu(self, mu):
		self.mu = mu
	#Lambda1 and lambda2
	def changeL1(self, lambda1):
		self.lambda1 = lambda1
	def changeL2(self, lambda2):
		self.lambda2 = lambda2
	def changeDT(self, dt):
		self.dt = dt
	def changeInitLvlSetChan(self, lvlSet):
		self.init_level_set_chan = lvlSet
	def changeInitLvlSetMorph(self, lvlSet):
		self.init_level_set_morph = lvlSet
	def changeSmoothing(self, smoothing):
		self.smoothing = smoothing
	def changeThresh(self, threshold):
		self.threshold = threshold
	def changeAlpha(self, alpha):
		self.alpha = alpha
	def changeSmoothing(self, smoothing):
		self.smoothing = smoothing
	def changeBalloon(self, balloon):
		self.balloon = balloon
	def changeSeedPoint(self, seedPt):
		self.seed_point = seedPt
	def changeNewVal(self, newVal):
		self.new_value = newVal
