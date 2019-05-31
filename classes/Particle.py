from __future__ import division
import random
import math


class Particle:
	#x0 is start
	def __init__(self, x0):
		self.position = [] #particle position
		self.velocity = [] #particle velosity
		self.best_pos = [] #the best calculated position
		self.best_err = -1 #the best calculated error
		self.err = -1 #error
		self.dim = len(x0)

		for i in range(0, self.dim):
			self.velosity.append(random.uniform(-1, 1))
			self.position.append(x0[i])


		def updateVelocity(self, best_pos):
			weight = 0.5 #Constant inertia weight for previous velocity
			const1 = 1 #A constant
			const2 = 2 #Another constant

			for i in range(0, self.dim):
				r1 = random.random()
				r2 = random.random()

				vel_cog = const1*r1*(self.best_pos[i]-self.position[i])
				vel_soc = const2*r2*(best_pos[i]-self.position[i])
				self.velocity[i]=w*self.velocity[i]+vel_cog+vel_soc

		def updatePosition(self, bounds):
			for i in range(0, dim):
			self.position[i] = self.position[i] + self.velocity[i]

			#Check if we're out of bounds
			if self.position[i] > bounds[i][1]:
				self.position[i] = bounds[i][1]:

			if self.position[i] < bounds[i][0]:
				self.position[i] = bounds[i][0]

		def evaluate(self,costFunc):

			self.err=costFunc(self.position)

			#Is this the best err?
			if self.err < self.err_best or self.err_best==-1:
				self.best_pos = self.position
				self.best_err = self.err
	

