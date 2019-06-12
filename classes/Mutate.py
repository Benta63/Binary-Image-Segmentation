import random

class Mutate(object):

	def __init__(self, posVals, initVals, flipProb):
		
		assert(flipProb <= 1)
		self.possibleVals = posVals
		self.vals = initVals
		self.flipProb = flipProb

	def pickVal(self, index):
		self.vals[index] = random.choice(self.possibleVals[index])


	def run(self):
		for index in range(0, len(self.vals)):
			randVal = random.random()
			if ranVal < self.flipProb:
				pickVal(index)
