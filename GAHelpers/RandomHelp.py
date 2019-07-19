#Helper functions to the random library
import random

class RandomHelp(object):

	#Returns a number from a list with certain values being weighted.
	#Variables:
	#seq is the sequence to be weighted
	#minVal is the minimum value to be weighted higher
	#maxVal is the maximum value to be weighted higher
	#weight is what the values from minVal to maxVal should be weighted
	def weighted_choice(seq, minVal, maxVal, weight):
		
		weights = []
		#Here we compute the number of values between minVal and maxVal
		counter = 0
		for i in seq:
			if minVal <= i <= maxVal:
				counter += 1

		for i in range(0, len(seq)):
			'''Populates the weights list. 
			Example: If weight is 0.5 and there are 5 values between
			minVal and maxVal, there is a 0.1 chance of each of those
			values
			'''
			if (i < counter):
				weights.append(weight/counter)
			else:
				weights.append((1-weight)/(len(seq) - counter))
		totals = []
		cur_total = 0
		
		for w in weights:
			cur_total += w
			totals.append(cur_total)
		#The randomization
		rand = random.random() * cur_total
		weightIndex = 0
		#And to select the correct index
		for i, total in enumerate(totals):
			if rand < total:
				weightIndex = i
				break
		#So we have the index
		return seq[weightIndex]