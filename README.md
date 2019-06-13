# Binary Image Segmentation

A project by Noah Stolz under Dr. Dirk Colbry of Michigan State University's Computational Mathematics, Science and Engineering (CMSE) as part of the Summer Research Opportunities Program (SROP). 

## Project Overview
Image segmentation is usually a time-consuming process. Usually, a researcher needs to manually segment all of their images. This project aims to use genetic algorithms for image segmentation and aims to save researchers time.

## Dependencies
* python 3.6.3 
  * conda 4.6.14
* numpy 1.13.3
* scikit-image 0.15.0
* skimage.segmentation
* deap 1.2.2
* random 
* math
* copy

## Todo
* Fix AC algorithm
* Refer to Deap's Basic.fitness
  * https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness
* Create mutation function

## Notes
* May want to weight QuickShift algorithm as it takes significantly more time to run. Would need to set a timeit function
* LabelOne changes already labeled images to binary.
  * Useful if the images are labeled with more than two colors
  
