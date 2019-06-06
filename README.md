# Binary Image Segmentation

A project by Noah Stolz under Dr. Dirk Colbry of Michigan State University's Computational Mathematics, Science and Engineering (CMSE) as part of the Summer Research Opportunities Program (SROP). 

## Project Overview
Image segmentation is usually a time-consuming process. Usually, a researcher needs to manually segment all of their images. This project aims to use genetic algorithms for image segmentation and aims to save researchers time.

## Dependencies
* numpy
* skimage
* skimage.segmentation
* deap
* random
* math
* copy

## Todo
* Go through labeled leaf images to change to one color for
  * This project is on binary image segmentation, thus the validation images should only be two colors.
* Refer to Deap's Basic.fitness
  * https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness
  * Need to correct creator.create("Fitness***", base.fitness, ????)
  
