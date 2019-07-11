# Binary Image Segmentation with Genetic Algorithms

A project by Noah Stolz under Dr. Dirk Colbry of Michigan State University's Computational Mathematics, Science and Engineering (CMSE) as part of the Summer Research Opportunities Program (SROP). 

## Project Overview
Image segmentation is usually a time-consuming process. Usually, a researcher needs to manually segment all of their images. This project aims to use genetic algorithms for image segmentation and aims to save researchers time.

## Running the program
First install all of the necessary packages.

### Dependencies
* python 3.5.3 
  * conda 4.6.14
   * https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
* numpy 1.13.3
  * https://anaconda.org/anaconda/numpy
* scikit-image 0.15.0
 * https://scikit-image.org/docs/dev/install.html 
 * skimage.segmentation
* deap 1.2.2
  * https://anaconda.org/conda-forge/deap
* scoop 0.7.1.1
  * https://scoop.readthedocs.io/en/0.7/install.html
* pillow 6.0.0
  * https://anaconda.org/anaconda/pillow
* random 
* math
* copy

### Commands
Type *python main.py* To run the program regularly
For parallelization *python -m scoop main.py*
For additional commands in scoop, refer to https://scoop.readthedocs.io/en/0.7/usage.html#how-to-launch-scoop-programs

#### Notes
* May want to weight QuickShift algorithm as it takes significantly more time to run. Would need to set a timeit function
* LabelOne changes already labeled images to binary.
  * Useful if the images are labeled with more than two colors
  
