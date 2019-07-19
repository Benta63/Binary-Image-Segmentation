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
* pandas 0.24.2
  * https://pandas.pydata.org/pandas-docs/version/0.23.3/install.html
* random 
* math
* copy

### Commands
Type *python main.py* To run the program regularly
For parallelization *python -m scoop main.py*
For additional commands in scoop, refer to https://scoop.readthedocs.io/en/0.7/usage.html#how-to-launch-scoop-programs

## Adding Additional Algorithms
In order to add additional algorithms, it is necessary to edit three different files. Additionally, it is necessary to come up with some information about said algorithm. These are namely:
* A 2-3 character string to represent the algorithm
* The implementation of the algorithm
* The parameters associated with the algorithm
* What channel does this algorithm operate on (e.g. multichannel, grayscale or both)
* If the algorithm returns only a boolean mask of the segmentation
* The range of values that for each parameter

The files are as follows:
### *AlgorithmParams.py*
This files determines all of the parameters for all of the algorithms. At the bottom of the *__init__* constructor, there is a capitalized comment calling to add additional parameters. This is where the parameters of of the new algorithm are written. Additionally, you should also specify an accessor for each parameter. Currently, there is not a use for any of the modifiers, but feel free to create one.

### *AlgorithmHelper.py*
This file provides data and specifications for each algorithm.

#### Constructor
In the *__init__* constructor, there are a few places to edit. The first is in *self.indexes* which is a dictionary. You should add the character code as a key and the indices associated with it as the value. The indices are specified in the *AlgorithmParams.py* file. 
The next part to edit has to do with the channel of the algorithm. If the algorithm runs on grayscale images, append the character code to the *self.GrayAlgos* list. If the algorithm runs on multichannel images, append the character code to the *self.RGBAlgos* list. 
Next, if the algorithm returns a boolean mask, add the character code to the *self.mask* list. Additionally, add the character code to the *self.usedAlgos* list.
It is also important to have the range of values for each parameter. So, for each parameter, with list comprehension, add the range of values for each parameter to *self.PosVals* in the same order that the parameters are listed in *AlgorithmParams.py*

#### *makeToolbox*
Make toolbox creates a Toolbox to use with the deap library. It is important to register the parameters to the toolbox. Right before the *func_seq* list, there is a capitalized comment to add more parameters to the toolbox. Follow the format listed to add your parameters. If you want to weight certain values more than others, use *RandHelp.weighted_choice* as opposed to *random.choice*. It is also important to add the the parameters to the *func_seq* list in the same order that they appear in *AlgorithmParams.py*.

### *AlgorithmSpace.py*
The *AlgorithmSpace.py* file contains the implementation for each of the algorithms. To add to this file, look for the ADD NEW ALGORITHMS HERE comment which should be right above the *runAlgo* function and add the implemntation of the algorithm. Any parameters needed can be accesses by *self.params.youAccessor* and the numpy array of the image can be found with *self.params.getImage().getImage()*. 
Finally, add the algorithm to the *switcher* dictionary in *runAlgo*. The key will be the character code of the algorithm while the value will be the implementation of the algorithm. 

#### Notes
* May want to weight QuickShift algorithm as it takes significantly more time to run. Would need to set a timeit function
* LabelOne changes already labeled images to binary.
  * Useful if the images are labeled with more than two colors
* May have to clone scikit-image from git, as regular installation installs 0.14 and does not include flood_fill
  
