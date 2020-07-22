This is a README file containing the functions I used for a fourier analysis approach to uncertainty quantification in object detection.  I will be periodically updating this section and only include programs that I know have been tested.

## Computer and software requirements 

 I will include links to the tutorials and websites that were helpful to me in installing the software that is used.
You need a computer that had decent running memory space. At least 8gb, and 12 gb is comfortable.  You also will need an NVIDIA GPU if you want to train and test your models.  You also need CUDA (my version was 10.2).  

The code is written in python 2.7, or some variation of python 2, since the DIGITs software that was used was with python 2.  You will also need an Nvidia gpu to do work at a reasonable speed if you want to use the DIGITs software.  However, you can train the models on your own without the GUI framework provided by DIGITs.
DIGITS 6.11: You can download this version in the Nvidia DIGITS github: https://github.com/NVIDIA/DIGITS.

While DIGITS with docker was recommended, I had to tweak some of the numbers in the detectnet model (mostly changing the maximum object count from 50 to 250, or changing the IOU threshold to something higher than 0.5 because bounding boxes for cars could significantly overlap), so I opted for a local digits directory instead.  Another reason for the local DIGITS is that you can save your models within their directory and access them for additional analysis. 

A local caffe directory.   See here: https://github.com/BVLC/caffe

I used a caffe version distinct from the one in my computer because of compatibility issues  with my own computer vs the needed specifications of DIGITs.  Follow carefully the instructions there as well. My .prototxt files were saved in this local caffe directory.

A local protobuf directory, if necessary (I needed one), that is compatible with all the above. This is needed to be able to run DIGITS.  If you plan to use DIGITS for training, then follow the instructions given carefully when installing not just digits, but also your local caffe depository and the protobuff.

The figures were also created using jupyter-notebook.  All relevant notebooks were put into a file, so we start within the notebooks directory space, and then switch to the necessary files.  So your workspace directory, which includes all the files to be used here, should look something like this: 

### workspace

	|_ paper  
	|	- bunch of pictures  
 	|	- quantifying-uncertainty files  
	|_ notebook  
	|	- big notebook for creating necessary data for this paper
	|_ programs
	|	|_ left-right-pass-curves
       	|	|`	- 161 files containing notch filter results (these took a very,very long time to compute
    	|   	|_ cvgs_20_plus-lowpass-.npy
	|      	|	- 161 files containing numpy array files
    	|   	|_ cvgs_20_plus-highpass-.npy
	|      	|	- 161 files containing numpy array files	
    	|   	|_ cvgs_20_bandpass_inverse-notch-10.npy
	|      	|	- 161 files containing numpy array files	
    	|   	|_ cvgs_20_bandpass_inverse-notch-05.npy
	|      	|	- 161 files containing numpy array files	
	| 	- evaluation_and_analysis_functions.py   
	|	- visualization_and_graphics.py
    	|_ DIGITS
    	|   	|_ digits
    	|       	|_jobs
    	|           		|_20181112-163907-b425 (this is where my detector is)
    	|_ caffe
    	|   	|_ python
    	|   	|_ models
    	|       	|_ bvlc_refference_detectnet (yes I had two f's in refference)
    	|           		- deploy.prototxt
    	|
    	|_ train
    	|   	|_labels (#2853 images)
    	|   	|_images
    	|
    	|_ val
    	|   	|_labels (#243 images)
    	|   	|_images
    	|
    	|_ test
        	|_labels (#603 images)
        	|_images


Before you begin, make sure you have paths to the following in the very beginning of the  evaluation_and_analysis_functions library as well as your data generation jupyter-notebook:

1. curdir = '/path/to/programs/'
2. caffePath = '/path/to/caffe/python'
3. mod = '/path/to/model.prototxt'
4. weights = '/path/to/weights.caffemodel'
5. image_path = '/path/to/test/images/'
6. label_path = '/path/to/test/labels/'

There are two main function libraries that are used to obtain the data and visualize it: 

1. evaluation_and_analysis_functions is the function library containing model application and data analysis functions. 
2. visualizing_functions is the (much smaller) function library containing image making functions. 

To see the function descriptions, please look at the functions-README.md file.
All relevant datasets and figures were generated with the "Data generation and figures.ipynb" file. 

### Intermediate Data 

The following is a description of primary and intermediate data that was generated for this project:

1. Train/Test/Val: Files containing images and the ground truth data in DIGITS format.  The images are 352x352 pixel size RGB images.  The labels are in .txt files, and each row gives the data of a single car.  The relevant columns are:
	- column 0: label "car", 
	- column 1: a float between 0 and 1 representing the portion of the car that is actually in the image (this is in cases where the car is at the edge of an image and is partially cutoff
	-columns 4-7: the left, top, right, and bottom boundaries of the car's ground truth bounding box.  To get the bounding box contents of the image, we take the array  img\[top:bottom, left:right\]
	
2. snapshot_iter_2574.caffemodel:  The weights for a trained detectnet model used for this project.  

3. deploy.protoxt: the file describing the model's layers using caffe

4. cvgs_20_plus-lowpass- : a file containing 161 numpy arrays of size 151x88x88.  if one of these files is called cvg, then cvg\[i,:,:\] gives the coverage map of the image when filtered through a lowpass gaussian filter with filter width sigma = 150/ (24i).  (i = 0 gives the unfiltered image).  See the "lowpass" case in the freqfilter function description.

5. cvgs_20_plus-highpass-: a file containing 161 numpy arrays of size 151x88x88.  if one of these files is called cvg, then cvg\[i,:,:\] gives the coverage map of the image when filtered through a lowpass gaussian filter with filter width sigma = i/150.  (i = 0 gives the unfiltered image).  See the "lowpass" case in the freqfilter function description for a complete description.

6. cvgs_20_plus-bandpass_inverse-notch-05: a file containing 161 numpy arrays of size 71x88x88.  if one of these files is called cvg, then cvg\[i, :, :\] gives the coverage map of the image when filtered through a notch gaussian filter with band width 2epsilon = 0.1 and band radius r = 0.01i.  See the "bandpass_inverse" case in the freqfilter function description for a complete description.

7. cvgs_20_plus-bandpass_inverse-notch-10:  a file containing 161 numpy arrays of size 71x88x88.  if one of these files is called cvg, then cvg\[i, :, :\] gives the coverage map of the image when filtered through a notch gaussian filter with band width 2epsilon = 0.2 and band radius r = 0.01i.  See the "bandpass_inverse" case in the freqfilter function description for a complete description.

8. left-right-pass-curves: a file containing 161 numpy arrays of size 2 x numberOfCars x 12.  These give the cutoff epsilons for left and right justified notch filters for each car.

The root function that we use for generating filters is "freqfilter,"  located in the evaluation_and_analysis_functions library. Other function arguments are based on this one.




