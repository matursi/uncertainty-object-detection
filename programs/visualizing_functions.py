import argparse
import os
import time

#os.chdir('~/caffe/python/')
from filter_evaluation_functions import *
from google.protobuf import text_format
import numpy as np
import PIL.Image
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter
import matplotlib as mpl


#REPLACE The below variables with paths to each necessary file.  You can use absolute file paths if necessary.  

############################

#caffePath is the path to the python file in your local caffe directory
caffePath = 'path/to/caffe/python'

#curdir is the file with all the python programs saved
curdir = '/path/to/programs/'

#mod = filepath where the deploy.prototxt file is located
#weights = filepath where model is located

mod = '/path/to/caffe/models/bvlc_refference_detectnet/deploy.prototxt'

#Weights that you want for code: MAKE SURE THAT WEIGHTS HERE MATCH THE ONES YOU ARE USING IN YOUR NOTEBOOK!!!!!

weights = '/path/to/digits/jobs/jobtitle/snapshot.caffemodel'

# filepaths for images and labels are located

image_path = '/path/to/test/images/'
label_path = '/path/to/test/labels/'

#############################


os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output
os.chdir(caffePath)
import caffe
from caffe.proto import caffe_pb2

def Q(arr):
	"""
	logarithmic function used for visualizing arrays with values ranging at exponential-like levels.
	Here, if A = an array, m = max(A), Q(A) = 255/ln(1+m) * ln(1+|A|).  
	ARGUMENT: 
	arr: array (typically used for 2-D arrays)
	OUTPUT:
	Y: also array, rounded to int values and multiplied by 255 to make it an rgb image.
	"""
	array = np.abs(arr)
	R = np.max(np.abs(array))

	c = 1.0#/np.log(1+R)
	return c*np.log(1+array)
	return np.round(c*np.log(1+array)).astype(int)


def drawbox(img,coords,points=0,sigmas=0):
	"""
	Draws a bounding box on an image. Points and sigmas args no longer used, but can be reused by commenting out lines 1-5.  Will also generate the certainty curve of the corresponding car and a visually optimal piecewise function.

	INPUTS:
	img: image array
	coords: list or array of ints, of length 4. [left, up, right, bottom] on image.  make sure that coords are within image dimensions

	PROCESS:
	shows the image with a yellow bounding box. gives numerical dimensions under image to more easily locate box if hard to find
	"""
	#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5)) #1
	f, ax1 = plt.subplots(figsize=(10,10))
	#ax2.figsize = (20,20)								#2
	draw = ax1.imshow(img)
	rect = patches.Rectangle((coords[0],coords[1]),coords[2]-coords[0],coords[3]-coords[1],linewidth=1,edgecolor='y',facecolor='none' )
	ax1.add_patch(rect)
	#[a,b]=optimal_shift(points) 	#3
	#ax2.scatter(sigmas,points)		#4
	#ax2.plot([0,a,b,3],[1,1,0,0])	#5
	#plt.show()
	return draw

        
def wholesome_uncertainty_stats(img,n, data, z, sigmas, prediction):
    """For a car,  visualizes change in uncertainties.  Includes plotline, color plot of individual uncertainties also will give ground truth vs predicted truth behavior. We should do this by image
	INPUTS:
	img: rgb image array
	n: the nth car in img
	data: the ground truth boxes in img
	z: the uncertainty arrays of the cars in img.  z[i][k] is an array of coverage pixels intersecting car i, where uncertainty levels are occuring over the image with blurring level sigmas[k]
	sigmas: the blurring levels for img
	prediction: the predicted boxes for img

	PROCESS:
	plots len(sigmas) images depicting the uncertainty arrays.  For each array, the title tells you the blurring level, the maximal iou for car n, and the iou rank (the number of cars who iou with maxiou ind >= car n).
	"""
    f= plt.figure()
    drawbox(img, data[n])
    plt.imshow(img[data[n][1]:data[n][3],data[n][0]:data[n][2]])

    for s in range(len(z[n])):
        y_predict = [list(prediction[s][m][0][:4].astype(int)) for m in range(len(prediction[s]))]
        maxiou, maxiou_ind, maxiou_pred, maxiou_pred_ind, rank = maximal_iou_match(data*4,y_predict,n)
        f = plt.figure()
        plt.imshow(z[n][s])
        plt.colorbar()
        plt.clim((0,1))
        plt.title('$\sigma = $'+str(sigmas[s])+', max-iou = '+str(maxiou)+'\n'
                 +'rank for maximal prediction: '+str(rank))


def prediction_uncertainty_stats_image(img, n, k, coords, z, sigmas, prediction):
    """function used in the image notebook for the 2019 fall report.  displays local image of car n in image k.  This function is a workable one, but only if prediction is defined beforehand
    """
    f= plt.figure()
    drawbox(img, data[n])
    plt.imshow(img[coords[1]:coords[3],coords[0]:coords[2]])
    plt.savefig('../internship-report-2019/image-'+good_filenames[k]+'-car-'+str(n)+'box.png')
    for s in range(len(z[n])):
        y_predict = [list(prediction[s][m][0][:4].astype(int)) for m in range(len(prediction[s]))]
        #print y_predict
        maxiou, maxiou_ind, maxiou_pred, maxiou_pred_ind, rank = maximal_iou_match(data*4,y_predict,n)
        f = plt.figure()
        plt.imshow(z[n][s])
        plt.colorbar()
        plt.clim((0,1))
        plt.title('$\sigma = $'+str(sigmas[s])[0:5]+', max-iou = '+str(maxiou)[0:5]+'\n'
                 +'rank for maximal prediction: '+str(rank))
        plt.savefig('../internship-report-2019/image-'+good_filenames[k]+'-car-'+str(n)+'-sigma-'+str(s)+'.png')


def show_cutoffs(z,z2, title): 
	""" Shows low and high pass curves and their cutoffs in a nice little graphic. 
    INPUT:
    z: lowpass certainty curve
    z2: highpass certainty curve
    title: main title of image

    PROCESS: plots two certainty curves with their cutoff points.

    """

	size = 15

	#Find lowpass cutoff
	sigmas = np.linspace(0,24,151)
	[cutoff, sigmax, sigmin],zargmax, zargmin = find_midpoint_of_curve(z, sigmas)
	[sigmax, sig1, sig2],_,_ = find_midpoint_of_curve(z, sigmas,threshold = 0.95)
	[sigmin, sig1, sig2],_,_ = find_midpoint_of_curve(z, sigmas, threshold = 0.05)

	g, axx = plt.subplots(1,2, figsize = (10,5))
	axx[0].scatter(sigmas, z)
	axx[0].scatter([cutoff,sigmax,sigmin],[0.5*(z[zargmax]+z[zargmin]),
		                                  0.95*z[zargmax]+0.05*z[zargmin],
		                                  0.05*z[zargmax]+0.05*z[zargmin]])

	axx[0].set_title('Low-pass certainty curve',size = size)
	axx[0].legend(['curve','cutoff and \n threshold points'])

	axx[0].set_xlabel('$\\alpha$', size = size+2)
	axx[1].set_xlabel('$\sigma$', size = size+2)
	axx[0].set_ylabel('certainty',size = size)

	axx[0].set_yticklabels([0,0,0.2,0.4,0.6,0.8,1],fontsize=size)
	axx[1].set_yticklabels([0,0,0.2,0.4,0.6,0.8,1],fontsize=size)
	axx[0].set_xticklabels([0,0,5,10,15,20,25],fontsize=size)
	axx[1].set_xticklabels([0,0,0.2,0.4,0.6,0.8,1],fontsize=size)

	#Find highpass cutoff
	sigmas = np.linspace(0,1,151)
	[cutoff, sigmax, sigmin],zargmax,zargmin = find_midpoint_of_curve(z2, sigmas)
	[sigmax, sig1, sig2], _ ,_ = find_midpoint_of_curve(z2, sigmas,threshold = 0.95)
	[sigmin, sig1, sig2],_,_ = find_midpoint_of_curve(z2, sigmas, threshold = 0.05)

	axx[1].scatter(sigmas, z2,color = 'purple')        
	axx[1].scatter([cutoff,sigmax,sigmin],[0.5*(z2[zargmax]+z2[zargmin]),
		                                  0.95*z2[zargmax]+0.05*z2[zargmin],
		                                  0.05*z2[zargmax]+0.05*z2[zargmin]],   color ='orange')
	plt.suptitle(title,size = 15)
	axx[1].set_title('High-pass certainty curve',size = size)
	axx[1].legend(['curve','transition and \n threshold points'])

os.chdir(curdir)

