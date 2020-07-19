"""
This program contains all of the functions used to derive the results for this paper.  
Below is a list of the functions, and what they do.  For more information in python, type ?mef.name_of_function for more information.  They were later executed in a jupyter-notebook to get all the necessary graphics.  I will also include the notebook file.

I worked with DIGITS 6.11 and a local caffe directory, since the configuration needed to be just right. However, you can take some of the functions here on their own without the additional caffe manipulation.
"""
import argparse
import os
import time

#os.chdir('~/caffe/python/')

from google.protobuf import text_format
import numpy as np
import PIL.Image
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter
import matplotlib as mpl
#import get_data_functions as gdf
#from model_classes import *

os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output

#change to local caffe/python file.  I used a local caffe file that was compatible with DIGITS 6.11. 
os.chdir('path/to/caffe/python')

import caffe
from caffe.proto import caffe_pb2

#change these to /path/from/home/to/file/
curdir = '/path/to/programs/'
caffePath = 'path/to/caffe/python'

#mod = filepath where the deploy.prototxt file is located
#weights = filepath where model is located. MAKE SURE THAT WEIGHTS HERE MATCH THE ONES YOU ARE USING IN YOUR NOTEBOOK!!!!!


mod = '/path/to/caffe/models/bvlc_refference_detectnet/deploy.prototxt'
weights = '/path/to/digits/jobs/jobtitle/snapshot.caffemodel


#change to /path/to/image/file
image_path = 'path/to/test/images/'
label_path = 'path/to/test/labels/'
good_filenames = []

#The following were taken from the DIGITS code and adapted to my purposes.  I trained the models with the GUI, but then would do detections from the notebooks because I needed some outputs in prior layers

def get_net(caffemodel, deploy_file, use_gpu=True): #get caffe model.  Use GPU if computer has it
    """
    Returns an instance of caffe.Net
    INPUTS:
    caffemodel -- 'path/to/a.caffemodel' 
    deploy_file -- 'path/to/a.prototxt' 
    use_gpu -- if True, use the GPU for inference

    OUTPUTS: get a caffe.net

    """

    if use_gpu:
        caffe.set_mode_gpu()

    # load a new model
    return caffe.Net(deploy_file, caffemodel, caffe.TEST)

#I didn't use mean_file, but you can if you like!
def get_transformer(deploy_file, mean_file=None):
    """
    Returns an instance of caffe.io.Transformer
    INPUTS:
    deploy_file -- 'path/to/a.prototxt'
    Keyword INPUTS:
    mean_file -- 'path/to/a.binaryproto' file..
    """
    network = caffe_pb2.NetParameter()
    with open(deploy_file) as infile:
        text_format.Merge(infile.read(), network)

    if network.input_shape:
        dims = network.input_shape[0].dim
    else:
        dims = network.input_dim[:4]

    t = caffe.io.Transformer(
            inputs = {'data': dims}
            )
    t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

    # color images
    if dims[1] == 3:
        # channel swap
        t.set_channel_swap('data', (2,1,0))

    if mean_file:
        # set mean pixel
        with open(mean_file,'rb') as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'):
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            t.set_mean('data', pixel)

    return t

#This function loads an image with the appropriate filtration.  I used it to load a 352x352 image, apply filtration, and then make it into a 1408x1408 image for detectnet.

def load_image(path, height, width, mode='RGB', args = [None,0.0, 0.0, 0.0], transform = None):
	"""
	Load an image from disk or transforms an already loaded image by filtering it
	Returns an np.ndarray (channels x width x height)
	INPUTS:
	path -- path to an image on disk
	width -- resize dimension
	height -- resize dimension
	Keyword INPUTS:
	mode -- the PIL mode that the image should be converted to	
    args -- arguments for filtration function: is of the form 
                ['filtertype', 
                  sigma, 
                  radius (for bandpass/notch), 
                  bandwidth (for bandpass/notch),
                  
                  height (for any filter,optional),
                  'justify' (for bandpass/notch, optional), ]

	"""
    #Isolates filtertype, and applies filtering if a filter is given.  Otherwise, it just does the interpolation and gives you the dilated image.

	filtertype = args[0]    

	if filtertype!= None:

		if transform !=None:

			image = filter_application(transform, args = args)
		else: 
			image = PIL.Image.open(path)
			image = image.convert(mode)
			image = np.array(image)
			image = filter_image(image, args)

	else:

		image = PIL.Image.open(path)
		image = image.convert(mode)
		image = np.array(image)		
	image = scipy.misc.imresize(image, (height, width), 'bilinear')
	return image

# This is a copy of a caffe function, but I extracted the coverage layer as well as the predictions.  coverage maps were size 88 x 88 pixels.  Images should be those from "load image", not directly from the file, because load_image will properly dilate them first.
def forward_pass(images, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    INPUTS:
    images -- a list of np.ndarrays... I tended to just put one image at a time.
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword INPUTS:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)

    OUTPUTS: in a list form:
    scores: a  250x5 np.array of predictions
    cvg_scores: an n/16 x n/16 np.array.  
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    cvg_scores = None
    bboxes = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()

        #output is a list of predictions
        output = net.forward()[net.outputs[-1]]

        #b is our coverage map.
        b= net.blobs['coverage'].data[0,0]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
            cvg_scores = np.copy(b)
        else:
            scores = np.vstack((scores, output))
            cvg_scores = np.vstack((cvg_scores, b))
        #print 'Processed image in %f seconds ...' % ( (end - start)), uncomment this to measure how long a detection takes.
    
    return [scores, cvg_scores]

#I didn't really use this function, but this may still be necessary if you want a variable label.
def read_labels(labels_file):
    """
    Returns a list of strings
    INPUTS:
    labels_file -- path to a .txt file
    """
    if not labels_file:
        #print 'WARNING: No labels file provided. Results will be difficult to interpret.'
        return None

    labels = []
    with open(labels_file) as infile:
        for line in infile:
            label = line.strip()
            if label:
                labels.append(label)
    assert len(labels), 'No labels found'
    return labels

# This function detect the image.  There are a lot of arguments to break down
def detect(caffemodel, deploy_file, image_files,
        mean_file=None, labels_file=None, batch_size=None, use_gpu=False, transform = None, args =['lowpass',0.0, 0.0,0.0]):
	"""
	Classify some images against a Caffe model and print the results
	INPUTS:
	caffemodel -- path to a .caffemodel
	deploy_file -- path to a .prototxt
	image_files -- list of paths to images... I would just put on image inside list 
                    brackets
	Keyword INPUTS:
	mean_file -- path to a .binaryproto
	labels_file path to a .txt file
	use_gpu -- if True, run inference on the GPU
    transform:  This gives the option to put a (703 x 703) size transform for our 352 x 352 size image. 
            It's helpful when using this function in loop to avoid having to take the same transform.
    args: freqfilter arguments.  See freqfilter for more info.
	"""
	# Load the model 
	net = get_net(caffemodel, deploy_file, use_gpu)
	transformer = get_transformer(deploy_file, mean_file)
	_, channels, height, width = transformer.inputs['data']
	if channels == 3:
		mode = 'RGB'
	elif channels == 1:
		mode = 'L'
	else:
		raise ValueError('Invalid number for channels: %s' % channels)

    #gets image and makes detection
	images = [load_image(image_file, height, width, mode, args, transform = transform) for image_file in image_files]

	[scores, b] = forward_pass(images, net, transformer, batch_size=batch_size)

    #returns predictions and coverage map.
	return scores, b




stride = 16

def probabilities(scores, cvg_scores, lab_path = None,avg = 1.0, carnums = None): 
	"""#returns a list of cars with bounding boxes and a "probability" of objectness
	#if test is the filename, compare to the ground truth, if not, base it off of the results alone.
	#curently three ways of measuring objectness scores: averages for pixels above
	# a certain threshold, taking maximal pixel, and aking the center pixel.
	INPUTS:
		scores: an nx6 array. n being the number of ground_truth boxes.  6 columns representing
				[left, top, right, bottom, coverage, internal score
		cvg_scores: array of pixels of form m x n, where dimensions are the image size over stride
		lab_path: path/to/file/labels.txt.  Use if you want ground truth pixel info.  Type None to just get prediction info
		avg: a code for determining what kind of way to measure objectness scores. 
				- type a float x from 0 to 1 to get all pixels at least x * max coverage pixel, and then 		      		averaging those pixels.
				- type 'center' to get the uncertainty of the center pixel in the ground truth box. If even 			dimensions, then bottom right from center is used.
				- type 'block' to just get array of all coverage pixels.  Note that this will change the format 			of the output
		carnums: cars whose certainty values we want by index
	OUTPUTS:
		cars: if avg = float or 'center', is an nx6 array, where n is the number of predicted cars in the 			image. Each row is [left, top, right, bottom, cvg_score, uncertainty_score].  If avg = 'block', returns a n-length list of elements formated in the form [[left,top,right,bottom, cvg_score], 		 			uncertainty_array], where the latter is an array of uncertainty levels of all the coverage_pixels 			intersecting the predicted box. (This is defunct right now for all options but 'center')

		real_cars: same as above, except gives info for the ground truth boxes.

	"""
	cars = []
	real_cars = []
	count = 0

	#if no labpath is included... this will be for false positives as well.
	#This is to get certainty data for predicted boxes
	while (scores[count, 4]!=0):
		block=np.array([ int(max(0, n/stride)) for n in scores[count,:4]] )
		pixel_probs = cvg_scores[block[1]:block[3]+1,block[0]:block[2]+1]

		if type(avg)!=str:
			pixel_probs = pixel_probs.flatten()
			prob = max(pixel_probs)
			avg_probs = np.average(pixel_probs[pixel_probs>= avg*prob])
			new_row = np.append(scores[count,:], avg_probs)
			if len(cars) > 0:
				cars = np.vstack([cars,new_row])
			else:
				cars = np.array([new_row])

		else: 
			if avg=='block':

				new_row = [scores[count,:]] +[pixel_probs]
				if len(cars) > 0:
					cars = cars +[new_row]
				else:
					cars = [new_row]

			if avg == 'center':
				c = scores[count,:4]
				blen,bhi = ((c[2]+c[0])/(2*stride)).astype(int),((c[3]+c[1])/(2*stride)).astype(int)
				new_row = np.append(scores[count,:],cvg_scores[min(bhi,87),min(blen,87)])

				if len(cars) > 0:
					cars = np.vstack([cars,new_row])
				else:
				 	cars = np.array([new_row])

		count+=1
		if count == len(scores): break

	if type(lab_path)!=str:
		return cars

	else: #If truth labels are included, we provide ground truth data as well
		f = open(lab_path, 'r')
		data = np.array([[int(n) for n in line.split(' ')[4:8]] for line in f.read().split('\n')[:-1]])
		if carnums == None:
			carnums = range(len(data))

		f.close()  
		#this is to get certainty data fo ground truth boxes
		for count in carnums:
			dataline = data[count]/4 #To get overlapping coverages
			real_pixels = cvg_scores[dataline[1]:dataline[3]+1,dataline[0]:dataline[2]+1]

			if type(avg)!=str:
				real_pixels = real_pixels.flatten()
				real_prob_max = max(real_pixels)
				real_avg_probs = np.average(real_pixels[real_pixels>= avg*real_prob_max])
				new_real_row = np.append(data[count], real_avg_probs)
				if len(real_cars) > 0:
					real_cars = np.vstack([real_cars,new_real_row])
				else:
					real_cars = new_real_row
			else: 
				if avg == 'block':
					new_real_row = [data[count], real_pixels]

					if len(real_cars) > 0:
						real_cars = real_cars +[new_real_row]
					else:
						real_cars = [new_real_row]

				if avg == 'center':
					c = data[count]*4   #To get ground truth coordinates for dilated images                                        
					blen,bhi = ((c[2]+c[0])/(2*stride)).astype(int),((c[3]+c[1])/(2*stride)).astype(int)

					new_real_row = np.append(data[count,:],cvg_scores[min(bhi,87),min(blen,87)])

					if len(real_cars) > 0:
						real_cars = np.vstack([real_cars,new_real_row])
					else:
						real_cars = new_real_row                


		return cars, real_cars
	
#This is for RGB images of size 352x352.  It pads the image and then gives the fourier transform of each color channel	
def padded_image_transform(img, padding = 'reflect'):
    """ 
    returns the discrete fourier transform of each image for each color channel as a list, images are assumed to be (352,352,3), and each fourier transform is of size (703,703).  Change dimensions and padding values if you want to use this function for arbitrary images.

    INPUTS:
    img: a (352,352,3) array of integers
    padding: padding for your fourier transform.  I used reflect bevause it enabled continuity in the image without having sharp edges in the periodic function

    OUTPUT: a three element list of transforms for each color channel.  transforms are size (703,703) arrays.
    
    """
	c = (352,352)

	new_img = np.pad(img, ((176,175),(176,175),(0,0)), mode=padding)
	new_img_avg = np.average(new_img, axis = (0,1))
	new_img = new_img -new_img_avg

	fftrawr = np.fft.fft2(new_img[:,:,0]) 
	#print fftrawr[:5,:5]
	fftrawg = np.fft.fft2(new_img[:,:,1])
	fftrawb = np.fft.fft2(new_img[:,:,2])
	#print transform.shape
	return [fftrawr, fftrawg,fftrawb,new_img_avg]


def freqfilter(x,y, args = ['lowpass',0.0,0.0,0.0]):
"""
Returns a filter in the form of an array.  Multiply the frequency space by the filter and then go back to time space to get your filtered image. Assumes filter is of size (703,703).  Change the specifications to allow for filters of arbitrary size. The filters divide values by 702. If you want to include more options, you can do so here.  This is the set up so that you do not have to worry about adjusting code elsewhere when wanting to try new filters on these images. 

WARNING: the sigma values are with respect to fitler intensity (this is args[1]).  For highpass and notch filters filters, Sigma is the filter width.  For lowpass and bandpass filters, sigma variable is actually 1/ filter width.  Look through the filter specifications to tweak as necessary when using this program.  

Current options for args[0]:
lowpass, highpass, highpass2, highbandpass, highbandpass2, bandpass_inverse (this is the notch filter heading), bandpass

INPUTS: 
x, y: integer arrays.  I used x, y = np.arange(-351,352).
args: filter arguments in the form of a list [str, float, float, float, float, str].  
    the argument order is [filtertype, 
                            filter intensity, (either sigma or 1/sigma) see above
                            radius, (used for notch or bandpass filters)
                            bandwidth, (used for notch or bandpass filters)
                            filter height, (number between 0 and 1, default is 1)
                            justify (for bandpass and notch filters) ]

OUTPUT:
freq: a (703,703) size array.
"""
	[filtertype, sigma,r, eps] = args[:4]

	height = 1.0
	if len(args) >= 5:
		height = args[4]

	if len(args) >= 6:
		if args[5] == 'left':
		    r, eps = r+eps/2.0, eps/2.0
		if args[5] == 'right':
		    r, eps = r-eps/2.0, eps/2.0

	if sigma == 0:
		return 1.0

	X,Y = np.meshgrid(x,y)

	if filtertype == 'lowpass': #lowpass Gaussian filter
		Zfreq = np.fft.fftshift(np.exp((sigma/702.0)**2*(-X**2 -Y**2)*0.5))

	if filtertype == 'highpass': #highpass Gaussian filter

		Zfreq = 1-np.fft.fftshift(np.exp( -0.5* (X**2+Y**2)/(702.0* sigma)**(2) ) )


	if filtertype == 'highpass2': #highpass filter that is simply subtraction
		Zfreq = 1+ np.average(Zfreq)- Zfreq

	if filtertype == 'highbandpass': #a bandpass filter with ringsize sqrt(1/2)
		A = (X**2+Y**2)**0.5/702.0
        Zfreq = np.fft.fftshift(np.exp( -((0.5**0.5 - A)**2)*sigma**2*0.5 ))


	if filtertype == 'highbandpass2': #a bandpass filter with ringsize 0.5
		A = (X**2+Y**2)**0.5/702.0
		B=(A-0.5)*(A-0.5 < 0)+0.5
		Zfreq = np.fft.fftshift(np.exp( -((0.5 - B)**2)*sigma**2*0.5 ))

	if filtertype == 'bandpass': #gaussian bandpass filter
		A = (X**2+Y**2)**0.5/702.0
		B=(A-r+eps)*(A-r+eps< 0)+r-eps
		C=r+eps - (A-r-eps)*(A-r-eps> 0)
		Zfreq = np.fft.fftshift(np.minimum(np.exp( -((r-eps - B)**2)*sigma**2*0.5 ),
                    np.exp( -((r+eps - C)**2)*sigma**2*0.5 )))

	if filtertype == 'bandpass_inverse': #gaussian notch filter
		A = (X**2+Y**2)**0.5/702.0
		B=(A-r+eps)*(A-r+eps< 0)+r-eps
		C=r+eps - (A-r-eps)*(A-r-eps> 0) 
		Zfreq = 1-np.fft.fftshift(np.minimum(np.exp( -((r-eps - B)**2)*sigma**(-2.0)*0.5 ),
                    np.exp( -((r+eps - C)**2)*sigma**(-2.0)*0.5 )))
        
	return height*Zfreq + (1.0 - height)


def filter_application(img_transform, args):
    """
    applies a filter to an image.  
    INPUTS:
    Images are size(352,352,3),
    args are the freqfilter arguments

    OUTPUT:
    a (352,352,3) size array of integers representing a filtered image
    """

	c = (352,352)
	fftrawr = img_transform[0]
	fftrawg = img_transform[1]
	fftrawb = img_transform[2]
	new_img_avg = img_transform[3]

	x = np.arange(1-c[0],c[0])
	y =np.arange(1-c[1], c[1])

	Zfreq = freqfilter(x,y,args)

	new_sharpened_img = np.zeros((2*c[0]-1,2*c[1]-1, 3))

	new_sharpened_img[:,:,0] = np.fft.ifft2(fftrawr*Zfreq)
	new_sharpened_img[:,:,1] = np.fft.ifft2(fftrawg*Zfreq)
	new_sharpened_img[:,:,2] = np.fft.ifft2(fftrawb*Zfreq)

	new_sharpened_img = new_sharpened_img +new_img_avg

	return new_sharpened_img.astype(int)[c[0]/2:c[0]/2 + c[0], c[1]/2:c[1]/2+c[1]]    


def filter_image(img, args, padding = 'reflect'):
	"""
    takes an image and returns a filtered image (combines filter_application, padded_image_transform, and freqfilter).
    INPUTS:
    img: a (352,352,3) size array
    args: freqfilter arguments
    padding: How you want your image to be padded for fourier transforms to take place

    OUTPUTS:
    a (352,352,3) size array of filtered image
	"""
	c = img.shape
	new_img = np.pad(img, ((176,175),(176,175),(0,0)), mode=padding)
	new_img_avg = np.average(new_img, axis = (0,1))
	new_img = new_img - new_img_avg
	fftrawr = np.fft.fft2(new_img[:,:,0])
	fftrawg = np.fft.fft2(new_img[:,:,1])
	fftrawb = np.fft.fft2(new_img[:,:,2])

	x = np.arange(1-c[0],c[0])
	y =np.arange(1-c[1], c[1])

	Zfreq = freqfilter(x,y,args)

	new_sharpened_img = np.zeros((703,703,3))

	new_sharpened_img[:,:,0] = np.fft.ifft2(fftrawr*Zfreq)
	new_sharpened_img[:,:,1] = np.fft.ifft2(fftrawg*Zfreq)
	new_sharpened_img[:,:,2] = np.fft.ifft2(fftrawb*Zfreq)

	new_sharpened_img = new_sharpened_img + new_img_avg
	return new_sharpened_img.astype(int)[176:176+352,176:176+352]		
	
sample_big_args = [['lowpass', i*24/150.0, i/150.0, i*0.25/150.0] for i in range(151)]	
        
def visualize_individual_certainty(filename, weights, mod, bigargs = sample_big_args, number_times = 151,  avg = 1.0):
	"""
	gives uncertainty levels of cars in an image over increasing levels of filtration
	INPUTS:
	filename: name of image file, but has no .*** at the end.
	weights: filepath/to/weights.caffemodel
	mod = file/path/to/deploy_file.prototxt
	bigargs = list of freqfilter args lists: a sample is 
            sample_big_args = [['lowpass', i*24/150.0, i/150.0, i*0.25/150.0] for i in range(151)]	
    number_times = length of bigargs, or if you want, it can be less that that
	avg: a float between 0 or 1, or strings 'center' or 'block'.

	OUTPUT:
	z: a list of lists of uncertainty levels.  z itself is of length n, where n is the number of cars in the 		image.  z[i] is of length number_times.   z[i][k] is the uncertainty level or uncertainty array (depending on choice of avg) for blurring level = sigmas[k] (see description of probabilities() output)
	sigmas: the array of blurring levels for z
	prediction: a list of the prediction boxes for each blurring level.  The list is of length number_times.  prediction[k] is the predicted boxes for image with blurring level sigma[k].  Prediction[k][i] is the i^th car, and the rows are of the form of rows in probabilities(), depending on the choice of 'avg'
	
	"""

	im_path = image_path+filename+'.png'
	lab_path = label_path+filename+'.txt'
	height = 1408
	width = 1408

	
	image = PIL.Image.open(im_path)
	image = image.convert('RGB')
	image = np.array(image)	

	img = load_image(im_path, height,width,args = bigargs[0])
	img_transform = padded_image_transform(image)

	scores, cvg_scores =detect(caffemodel= weights,deploy_file= mod, image_files=  [im_path], use_gpu=True, args = bigargs[0])
	rows,real = probabilities(scores[0], cvg_scores, lab_path,avg=avg)

	prediction = [rows]
	counter=1
	if avg =='block':
	    s=[[n[1]] for n in real] 
	else:
		s = [[n[4]] for n in real]
	z=s



	for i in range(1, number_times):

		scores, cvg_scores =detect(caffemodel= weights,deploy_file= mod, image_files=  [im_path], use_gpu=True, transform = img_transform, args= bigargs[i])
		rows,real = probabilities(scores[0], cvg_scores, lab_path,avg)

		if avg == 'block':
			z = [z[m]+[real[m][1]] for m in range(len(z))]
		else: 
			z = [z[m]+[real[m,4]] for m in range(len(z))]
		prediction = prediction + [rows]
		counter+=1

	return z,bigargs,prediction


def iou(box1,box2):
	"""
	Calculates the iou (intersection over union) threshold for two bounding boxes
	ARGUMENT:
	box1 and box2: two lists/arrays of the form [left, top, right, bottom]

	OUTPUT:
	a float giving the area of the intersection over the area of the union of box1 and box2
	"""
	if box1[0]>= box2[0]:
		leftbox = box2
		rightbox = box1
	else:
		leftbox = box1
		rightbox = box2
	lbound = rightbox[0]
	if rightbox[0] <= leftbox[2]:
		rbound = min(leftbox[2], rightbox[2])
        
		if leftbox[1]>= rightbox[1]:
			bottombox = leftbox
			topbox = rightbox
		else:
			bottombox = rightbox
			topbox = leftbox
		tbound = bottombox[1]
        
		if topbox[3] >= bottombox[1]:
			bbound = min(topbox[3], bottombox[3])
		else:
			return 0
        
		ia = (rbound-lbound +1.0 )*(bbound - tbound +1.0)
		
		oa = (topbox[2] - topbox[0]+1.0)*(topbox[3] - topbox[1]+1.0)+(bottombox[2] - bottombox[0]+1.0)*(bottombox[3] - bottombox[1]+1.0) -ia
        #print [ia, oa]    
		return ia/oa

	else:
		return 0
        

def pr_area(x,y):
	"""
	calculates the area under the pr-curve.  Is calculated by adding up the areas of trapezoids formed by the lines formed between points and the boundaries from the x values.
	INPUTS:
	x: x values of curve, in array or list form
	y: y values of curve, in array or list form

	OUTPUT:
	total_area: a float value
	"""
	total_area = 0
	trap_nums = len(x)
	for i in range(trap_nums-1):
		total_area += (x[i]-x[i+1])*(y[i]+y[i+1])*0.5
        
	return total_area

	
def get_pr_values(y_true, y_predict, t, match_thresh = 0.7): 
	"""
	breaks down predictions and ground truths into true positive, false positive, and false negative categories.

	INPUTS:
	y_true: an n x 4 size array, where n is the number of cars in the image
	y_predict: an n x 6 size array, where the 6th column is the uncertainty value
	match_thresh: a float between 0 and 1 to indicate the minimu iou value to designate true positives

	OUTPUT:
	tp: true positive match
	fn = false negative
	fp = false positive
	truth = number of true images
	prediction number of prediction
	"""

	#gets pr values for a specific threshold
	predict_thresh = y_predict[y_predict[:,5] > t]
	truth = len(y_true)
	predict = len(predict_thresh)
	#print predict
	match_true = np.zeros(truth)
	match_predict = np.zeros(len(y_predict))
	for j in range(predict):
		for i in range(truth): 
			ious = iou(y_true[i,:],predict_thresh[j,:])
			#print ious
			if ious > match_thresh:
				if match_true[i] == 0:
					match_true[i]=1
					match_predict[j] = 1
					continue
    
	tp = sum(match_true)
	fn = truth -tp
	fp = predict - tp

	return tp, fn, fp, truth, predict


def pr_curve(y_true,y_predict, match_thresh = 0.7):
	"""
	#calculates the precision recall curve over increments of 0.02 for a single image

	INPUTS:
		y_true: a list or array of shape (n,4) with n being the number of cars, and each row consisting of the left,top,right,bottom coordinates.
		y_predict: list/ array of size (m,4) with predictions of cars
		match_thresh: iou threshold, float number between 0 and 1, default is 0.7

	OUTPUT:
		[x,y], where x is a list the true positive rate (true positive/ truth), and y is a list of the false positive rate (true positive/ prediction).  the values range over certainty thresholds from 0 to 1., but also include the theoretical cases for absolute 0 and absolute 1.
	"""
	#makes list data into array data for easier computation 
	if type(y_predict[0])!=np.ndarray:
		y_predict = np.array([y_predict])


	truth = len(y_true)			# number of cars in image
	thresholds = np.linspace(0,.98,50)	# array of threshold values
	fpr=[]				# rate will be added
	tpr =[]
	for t in thresholds:
		
		tp, fn, fp, truth, predict = get_pr_values(y_true, y_predict,t, match_thresh = 0.7, ratio = True)

	if predict!=0: fpr = fpr +[tp/predict]
	else: 
		fpr = fpr+ [0]
	tpr = tpr +[tp/truth]

	return [[1]+tpr+[0], [0]+fpr+[1]] , 



def hamham(img, dim = 3):
	"""
	Hamming function for diminishing spectral leak.
	ARGUMENT:
	img: image in array form. can be in rgb or grayscale, 
	dim: dimensions of image (either 1 or 3).  must correspond to dimensions of image
	OUTPUT:
	I: image array with integer vals

	"""
	s = img.shape
	x = np.hamming(s[0])
	y = np.hamming(s[1])

	X,Y = np.meshgrid(y,x)

	if dim == 3:
		I=np.zeros((s[0],s[1],s[2]))
		Z = X*Y

		for i in range(3):
			I[:,:,i] = img[:,:,i]*Z

		return np.round(I).astype(int)
    
	I=img*X*Y
	return np.round(I).astype(int)
    

#This function is somehwat complicated.  Please read the information below on how to use it!    
def small_fourier(im, coords, bordsize, edge = 0, color = 0,make_odd = True):
	"""
	Gives the local fourier transform of a region in an image.  Transforms are controlled with hamming windows.  See hamham() function for better description. 
	INPUTS:
		im: Array, is the original image.  Can be grayscale or rgb
		coords: the integer list of coordinates of the image for which you want the small fourier window.  Is of the form [left, up, right, down]
		bordsize: is either an int or a list of two ints.  if an int, it's the same as [width, height] where width = int = height.  
		edge: tells you whether bordsize corresponds to the size of the raw window with the center of said window being the coords' center, or whether you want it to be the size of the border around the window.  This is helpful if you want to study set window sizes for certain comparisons, or just want the windows to be dependent on the original coord box dimensions.  If edge = 0, bordsize gives edge dimensions.  if edge = 1, bordsize gives raw window dimensions.
		color: choose 1 for grayscale or 3 for color
	OUTPUT:
		freq: array, Fourier transform of image (has 3rd dimension if you wanted RGB transform)
		image: cropped image of region with borders determined by coords.
	"""
	#print coords
	if make_odd == True:
		coords[3] = coords[3] + (1 - (coords[3] - coords[1])%2 )
		coords[2] = coords[2] + (1 - (coords[2] - coords[0])%2 )
	#color states whether you want to grayscale the image or not.
	if type(bordsize) == int: 
		bordsizeW = bordsize
		bordsizeH = bordsize
		#print bordsizeW, bordsizeH
	else: 
		bordsizeW = bordsize[0]
		bordsizeH = bordsize[1]
	im2 = im #- np.average(im)
	if edge == 0:
		left = max(0, int(coords[0]-bordsizeW))
		up = max(0, coords[1]-bordsizeH)
		right = min(352, coords[2]+bordsizeW)
		down = min(352, coords[3]+bordsizeH)
		#print up,down,left,right
	else:
		a_center = (coords[2]+coords[0])/2
		b_center = (coords[3]+coords[1])/2
		#print a_center, b_center
		left = a_center-bordsizeW/2-1
		right = a_center+bordsizeW/2
		up = b_center-bordsizeH/2-1
		down = b_center+bordsizeH/2
		#print left,right,up,down
	
	if color == 1:
		r = im2[up:down,left:right,0]/1.0
		g = im2[up:down,left:right,1]/1.0
		b = im2[up:down,left:right,2]/1.0

		r_transform = np.fft.fft2(np.fft.fftshift(hamham(r,1)))
		g_transform = np.fft.fft2(np.fft.fftshift(hamham(g,1)))
		b_transform = np.fft.fft2(np.fft.fftshift(hamham(b,1)))


		s = r.shape

		freq = np.zeros((s[0],s[1],3))

		freq[:,:,0] = r_transform
		freq[:,:,1] = g_transform
		freq[:,:,2] = b_transform
        
	else:
		if len(im.shape) ==3: 
			im2 = rgb2gray(im2)


		r = im2[up:down,left:right]/1.0
		freq = np.fft.fftshift(np.fft.fft2(hamham(r,1)))
        
        
	return freq, im[up:down, left:right]


def sift_data(data, border_size):
    """
    Goes through bounding boxes and removes those too close to the edge of an image.  Use this when wanting to examine car behavior over blurring areas, since even with the window-dampeners, car blurring behavior around the edges may be weird.
	ARGUMENT:
	data: a list of arrays/lists representing the bounding boxes of a car data[i] =[left,up,right,down].
	border_size: int, how far from the edge we require our cars to be, in pixels
	OUTPUTS:
	new_data: sublist of data
	data_index: indices of the rows in data that made up new_data
    """
    new_data = []
    data_index = []
    for n in range(len(data)):
        if (data[n][0] - border_size >= 0) and (data[n][2]+border_size < 352):
            if (data[n][1] - border_size >= 0) and (data[n][3]+border_size < 352):
                new_data = new_data +[data[n]]
                data_index = data_index + [n]
    return new_data,data_index  
    

def rgb2gray(rgb):
	"""
	Turns color image into grayscale image.
	ARGUMENT: rgb array of image (3-dim)
	OUTPUT: grayscale image (2-dim)
	"""
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    

#This is also a complicated function!!! read carefully
def fourier_of_blur(img,boxsize,coords,i,transform = None, bigargs=sample_big_args, edge = 1,blurchoice = 1):
	"""
	gets the fourier transform of an image blurred a certain sigma level. Used as an assistant function to fourier_chage()

	INPUTS:
	img: is a list of the choice images, in rgb array format, or a single image.  if a list, then set blurchoice to 0, since it will blur the ith image
	boxsize: int, is the size of the fourier transform (edges or actual size), 
	coords: is the coordinates where you want to get the fourier window, in [left,top,right,bottom] form 
	i: is the ith image in the image list that you want filtered (it's presumed to be an already filtered iamge).  
    transform: fourier transform of initial image
    bigargs: list of freqfilter args (see freqfilter() )
	edge: boolean, see small_fourier() for description
	blurchoice: set to 0 if img is a list of images, set to 1 if it's a single image.   

	OUTPUT:
	fourier_image: fourier transform of blurred image
	"""   

	if blurchoice == 1: 
		args = bigargs[i]
		new_img = filter_application(transform, args)

	else: new_img = img[i]
    
	fourier_img,_ = small_fourier(rgb2gray(new_img), coords, boxsize,0, edge)

	return fourier_img



def adapted_fourier_change(img,coords,edgesize, n = 151, transform = None, blurchoice = 1, bigargs = sample_big_args):
	"""
	like fourier change, but instead is adapted to boundaries of different sizes.  Edgesize gives the size of the border.  See description for fourier_change()
	"""
	#print coords
	if (coords[2] - coords[0]) % 2 == 0:
		coords[2]+=1
	if (coords[3] - coords[1]) % 2 == 0:
		coords[3]+=1
	#print coords
	vals = np.zeros((coords[3]-coords[1]+2*edgesize, coords[2] - coords[0]+2*edgesize,n))
	#print vals.shape, #fourier_of_blur(img,edgesize,coords,10, edge = 0,blurchoice=0).shape
	
	for i in range(n):
		#print i
		#fourier_of_blur(img,boxsize,coords,i,transform = None, bigargs=sample_big_args, edge = 1,blurchoice = 1):
		vals[:,:,i] = fourier_of_blur(img,edgesize,coords,i, edge = 0, transform = transform, blurchoice=blurchoice, bigargs = bigargs)
	#print sigmas[-1]	
	return vals

def get_good_img_data(m,good_filenames,label_path,image_path):
	"""
	Gets important data for a particular image for working with other functions.  This is a convenience function.
	INPUTS:
	m: the mth image in good_filenames
	good_filenames: a list of filenames of images in a folder (these are without endings)
	label_path: str, /path/to/label/folder/
	image_path: str, /path/to/image/folder/
	OUTPUTS:
	new_data: list of boundaries for cars at least 20 pixels away from the edges
	data_index: indices of new_data cars in "data" parameter 
	data: list of all car boundaries in image
	img: 3-D integer array of the mth image in good_filenames
	sigmas: array of default filtration intensity levels (don't use this, it's like an appendix)
	img_transform: patted fourier transform of image
	"""	
	#gets necessary data from image to work on


	image = PIL.Image.open(image_path+good_filenames[m]+'.png')
	image = image.convert('RGB')
	image = np.array(image)	
	img_transform = padded_image_transform(image, padding = 'reflect')
    

	sigmas = np.linspace(0,3,151)
	img = load_image(image_path+good_filenames[m]+'.png',352,352)
	f = open(label_path+good_filenames[m]+'.txt', 'r')
	data = np.array([[int(n) for n in line.split(' ')[4:8]] for line in f.read().split('\n')[:-1]])

	new_data, data_index = sift_data(data,20)

	return new_data, data_index, data, img, sigmas, img_transform


		
def maximal_iou_match(y_true, y_predict,n): 
    """
	returns iou data for a specific car.  This is to gather more information about prediction matches.  This is a local function. 
	INPUTS:     
	y_true: the ground truth boxes of an image
    y_predict: is the predicted boxes
    n: the nth car in the truth box
	 	PROTIP: This function can be used to weed out both false positives and false negatives.  To find the false negatives, put the inputs in as designated.  To find the false positives, you can set the predicted boxes to y_true, and the ground truth boxes to y_predict.
	OUTPUTS: 
	maxiou: the max iou value between the nth car and the predicted cars
    maxiou_ind: the predicted dimensions of the car with maxiou
    maxiou_prediction: the max iou value between maxiou_ind and the true cars
	maxiou_pred_ind: the dimensions of the ground_truth box generating maxiou_prediction for car n
	rank: number of prediction boxes whose iou with than that of car maxiou_ind is at least that of car n.  This is helpful in determining if the prediction box may actually be going to a different car that happens to be close by.
	"""

    ious = [iou(y_true[n,:],y_predict[k]) for k in range(len(y_predict))]
    
    if len(ious)>0:
        maxiou = np.max(ious)
    #print maxiou
        if maxiou > 0:
            maxiou_ind = y_predict[np.argmax(ious)]
         #   print maxiou_ind
            ious_for_pred = np.array([iou(maxiou_ind,y_true[k]) for k in range(len(y_true))])
         #   print ious_for_pred
            maxiou_pred = np.max(ious_for_pred)
         #   print maxiou_pred
            maxiou_pred_ind = y_true[np.argmax(ious_for_pred)]
         #   print maxiou_pred_ind
            rank = len(ious_for_pred[ious_for_pred >= maxiou])

            return maxiou, maxiou_ind, maxiou_pred, maxiou_pred_ind, rank
    return 0,0,0,0,0


            
def extract_false_positives(ground_truths, predictions, thresh=0):
    """extracts the isolated false positives of an image detection.  These are the false positives for which there is little to no overlap
    INPUTS:
    ground_truths: the ground truth boxes of an image
    predictions: the predicted bounding boxes of an image
    threshold: a flot between 0 and 1 telling you the upper bound for the IOU threshold separating false from true positives

    OUTPUTS: an array of the coordinates of false positive cars
    """
    
    false_positives =[]
    false_positive_index = []
    for i in range(len(predictions)):
        ious = [iou(ground_truths[k],predictions[i]) for k in range(len(ground_truths))]
        maxiou = np.max(ious)
        if maxiou <= thresh:
            false_positives = false_positives + [predictions[i]]
            false_positive_index = false_positive_index +[i]
            
    return np.array(false_positives), false_positive_index

def  prediction_certainty_curve(prediction,cvgs, avg='center', thresh =0,sigtimes = 151):
    """

    produces the uncertainty curves for the isolated false positive predictions for a single image, but I actually used this function for ground truth bounding boxes as well.
    INPUTS:
    prediction: prediction boxes for testing (a nx4 array) of the image
    cvgs: a list of coverage maps, size (88,88), each coverage map corresponds to one point in the certainty curve

    avg: mode of determining uncertainty: put forward a float between 0 and 1 or 'center' for now.  The block version doesn't work well.
    sigtimes: length of certainty curve.
    OUTPUTS:
    z: a list of certainty curves for the image.
    """
    
    z=[[]]*len(prediction)
    #print prediction
    pred_bounds = prediction[:,:5]
    #print (pred_bounds/4)[:,:4].astype(int)
    #sigind = np.where((sigmas>=s1)*(sigmas<=s2))[0]
    sigind = range(sigtimes)
    
    if avg != 'block':
        for i in sigind:
            #print i
            x=probabilities(pred_bounds, cvgs[i],avg=avg)
            #print x
            z = [z[m]+[x[m,-1]] for m in range(len(z))]

        return z
    for i in sigind:
        x= probabilities(pred_bounds*4, cvgs[i], avg = avg)
        x = [x[i][1] for i in range(len(x))]
        z= [z[m]+[x[m]] for m in range(len(z))]
        #print i
    return [np.asarray(zz) for zz in z]

def uncertainty_cvg_maps(good_filenames, bigargs, weights, mod, filenamelist=None, mode='RGB',cvg_z = [], modify_name=''):
	    """ gives back a set list of coverage maps for images of good_filenames numbered by the elements in filenamelist.   Also saves predictions in a file so that work won't be lost should something happen.
    INPUTS:
    good_filenames: names of files to be detected, list of strings, don't add .txt or .png
    bigargs = list of freqfilter "args" (see freqfilter() )
    weights = path/to/file.caffemodel 
    mod = path/to/file.prototxt
    filenamelist: indices for choosing parts that you want of good_filenames. 
    mode: choice of # of image channels.  I kept it at RGB throughout
    cvg_z = use as recovery.  If you ran the code and it didn't finish, load 

    bigargs should be an number_of_times x args_length list  
    """
	os.chdir(caffePath)
	if filenamelist == None:
		filenamelist=range(len(good_filenames))
	height = 1408
	width = 1408
	filtertype = bigargs[0][0]

   	prediction_z = []

	for fnum in filenamelist: #finds image
		start = time.time()
		filename = good_filenames[fnum] 
		im_path = image_path+filename+'.png'
		lab_path = label_path+filename+'.txt'
		image = PIL.Image.open(im_path)
		image = image.convert(mode)
		image = np.array(image)
		img_transform = padded_image_transform(image)
		little_z = []
		little_predictions = []
		for s in range(len(bigargs)): #detects image under filtration defined by bigargs[s]
			scores, cvg_scores =detect(caffemodel= weights,deploy_file= mod, image_files=  [im_path], use_gpu=True, transform = img_transform, args = bigargs[s])
			little_z = little_z +[cvg_scores] #adds coverage map
			little_predictions = little_predictions+[scores[0]] #adds predictions

		cvg_z = cvg_z + [little_z]
		prediction_z = prediction_z +[little_predictions]

        #saves files
		np.save(curdir+'cvgs_20_plus-'+filtertype+'-'+modify_name+'/'+good_filenames[fnum]+'.npy',little_z)
		np.save(curdir+'cvgs_20_plus-'+filtertype+'-'+modify_name+'-predictions.npy',prediction_z)
		end = time.time()
		print " processed image "+str(fnum)+" in "+ str((end-start)/60.0)[:5]+ " minutes..."
	os.chdir(curdir)
	return cvg_z



def frequency_magnitudes(m,n,typenum = 'real'):
	"""
	Gives an array of m rows x n columns that gives complex values 
	in [-0.5, 0.5] x i[-0.5, 0.5]

	"""
	assert (typenum == 'real') or (typenum == 'complex')
	#A = np.zeros((m,n))
	centii = (m-1)/2
	centjj = (n-1)/2

	if typenum == 'real':
		for i in range(m):
			for j in range(n):
				A[i,j] = np.sqrt(((i-centii)/np.float(m-1))**2 + ((j-centjj)/np.float(n-1))**2)
	else:
		x = np.linspace(-0.5,0.5,m)
		y = np.linspace(-0.5,0.5,n)

		Y,X = np.meshgrid(y,x)
		A = X +complex(0,1)*Y

	#    for i in range(m):
	#        for j in range(n):
		        #print (i-centii)/np.float(m-1), (j-centjj)/np.float(n-1)
	#            A[i,j] = complex((i-centii)/np.float(m-1), (j-centjj)/np.float(n-1) )
	return A
    
def interpolated_frequency_spectrum(A,r=0.5, n=60):
    """ Takes a discrete fourier transform matrix A and returns the average amplitude in a given frequency ring r, using interpolation where necessary.

    INPUTS:    
    A:  a 2-D array with odd dimensions. Will be treated like frquency space
    r: radius of ring for which you want to measure energy.  Pick value between 0 and sqrt(0.5)
    n: number of points to sample from to get energy levels. If measuring a fourier transform of an image, there is a rotational symmetry, so you only need to do the top half of the image.  So n = 60 means you have 60 points sampled over the top half of the transform.  This cuts computational time in half

    OUTPUTS:
    a float, returns average energy of ring of radius r in frequency space A.
    """    

	assert A.shape[0]*A.shape[1]%2 == 1, 'need both dimensions of matrix to be odd'
	assert 0<= r<= np.sqrt(0.5), 'need r to be within fourier frequency magnitude'

    #Gives absolute value for amplitudes
	A =np.abs(A)

	l=[]
	dt = np.pi/n
	theta = np.linspace(0, np.pi-dt, n)

	y = np.linspace(-0.5,0.5,A.shape[0])
	x = np.linspace(-0.5,0.5,A.shape[1])  
	dx = x[1]-x[0]
	dy = y[1]-y[0] 

    #selects sampling of point. working with fourier transform of real image, so only needs top half.
	coordpoints = r*np.cos(theta) + complex(0,1)*r*np.sin(theta)

	for point in coordpoints:

        #ensures we only stay within confines of discrete frequency space.
		if (np.abs(point.real)<= 0.5) & (np.abs(point.imag) <= 0.5):
			left = np.where(x<= point.real)[0][-1]
			top = np.where(y<= point.imag)[0][-1]
			if (point.real == 0.5 ): 
				left = left - 1
			if (point.imag == 0.5): 
				top = top -1

			l = l+ [ (A[top,left]*((x[left+1]- point.real)/dx)+A[top,left+1]*(np.abs(point.real- x[left])/dx) )*((y[top+1]-point.imag)/dy)
				+(A[top+1,left]*((x[left+1]-point.real)/dx)+A[top+1,left+1]*(np.abs(point.real- x[left])/dx) )*((point.imag - y[top])/dy)  ]

    #gets average energy
	l = np.sum(l)/len(l)
	return l
  

def find_midpoint_of_curve(z, sigma,threshold = 0.5, absolute = 0):
	"""
	Finds the point at a certain percentile of the range of your curve.  Used for certainty curves and generally assumes that the curve is decreasing
	INPUTS:
		z = array/ list of y-points
		sigma: array/ list of x-points
		threshold: the precentile you want
        absolute: 0 or 1: If 0, then sets theshold to a relative threshold (i.e., of threshold is 0.6, then setting absolute = 1 finds point in the curve where f(x) = 0.6*max(f) + (1-0.6) min(f).  If set to 1, finds x such that f(x) = 0.6
 
	OUTPUTS:
		[x-value of the threshold cutoff, max x-value, min x-value], max, min
	"""

    #Ensures that cutoff is between maximum and minimums
	zargmax = np.argmax(z)
	zargmin = np.argmin(z)

	if absolute == 0: zcenter = threshold*z[zargmax]+(1-threshold)*z[zargmin]
	else: zcenter = threshold

	a = zargmax
	b = zargmin

    #finds n where z[n] > threshold and z[n+1] < threshold.
	while b - a > 1:
		c = (b+a)/2
		if z[c] > zcenter:
			(a,b) = (c,b)
		else:
			(a,b) = (a,c)
    #interpolates between f[n] and f[n+1] to find threshold.
	t = (z[b] - zcenter)/(z[b]-z[a])
	cutoff_sigma = t*sigma[a] + (1-t)*sigma[b]

    #returns [cutoff point, argmax, argmin], max, min
	return [cutoff_sigma, sigma[zargmax], sigma[zargmin]], zargmax,zargmin

def get_all_image_stuff(filename, args = ['lowpass',0.0,0.0,0.0]):
    """
    Retrieves different pieces of data from a single image.
    INPUTS:
    filename: name of file, no endings like .png or .txt
    args: freqfile args (see freqfile())
    
    OUTPUTS:
    im_path: image path to specific image
    lab_path: label_path to bounding boxes in image
    height,width : height and width you want the image to dilate (right now it's set to 1408.  Change the code if you want it to be different?=)
    image: the dilated and transformed image using args
    """

    #get im_path, lab_path, height and width of image
	im_path = image_path+filename+'.png'
	lab_path = label_path+filename+'.txt'
	height = 1408
	width = 1408

    #load image with attached filters.
	image = PIL.Image.open(im_path)
	image = image.convert('RGB')
	image = np.array(image)	
	img = load_image(im_path, height,width,args = args)
	img_transform = padded_image_transform(image)

	return im_path, lab_path, height, width, image, img, img_transform
    
def get_cutoff(filename, k, weights, mod, 
               image_path, label_path, args = ['lowpass', 0.0, 0.0, 0.0],
               jump = 10.0, thresh = 0.5, xeps = 0.0001, yeps = 0.0001, loops = 20, i=1, lowbound = -100000, highbound = 1000000):
    """
    Gets the cutoff point for a car without a prior certainty curve.
    INPUTS:
        filename, weights, mod, image_path_label_path: see info for detect()
        args: freqfilter args (see freqfilter() )
        jump: initial sigma jump amount.  The bigger, the larger the starting point.
        thresh: the final cutoff sigma that you want
        xeps: the sigma error bound that you want for stopping the loop
        yeps: the certainty error bound that you want for stopping the loop
        loops: the max number of times you want this function to loop before you stop the function.  
            Default is set to 20, but don't make it too high!
        i: index of args that you want to find cutoff for.  Should be >= 1.  For example, if you are adjusting over args[2], then the other arguments are fixed and you are trying to find the cutoff for an args[i] variable.
        lowbound,highbound: use this when finding cutoffs for notch or bandpass filters.  Essentially, we assign limits to where the cutoff can be.  If I am finding a cutoff over epsilon with a notch filter, I expect the cutoff bounds to be between 0 and sqrt(0.5), for example.  If there is no bound, just put in a very large number

    OUTPUTS:
        cutoff: the sigma approximating your cutoff point
        cert_score: the certainty score c_k^{blah}(x)
    """
    
    im_path, lab_path, height, width, image, img, img_transform = get_all_image_stuff(filename, args)
    filtertype = args[0]
    #x = 0.0 
    old_args, new_args = args, args
    #dx = jump
    dy = 1.0
    scores, cvg_scores =detect(caffemodel= weights,deploy_file= mod, 
                                   image_files=  [im_path], use_gpu=True, 
                                   transform = img_transform,
                                   args = new_args)
    rows,real =probabilities(scores[0][k].reshape((1,5)), cvg_scores, lab_path, avg = 'center', carnums = [k])
    cert_score = real[4]
    #print "initial certainty score: "+ str(cert_score)
    if cert_score < thresh:
        return -0.1, cert_score
    counter = 0
    direction = -1.0
    while cert_score > thresh:
        #print new_args
        if (new_args[i] < lowbound) or (new_args[i] > highbound):
            return -1, cert_score
        if jump == 0:
            return -1, cert_score
        
        old_args[i],new_args[i] = new_args[i], new_args[i] + jump 

        #keep adding jumps until the certainty gets below the threshold
        old_cert_score = cert_score
        scores, cvg_scores =detect(caffemodel= weights,deploy_file= mod, 
                                   image_files=  [im_path], use_gpu=True, 
                                   transform = img_transform,
                                   args = new_args)
        rows,real = probabilities(scores[0][k].reshape((1,5)), cvg_scores, lab_path, avg = 'center', carnums = [k])
        cert_score = real[4]
        

    #now start decreasing the size of the jump
    dy = old_cert_score - cert_score  

    #use binary cuts to find cutoff.
    while ((abs(jump) > xeps) or (max(abs(old_cert_score - thresh), abs(old_cert_score - thresh) ) > yeps)) and (counter < loops):
        jump = jump/2.0

        #find midpoint of interval where cutoff should be
        old_args[i],new_args[i] = new_args[i], new_args[i] + jump*direction
        scores, cvg_scores =detect(caffemodel= weights,deploy_file= mod, 
                                   image_files=  [im_path], use_gpu=True, 
                                   transform = img_transform,
                                   args = new_args)

        rows,real = probabilities(scores[0][k].reshape((1,5)), cvg_scores, lab_path, avg = 'center', carnums = [k])

        # find the half of the interval where the threshold crossing occurs, make this new interval
        if real[4] < thresh:
            if direction == 1:
                old_cert_score, cert_score = cert_score, real[4]
            else:
                cert_score = real[4]
            direction = -1.0
        else:
            if direction == -1:
                old_cert_score, cert_score = cert_score, real[4]
            else:
                cert_score = real[4]
            direction = 1.0
        
        
        dy = abs(old_cert_score - cert_score)
        counter+=1

    #pick final interval
    old_test_sig, test_sig = old_args[i],new_args[i] 

    if old_cert_score == cert_score:
        return 0.5*(test_sig +old_test_sig)

    #and use interplation to approximate where cutoff point is.
    max_sig,min_sig = max(test_sig, test_sig + jump*direction), min(test_sig, test_sig + 2*jump*direction)
    top,bottom = max(old_cert_score, cert_score), min(old_cert_score, cert_score)

    t = -(bottom - thresh)/dy
    
    cutoff = t*min_sig + (1-t)*max_sig
    
    return cutoff, cert_score
 
