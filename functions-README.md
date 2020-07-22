This is a README file containing the information on each of the functions in evaluation_and_analysis_functions.py and visualize_data.py.  

FOR OBJECT DETECTION (a lot of these are adaptation of DIGITS functions)

1. get_net(caffemodel, deploy_file, use_gpu=True): #get caffe model.  Use GPU if computer has it
   
    Returns an instance of caffe.Net

    INPUTS:
    caffemodel -- 'path/to/a.caffemodel' 
    deploy_file -- 'path/to/a.prototxt' 
    use_gpu -- if True, use the GPU for inference

    OUTPUTS: get a caffe.net
    

2. get_transformer(deploy_file, mean_file=None):
 
    Returns an instance of caffe.io.Transformer

    INPUTS:
    deploy_file -- 'path/to/a.prototxt'
    Keyword INPUTS:
    mean_file -- 'path/to/a.binaryproto' file..
  

3. load_image(path, height, width, mode='RGB', args = [None,0.0, 0.0, 0.0], transform = None):

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

4.  forward_pass(images, net, transformer, batch_size=None):
 
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
    
5. read_labels(labels_file):

    Returns a list of strings

    INPUTS:
    labels_file -- path to a .txt file

6. detect(caffemodel, deploy_file, image_files,
        mean_file=None, labels_file=None, batch_size=None, use_gpu=False, transform = None, args =['lowpass',0.0, 0.0,0.0]):

	Classify some images against a Caffe model and get bounding box prediction as well as associated coverage map
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

28. uncertainty_cvg_maps(good_filenames, bigargs, weights, mod, filenamelist=None, mode='RGB',cvg_z = None, modify_name=''):
    gives back a set list of  coverage maps for images of good_filenames numbered by the elements in filenamelist.   Also saves predictions in a file.

    INPUTS:
    good_filenames: list of filenames in the test/images folder, just have the names without file endings '.*'
    bigargs: list of freqfilter args (see freqfilter() )
    weights: str, path/to/weights.caffemodel
    mod: str, path/to/deploy.prototxt
    filenamelist: list of integers... if this isn't None, it will only do coverage maps for the good_filenames files listed by  numbers in filenamelist
    mode: str, keep it at 'rgb' for this project
    cvg_z: an incomplete file containing the coverage maps. this was here in case my program stopped running for some reason.  Since I was periodically saving a file containing the coverage maps, I needed a way not to let my work go to waste.

    modify_name= str, anything additional for the file to be saved.  will be put at the end of the .npy file.

    OUTPUT:
    cvg_z: a (n, len(bigargs), 88,88) size array, where n is the number of images.  
    saves cvg_z in a file, as well as the predictions of detection for all n x len(bigargs).

    bigargs should be an number_of_times x args_length list 

<b>FOR DATA PROCESSING</b>
 (these functions are used to remove potentially problematic types of data.  No, I don't mean data that will go against my preconceived hypotheses)

20. sift_data(data, border_size):

    Goes through bounding boxes and removes those too close to the edge of an image.  Use this when wanting to examine car behavior over blurring areas, since car's behavior filtration behavior around the edges may be weird.  Also, detectnet sometimes gives negative coordinates.  It's a smart little machine, but it's annoying to deal with these.

	INPUTS:
	data: a list of arrays/lists representing the bounding boxes of a car data[i] =[left,up,right,down].
	border_size: int, how far from the edge we require our cars to be, in pixels
	OUTPUTS:
	new_data: sublist of data
	data_index: indices of the rows in data that made up new_data

24. get_good_img_data(m,good_filenames,label_path,image_path):
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

32. get_all_image_stuff(filename, args = ['lowpass',0.0,0.0,0.0]):
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

<b> FOR GENERATING CERTAINTY CURVES <b> 

7. probabilities(scores, cvg_scores, lab_path = None,avg = 1.0, carnums = None): 
	returns a list of cars with bounding boxes and a "probability" of objectness
    if test is the filename, compare to the ground truth, if not, base it off of the results alone.
	curently three ways of measuring objectness scores: averages for pixels above
	a certain threshold, taking maximal pixel, and Taking the center pixel. 

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

12. visualize_individual_certainty(filename, weights, mod, bigargs = sample_big_args, number_times = 151,  avg = 1.0):
	gives uncertainty levels of cars in an image over increasing levels of filtration.  This is an older function, but it still is usable.

	INPUTS:
	filename: name of image file, but has no .*** at the end.
	weights: filepath/to/weights.caffemodel
	mod = file/path/to/deploy_file.prototxt
	bigargs = list of freqfilter args lists: a sample is 
            sample_big_args = [['lowpass', i*24/150.0, i/150.0, i*0.25/150.0] for i in range(151)]	
    number_times = length of bigargs, or if you want, it can be less that that
	avg: a float between 0 or 1, or strings 'center' or 'block'.

	OUTPUTS:
	z: a list of lists of uncertainty levels.  z itself is of length n, where n is the number of cars in the 		image.  z[i] is of length number_times.   z[i][k] is the uncertainty level or uncertainty array (depending on choice of avg) for blurring level = sigmas[k] (see description of probabilities() output)
	sigmas: the array of blurring levels for z
	prediction: a list of the prediction boxes for each blurring level.  The list is of length number_times.  prediction[k] is the predicted boxes for image with blurring level sigma[k].  Prediction[k][i] is the i^th car, and the rows are of the form of rows in probabilities(), depending on the choice of 'avg'
	

27. prediction_certainty_curve(prediction,cvgs, avg='center', thresh =0,sigtimes = 151):
    produces the uncertainty curves for the isolated false positive predictions for a single image, but I actually used this function for ground truth bounding boxes as well.
    INPUTS:

    prediction: prediction boxes for testing (a nx4 array) of the image
    cvgs: a list of coverage maps, size (88,88), each coverage map corresponds to one point in the certainty curve

    avg: mode of determining uncertainty: put forward a float between 0 and 1 or 'center' for now.  The block version doesn't work well.
    sigtimes: length of certainty curve.
    OUTPUTS:
    z: a list of certainty curves for the image.
    """

<b>FOR IMAGE MANIPULATION <b>

8. padded_image_transform(img, padding = 'reflect'):
    returns the discrete fourier transform of each image for each color channel as a list, images are assumed to be (352,352,3), and each fourier transform is of size (703,703).  Change dimensions and padding values if you want to use this function for arbitrary images.

    INPUTS:
    img: a (352,352,3) array of integers
    padding: padding for your fourier transform.  I used reflect because it enabled continuity in the image without having sharp edges in the periodic function
   
9. freqfilter(x,y, args = ['lowpass',0.0,0.0,0.0]):
    Returns a filter in the form of an array.  Multiply the frequency space by the filter and then go back to time space to get your filtered image. Assumes filter is of size (703,703).  Change the specifications to allow for filters of arbitrary size. The filters divide values by 702. If you want to include more options, you can do so here.  This is the set up so that you do not have to worry about adjusting code elsewhere when wanting to try new filters on these images. 

    WARNING: the sigma values are with respect to fitler intensity (this is args[1]).  For highpass and notch filters, Sigma is the filter width.  For lowpass and bandpass filters, sigma variable is actually 1/ filter width.  Look through the filter specifications to tweak as necessary when using this program.  

    Current options for args[0]:
    lowpass, highpass, highpass2, highbandpass, highbandpass2, bandpass_inverse (this is the notch filter heading), bandpass

    Current options for args[5]: left, right

    INPUTS: 
    x, y: integer arrays.  I used x, y = np.arange(-351,352).
    args: filter arguments in the form of a list [str, float, float, float, float, str].  
    the argument order is [filtertype, 
                            filter intensity, (either sigma or 1/sigma) see above
                            radius, (used for notch or bandpass filters)
                            bandwidth, (used for notch or bandpass filters)
                            filter height, (number between 0 and 1, default is 1)
                            justify (for bandpass and notch filters) ]

    Args needs to be at least of length 4.  But you can make it longer, and you can also add extra parameters or filters.  You only need to change it here though.
 
    OUTPUT:
    freq: a (703,703) size array.

10. filter_application(img_transform, args):
    applies a filter to an image.  

    INPUTS:
    Images are size(352,352,3),
    args are the freqfilter arguments
    
    OUTPUTS: a (352,352,3) size array of integers represening a filtered image

11. filter_image(img, args, padding = 'reflect'):
    takes an image and returns a filtered image (combines filter_application, padded_image_transform, and freqfilter).

    INPUTS:
    img: a (352,352,3) size array
    args: freqfilter arguments
    padding: How you want your image to be padded for fourier transforms to take place

    OUTPUTS:
    a (352,352,3) size array of filtered image


<b>LOCAL FOURIER ANALYSIS </b> (these are function used for looking at frequencies of individual cars.

17. hamham(img, dim = 3):
	Hamming function for diminishing spectral leak.

	INPUT:
	img: image in array form. can be in rgb or grayscale, 
	dim: dimensions of image (either 1 or 3).  must correspond to dimensions of image
	OUTPUT:
	I image array with integer vals


18. small_fourier(im, coords, bordsize, edge = 0, color = 0,make_odd = True):
	Gives the local fourier transform of a region in an image.  Transforms are controlled with hamming windows.  See hamham() function for better description. This function was often used in a loop over mutiple coordinate lists "coords" for a single image "im," which was why I set it up this way.

    WARNING: this is a complicated function.  Look carefully at how the arguments work!  For this research, my default settings were:

    boordsize = 2 (it adds a small window around the bounding box.  This was both to ensure that the box actually surrounded the car, which occasionally wasn't the case due to labelling issues in the xview data set, and also because immediate surroundings had an influence on the car's detection)
    edge = 0    (cars were of various sizes)
    color = 1 (I would put rgb cars into this function)
    make_odd = True (to ensure symmetry of the local transform and the zero offset in the center)
    

	INPUTS:
		im: Array, is the original image.  Can be grayscale or rgb
		coords: the integer list of coordinates of the image for which you want the small fourier window.  Is of the form [left, up, right, down]
		bordsize: is either an int or a list of two ints.  if an int, it's the same as [width, height] where width = int = height.  
		edge: tells you whether bordsize corresponds to the size of the raw window with the center of said window being the coords' center, or whether you want it to be the size of the border around the window.  This is helpful if you want to study set window sizes for certain comparisons, or just want the windows to be dependent on the original coord box dimensions.  If edge = 0, bordsize gives edge dimensions.  if edge = 1, bordsize gives raw window dimensions.
		color: choose 1 for grayscale or 3 for color
        make_odd: adds at most one row or column of pixels to ensure that cropped image has odd dimensions.

	OUTPUT:
		freq: array, Fourier transform of image (has 3rd dimension if you wanted RGB transform)
		image: cropped image of region with borders determined by coords.


19. hamham(img, dim = 3):
	Hamming function for diminishing spectral leak.

	INPUT:
	img: image in array form. can be in rgb or grayscale, 
	dim: dimensions of image (either 1 or 3).  must correspond to dimensions of image
	OUTPUT:
	I: image array with integer vals

21. rgb2gray(rgb):
	Turns color image into grayscale image.  This is used to get individual fourier transforms for each car.
	ARGUMENT: rgb array of image size (n,m,3)
	OUTPUT: grayscale image  size (n,m)

22. fourier_of_blur(img,boxsize,coords,i,transform = None, bigargs=sample_big_args, edge = 1,blurchoice = 1):
	gets the fourier transform of an image blurred a certain sigma level. Used as an assistant function to fourier_chage().  This function is used mainly for local fourier transforms of cars.  Most of the times, I had img as a list, and this function was used as part of a loop (doing this saved a lot of computing time.  There are a lot of arguments, but these mirror the small Fourier arguments

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

23. adapted_fourier_change(img,coords,edgesize, n = 151, transform = None, blurchoice = 1, bigargs = sample_big_args):
	"""
	Gives a list of local fourier transforms for a car.   

    INPUTS:
	img: image or list of images of increasing blur
	coords: boundaries of bounding box in list form [left,top,right,bottom]
	edgesize: int, edge size of fourier window
    n = number less than or equal to length of bigargs
    transform: initial transform of img (included here because this was also called in a loop)
    blurchoice: put 1 if img is a single image, put 0 if img is a list of images.


	OUTPUT:
	list of fourier transforms of length n, each element being a 2-D array of the local fourier transform
	"""

<b> CUTOFFS AND THRESHOLD FUNCTIONS </b>
29. frequency_magnitudes(m,n,typenum = 'real'):

	Gives an array of m rows x n columns that gives complex values 
	in [-0.5, 0.5] x i[-0.5, 0.5]

    INPUTS: 
    m,n: dimensions of the array that you want.
    typenum: gives array with distance from center if real, gives complex values (complex is much better,  don't do real)

    OUTPUTS: the m x n array

30. interpolated_frequency_spectrum(A,r=0.5, n=60):
    Takes a discrete fourier transform matrix A and returns the average amplitude in a given frequency ring r, using interpolation where necessary.

    INPUTS:    
    A:  a 2-D array with odd dimensions. Will be treated like frquency space
    r: radius of ring for which you want to measure energy.  Pick value between 0 and sqrt(0.5)
    n: number of points to sample from to get energy levels. If measuring a fourier transform of an image, there is a rotational symmetry, so you only need to do the top half of the image.  So n = 60 means you have 60 points sampled over the top half of the transform.  This cuts computational time in half

    OUTPUTS:
    a float, returns average energy of ring of radius r in frequency space A.
    
31. find_midpoint_of_curve(z, sigma,threshold = 0.5, absolute = 0):
	Finds the point at a certain percentile of the range of your curve.  Used for certainty curves and generally assumes that the curve is decreasing
	INPUTS:
		z = array/ list of y-points
		sigma: array/ list of x-points
		threshold: the precentile you want
        absolute: 0 or 1: If 0, then sets theshold to a relative threshold (i.e., of threshold is 0.6, then setting absolute = 1 finds point in the curve where f(x) = 0.6*max(f) + (1-0.6) min(f).  If set to 1, finds x such that f(x) = 0.6
 
	OUTPUTS:
		[x-value of the threshold cutoff, max x-value, min x-value], max, min  

33. get_cutoff(filename, k, weights, mod, 
               image_path, label_path, args = ['lowpass', 0.0, 0.0, 0.0],
               jump = 10.0, thresh = 0.5, xeps = 0.0001, yeps = 0.0001, loops = 20, i=1, lowbound = -100000, highbound = 1000000):

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
 

ADDITIONAL FUNCTIONS (These are functions that I have been using throughout our research, but they were not used to create the contents of this paper)

PREDICTION BOX FUNCTIONS

13. iou(box1,box2):
	Calculates the iou (intersection over union) threshold for two bounding boxes
	INPUT:
	box1 and box2: two lists/arrays of the form [left, top, right, bottom]

	OUTPUT:
	a float giving the area of the intersection over the area of the union of box1 and box2
 

14. pr_area(x,y):
	calculates the area under the pr-curve.  Is calculated by adding up the areas of trapezoids formed by the lines formed between points and the boundaries from the x values.

	INPUTS:
	x: x values of curve, in array or list form
	y: y values of curve, in array or list form

	OUTPUT:
	total_area: a float value

15. get_pr_values(y_true, y_predict, t, match_thresh = 0.7): 

	breaks down predictions and ground truths into true positive, false positive, and false negative categories, according to bounding box predictions.

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

16. pr_curve(y_true,y_predict, match_thresh = 0.7):
    calculates the precision recall curve over increments of 0.02 for a single image

	INPUTS:
		y_true: a list or array of shape (n,4) with n being the number of cars, and each row consisting of the left,top,right,bottom coordinates.
		y_predict: list/ array of size (m,4) with predictions of cars
		match_thresh: iou threshold, float number between 0 and 1, default is 0.7

	OUTPUT:
		[x,y], where x is a list the true positive rate (true positive/ truth), and y is a list of the false positive rate (true positive/ prediction).  the values range over certainty thresholds from 0 to 1., but also include the theoretical cases for absolute 0 and absolute 1.
	

25. maximal_iou_match(y_true, y_predict,n): 
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


26. extract_false_positives(ground_truths, predictions, thresh=0):
    extracts the isolated false positives of an image detection.  These are the false positives for which there is little to no overlap.  This function was used in some older analyses and still will be used later, but it isn't used in this paper.

    INPUTS:
    ground_truths: the ground truth boxes of an image
    predictions: the predicted bounding boxes of an image
    threshold: a flot between 0 and 1 telling you the upper bound for the IOU threshold separating false from true positives

    OUTPUTS: an array of the coordinates of false positive cars


