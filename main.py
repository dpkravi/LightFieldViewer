#  Deepak Ravindran
#  CS 6475 - Computational Photography
#  Georgia Institute of Technology

#  Final Project - Lightfield viewer 
#
#  Input images fot this project can be found at the 
#  The (New) Stanford Light Field Archive - http://lightfield.stanford.edu/lfs.html

import numpy as np
import skimage.io as skio
import skimage.transform
from skimage import img_as_float
import matplotlib.pyplot as plt
import sys
import math
import glob

# The folder containing the input images. Change the folder for the required input image set
inputFolder = "P:/LightField/chess"

# Currently the image sets available on the Stanford archive have a very specific naming scheme
# For example out_00_00_-859.738525_1022.898743.png
# in the pattern out_yy_xx_zzzzzzz, the xx and yy are the x and y positions on the light grid.
#Setting the grid size. Most image sets in the stanford archive are 17x17
gridX = 17
gridY = 17
#Size of the square patch that will be used for alignment
windowSize = 100
#Initial value of the radius for aperture
initAperture = 10
#Size of the search window
searchWindow = 20
#No of layers in gaussian pyramid. Program crashes on very high values. Memory requirement goes up drastically
gausslayer = 9

# Code

# Reads the images, computes the gaussian pyramid and stores it in memory - This is the cause of the poor efficiency of the system
def getGridPyramid():
	grid = []
	
	for y in xrange(gridY):
		print "Processing Images: %3d%% complete" % int(100*(y+1)/float(gridY))
		newRow = []
		for x in xrange(gridX):
			img = getGridImageXY(x,y)
			pyramid = gaussianPyramid(img)
			newRow.append(pyramid)
		grid.append(newRow)
		
	return np.array(grid)

#--------------------------------------------------------------------------------

# Returns a square patch (an image) cropped from img,
# centered in p with dimensions w by w
def cropSquarePatch(img,p,w):
	x = p[0]
	y = p[1]

	# Cases near the border or with too low resolution for w by w
	k = max(1,int(w/2))
	min_dim = min(img.shape[0],img.shape[1])
	k = min(k,min_dim/2)
	min_x = max(x-k,0)
	max_x = min(x+k,img.shape[1])
	min_y = max(y-k,0)
	max_y = min(y+k,img.shape[0])

	return img[min_y:max_y,min_x:max_x]

# Returns the image located at position (x,y) in the grid
def getGridImageXY(x,y):

	#Rounding off the float values to get approximately correct image
	x = int(x)
	y = int(y)

	# Creating filename string for that particular folder with format out_yy_xx_zzzzzzz
	fileNameString = "%s/out_%02d_%02d_*" % (inputFolder,y,x)
	#glob is a filename pattern matching library
	results = glob.glob(fileNameString)
	num_results = len(results)

	if num_results == 0:
		sys.exit("Error: Image Not Found")
	elif num_results > 1:
		sys.exit("Error: Multiple Images Found")
	else:
		img = skio.imread(results[0])
		return img_as_float(img)


# Return a list that contains the layers of the gaussian pyramid of image
def gaussianPyramid(image):
	return list(skimage.transform.pyramid_gaussian(image,gausslayer=gausslayer))


# Translates img by dx in x and dy in y
def translateImage(img,dx,dy):
	result = np.roll(img, dx, axis = 1)
	result = np.roll(result, dy, axis = 0)
	return result

# Compute the SSD between image1 and image2
def ssd(image1, image2):
	return np.sum(((image1 - image2)**2))

def alignByPyramids(referencePatch,py_target,point):
	len_py_ref = len(referencePatch)
	len_py_target = len(py_target)
	if len_py_ref != len_py_target:
		sys.exit("Error: Pyramids do not have the same size.")

	bestDX, bestDY = 0, 0
	for ref,target,level_number in reversed(zip(referencePatch,py_target,range(len_py_ref))):
		bestDX *= 2
		bestDY *= 2

		ptx = point[0]/(2**level_number) + bestDX
		pty = point[1]/(2**level_number) + bestDY

		adjusted_target = translateImage(target,bestDX,bestDY)

		w = windowSize
		ref_patch = cropSquarePatch(ref,(ptx,pty),w)
		target_patch = cropSquarePatch(adjusted_target,(ptx,pty),w)

		new_dx, new_dy = bestAlignment(ref_patch,target_patch)

		bestDX += new_dx
		bestDY += new_dy

	return (bestDX,bestDY)

def averageAlignedGrid(grid,dx,dy,center,radius,level):
	average = np.zeros(grid[0,0][level].shape)

	dx = dx/(2**level)
	dy = dy/(2**level)

	images_used = gridY*gridX

	for y in xrange(gridY):
		for x in xrange(gridX):
			# Do rejection sampling to discard images outside the radius
			if L1Dist((x,y),center) > radius:
				images_used -= 1
				continue

			# Get image, align, average
			img = grid[y,x][level]
			trans_x = int(-x*dx)/(2**level)
			trans_y = int(-y*dy)/(2**level)
			img_aligned = translateImage(img,trans_x,trans_y)
			average += img_aligned
	average /= images_used
	return average

	
def bestAlignment(ref, target):
	best_value = float("inf")
	
	bestDX = 0
	bestDY = 0

	width_search_window = min(searchWindow,ref.shape[0]/2)
	height_search_window = min(searchWindow,ref.shape[1]/2)

	# Searches in the displacement search window
	for dx in reversed(xrange(-width_search_window,width_search_window)):
		for dy in reversed(xrange(-height_search_window,height_search_window)):
		
		
			traslated_target = translateImage(target,dx,dy)
			dist = ssd(ref,traslated_target)
			# Get the minimum distances
			if dist < best_value:
				best_value = dist
				bestDX = dx
				bestDY = dy

	return (bestDX, bestDY)
	
# Returns the manhattan/L1 distance between two points
def L1Dist(a,b):
	return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Parameters to be maintened during out application
radius = initAperture
dx = 0.0
dy = 0.0
axis = None
referencePatch = None
referencePatchEnd = None
grid = None
center = None
image = None

# User event handling. Handles the mouse click and the mouse scroll
def userEventHandle(event):

	global center
	global image
	global radius
	global dx
	global dy
	global axis
	
	# These two variables are the reference patch locations. The top left and bottom right corner of the grid
	global referencePatch
	global referencePatchEnd
	
	global grid

	update = False

	if event.button == 1:
		if event.inaxes == axis:
			point = (int(event.xdata),int(event.ydata))
			delta_x,delta_y = alignByPyramids(referencePatch,referencePatchEnd,point)

			dx = float(-delta_x)/float(gridX)
			dy = float(-delta_y)/float(gridY)

		update = True
	elif event.button == 'up':
		if event.inaxes == axis:
			# Decreasing aperture radius
			radius += event.step
			if radius < 1:
				radius = 1
			update = True

	elif event.button == 'down':
		if event.inaxes == axis:
			# Increasing aperture radius
			radius += event.step
			if radius < 1:
				radius = 1
			update = True
	
	if update:
		# Drawing in levels to give immediate feedback to user and hide the loading time
		for level in reversed(xrange(len(grid[0,0]))):
			avg = averageAlignedGrid(grid,dx,dy,center,radius,level)
			image.set_data(avg)
			plt.draw()

def run():

	global center
	global image
	global radius
	global dx
	global dy
	global axis
	
	# These two variables are the reference patch locations. The top left and bottom right corner of the grid
	global referencePatch
	global referencePatchEnd
	
	global grid

	grid = getGridPyramid()

	# Setting the reference images to the top left corner and bottom right corner
	referencePatch = grid[0,0]
	radius = initAperture
	center = (gridX/2,gridY/2)
	referencePatchEnd = grid[(gridY-1),(gridX-1)]

	point = (0,0)
	delta_x,delta_y = alignByPyramids(referencePatch,referencePatchEnd,point)
	dx = float(-delta_x)/float(gridX)
	dy = float(-delta_y)/float(gridY)

	# Find the averaged image. This will make parts of the image look blurry and parts of it look sharp and in focus
	avg = averageAlignedGrid(grid,dx,dy,center,radius,0)

	# Create the matplotlib plot for the application. This will be the UI that the user interacts with
	figure, axis = plt.subplots(ncols=1)
	image = axis.imshow(avg, vmin=0, vmax=1)

	# Setting the mouse event handlers
	listener_button_press = figure.canvas.mpl_connect('button_press_event', userEventHandle)
	listener_scroll = figure.canvas.mpl_connect('scroll_event', userEventHandle)

	plt.show()
	plt.draw()
	figure.canvas.mpl_disconnect(listener_button_press)
	figure.canvas.mpl_disconnect(listener_scroll)
	plt.close()

if __name__ == '__main__':

	run()

	print "------------- Running application -------------"


