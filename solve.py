import cv2
import numpy as np
import sys
from number_ocr import test_number

MIN_AREA = 300

def process_img(img_name):
	img =  cv2.imread(img_name)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = cv2.GaussianBlur(gray,(5,5),0)

	show_img('poop',gray)
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	thresh
	show_img('poop', thresh)
	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	show_img('after_contours', thresh)


	max_area = 0
	contour = None
	M = None

	# Determine which contour is the largest, aka the actual contour.
	for cnt in contours:
		moments = cv2.moments(cnt)
		area = moments['m00']
		if area > max_area:
			contour = cnt
			max_area = area
			M = moments

	peri = cv2.arcLength(contour,True)

	# approx is the 4 corners of the puzzle
	approx = cv2.approxPolyDP(contour,0.02*peri,True)

	cv2.drawContours(thresh,[approx],-1,(0,255,0),3)
	show_img('poop', thresh)

	approx = rectify(approx)
	h = np.array([ [0,0],[179,0],[179,179],[0,179] ],np.float32)

	retval = cv2.getPerspectiveTransform(approx,h)
	warp = cv2.warpPerspective(thresh,retval,(180,180))

	# Now we split the image to 81 cells, each 20x20 size
	cells = [np.hsplit(row,9) for row in np.vsplit(warp,9)]

	# Make it into a Numpy array. It size will be (50,100,20,20)
	x = np.array(cells)

	for i in range(9):
		for j in range(9):
			x[i,j] = remove_edges(x[i,j])
			test = x[i,j].reshape(-1,400).astype(np.float32) # Size = (2500,400)
			print x[i,j]
			area = sum(sum(x[i,j]))
			print 'Area: {}'.format(area)
			if area < MIN_AREA:
				label = [['BLANK']]
			else:
				label = test_number(test)

			print 'Label: {}'.format(label)
			show_img('puzzle', x[i,j])


	show_img('Puzzle', warp)

def rectify(h):
	h = h.reshape((4,2))
	hnew = np.zeros((4,2),dtype = np.float32)

	add = h.sum(1)
	hnew[0] = h[np.argmin(add)]
	hnew[2] = h[np.argmax(add)]
	 
	diff = np.diff(h,axis = 1)
	hnew[1] = h[np.argmin(diff)]
	hnew[3] = h[np.argmax(diff)]

	return hnew

def remove_edges(x):
	rows, cols = x.shape
	x[:3,:] = 0
	x[:,:3] = 0
	x[:,cols-4:] = 0
	x[rows-4:,:] = 0

	return x

def show_img(title, img):
	cv2.imshow(title,img)
	cv2.waitKey(0)

if __name__ == '__main__':
	if len(sys.argv) != 2: # Expect exactly one argument: the port number
		#usage()
		sys.exit(2)

	img_name = sys.argv[1]
	process_img(img_name)


