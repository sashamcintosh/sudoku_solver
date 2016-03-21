#################################
# 
# Sasha McIntosh - sam2270
# Michael Saltzman - mjs2287
# 
# Visual Interfaces to Computers
# Final Project
# 
# solve.py
# 
#################################

import cv2
import numpy as np
import sys
from number_ocr import test_number
from solutions import PUZZLES
import sudoku_solver as solver


MIN_AREA = 500

def process_img(img_name, model, k, DEMO):
	img =  cv2.imread(img_name)
	if DEMO: show_img('Original Image',img)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(5,5),0)
	if DEMO: show_img('Grayscale with Gaussian Blur',gray)
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	if DEMO: show_img('Adaptive Thresholding', thresh)
	contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#show_img('after_contours', thresh)

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
	#print approx

	img1 = img.copy()
	cv2.drawContours(img1,[approx],-1,(0,255,0),3)
	
	if DEMO: show_img('Boundary of Sudoku', img1)

	approx = order_corners(approx)
	corners = approx
	h = np.array([ [0,0],[179,0],[179,179],[0,179] ],np.float32)
	#h = np.array([ [0,0],[145*9,0],[145*9,128*9],[0,128*9] ],np.float32)

	retval = cv2.getPerspectiveTransform(approx,h)
	warp = cv2.warpPerspective(gray,retval,(180,180))#(145*9,128*9)) #,(180,180))
	warp_thresh = cv2.warpPerspective(thresh,retval,(180,180))#(145*9,128*9)) #,(180,180))
	
	if DEMO: show_img('Perspective Warp',warp_thresh)

	h = np.array([ [0,0],[539,0],[539,539],[0,539] ],np.float32)
	retval = cv2.getPerspectiveTransform(approx,h)
	warp_orig = cv2.warpPerspective(img.copy(),retval,(540,540))#(145*9,128*9)) #,(180,180))

	# Now we split the image to 81 cells, each 20x20 size
	cells = [np.hsplit(row,9) for row in np.vsplit(warp,9)]
	cells_thresh = [np.hsplit(row,9) for row in np.vsplit(warp_thresh,9)]

	# Make it into a Numpy array. It size will be (50,100,20,20)
	p = np.array(cells)
	p_thresh = np.array(cells_thresh)
	p_new = np.zeros([9,9,20,20])
	labels = []

	for i in range(9):
		for j in range(9):
			#p_thresh[i,j] = remove_edges(p_thresh[i,j])
			contours, hierarchy = cv2.findContours(p_thresh[i,j].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			area = sum(sum(remove_edges(p_thresh[i,j].copy())))

			if (len(contours) == 0) or area < MIN_AREA:
				label = 0
				warp_test = p_thresh[i,j]
				p_new[i,j] = warp_test
			else:
				contour = contours[0]
				if cv2.contourArea(contour) < 5.0:
					approx = np.array([ [0,0],[19,0],[19,19],[0,19] ],np.float32)

				else:
					x,y,w,h = cv2.boundingRect(contour)
					approx = np.array([ [y,x],[y+h,x],[y+h,x+w],[y,x+w] ],np.float32)

				new_corners = np.array([ [0,0],[19,0],[19,19],[0,19] ],np.float32)

				retval = cv2.getPerspectiveTransform(approx,new_corners)
				warp_test = cv2.warpPerspective(p_thresh[i,j],retval,(20,20))#(145*9,128*9)) #,(180,180))
				
				#p_new[i,j] = warp_test
				test = p_thresh[i,j].reshape(-1,400).astype(np.float32)

				label = test_number(model,test, k).tolist()[0]
				label = int(label[0])

			labels.append(label)
			#print 'Area: {}'.format(area)
			#print 'Label: {}'.format(label)
			#show_img('puzzle', p_thresh[i,j])

	#labels = test_number(model, p_new.reshape(-1,400).astype(np.float32), k)
	return warp_orig, corners, labels
	#show_img('Puzzle', warp)

def order_corners(h):
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
	x[:2,:] = 0
	x[:,:2] = 0
	x[:,cols-2:] = 0
	x[rows-2:,:] = 0

	return x

def show_img(title, img):
	cv2.imshow(title,img)
	cv2.waitKey(0)

def draw_solution(img, corners, puzzle, solution):
	#print corners
	#w = corners[1][0] - corners[0][0]
	#h = corners[3][1] - corners[0][1]
	#print puzzle
	#print solution
	for i in range(9):
		for j in range(9):
			if puzzle[i, j] == 0:
				num = str(solution[i,j])
				org = (j*60+15, (i+1)*60-10)
				#print org
				cv2.putText(img, num, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
				#show_img('poop', img)

if __name__ == '__main__':
	img_name = sys.argv[1]

	if len(sys.argv) == 2: # Expect exactly one argument: the image
		DEMO = True
		img, corners, labels = process_img(img_name,'knn_typed.npz', 1, DEMO)

		ext_puzzle = solver.stringToArray(''.join(map(str, labels)))
		print 'Extracted Sudoku:\n{}'.format(ext_puzzle)
		puzzle = solver.stringToArray(''.join(map(str, PUZZLES[img_name])))
		print 'Actual Sudoku:\n{}'.format(puzzle)

		matches = np.array(PUZZLES[img_name])==np.array(labels)
		correct = np.count_nonzero(matches)
		accuracy = correct*100.0/len(labels)
		print 'Digit Recognition Accuracy: {}'.format(accuracy)

		solution = solver.sudoku(puzzle.copy())
		print 'Solution:\n{}'.format(solution)

		draw_solution(img, corners, puzzle, solution)
		show_img('Solution', img)

	else:
		for k in range(1,21):
			labels = process_img(img_name,'knn_typed.npz', k, False)
			#print labels
			matches = np.array(PUZZLES[img_name])==np.array(labels)
			correct = np.count_nonzero(matches)
			accuracy = correct*100.0/len(labels)
			print '{}: {}'.format(k, accuracy)


