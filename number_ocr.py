import numpy as np
import cv2
from matplotlib import pyplot as plt

def train():
	img = cv2.imread('training/digits.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,127,255,0)

	# Now we split the image to 5000 cells, each 20x20 size
	cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
	cells_thresh = [np.hsplit(row,100) for row in np.vsplit(thresh,50)]

	# Make it into a Numpy array. It size will be (50,100,20,20)
	p = np.array(cells)[5:, :]
	p_thresh = np.array(cells_thresh)[5:, :]
	p_new = np.zeros([45,100,20,20])

	# Create labels for train and test data
	k = np.arange(1,10)
	train_labels = np.repeat(k,500)[:,np.newaxis]
	#print train_labels

	
	for i in range(45):
		for j in range(100):
			test = p_thresh[i,j]

			contours, hierarchy = cv2.findContours(p_thresh[i,j].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			'''
			max_area = 0
			contour = None

			# Determine which contour is the largest, aka the actual contour.
			for cnt in contours:
				moments = cv2.moments(cnt)
				area = moments['m00']
				if area > max_area:
					contour = cnt
					max_area = area
					M = moments
			'''
			contour = contours[0]

			x,y,w,h = cv2.boundingRect(contour)

			new_corners = np.array([ [0,0],[19,0],[19,19],[0,19] ],np.float32)
			approx = np.array([ [y,x],[y+h,x],[y+h,x+w],[y,x+w] ],np.float32)

			retval = cv2.getPerspectiveTransform(approx,new_corners)
			warp = cv2.warpPerspective(test,retval,(20,20))#(145*9,128*9)) #,(180,180))

			p_new[i,j] = warp
			#print x,y,w,h
			#cv2.rectangle(test,(x,y),(x+w,y+h),255,1)

			#print test
			#cv2.imshow('warp',warp)
			#cv2.waitKey(0)	
	
	#test_labels = train_labels.copy()

	#cv2.imshow('img',img)
	#cv2.waitKey(0)	

	#test_number(test)

	# Now we prepare train_data and test_data.
	train = p_thresh[:,:].reshape(-1,400).astype(np.float32) # Size = (5000,400)
	#test = x[0,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.KNearest()
	knn.train(train,train_labels)

	file_name = 'knn_handwritten.npz'
	'''
	ret,result,neighbours,dist = knn.find_nearest(test,k=5)

	# Now we check the accuracy of classification
	# For that, compare the result with test_labels and check which are wrong
	matches = result==test_labels
	correct = np.count_nonzero(matches)
	accuracy = correct*100.0/result.size
	print accuracy
	'''
	# save the data
	np.savez(file_name,train=train, train_labels=train_labels)

	print 'Model saved in {}'.format(file_name)
	
def train_typed():
	img = cv2.imread('training/typed.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV)

	# Now we split the image to 5000 cells, each 128x145 size
	cells = [np.hsplit(row,10) for row in np.vsplit(thresh,11)]

	# Make it into a Numpy array. It size will be (11,10,145,128)
	x = np.array(cells)
	x_new = np.zeros([11,10,20,20])

	
	for i in range(11):
		for j in range(10):
			test = x[i,j]
			#print test
			new_corners = np.array([ [0,0],[19,0],[19,19],[0,19] ],np.float32)
			approx = np.array([ [0,0],[144,0],[144,127],[0,127] ],np.float32)

			retval = cv2.getPerspectiveTransform(approx,new_corners)
			warp = cv2.warpPerspective(test,retval,(20,20))
			x_new[i,j] = warp
			cv2.imshow('img',warp)
			cv2.waitKey(0)	
	

	# Now we prepare train_data and test_data.
	train = x_new[:,1:].reshape(-1,400).astype(np.float32) # Size = (5000,400)
	#test = x[0,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	# Create labels for train and test data
	train_labels = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9]] * 11)

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.KNearest()
	knn.train(train,train_labels)

	file_name = 'knn_typed.npz'

	# save the data
	np.savez(file_name,train=train, train_labels=train_labels)

	print 'Model saved in {}'.format(file_name)

def test_number(model, test_data, k):
	# Now load the data
	with np.load(model) as data:
		#print data.files
		train = data['train']
		train_labels = data['train_labels']

		knn = cv2.KNearest()
		knn.train(train,train_labels)
		ret,result,neighbours,dist = knn.find_nearest(test_data,k=k)
		return result


if __name__ == '__main__':
	train()
	train_typed()
	#test_number()