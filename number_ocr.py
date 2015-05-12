import numpy as np
import cv2
from matplotlib import pyplot as plt

def train():
	img = cv2.imread('digits.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Now we split the image to 5000 cells, each 20x20 size
	cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]

	# Make it into a Numpy array. It size will be (50,100,20,20)
	x = np.array(cells)
	print x

	# Now we prepare train_data and test_data.
	train = x[:,:].reshape(-1,400).astype(np.float32) # Size = (5000,400)
	#test = x[0,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)

	# Create labels for train and test data
	k = np.arange(10)
	train_labels = np.repeat(k,500)[:,np.newaxis]

	'''
	for i in range(50):
		for j in range(100):
			test = x[i,j]
			print test
			cv2.imshow('img',test)
			cv2.waitKey(0)	
	'''
	#test_labels = train_labels.copy()

	#cv2.imshow('img',img)
	#cv2.waitKey(0)	

	#test_number(test)

	# Initiate kNN, train the data, then test it with test data for k=1
	knn = cv2.KNearest()
	knn.train(train,train_labels)

	file_name = 'knn_data.npz'
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
	

def test_number(test_data):
	# Now load the data
	with np.load('knn_data.npz') as data:
		#print data.files
		train = data['train']
		train_labels = data['train_labels']

		knn = cv2.KNearest()
		knn.train(train,train_labels)
		ret,result,neighbours,dist = knn.find_nearest(test_data,k=2)
		return result


if __name__ == '__main__':
	train()
	#test_number()