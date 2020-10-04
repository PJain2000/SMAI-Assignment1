import numpy as np
import csv
import random
import math
from sklearn.metrics import accuracy_score

class KNNClassifier:
	k = 1

	def euclidean_distance(self, a, b):
	    dist = np.linalg.norm(a-b)
	    return dist

	def manhattan_distance(self, a, b):
	    n = a.shape
	    dist = 0
	    for i in range(0,n[0]):
	        dist += math.abs(a[i]-b[i])
	        
	    return dist

	def split_data(self, datafile):
		data = np.genfromtxt(datafile, delimiter=',').astype(int)

		a = data.shape
		n1 = int(0.8*a[0])
		n2 = a[1]

		np.random.shuffle(data)

		train_data = data[0:n1,:]
		validation_data = data[n1:a[0],:]

		x_train = train_data[:,1:n2]
		y_train = train_data[:,0]

		x_validation = validation_data[:,1:n2]
		y_validation = validation_data[:,0]

		return x_train, y_train, x_validation, y_validation

	def evaluate(self, validation_point, k, x_train, y_train):
	    distance = []

	    # print(k)
	    for train_point in x_train:
	        distance.append(self.euclidean_distance(train_point, validation_point))
	        
	    idx = np.argpartition(distance, k)
	    
	    y_predicted = []
	    
	    for i in range(0,k):
	        y_predicted.append(y_train[np.where(idx == i)])
	        
	    # ans = random.choice(y_predicted)
	    
	    d = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
	    
	    for i in y_predicted:
	        d[int(i)] += 1
	    
	    ans = max(d, key=d.get)
	        
	    return int(ans)

	def train(self, datafile):

		x_train, y_train, x_validation, y_validation = self.split_data(datafile)

		accuracy = []

		for k in range(1,3):
			# print(k)
			y_predicted = []
			for i in x_validation:
			    y_predicted.append(self.evaluate(i, k, x_train, y_train))

			accuracy.append(accuracy_score(y_validation, y_predicted))
			# print(accuracy)

		ind = accuracy.index(max(accuracy))

		self.k = ind+1

		# print(accuracy)
		print(self.k)
		print("Done training")

	def predict(self, datafile):
		data = np.genfromtxt(datafile, delimiter=',').astype(int)
		train_data = np.genfromtxt('./Datasets/q1/train.csv', delimiter=',').astype(int)

		x_train, y_train, x_validation, y_validation = self.split_data(train_data)

		y_predicted = []
		for i in data:
			y_predicted.append(self.evaluate(i, self.k, x_train, y_train))

		return y_predicted



