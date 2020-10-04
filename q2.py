import numpy as np
import csv
import random
import math
from sklearn.metrics import accuracy_score

class KNNClassifier:
	k = 1

	def hamming_distance(self, a, b):
	    n = a.shape
	    dist = 0
	    for i in range(0,n[0]):
	        if a[i] != b[i]:
	            dist += 1
	        
	    return dist

	def fill_missing_data(self, datafile):
		data = np.genfromtxt(datafile, delimiter=',', dtype=str)

		d = {'b':0, 'c':0, 'u':0, 'e':0, 'z':0, 'r':0}
		for row in data:
		    if row[11] != '?':
		        d[row[11]] += 1

		mode = max(d, key=d.get)

		for row in data:
		    if row[11] == '?':
		        d[row[11]] = mode

		return data

	def split_data(self, data):
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

	    for train_point in x_train:
	        distance.append(self.hamming_distance(train_point, validation_point))
	        
	    idx = np.argpartition(distance, k)
	    y_predicted = []
	    
	    for i in range(0,k):
	        y_predicted.append(y_train[np.where(idx == i)])

	#     ans = random.choice(y_predicted)
	    
	    d = {'e': 0, 'p': 0}
	    
	    for i in y_predicted:
	        d[i[0]] += 1
	    
	    ans = max(d, key=d.get)
	        
	    return ans

	def train(self, datafile):

		data = self.fill_missing_data(datafile)

		x_train, y_train, x_validation, y_validation = self.split_data(data)

		accuracy = []
		for k in range(1,3):
			y_predicted = []
			for i in x_validation:
			    y_predicted.append(self.evaluate(i, k, x_train, y_train))

			accuracy.append(accuracy_score(y_validation, y_predicted))
			# print(accuracy)

		ind = accuracy.index(max(accuracy))

		self.k = ind+1
		print(self.k)
		print("Done training")

	def predict(self, datafile):
		data = np.genfromtxt(datafile, delimiter=',', dtype=str)
		train_data = np.genfromtxt('./Datasets/q2/train.csv', delimiter=',', dtype=str)

		x_train, y_train, x_validation, y_validation = self.split_data(train_data)

		y_predicted = []
		for i in data:
			y_predicted.append(self.evaluate(i, self.k, x_train, y_train))

		return y_predicted


