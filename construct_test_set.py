#!/usr/bin/env python

import numpy as np
import os

'''
Construct test cases with the following set random indices

Test set indexes for Iris-setosa: 20,34,30,28,32,26,0,5,4,15,24,45,19,33,47,46,13
Test set indexes for Iris-versicolor: 10,12,27,9,43,19,6,31,3,46,18,21,24,44,11,14,35
Test set indexes for Iris-virginica: 27,31,4,48,32,13,46,24,45,20,26,44,38,34,49,5,12

'''

test_indices = {"Iris-setosa" : [20,34,30,28,32,26,0,5,4,15,24,45,19,33,47,46,13],
"Iris-versicolor" : [10,12,27,9,43,19,6,31,3,46,18,21,24,44,11,14,35], 
"Iris-virginica" : [27,31,4,48,32,13,46,24,45,20,26,44,38,34,49,5,12]}

if __name__ == "__main__":
	FILE = open("Iris.txt", "r")
	data = []
	for line in FILE:
		data.append(line)

	ddata = {}
	classes = []

	for d in data[1:]:
		ddata[d.split('\t')[-1].replace('\n', '')] = []

	classes = ddata.keys()

	for d in data[1:]:
		d = d.split('\t')
		ddata[d[-1].replace('\n', '')].append([float(sd) for sd in d[:-1]])

	print ddata
	print "Keys:"
	print ddata.keys()
	print classes

	fdata = []
	train_set = []
	test_set = []
	for i in range(len(ddata)):
		fd = []
		tr_set = []
		ts_set = []
		for j in range(len(ddata[ddata.keys()[i]])):
			fd.append([i] + ddata[ddata.keys()[i]][j])
			if j in test_indices[ddata.keys()[i]]:
				ts_set.append([i] + ddata[ddata.keys()[i]][j])
			else:
				tr_set.append([i] + ddata[ddata.keys()[i]][j])
		fdata.append(fd)
		train_set.append(tr_set)
		test_set.append(ts_set)


   	#save this to file for later use
   	#first create directory Iris_Separated if it doesn't already exist
	if os.listdir('.').count('Iris_Separated') == 0:
		os.mkdir('Iris_Separated')

	#if train and test data directories don't exist creat them
	if os.listdir('./Iris_Separated/').count('test_data') == 0:
		os.mkdir('Iris_Separated/test_data')
	if os.listdir('./Iris_Separated/').count('train_data') == 0:
		os.mkdir('Iris_Separated/train_data')

	for i in range(len(fdata)):
		np.savetxt('./Iris_Separated/'+ddata.keys()[i]+'.txt', fdata[i], fmt='%.2f', delimiter='\t')
		np.savetxt('./Iris_Separated/train_data/'+ddata.keys()[i]+'.txt', train_set[i], fmt='%.2f', delimiter='\t')
		np.savetxt('./Iris_Separated/test_data/'+ddata.keys()[i]+'.txt', test_set[i], fmt='%.2f', delimiter='\t')

	#merging the separate data
	merged_ts = []
	merged_tr = []
	for i in range(len(fdata)):
		merged_ts += test_set[i]
		merged_tr += train_set[i]
	
	np.savetxt('./Iris_Separated/train_data/merged_train.txt', merged_tr, fmt='%.2f', delimiter='\t')
	np.savetxt('./Iris_Separated/test_data/merged_test.txt', merged_ts, fmt='%.2f', delimiter='\t')
	print merged_tr

