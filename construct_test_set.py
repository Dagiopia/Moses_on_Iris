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

NO_BINS = 4

def binization(data):
	'''
	How the binerization works
		First, the maximum of all features is collected into a list maxs[].
		To collect the maximums, the data which is col=features and row=observations
		is transposed and by iterating over the values of each feature the max
		in that array is appended to the list maxs[]

		After, maxs contains [x, y, z, ...] where x is the max value of the first 
		feature, y is the max value of the second feature and so on so on

		Second, construct a an array for each feature with the number of bins 
		required.
		To do this, a top loop iterated over the number of features and a second 
		loop over the number of bins to create. During the first bin, the range is 
		assumed to start from zero to (max/BIN_NO) * bin_position
		That is, if max is 4 and BIN_NO is 3 then the first bin is in the range of 
		values from 0 to (4/3)*1 = 1.3333 then the second bin is in the range of
		values from 1.3333 to (4/3)*2 = 2.666 and the third and last is in the range
		of values from 2.666 to (4/3)*3 = 4. Thus the array of a certain feature that
		contains the range of values of the bins would be [1.3333, 2.6666, 4] 

		Obviously this method is yelele bad when it comes to features that may 
		contain negative numbers. To deal with that the above can be modified to 
		consider the min value of each feature like the following.

		while collecting the maxs in the first step, it would be necessary to collect
		the mins too. After, The first loop would iterate over the number of feaures
		as in the first method and the second would iterate over the number of bins
		required plus one(in order to consider the initial negative value which was
		assumed to be 0 in the first way). Additionally, diff MAX and MIN to get the 
		span. 
		Thus, for a bin number of 3, it would loop 4 times like the follwing. 
		MIN+((MAX-MIN)/BIN_NO)*ITER
		If max is 4, min is -3 and BIN_NO is 3 then the first value of the ranges
		array would be (-3+((4 - -3)/3)*0) = -3 then the second value would be
		(-3+((4 - -3)/3)*1) = -0.66666 the third would be 1.666666 and the last one
		would be 4 and so the array would contain the value
		[-3.0, -0.6666666666666665, 1.666666666666667, 4.0]

		since this second method works for whatever min value (even 0) it is used.
		
		come to think of it's actually better because now the range of each bin is
		reduced or has more resolution.
		For example for min = 1, max = 4 and BIN_NO = 3

		FIRST METHOD BINS:
			bin1 = (0 - 1.3333)
			bin2 = (1.3333 - 2.66666)
			bin3 = (2.6666 - 4)

		SECOND METHOD BINS:
			bin1 = (1 - 2)
			bin2 = (2 -3)
			bin3 = (3 -4)
		
		so each range with the first method is 1.333 where the second is 1.

		After this, the range of values for binization of each feature are stored
		in 2D array.

		Third,
	'''


	np.savetxt('iris_parsed.csv', data, fmt='%.3f', delimiter=',')
	width = len(data[0]) - 1
	maxs = []
	mins = []
	dt = np.transpose(data)
	for i in range(1, width+1):
		maxs.append(max(dt[i]))
		mins.append(min(dt[i]))
	#maxs is right - verified
	#print "Max of each category:"
	#print maxs

	bin_dic = []
	for j in range(width):
		t_dic = []
		for i in range(NO_BINS+1):
			t_dic.append(mins[j] + ((maxs[j] - mins[j]) / float(NO_BINS))*i)
		bin_dic.append(t_dic)

	print bin_dic
	print bin_dic[0]

	#start binization
	n_data = []
	for d in data:
		for i in range(1, width+1):
			bin_no = 0
			for b in range(NO_BINS):
				#print "if %f is between %f and %f then BIN=%d" % (d[i], bin_dic[i-1][b], bin_dic[i-1][b+1], bin_no)
				if d[i] <= bin_dic[i-1][b+1]:
					break
				bin_no += 1
			d[i] = bin_no
		n_data.append(d)
	print n_data
	np.savetxt('binized_iris.csv', n_data, fmt='%d', delimiter=',')
	return n_data







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

	#save the separated data to file
	#all data for a certain class of flower in Iris_Separated
	#training and test data of each class in train_data and test_data 
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

	# for i in range(len(merged_tr)):
	# 	merged_tr[i][0] += 2

	binization(merged_tr)