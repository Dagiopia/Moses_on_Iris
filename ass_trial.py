#!/usr/bin/env python 

from pymoses import moses
import numpy as np


NO_BINS = 3

def binization(data):
	np.savetxt('iris_parsed.csv', data, fmt='%.3f', delimiter=',')
	width = len(data[0]) - 1
	maxs = []
	dt = np.transpose(data)
	for i in range(width):
		maxs.append(max(dt[i]))
		
	#maxs is right - verified
	#print "Max of each category:"
	#print maxs
	
	bin_dic = []
	for j in range(width):
		t_dic = []
		for i in range(NO_BINS):
			t_dic.append((maxs[j] / float(NO_BINS)) * (i+1))
		bin_dic.append(t_dic)
	
	print bin_dic
	print bin_dic[0]
	
	#start binization
	n_data = []
	for d in data:
		for i in range(width):
			bin_no = 0
			for b in range(len(bin_dic[i])):
				if bin_dic[i][b] > d[i]:
					break
				bin_no += 1
			d[i] = bin_no
		n_data.append(d)
	#print n_data
	np.savetxt('binized_iris.csv', n_data, fmt='%d', delimiter=',')
	return n_data



f = open('Iris.txt', 'r')


data = []
f_data = []

col_names = f.readline()

for line in f:
	data.append(line)
	
	
d_n = {}

for d in data:
	f_data.append(d.split('\t')) #get rid of \t in text
	f_data[-1][-1] = f_data[-1][-1].replace('\n', '') #remove the trailing \n
	d_n[f_data[-1][-1]] = len(d_n)	#gather keys
	f_data[-1][-1] = d_n[f_data[-1][-1]]  #replace names with keys
	f_data[-1] = [float(j) for j in f_data[-1]]
	f_data[-1][-1] = int(f_data[-1][-1])

#this is an idiotic hack 
#when keying the flower names into a dictionary, I used the size of the dictionary
#as value and the problem is that the first time a certain flower is seen,
#the size of the dictionary is 0 and 0 is set but the next time the same flower is seen
#the size has increased to one and so the next value is 1 for the same flower
#so here I add 1 to the flower key of the first observation of the flower in the data
#GET RID OF THIS SHIT!!!!!!!!!
f_data[0][-1] += 1
	
print data[0]
print f_data[0]
print d_n


f_data = binization(f_data)

print f_data[0:4]
print f_data[70:74]
print f_data[125:129]

f_data = [[1, 2, 0, 0, 1], [1, 2, 0, 0, 1], [1, 2, 0, 0, 1], [1, 2, 0, 0, 1], [2, 2, 2, 2, 2], [2, 1, 1, 1, 2], [2, 1, 2, 1, 2], [2, 1, 2, 1, 2], [2, 2, 2, 2, 3], [2, 1, 2, 2, 3], [2, 2, 2, 2, 3], [2, 1, 2, 2, 3]]

f_data = [[0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [10, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [0, 10, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1], [10, 1, 1, 1, 1], [0, 1, 1, 1, 1]]


m = moses()
output = m.run(input=f_data, python=True)

#output has 'eval', 'program', 'program_type', 'score'

print output[0].program
print "\n"
print d_n

