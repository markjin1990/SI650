import csv
import glob
import os
import numpy as np

def find_majority(k):
	myMap = {}
	maximum = ( '', 0 ) # (occurring element, occurrences)
	for n in k:
		if n in myMap: myMap[n] += 1
		else: myMap[n] = 1

		# Keep track of maximum on the go
		if myMap[n] > maximum[1]: maximum = (n,myMap[n])
	return maximum[0]

os.chdir(".")
labelList = []
for csvfile in glob.glob("*_result.csv"):
	newList = []
	with open(csvfile, 'rb') as resultfile:
		spamreader = csv.reader(resultfile, delimiter=',')
		for idx,row in enumerate(spamreader):
			if idx==0: 
				continue
			newList.append(str(row[1]))
		labelList.append(newList)



labelList = np.array(labelList).T.tolist()

finalresult = []
for row in labelList:
	finalresult.append(find_majority(row))

with open('finalresult.csv', 'wb') as outputfile:
	writer = csv.writer(outputfile, delimiter=',')
	writer.writerow(['Id','Category'])
	length = len(finalresult)
	for idx in range(0,length):
		writer.writerow([idx+1,finalresult[idx]])
