import csv
import nltk
from sklearn import svm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

trainname = 'train.csv'
testname = 'test.csv'
outputname = 'result.csv'

finaltest = []

def precision(a,b):
	if len(a)!=len(b):
		return 

	mylen = len(a)
	count = 0
	for i in range(0,mylen):
		if a[i] == b[i]:
			count += 1

	return count/float(mylen)

with open(trainname, 'rb') as trainfile:
	#spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	label = list()
	content = list()
	for row in trainfile:
		idx = row.index(',')
		label.append(row[:idx])
		content.append(row[idx+1:])
	label = map(int, label[1:])
	content = content[1:]


	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  

	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words ='english', max_features = 60,lowercase=True,encoding='ascii') 
	#vectorizer = CountVectorizer(analyzer = "word",stop_words ='english', max_features = 80)

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	contentVectors = vectorizer.fit_transform(content).toarray()
	#train_data_features = vectorizer.transform(content)
	print "Vectorization finished"


	#clf = svm.SVC(kernel='rbf', C=1e3, gamma=0.1)
	#clf = svm.SVC(kernel='linear', C=1e3)
	clf = svm.LinearSVC(C=1.0)
	clf.fit(np.array(contentVectors),np.array(label))

	'''
	predicted_result = clf.predict(np.array(contentVectors[1001:]))
	print predicted_result
	print precision(predicted_result,label[1001:])
	'''




	print "Training finished"

	with open(testname, 'rb') as testfile:
		#spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		
			print "Start testing"
			for idx,row in enumerate(testfile):
				if idx==0:
					continue
				commaidx = row.index(',')
				testidx=row[:commaidx]
				testcontent = row[commaidx+1:]
				finaltest.append(testcontent)

	print "Labeling starteded"
	newVector = vectorizer.transform(finaltest).toarray()
	predicted_label = clf.predict(np.array(newVector))
	print "Labeling finished"

	with open(outputname, 'wb') as outputfile:
		writer = csv.writer(outputfile, delimiter=',')
		writer.writerow(['Id','Category'])
		length = len(finaltest)
		for idx in range(0,length):
			writer.writerow([idx+1,predicted_label[idx]])
