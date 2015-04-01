import csv
import nltk
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

trainname = 'train.csv'
testname = 'test.csv'
outputname = 'result.csv'

with open(trainname, 'rb') as trainfile:
	#spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	label = list()
	content = list()
	for row in trainfile:
		label.append(row[0])
		content.append(row[2:])
	label = label[1:]
	content = content[1:]


	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  

	vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
	

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(content)
	print "Vectorization finished"


	# Numpy arrays are easy to work with, so convert the result to an 
	# array
	contentVectors = train_data_features.toarray()

	clf = svm.SVC()
	clf.fit(contentVectors[:100],label[:100])
	print "Training finished"

	with open(testname, 'rb') as testfile:
		#spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		with open(outputname, 'wb') as outputfile:
			writer = csv.writer(outputfile, delimiter=',')
			writer.writerow(['Id','Category'])
			print "Start testing"
			for idx,row in enumerate(testfile):
				if idx%100==0:
					print idx
				if idx > 10:
					break
				newVector = vectorizer.transform([row[2:]]).toarray()
				print newVector.shape
				print clf.predict(newVector[0])
				#writer.writerow([str(row[0]),str(clf.decision_function(newVector))])
