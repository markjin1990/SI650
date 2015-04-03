import csv
import nltk
from sklearn import svm
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import ExtraTreesClassifier
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification

from sklearn.feature_selection import RFE

from sklearn.linear_model import SGDClassifier

from textblob import TextBlob

from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import linear_model

from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import RandomizedLogisticRegression

class LemmaTokenizer(object):
	def __init__(self):
		self.wnl = WordNetLemmatizer()
	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


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

	# Data clean
	# Spelling correction
	#for idx,sentence in enumerate(content): 
	#	if idx % 400 == 0:
	#		print "Spelling check "+str(idx)
	#	b = TextBlob(sentence)
	#	content[idx] = b.correct()

	# Initialize the "CountVectorizer" object, which is scikit-learn's
	# bag of words tool.  

	# Vectorizer: count or tfidf? We have proved that tfidf is better. Lemetizing is not helping.
	#vectorizer = CountVectorizer(analyzer = "word",  preprocessor = None, stop_words ='english', lowercase=True,encoding='ascii') 
	vectorizer = TfidfVectorizer(analyzer = "word",  preprocessor = None, stop_words ='english', lowercase=True,encoding='ascii') 

	#vectorizer = CountVectorizer(analyzer = "word",stop_words ='english', max_features = 80)

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of 
	# strings.
	contentVectors = vectorizer.fit_transform(content).toarray()
	#train_data_features = vectorizer.transform(content)
	print "Vectorization finished"

	#clf = svm.LinearSVC(C=1)
	
	# feature selection: variance
	#sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
	#contentVectors = sel.fit_transform(contentVectors)
	#print len(contentVectors[0])

	# feature selection: Tree classifier importance
	#clf = ExtraTreesClassifier()
	#selector = clf.fit(contentVectors , label)
	#contentVectors = selector.transform(contentVectors)

	# feature selection: SGDClassifie importance
	#contentVectors = SGDClassifier(loss="hinge", penalty="l1").fit_transform(contentVectors,label)

	# feature selection: SVM importance
	#selector =  svm.LinearSVC(C=1, penalty="l1", dual=False).fit(contentVectors,label)
	#contentVectors = selector.transform(contentVectors)

	selector =  RandomizedLogisticRegression().fit(contentVectors,label)
	contentVectors = selector.transform(contentVectors)

	# LARS feature selection
	
	#l1-based feature selection
	#contentVectors = SGDClassifier(loss="hinge", penalty="l1").fit_transform(contentVectors,label)
	#contentVectors =  svm.LinearSVC(C=1, penalty="l1", dual=False).fit_transform(contentVectors,label)
	
	#clf = svm.LinearSVC(C=1)
	#clf = SGDClassifier(loss="hinge", penalty="l1")
	print "Feature selection finished"

	# Cross validation
	#rfecv = RFECV(estimator=clf, step=2, cv=StratifiedKFold(label, 2),scoring='accuracy')
	#selector = rfecv.fit(contentVectors, label)
	#contentVectors = selector.transform(contentVectors)


	# RFE
	#rfe = RFE(estimator=clf, n_features_to_select=5000, step=10)
	#selector = rfe.fit(contentVectors[:3000], label[:3000])
	#contentVectors = selector.transform(contentVectors)
	print "Feature selection finished"

	#clf = DummyClassifier()
	#clf = RandomForestClassifier()

	clf = SGDClassifier(loss="hinge", penalty="l1")
	#clf = OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))
	#clf = svm.SVC(kernel='rbf', C=1e3, gamma=0.1)
	#clf = svm.SVC(kernel='linear', C=1e3)

	#clf.fit(np.array(contentVectors[:3000]),np.array(label[:3000]))
	clf.fit(np.array(contentVectors),np.array(label))

	predicted_result = clf.predict(np.array(contentVectors[3001:]))
	print precision(predicted_result,label[3001:])

	print "Training finished"
'''
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
	newVector = selector.transform(newVector)
	predicted_label = clf.predict(np.array(newVector))
	print "Labeling finished"

	with open(outputname, 'wb') as outputfile:
		writer = csv.writer(outputfile, delimiter=',')
		writer.writerow(['Id','Category'])
		length = len(finaltest)
		for idx in range(0,length):
			writer.writerow([idx+1,predicted_label[idx]])
'''