import nltk
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import math
import numpy as np



def readText(fn):
	f = open(fn, "r")		
	text = f.read()
	f.close()
	return text


stoptext = readText("stoplist.txt")
stopword = stoptext.split("\r\n")

def removeNonAlphaNumeric(tokenList):
	for idx,m in enumerate(tokenList):
		if not m.isalnum():
			tokenList.pop(idx)
	return tokenList

def checkIfStopWord(word):
	if word in stopword:
		return True
	else:
		return False

def numCapital(word):
	n = 0
	for letter in word:
		if letter.isupper():
			n += 1
	return n

def getProportion(mylist):
	proportionList = list()
	total = sum(mylist)
	for m in mylist:
		proportionList.append(m/float(total))
	return sorted(proportionList)


def plotLogLog(mydict):
	freqList = mydict.values()

	sumFreq = sum(freqList)

	getProportion(freqList)
	freqCount = dict.fromkeys(set(freqList),0)
	for freq in freqList:
		freqCount[freq] += 1

	x = list(freqList)
	y = list()
	for freq in freqList:
		y.append(freqCount[freq]/float(sumFreq))


	#plt.loglog(x,y)
	#plt.plot(x,y, color='blue', lw=2)


	fig = plt.figure()
	ax = plt.gca()
	#ax.scatter(x ,y , c='blue', alpha=0.05, edgecolors='none')
	#fig = plt.figure()
	ax.plot(x ,y, 'o', c='blue', alpha=0.5, markeredgecolor='none')


	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.show()

def countWordFrequency(tokenList):
	tokenSet = list(set(tokenList)-set(stopword))
	frequencyDict = dict.fromkeys(tokenSet,0)
	for token in tokenList:
		if token not in stopword:
			frequencyDict[token] = frequencyDict[token] + 1
	return frequencyDict

def plusOne(mydict,key):
	if key not in mydict.keys():
		mydict[key] = 1
	else:
		mydict[key] += 1


blogText = readText("blog.txt")
blogToken = nltk.word_tokenize(blogText)
blogToken = [token.lower() for token in blogToken]
blogTag = nltk.pos_tag(blogToken)
#blogToken = removeNonAlphaNumeric(blogToken)
blogFreq = countWordFrequency(blogToken)
plotLogLog(blogFreq)


print "vocabulary size: ",len(blogFreq.keys())




n_stop = 0
n_capital = 0
total_char = 0
for token in blogToken:
	if checkIfStopWord(token):
		n_stop += 1
	n_capital += numCapital(token)
	total_char += len(token)
print "frequency of stopwords: ",n_stop/float(len(blogToken))
print "number of capital letters: ",n_capital
print "average number of characters per word: ",total_char/float(len(blogToken))

n_noun = 0
n_adv = 0
n_adj = 0
n_verb = 0
n_pron = 0
noun_dict = dict()
adv_dict = dict()
adj_dict = dict()
verb_dict = dict()
pron_dict = dict()
for item in blogTag:
	if item[1] == "NN":
		n_noun += 1
		plusOne(noun_dict,item[0])

	elif item[1] == "JJ":
		n_adj += 1
		plusOne(adj_dict,item[0])

	elif item[1] == "RB":
		n_adv += 1
		plusOne(adv_dict,item[0])

	elif item[1] == "PRP":
		n_pron += 1
		plusOne(pron_dict,item[0])

	elif item[1].startswith("VB"):
		n_verb += 1
		plusOne(verb_dict,item[0])

print "number of nouns: ",n_noun
print "number of adverbs: ",n_adv
print "number of adjectives: ",n_adj
print "number of verbs: ",n_verb
print "number of pronouns: ",n_pron

print "noun\n"
sortDict = OrderedDict(sorted(noun_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "adverbs\n"
sortDict = OrderedDict(sorted(adv_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "adjectives\n"
sortDict = OrderedDict(sorted(adj_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "verbs\n"
sortDict = OrderedDict(sorted(verb_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "pronouns\n"
sortDict = OrderedDict(sorted(pron_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break





print

n_stop = 0
n_capital = 0
total_char = 0
speechText = readText("congress_speech.txt")
speechToken = nltk.word_tokenize(speechText)
speechToken = [token.lower() for token in speechToken]
speechTag = nltk.pos_tag(speechToken)
#speechToken = removeNonAlphaNumeric(speechToken)
speechFreq = countWordFrequency(speechToken)
plotLogLog(speechFreq)


print "vocabulary size: ",len(speechFreq.keys())

n_stop = 0
for token in speechToken:
	if checkIfStopWord(token):
		n_stop += 1
	n_capital += numCapital(token)
	total_char += len(token)
print "frequency of stopwords: ",n_stop/float(len(speechToken))
print "number of capital letters: ",n_capital
print "average number of characters per word: ",total_char/float(len(speechToken))

n_noun = 0
n_adv = 0
n_adj = 0
n_verb = 0
n_pron = 0
noun_dict = dict()
adv_dict = dict()
adj_dict = dict()
verb_dict = dict()
pron_dict = dict()
for item in speechTag:
	if item[1] == "NN":
		n_noun += 1
		plusOne(noun_dict,item[0])

	elif item[1] == "JJ":
		n_adj += 1
		plusOne(adj_dict,item[0])

	elif item[1] == "RB":
		n_adv += 1
		plusOne(adv_dict,item[0])

	elif item[1] == "PRP":
		n_pron += 1
		plusOne(pron_dict,item[0])

	elif item[1].startswith("VB"):
		n_verb += 1
		plusOne(verb_dict,item[0])

print "number of nouns: ",n_noun
print "number of adverbs: ",n_adv
print "number of adjectives: ",n_adj
print "number of verbs: ",n_verb
print "number of pronouns: ",n_pron

print "noun\n"
sortDict = OrderedDict(sorted(noun_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "adverbs\n"
sortDict = OrderedDict(sorted(adv_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "adjectives\n"
sortDict = OrderedDict(sorted(adj_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "verbs\n"
sortDict = OrderedDict(sorted(verb_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break
print "pronouns\n"
sortDict = OrderedDict(sorted(pron_dict.items(), key=lambda x: x[1],reverse=True))
n = 0
for key,value in sortDict.items():
	print key,value
	n += 1
	if n>= 10:
		break