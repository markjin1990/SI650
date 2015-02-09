import nltk
import re
from collections import OrderedDict


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


def countWordFrequency(tokenList):
	tokenSet = list(set(tokenList))
	frequencyDict = dict.fromkeys(tokenSet,0)
	for token in tokenList:
		frequencyDict[token] = frequencyDict[token]+1
	return frequencyDict

def plusOne(mydict,key):
	if key not in mydict.keys():
		mydict[key] = 1
	else:
		mydict[key] += 1


blogText = readText("blog.txt")
blogToken = nltk.word_tokenize(blogText)
blogTag = nltk.pos_tag(blogToken)
blogToken = removeNonAlphaNumeric(blogToken)
blogFreq = countWordFrequency(blogToken)
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

	elif item[1] == "IN":
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
speechTag = nltk.pos_tag(speechToken)
speechToken = removeNonAlphaNumeric(speechToken)
speechFreq = countWordFrequency(speechToken)
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
for item in speechTag:
	if item[1] == "NN":
		n_noun += 1
	elif item[1] == "JJ":
		n_adj += 1
	elif item[1] == "IN":
		n_adv += 1
	elif item[1] == "PRP":
		n_pron += 1
	elif item[1].startswith("VB"):
		n_verb += 1

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

	elif item[1] == "IN":
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