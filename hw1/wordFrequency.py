import nltk
import re

def readText(fn):
	f = open(fn, "r")		
	text = f.read()
	f.close()
	return text

def removeNonAlphaNumeric(tokenList):
	for idx,m in enumerate(tokenList):
		if not m.isalnum():
			tokenList.pop(idx)
	return tokenList


def countWordFrequency(tokenList):
	tokenSet = list(set(tokenList))
	frequencyDict = dict.fromkeys(tokenSet,0)
	for token in tokenList:
		frequencyDict[token] = frequencyDict[token]+1
	return frequencyDict


blogText = readText("blog.txt")
blogToken = nltk.word_tokenize(blogText)
print len(blogToken)
blogToken = removeNonAlphaNumeric(blogToken)
print len(blogToken)
blogFreq = countWordFrequency(blogToken)


speechText = readText("congress_speech.txt")
speechToken = nltk.word_tokenize(speechText)
print len(speechToken)
speechToken = removeNonAlphaNumeric(speechToken)
print len(blogToken)
speechFreq = countWordFrequency(speechToken)