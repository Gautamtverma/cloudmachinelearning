import os, sys, numpy as np, re
from os import listdir
from os.path import isfile, join

maxSeqLength = 250

def cleanSentences(string, strip_special_chars):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def creating_ids_matrix():

    positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    numWords = []
    wordsList = np.load('wordsList.npy')

    for pf in positiveFiles:
        with open(pf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            numWords.append(counter)
    print('Positive files finished')

    for nf in negativeFiles:
        with open(nf, "r", encoding='utf-8') as f:
            line = f.readline()
            counter = len(line.split())
            numWords.append(counter)
    print('Negative files finished')

    numFiles = len(numWords)

    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
    fileCounter = 0
    for pf in positiveFiles:
       with open(pf, "r") as f:
           indexCounter = 0
           line=f.readline()

           cleanedLine = cleanSentences(line, strip_special_chars)
           split = cleanedLine.split()
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = wordsList.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
               indexCounter = indexCounter + 1
               if indexCounter >= maxSeqLength:
                   break
           fileCounter = fileCounter + 1

    for nf in negativeFiles:
       with open(nf, "r") as f:
           indexCounter = 0
           line=f.readline()
           cleanedLine = cleanSentences(line, strip_special_chars)
           split = cleanedLine.split()
           for word in split:
               try:
                   ids[fileCounter][indexCounter] = wordsList.index(word)
               except ValueError:
                   ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
               indexCounter = indexCounter + 1
               if indexCounter >= maxSeqLength:
                   break
           fileCounter = fileCounter + 1
    #Pass into embedding function and see if it evaluates.

    np.save('idsMatrix', ids)