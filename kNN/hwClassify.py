import kNN
from numpy import *
from os import listdir

#convert the img to (1,1024) vector
def img2vector(filename): 
    vect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            vect[0,32*i+j] = int(linestr[j])
    return vect

def hwClassifyTest():
    hwlabels = []
    #listdir is an os function, get a list of filename in the dir 
    trainfilelist = listdir('trainingDigits')
    m = len(trainfilelist)
    traindata = zeros((m,1024))
    for i in range(m):
        filenamestr = trainfilelist[i]
        #0_12.txt filestr=0_12; numclass=0;split by '.' and '_'
        filestr = filenamestr.split('.')[0]
        numclass = int(filestr.split('_')[0])
        hwlabels.append(numclass)
        traindata[i,:] = img2vector('trainingDigits/%s' % filenamestr)
    testfilelist = listdir('testDigits')
    errorcount = 0
    mtest = len(testfilelist)
    for i in range(10):#change the data of range(mtest)
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('.')[0]
        numclass = int(filestr.split('_')[0])
        testvect = img2vector('testDigits/%s' % filenamestr)
        result = kNN.classify0(testvect, traindata, hwlabels, 3)
        print "handwriting classifier result: %d, the resl answer is: %d"\
              % (result, numclass)  #CARE the (..)
        if (result != numclass): errorcount += 1.0
    print "\nthe error rate is:%f " % (errorcount/float(mtest))#CARE the (..)
#excute the test
hwClassifyTest()
