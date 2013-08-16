from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffmat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffmat = diffmat**2
    sqDistances = sqDiffmat.sum(axis=1) #axis=0 column axis=1 row
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         
    returnMat = zeros((numberOfLines,3))  #(numberOfLines,3) is a shape of a matrix
    classLabelVector = []                       
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()   #get rid of the \n
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataset):
    minv = dataset.min(0)
    maxv = dataset.max(0)
    ranges = maxv - minv
    m = dataset.shape[0]
    normdata = dataset - tile(minv,(m,1)) #tile(minv,(m,n)),expand minv:([x,y,z])
                                          #to m*n matrix in which each element is [x,y,z]
    normdata = normdata/tile(ranges,(m,1))
    return normdata,ranges,minv
