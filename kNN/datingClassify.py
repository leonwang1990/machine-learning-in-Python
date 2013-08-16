import kNN
from numpy import *
def datingClassTest():
    hoRatio = 0.10
    datingdata, datinglabel = file2matrix('datingTestSet2.txt')
    normdata, ranges, minv = autoNorm(datingdata)
    m = normdata.shape[0]
    numoftest = int(m*hoRatio)
    errorcount = 0
    for i in range(numoftest):
        result = kNN.classify0(normdata[i,:],normdata[numoftest:m,:],
                           datinglabel[numoftest:m],3)
        print "the classifier came back with: %d, the real answer is: %d"\
              % (result, datinglabel[i])
        if (result != datinglabel[i]): errorcount +=1.0
    print "the total error rate is: %f" % (errorcount/float(numoftest))

def classifyPerson():
    resultlist = ['not at all','in small doses','in large doses']
    games = float(raw_input(
        "percentage of time spent playing video games?"))
    flymiles = float(raw_input(
        "frequent flier miles earned per year?"))
    icecream = float(raw_input(
        "liters of ice cream consumed per year?"))
    datingdata, datinglabel = kNN.file2matrix('datingTestSet2.txt')
    normdata, ranges, minv = kNN.autoNorm(datingdata)
    inarr = array([flymiles, games, icecream])
    result = kNN.classify0((inarr - minv)/ranges, normdata, datinglabel, 3)
    print "you will probably like this person:", resultlist[result-1]
classifyPerson()
