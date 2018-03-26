'''Author = Morten Goodwin, python 3 from scratch'''
import numpy as np
import math

X = np.array(([[1,-1,-1,1,  -1,1,1,-1, -1,1,1,-1,  1,-1,-1,1],] ),dtype=float)
Y = np.array(([[1,0],])) # 1,0 is X, 0,1 is Y
print(X)

#layer = 0
def printLayer(i):
    global layer
    print("Layer:",layer,":",i)
    layer += 1

def conv(inputdata,filterdata):
    retval = []
    width = int(len(inputdata)**0.5)
    for i in range(len(inputdata)):
        if(i//width==(i+1)//width and i+width<len(inputdata)):
            indexes = i,i+1,i+width,i+width+1
            data = [inputdata[i] for i in indexes]
            multi = [data[i]*filterdata[i] for i in range(len(filterdata))]
            retval.append((sum(multi)/len(multi)))
    return np.array(retval)

afilter = np.array(([1,-1,-1,1]))

def manyRelu(inputdata):
    relu = lambda z: z if z>0 else 0
    return np.array([relu(z) for z in inputdata])

def maxPooling(inputdata):
    retval = []
    width = int(len(inputdata)**0.5)
    for i in range(len(inputdata)):
        if(i//width==(i+1)//width and i+width<len(inputdata)):
            indexes = i,i+1,i+width,i+width+1
            data = [inputdata[i] for i in indexes]
            retval.append(max(data))
    return np.array(retval)

fullyLayerX = [0.9,0.1,0.1,0.9]
fullyLayerY = [0.1,0.9,0.9,0.1]

def fullyConnected(inputdata):
    retval = []
    sigmoid = lambda x:1/(1+math.exp(-x))
    X = sum([inputdata[i]*fullyLayerX[i] for i in range(len(fullyLayerX))])
    Y = sum([inputdata[i]*fullyLayerY[i] for i in range(len(fullyLayerY))])
    return np.array((sigmoid(X),sigmoid(Y)))

def loss(inputdata,target):
   return np.array(sum([(inputdata[i]*target[i])**2 for i in range(len(inputdata))])/2.)

def backprob(inputdata,y):
    loss = lambda x,y:(x-y)**2
    lossder = lambda x,y:2*(x-y)
    LEARNINGRATE = 0.01
    global fullyLayerX
    fullyLayerX = [fullyLayerX[i]-LEARNINGRATE*lossder(y[0],inputdata[0]) for i in range(len(fullyLayerX))]
    global fullyLayerY
    fullyLayerY = [fullyLayerY[i]-LEARNINGRATE*lossder(y[0],inputdata[0]) for i in range(len(fullyLayerY))]
    print(fullyLayerX)
    print(fullyLayerY)
#Actual Neural Network
for j in range(10000):
  for i in range(len(X)):
    global layer
    layer = 0
    x = X[i]
    y = Y[i]
    #printLayer(x)
    x2 = conv(x,afilter)
    #printLayer(x2)
    x3 = manyRelu(x2)
    #printLayer(x3)
    x4 = maxPooling(x3)
    #printLayer(x4)
    x5 = fullyConnected(x4)
    #printLayer(x5)
    x6 = loss(x5,y)
    #printLayer(x6)
    backprob(x5,y)