import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

import warnings

warnings.simplefilter("ignore")

training = [[int(i) for i in i.split(",")] for i in open("pendigits.tra").readlines() if i.strip()]
testing = [[int(i) for i in i.split(",")] for i in open("pendigits.tes").readlines() if i.strip()]

#split training and test data
training_0 = [i[:-1] for i in training if i[-1] == 0]
training_1 = [i[:-1] for i in training if i[-1] == 1]

testing_0 = [i[:-1] for i in testing if i[-1] == 0]
testing_1 = [i[:-1] for i in testing if i[-1] == 1]


# Mapping to 2d --- for plot purposes

def mapTo2D(data):
    retval = []
    for i in range(0, len(data), 2):
        x = data[i]
        y = data[i + 1]
        retval.append((x, y))
    return retval


training_2d_0 = []
training_2d_1 = []

for d in training_0:
    training_2d_0 += mapTo2D(d)

for d in training_1:
    training_2d_1 += mapTo2D(d)

# Plotting 2d

plt.subplot(2, 2, 1)
plt.plot([i[0] for i in training_2d_0], [i[1] for i in training_2d_0], "-o", color="green")

plt.subplot(2, 2, 2)
plt.plot([i[0] for i in training_2d_1], [i[1] for i in training_2d_1], "-o", color="green")

plt.subplot(2, 2, 3)
plt.plot([i[0] for i in training_2d_0][:8], [i[1] for i in training_2d_0][:8], "-o", color="green")

plt.subplot(2, 2, 4)
plt.plot([i[0] for i in training_2d_1][:8], [i[1] for i in training_2d_1][:8], "-o", color="green")
plt.show()

#create x and y axis
X = np.array(training_2d_0[:8] + training_2d_1[:8])
Y = np.array([0 for i in training_2d_0[:8]] + [1 for i in training_2d_1[:8]])

#use soft margin C
C = 1.0 #learning rate
gamma = 0.5  #learning rate
#create different functions of svm
svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)
#svm_gaussian = svm.SVC(kernel='gaussian',C=C, gamma = gamma).fit(X, Y)
h = 0.2  # Mesh step
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1   #
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

def plotSVM(svm, n, title):
    plt.subplot(2, 2, n)
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.title(title)

#3 models: Liner, RBF, Sigmoid
#svm_model, number, "title"
plotSVM(svm_linear, 1, "Linear")
plotSVM(svm_rbf, 2, "RBF")
plotSVM(svm_sigmoid, 3, "Sigmoid")
#plotSVM(svm_gaussian, 4, "Gaussian")

plt.show()

testing_2d_0 = []
testing_2d_1 = []
for d in testing_0:
    testing_2d_0 += mapTo2D(d)

for d in testing_1:
    testing_2d_1 += mapTo2D(d)


def testSVM(svm, zero, one):
    numcorrect = 0
    numwrong = 0
    for correct, testing in ((0, zero), (1, one)):
        for d in testing:
            #print(d)
            r = svm.predict(np.reshape(d,(1,-1)))[0]
            #print(r)
            if (r == correct):
                numcorrect += 1
            else:
                numwrong += 1
    print("Correct", numcorrect)
    print("Wrong", numwrong)
    print("Accuracy", (numcorrect)/(numcorrect + numwrong))


print("Linear")
testSVM(svm_linear, testing_2d_0, testing_2d_1)
print('-------------------------------- '*2)
print('-------------------------------- '*2)
print("RBF")
testSVM(svm_rbf, testing_2d_0, testing_2d_1)
print('-------------------------------- '*2)
print('-------------------------------- '*2)
print ("Sigmoid")
testSVM(svm_sigmoid, testing_2d_0, testing_2d_1)
print('-------------------------------- '*2)
print('-------------------------------- '*2)

# 16d data
X = np.array(training_0 + training_1)
Y = np.array([0 for i in training_0] + [1 for i in training_1])

svm_linear = svm.SVC(kernel='linear', C=C, gamma=gamma).fit(X, Y)
svm_poly = svm.SVC(kernel='poly', C=C, gamma=gamma).fit(X, Y)
svm_rbf = svm.SVC(kernel='rbf', C=C, gamma=gamma).fit(X, Y)
svm_sigmoid = svm.SVC(kernel='sigmoid', C=C, gamma=gamma).fit(X, Y)

print("16d data")
print('-------------------------------- '*2)
print("Linear")
testSVM(svm_linear, testing_0, testing_1)
print('-------------------------------- '*2)
print('-------------------------------- '*2)
print("Polinomial")
testSVM(svm_poly, testing_0, testing_1)
print('-------------------------------- '*2)
print('-------------------------------- '*2)
print("RBF")
testSVM(svm_rbf, testing_0, testing_1)
print('-------------------------------- '*2)
print('-------------------------------- '*2)
print("Sigmoid")
testSVM(svm_sigmoid, testing_0, testing_1)
