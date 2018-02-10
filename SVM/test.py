import numpy as np
from sklearn import svm

training = open("iris.data").readlines()[0::2]
testing = open("iris.data").readlines()[1::2]


def getData(lines, labeltype):
    t = [list(map(float, i)) for i in [[i for i in i.split(",")][:-1] for i in lines if i.strip()]]
    t_labels = [i.split(",")[-1:][0].strip('\n') for i in lines if i.strip()]
    t_cat = [j for i, j in enumerate(t) if t_labels[i] == labeltype]
    return t_cat


training_0 = getData(training, 'Iris-setosa')
training_1 = getData(training, 'Iris-versicolor')
training_2 = getData(training, 'Iris-virginica')

testing_0 = getData(testing, 'Iris-setosa')
testing_1 = getData(testing, 'Iris-versicolor')
testing_2 = getData(testing, 'Iris-virginica')

C = 1.0
gamma = 0.5


def testSVM(svm, zero, one, two):
    numcorrect = 0.
    numwrong = 0.
    for correct, testing in ((0, zero), (1, one), (2, two)):
        for d in testing:
            r = svm.predict(np.reshape(d, (1, -1)))[0]
            if r == correct:
                numcorrect += 1
            else:
                numwrong += 1
    print("Wrong", numwrong)
    print("Correct", numcorrect)
    print("Accuracy", (numcorrect) / (numcorrect + numwrong), "\n")


X = np.array(training_0 + training_1 + training_2)
Y = np.array([0 for i in training_0] + [1 for j in training_1] + [2 for k in training_2])

svm_linear = svm.SVC(kernel='linear',C=C,gamma=gamma).fit(X, Y)
svm_poly = svm.SVC(kernel='poly',C=C,gamma=gamma).fit(X, Y)
svm_sigmoid = svm.SVC(kernel='sigmoid',C=C,gamma=gamma).fit(X, Y)
svm_rbf = svm.SVC(kernel='rbf',C=C,gamma=gamma).fit(X, Y)

print("Linear")
testSVM(svm_linear, testing_0, testing_1, testing_2)

print("RBF")
testSVM(svm_rbf, testing_0, testing_1, testing_2)

print("Sigmoid")
testSVM(svm_sigmoid, testing_0, testing_1, testing_2)

print("Poly")
testSVM(svm_poly, testing_0, testing_1, testing_2)
