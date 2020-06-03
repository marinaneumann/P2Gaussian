import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Programming Assignment #3")
    print("By Marina Neumann")
    print("Spring 2020 Machine Learning")

    dataLoad()
    choice = ''
    while choice != 'e':
        choice = input("Enter 'k' for K-Means clustering, 'f' for Fuzzy C-Means, and 'e' for exit: ")
        print("\n")
        if choice == 'k':
            print("K-Means Clustering")
            kMeansAlg()
        elif choice == 'f':
            fuzzyCMeansAlg()



def dataLoad():
    global data, dX, dY, dataNum, fNum
    data = np.loadtxt("data/cluster_dataset.txt", usecols=(0,1))
    dX = data[:,0]
    dY = data[:,1]
    dataNum = data.shape[0]
    fNum = data.shape[1]
    plt.scatter(dX, dY, c="blue")
    plt.show()

def kMeansAlg():

    k = int(input("What is the value of K?"))
    r = int(input("How many times would you like to run?"))

    for i in range(r):
        print("wassssup")


def fuzzyCMeansAlg():
    print("hi")

class kMeans:
    def __init__(self,k):
        self.k = k


class fuzzyC:
    def __init__(self, k):
        self.k = k
main()
