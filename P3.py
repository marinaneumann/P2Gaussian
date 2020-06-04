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
    # plt.scatter(dX, dY, c="blue")
    # plt.show()

def kMeansAlg():
    #clusterVars = []
    k = int(input("What is the value of K?"))
    r = int(input("How many times would you like to run?"))
    kCluster = kMeans(k, dataNum, fNum,data)
    rseed =2
    rng =np.random.RandomState(rseed)
    i = rng.permutation(data.shape[0])[:k]
    centers = data[i]
    kCluster.assign(centers,data)






    # initial_clusters = np.zeros(centers.shape)
    # updatedClusters = centers
    #
    # clusters = np.zeros(dataNum)
    # distances = np.zeros((dataNum, k))
    # e = kCluster.euclideanDistance(updatedClusters, initial_clusters)



    # colors = ['green','red', 'purple']
    # for i in range(dataNum):
    #     plt.scatter(data[i,0], data[i,1], s= 7, color = colors[int(i)])
    # plt.scatter(updatedClusters[:,0], updatedClusters[:,1], marker="*", c='g', s=20)
    # plt.show()


def fuzzyCMeansAlg():
    print("hi")

class kMeans:
    def __init__(self,k, dataNum, fNum, data):
        self.k = k
        self.numData = dataNum
        self.features = fNum
        self.data = data


    def euclideanDistance(self,center, data):
        return (sum((center - data)**2))**0.5

    def assign(self, centers, data):
        clusters = []
        for d in range(dataNum):
            distances = []
            for c in centers:
                distances.append(self.euclideanDistance(c, data[d]))

            cluster = [z for z, val in enumerate(distances) if val == min(distances)]
            clusters.append(cluster[0])
        print(clusters)

    def calcCentroids(self, clusters, c_array):
        #newCentroids = np.array([c_array[]])
        print("sup")

class fuzzyC:
    def __init__(self, k):
        self.k = k
main()
