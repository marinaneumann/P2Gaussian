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
    kCluster = kMeans(k)
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    centers = np.random.rand(k,fNum)*std + mean

     for i in range(k):
        distances[:,i] = kCluster.euclideanDistance(data, centers[i])
        print("Distances: ", distances)


    # initial_clusters = np.zeros(centers.shape)
    # updatedClusters = centers
    #
    # clusters = np.zeros(dataNum)
    # distances = np.zeros((dataNum, k))
    # e = kCluster.euclideanDistance(updatedClusters, initial_clusters)
    # print("Value of error:", e)
    #centroids = [data[i+2] for i in range(k)]
    #clusters = kCluster.assign(centroids,data)
    #initial_clusters = clusters

    # while (e !=0).all() :
    #     for i in range(k):
    #         distances[:,i] = kCluster.euclideanDistance(data[i], updatedClusters[i])
    #     clusters = np.argmin(distances, axis =1)
    #
    #     initial_clusters = updatedClusters

    #     for i in range(k):
    #         updatedClusters[i] = np.mean(data[clusters ==i], axis =0)
    #     e = kCluster.euclideanDistance(updatedClusters, initial_clusters)
    # print("These are the updated clusters:", updatedClusters)
    #
    # colors = ['green','red', 'purple']
    # for i in range(dataNum):
    #     plt.scatter(data[i,0], data[i,1], s= 7, color = colors[int(i)])
    # plt.scatter(updatedClusters[:,0], updatedClusters[:,1], marker="*", c='g', s=20)
    # plt.show()


def fuzzyCMeansAlg():
    print("hi")

class kMeans:
    def __init__(self,k):
        self.k = k

    def euclideanDistance(self,centroid, c):
        return (sum((centroid - c)**2))**0.5

    def assign(self, centroids, c_array):
        print("Assigning stuff")
        clusters = []
        for i in range(c_array.shape[0]):
            distances = []
            for centroid in centroids:
                distances.append(self.euclideanDistance(centroid,c_array[i]))
            cluster = [z for z, val in enumerate(distances) if val== min(distances)]
            clusters.append(cluster[0])
        return clusters

    def calcCentroids(self, clusters, c_array):
        #newCentroids = np.array([c_array[]])
        print("sup")

class fuzzyC:
    def __init__(self, k):
        self.k = k
main()
