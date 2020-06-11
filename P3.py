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
    k = int(input("What is the value of K?")) #Takes user input of k
    kCluster = kMeans(k, dataNum, fNum,data) #Creates kCluster object

    rseed =2
    rng =np.random.RandomState(rseed)
    i = rng.permutation(data.shape[0])[:k] # Creates random indexes to assign centers
    kCluster.centers = data[i] #Creates centers at those random data points
    n = int(input("How many iterations?: ")) #Asks user for how many iterations
    squaredSums = np.zeros(n)
    #z = 0
    for z in range(n):
        kCluster.clusters = {}
        kCluster.assign(data)
        #kCluster.calcCenters()
        break
        # if np.all(self.centers == self.prev_centers):
        #     break

        #calculated squared sums and push into array
        #ss = kCluster.squareSums(data, centers)
        # centers = new_centers

        #Track numbers of least squares error variance????


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

    def assign(self, data):

        for i in range(self.k): #Arranges for K different clusters
            self.clusters[i] = []

        for f in data:
            distances = [self.euclideanDistance(f,self.centers[c]) for c in range(self.k)] #calculates distances in data
            cluster = distances.index(min(distances))
            self.clusters[cluster] =np.append(self.clusters[cluster], f)

        self.prev_centers = dict(self.centers)


    def calcCenters(self, clusters):
        for c in self.clusters:
            self.centers[c] = np.average(self.clusters[c], axis =0)


        # count = np.zeros(self.k)
        #
        # for i in range(self.k):
        #     z = 0
        #     for c in clusters:
        #         if c == i:
        #             z +=1
        #     count[i] == z
        #
        # print("Count", count)


    def squareSums(self, data, centers):
        print("blah")

class fuzzyC:
    def __init__(self, k):
        self.k = k
main()
