import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

#Very important function for clustering algorithms below
def euclideanDistance(X, Y):
        return (sum((X-Y)**2))**0.5

def kMeansAlg():

    k = int(input("What is the value of K?")) #Takes user input of k
    kCluster = kMeans(k, dataNum, fNum,data) #Creates kCluster object
    n = int(input("How many iterations?: "))  # Asks user for how many iterations

    rseed =2
    rng =np.random.RandomState(rseed)
    i = rng.permutation(data.shape[0])[:k] # Creates random indexes to assign centers
    kCluster.centers = data[i] #Creates centers at those random data points
    squaredSums = []

    for z in range(n):
        kCluster.clusters = {}
        kCluster.assign(data)
        kCluster.calcCenters()

        if np.all(kCluster.centers == kCluster.prev_centers):
            break
        #Sum of square values appened to array outside loop to be evaluated later for lowest SSS.
        ss = kCluster.squareSums()
        squaredSums =np.append(squaredSums, ss)


    #Code for sum of squares results from different clustering iterations w/ different K values
    sumMin = np.argmin(squaredSums)
    print("Solution with lowest sum was at itertion:", sumMin)
    mSS = squaredSums[sumMin]
    print("Lowest sum was :", mSS)

    #Code used to graph clustering of algorithm
    colors = ['green', 'purple', 'blue', 'orange', 'brown', 'pink', 'black', 'violet', 'teal','tomato','maroon', 'olive', 'gold']

    for c in kCluster.centers:
        plt.scatter(c[0], c[1], marker='*', color="red", s=200)

    for d in range(kCluster.k):
        color = colors[d]
        num = len(kCluster.clusters[d])/2
        kCluster.clusters[d] = np.array_split(kCluster.clusters[d],num )

        for k in kCluster.clusters[d]:
            plt.scatter(k[0],k[1], marker="o", color=color, s=2)
    plt.title('K-Means Algorithm' ,size=16)
    plt.show()

#Kmeans class and functions
class kMeans:
    def __init__(self,k, dataNum, fNum, data):
        self.k = k
        self.numData = dataNum
        self.features = fNum
        self.data = data

    #Responsible for assigning data to different clusters
    def assign(self, data):
        for i in range(self.k): #Arranges for K different clusters
            self.clusters[i] = []
        for f in data:
            distances = [euclideanDistance(f,self.centers[c]) for c in range(self.k)] #calculates distances in data
            cluster = distances.index(min(distances))
            self.clusters[cluster] =np.append(self.clusters[cluster],f)
        self.prev_centers = dict(self.centers)

    #Calculates centers for each clusters
    def calcCenters(self):
        for c in self.clusters:
            self.centers[c] = np.mean(self.clusters[c], axis =0)

    #Calculates square sums for value at K, with state of clusters and centers at that iteration.
    def squareSums(self ):
        sums = 0
        for i in range(self.k):
            cSUM = 0
            for z in self.clusters[i]:
                cSUM += euclideanDistance(z, self.centers[i])
            sums += cSUM
        return sums

#Function to declare fuzzy C-means functions and such
def fuzzyCMeansAlg():
    print("hi")
    k = int(input("What is the value of K?"))  # Takes user input of k
    fMeans = fuzzyC(k, dataNum, fNum, data)  # Creates kCluster object
    fMeans.membership = np.random.rand(dataNum,k)
    n = int(input("How many iterations?: "))
    squaredSums = []
    for i in range(n):

        fMeans.clusters = {}
        fMeans.computeCenter()
        fMeans.updateMembership()
        fMeans.clustering()

        if np.sum(~(fMeans.centers == fMeans.old_centers)) < 1:
            print("Centers Optimized")
            break

        MS = fMeans.sumSquares()
        squaredSums = np.append(squaredSums, MS)

    print(squaredSums)
    SM = np.argmin(squaredSums)
    print("Solution with lowest sum was at itertion:", SM)
    mSS = squaredSums[SM]
    print("Lowest sum was :", mSS)

    colors = ['green', 'purple', 'blue', 'orange', 'brown', 'pink', 'black', 'violet', 'teal', 'tomato', 'maroon',
              'olive', 'gold']

    for c in fMeans.centers:
        plt.scatter(c[0], c[1], marker='*', color="red", s=200)

    for d in range(fMeans.k):
        color = colors[d]
        num = len(fMeans.clusters[d]) / 2
        fMeans.clusters[d] = np.array_split(fMeans.clusters[d], num)

        for k in fMeans.clusters[d]:
            plt.scatter(k[0], k[1], marker="o", color=color, s=2)
    plt.title('Fuzzy C-Means Algorithm', size=16)
    plt.show()

#Fuzzy C class and functions
class fuzzyC:
    def __init__(self, k, dataNum, fNum, data):
        self.k = k
        self.dNum = dataNum
        self.fNum = fNum
        self.data = data
        self.m = 2  #Default set value,for fuzzier parameter
        self.centers ={}

    #Function to computer centers for each cluster, using fuzzy c centers equation
    def computeCenter(self):
        self.old_centers = self.centers
        wkM = np.power(self.membership, self.m)
        self.centers = np.transpose(np.matmul(self.data.T, wkM) / np.sum(wkM, axis=0))

    #Function to update membership of each data and its relation to specific cluster
    def  updateMembership(self):
        print("Blah Blah Blah ")
        self.old_membership = self.membership
        d = float(2 / (self.m - 1))
        for i in range(self.dNum):
            distances = [euclideanDistance(self.data[i],self.centers[j])for j in range(self.k)]
            for j in range(self.k):
                den = sum([pow(float(distances[j] / distances[c]), d) for c in range(self.k)])
                self.membership[i][j] = float(1 / den)

    #Function for the clustering of data depending on data X to cluster C
    def clustering(self):
        for i in range(self.k):
            self.clusters[i] = []

        for f in self.data:
            distances = [euclideanDistance(f, self.centers[c]) for c in range(self.k)]  # calculates distances in data
            cluster = distances.index(min(distances))
            self.clusters[cluster] = np.append(self.clusters[cluster], f)
        self.prev_centers = dict(self.centers)

    #Function to calculate sum of squares for Fuzzy C clusters and centers at a specific iteration
    def sumSquares(self):
        print("BLOOOPPPP?")
        sums = 0
        for i in range(self.k):
            cSUM = 0
            for z in self.clusters[i]:
                cSUM += euclideanDistance(z, self.centers[i])
            sums += cSUM
        print("These are the sums?:", sums)
        return sums

main() #Function call which puts everything into motion..
