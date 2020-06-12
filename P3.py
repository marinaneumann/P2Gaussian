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

def kMeansAlg():
    #clusterVars = []
    k = int(input("What is the value of K?")) #Takes user input of k
    kCluster = kMeans(k, dataNum, fNum,data) #Creates kCluster object

    rseed =2
    rng =np.random.RandomState(rseed)
    i = rng.permutation(data.shape[0])[:k] # Creates random indexes to assign centers
    kCluster.centers = data[i] #Creates centers at those random data points
    n = int(input("How many iterations?: ")) #Asks user for how many iterations
    squaredSums = []
    #z = 0
    for z in range(n):
        kCluster.clusters = {}
        kCluster.assign(data)
        kCluster.calcCenters()

        if np.all(kCluster.centers == kCluster.prev_centers):
            break

        ss = kCluster.squareSums()
        squaredSums =np.append(squaredSums, ss)
    sumMin = np.argmin(squaredSums)
    print("Solution with lowest sum was at itertion:", sumMin)
    mSS = squaredSums[sumMin]
    print("Lowest sum was :", mSS)

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
    #plt.savefig('Kmeans2.png', dpi=150)

def fuzzyCMeansAlg():
    print("hi")

class kMeans:
    def __init__(self,k, dataNum, fNum, data):
        self.k = k
        self.numData = dataNum
        self.features = fNum
        self.data = data


    def euclideanDistance(self,X, Y):
        return (sum((Y-X)**2))**0.5

    def assign(self, data):
        for i in range(self.k): #Arranges for K different clusters
            self.clusters[i] = []
        for f in data:
            distances = [self.euclideanDistance(f,self.centers[c]) for c in range(self.k)] #calculates distances in data
            cluster = distances.index(min(distances))
            self.clusters[cluster] =np.append(self.clusters[cluster],f)
        self.prev_centers = dict(self.centers)

    def calcCenters(self):
        for c in self.clusters:
            self.centers[c] = np.average(self.clusters[c], axis =0)

    def squareSums(self ):
        sums = 0
        for i in range(self.k):
            cSUM = 0
            for z in self.clusters[i]:
                cSUM += self.euclideanDistance(z, self.centers[i])
            sums += cSUM
        return sums

class fuzzyC:
    def __init__(self, k):
        self.k = k
main()
