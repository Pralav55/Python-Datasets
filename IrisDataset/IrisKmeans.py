#importing required python libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#Functions Demonstrate the Kmean algorithm using Iris-Dataset
def Iris_Kmean():
    #Loading Data into Program
    dataset = pd.read_csv('Iris.csv')
    x = dataset.iloc[:,[0,1,2,3]].values
    wcss = []

    #
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i,init='k-means++',max_iter=300,n_init=10,random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,11),wcss)
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    kmeans = KMeans(n_clusters=3,init ='k-means++',max_iter=300,n_init=10,random_state=0)
    y_Kmeans = kmeans.fit_predict(x)

    plt.scatter(x[y_Kmeans==0,0],x[y_Kmeans==0,1],s=30,c='red',label='Iris-Sentosa')
    plt.scatter(x[y_Kmeans==1,0],x[y_Kmeans==1,1],s=30,c='blue',label='Iris-Versicolor')
    plt.scatter(x[y_Kmeans==2,0],x[y_Kmeans==2,1],s=30,c='green',label='Iris-Verginica')
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='yellow',label='Centroids')
    plt.legend()
    plt.savefig("IrisClusterGraph")
    plt.show()

#Execution of Program 
def main():
    #Displays Header
    print("-----Python Machine Learning Algorithm-----")
    print(" Iris Dataset using K-Means Machine Learning Algorithm")

    #Calling the function which Demonstrates the K-Mean algorithm using Iris Dataset
    Iris_Kmean()

#Application starter
if __name__ =="__main__":
    main()