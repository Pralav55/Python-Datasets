import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def Diabetes_KNeighborsClassifier():
    Diabetes = pd.read_csv('diabetes.csv')

    print("Column of Datasets")
    print(Diabetes.columns)

    print("First 5 records of Dataset")
    print(Diabetes.head())

    print("Dimension of Diabetes data  : {}".format(Diabetes.shape))

    train_x,test_x,train_y,test_y = train_test_split(Diabetes.loc[:,Diabetes.columns !='Outcome'],Diabetes['Outcome'],stratify = Diabetes['Outcome'],random_state = 66)

    training_accuracy = []
    testing_accuracy = []

    neighbors_settings = range(1,11)

    for n_neighbors in neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_x,train_y)

        training_accuracy.append(knn.score(train_x,train_y))

        testing_accuracy.append(knn.score(test_x,test_y))
    
    plt.plot(neighbors_settings,training_accuracy,label ="training accuracy")
    plt.plot(neighbors_settings,testing_accuracy,label = "testing accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.show()

    Knn = KNeighborsClassifier(n_neighbors=3)
    Knn.fit(train_x,train_y)

    print("Accuracy of KNN classifier on training data set : {:.2f}".format(Knn.score(train_x,train_y)))

    print("Accuracy of KNN classifier on testing data set : {:.2f}".format(Knn.score(test_x,test_y)))


def main():
    print("-----Python Machine Learning Algorithm-----")
    print("Diabetes predictor using K-Nearest Neighbors")

    Diabetes_KNeighborsClassifier()

if __name__ =="__main__":
    main()