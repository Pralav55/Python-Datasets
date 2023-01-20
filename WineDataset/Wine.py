#Importing required libraries 
from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#Function Performing Accuracy to Wine Dataset
def WinePredict():

    #Loading wine data into program
    wine = datasets.load_wine()

    #Printing all Independant variable
    print(wine.feature_names)

    #Printing all Dependant variable
    print(wine.target_names)

    #Performs data splitting into training and testing
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

    #Creating Object of KNN with neighbors as 3
    Knn = KNeighborsClassifier(n_neighbors = 3)

    #Training the model using fit method
    Knn.fit(Xtrain,Ytrain)

    #Testing the model using predict method
    y_predict = Knn.predict(Xtest)

    #Printing Accuracy of the algorithm using accuracy_score method
    print("Accuracy : ",(metrics.accuracy_score(Ytest,y_predict))*100)

#Execution starts from main function
def main():
    #Displays Header
    print("-----Python Machine Learning Algorithm-----")
    print("Wine Dataset using K-Nearest Neighbors classifier")

    #Calling Function to Calculate Accuracy
    WinePredict()

#Application Starter
if __name__ == "__main__":
    main()