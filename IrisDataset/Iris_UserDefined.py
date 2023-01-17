#importing required Libraries 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.spatial import distance

#Function which Calculates Euclidean Distance and returns the Distance between two points
def euc_distance(a,b):
    return distance.euclidean(a,b)

#User-Defined K-Nearest Neighbors Classifier Function
class User_Defined_KNeighbor_Classifier():
    #User-Defined fit method which trains the model
    def fit(self,trainingdata,trainingtarget):
        self.TrainingData = trainingdata
        self.TrainingTarget = trainingtarget
    
    #User-Defined method which returns the closest distance from the given point
    def closest(self,row):
        minimumdistance = euc_distance(row,self.TrainingData[0])
        minimumindex=0

        for i in range(1,len(self.TrainingData)):
            Distance = euc_distance(row,self.TrainingData[i])

            if Distance < minimumdistance:
                minimumdistance = Distance
                minimumindex = i
        
        return self.TrainingTarget[minimumindex]
    
    #User-Defined predict method which Tests the model
    def predict(self,TestData):
        predictions = []

        for value in TestData:
            result = self.closest(value)
            predictions.append(result)
        return predictions

#Functions Performs loading data, training data ,testing data and returning Accuracy 
def Iris_ML():
    #Loading Data set into the program
    Dataset = load_iris()

    #Dividing the independant variable and dependant variables
    Data = Dataset.data
    Target = Dataset.target

    #performs splitting of data into 4 parts
    Data_Train,Data_Test,Target_Train,Target_Test = train_test_split(Data,Target,test_size = 0.5)

    #Creating an object of User-Defined KNN class
    Classifier = User_Defined_KNeighbor_Classifier()

    #Training the data using User-defined fit method
    Classifier.fit(Data_Train,Target_Train)

    #Testing the data using User-defined predict method
    Predictions = Classifier.predict(Data_Test)

    #Calculating accuracy with in-Built Function 
    Accuracy = accuracy_score(Target_Test,Predictions)
    return Accuracy

#Execution of Program starts from main 
def main():
    #Displays header
    print("-----Python Machine Learning Algorithm-----")
    print(" Iris Dataset using User-Defined k-Nearest Neighbors Function")

    #Calling the Algorithm and accepting the accuracy
    Ret = Iris_ML()

    #Displays the Accuracy of Iris Dataset 
    print("Accuracy of Iris Dataset with User-Defined KNN algorithm is ",Ret*100)

#Starter of Application
if __name__ =="__main__":
    main()