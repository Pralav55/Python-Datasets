#importing required Libraries
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Function to Plot bar-graph of Accuracy of both Algorithm
def Accuracy_graph(Acc1,Acc2):

    Names = ['KNeighborClassifier','DecisionTreeClassifier']
    Acc = [Acc1,Acc2]
    bar_label = ['KNN','DT']
    bar_colors = ['tab:red','tab:blue']
    fig,ax = plt.subplots()

    hbar = ax.bar(Names,Acc,label=bar_label,color=bar_colors,width=0.4)
    ax.bar_label(hbar,fmt='%.2f')
    ax.set_ylabel("Percentage")
    ax.set_title("Iris Accuracy using KNN and DT")
    ax.legend(title="Iris-Dataset")
    plt.show()

#Function which Calculates Accuracy using Decision Tree Algorithm
def Decision_Tree_Classifier(Data_Train,Data_Test,Target_Train,Target_Test):

    #Creating Object of Decision Tree Algorithm
    Classifier = DecisionTreeClassifier()

    #Training the dataset using fit method
    Classifier.fit(Data_Train,Target_Train)

    #Testing the dataset using predict method
    Predictions = Classifier.predict(Data_Test)

    #Calculating Accuracy using accuracy_score method
    Accuracy = accuracy_score(Target_Test,Predictions)

    return Accuracy

#Function which Calculates Accuracy using KNN Algorithm
def KNearest_NeighborsClassifier(Data_Train,Data_Test,Target_Train,Target_Test):

    #Creating Object of KNN Algorithm
    Classifier = KNeighborsClassifier()

    #Training the dataset using fit method
    Classifier.fit(Data_Train,Target_Train)

    #Testing the dataset using predict method
    Predictions = Classifier.predict(Data_Test)

    #Calculating Accuracy using accuracy_score method
    Accuracy = accuracy_score(Target_Test,Predictions)

    return Accuracy

#Function which performs Loading of dataset
def Load_Data():

    #Loading the Dataset into the Program
    Dataset = load_iris()

    #Seperating the independant and dependant variables 
    Data = Dataset.data
    Target = Dataset.target

    #Shuffling and splitting of data
    Data_Train,Data_Test,Target_Train,Target_Test = train_test_split(Data,Target,test_size=0.7)

    #Calculating Accuracy with KNN algorithm
    Acc1 = KNearest_NeighborsClassifier(Data_Train,Data_Test,Target_Train,Target_Test)
    
    #Calculating Accuracy with Decision Tree Algorithm
    Acc2 = Decision_Tree_Classifier(Data_Train,Data_Test,Target_Train,Target_Test)

    #Displaying the Accuracy of both the Algorithm
    print("Accuracy of Iris Dataset using KNearest Neighbor Classifier is : ",Acc1*100)
    print("Accuracy of Iris Dataset using Decision Tree Classifier is : ",Acc2*100)

    #Function to Plot bar-graph of Accuracy of both Algorithm
    Accuracy_graph(Acc1*100,Acc2*100)

#Execution starting from main
def main():
    #Displays Header
    print("-----Python Machine Learning Algorithm-----")
    print(" Iris Dataset using Decision tree and KNN algorithm with same training and testing data")
    print("")

    #Calling Function which performs Loading,splitting of data
    Load_Data()

#Application Starter
if __name__ =="__main__":
    main()