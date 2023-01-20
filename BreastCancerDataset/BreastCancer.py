#Importing Required Header File Library
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#CSV file name into variable
INPUT_PATH = 'breast-cancer-wisconsin.data'
OUTPUT_PATH = 'breast-cancer-wisconsin.csv'

#Headers of the csv file
HEADERS = ['CodeNumber','ClumpThickness','UniformityCellSize','UniformityCellShape','MarginalAdhesion','SingleEpithelialCellSize','BareNuclei','BlandChromatin','NormalNucleoli','Mitoses','CancerType']

#Function which creates Log of predicted and Actual Output
def PredictedLog(Y_test,predictions,log_dir = 'DatasetLog'):
    
    #Creates a Directory on same path of .py file
    if not os.path.exists(log_dir):
        try:
            os.mkdir(log_dir)
        except:
            pass
    
    seperator = "-"*80

    #Creates a file inside the Directory created
    log_path = os.path.join(log_dir,"PredictedLog%s.log"%(time.ctime()))

    #Opens the file into write mode
    f = open(log_path,'w')

    #Write contents into file
    f.write(seperator+"\n")
    f.write("Breast Cancer Predicted Logger : "+time.ctime()+"\n")
    f.write(seperator+"\n")

    #Writes the actual output and predicted output inside the file
    for i in range(0,205):
        f.write("Actual Outcome :: {} and Predicted Outcome :: {}\n".format(list(Y_test)[i],predictions[i]))

#Function which creates object of Random Forest Algorithm and train the algorithm
def BreastCancer_RandomForest(features,target):
    clf = RandomForestClassifier()
    clf.fit(features,target)
    return clf

#Function which splits the data of training and testing as well as splits featurs and label 
def Split_dataset(dataset,train_percentage,feature_headers,target_header):
    X_train,X_test,Y_train,Y_test = train_test_split(dataset[feature_headers],dataset[target_header],train_size = train_percentage)
    return X_train,X_test,Y_train,Y_test

#Function which Filters missing values from dataset
def handel_missing_values(dataset,missing_value_header,missing_label):
    return dataset[dataset[missing_value_header]!=missing_label]

#Function Prints basic statistics of dataset
def dataset_statistics(dataset):
    print(dataset.describe())

#Main Function from where the Execution starts
def main():
    #Load the csv file into pandas dataframe
    dataset = pd.read_csv(OUTPUT_PATH)

    #Get basic statistics of the loaded dataset
    dataset_statistics(dataset)

    #Filter missing values
    dataset = handel_missing_values(dataset,HEADERS[6],'?')

    #Split the data
    X_train,X_test,Y_train,Y_test = Split_dataset(dataset,0.7,HEADERS[1:-1],HEADERS[-1])

    #Prints the size details of train and test dataset
    print("X_Train shape :: ",X_train.shape)
    print("Y_Train shape :: ",Y_train.shape)
    print("X_Test shape :: ",X_test.shape)
    print("Y_Test shape :: ",Y_test.shape)

    #Creates Random forest classifier instance
    trained_model = BreastCancer_RandomForest(X_train,Y_train)
    print("Trained model :: ",trained_model)
    predictions = trained_model.predict(X_test)

    #Creating log of Predicted and actual output
    PredictedLog(Y_test,predictions)

    #Printing Accuracy of Training and Testing
    print("Train Accuracy :: ",accuracy_score(Y_train,trained_model.predict(X_train)))
    print("Test Accuracy :: ",accuracy_score(Y_test,predictions))
    
    #Printing Confusion matrix
    print("Confusion Matrix : ")
    print(confusion_matrix(Y_test,predictions))

#Application Starter
if __name__ == "__main__":
    main()