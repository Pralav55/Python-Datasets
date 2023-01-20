#Importing required Libraries 
import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure,show
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Function Handling Titanic Dataset using Logistic Regression
def TitanicLogistic():
    #####################################################################
    #Step 1 : Load Data
    #####################################################################
    titanic_data = pd.read_csv('TitanicDataset.csv')

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passengers are "+str(len(titanic_data)))

    #####################################################################
    #Step 2 : Analyse the Data
    #####################################################################
    print("Visualisation : Survived and Non-Survived Passengers")
    figure()
    target = 'Survived'
    countplot(data = titanic_data,x=target).set_title("Survived and Non-survived passengers")
    show()

    print("Visualisation : Survived and Non-Survived passengers based on gender")
    figure()
    target="Survived"
    countplot(data=titanic_data,x=target,hue="Sex").set_title("Survived and Non-survived passengers based on gender")
    show()

    print("Visualisation : Survived and Non-Survived passengers based on passenger class")
    figure()
    target="Survived"
    countplot(data=titanic_data,x=target,hue="Pclass").set_title("Survived and Non-survived passengers based on passenger class")
    show()

    print("Visualisation : Survived and non-Survived passengers based on age")
    figure()
    titanic_data['Age'].plot.hist().set_title("Survived and non-survived passengers based on Age")
    show()

    print("Visualisation : Survived and non survived passengers based on the Fare")
    figure()
    titanic_data['Fare'].plot.hist().set_title("Survived and non survived passengers based on fare")
    show()

    #####################################################################
    #Step 3 : Data Cleaning
    #####################################################################
    titanic_data.drop("zero",axis=1,inplace=True)

    print("First five entries from loaded dataset after removal zero column")
    print(titanic_data.head(5))

    print("Values of Sex column")
    print(pd.get_dummies(titanic_data['Sex']))

    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies(titanic_data["Sex"],drop_first=True)
    print(Sex.head(5))

    print("Values of Pclass column after removing one field")
    Pclass = pd.get_dummies(titanic_data["Pclass"],drop_first=True)
    print(Pclass.head(5))

    print("Values of data set after concatinating new column")
    titanic_data = pd.concat([titanic_data,Sex,Pclass],axis=1)
    print(titanic_data.head(5))

    print("Values of dataset after removing irrelevant columns")
    titanic_data.drop(["Sex","sibsp","Parch","Embarked"],axis=1,inplace=True)
    print(titanic_data.head())

    X = titanic_data.drop("Survived",axis=1)
    Y = titanic_data["Survived"]

    #####################################################################
    #Step 4 : Data Training
    #####################################################################
    xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.5)

    ObjModel = LogisticRegression()

    ObjModel.fit(xtrain.values,ytrain.values)

    #####################################################################
    #Step 5 : Data Testing
    #####################################################################
    prediction = ObjModel.predict(xtest.values)

    #####################################################################
    #Step 6 : Calculate Accuracy
    #####################################################################
    print("Classification report of Logistic Regression is : ")
    print(classification_report(ytest,prediction))

    print("Confusion Matrix of Logistic Regression is : ")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of Logistic Regression is : ",(accuracy_score(ytest,prediction))*100)

#Execution of program starts from main
def main():
    #Displays header
    print("-----Python Machine Learning Algorithm-----")
    print("Titanic Dataset using Logistic Regression")

    #Calling Function Titanic 
    TitanicLogistic()

#Application Starter
if __name__ =="__main__":
    main()