import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter

def Diabetes_LogisticRegression():
    Diabetes = pd.read_csv('diabetes.csv')

    print("Column of datasets")
    print(Diabetes.columns)

    print("First 5 records of dataset")
    print(Diabetes.head())

    print("Dimension of diabetes data : {}".format(Diabetes.shape))

    X_train,X_test,Y_train,Y_test = train_test_split(Diabetes.loc[:,Diabetes.columns != 'Outcome'],Diabetes['Outcome'],stratify = Diabetes['Outcome'],random_state = 66)

    logreg = LogisticRegression().fit(X_train,Y_train)
    print("Training set Accuracy : {:.3f}".format(logreg.score(X_train,Y_train)))
    print("Testing set accuracy : {:.3f}".format(logreg.score(X_test,Y_test)))

    logreg2 = LogisticRegression(C=0.01).fit(X_train,Y_train)
    print("Training set Accuracy : {:.3f}".format(logreg2.score(X_train,Y_train)))
    print("Testing set accuracy : {:.3f}".format(logreg2.score(X_test,Y_test)))

def main():
    print("-----Python Machine Learning Algorithm-----")
    print("Diabetes predictor using Logistic Regression")

    Diabetes_LogisticRegression()

if __name__ == "__main__":
    main()