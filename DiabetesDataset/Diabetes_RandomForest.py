import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter

def Diabetes_RandomForest():
    Diabetes = pd.read_csv('diabetes.csv')

    print("Columns of Dataset")
    print(Diabetes.columns)

    print("First 5 records of dataset")
    print(Diabetes.head())

    print("Dimensions of diabetes data : {}".format(Diabetes.shape))

    X_train,X_test,Y_train,Y_test = train_test_split(Diabetes.loc[:,Diabetes.columns !='Outcome'],Diabetes['Outcome'],stratify = Diabetes['Outcome'],random_state=66)
    
    rf = RandomForestClassifier(n_estimators = 100,random_state =0)
    rf.fit(X_train,Y_train)

    print("Accuracy on training set : {:.3f}".format(rf.score(X_train,Y_train)))
    print("Accuracy on test set : {:.3f}".format(rf.score(X_test,Y_test)))


    rf1 = RandomForestClassifier(max_depth = 3,n_estimators = 100,random_state =0)
    rf1.fit(X_train,Y_train)

    print("Accuracy on training set : {:.3f}".format(rf1.score(X_train,Y_train)))
    print("Accuracy on test set : {:.3f}".format(rf1.score(X_test,Y_test)))

    plt.figure(figsize=(8,6))
    n_features = 8

    plt.barh(range(n_features),rf.feature_importances_,align='center')
    diabetes_features = [x for i,x in enumerate(Diabetes.columns) if i!=8]

    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.ylim(-1,n_features)
    plt.show()

def main():
    print("-----Python Machine Learning Algorithm-----")
    print("Diabetes predictor using Random Forest")

    Diabetes_RandomForest()

if __name__ == "__main__":
    main()
