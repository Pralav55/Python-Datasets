import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def Diabetes_KNN(X_train,X_test,Y_train,Y_test):
    Knn = KNeighborsClassifier(n_neighbors=3)

    Knn.fit(X_train,Y_train)

    Train = Knn.score(X_train,Y_train)
    Test = Knn.score(X_test,Y_test)

    return Train,Test

def Diabetes_DT(X_train,X_test,Y_train,Y_test):
    Tree = DecisionTreeClassifier(random_state=0)

    Tree.fit(X_train,Y_train)

    Train = Tree.score(X_train,Y_train)
    Test = Tree.score(X_test,Y_test)

    return Train,Test

def Diabetes_LR(X_train,X_test,Y_train,Y_test):
    logreg = LogisticRegression()

    logreg.fit(X_train,Y_train)

    Train = logreg.score(X_train,Y_train)
    Test = logreg.score(X_test,Y_test)

    return Train,Test

def Diabetes_RF(X_train,X_test,Y_train,Y_test):
    rf = RandomForestClassifier(n_estimators = 100,random_state =0)
    rf.fit(X_train,Y_train)

    Train = rf.score(X_train,Y_train)
    Test = rf.score(X_test,Y_test)

    return Train,Test

def LoadData():
    Diabetes = pd.read_csv('diabetes.csv')

    print("Column of Datasets")
    print(Diabetes.columns)

    print("First 5 records of Dataset")
    print(Diabetes.head())

    print("Dimension of Diabetes data : {}".format(Diabetes.shape))

    x = []
    y = []

    X_train,X_test,Y_train,Y_test = train_test_split(Diabetes.loc[:,Diabetes.columns!='Outcome'],Diabetes['Outcome'],stratify = Diabetes['Outcome'],random_state=66)

    Train1,test1 = Diabetes_KNN(X_train,X_test,Y_train,Y_test)
    print("Accuracy of KNN classifier on training data set : {:.3f}".format(Train1))
    print("Accuracy of KNN classifier on test data : {:.3f}".format(test1))

    x.append(Train1)
    y.append(test1)

    Train2,test2 = Diabetes_DT(X_train,X_test,Y_train,Y_test)
    print("Accuracy of DT classifier on training data set : {:.3f}".format(Train2))
    print("Accuracy of DT classifier on test data : {:.3f}".format(test2))    

    x.append(Train2)
    y.append(test2)

    Train3,test3 = Diabetes_LR(X_train,X_test,Y_train,Y_test)
    print("Accuracy of LR classifier on training data set : {:.3f}".format(Train3))
    print("Accuracy of LR classifier on test data : {:.3f}".format(test3))

    x.append(Train3)
    y.append(test3)

    Train4,test4 = Diabetes_RF(X_train,X_test,Y_train,Y_test)
    print("Accuracy of RF classifier on training data set : {:.3f}".format(Train4))
    print("Accuracy of RF classifier on test data : {:.3f}".format(test4))

    x.append(Train4)
    y.append(test4)

    labels = ['K-Nearest Neighbors','Decision Tree','Logistic Regression','Random Forest']

    sh = np.arange(len(labels))
    width = 0.3

    fig,ax = plt.subplots()

    rects1 = ax.bar(sh - width/2,x,width,label='Training')
    rects2 = ax.bar(sh+width/2,y,width,label="Testing")

    ax.set_ylabel('Accuracy')
    ax.set_title("Training and Testing of Different algorithm")
    ax.set_xticks(sh,labels)
    ax.legend()

    ax.bar_label(rects1,padding=3)
    ax.bar_label(rects2,padding=3)
    fig.tight_layout()
    plt.show()    

def main():
    print("-----Python Machine Learning Algorithm-----")
    print("Diabetes case study predictor using four Different Algorithm")

    LoadData()

if __name__ == "__main__":
    main()