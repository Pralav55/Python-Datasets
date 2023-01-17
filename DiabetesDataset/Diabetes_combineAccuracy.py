import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def Bar_Graph(Accuracy):
    Name = ['Decision Tree','KNN','Logistic Reg','Random Forest']

    x = np.arange(len(Name))
    width = 0.5

    fig,ax = plt.subplots()

    bar_labels = ['DT','KNN','LR','RF']
    bar_colors = ['tab:red','tab:blue','tab:orange','tab:pink']

    ch = ax.bar(Name,Accuracy,width,label = bar_labels,color = bar_colors)

    ax.set_ylabel("Percentage")
    ax.set_title("Diabetes Dataset Accuracy with different algorithm")
    ax.legend(title ="Algorithm")
    ax.bar_label(ch,padding=3)
    fig.tight_layout()
    plt.show()

def Diabetes_KNN(X_train,X_test,Y_train,Y_test):
    Knn = KNeighborsClassifier(n_neighbors=3)

    Knn.fit(X_train,Y_train)

    Predictions = Knn.predict(X_test)

    Accuracy = accuracy_score(Y_test,Predictions)

    return Accuracy

def Diabetes_DT(X_train,X_test,Y_train,Y_test):
    Tree = DecisionTreeClassifier(random_state=0)

    Tree.fit(X_train,Y_train)

    Predictions = Tree.predict(X_test)

    Accuracy = accuracy_score(Y_test,Predictions)

    return Accuracy

def Diabetes_LR(X_train,X_test,Y_train,Y_test):
    logreg = LogisticRegression()

    logreg.fit(X_train,Y_train)

    Predictions = logreg.predict(X_test)

    Accuracy = accuracy_score(Y_test,Predictions)

    return Accuracy

def Diabetes_RF(X_train,X_test,Y_train,Y_test):
    rf = RandomForestClassifier(n_estimators = 100,random_state =0)
    rf.fit(X_train,Y_train)

    Predictions = rf.predict(X_test)

    Accuracy = accuracy_score(Y_test,Predictions)

    return Accuracy

def LoadData():
    Diabetes = pd.read_csv('diabetes.csv')

    print("Column of Datasets")
    print(Diabetes.columns)

    print("First 5 records of Dataset")
    print(Diabetes.head())

    print("Dimension of Diabetes data : {}".format(Diabetes.shape))

    X_train,X_test,Y_train,Y_test = train_test_split(Diabetes.loc[:,Diabetes.columns!='Outcome'],Diabetes['Outcome'],stratify = Diabetes['Outcome'],random_state=66)

    Accuracy = []

    Accuracy.append((Diabetes_DT(X_train,X_test,Y_train,Y_test))*100)
    Accuracy.append((Diabetes_KNN(X_train,X_test,Y_train,Y_test))*100)
    Accuracy.append((Diabetes_LR(X_train,X_test,Y_train,Y_test))*100)
    Accuracy.append((Diabetes_RF(X_train,X_test,Y_train,Y_test))*100)

    Bar_Graph(Accuracy)

def main():
    print("-----Python Machine Learning Algorithm-----")
    print("Diabetes case study predictor using four Different Algorithm")

    LoadData()

if __name__ == "__main__":
    main()