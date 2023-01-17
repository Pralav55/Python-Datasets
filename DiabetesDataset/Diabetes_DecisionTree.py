import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,roc_curve,RocCurveDisplay,precision_recall_curve,PrecisionRecallDisplay

def Confusion_Matrix(y_test,y_predicted,name):
    cm = confusion_matrix(y_test,y_predicted)
    print(cm)
    disp = ConfusionMatrixDisplay(cm)
    
    disp.plot()
    plt.title(name)
    plt.savefig(name)

def plotting_graph(p1,p2,name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    p1.plot(ax=ax1)    
    p2.plot(ax=ax2)
    plt.savefig(name)


def Decision_Tree_Classifier(data,x_train,x_test,y_train,y_test):
    Tree1 = DecisionTreeClassifier(random_state=0)
    Tree2 = DecisionTreeClassifier(max_depth=3,random_state=0)

    Tree1.fit(x_train,y_train)
    Tree2.fit(x_train,y_train)

    pred1 = Tree1.predict(x_test)
    pred2 = Tree2.predict(x_test)

    print("Accuracy of Training data of Tree1 is {:.3f}".format(Tree1.score(x_train,y_train)))
    print("Accuracy of Training data of Tree2 is {:.3f}".format(Tree2.score(x_train,y_train)))

    print("Accuracy of Testing data of Tree1 is {:.3f}".format(Tree1.score(x_test,y_test)))
    print("Accuracy of Testing data of Tree2 is {:.3f}".format(Tree2.score(x_test,y_test)))

    print("Confusion matrix for Tree1 : ")
    Confusion_Matrix(y_test,Tree1.predict(x_test),"Tree1_graph")

    print("Confusion matrix for Tree2 : ")
    Confusion_Matrix(y_test,Tree2.predict(x_test),"Tree2_graph")

    obj = RocCurveDisplay.from_estimator(Tree1,x_test,y_test,alpha=0.8)
    obj2 = RocCurveDisplay.from_estimator(Tree2,x_test,y_test,alpha=0.8)

    plotting_graph(obj,obj2,"TP-TN_graph")
      
    prec, recall, _ = precision_recall_curve(y_test, pred1)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

    prec1, recall1, _ = precision_recall_curve(y_test, pred2)
    pr_display2 = PrecisionRecallDisplay(precision=prec1, recall=recall1)

    plotting_graph(pr_display,pr_display2,"Precision_recall_graph")

def main():
    print("Python Machine Learning Algorithm")

    Data = pd.read_csv("diabetes.csv")

    print("Name of Feature and Label in the csv")
    print(Data.columns)

    print("First 5 data from the csv")
    print(Data.head())

    print("Dimensions of the diabetes csv file {}".format(Data.shape))

    X_train,X_test,Y_train,Y_test = train_test_split(Data.loc[:,Data.columns != 'Outcome'],Data['Outcome'],stratify=Data['Outcome'],random_state = 66)

    Decision_Tree_Classifier(Data,X_train,X_test,Y_train,Y_test)


if __name__ =="__main__":
    main()