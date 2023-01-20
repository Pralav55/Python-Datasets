import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def Digit():

    digits = load_digits()

    print(dir(digits))

    plt.gray()
    for i in range(4):
        plt.matshow(digits.images[i])
        plt.title(i)
        plt.savefig("Fig_{}".format(i))
    
    df = pd.DataFrame(digits.data)
    print(df.head())

    df['target'] = digits.target

    X = df.drop('target',axis='columns')
    Y = df.target

    Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size=0.2)

    model = RandomForestClassifier(n_estimators = 20)

    model.fit(Xtrain,Ytrain)

    y_predict = model.predict(Xtest)

    print("Training Accuracy : {:.3f}".format((model.score(Xtrain,Ytrain))*100))
    print("Testing Accuracy : {:.3f}".format((model.score(Xtest,Ytest))*100))
    print("Overall Accuracy : {:.3f}".format((accuracy_score(Ytest,y_predict))*100))

    print("Confusion Matrix")
    cm = confusion_matrix(Ytest,y_predict)
    print(cm)


def main():
    print("-----Python machine Learning Algorithm")
    print("Digits Predictor using Random Forest")

    Digit()
    
if __name__ =="__main__":
    main()