#Importing Required Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Function to Plot graph for Linear Regression
def plot_graph(X,Y,predict):
    plt.scatter(X,Y,c='red',s=1.5)
    plt.plot(X,predict,c='blue')
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weight(grams)")
    plt.title("Head Brain Dataset using Linear Regression")

    plt.savefig('Head_Brain_plt')
    plt.show()

#Function to Load CSV into program 
def Load_CSV():
    data = pd.read_csv('HeadBrain.csv')
    return data

#Function which manipulates the data of Head Brain using Linear Regression
def Head_Brain_LR(data):

    print("Size of dataset is ",data.shape)

    #Seperating out feature in X and Label in Y
    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values

    X = X.reshape((-1,1))

    #Creating Object of LinearRegression algorithm
    reg = LinearRegression()

    #Training the dataset
    reg = reg.fit(X,Y)

    #Testing with same dataset
    predictions = reg.predict(X)

    #Accuracy calculated
    Accuracy = reg.score(X,Y)

    #Function to plot Accuracy
    plot_graph(X,Y,predictions)

    return Accuracy

#Execution of Program starts from here
def main():
    #Displays Header
    print("-----Python Machine Learning Algorithm-----")
    print("Head Brain Data set using Linear Regression")

    #Loading dataset
    dataset = Load_CSV()

    #Calculating accuracy
    Accuracy = Head_Brain_LR(dataset)
    print("Accuracy of Head Brain Data is ",Accuracy*100,"%")

#Application starter
if __name__ == "__main__":
    main()