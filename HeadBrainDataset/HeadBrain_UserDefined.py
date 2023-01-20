#Importing required Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Function Plotting graph of Head Brain Dataset
def Graph(x,y,X,Y):
    plt.plot(x,y,color='#58b970',label='Regression Line')
    plt.scatter(X,Y,color='#ef5423',label = 'scatter plot',s=5)
    plt.xlabel('Head Size(cm^3)')
    plt.ylabel('Brain Weight(grams)')
    plt.legend()
    plt.show()

#Function to Load Data into program
def Load_CSV():
    data = pd.read_csv('HeadBrain.csv')
    return data

#User-Defined Function of Linear Regression 
def HeadBrainUser(dataset):
    #printing shape of the dataset
    print("Size of dataset is ",dataset.shape)

    #Seperating out feature in X and Label in Y
    #Here X is Independant Variable
    X = dataset['Head Size(cm^3)'].values

    #Here Y is Dependant Variable
    Y = dataset['Brain Weight(grams)'].values

    #Calculating mean of X and Y using numpy
    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    #Calculating length of X
    n = len(X)

    numerator = 0
    denominator = 0

    #Calculating Summation of numerator and denominator seperately for slope of line
    for i in range(n):
        numerator += (X[i]-mean_x)*(Y[i]-mean_y)
        denominator += (X[i]-mean_x)**2
    
    #Calculating Slope of line
    m = numerator/denominator

    #Calculating Y intercept of Line
    c = mean_y - (m*mean_x)

    #Displaying Slope and Y-intercept
    print("Slope of Regression line is : {:.3f}".format(m))
    print("Y intercept of regression line is : {:.3f}".format(c))

    #Finding Maximum value in X and Minumum value in X
    max_x = np.max(X)+100
    min_x = np.min(X)-100

    #generate linear sequence out of X
    x = np.linspace(min_x,max_x,n)

    #Formula for Equation of Line
    y = (m*x)+c

    ss_t = 0
    ss_r = 0

    #Calculating R Square value
    for i in range(n):
        y_predict = (m*X[i])+c

        ss_t += (Y[i] - mean_y)**2
        ss_r += (y_predict - mean_y)**2

    r2 = 1-(ss_r/ss_t)

    print("R Square value : ",r2)

    Graph(x,y,X,Y)

#Execution starts from main function
def main():
    #Displays Header
    print("-----Python Machine Learning Algorithm-----")
    print("Head Brain Dataset using user Defined Function")

    #Loading csv into program
    dataset = Load_CSV()

    #Function which calculates Accuracy of Head Brain Dataset
    HeadBrainUser(dataset)

#Application starter
if __name__ == "__main__":
    main()