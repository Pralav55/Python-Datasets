#importing required library from Python Library
from sklearn.tree import DecisionTreeClassifier

#Function which predicts the Label 
def Ball_Dataset_Decision_Tree(weight,surface):

    #Loading the Dataset into the program
    #smooth- 0        rough- 1
    Features = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    Labels = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]    
    #Tennis- 1     Cricket- 2

    #Creating Object of a Algorithm 
    clf = DecisionTreeClassifier()

    #Performs Training on Loaded Dataset 
    clf = clf.fit(Features,Labels)

    #Performs Testing on Trained Dataset with user input data 
    result = clf.predict([[weight,surface]])

    #Displaying the predicted data 
    if result == 1:
        print("Your object looks like a Tennis ball")
    elif result == 2:
        print("Your object looks like a Cricket Ball")

#Execution of algorithm starts from main
def main():
    #Displays Header
    print("-----Python Machine Learning Algorithm-----")
    print(" Ball Dataset using Decision Tree Algorithm")

    #Accepting a weight of object from user
    print("Enter weight of object : ")
    weight = input()

    #Accepting a surface of object from user
    print("What is the surface type of object Rough or Smooth : ")
    surface = input()

    #checking if the accepted value is smooth or rough with handling case
    if surface.lower() == 'rough':
        surface = 1
    elif surface.lower() == 'smooth':
        surface = 0
    else:
        print("Error : Wrong input")
        exit()
    
    #Calling the Function with 2 parameters as weight and surface
    Ball_Dataset_Decision_Tree(weight,surface)

# Starter of Application
if __name__ == "__main__":
    main()