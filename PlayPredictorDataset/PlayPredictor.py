import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def Play_Predictor_KNN(data_path):
    data = pd.read_csv(data_path,index_col=0)
    print("Size of Actual Dataset",len(data))

    features_names = ['Whether','Temperature']

    whether = data.Whether
    Temp = data.Temperature
    Label = data.Play

    le = preprocessing.LabelEncoder()

    weather_encoded = le.fit_transform(Temp)
    print(weather_encoded)

    temp_encoded = le.fit_transform(Temp)
    label = le.fit_transform(Label)

    print(temp_encoded)

    features = list(zip(weather_encoded,temp_encoded))

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(features,label)

    predicted = model.predict([[0,2]])

    print(predicted)



def main():
    print("-----Python Machine Learning Application-----")
    print(" Play Predictor application using K-Nearest Neighbors Classifier")

    Play_Predictor_KNN("PlayPredictor.csv")

if __name__ =="__main__":
    main()