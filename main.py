# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

class LinearRegression:
    
    def __init__(self):
        """
         Retrieving data from the iris data set file into a dataframe
         Preprocessing includes
         1. Define the column names
         2. Label Encoding the final column
             2.1 Changing the column 'Species' type to "category"
             2.2 Assigning numerical values and storing it in column 'Species_cat'
             

        Returns
        -------
        None.

        """
        self.data = pd.read_csv("data/iris.data", sep=",", header=None)
        self.data.columns = ["Sepal_length", 
                             "Sepal_width", 
                             "Petal_length", 
                             "Petal_width", 
                             "Species"]
        self.data["Species"] = self.data["Species"].astype('category')
        self.data["Species_cat"] = self.data["Species"].cat.codes
    
    def describe_data(self):
        """
        Generate numeric meta data on the data
        1. Number of records for each class
        2. Std dev, Mean, variance and other values for each of the columns

        Returns
        -------
        None.

        """
        iris_cross = pd.crosstab(index = self.data["Species"], 
                                 columns = "count")
        print(iris_cross)
        print(self.data.describe())
    
    def visualize_data(self):
        """
        Creates graphs to visualize the preprocessed data
        1. Pair plot for the data - provides the std distribution of each column
           and column data in comparison to other columns.

        Returns
        -------
        None.

        """
        
        sns.set_style("whitegrid")
        sns.pairplot(self.data, hue="Species", size=3)
        plt.show()
    
    def fit(self):
        pass
    
    def predict(self):
        pass

    def test_fit(self):
        """
        Testing fit algorithm on the entire data set
        create A and Y
        generate B using the formula
        Returns
        -------
        None.

        """
        A = self.data.iloc[:, : 4].to_numpy()
        Y = self.data.iloc[:, -1].to_numpy()
        
        B = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)),A.transpose()),Y)
        print(B)
        

if __name__ == "__main__":
    linreg = LinearRegression()