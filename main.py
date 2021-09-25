# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import metrics

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
    
    def fit(self, train_A, train_Y):
        """
        Fit method calculates the beta values 

        Parameters
        ----------
        train_A : Pandas Dataframe
            This contains only the feature columns
        train_Y : Pandas Dataframe
            This contains only the record of the species its corresponding
            features in A belong to

        Returns
        -------
        B : Numpy array
            This contains the Beta values trained by the input training data.

        """
        A = train_A.to_numpy()
        Y = train_Y.to_numpy()
        B = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)),A.transpose()),Y)
        return B
    
    def predict(self, test_A, B):
        """
        This function is to predict the species provided the features from 
        Test records and the beta values we generated from the fit method

        Parameters
        ----------
        test_A : Pandas dataframe
            Features dataframe that we want to use for testing
        B : numpy array
            Beta values required for predicting the species of the test set.

        Returns
        -------
        Array of predicted values as a numpy array of type float
            The predict function generated the below mentioned values 
            representing each species

        """
        A = test_A.to_numpy()
        Y = np.matmul(A, B)
        return Y.astype(np.float64)
    
    def cross_validation(self, per_split, bins):
        """
        Cross validation function
        Split the data into training and test sets to validate it across the
        whole set.
        It needs to split for each of the species available in iris data
        1. iris-setosa
        2. iris-virginica
        3. iris-versicolor

        Returns
        -------
        None.

        """
        #Get the unique values in column species
        labels = self.data.Species.unique().to_list()
        
        #Get the number of unique values
        num_labels = len(labels)
        
        #Numeric values to be used for splicing data
        total_rows = self.data.shape[0]
        total_train_data = int(total_rows*(1 - per_split))
        total_test_data = int(total_rows*per_split)
        records_per_bin = int(total_rows/bins)
        species_per_bin = int(records_per_bin/num_labels)
        
        #To store the data split by species
        label_split_data = {}
        
        #To store the data in k bins as in k folds cross validation
        binned_data = {}
        
        #initialize binned data
        for i in range(0, bins):
            new_df = pd.DataFrame(columns=self.data.columns)
            new_dict = {"train": new_df, "test": new_df}
            binned_data[i] = new_dict
        
        #Splitting data for each species and storing in the dictionary with key
        #as the species name
        for label in labels:
            label_split_data[label] = self.data[self.data["Species"] == label]
        
        #Splitting the data and putting in bins for each of the species
        #for each species in data
        for label in labels:
            label_df = label_split_data[label]
            #for each bin available in data
            for i in range(0, bins):
                new_train_df = binned_data[i]["train"]
                new_test_df = binned_data[i]["test"]
                
                #concatenate dataframes
                a = i*species_per_bin
                b = (i*species_per_bin+species_per_bin)
                train_df, test_df = self.train_test_split(label_df.iloc[a:b, :], per_split)
                frames = [train_df, new_train_df]
                new_train_df = pd.concat(frames, ignore_index=True)
                binned_data[i]["train"] = new_train_df
                frames = [test_df, new_test_df]
                new_test_df = pd.concat(frames, ignore_index=True)
                binned_data[i]["test"] = new_test_df
                i += 1
        
        B_list = []
        for i in range(0, bins):
            data_df = binned_data[i]["train"]
            train_A = data_df.iloc[:,:4]
            train_Y = data_df.iloc[:,-1:]
            test_data_df = binned_data[i]["test"]
            test_A = test_data_df.iloc[:, :4]
            test_Y = test_data_df.iloc[:,-1]
            B = self.fit(train_A, train_Y)
            Y = np.rint(np.positive(self.predict(test_A, B)), casting='unsafe')
            test_Y = test_Y.to_list()
            print(metrics.accuracy_score(test_Y, Y.tolist()))
            B_list.append(B)
            i += 1
        

    def test_fit(self):
        """
        Testing fit algorithm on the entire data set
        create A and Y
        generate B using the formula

        Returns
        -------
        B : TYPE numpy array of size 4x1
            the beta values from the full data set

        """
        
        A = self.data.iloc[:, : 4].to_numpy()
        Y = self.data.iloc[:, -1].to_numpy()
        
        B = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)),A.transpose()),Y)
        print(B)
        
        return B
        

if __name__ == "__main__":
    linreg = LinearRegression()
    linreg.describe_data()
    #linreg.visualize_data()
    
    linreg.cross_validation()