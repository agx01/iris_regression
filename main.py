# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
             
        Initializes B_list for storing Beta values from each of the bins
        
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
        
        #To store the Beta values in a numpy array
        self.B_list = np.array([])
    
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
        
        #Processing Y values to make sense
        Y = Y.astype(np.float64)
        Y = np.positive(Y)
        Y = np.int32(np.rint(Y, casting='unsafe'))
        return Y
    
    def train_test_split(self, df, per_split, species_per_bin):
        """
        Splits the dataframe input as per the percentage split required

        Parameters
        ----------
        df : Pandas dataframe
            Input dataframe.
        per_split : float
            Percentage split between training set and test set.
        species_per_bin : int
            Number of records of species per bin.

        Returns
        -------
        train_data : Pandas dataframe
            Training data to be used for function fit.
        test_data : Pandas dataframe
            Training data to be used for the function predict.

        """
        train_per_bin = int(species_per_bin * (1 - per_split))
        test_per_bin = int(species_per_bin * per_split)
        train_data = df.iloc[:train_per_bin, :]
        test_data = df.iloc[-test_per_bin:, :]
        
        return train_data, test_data
    
    def cross_validation(self, per_split=0.2, bins=5):
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
                train_df, test_df = self.train_test_split(label_df.iloc[a:b, :], 
                                                          per_split,
                                                          species_per_bin)
                frames = [train_df, new_train_df]
                new_train_df = pd.concat(frames, ignore_index=True)
                binned_data[i]["train"] = new_train_df
                frames = [test_df, new_test_df]
                new_test_df = pd.concat(frames, ignore_index=True)
                binned_data[i]["test"] = new_test_df
                i += 1
        
        #full results
        total_results = 0
        
        #storing results
        predicted_Y = {}
        
        #accuracyscores for each bin
        self.accuracy_scores = {}
        #Using the binned data we do training and testing on the binned data
        for i in range(0, bins):
            data_df = binned_data[i]["train"]
            train_A = data_df.iloc[:,:4]
            train_Y = data_df.iloc[:,-1:]
            test_data_df = binned_data[i]["test"]
            test_A = test_data_df.iloc[:, :4]
            test_Y = test_data_df.iloc[:,-1]
            B = self.fit(train_A, train_Y)
            Y = self.predict(test_A, B).tolist()
            test_Y = test_Y.to_list()
            predicted_Y[i] = Y
            
            self.accuracy_scores[i] = metrics.accuracy_score(y_true=test_Y, y_pred=Y)
            """
            #Checking accuracy
            for j in range(0, len(Y)):
                predicted_value = Y[j]
                actual_value = test_Y[j]
                if predicted_value == actual_value:
                    
                    print(total_results)
            """
            self.B_list = np.append(self.B_list, B)
            i += 1
        
        self.B_list = np.reshape(self.B_list, (bins, 4))
        print("Beta values generated for each bin are:")
        print(self.B_list)
        
        self.B_mean = np.mean(self.B_list, axis=0)
        print("Mean Beta values generated:")
        print(self.B_mean)

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
    
    def test_Bmean(self):
        num_records = self.data.shape[0]
        decision = True
        record_no = 0
        while decision:
            record_no = int(input("Select record number from dataset to use for test(0 to %d):", num_records))
            print("Record number %d has data:", record_no)
            print(self.data[record_no])
            continue_loop = input("Input Yes or y to select anything else:")
            if continue_loop == 'yes' or continue_loop == 'Yes' or continue_loop == 'y' or continue_loop == 'Y':
                decision = True
            else:
                decision = False
        df = self.data[record_no].iloc[:, :4]
        actual_value = self.data[record_no.iloc[:, -1:]]
        B = self.B_mean
        predicted_Y =  self.fit(df, B)
        print("The Predicted value is:")
        print(predicted_Y)
        print("The Actual Value is:")
        print(actual_value)
        
if __name__ == "__main__":
    linreg = LinearRegression()
    linreg.describe_data()
    #linreg.visualize_data()
    per_split = float(input("Please input the training and test split in decimal(20% = 0.2):"))
    bins = int(input("Please input the number of k-folds for cross validation:"))
    linreg.cross_validation(per_split, bins)