# Linear Regression on Iris dataset

## Problem Statement
The use of iris data set for the prediction of species is a classic example for classification problem. This classification problem needs to be solved by the Linear Regression which is a supervised learning problem. A linear regression algorithm needs to be developed that can predict the species of input provided to the algorithm with almost certainty (close to 100% accuracy).

## Strategy
The strategy for implementing a solution is to use k-folds cross validation to create k number of bins in the data and then train and test data on each of these bins which will give Beta for each the bins. To reduce the overfitting, we will find the mean of each Beta value generates for each feature as Beta Mean.

## Folders:
data - Iris dataset is stored in the folder

Please check if the below mentioned dependencies are met 
(Possibly can run on lower versions as I use very basic methods and functions)

## Dependecies:
1. Python : 3.8.8
2. pandas : 1.2.4
3. numpy : 1.20.1
4. matplotlib : 3.3.4
5. seaborn : 0.11.1
6. sklearn : 0.24.1

## Run Program:
To Run the code, download the folder wherever required.
In command prompt, first navigate to the iris_regression folder. 
Then, input the below mentioned command:
```
python main.py
```