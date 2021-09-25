# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv("data/iris.data", sep=",", header=None)
data.columns = ["Sepal_length", "Sepal_width", "Petal_length", "Petal_width", "Species"]
print(data.describe())

print("Change the column Class to category type")
data["Species"] = data["Species"].astype('category')

print("Assigning numberical values and storing in another column")
data["Species_cat"] = data["Species"].cat.codes

print(data)


"""
np_array = data.to_numpy()
print(np_array)
print(np_array.shape)
"""

#Frequency distrbiution of speciies
iris_outcome = pd.crosstab(index = data["Species"],
                           columns="count")


print(iris_outcome)

iris_setosa = data.loc[data["Species"] == "Iris-setosa"]
iris_virginica = data.loc[data["Species"] == "Iris-virginica"]
iris_versicolor = data.loc[data["Species"] == "Iris-versicolor"]
"""
sns.FacetGrid(data, hue="Species", size=3).map(sns.distplot, "Petal_length").add_legend()
sns.FacetGrid(data, hue="Species", size=3).map(sns.distplot, "Petal_width").add_legend()
sns.FacetGrid(data, hue="Species", size=3).map(sns.distplot, "Sepal_length").add_legend()
sns.FacetGrid(data, hue="Species", size=3).map(sns.distplot, "Sepal_width").add_legend()
plt.show()
"""

"""
sns.set_style("whitegrid")
sns.pairplot(data, hue="Species", size=3)
plt.show()
"""

print("Correlation between sepal length and petal length"+str(data["Sepal_length"].corr(data["Petal_length"])))
print("Correlation between sepal length and petal width"+str(data["Sepal_length"].corr(data["Petal_width"])))
print("Correlation between sepal width and petal width"+str(data["Sepal_width"].corr(data["Petal_width"])))
print("Correlation between petal length and petal width"+str(data["Petal_length"].corr(data["Petal_length"])))

"""
kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
"""

print("A:")
A = data.iloc[:, : 4].to_numpy()
print(A)

print("Y:")
Y = data.iloc[:, -1].to_numpy()
print(Y)

print("Checking the data types:\nA:"+ str(A.dtype)+ "Y:"+str(Y.dtype))

"""
Y = Y.astype(float)
print("Checking the data types:\nA:"+ str(A.dtype)+ "Y:"+str(Y.dtype))

print(Y)

"""

B = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)),A.transpose()),Y)

print(B)


iris_setosa = data[data["Species"] == "Iris-setosa"]
iris_virginica = data[data["Species"] == "Iris-virginica"]
iris_versicolor = data[data["Species"] == "Iris-versicolor"]
        
#Get the unique values in column species
labels = data.Species.unique().to_list()
        
#Get the number of unique values
num_labels = len(labels)
        
#Dictionary to hold all the splits
label_split_data = {}

per_split = 0.2
print(data.shape[0])
total_rows = data.shape[0]

print("Training data required:")
total_train_data = int(total_rows*(1 - per_split))
print(total_train_data)

print("Testing data required:")
total_test_data = int(total_rows*per_split)
print(total_test_data)

"""
print("Training data per species required:")
cat_train_data = int(total_train_data/num_labels)
print(cat_train_data)

print("Testing data per species required:")
cat_test_data = int(total_test_data/num_labels)
print(cat_test_data)
"""

#take input for the number of sets to be created
bins = int(input("Input the value for k:"))

for label in labels:
    label_split_data[label] = data[data["Species"] == label]
    
records_per_bin = int(total_rows/bins)
species_per_bin = int(records_per_bin/num_labels)
binned_data = {}

#initialize binned data
for i in range(0, bins):
    new_df = pd.DataFrame(columns=data.columns)
    new_dict = {"train": new_df, "test": new_df}
    binned_data[i] = new_dict

"""
#for each species in data
for label in labels:
    label_df = label_split_data[label]
    #for each bin available in data
    for i in range(0, bins):
        #check if any data exist in the bins
        if i in binned_data.keys():
            new_df = binned_data[i]
        else:
            #if no create a new empty data frame
            new_df = pd.DataFrame(columns=data.columns)
        #concatenate dataframes
        a = i*species_per_bin
        b = (i*species_per_bin+species_per_bin)
        frames = [label_df.iloc[a:b, :], new_df]
        new_df = pd.concat(frames, ignore_index=True)
        binned_data[i] = new_df
        i += 1
"""
def train_test_split(df, per_split):
    train_per_bin = int(species_per_bin * (1 - per_split))
    test_per_bin = int(species_per_bin * per_split)
    train_data = df.iloc[:train_per_bin, :]
    test_data = df.iloc[-test_per_bin:, :]
    
    return train_data, test_data

def fit(train_A, train_Y):
    A = train_A.to_numpy()
    Y = train_Y.to_numpy()
    B = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.transpose(), A)),A.transpose()),Y)
    return B

def predict(test_A, B):
    A = test_A.to_numpy()
    Y = np.matmul(A, B)
    #Y = np.matmul(np.matmul(np.matmul(A.transpose(), A), B), np.linalg.inv(A.transpose()))
    return Y.astype(np.float64)

from sklearn import metrics

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
        train_df, test_df = train_test_split(label_df.iloc[a:b, :], per_split)
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
    B = fit(train_A, train_Y)
    Y = np.rint(np.positive(predict(test_A, B)), casting='unsafe')
    test_Y = test_Y.to_list()
    print(metrics.accuracy_score(test_Y, Y.tolist()))
    B_list.append(B)
    i += 1

print(B_list)
"""    
iris_setosa = data[data["Species"] == "Iris-setosa"]
iris_virginica = data[data["Species"] == "Iris-virginica"]
iris_versicolor = data[data["Species"] == "Iris-versicolor"]
#Split the data into the species and store the data in dictionary
#for label in labels:
    #label_split_data[label] = self.data[self.data["Species"] == label]
        
#Each species has 50 records, creating a 20% split in each species
#40 records for training and 10 records for testing in each species
        
#Creating 3 distinct sets for each of the species
train_setosa1 = iris_setosa.iloc[40, :]
test_setosa1 = iris_setosa.iloc[-10, :]
        
train_setosa2 = iris_setosa.iloc[-40, :]
test_setosa2 = iris_setosa.iloc[10, :]
        
train_setosa3 = iris_setosa.iloc[5:45, :]
test_setosa3 = iris_setosa.iloc[6:-4, :]
        
train_virginica1 = iris_virginica.iloc[40, :]
test_virginica1 = iris_virginica.iloc[-10, :]
        
train_virginica2 = iris_virginica.iloc[-40, :]
test_virginica2 = iris_virginica.iloc[10, :]
        
train_virginica3 = iris_virginica.iloc[5:45, :]
test_virginica3 = iris_virginica.iloc[6:-4, :]
        
train_versicolor1 = iris_versicolor.iloc[40, :]
test_versicolor1 = iris_versicolor.iloc[-10, :]
        
train_versicolor2 = iris_versicolor.iloc[-40, :]
test_versicolor2 = iris_versicolor.iloc[10, :]
        
train_versicolor3 = iris_versicolor.iloc[5:45, :]
test_versicolor3 = iris_versicolor.iloc[6:-4, :]
        
#print(iris_setosa)

"""

"""        
#creating 1st dataset
train_data1 = pd.concat(train_versicolor1, pd.concat(train_setosa1, train_virginica1, ignore_index=True), ignore_index=True)
test_data1 = pd.concat(test_versicolor1, pd.concat(test_setosa1, test_virginica1, ignore_index=True), ignore_index=True)
        
#creating 2nd dataset
train_data2 = pd.concat(train_versicolor2, pd.concat(train_setosa2, train_virginica2, ignore_index=True), ignore_index=True)
test_data2 = pd.concat(test_versicolor2, pd.concat(test_setosa2, test_virginica2, ignore_index=True), ignore_index=True)
        
#creating 3rd dataset
train_data3 = pd.concat(train_versicolor3, pd.concat(train_setosa3, train_virginica3, ignore_index=True), ignore_index=True)
test_data3 = pd.concat(test_versicolor3, pd.concat(test_setosa3, test_virginica3, ignore_index=True), ignore_index=True)
        
print("Train test set 1:")
print(train_data1)
print(test_data1)
        
print("Train test set 2:")
print(train_data2)
print(test_data2)
        
print("Train test set 3:")
print(train_data3)
print(test_data3)
"""

"""
#Test code for stackexchange question
new_df = pd.DataFrame(columns=["ID", "Date", "Earthquake", "Fire","Storm Damage"])
new_df = new_df.append({"ID":1, "Date":'1/21/21', "Earthquake":0, "Fire":0, "Storm Damage":0}, ignore_index = True)
new_df = new_df.append({"ID":2, "Date":'2/3/21', "Earthquake":1, "Fire":0, "Storm Damage":0}, ignore_index = True)
new_df = new_df.append({"ID":3, "Date":'2/4/21', "Earthquake":0, "Fire":1, "Storm Damage":0}, ignore_index = True)
new_df = new_df.append({"ID":1, "Date":'2/10/21', "Earthquake":1, "Fire":0, "Storm Damage":0}, ignore_index = True)
new_df = new_df.append({"ID":1, "Date":'2/28/21', "Earthquake":0, "Fire":1, "Storm Damage":1}, ignore_index = True)
new_df = new_df.append({"ID":2, "Date":'3/5/21', "Earthquake":0, "Fire":0, "Storm Damage":1}, ignore_index = True)

print(new_df)

grouped = new_df.groupby("ID")
print(grouped.first())
print("Type of grouped:"+str(type(grouped)))
sorted_df = new_df.sort_values(["ID"], ascending=[1])
print("Type of Sort values:"+str(type(sorted_df)))
print(sorted_df)
"""