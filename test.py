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

sns.set_style("whitegrid")
sns.pairplot(data, hue="Species", size=3)
plt.show()


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