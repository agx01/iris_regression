# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data = pd.read_csv("data/iris.data", sep=",", header=None)
data.columns = ["Sepal_length", "Sepal_width", "Petal_length", "Petal_width", "Class"]
print(data.describe())

"""
np_array = data.to_numpy()
print(np_array)
print(np_array.shape)
"""

#Frequency distrbiution of speciies
iris_outcome = pd.crosstab(index = data["Class"],
                           columns="count")


print(iris_outcome)

iris_setosa = data.loc[data["Class"] == "Iris-setosa"]
iris_virginica = data.loc[data["Class"] == "Iris-virginica"]
iris_versicolor = data.loc[data["Class"] == "Iris-versicolor"]
"""
sns.FacetGrid(data, hue="Class", size=3).map(sns.distplot, "Petal_length").add_legend()
sns.FacetGrid(data, hue="Class", size=3).map(sns.distplot, "Petal_width").add_legend()
sns.FacetGrid(data, hue="Class", size=3).map(sns.distplot, "Sepal_length").add_legend()
sns.FacetGrid(data, hue="Class", size=3).map(sns.distplot, "Sepal_width").add_legend()
plt.show()
"""

sns.set_style("whitegrid")
sns.pairplot(data, hue="Class", size=3)
plt.show()


print("Correlation between sepal length and petal length"+str(data["Sepal_length"].corr(data["Petal_length"])))
print("Correlation between sepal length and petal width"+str(data["Sepal_length"].corr(data["Petal_width"])))
print("Correlation between sepal width and petal width"+str(data["Sepal_width"].corr(data["Petal_width"])))
print("Correlation between petal length and petal width"+str(data["Petal_length"].corr(data["Petal_length"])))


kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
