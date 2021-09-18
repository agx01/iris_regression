# -*- coding: utf-8 -*-

import pandas as pd

data = pd.read_csv("data/iris.data", sep=",", header=None)
data.columns = ["Sepal length", "Sepal width", "Petal length", "Petal width", "Class"]
print(data)

np_array = data.to_numpy()
print(np_array)