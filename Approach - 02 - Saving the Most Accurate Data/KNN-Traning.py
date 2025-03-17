import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv("car.data")
print(data.head())
label = preprocessing.LabelEncoder()
buying = label.fit_transform(list(data["buying"]))
maint = label.fit_transform(list(data["maint"]))
door = label.fit_transform(list(data["door"]))
persons = label.fit_transform(list(data["persons"]))
lug_boot = label.fit_transform(list(data["lug_boot"]))
safety = label.fit_transform(list(data["safety"]))
Class = label.fit_transform(list(data["class"]))

prerdict = "class"

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(Class)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)

# Save the trained model
with open("knn_model.pickle", "wb") as f:
    pickle.dump(model, f)

accuracy = model.score(x_test, y_test)
print(f"Accuracy = {accuracy*100}%")
